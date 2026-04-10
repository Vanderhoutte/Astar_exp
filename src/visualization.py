"""TSP 静态图（PNG）与 Tk 过程演示 GUI。"""

from __future__ import annotations

import json
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, List, Optional

from . import mpl_compat  # noqa: F401

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from .astar_tsp_solver import (
    EVENT_DONE,
    EVENT_GOAL,
    EVENT_POP,
    EVENT_PUSH,
    AStarTSPSolver,
    SearchResult,
)
from .heuristics import get_heuristic
from .solve_policy import (
    astar_search_progress_kwargs,
    recommend_solver_mode,
    solve_tsp_auto,
    state_space_upper_bound,
)
from .feasibility_estimate import estimate_feasible_closed_tour, format_feasibility_log_lines
from .solution_verifier import exact_optimal_tour_small_n, validate_closed_tour
from .tsp_heuristic_solvers import search_heuristic_stepwise
from .map_generator import (
    TSPGraph,
    ensure_connected_fallback,
    generate_random_map,
    load_instance,
)


def draw_tsp(
    ax: Axes,
    tsp: TSPGraph,
    *,
    tour: List[int] | None = None,
    partial_path: List[int] | None = None,
    city_color: str = "steelblue",
    city_size: float = 40,
    annotate_fontsize: int = 8,
    edge_color: str = "lightgray",
    edge_lw: float = 0.8,
    tour_lw: float = 1.5,
    overlay_lw: float = 1.4,
) -> None:
    pts = tsp.points
    ax.scatter(pts[:, 0], pts[:, 1], c=city_color, s=city_size, zorder=3)
    for i in range(tsp.n):
        ax.annotate(
            str(i),
            (pts[i, 0], pts[i, 1]),
            fontsize=annotate_fontsize,
            xytext=(3, 3),
            textcoords="offset points",
        )
    drawn: set[tuple[int, int]] = set()
    for u in range(tsp.n):
        for v, _ in tsp.adjacency[u]:
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in drawn:
                continue
            drawn.add((a, b))
            ax.plot(
                [pts[u, 0], pts[v, 0]],
                [pts[u, 1], pts[v, 1]],
                c=edge_color,
                lw=edge_lw,
                zorder=1,
            )
    if tour and len(tour) >= 2:
        tx = [pts[i, 0] for i in tour]
        ty = [pts[i, 1] for i in tour]
        ax.plot(tx, ty, "r--", lw=tour_lw, zorder=2, label="tour")
        ax.legend()
    if partial_path and len(partial_path) >= 2:
        xs = [pts[i, 0] for i in partial_path]
        ys = [pts[i, 1] for i in partial_path]
        ax.plot(xs, ys, "r-", lw=overlay_lw, zorder=4, alpha=0.85)
    ax.set_aspect("equal", adjustable="box")


def save_map_png(
    tsp: TSPGraph,
    out_path: str | Path,
    tour: List[int] | None = None,
    title: str | None = None,
    dpi: int = 120,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    draw_tsp(ax, tsp, tour=tour)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title or f"TSP map (n={tsp.n})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _gui_defaults() -> Dict[str, Any]:
    return {
        "default_n": 10,
        "default_k": 4,
        "default_heuristic": "euclidean",
        "default_start": "random",
        "default_max_expansions": "500000",
        "default_time_limit": "",
        "auto_solver": True,
        "force_exact_astar": False,
        "force_stronger_solver": False,
        "force_monster_solver": False,
        "force_bruteforce": False,
        "monster_max_n": 24,
        "monster_time_limit_sec": 20.0,
        "bruteforce_max_n": 11,
        "bruteforce_time_limit_sec": None,
        "stronger_exact_max_n": 18,
        "stronger_exact_time_limit_sec": None,
        "astar_max_n": 20,
        "weighted_max_n": None,
        "weighted_astar_epsilon": None,
        "fallback_sa_iterations": 8000,
        "fallback_sa_T0": 1.0,
        "fallback_sa_T_min": 1e-4,
        "fallback_sa_cool": 0.995,
        "fallback_sa_seed": None,
        "fallback_random_nn_tries": 500,
        "fallback_rcl_size": 4,
        "fallback_backtrack": True,
        "fallback_backtrack_max_steps": 3000000,
        "fallback_backtrack_restarts": 24,
        "fallback_backtrack_time_sec": None,
        "verify_exact_max_n": 14,
        "verify_exact_time_sec": None,
    }


class AStarGUI:
    def __init__(self, root: tk.Tk, gui_cfg: Optional[Dict[str, Any]] = None) -> None:
        self.root = root
        root.title("A* TSP 实验")
        self.g = {**_gui_defaults(), **(gui_cfg or {})}
        self.tsp: Optional[TSPGraph] = None
        self.gen: Optional[Any] = None
        self.step_job: Optional[str] = None
        self.paused = False
        self.auto_play = tk.BooleanVar(value=False)
        self.var_force_exact = tk.BooleanVar(value=bool(self.g.get("force_exact_astar", False)))
        self.var_force_stronger = tk.BooleanVar(value=bool(self.g.get("force_stronger_solver", False)))
        self.var_force_monster = tk.BooleanVar(value=bool(self.g.get("force_monster_solver", False)))
        self.var_force_bruteforce = tk.BooleanVar(value=bool(self.g.get("force_bruteforce", False)))
        self._frozen_random_start: Optional[int] = None
        self._init_best_tour: Optional[List[int]] = None
        self._init_best_tour_certified_optimal: bool = False
        self._cum_search_s: float = 0.0
        self._step_start: int = 0
        self._cancel_astar = threading.Event()
        self._astar_bg_running = False
        self._closing = False
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._build()

    def _is_alive(self) -> bool:
        if self._closing:
            return False
        try:
            return bool(self.root.winfo_exists())
        except tk.TclError:
            return False

    def _safe_after(self, delay_ms: int, fn: Any) -> None:
        if not self._is_alive():
            return
        try:
            self.root.after(delay_ms, fn)
        except tk.TclError:
            pass

    def on_close(self) -> None:
        if self._closing:
            return
        self._closing = True
        self._cancel_astar.set()
        self.paused = True
        self.gen = None
        if self.step_job:
            try:
                self.root.after_cancel(self.step_job)
            except tk.TclError:
                pass
            self.step_job = None
        try:
            self.root.quit()
        except tk.TclError:
            pass
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def _build(self) -> None:
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill=tk.BOTH, expand=True)

        ctrl = ttk.LabelFrame(main, text="参数与控制", padding=6)
        ctrl.pack(side=tk.RIGHT, fill=tk.Y)

        g = self.g
        ttk.Label(ctrl, text="城市数 n").grid(row=0, column=0, sticky=tk.W)
        self.var_n = tk.StringVar(value=str(g.get("default_n", 10)))
        ttk.Entry(ctrl, textvariable=self.var_n, width=8).grid(row=0, column=1)

        ttk.Label(ctrl, text="k 近邻(3–5)").grid(row=1, column=0, sticky=tk.W)
        self.var_k = tk.StringVar(value=str(g.get("default_k", 4)))
        ttk.Entry(ctrl, textvariable=self.var_k, width=8).grid(row=1, column=1)

        ttk.Label(ctrl, text="启发函数").grid(row=2, column=0, sticky=tk.W)
        self.var_h = tk.StringVar(value=str(g.get("default_heuristic", "euclidean")))
        ttk.Combobox(
            ctrl,
            textvariable=self.var_h,
            values=["euclidean", "manhattan", "custom"],
            width=10,
            state="readonly",
        ).grid(row=2, column=1)

        ttk.Label(ctrl, text="起点(空/random=随机)").grid(row=3, column=0, sticky=tk.W)
        self.var_start = tk.StringVar(value=str(g.get("default_start", "random")))
        ttk.Entry(ctrl, textvariable=self.var_start, width=10).grid(row=3, column=1)

        ttk.Label(ctrl, text="扩展上限(空=无)").grid(row=4, column=0, sticky=tk.W)
        self.var_max_exp = tk.StringVar(value=str(g.get("default_max_expansions", "500000")))
        ttk.Entry(ctrl, textvariable=self.var_max_exp, width=10).grid(row=4, column=1)

        ttk.Label(ctrl, text="时限秒(空=无)").grid(row=5, column=0, sticky=tk.W)
        self.var_time = tk.StringVar(value=str(g.get("default_time_limit", "")))
        ttk.Entry(ctrl, textvariable=self.var_time, width=10).grid(row=5, column=1)

        ttk.Checkbutton(
            ctrl, text="强制精确 A*（忽略规模估算）", variable=self.var_force_exact
        ).grid(row=6, column=0, columnspan=2, sticky=tk.W)
        ttk.Checkbutton(
            ctrl, text="强制更强算法（禁用A*）", variable=self.var_force_stronger
        ).grid(row=7, column=0, columnspan=2, sticky=tk.W)
        ttk.Checkbutton(
            ctrl, text="强制 Monster 分支定界（禁用A*）", variable=self.var_force_monster
        ).grid(row=8, column=0, columnspan=2, sticky=tk.W)
        ttk.Checkbutton(
            ctrl, text="强制纯暴力（小规模）", variable=self.var_force_bruteforce
        ).grid(row=9, column=0, columnspan=2, sticky=tk.W)

        ttk.Button(ctrl, text="随机生成实例", command=self.on_generate).grid(
            row=10, column=0, columnspan=2, pady=4, sticky=tk.EW
        )
        ttk.Button(ctrl, text="加载 CSV…", command=self.on_load).grid(
            row=11, column=0, columnspan=2, pady=2, sticky=tk.EW
        )
        ttk.Separator(ctrl, orient=tk.HORIZONTAL).grid(
            row=12, column=0, columnspan=2, sticky=tk.EW, pady=6
        )
        ttk.Button(ctrl, text="运行到结束(后台)", command=self.on_run_full).grid(
            row=13, column=0, columnspan=2, pady=2, sticky=tk.EW
        )
        self.btn_cancel_astar = ttk.Button(
            ctrl,
            text="终止 A* 搜索",
            command=self.on_cancel_astar,
            state=tk.DISABLED,
        )
        self.btn_cancel_astar.grid(row=14, column=0, columnspan=2, pady=2, sticky=tk.EW)
        ttk.Button(ctrl, text="单步开始/下一步", command=self.on_step_start).grid(
            row=15, column=0, columnspan=2, pady=2, sticky=tk.EW
        )
        ttk.Checkbutton(ctrl, text="自动连续步进", variable=self.auto_play).grid(
            row=16, column=0, columnspan=2, sticky=tk.W
        )
        ttk.Button(ctrl, text="暂停/继续步进", command=self.on_toggle_pause).grid(
            row=17, column=0, columnspan=2, pady=2, sticky=tk.EW
        )
        ttk.Separator(ctrl, orient=tk.HORIZONTAL).grid(
            row=18, column=0, columnspan=2, sticky=tk.EW, pady=6
        )
        ttk.Button(ctrl, text="保存地图 PNG", command=self.on_save_png).grid(
            row=19, column=0, columnspan=2, sticky=tk.EW
        )
        ttk.Separator(ctrl, orient=tk.HORIZONTAL).grid(
            row=20, column=0, columnspan=2, sticky=tk.EW, pady=6
        )

        adv = ttk.LabelFrame(ctrl, text="高级策略参数", padding=4)
        adv.grid(row=21, column=0, columnspan=2, sticky=tk.EW)

        self.var_auto_solver = tk.BooleanVar(value=bool(g.get("auto_solver", True)))
        self.var_fallback_backtrack = tk.BooleanVar(value=bool(g.get("fallback_backtrack", True)))
        ttk.Checkbutton(adv, text="启用自动策略", variable=self.var_auto_solver).grid(
            row=0, column=0, columnspan=2, sticky=tk.W
        )
        ttk.Checkbutton(adv, text="回溯兜底", variable=self.var_fallback_backtrack).grid(
            row=1, column=0, columnspan=2, sticky=tk.W
        )

        self.var_astar_max_n = tk.StringVar(value=str(g.get("astar_max_n", 20)))
        self.var_weighted_max_n = tk.StringVar(
            value="" if g.get("weighted_max_n") is None else str(g.get("weighted_max_n"))
        )
        self.var_weighted_eps = tk.StringVar(
            value="" if g.get("weighted_astar_epsilon") is None else str(g.get("weighted_astar_epsilon"))
        )
        self.var_sa_iterations = tk.StringVar(value=str(g.get("fallback_sa_iterations", 8000)))
        self.var_sa_t0 = tk.StringVar(value=str(g.get("fallback_sa_T0", 1.0)))
        self.var_sa_tmin = tk.StringVar(value=str(g.get("fallback_sa_T_min", 1e-4)))
        self.var_sa_cool = tk.StringVar(value=str(g.get("fallback_sa_cool", 0.995)))
        self.var_sa_seed = tk.StringVar(
            value="" if g.get("fallback_sa_seed") is None else str(g.get("fallback_sa_seed"))
        )
        self.var_rnn_tries = tk.StringVar(value=str(g.get("fallback_random_nn_tries", 500)))
        self.var_rcl_size = tk.StringVar(value=str(g.get("fallback_rcl_size", 4)))
        self.var_bt_max_steps = tk.StringVar(value=str(g.get("fallback_backtrack_max_steps", 3_000_000)))
        self.var_bt_restarts = tk.StringVar(value=str(g.get("fallback_backtrack_restarts", 24)))
        self.var_bt_time = tk.StringVar(
            value="" if g.get("fallback_backtrack_time_sec") is None else str(g.get("fallback_backtrack_time_sec"))
        )
        self.var_verify_max_n = tk.StringVar(value=str(g.get("verify_exact_max_n", 14)))
        self.var_verify_time = tk.StringVar(
            value="" if g.get("verify_exact_time_sec") is None else str(g.get("verify_exact_time_sec"))
        )
        self.var_stronger_max_n = tk.StringVar(value=str(g.get("stronger_exact_max_n", 18)))
        self.var_stronger_time = tk.StringVar(
            value=""
            if g.get("stronger_exact_time_limit_sec") is None
            else str(g.get("stronger_exact_time_limit_sec"))
        )
        self.var_bf_max_n = tk.StringVar(value=str(g.get("bruteforce_max_n", 11)))
        self.var_bf_time = tk.StringVar(
            value="" if g.get("bruteforce_time_limit_sec") is None else str(g.get("bruteforce_time_limit_sec"))
        )
        self.var_monster_max_n = tk.StringVar(value=str(g.get("monster_max_n", 24)))
        self.var_monster_time = tk.StringVar(
            value="" if g.get("monster_time_limit_sec") is None else str(g.get("monster_time_limit_sec"))
        )

        fields = [
            ("A*阈值 n≤", self.var_astar_max_n),
            ("Weighted阈值 n≤(空禁用)", self.var_weighted_max_n),
            ("Weighted ε(空禁用)", self.var_weighted_eps),
            ("SA迭代数", self.var_sa_iterations),
            ("SA T0", self.var_sa_t0),
            ("SA T_min", self.var_sa_tmin),
            ("SA 冷却率", self.var_sa_cool),
            ("SA seed(空随机)", self.var_sa_seed),
            ("随机NN次数", self.var_rnn_tries),
            ("RCL大小", self.var_rcl_size),
            ("回溯最大步数", self.var_bt_max_steps),
            ("回溯重启数", self.var_bt_restarts),
            ("回溯时限s(空无)", self.var_bt_time),
            ("验证器最大n", self.var_verify_max_n),
            ("验证器时限s(空无)", self.var_verify_time),
            ("强算法DP最大n", self.var_stronger_max_n),
            ("强算法DP时限s(空无)", self.var_stronger_time),
            ("Monster最大n", self.var_monster_max_n),
            ("Monster时限s(空无)", self.var_monster_time),
            ("暴力最大n", self.var_bf_max_n),
            ("暴力时限s(空无)", self.var_bf_time),
        ]
        rr = 2
        for label, var in fields:
            ttk.Label(adv, text=label).grid(row=rr, column=0, sticky=tk.W)
            ttk.Entry(adv, textvariable=var, width=14).grid(row=rr, column=1, sticky=tk.EW)
            rr += 1

        ttk.Button(ctrl, text="导出当前参数JSON", command=self.on_export_gui_params).grid(
            row=22, column=0, columnspan=2, pady=2, sticky=tk.EW
        )
        ttk.Button(ctrl, text="导入参数JSON", command=self.on_import_gui_params).grid(
            row=23, column=0, columnspan=2, pady=2, sticky=tk.EW
        )

        plt.ioff()
        self.fig, self.ax = plt.subplots(figsize=(6.5, 6.5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=main)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        log_frame = ttk.LabelFrame(main, text="日志 (f, g, h, 扩展次数)", padding=4)
        log_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, before=ctrl)
        self.log = tk.Text(log_frame, width=36, height=28, wrap=tk.WORD)
        self.log.pack(fill=tk.BOTH, expand=True)

    def _log(self, s: str) -> None:
        self.log.insert(tk.END, s + "\n")
        self.log.see(tk.END)

    def _log_run_summary(self, res: SearchResult) -> None:
        """与实验表格字段对齐；单次成功时最优/最差/平均路径长度均为本次 cost。"""
        n = self.tsp.n if self.tsp else 0
        self._log("—— 汇总（单次当前地图；多样本统计请用 main.py tables）——")
        self._log(f"城市规模:\t{n}")
        if res.success:
            c = res.cost
            self._log(f"最优路径长度:\t{c:.6f}")
            self._log(f"最差路径长度:\t{c:.6f}")
            self._log(f"平均路径长度:\t{c:.6f}")
        else:
            self._log("最优路径长度:\t(未成功)")
            self._log("最差路径长度:\t(未成功)")
            self._log("平均路径长度:\t(未成功)")
        self._log(f"总迭代次数:\t{res.expansions}")
        self._log(f"总运行时间_s:\t{res.elapsed_sec:.6f}")
        self._log(f"平均运行时间_s:\t{res.elapsed_sec:.6f}\t(本行为单次搜索总耗时)")
        if res.expansions > 0:
            self._log(f"平均每步时间_s:\t{res.elapsed_sec / res.expansions:.6f}")
        if res.optimal:
            optimal_text = "是(已认证)"
        elif "verified suboptimal" in (res.message or ""):
            optimal_text = "否(已证次优)"
        else:
            optimal_text = "未认证"
        self._log(f"求解方法:\t{res.method}\t最优解:\t{optimal_text}")

    def _log_feasibility_estimate(self, start: Optional[int] = None) -> None:
        if self.tsp is None:
            return
        try:
            if start is None:
                start = self._resolve_start(self.tsp.n)
            est = estimate_feasible_closed_tour(self.tsp, start=start)
            self._log(f"可行性估算起点: {start}")
            for line in format_feasibility_log_lines(est):
                self._log(line)
        except Exception as e:
            self._log(f"可行解估算失败: {e}")

    def _post_verify_result(self, res: Optional[SearchResult], start: int) -> Optional[SearchResult]:
        if self.tsp is None or res is None:
            return res
        if res.success:
            chk = validate_closed_tour(self.tsp, res.tour, start=start, cost=res.cost)
            if not chk.valid:
                return SearchResult(
                    False,
                    list(res.tour),
                    float(chk.computed_cost),
                    res.expansions,
                    res.elapsed_sec,
                    f"solution invalid: {chk.message}",
                    method=res.method,
                    optimal=False,
                )
            res.tour = chk.normalized_tour
            res.cost = chk.computed_cost

        try:
            vmax = self._parse_required_int(self.var_verify_max_n.get(), "验证器最大n")
            vtime = self._parse_optional_float(self.var_verify_time.get())
        except Exception:
            vmax = int(self.g.get("verify_exact_max_n", 14))
            vtime_raw = self.g.get("verify_exact_time_sec")
            vtime = float(vtime_raw) if vtime_raw is not None else None
        if self.tsp.n <= vmax:
            cert = exact_optimal_tour_small_n(self.tsp, start=start, time_limit_sec=vtime)
            if cert.certified and cert.optimal_cost is not None and res.success:
                if abs(res.cost - cert.optimal_cost) <= 1e-8:
                    res.optimal = True
                else:
                    res.optimal = False
                    res.message = (
                        (res.message + "; ") if res.message else ""
                    ) + f"verified suboptimal: gap={res.cost - cert.optimal_cost:.6g}"
            elif not cert.certified and res.success and cert.message:
                res.message = ((res.message + "; ") if res.message else "") + cert.message
        elif res.success:
            note = f"optimality uncertified: n={self.tsp.n} > verify_exact_max_n={vmax}"
            res.message = ((res.message + "; ") if res.message else "") + note
        return res

    def _disable_cancel_astar_button(self) -> None:
        self._astar_bg_running = False
        self.btn_cancel_astar.config(state=tk.DISABLED)

    def on_cancel_astar(self) -> None:
        if self._astar_bg_running or self.gen is not None:
            self._cancel_astar.set()
            self._log("已请求终止 A*（在扩展边界与邻接扩展处检查，稍后结束）…")

    def _policy_cfg(self) -> Dict[str, Any]:
        return {
            "auto_solver": bool(self.var_auto_solver.get()),
            "force_exact_astar": bool(self.var_force_exact.get()),
            "force_stronger_solver": bool(self.var_force_stronger.get()),
            "force_monster_solver": bool(self.var_force_monster.get()),
            "force_bruteforce": bool(self.var_force_bruteforce.get()),
            "monster_max_n": self._parse_required_int(self.var_monster_max_n.get(), "Monster最大n"),
            "monster_time_limit_sec": self._parse_optional_float(self.var_monster_time.get()),
            "bruteforce_max_n": self._parse_required_int(self.var_bf_max_n.get(), "暴力最大n"),
            "bruteforce_time_limit_sec": self._parse_optional_float(self.var_bf_time.get()),
            "stronger_exact_max_n": self._parse_required_int(self.var_stronger_max_n.get(), "强算法DP最大n"),
            "stronger_exact_time_limit_sec": self._parse_optional_float(self.var_stronger_time.get()),
            "astar_max_n": self._parse_required_int(self.var_astar_max_n.get(), "A*阈值"),
            "weighted_max_n": self._parse_optional_int(self.var_weighted_max_n.get()),
            "weighted_astar_epsilon": self._parse_optional_float(self.var_weighted_eps.get()),
            "fallback_sa_iterations": self._parse_required_int(self.var_sa_iterations.get(), "SA迭代数"),
            "fallback_sa_T0": self._parse_required_float(self.var_sa_t0.get(), "SA T0"),
            "fallback_sa_T_min": self._parse_required_float(self.var_sa_tmin.get(), "SA T_min"),
            "fallback_sa_cool": self._parse_required_float(self.var_sa_cool.get(), "SA冷却率"),
            "fallback_sa_seed": self._parse_optional_int(self.var_sa_seed.get()),
            "fallback_random_nn_tries": self._parse_required_int(self.var_rnn_tries.get(), "随机NN次数"),
            "fallback_rcl_size": self._parse_required_int(self.var_rcl_size.get(), "RCL大小"),
            "fallback_backtrack": bool(self.var_fallback_backtrack.get()),
            "fallback_backtrack_max_steps": self._parse_required_int(self.var_bt_max_steps.get(), "回溯最大步数"),
            "fallback_backtrack_restarts": self._parse_required_int(self.var_bt_restarts.get(), "回溯重启数"),
            "fallback_backtrack_time_sec": self._parse_optional_float(self.var_bt_time.get()),
            "verify_exact_max_n": self._parse_required_int(self.var_verify_max_n.get(), "验证器最大n"),
            "verify_exact_time_sec": self._parse_optional_float(self.var_verify_time.get()),
        }

    def _parse_optional_int(self, s: str) -> Optional[int]:
        s = s.strip()
        return None if not s else int(s)

    def _parse_optional_float(self, s: str) -> Optional[float]:
        s = s.strip()
        return None if not s else float(s)

    def _parse_required_int(self, s: str, field: str) -> int:
        s = s.strip()
        if not s:
            raise ValueError(f"{field} 不能为空")
        return int(s)

    def _parse_required_float(self, s: str, field: str) -> float:
        s = s.strip()
        if not s:
            raise ValueError(f"{field} 不能为空")
        return float(s)

    def _to_bool(self, v: Any) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, np.integer)):
            return int(v) != 0
        if isinstance(v, str):
            s = v.strip().lower()
            if s in {"1", "true", "yes", "y", "on"}:
                return True
            if s in {"0", "false", "no", "n", "off"}:
                return False
        raise ValueError(f"invalid bool value: {v!r}")

    def _collect_gui_params(self) -> Dict[str, Any]:
        return {
            "n": self.var_n.get(),
            "k": self.var_k.get(),
            "heuristic": self.var_h.get(),
            "start": self.var_start.get(),
            "max_expansions": self.var_max_exp.get(),
            "time_limit_sec": self.var_time.get(),
            "force_exact_astar": bool(self.var_force_exact.get()),
            "force_stronger_solver": bool(self.var_force_stronger.get()),
            "force_monster_solver": bool(self.var_force_monster.get()),
            "force_bruteforce": bool(self.var_force_bruteforce.get()),
            "auto_solver": bool(self.var_auto_solver.get()),
            "fallback_backtrack": bool(self.var_fallback_backtrack.get()),
            "astar_max_n": self.var_astar_max_n.get(),
            "weighted_max_n": self.var_weighted_max_n.get(),
            "weighted_astar_epsilon": self.var_weighted_eps.get(),
            "fallback_sa_iterations": self.var_sa_iterations.get(),
            "fallback_sa_T0": self.var_sa_t0.get(),
            "fallback_sa_T_min": self.var_sa_tmin.get(),
            "fallback_sa_cool": self.var_sa_cool.get(),
            "fallback_sa_seed": self.var_sa_seed.get(),
            "fallback_random_nn_tries": self.var_rnn_tries.get(),
            "fallback_rcl_size": self.var_rcl_size.get(),
            "fallback_backtrack_max_steps": self.var_bt_max_steps.get(),
            "fallback_backtrack_restarts": self.var_bt_restarts.get(),
            "fallback_backtrack_time_sec": self.var_bt_time.get(),
            "verify_exact_max_n": self.var_verify_max_n.get(),
            "verify_exact_time_sec": self.var_verify_time.get(),
            "stronger_exact_max_n": self.var_stronger_max_n.get(),
            "stronger_exact_time_limit_sec": self.var_stronger_time.get(),
            "monster_max_n": self.var_monster_max_n.get(),
            "monster_time_limit_sec": self.var_monster_time.get(),
            "bruteforce_max_n": self.var_bf_max_n.get(),
            "bruteforce_time_limit_sec": self.var_bf_time.get(),
        }

    def _apply_gui_params(self, p: Dict[str, Any]) -> None:
        if "n" in p:
            self.var_n.set(str(p["n"]))
        if "k" in p:
            self.var_k.set(str(p["k"]))
        if "heuristic" in p:
            self.var_h.set(str(p["heuristic"]))
        if "start" in p:
            self.var_start.set(str(p["start"]))
        if "max_expansions" in p:
            self.var_max_exp.set(str(p["max_expansions"]))
        if "time_limit_sec" in p:
            self.var_time.set(str(p["time_limit_sec"]))
        if "force_exact_astar" in p:
            self.var_force_exact.set(self._to_bool(p["force_exact_astar"]))
        if "force_stronger_solver" in p:
            self.var_force_stronger.set(self._to_bool(p["force_stronger_solver"]))
        if "force_monster_solver" in p:
            self.var_force_monster.set(self._to_bool(p["force_monster_solver"]))
        if "force_bruteforce" in p:
            self.var_force_bruteforce.set(self._to_bool(p["force_bruteforce"]))
        if "auto_solver" in p:
            self.var_auto_solver.set(self._to_bool(p["auto_solver"]))
        if "fallback_backtrack" in p:
            self.var_fallback_backtrack.set(self._to_bool(p["fallback_backtrack"]))

        txt_map: Dict[str, tk.StringVar] = {
            "astar_max_n": self.var_astar_max_n,
            "weighted_max_n": self.var_weighted_max_n,
            "weighted_astar_epsilon": self.var_weighted_eps,
            "fallback_sa_iterations": self.var_sa_iterations,
            "fallback_sa_T0": self.var_sa_t0,
            "fallback_sa_T_min": self.var_sa_tmin,
            "fallback_sa_cool": self.var_sa_cool,
            "fallback_sa_seed": self.var_sa_seed,
            "fallback_random_nn_tries": self.var_rnn_tries,
            "fallback_rcl_size": self.var_rcl_size,
            "fallback_backtrack_max_steps": self.var_bt_max_steps,
            "fallback_backtrack_restarts": self.var_bt_restarts,
            "fallback_backtrack_time_sec": self.var_bt_time,
            "verify_exact_max_n": self.var_verify_max_n,
            "verify_exact_time_sec": self.var_verify_time,
            "stronger_exact_max_n": self.var_stronger_max_n,
            "stronger_exact_time_limit_sec": self.var_stronger_time,
            "monster_max_n": self.var_monster_max_n,
            "monster_time_limit_sec": self.var_monster_time,
            "bruteforce_max_n": self.var_bf_max_n,
            "bruteforce_time_limit_sec": self.var_bf_time,
        }
        for k, var in txt_map.items():
            if k in p:
                var.set(str(p[k]))

    def on_export_gui_params(self) -> None:
        path = filedialog.asksaveasfilename(
            title="导出参数 JSON",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        payload = {
            "schema": "astar_tsp_gui_params_v1",
            "params": self._collect_gui_params(),
        }
        try:
            Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            self._log(f"已导出参数: {path}")
        except Exception as e:
            messagebox.showerror("导出失败", str(e))

    def on_import_gui_params(self) -> None:
        path = filedialog.askopenfilename(title="导入参数 JSON", filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            obj = json.loads(Path(path).read_text(encoding="utf-8"))
            params = obj["params"] if isinstance(obj, dict) and "params" in obj else obj
            if not isinstance(params, dict):
                raise ValueError("JSON 顶层应为对象或包含 params 对象")
            self._apply_gui_params(params)
            self._log(f"已导入参数: {path}")
        except Exception as e:
            messagebox.showerror("导入失败", str(e))

    def _resolve_start(self, n: int) -> int:
        raw = self.var_start.get().strip()
        if raw == "" or raw.lower() in {"random", "rand"}:
            if self._frozen_random_start is None:
                self._frozen_random_start = int(np.random.default_rng().integers(0, n))
            return int(self._frozen_random_start) % n
        self._frozen_random_start = None
        return int(raw) % n

    def _draw_static(
        self,
        partial_path: Optional[List[int]] = None,
        *,
        tour: Optional[List[int]] = None,
    ) -> None:
        if self.tsp is None:
            return
        self.ax.clear()
        tr: Optional[List[int]] = None
        pp: Optional[List[int]] = None
        if tour is not None and len(tour) >= 2:
            tr = list(tour)
        elif partial_path is not None and len(partial_path) >= 2:
            p = list(partial_path)
            if p[0] == p[-1] and len(p) >= 3:
                tr = p
            else:
                pp = p
        draw_tsp(
            self.ax,
            self.tsp,
            city_size=35,
            annotate_fontsize=7,
            edge_lw=0.7,
        )
        # 始终绘制初始化最佳回路，使其在搜索过程中持续可见。
        if self._init_best_tour and len(self._init_best_tour) >= 2:
            pts = self.tsp.points
            gx = [pts[i, 0] for i in self._init_best_tour]
            gy = [pts[i, 1] for i in self._init_best_tour]
            ring_color = "gold" if self._init_best_tour_certified_optimal else "dodgerblue"
            ring_label = (
                "best-known (certified optimal)"
                if self._init_best_tour_certified_optimal
                else "best-known (uncertified)"
            )
            # 双层线实现“中空金色”效果：外层金色，内层白色。
            self.ax.plot(gx, gy, color=ring_color, lw=3.6, zorder=5, alpha=0.95, label=ring_label)
            self.ax.plot(gx, gy, color="white", lw=1.8, zorder=6, alpha=0.95)
        # 搜索中的路径置于更上层，便于同时观察“最佳已知”与“当前搜索”。
        if tr is not None and len(tr) >= 2:
            tx = [self.tsp.points[i, 0] for i in tr]
            ty = [self.tsp.points[i, 1] for i in tr]
            self.ax.plot(tx, ty, "r--", lw=1.7, zorder=7, label="tour")
        if pp is not None and len(pp) >= 2:
            xs = [self.tsp.points[i, 0] for i in pp]
            ys = [self.tsp.points[i, 1] for i in pp]
            self.ax.plot(xs, ys, "r-", lw=1.6, zorder=8, alpha=0.9)
        if self._init_best_tour or tr is not None:
            self.ax.legend()
        self.ax.set_title(f"TSP n={self.tsp.n}")
        self.canvas.draw()

    def _compute_and_set_initial_best_tour(self) -> None:
        if self.tsp is None:
            return
        tsp = self.tsp
        try:
            start_i = self._resolve_start(tsp.n)
        except Exception:
            start_i = 0
        self._init_best_tour = None
        self._init_best_tour_certified_optimal = False

        # 先尝试精确 Held-Karp（小规模），否则回退当前策略求高质量可行解。
        try:
            hk_max = self._parse_required_int(self.var_stronger_max_n.get(), "强算法DP最大n")
            hk_tl = self._parse_optional_float(self.var_stronger_time.get())
        except Exception:
            hk_max = int(self.g.get("stronger_exact_max_n", 18))
            hk_tl_raw = self.g.get("stronger_exact_time_limit_sec")
            hk_tl = float(hk_tl_raw) if hk_tl_raw is not None else None

        if tsp.n <= hk_max:
            cert = exact_optimal_tour_small_n(tsp, start=start_i, time_limit_sec=hk_tl)
            if cert.certified and cert.optimal_cost is not None and cert.optimal_tour:
                self._init_best_tour = list(cert.optimal_tour)
                self._init_best_tour_certified_optimal = True
                self._log(
                    f"初始化金色高亮: 已找到可证最优回路 "
                    f"(Held-Karp, cost={cert.optimal_cost:.6f}, start={start_i})"
                )
                return
            if cert.certified and cert.optimal_cost is None:
                self._log("初始化金色高亮: 该图在当前起点下无可行闭合回路。")
                return

        try:
            pcfg = self._policy_cfg()
        except Exception:
            pcfg = dict(self.g)
        try:
            max_e = self._parse_optional_int(self.var_max_exp.get())
            max_t = self._parse_optional_float(self.var_time.get())
        except Exception:
            max_e = None
            max_t = None
        res = solve_tsp_auto(
            tsp,
            self.var_h.get(),
            start_i,
            pcfg,
            max_expansions=max_e,
            time_limit_sec=max_t,
        )
        if res.success and res.tour:
            self._init_best_tour = list(res.tour)
            self._init_best_tour_certified_optimal = bool(res.optimal)
            self._log(
                f"初始化{'金色' if self._init_best_tour_certified_optimal else '蓝色'}高亮: 已找到最佳已知回路 "
                f"(method={res.method}, cost={res.cost:.6f}, start={start_i})"
            )
        else:
            self._log("初始化金色高亮: 未得到可行回路。")

    def on_generate(self) -> None:
        try:
            n = int(self.var_n.get())
            k = int(self.var_k.get())
        except ValueError:
            messagebox.showerror("错误", "n 与 k 须为整数")
            return
        if k < 3 or k > 5:
            messagebox.showwarning("提示", "实验建议 k 近邻取 3～5")
        seed = int(np.random.default_rng().integers(0, 2**31 - 1))
        g = generate_random_map(n, k, seed=seed)
        g = ensure_connected_fallback(g, np.random.default_rng(seed))
        self.tsp = g
        self._frozen_random_start = None
        self.gen = None
        self._cancel_astar.clear()
        self._disable_cancel_astar_button()
        self._log(f"已生成 n={n}, k={k}, seed={seed}")
        self._log_feasibility_estimate()
        self._compute_and_set_initial_best_tour()
        self._draw_static()

    def on_load(self) -> None:
        pcsv = filedialog.askopenfilename(title="选择 points.csv", filetypes=[("CSV", "*.csv")])
        if not pcsv:
            return
        ecsv = filedialog.askopenfilename(title="选择 edges.csv", filetypes=[("CSV", "*.csv")])
        if not ecsv:
            return
        try:
            self.tsp = load_instance(pcsv, ecsv)
        except Exception as e:
            messagebox.showerror("加载失败", str(e))
            return
        self._frozen_random_start = None
        self.gen = None
        self._cancel_astar.clear()
        self._disable_cancel_astar_button()
        self.var_n.set(str(self.tsp.n))
        self._log(f"已加载 n={self.tsp.n}")
        self._log_feasibility_estimate()
        self._compute_and_set_initial_best_tour()
        self._draw_static()

    def on_save_png(self) -> None:
        if self.tsp is None:
            messagebox.showwarning("提示", "请先生成或加载实例")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png")],
        )
        if not path:
            return
        save_map_png(self.tsp, path, title=f"TSP n={self.tsp.n}")
        self._log(f"已保存 {path}")

    def on_run_full(self) -> None:
        if self.tsp is None:
            messagebox.showwarning("提示", "请先生成或加载实例")
            return
        hname = self.var_h.get()
        try:
            get_heuristic(hname)
        except KeyError as e:
            messagebox.showerror("错误", str(e))
            return
        try:
            start_i = self._resolve_start(self.tsp.n)
        except ValueError:
            messagebox.showerror("错误", "起点须为整数，或留空/random 表示随机")
            return
        try:
            max_e = self._parse_optional_int(self.var_max_exp.get())
            max_t = self._parse_optional_float(self.var_time.get())
            pcfg = self._policy_cfg()
        except ValueError as e:
            messagebox.showerror("错误", f"参数格式错误：{e}")
            return
        tsp = self.tsp
        mode = recommend_solver_mode(tsp.n, pcfg)
        force_non_astar = bool(
            pcfg.get("force_stronger_solver")
            or pcfg.get("force_monster_solver")
            or pcfg.get("force_bruteforce")
        ) and (not pcfg.get("force_astar_only", False))
        ub = state_space_upper_bound(tsp.n)
        self._cancel_astar.clear()
        self._astar_bg_running = (mode in ("exact", "weighted")) and (not force_non_astar)
        if self._astar_bg_running:
            self.btn_cancel_astar.config(state=tk.NORMAL)
        else:
            self._disable_cancel_astar_button()

        def work() -> None:
            res: Optional[SearchResult] = None
            if mode == "heuristic" or force_non_astar:
                res = solve_tsp_auto(
                    tsp, hname, start_i, pcfg, max_expansions=max_e, time_limit_sec=max_t
                )
                res = self._post_verify_result(res, start_i)

                def finish_h() -> None:
                    self._disable_cancel_astar_button()
                    if res is None:
                        self._draw_static()
                        self._log("完成: (无结果)")
                        return
                    self._log(
                        f"规模策略={mode}，状态空间上界≈{ub}；"
                        f"method={res.method}"
                    )
                    if res.success and res.tour:
                        self._draw_static(tour=list(res.tour))
                    else:
                        if res.tour and len(res.tour) >= 2:
                            self._draw_static(tour=list(res.tour))
                        else:
                            self._draw_static()
                    self._log(
                        f"完成: cost={res.cost:.6f}, expansions={res.expansions}, "
                        f"time={res.elapsed_sec:.4f}s"
                    )
                    self._log_run_summary(res)

                self._safe_after(0, finish_h)
                return

            eps = (
                float(pcfg["weighted_astar_epsilon"])
                if mode == "weighted" and pcfg.get("weighted_astar_epsilon") is not None
                else 1.0
            )
            solver = AStarTSPSolver(tsp, heuristic=hname, start=start_i)
            prog_kw = astar_search_progress_kwargs(pcfg)
            for ev in solver.search_stepwise(
                max_expansions=max_e,
                time_limit_sec=max_t,
                epsilon=eps,
                cancel_check=self._cancel_astar.is_set,
                **prog_kw,
            ):
                et = ev.get("event")
                if et == EVENT_POP:
                    path = ev.get("path")
                    if isinstance(path, list) and len(path) >= 2:
                        snap = list(path)
                        if snap[0] == snap[-1] and len(snap) >= 3:
                            self._safe_after(0, lambda t=snap: self._draw_static(tour=t))
                        else:
                            self._safe_after(0, lambda p=snap: self._draw_static(p))
                elif et == EVENT_GOAL:
                    tour = ev.get("tour", [])
                    if isinstance(tour, list) and len(tour) >= 2:
                        tr = list(tour)
                        self._safe_after(0, lambda t=tr: self._draw_static(tour=t))
                elif et == EVENT_DONE:
                    r = ev.get("result")
                    if isinstance(r, SearchResult):
                        res = self._post_verify_result(r, start_i)

            def finish() -> None:
                self._disable_cancel_astar_button()
                if res is None:
                    self._draw_static()
                    self._log("完成: (无结果)")
                    return
                if res.success and res.tour:
                    self._draw_static(tour=list(res.tour))
                else:
                    if res.tour and len(res.tour) >= 2:
                        self._draw_static(tour=list(res.tour))
                    else:
                        self._draw_static()
                self._log(
                    f"完成: mode={mode}, cost={res.cost:.6f}, expansions={res.expansions}, "
                    f"time={res.elapsed_sec:.4f}s"
                )
                self._log_run_summary(res)

            self._safe_after(0, finish)

        threading.Thread(target=work, daemon=True).start()
        if mode == "heuristic" or force_non_astar:
            self._log(f"后台运行 非A*模式 … (start={start_i}, mode={mode}, 状态空间上界≈{ub})")
        else:
            self._log(f"后台运行 A* … (start={start_i}, mode={mode}, 状态空间上界≈{ub})")

    def on_step_start(self) -> None:
        if self.tsp is None:
            messagebox.showwarning("提示", "请先生成或加载实例")
            return
        if self.gen is None:
            try:
                get_heuristic(self.var_h.get())
            except KeyError as e:
                messagebox.showerror("错误", str(e))
                return
            try:
                start_i = self._resolve_start(self.tsp.n)
            except ValueError:
                messagebox.showerror("错误", "起点须为整数，或留空/random 表示随机")
                return
            try:
                pcfg = self._policy_cfg()
                max_e = self._parse_optional_int(self.var_max_exp.get())
                max_t = self._parse_optional_float(self.var_time.get())
            except ValueError as e:
                messagebox.showerror("错误", f"参数格式错误：{e}")
                return
            if bool(
                pcfg.get("force_stronger_solver")
                or pcfg.get("force_monster_solver")
                or pcfg.get("force_bruteforce")
            ) and (not pcfg.get("force_astar_only", False)):
                self._log("当前模式不支持步进展示，自动改为后台运行到结束。")
                self.on_run_full()
                return
            mode = recommend_solver_mode(self.tsp.n, pcfg)
            self._cancel_astar.clear()
            if mode == "heuristic":
                self.btn_cancel_astar.config(state=tk.DISABLED)
                self.gen = search_heuristic_stepwise(
                    self.tsp,
                    start_i,
                    pcfg,
                    max_expansions=max_e,
                    time_limit_sec=max_t,
                )
            else:
                self.btn_cancel_astar.config(state=tk.NORMAL)
                eps = (
                    float(pcfg["weighted_astar_epsilon"])
                    if mode == "weighted" and pcfg.get("weighted_astar_epsilon") is not None
                    else 1.0
                )
                solver = AStarTSPSolver(self.tsp, heuristic=self.var_h.get(), start=start_i)
                prog_kw = astar_search_progress_kwargs(pcfg)
                self.gen = solver.search_stepwise(
                    max_expansions=max_e,
                    time_limit_sec=max_t,
                    epsilon=eps,
                    cancel_check=self._cancel_astar.is_set,
                    **prog_kw,
                )
            self.paused = False
            self._cum_search_s = 0.0
            self._step_start = start_i
            self._draw_static()
            self._log(
                "--- 新步进搜索（耗时仅计搜索代码，不含挂机等待）---"
                + f" [start={start_i}]"
                + (" [启发式: 贪心→SA]" if mode == "heuristic" else "")
            )
        self._pump_one_step()

    def on_toggle_pause(self) -> None:
        self.paused = not self.paused
        self._log("暂停" if self.paused else "继续")
        if not self.paused and self.auto_play.get():
            self._schedule_auto()

    def _schedule_auto(self) -> None:
        if self.step_job:
            try:
                self.root.after_cancel(self.step_job)
            except tk.TclError:
                pass
            self.step_job = None
        if (not self._is_alive()) or self.gen is None or self.paused or not self.auto_play.get():
            return
        try:
            self.step_job = self.root.after(30, self._auto_tick)
        except tk.TclError:
            self.step_job = None

    def _auto_tick(self) -> None:
        self.step_job = None
        if (not self._is_alive()) or self.gen is None or self.paused:
            return
        self._pump_one_step()
        if self.auto_play.get() and not self.paused and self.gen is not None and self._is_alive():
            try:
                self.step_job = self.root.after(30, self._auto_tick)
            except tk.TclError:
                self.step_job = None

    def _pump_one_step(self) -> None:
        """一次「步进」对应一次扩展（POP）；内部吞掉紧随其后的 PUSH，避免连点无反馈。"""
        if self.gen is None:
            return
        t0 = time.perf_counter()
        try:
            ev = next(self.gen)
            while ev.get("event") == EVENT_PUSH:
                ev = next(self.gen)
        except StopIteration:
            self.gen = None
            self.btn_cancel_astar.config(state=tk.DISABLED)
            return
        dt_step = time.perf_counter() - t0
        self._cum_search_s += dt_step

        et = ev.get("event")
        if et == EVENT_POP:
            path = ev.get("path")
            if isinstance(path, list) and len(path) >= 2:
                p = list(path)
                if p[0] == p[-1] and len(p) >= 3:
                    self._draw_static(tour=p)
                else:
                    self._draw_static(p)
            else:
                self._draw_static()
            if ev.get("heuristic"):
                st = ev.get("stage", "")
                self._log(
                    f"[启发式:{st}] 路径顶点数={len(path) if isinstance(path, list) else 0} "
                    f"exp={ev.get('expansions')} "
                    f"本步搜索耗时_s={dt_step:.6f} 累计搜索耗时_s={self._cum_search_s:.6f}"
                )
            else:
                self._log(
                    f"POP  city={ev.get('current')} phase={ev.get('phase')} "
                    f"g={ev.get('g'):.4f} h={ev.get('h'):.4f} f={ev.get('f'):.4f} "
                    f"exp={ev.get('expansions')} "
                    f"本步搜索耗时_s={dt_step:.6f} 累计搜索耗时_s={self._cum_search_s:.6f}"
                )
        elif et == EVENT_GOAL:
            tour = ev.get("tour", [])
            if isinstance(tour, list) and len(tour) >= 2:
                tr = list(tour)
                if tr[0] == tr[-1] and len(tr) >= 3:
                    self._draw_static(tour=tr)
                else:
                    self._draw_static(tr)
            else:
                self._draw_static()
            st = ev.get("stage", "")
            if ev.get("heuristic"):
                self._log(
                    f"GOAL [{st}] cost={ev.get('cost'):.6f} tour={tour} "
                    f"本步搜索耗时_s={dt_step:.6f} 累计搜索耗时_s={self._cum_search_s:.6f}"
                )
                return
            self._log(
                f"GOAL cost={ev.get('cost'):.6f} tour={tour} "
                f"本步搜索耗时_s={dt_step:.6f} 累计搜索耗时_s={self._cum_search_s:.6f}"
            )
            t1 = time.perf_counter()
            try:
                ev2 = next(self.gen)
            except StopIteration:
                self.gen = None
                self.btn_cancel_astar.config(state=tk.DISABLED)
                self._cum_search_s += time.perf_counter() - t1
                return
            self._cum_search_s += time.perf_counter() - t1
            if ev2.get("event") == EVENT_DONE:
                r = ev2.get("result")
                if isinstance(r, SearchResult):
                    r = self._post_verify_result(r, self._step_start) or r
                    if r.success:
                        self._log(f"DONE 成功 cost={r.cost:.6f} 累计搜索耗时_s={self._cum_search_s:.6f}")
                    else:
                        self._log(f"DONE 失败 {r.message} 累计搜索耗时_s={self._cum_search_s:.6f}")
                    self._log_run_summary(r)
            self.gen = None
            self.btn_cancel_astar.config(state=tk.DISABLED)
            return
        elif et == EVENT_DONE:
            r = ev.get("result")
            if isinstance(r, SearchResult):
                r = self._post_verify_result(r, self._step_start) or r
                if r.tour and len(r.tour) >= 2:
                    tt = list(r.tour)
                    if tt[0] == tt[-1] and len(tt) >= 3:
                        self._draw_static(tour=tt)
                    else:
                        self._draw_static(tt)
                if r.success:
                    self._log(
                        f"DONE 成功 cost={r.cost:.6f} "
                        f"本步搜索耗时_s={dt_step:.6f} 累计搜索耗时_s={self._cum_search_s:.6f}"
                    )
                else:
                    self._log(
                        f"DONE 失败 {r.message} "
                        f"本步搜索耗时_s={dt_step:.6f} 累计搜索耗时_s={self._cum_search_s:.6f}"
                    )
                self._log_run_summary(r)
            self.gen = None
            self.btn_cancel_astar.config(state=tk.DISABLED)
            return

        if self.auto_play.get() and not self.paused:
            self._schedule_auto()


def launch_gui(gui_cfg: Optional[Dict[str, Any]] = None) -> None:
    root = tk.Tk()
    app = AStarGUI(root, gui_cfg)
    root.minsize(900, 640)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.on_close()
