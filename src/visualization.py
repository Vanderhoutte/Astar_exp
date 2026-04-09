"""TSP 静态图（PNG）与 Tk 过程演示 GUI。"""

from __future__ import annotations

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
        "default_max_expansions": "500000",
        "default_time_limit": "",
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
        self._cum_search_s: float = 0.0
        self._build()

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

        ttk.Label(ctrl, text="扩展上限(空=无)").grid(row=3, column=0, sticky=tk.W)
        self.var_max_exp = tk.StringVar(value=str(g.get("default_max_expansions", "500000")))
        ttk.Entry(ctrl, textvariable=self.var_max_exp, width=10).grid(row=3, column=1)

        ttk.Label(ctrl, text="时限秒(空=无)").grid(row=4, column=0, sticky=tk.W)
        self.var_time = tk.StringVar(value=str(g.get("default_time_limit", "")))
        ttk.Entry(ctrl, textvariable=self.var_time, width=10).grid(row=4, column=1)

        ttk.Button(ctrl, text="随机生成实例", command=self.on_generate).grid(
            row=5, column=0, columnspan=2, pady=4, sticky=tk.EW
        )
        ttk.Button(ctrl, text="加载 CSV…", command=self.on_load).grid(
            row=6, column=0, columnspan=2, pady=2, sticky=tk.EW
        )
        ttk.Separator(ctrl, orient=tk.HORIZONTAL).grid(
            row=7, column=0, columnspan=2, sticky=tk.EW, pady=6
        )
        ttk.Button(ctrl, text="运行到结束(后台)", command=self.on_run_full).grid(
            row=8, column=0, columnspan=2, pady=2, sticky=tk.EW
        )
        ttk.Button(ctrl, text="单步开始/下一步", command=self.on_step_start).grid(
            row=9, column=0, columnspan=2, pady=2, sticky=tk.EW
        )
        ttk.Checkbutton(ctrl, text="自动连续步进", variable=self.auto_play).grid(
            row=10, column=0, columnspan=2, sticky=tk.W
        )
        ttk.Button(ctrl, text="暂停/继续步进", command=self.on_toggle_pause).grid(
            row=11, column=0, columnspan=2, pady=2, sticky=tk.EW
        )
        ttk.Separator(ctrl, orient=tk.HORIZONTAL).grid(
            row=12, column=0, columnspan=2, sticky=tk.EW, pady=6
        )
        ttk.Button(ctrl, text="保存地图 PNG", command=self.on_save_png).grid(
            row=13, column=0, columnspan=2, sticky=tk.EW
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

    def _parse_optional_int(self, s: str) -> Optional[int]:
        s = s.strip()
        return None if not s else int(s)

    def _parse_optional_float(self, s: str) -> Optional[float]:
        s = s.strip()
        return None if not s else float(s)

    def _draw_static(self, partial_path: Optional[List[int]] = None) -> None:
        if self.tsp is None:
            return
        self.ax.clear()
        pp = partial_path if partial_path and len(partial_path) >= 2 else None
        draw_tsp(
            self.ax,
            self.tsp,
            city_size=35,
            annotate_fontsize=7,
            edge_lw=0.7,
            partial_path=pp,
        )
        self.ax.set_title(f"TSP n={self.tsp.n}")
        self.canvas.draw()

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
        self.gen = None
        self._log(f"已生成 n={n}, k={k}, seed={seed}")
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
        self.gen = None
        self.var_n.set(str(self.tsp.n))
        self._log(f"已加载 n={self.tsp.n}")
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
        max_e = self._parse_optional_int(self.var_max_exp.get())
        max_t = self._parse_optional_float(self.var_time.get())
        tsp = self.tsp

        def work() -> None:
            solver = AStarTSPSolver(tsp, heuristic=hname, start=0)
            res: Optional[SearchResult] = None
            for ev in solver.search_stepwise(
                max_expansions=max_e, time_limit_sec=max_t
            ):
                et = ev.get("event")
                if et == EVENT_POP:
                    path = ev.get("path")
                    if isinstance(path, list) and len(path) >= 2:
                        snap = list(path)
                        self.root.after(0, lambda p=snap: self._draw_static(p))
                elif et == EVENT_GOAL:
                    tour = ev.get("tour", [])
                    if isinstance(tour, list) and len(tour) >= 2:
                        tr = list(tour)
                        self.root.after(0, lambda t=tr: self._draw_static(t))
                elif et == EVENT_DONE:
                    r = ev.get("result")
                    if isinstance(r, SearchResult):
                        res = r

            def finish() -> None:
                if res is None:
                    self._draw_static()
                    self._log("完成: (无结果)")
                    return
                if res.success and res.tour:
                    self._draw_static(list(res.tour))
                else:
                    self._draw_static()
                self._log(
                    f"完成: cost={res.cost:.6f}, expansions={res.expansions}, "
                    f"time={res.elapsed_sec:.4f}s"
                )
                self._log_run_summary(res)

            self.root.after(0, finish)

        threading.Thread(target=work, daemon=True).start()
        self._log("后台运行 A* …")

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
            max_e = self._parse_optional_int(self.var_max_exp.get())
            max_t = self._parse_optional_float(self.var_time.get())
            solver = AStarTSPSolver(self.tsp, heuristic=self.var_h.get(), start=0)
            self.gen = solver.search_stepwise(
                max_expansions=max_e, time_limit_sec=max_t
            )
            self.paused = False
            self._cum_search_s = 0.0
            self._draw_static()
            self._log("--- 新步进搜索（耗时仅计搜索代码，不含挂机等待）---")
        self._pump_one_step()

    def on_toggle_pause(self) -> None:
        self.paused = not self.paused
        self._log("暂停" if self.paused else "继续")
        if not self.paused and self.auto_play.get():
            self._schedule_auto()

    def _schedule_auto(self) -> None:
        if self.step_job:
            self.root.after_cancel(self.step_job)
            self.step_job = None
        if self.gen is None or self.paused or not self.auto_play.get():
            return
        self.step_job = self.root.after(30, self._auto_tick)

    def _auto_tick(self) -> None:
        self.step_job = None
        if self.gen is None or self.paused:
            return
        self._pump_one_step()
        if self.auto_play.get() and not self.paused and self.gen is not None:
            self.step_job = self.root.after(30, self._auto_tick)

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
            return
        dt_step = time.perf_counter() - t0
        self._cum_search_s += dt_step

        et = ev.get("event")
        if et == EVENT_POP:
            path = ev.get("path")
            if isinstance(path, list) and len(path) >= 2:
                self._draw_static(path)
            else:
                self._draw_static()
            self._log(
                f"POP  city={ev.get('current')} phase={ev.get('phase')} "
                f"g={ev.get('g'):.4f} h={ev.get('h'):.4f} f={ev.get('f'):.4f} "
                f"exp={ev.get('expansions')} "
                f"本步搜索耗时_s={dt_step:.6f} 累计搜索耗时_s={self._cum_search_s:.6f}"
            )
        elif et == EVENT_GOAL:
            tour = ev.get("tour", [])
            self._draw_static(list(tour) if isinstance(tour, list) and len(tour) >= 2 else None)
            self._log(
                f"GOAL cost={ev.get('cost'):.6f} tour={tour} "
                f"本步搜索耗时_s={dt_step:.6f} 累计搜索耗时_s={self._cum_search_s:.6f}"
            )
            t1 = time.perf_counter()
            try:
                ev2 = next(self.gen)
            except StopIteration:
                self.gen = None
                self._cum_search_s += time.perf_counter() - t1
                return
            self._cum_search_s += time.perf_counter() - t1
            if ev2.get("event") == EVENT_DONE:
                r = ev2.get("result")
                if isinstance(r, SearchResult):
                    if r.success:
                        self._log(f"DONE 成功 cost={r.cost:.6f} 累计搜索耗时_s={self._cum_search_s:.6f}")
                    else:
                        self._log(f"DONE 失败 {r.message} 累计搜索耗时_s={self._cum_search_s:.6f}")
                    self._log_run_summary(r)
            self.gen = None
            return
        elif et == EVENT_DONE:
            r = ev.get("result")
            if isinstance(r, SearchResult):
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
            return

        if self.auto_play.get() and not self.paused:
            self._schedule_auto()


def launch_gui(gui_cfg: Optional[Dict[str, Any]] = None) -> None:
    root = tk.Tk()
    AStarGUI(root, gui_cfg)
    root.minsize(900, 640)
    root.mainloop()
