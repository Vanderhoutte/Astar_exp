#!/usr/bin/env python3
"""入口：tables / demo / gui。参数来自 config.json + 可选 CLI 覆盖。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.astar_tsp_solver import (
    EVENT_DONE,
    EVENT_GOAL,
    EVENT_POP,
    EVENT_PUSH,
    AStarTSPSolver,
    SearchResult,
)
from src.feasibility_estimate import estimate_feasible_closed_tour, format_feasibility_log_lines
from src.solve_policy import (
    recommend_solver_mode,
    solve_tsp_auto,
    state_space_upper_bound,
)
from src.solution_verifier import exact_optimal_tour_small_n, validate_closed_tour
from src.map_generator import (
    ensure_connected_fallback,
    generate_random_map,
    save_instance,
)
from src.visualization import launch_gui, save_map_png

_DEFAULT: Dict[str, Any] = {
    "default_cmd": "demo",
    "results_dir": "data/results",
    "instances_dir": "data/instances",
    "intermediate_dir": "data/intermediate",
    "sizes": [10, 20, 50, 100],
    "k": 4,
    "repeats": 10,
    "seed": 42,
    "max_expansions": None,
    "time_limit": None,
    "save_sample_maps": False,
    "auto_time_limit_n_ge_50": True,
    "demo_n": 10,
    "demo_k": 4,
    "demo_seed": 0,
    "demo_heuristic": "euclidean",
    "demo_max_expansions": 500000,
    "demo_time_limit": None,
    "demo_start": "random",
    "demo_save_instance": False,
    "save_png": True,
    "save_events": True,
    "gui_default_n": 10,
    "gui_default_k": 4,
    "gui_default_heuristic": "euclidean",
    "gui_default_start": "random",
    "gui_default_max_expansions": "500000",
    "gui_default_time_limit": "",
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
    "tables_start": "random",
    "verify_exact_max_n": 14,
    "verify_exact_time_sec": None,
}

_EVENT_FIELDS = [
    "event",
    "expansions",
    "g",
    "h",
    "f",
    "current",
    "phase",
    "push_to",
    "push_from",
    "push_phase",
    "cost",
    "tour",
    "success",
    "message",
    "elapsed_sec",
    "method",
    "optimal",
]


def _load_config(path: Path) -> Dict[str, Any]:
    cfg = dict(_DEFAULT)
    if path.is_file():
        with path.open(encoding="utf-8") as f:
            user = json.load(f)
        if isinstance(user, dict):
            cfg.update(user)
    return cfg


def _resolve_start_vertex(raw: Any, n: int, rng: np.random.Generator) -> int:
    """解析起点配置：支持 random/空/整数，最终归一化到 [0, n)。"""
    if n <= 0:
        raise ValueError("n must be positive")
    if raw is None:
        return int(rng.integers(0, n))
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in {"", "random", "rand"}:
            return int(rng.integers(0, n))
        try:
            raw_i = int(s)
        except ValueError as e:
            raise ValueError(f"invalid start value: {raw!r}") from e
    elif isinstance(raw, (int, np.integer)):
        raw_i = int(raw)
    else:
        raise ValueError(f"invalid start type: {type(raw).__name__}")
    return raw_i % n


def _p(root: Path, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else root / p


def _cell_num(v: Any) -> str:
    if v == "" or v is None:
        return "-"
    if isinstance(v, float) and np.isnan(v):
        return "-"
    if isinstance(v, (int, float, np.floating)):
        return f"{float(v):.6f}"
    return str(v)


def _lines_benchmark_table(title: str, rows: List[Dict[str, Any]]) -> List[str]:
    """实验表字段：城市规模、路径统计、迭代与时间、成功/失败、求解方法、是否最优。"""
    w = 118
    lines = [title, "=" * w]
    head = (
        f"{'城市规模':>6} | {'最优路径长度':>14} | {'最差路径长度':>14} | {'平均路径长度':>14} | "
        f"{'总迭代次数':>10} | {'平均运行时间_s':>14} | {'成功':>4} | {'失败':>4} | "
        f"{'求解方法':>12} | {'最优':>4}"
    )
    lines.append(head)
    lines.append("-" * w)
    for r in rows:
        line = (
            f"{int(r['城市规模']):>6} | {_cell_num(r.get('最优路径长度')):>14} | "
            f"{_cell_num(r.get('最差路径长度')):>14} | {_cell_num(r.get('平均路径长度')):>14} | "
            f"{str(r.get('总迭代次数', '')):>10} | {_cell_num(r.get('平均运行时间_s')):>14} | "
            f"{int(r.get('成功次数', 0)):>4} | {int(r.get('失败次数', 0)):>4} | "
            f"{str(r.get('求解方法', '-')):>12} | {str(r.get('是否最优', '-')):>4}"
        )
        lines.append(line)
    lines.append("")
    return lines


def _lines_demo_result(n: int, k: int, heuristic: str, final: Optional[SearchResult]) -> List[str]:
    lines = [
        "========== 单次演示结果（与实验表字段对应；单次最优=最差=平均）==========",
        f"城市规模:\t{n}",
        f"k近邻:\t{k}",
        f"启发:\t{heuristic}",
        "",
    ]
    if final is None:
        lines.append("最优路径长度:\t(无结果)")
        lines.append("最差路径长度:\t(无结果)")
        lines.append("平均路径长度:\t(无结果)")
        lines.append("总迭代次数:\t-")
        lines.append("平均运行时间_s:\t-")
        return lines
    if final.success:
        c = float(final.cost)
        lines.append(f"最优路径长度:\t{c:.6f}")
        lines.append(f"最差路径长度:\t{c:.6f}")
        lines.append(f"平均路径长度:\t{c:.6f}")
    else:
        lines.append("最优路径长度:\t(未成功)")
        lines.append("最差路径长度:\t(未成功)")
        lines.append("平均路径长度:\t(未成功)")
    lines.append(f"总迭代次数:\t{final.expansions}")
    lines.append(f"总运行时间_s:\t{final.elapsed_sec:.6f}")
    lines.append(f"平均运行时间_s:\t{final.elapsed_sec:.6f}\t(单次搜索总耗时)")
    if final.expansions > 0:
        lines.append(f"平均每步时间_s:\t{final.elapsed_sec / final.expansions:.6f}")
    lines.append(f"success:\t{final.success}")
    lines.append(f"message:\t{final.message}")
    lines.append(f"method:\t{final.method}")
    lines.append(f"optimal:\t{final.optimal}")
    lines.append(f"tour:\t{final.tour}")
    lines.append("")
    return lines


def _benchmark_scale(
    n: int,
    k_neighbors: int,
    heuristic_name: str,
    repeats: int,
    base_seed: int = 0,
    max_expansions: Optional[int] = None,
    time_limit_sec: Optional[float] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    policy_cfg = dict(cfg or {})
    rng = np.random.default_rng(base_seed)
    costs: List[float] = []
    expansions_list: List[int] = []
    times: List[float] = []
    failures = 0
    methods: List[str] = []

    for _ in range(repeats):
        seed = int(rng.integers(0, 2**31 - 1))
        g = generate_random_map(n, k_neighbors, seed=seed)
        g = ensure_connected_fallback(g, np.random.default_rng(seed))
        start_i = _resolve_start_vertex(
            policy_cfg.get("tables_start", "random"),
            n,
            np.random.default_rng(seed ^ 0xA5A5A5A5),
        )
        res: SearchResult = solve_tsp_auto(
            g,
            heuristic_name,
            start_i,
            policy_cfg,
            max_expansions=max_expansions,
            time_limit_sec=time_limit_sec,
        )
        methods.append(res.method)
        if not res.success:
            failures += 1
            continue
        costs.append(res.cost)
        expansions_list.append(res.expansions)
        times.append(res.elapsed_sec)

    mc = Counter(methods).most_common(1)
    method_summary = mc[0][0] if mc else "-"
    all_exact = bool(methods) and all(m == "astar_exact" for m in methods)

    row: Dict[str, Any] = {
        "城市规模": n,
        "k近邻": k_neighbors,
        "启发": heuristic_name,
        "重复次数": repeats,
        "成功次数": repeats - failures,
        "失败次数": failures,
        "状态空间上界": state_space_upper_bound(n),
        "求解方法": method_summary,
        "是否最优": "是" if all_exact else "否",
    }
    if costs:
        row["最优路径长度"] = float(np.min(costs))
        row["最差路径长度"] = float(np.max(costs))
        row["平均路径长度"] = float(np.mean(costs))
        row["总迭代次数"] = int(np.sum(expansions_list))
        row["平均运行时间_s"] = float(np.mean(times))
    else:
        row["最优路径长度"] = ""
        row["最差路径长度"] = ""
        row["平均路径长度"] = ""
        row["总迭代次数"] = 0
        row["平均运行时间_s"] = ""
    return row


def _write_table_csv(rows: List[Dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return path
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return path


def _fmt(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


def _event_row(ev: Dict[str, Any]) -> Dict[str, str]:
    row = {k: "" for k in _EVENT_FIELDS}
    et = ev.get("event", "")
    row["event"] = _fmt(et)
    if et == EVENT_POP:
        row["current"] = _fmt(ev.get("current"))
        row["phase"] = _fmt(ev.get("phase"))
        row["g"] = _fmt(ev.get("g"))
        row["h"] = _fmt(ev.get("h"))
        row["f"] = _fmt(ev.get("f"))
        row["expansions"] = _fmt(ev.get("expansions"))
    elif et == EVENT_PUSH:
        row["push_to"] = _fmt(ev.get("to"))
        row["push_from"] = _fmt(ev.get("from"))
        row["push_phase"] = _fmt(ev.get("phase"))
        row["g"] = _fmt(ev.get("g"))
        row["h"] = _fmt(ev.get("h"))
        row["f"] = _fmt(ev.get("f"))
    elif et == EVENT_GOAL:
        row["cost"] = _fmt(ev.get("cost"))
        row["tour"] = _fmt(ev.get("tour"))
        row["expansions"] = _fmt(ev.get("expansions"))
    elif et == EVENT_DONE:
        r = ev.get("result")
        if isinstance(r, SearchResult):
            row["success"] = _fmt(r.success)
            row["message"] = _fmt(r.message)
            row["cost"] = _fmt(r.cost)
            row["tour"] = _fmt(r.tour)
            row["elapsed_sec"] = _fmt(r.elapsed_sec)
            row["expansions"] = _fmt(r.expansions)
            row["method"] = _fmt(r.method)
            row["optimal"] = _fmt(r.optimal)
    return row


def _post_verify_result(
    tsp: Any, final: Optional[SearchResult], start: int, cfg: Dict[str, Any]
) -> Optional[SearchResult]:
    if final is None:
        return None
    if final.success:
        chk = validate_closed_tour(tsp, final.tour, start=start, cost=final.cost)
        if not chk.valid:
            return SearchResult(
                False,
                list(final.tour),
                float(chk.computed_cost),
                final.expansions,
                final.elapsed_sec,
                f"solution invalid: {chk.message}",
                method=final.method,
                optimal=False,
            )
        final.tour = chk.normalized_tour
        final.cost = chk.computed_cost
    vmax = int(cfg.get("verify_exact_max_n", 14))
    vtime_raw = cfg.get("verify_exact_time_sec")
    vtime = float(vtime_raw) if vtime_raw is not None else None
    if tsp.n <= vmax:
        cert = exact_optimal_tour_small_n(tsp, start=start, time_limit_sec=vtime)
        if cert.certified and cert.optimal_cost is not None and final.success:
            if abs(final.cost - cert.optimal_cost) <= 1e-8:
                final.optimal = True
            else:
                final.optimal = False
                final.message = (
                    (final.message + "; ") if final.message else ""
                ) + f"verified suboptimal: gap={final.cost - cert.optimal_cost:.6g}"
        elif not cert.certified and final.success and cert.message:
            final.message = ((final.message + "; ") if final.message else "") + cert.message
    elif final.success:
        note = f"optimality uncertified: n={tsp.n} > verify_exact_max_n={vmax}"
        final.message = ((final.message + "; ") if final.message else "") + note
    return final


def cmd_tables(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    sizes = args.sizes if args.sizes is not None else list(cfg["sizes"])
    k = args.k if args.k is not None else cfg["k"]
    repeats = args.repeats if args.repeats is not None else cfg["repeats"]
    seed = args.seed if args.seed is not None else cfg["seed"]
    max_exp = args.max_expansions if args.max_expansions is not None else cfg["max_expansions"]
    time_limit = args.time_limit if args.time_limit is not None else cfg["time_limit"]
    save_maps = args.save_sample_maps if args.save_sample_maps else cfg["save_sample_maps"]

    if (
        cfg.get("auto_time_limit_n_ge_50", True)
        and max_exp is None
        and time_limit is None
        and max(sizes, default=0) >= 50
    ):
        time_limit = 120.0
        print(
            "提示: n≥50 时默认每 run 时限 120s（可在 config 关 auto_time_limit_n_ge_50 或设 max_expansions/time_limit）",
            file=sys.stderr,
        )

    results_dir = _p(ROOT, str(cfg["results_dir"]))
    instances_dir = _p(ROOT, str(cfg["instances_dir"]))
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_chunks: List[str] = []
    summary_chunks.append(
        f"A* TSP 批量结果输出  k={k}  repeats={repeats}  seed={seed}  "
        f"启发分组=表1欧氏/表2曼哈顿/表3自定义\n"
    )

    for name, hkey in [
        ("table1_euclidean", "euclidean"),
        ("table2_manhattan", "manhattan"),
        ("table3_custom", "custom"),
    ]:
        rows = []
        for n in sizes:
            rows.append(
                _benchmark_scale(
                    n,
                    k,
                    hkey,
                    repeats,
                    base_seed=seed + n * 1000,
                    max_expansions=max_exp,
                    time_limit_sec=time_limit,
                    cfg=cfg,
                )
            )
        out = _write_table_csv(rows, results_dir / f"{name}.csv")
        print("Wrote", out, flush=True)
        block = _lines_benchmark_table(f"【{name}】启发={hkey}", rows)
        summary_chunks.append("\n".join(block))
        for line in block:
            print(line, flush=True)

    run_summary_path = results_dir / "run_summary.txt"
    run_summary_path.write_text("\n".join(summary_chunks), encoding="utf-8")
    print("Wrote", run_summary_path, flush=True)

    if save_maps:
        for n in sizes:
            s = seed + n
            g = generate_random_map(n, k, seed=s)
            g = ensure_connected_fallback(g, np.random.default_rng(s))
            base = f"sample_n{n}_k{k}_s{s}"
            save_instance(g, instances_dir, base)
            save_map_png(g, instances_dir / f"{base}.png", title=f"Sample n={n} k={k}")


def cmd_demo(cfg: Dict[str, Any]) -> None:
    mid = _p(ROOT, str(cfg["intermediate_dir"]))
    mid.mkdir(parents=True, exist_ok=True)

    n, k = int(cfg["demo_n"]), int(cfg["demo_k"])
    dseed = int(cfg["demo_seed"])
    g = generate_random_map(n, k, seed=dseed)
    g = ensure_connected_fallback(g, np.random.default_rng(dseed))

    if cfg.get("demo_save_instance"):
        save_instance(g, mid, "demo")

    if cfg.get("save_png", True):
        save_map_png(g, mid / "map.png", title=f"n={n} k={k} seed={dseed}")

    hname = str(cfg["demo_heuristic"])
    start_i = _resolve_start_vertex(
        cfg.get("demo_start", "random"),
        n,
        np.random.default_rng((dseed ^ 0x9E3779B9) & 0xFFFFFFFF),
    )
    max_e = cfg["demo_max_expansions"]
    max_t = cfg["demo_time_limit"]
    final: Optional[SearchResult] = None
    mode = recommend_solver_mode(n, cfg)
    feas = estimate_feasible_closed_tour(
        g,
        start=start_i,
        seed=int(dseed) & 0x7FFFFFFF,
    )
    feas_lines = format_feasibility_log_lines(feas)

    if cfg.get("save_events", True) and mode == "heuristic":
        tsv_path = mid / "events.tsv"
        final = solve_tsp_auto(
            g,
            hname,
            start_i,
            cfg,
            max_expansions=max_e,
            time_limit_sec=max_t,
        )
        with tsv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=_EVENT_FIELDS, delimiter="\t", extrasaction="ignore")
            w.writeheader()
            w.writerow(_event_row({"event": EVENT_DONE, "result": final}))
    elif cfg.get("save_events", True):
        eps = (
            float(cfg["weighted_astar_epsilon"])
            if mode == "weighted" and cfg.get("weighted_astar_epsilon") is not None
            else 1.0
        )
        solver = AStarTSPSolver(g, heuristic=hname, start=start_i)
        tsv_path = mid / "events.tsv"
        with tsv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=_EVENT_FIELDS, delimiter="\t", extrasaction="ignore")
            w.writeheader()
            for ev in solver.search_stepwise(
                max_expansions=max_e,
                time_limit_sec=max_t,
                epsilon=eps,
            ):
                w.writerow(_event_row(ev))
                if ev.get("event") == EVENT_DONE:
                    r = ev.get("result")
                    if isinstance(r, SearchResult):
                        final = r
    else:
        final = solve_tsp_auto(
            g,
            hname,
            start_i,
            cfg,
            max_expansions=max_e,
            time_limit_sec=max_t,
        )

    final = _post_verify_result(g, final, start_i, cfg)

    demo_lines = _lines_demo_result(n, k, hname, final)
    ins = 5
    demo_lines.insert(ins, f"起点:\t{start_i}")
    ins += 1
    for j, fl in enumerate(feas_lines):
        demo_lines.insert(ins + j, fl)
    demo_lines.insert(
        ins + len(feas_lines),
        f"规模策略:\t{mode}\t(状态空间上界≈{state_space_upper_bound(n)})",
    )
    demo_text = "\n".join(demo_lines)
    sum_path = mid / "summary.txt"
    sum_path.write_text(demo_text, encoding="utf-8")
    print(demo_text, flush=True)
    print("Wrote", sum_path, flush=True)

    if cfg.get("save_png", True) and final and final.success and final.tour:
        save_map_png(g, mid / "solution.png", tour=final.tour, title="solution")


def cmd_gui(cfg: Dict[str, Any]) -> None:
    launch_gui(
        {
            "default_n": cfg.get("gui_default_n", 10),
            "default_k": cfg.get("gui_default_k", 4),
            "default_heuristic": str(cfg.get("gui_default_heuristic", "euclidean")),
            "default_start": str(cfg.get("gui_default_start", "random")),
            "default_max_expansions": str(cfg.get("gui_default_max_expansions", "500000")),
            "default_time_limit": str(cfg.get("gui_default_time_limit", "")),
            "auto_solver": cfg.get("auto_solver", True),
            "force_exact_astar": cfg.get("force_exact_astar", False),
            "force_stronger_solver": cfg.get("force_stronger_solver", False),
            "force_monster_solver": cfg.get("force_monster_solver", False),
            "force_bruteforce": cfg.get("force_bruteforce", False),
            "monster_max_n": cfg.get("monster_max_n", 24),
            "monster_time_limit_sec": cfg.get("monster_time_limit_sec", 20.0),
            "bruteforce_max_n": cfg.get("bruteforce_max_n", 11),
            "bruteforce_time_limit_sec": cfg.get("bruteforce_time_limit_sec"),
            "stronger_exact_max_n": cfg.get("stronger_exact_max_n", 18),
            "stronger_exact_time_limit_sec": cfg.get("stronger_exact_time_limit_sec"),
            "astar_max_n": cfg.get("astar_max_n", 20),
            "weighted_max_n": cfg.get("weighted_max_n"),
            "weighted_astar_epsilon": cfg.get("weighted_astar_epsilon"),
            "fallback_sa_iterations": cfg.get("fallback_sa_iterations", 8000),
            "fallback_sa_T0": cfg.get("fallback_sa_T0", 1.0),
            "fallback_sa_T_min": cfg.get("fallback_sa_T_min", 1e-4),
            "fallback_sa_cool": cfg.get("fallback_sa_cool", 0.995),
            "fallback_sa_seed": cfg.get("fallback_sa_seed"),
            "fallback_random_nn_tries": cfg.get("fallback_random_nn_tries", 500),
            "fallback_rcl_size": cfg.get("fallback_rcl_size", 4),
            "fallback_backtrack": cfg.get("fallback_backtrack", True),
            "fallback_backtrack_max_steps": cfg.get("fallback_backtrack_max_steps", 3_000_000),
            "fallback_backtrack_restarts": cfg.get("fallback_backtrack_restarts", 24),
            "fallback_backtrack_time_sec": cfg.get("fallback_backtrack_time_sec"),
            "verify_exact_max_n": cfg.get("verify_exact_max_n", 14),
            "verify_exact_time_sec": cfg.get("verify_exact_time_sec"),
        }
    )


def main() -> None:
    p = argparse.ArgumentParser(description="A* TSP：tables / demo / gui（见 config.json）")
    p.add_argument("--config", default="config.json", help="JSON 配置路径")
    sub = p.add_subparsers(dest="cmd", required=False)

    pg = sub.add_parser("gui", help="Tk 图形界面：过程展示（实验要求）")
    pg.set_defaults(func="gui")

    pt = sub.add_parser("tables", help="生成表 1–3 CSV")
    pt.add_argument("--sizes", type=int, nargs="+", default=None)
    pt.add_argument("--k", type=int, default=None)
    pt.add_argument("--repeats", type=int, default=None)
    pt.add_argument("--seed", type=int, default=None)
    pt.add_argument("--max-expansions", type=int, default=None)
    pt.add_argument("--time-limit", type=float, default=None)
    pt.add_argument("--save-sample-maps", action="store_true", default=False)
    pt.set_defaults(func="tables")

    pd = sub.add_parser("demo", help="单次搜索，写 data/intermediate/ 下 TSV/TXT/PNG")
    pd.set_defaults(func="demo")

    args = p.parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    cfg = _load_config(cfg_path)

    selected = getattr(args, "func", None) or str(cfg.get("default_cmd", "demo")).strip().lower()
    if selected not in {"tables", "demo", "gui"}:
        p.error(f"invalid default_cmd in config: {selected!r}, choose from gui/tables/demo")

    if selected == "tables":
        cmd_tables(cfg, args)
    elif selected == "demo":
        cmd_demo(cfg)
    else:
        cmd_gui(cfg)


if __name__ == "__main__":
    main()
