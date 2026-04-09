#!/usr/bin/env python3
"""入口：tables / demo / gui。参数来自 config.json + 可选 CLI 覆盖。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
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
from src.map_generator import (
    ensure_connected_fallback,
    generate_random_map,
    save_instance,
)
from src.visualization import launch_gui, save_map_png

_DEFAULT: Dict[str, Any] = {
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
    "demo_start": 0,
    "demo_save_instance": False,
    "save_png": True,
    "save_events": True,
    "gui_default_n": 10,
    "gui_default_k": 4,
    "gui_default_heuristic": "euclidean",
    "gui_default_max_expansions": "500000",
    "gui_default_time_limit": "",
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
]


def _load_config(path: Path) -> Dict[str, Any]:
    cfg = dict(_DEFAULT)
    if path.is_file():
        with path.open(encoding="utf-8") as f:
            user = json.load(f)
        if isinstance(user, dict):
            cfg.update(user)
    return cfg


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
    """实验表字段：城市规模、最优/最差/平均路径长度、总迭代次数、平均运行时间 + 成功/失败次数。"""
    w = 92
    lines = [title, "=" * w]
    head = (
        f"{'城市规模':>6} | {'最优路径长度':>14} | {'最差路径长度':>14} | {'平均路径长度':>14} | "
        f"{'总迭代次数':>10} | {'平均运行时间_s':>14} | {'成功':>4} | {'失败':>4}"
    )
    lines.append(head)
    lines.append("-" * w)
    for r in rows:
        line = (
            f"{int(r['城市规模']):>6} | {_cell_num(r.get('最优路径长度')):>14} | "
            f"{_cell_num(r.get('最差路径长度')):>14} | {_cell_num(r.get('平均路径长度')):>14} | "
            f"{str(r.get('总迭代次数', '')):>10} | {_cell_num(r.get('平均运行时间_s')):>14} | "
            f"{int(r.get('成功次数', 0)):>4} | {int(r.get('失败次数', 0)):>4}"
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
) -> Dict[str, Any]:
    rng = np.random.default_rng(base_seed)
    costs: List[float] = []
    expansions_list: List[int] = []
    times: List[float] = []
    failures = 0

    for _ in range(repeats):
        seed = int(rng.integers(0, 2**31 - 1))
        g = generate_random_map(n, k_neighbors, seed=seed)
        g = ensure_connected_fallback(g, np.random.default_rng(seed))
        solver = AStarTSPSolver(g, heuristic=heuristic_name, start=0)
        res: SearchResult = solver.search(
            max_expansions=max_expansions,
            time_limit_sec=time_limit_sec,
        )
        if not res.success:
            failures += 1
            continue
        costs.append(res.cost)
        expansions_list.append(res.expansions)
        times.append(res.elapsed_sec)

    row: Dict[str, Any] = {
        "城市规模": n,
        "k近邻": k_neighbors,
        "启发": heuristic_name,
        "重复次数": repeats,
        "成功次数": repeats - failures,
        "失败次数": failures,
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
    return row


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

    solver = AStarTSPSolver(
        g,
        heuristic=str(cfg["demo_heuristic"]),
        start=int(cfg.get("demo_start", 0)),
    )
    max_e = cfg["demo_max_expansions"]
    max_t = cfg["demo_time_limit"]
    final: Optional[SearchResult] = None

    if cfg.get("save_events", True):
        tsv_path = mid / "events.tsv"
        with tsv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=_EVENT_FIELDS, delimiter="\t", extrasaction="ignore")
            w.writeheader()
            for ev in solver.search_stepwise(
                max_expansions=max_e,
                time_limit_sec=max_t,
            ):
                w.writerow(_event_row(ev))
                if ev.get("event") == EVENT_DONE:
                    r = ev.get("result")
                    if isinstance(r, SearchResult):
                        final = r
    else:
        final = solver.search(max_expansions=max_e, time_limit_sec=max_t)

    hname = str(cfg["demo_heuristic"])
    demo_lines = _lines_demo_result(n, k, hname, final)
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
            "default_max_expansions": str(cfg.get("gui_default_max_expansions", "500000")),
            "default_time_limit": str(cfg.get("gui_default_time_limit", "")),
        }
    )


def main() -> None:
    p = argparse.ArgumentParser(description="A* TSP：tables / demo / gui（见 config.json）")
    p.add_argument("--config", default="config.json", help="JSON 配置路径")
    sub = p.add_subparsers(dest="cmd", required=True)

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

    if args.func == "tables":
        cmd_tables(cfg, args)
    elif args.func == "demo":
        cmd_demo(cfg)
    else:
        cmd_gui(cfg)


if __name__ == "__main__":
    main()
