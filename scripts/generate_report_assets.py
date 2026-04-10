#!/usr/bin/env python3
"""
生成实验报告所需数据与配图（表1–3 CSV、样例地图、启发对比图、同一实例三启发图）。

报告批量表数据由 ``config.report.json`` 驱动；其中 ``force_astar_only: true`` 表示仅使用 A*（不含 SA/Monster 等），
进度输出见同文件中的 ``astar_progress_*``。普通 ``config.json`` / demo / GUI 默认不受此限制。

用法：
  python scripts/generate_report_assets.py              # 完整（10/20/50/100，repeats 见 config.report.json）
  python scripts/generate_report_assets.py --fast       # 快速冒烟：小规模、少重复、不保存样例地图

流程图 / GUI 思维导图 PNG 需用 draw.io 桌面版从 .drawio 导出，见 docs/report/README.md。
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from main import ROOT as PROJECT_ROOT, _load_config, cmd_tables
from src.map_generator import ensure_connected_fallback, generate_random_map
from src.solve_policy import solve_tsp_auto
from src.visualization import draw_tsp


def _tables_args(save_sample_maps: bool) -> argparse.Namespace:
    return argparse.Namespace(
        sizes=None,
        k=None,
        repeats=None,
        seed=None,
        max_expansions=None,
        time_limit=None,
        save_sample_maps=save_sample_maps,
    )


def run_tables(cfg_path: Path, *, save_sample_maps: bool) -> None:
    cfg = _load_config(cfg_path)
    # 确保输出目录存在
    (PROJECT_ROOT / str(cfg["results_dir"])).mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / str(cfg["instances_dir"])).mkdir(parents=True, exist_ok=True)
    cmd_tables(cfg, _tables_args(save_sample_maps))


def _read_table_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _float_cell(s: str) -> Optional[float]:
    s = (s or "").strip()
    if not s or s == "-":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _int_cell(s: str) -> Optional[int]:
    s = (s or "").strip()
    if not s or s == "-":
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def plot_heuristic_compare(
    tables_dir: Path,
    out_png: Path,
    *,
    aliases: bool = True,
) -> None:
    """读取 table1/2/3 CSV，绘制平均路径长度与总迭代次数随规模变化。"""
    specs: List[Tuple[str, str, str]] = [
        ("table1_euclidean.csv", "euclidean", "C0"),
        ("table2_manhattan.csv", "manhattan", "C1"),
        ("table3_custom.csv", "custom", "C2"),
    ]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    for fname, label, color in specs:
        rows = _read_table_rows(tables_dir / fname)
        xs_c, ys_c = [], []
        xs_e, ys_e = [], []
        for row in rows:
            n = _int_cell(row.get("城市规模", ""))
            if n is None:
                continue
            c = _float_cell(row.get("平均路径长度", ""))
            e = _float_cell(row.get("总迭代次数", ""))
            if c is not None:
                xs_c.append(n)
                ys_c.append(c)
            if e is not None:
                xs_e.append(n)
                ys_e.append(e)

        ax1.plot(xs_c, ys_c, "o-", label=label, color=color)
        ax2.plot(xs_e, ys_e, "o-", label=label, color=color)

    ax1.set_xlabel("n (cities)")
    ax1.set_ylabel("mean tour cost")
    ax1.set_title("Mean tour cost vs n (3 heuristics)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("n (cities)")
    ax2.set_ylabel("sum expansions (table aggregate)")
    ax2.set_title("Total expansions vs n (log scale)")
    ax2.set_yscale("log")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    if aliases:
        # 可选：写一份带「平均运行时间」别名的 CSV 副本，便于粘贴 Word 表头
        for src_name in ("table1_euclidean.csv", "table2_manhattan.csv", "table3_custom.csv"):
            src = tables_dir / src_name
            if not src.is_file():
                continue
            dst = tables_dir / src_name.replace(".csv", "_for_word.csv")
            rows = _read_table_rows(src)
            if not rows:
                continue
            fieldnames = list(rows[0].keys())
            if "平均运行时间_s" in fieldnames and "平均运行时间" not in fieldnames:
                fieldnames = fieldnames + ["平均运行时间"]
                for r in rows:
                    r["平均运行时间"] = r.get("平均运行时间_s", "")
            with dst.open("w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)


def same_instance_three_heuristics(
    out_png: Path,
    *,
    n: int = 12,
    k: int = 4,
    seed: int = 4242,
    start: int = 0,
    max_expansions: int = 2_000_000,
    time_limit_sec: float = 120.0,
) -> None:
    g = generate_random_map(n, k, seed=seed)
    g = ensure_connected_fallback(g, np.random.default_rng(seed))
    cfg: Dict[str, Any] = {
        "auto_solver": True,
        "force_exact_astar": True,
        "astar_max_n": 20,
        "verify_exact_max_n": 14,
        "weighted_max_n": None,
        "weighted_astar_epsilon": None,
        "force_stronger_solver": False,
        "force_monster_solver": False,
        "force_bruteforce": False,
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    names = ["euclidean", "manhattan", "custom"]
    for ax, hname in zip(axes, names):
        res = solve_tsp_auto(
            g,
            hname,
            start,
            cfg,
            max_expansions=max_expansions,
            time_limit_sec=time_limit_sec,
        )
        tour = list(res.tour) if res.success and res.tour else None
        draw_tsp(ax, g, tour=tour, city_size=30, annotate_fontsize=6, edge_lw=0.5)
        subtitle = f"{hname}\n"
        if res.success:
            subtitle += f"cost={res.cost:.4f} exp={res.expansions}"
        else:
            msg = (res.message or "")[:48]
            subtitle += f"fail: {msg}"
        ax.set_title(subtitle, fontsize=9)
    fig.suptitle(f"Same instance: n={n} k={k} seed={seed} start={start}", fontsize=11)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def try_drawio_export(drawio_src: Path, png_out: Path) -> bool:
    """若系统存在 drawio CLI，则尝试导出 PNG。"""
    exe = shutil.which("drawio") or shutil.which("draw.io")
    if not exe:
        return False
    png_out.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [exe, "--export", "--format", "png", "--output", str(png_out), str(drawio_src)],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return png_out.is_file() and png_out.stat().st_size > 100
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="生成实验报告数据与配图")
    ap.add_argument(
        "--fast",
        action="store_true",
        help="小规模快速冒烟（不写完整实验表）",
    )
    ap.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config.report.json",
        help="报告用配置 JSON（基于 main._DEFAULT 合并）",
    )
    args = ap.parse_args()

    report_cfg_path = args.config
    if not report_cfg_path.is_absolute():
        report_cfg_path = ROOT / report_cfg_path

    figures_dir = ROOT / "data" / "report" / "figures"
    docs_report = ROOT / "docs" / "report"
    tables_dir = ROOT / "data" / "report" / "tables"

    if args.fast:
        # 临时合并配置：覆盖 sizes/repeats/save_sample_maps（报告数据仍仅走 A*）
        base = _load_config(report_cfg_path)
        base["results_dir"] = "data/report/tables"
        base["instances_dir"] = "data/report/instances"
        base["force_astar_only"] = True
        base["sizes"] = [6, 8]
        base["repeats"] = 1
        base["seed"] = 7
        base["save_sample_maps"] = False
        fast_path = ROOT / "data" / "report" / "_fast_config.json"
        fast_path.parent.mkdir(parents=True, exist_ok=True)
        fast_path.write_text(json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")
        run_tables(fast_path, save_sample_maps=False)
        print("Fast mode: wrote CSV under data/report/tables/", flush=True)
    else:
        run_tables(report_cfg_path, save_sample_maps=True)
        print("Wrote tables + sample maps (see config.report.json paths).", flush=True)

    # 配图：CSV 对比图（依赖刚生成的 CSV）
    compare_png = figures_dir / "heuristic_compare.png"
    try:
        plot_heuristic_compare(tables_dir, compare_png, aliases=True)
        print("Wrote", compare_png, flush=True)
    except Exception as e:
        print("Warning: heuristic_compare plot failed:", e, flush=True)

    same_png = figures_dir / "same_instance_three_heuristics.png"
    try:
        same_instance_three_heuristics(same_png)
        print("Wrote", same_png, flush=True)
    except Exception as e:
        print("Warning: same_instance_three_heuristics failed:", e, flush=True)

    # 可选：draw.io 导出
    for src, dst in (
        (ROOT / "docs" / "astar_tsp_flowchart.drawio", docs_report / "astar_tsp_flowchart.png"),
        (ROOT / "docs" / "gui_mindmap.drawio", docs_report / "gui_mindmap.png"),
    ):
        if src.is_file() and try_drawio_export(src, dst):
            print("Exported via draw.io CLI:", dst, flush=True)
        else:
            print(
                f"Skip auto-export {src.name} -> PNG (install draw.io CLI or export manually, see docs/report/README.md)",
                flush=True,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
