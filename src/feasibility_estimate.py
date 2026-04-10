"""闭合哈密顿回路（可行 TSP 回路）是否可能存在的快速估算（非严格判定）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal

import numpy as np

from .map_generator import TSPGraph
from .tsp_heuristic_solvers import hamiltonian_cycle_backtrack

Verdict = Literal["no", "probe_yes", "unknown"]


@dataclass
class FeasibilityEstimate:
    """verdict: 否 / 试探找到 / 不确定（必要条件允许但试探未找到）。"""

    verdict: Verdict
    summary_zh: str
    details_zh: List[str] = field(default_factory=list)


def graph_is_connected(tsp: TSPGraph) -> bool:
    n = tsp.n
    if n <= 1:
        return True
    seen = {0}
    stack = [0]
    while stack:
        u = stack.pop()
        for v, _ in tsp.adjacency[u]:
            vi = int(v)
            if vi not in seen:
                seen.add(vi)
                stack.append(vi)
    return len(seen) == n


def vertex_degrees(tsp: TSPGraph) -> List[int]:
    return [len(tsp.adjacency[i]) for i in range(tsp.n)]


def estimate_feasible_closed_tour(
    tsp: TSPGraph,
    start: int = 0,
    *,
    probe_max_steps: int = 80_000,
    probe_restarts: int = 8,
    probe_time_sec: float | None = 0.35,
    seed: int = 0,
) -> FeasibilityEstimate:
    """
    先做**必要条件**（不连通、度不足则必无可行回路），再用**限步回溯+随机邻序**试探。
    试探找到 ⇒ 一定存在可行解；未找到 ⇒ 不能推出无解（NP 完全问题无廉价精确判定）。
    """
    n = tsp.n
    details: List[str] = []
    start = int(start) % max(n, 1)

    if n <= 1:
        return FeasibilityEstimate(
            "unknown",
            "n≤1 时不按闭合回路模型估算。",
            ["顶点过少"],
        )

    if not graph_is_connected(tsp):
        return FeasibilityEstimate(
            "no",
            "图不连通 ⇒ 不存在经过全部顶点一次的闭合回路。",
            ["BFS 从城市 0 无法到达所有顶点"],
        )

    degs = vertex_degrees(tsp)
    min_d = min(degs)
    details.append(f"各点度数: min={min_d}, max={max(degs)}")

    if n == 2:
        ok = any(int(v) == 1 - start for v, _ in tsp.adjacency[start])
        if ok:
            return FeasibilityEstimate(
                "probe_yes",
                "n=2 且两点相邻 ⇒ 存在闭合回路。",
                details + ["两点间有边即可"],
            )
        return FeasibilityEstimate(
            "no",
            "n=2 但两点无边 ⇒ 无可行回路。",
            details + ["缺少连接两城的边"],
        )

    if n >= 3 and min_d < 2:
        return FeasibilityEstimate(
            "no",
            "存在度数 <2 的顶点 ⇒ 不可能存在哈密顿回路（n≥3）。",
            details + ["哈密顿回路要求每点在回路上度为 2，图上至少要有 2 条不同邻边"],
        )

    rng = np.random.default_rng(seed)
    restarts = max(1, int(probe_restarts))
    steps_each = max(10_000, int(probe_max_steps) // restarts)

    for _ in range(restarts):
        child = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        hc = hamiltonian_cycle_backtrack(
            tsp,
            start,
            child,
            max_steps=steps_each,
            time_limit_sec=probe_time_sec,
        )
        if hc is not None:
            return FeasibilityEstimate(
                "probe_yes",
                "快速随机回溯已找到一条合法闭合回路 ⇒ 可行解存在（未判断最优性）。",
                details
                + [
                    f"试探参数: {restarts} 轮 × 约 {steps_each} DFS 步, 时限≈{probe_time_sec}s/轮",
                ],
            )

    return FeasibilityEstimate(
        "unknown",
        "必要条件允许可行解，但快速试探未找到回路 ⇒ 不能据此断定无解，可增大 k、换 seed 或直接运行求解器。",
        details
        + [
            f"试探参数: {restarts} 轮 × 约 {steps_each} DFS 步",
            "图论上判定哈密顿回路为 NP 完全，此处仅为启发式估算",
        ],
    )


def format_feasibility_log_lines(est: FeasibilityEstimate) -> List[str]:
    lines = [f"可行解估算: {est.summary_zh}"]
    for d in est.details_zh:
        lines.append(f"  · {d}")
    return lines
