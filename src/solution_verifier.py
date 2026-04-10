"""TSP 解合法性校验与小规模最优性认证。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .map_generator import TSPGraph


@dataclass
class TourValidation:
    valid: bool
    normalized_tour: List[int]
    computed_cost: float
    message: str = ""


@dataclass
class OptimalityCertificate:
    certified: bool
    optimal_cost: Optional[float]
    optimal_tour: List[int]
    message: str = ""


def normalize_start(start: int, n: int) -> int:
    if n <= 0:
        raise ValueError("tsp.n must be positive")
    return int(start) % n


def _adj_weight_map(tsp: TSPGraph) -> Dict[Tuple[int, int], float]:
    w: Dict[Tuple[int, int], float] = {}
    for u in range(tsp.n):
        for v_raw, ww in tsp.adjacency[u]:
            v = int(v_raw)
            w[(u, v)] = float(ww)
    return w


def validate_closed_tour(
    tsp: TSPGraph,
    tour: List[int],
    *,
    start: int,
    cost: Optional[float] = None,
    atol: float = 1e-8,
) -> TourValidation:
    """严格检查回路是否合法，并返回按起点归一后的回路与重算代价。"""
    n = tsp.n
    s = normalize_start(start, n)
    if len(tour) != n + 1:
        return TourValidation(False, [], float("inf"), f"tour length must be n+1, got {len(tour)}")
    t = [int(x) for x in tour]
    if t[0] != s or t[-1] != s:
        return TourValidation(False, t, float("inf"), f"tour must start/end at start={s}")
    middle = t[:-1]
    if len(set(middle)) != n:
        return TourValidation(False, t, float("inf"), "tour must visit each vertex exactly once")
    if any(v < 0 or v >= n for v in middle):
        return TourValidation(False, t, float("inf"), "tour has vertex out of range")

    wmap = _adj_weight_map(tsp)
    total = 0.0
    for i in range(n):
        a, b = t[i], t[i + 1]
        if (a, b) not in wmap:
            return TourValidation(False, t, float("inf"), f"edge ({a}->{b}) not in graph")
        total += wmap[(a, b)]

    if cost is not None and abs(float(cost) - total) > atol:
        return TourValidation(
            False,
            t,
            total,
            f"reported cost={float(cost):.9f} mismatches computed cost={total:.9f}",
        )
    return TourValidation(True, t, total, "")


def exact_optimal_tour_small_n(
    tsp: TSPGraph,
    *,
    start: int,
    time_limit_sec: Optional[float] = None,
) -> OptimalityCertificate:
    """
    Held-Karp 动态规划（稀疏图）：
    - 证真最优（certified=True）
    - 若超时则 certified=False
    - 若图上无可行闭合回路则 optimal_cost=None 且 certified=True
    """
    n = tsp.n
    s = normalize_start(start, n)
    t0 = time.perf_counter()

    def over_time() -> bool:
        return time_limit_sec is not None and (time.perf_counter() - t0) >= time_limit_sec

    w = _adj_weight_map(tsp)
    start_mask = 1 << s
    dp: Dict[Tuple[int, int], float] = {(start_mask, s): 0.0}
    parent: Dict[Tuple[int, int], Tuple[int, int]] = {}

    for _ in range(1, n):
        if over_time():
            return OptimalityCertificate(False, None, [], "exact verifier time limit exceeded")
        next_dp: Dict[Tuple[int, int], float] = {}
        for (mask, u), gcost in dp.items():
            for v in range(n):
                if v == s:
                    continue
                bit = 1 << v
                if mask & bit:
                    continue
                wuv = w.get((u, v))
                if wuv is None:
                    continue
                nmask = mask | bit
                key = (nmask, v)
                ng = gcost + wuv
                if ng < next_dp.get(key, float("inf")):
                    next_dp[key] = ng
                    parent[key] = (mask, u)
        dp = next_dp
        if not dp:
            break

    full = (1 << n) - 1
    best = float("inf")
    best_last: Optional[int] = None
    for (mask, last), gcost in dp.items():
        if mask != full:
            continue
        back = w.get((last, s))
        if back is None:
            continue
        c = gcost + back
        if c < best:
            best = c
            best_last = last

    if best_last is None:
        return OptimalityCertificate(True, None, [], "no Hamiltonian cycle exists on graph")

    # 回溯路径（不含最后回到 start 的一跳）
    rev = [best_last]
    cur = (full, best_last)
    while cur in parent:
        pmask, pu = parent[cur]
        rev.append(pu)
        cur = (pmask, pu)
    rev.reverse()
    if not rev or rev[0] != s:
        rev = [s] + [v for v in rev if v != s]
    tour = rev + [s]
    return OptimalityCertificate(True, best, tour, "")
