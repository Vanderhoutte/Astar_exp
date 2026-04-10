"""Monster exact solver: branch-and-bound for sparse-graph TSP."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .astar_tsp_solver import SearchResult
from .map_generator import TSPGraph
from .solution_verifier import normalize_start, validate_closed_tour


@dataclass
class MonsterSolveStats:
    expansions: int = 0
    pruned_by_bound: int = 0
    pruned_by_dom: int = 0


def _edge_map(tsp: TSPGraph) -> Dict[Tuple[int, int], float]:
    w: Dict[Tuple[int, int], float] = {}
    for u in range(tsp.n):
        for v_raw, ww in tsp.adjacency[u]:
            w[(u, int(v_raw))] = float(ww)
    return w


def solve_tsp_monster_bnb(
    tsp: TSPGraph,
    *,
    start: int,
    initial_tour: Optional[List[int]] = None,
    initial_cost: Optional[float] = None,
    time_limit_sec: Optional[float] = None,
) -> SearchResult:
    """
    Exact branch-and-bound TSP on sparse graph (directed edge set view).

    - If search exhausts without timeout: returns certified optimal (optimal=True).
    - If timeout occurs: returns best-so-far if available (optimal=False, uncertified).
    """
    t0 = time.perf_counter()
    n = tsp.n
    s = normalize_start(start, n)
    full_mask = (1 << n) - 1
    w = _edge_map(tsp)

    def elapsed() -> float:
        return time.perf_counter() - t0

    def over_time() -> bool:
        return time_limit_sec is not None and elapsed() >= time_limit_sec

    stats = MonsterSolveStats()
    timed_out = False

    best_cost = float("inf")
    best_tour: List[int] = []
    if initial_tour and initial_cost is not None:
        chk = validate_closed_tour(tsp, list(initial_tour), start=s, cost=float(initial_cost))
        if chk.valid:
            best_cost = float(chk.computed_cost)
            best_tour = list(chk.normalized_tour)

    # Dominance pruning: same (mask, current), smaller g dominates.
    g_best: Dict[Tuple[int, int], float] = {(1 << s, s): 0.0}

    lb_cache: Dict[Tuple[int, int], float] = {}

    def lower_bound(mask: int, current: int) -> float:
        """
        Admissible directed-degree relaxation for remaining path current -> ... -> start.

        Remaining unvisited nodes U must each have one outgoing + one incoming.
        current needs one outgoing; start needs one incoming.
        For any feasible completion, completion_cost is >= max(sum_min_out, sum_min_in).
        """
        key = (mask, current)
        v = lb_cache.get(key)
        if v is not None:
            return v

        unvisited = [i for i in range(n) if ((mask >> i) & 1) == 0]
        if not unvisited:
            direct = w.get((current, s), float("inf"))
            lb_cache[key] = direct
            return direct

        in_candidates = [current, *unvisited]
        out_candidates = [*unvisited, s]

        out_sum = 0.0
        for u in [current, *unvisited]:
            best_out = float("inf")
            for vtx in out_candidates:
                if vtx == u:
                    continue
                ww = w.get((u, vtx))
                if ww is not None and ww < best_out:
                    best_out = ww
            if best_out == float("inf"):
                lb_cache[key] = float("inf")
                return float("inf")
            out_sum += best_out

        in_sum = 0.0
        for vtx in [*unvisited, s]:
            best_in = float("inf")
            for u in in_candidates:
                if u == vtx:
                    continue
                ww = w.get((u, vtx))
                if ww is not None and ww < best_in:
                    best_in = ww
            if best_in == float("inf"):
                lb_cache[key] = float("inf")
                return float("inf")
            in_sum += best_in

        lb = max(out_sum, in_sum)
        lb_cache[key] = lb
        return lb

    def dfs(mask: int, current: int, g: float, path: List[int]) -> None:
        nonlocal best_cost, best_tour, timed_out
        if timed_out:
            return
        if over_time():
            timed_out = True
            return

        stats.expansions += 1

        lb = lower_bound(mask, current)
        if lb == float("inf") or g + lb >= best_cost - 1e-12:
            stats.pruned_by_bound += 1
            return

        if mask == full_mask:
            back = w.get((current, s))
            if back is None:
                return
            total = g + back
            if total < best_cost - 1e-12:
                best_cost = total
                best_tour = [*path, s]
            return

        key = (mask, current)
        old = g_best.get(key)
        if old is not None and g > old + 1e-12:
            stats.pruned_by_dom += 1
            return
        g_best[key] = g

        # Expand likely-good edges first to tighten upper bound early.
        cand: List[Tuple[float, int]] = []
        for nxt_raw, ww in tsp.adjacency[current]:
            nxt = int(nxt_raw)
            if (mask >> nxt) & 1:
                continue
            cand.append((float(ww), nxt))
        cand.sort(key=lambda x: x[0])

        for ww, nxt in cand:
            ng = g + ww
            if ng >= best_cost - 1e-12:
                continue
            dfs(mask | (1 << nxt), nxt, ng, [*path, nxt])
            if timed_out:
                return

    dfs(1 << s, s, 0.0, [s])
    el = elapsed()

    if timed_out:
        if best_tour:
            return SearchResult(
                True,
                best_tour,
                float(best_cost),
                stats.expansions,
                el,
                (
                    "monster bnb timeout: returned best-so-far (uncertified); "
                    f"pruned_bound={stats.pruned_by_bound}, pruned_dom={stats.pruned_by_dom}"
                ),
                method="monster_bnb_anytime",
                optimal=False,
            )
        return SearchResult(
            False,
            [],
            0.0,
            stats.expansions,
            el,
            (
                "monster bnb timeout: no feasible tour found; "
                f"pruned_bound={stats.pruned_by_bound}, pruned_dom={stats.pruned_by_dom}"
            ),
            method="monster_bnb_anytime",
            optimal=False,
        )

    if best_tour:
        return SearchResult(
            True,
            best_tour,
            float(best_cost),
            stats.expansions,
            el,
            (
                f"monster bnb exact done; pruned_bound={stats.pruned_by_bound}, "
                f"pruned_dom={stats.pruned_by_dom}"
            ),
            method="monster_bnb_exact",
            optimal=True,
        )
    return SearchResult(
        False,
        [],
        0.0,
        stats.expansions,
        el,
        (
            f"monster bnb exact done: no Hamiltonian cycle; pruned_bound={stats.pruned_by_bound}, "
            f"pruned_dom={stats.pruned_by_dom}"
        ),
        method="monster_bnb_exact",
        optimal=True,
    )
