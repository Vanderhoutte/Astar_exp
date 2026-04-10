"""图上限定贪心 TSP + 仅合法边的模拟退火改进。"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

import numpy as np

from .astar_tsp_solver import EVENT_DONE, EVENT_GOAL, EVENT_POP, SearchResult
from .map_generator import TSPGraph
from .solution_verifier import normalize_start, validate_closed_tour


def build_undirected_edge_set(tsp: TSPGraph) -> Set[Tuple[int, int]]:
    s: Set[Tuple[int, int]] = set()
    for u in range(tsp.n):
        for v, _ in tsp.adjacency[u]:
            a, b = (u, v) if u < v else (v, u)
            s.add((a, b))
    return s


def build_directed_weight(tsp: TSPGraph) -> Dict[Tuple[int, int], float]:
    m: Dict[Tuple[int, int], float] = {}
    for u in range(tsp.n):
        for v, w in tsp.adjacency[u]:
            m[(u, v)] = float(w)
    return m


def tour_edges_valid(seq: List[int], edge_set: Set[Tuple[int, int]]) -> bool:
    """seq 为 n 个互异顶点按访问顺序，检查 Hamilton 路径边及末点回到首点。"""
    n = len(seq)
    if n < 2:
        return False
    for i in range(n - 1):
        a, b = seq[i], seq[i + 1]
        e = (a, b) if a < b else (b, a)
        if e not in edge_set:
            return False
    a, b = seq[-1], seq[0]
    e = (a, b) if a < b else (b, a)
    return e in edge_set


def closed_tour_cost(seq: List[int], wdir: Dict[Tuple[int, int], float]) -> float:
    n = len(seq)
    if n < 2:
        return float("inf")
    s = 0.0
    for i in range(n - 1):
        s += wdir[(seq[i], seq[i + 1])]
    s += wdir[(seq[-1], seq[0])]
    return s


def _closed_tour_from_path(path: List[int], start: int, adj: List[List[Tuple[int, float]]]) -> Tuple[List[int], float] | None:
    """path 为含 n 个互异顶点的访问序列且 path[0]==start；若末点能回到 start 则返回 (tour, cost)。"""
    if not path or path[0] != start:
        return None
    n = len(path)
    if n < 2:
        return None
    current = path[-1]
    back_w: float | None = None
    for nxt, w in adj[current]:
        if int(nxt) == start:
            back_w = float(w)
            break
    if back_w is None:
        return None
    total = back_w
    for i in range(n - 1):
        a, b = path[i], path[i + 1]
        found = False
        for nxt, w in adj[a]:
            if int(nxt) == b:
                total += float(w)
                found = True
                break
        if not found:
            return None
    return path + [start], total


def randomized_rcl_nn_tour(
    tsp: TSPGraph,
    start: int,
    rng: np.random.Generator,
    *,
    max_tries: int,
    rcl_size: int,
) -> SearchResult | None:
    """从候选边权最小的前 rcl_size 条未访问邻边中随机选下一步，多轮重启；任一轮得到闭合回路即返回最优一轮。"""
    t0 = time.perf_counter()
    n = tsp.n
    adj = tsp.adjacency
    rcl_size = max(1, int(rcl_size))
    best_tour: List[int] | None = None
    best_cost = float("inf")
    best_steps = 0

    for _ in range(max(1, int(max_tries))):
        visited = {start}
        path: List[int] = [start]
        current = start
        steps = 0
        ok = True
        while len(visited) < n:
            cand = [(int(v), float(w)) for v, w in adj[current] if v not in visited]
            if not cand:
                ok = False
                break
            cand.sort(key=lambda t: t[1])
            k = min(rcl_size, len(cand))
            pick = int(rng.integers(0, k))
            nxt, w = cand[pick]
            path.append(nxt)
            visited.add(nxt)
            current = nxt
            steps += 1
        if not ok:
            continue
        closed = _closed_tour_from_path(path, start, adj)
        if closed is None:
            continue
        tour, cost = closed
        if cost < best_cost:
            best_cost = cost
            best_tour = tour
            best_steps = steps

    if best_tour is None:
        return None
    return SearchResult(
        True,
        best_tour,
        best_cost,
        best_steps,
        time.perf_counter() - t0,
        "",
        method="greedy_nn",
        optimal=False,
    )


def hamiltonian_cycle_backtrack(
    tsp: TSPGraph,
    start: int,
    rng: np.random.Generator,
    *,
    max_steps: int,
    time_limit_sec: float | None = None,
) -> List[int] | None:
    """深度优先搜索图上一条经过全部顶点并回到起点的回路；限步数/时限。"""
    n = tsp.n
    adj = tsp.adjacency
    path: List[int] = [start]
    visited: Set[int] = {start}
    steps = 0
    t0 = time.perf_counter()

    def over_time() -> bool:
        return time_limit_sec is not None and (time.perf_counter() - t0) >= time_limit_sec

    def recurse() -> bool:
        nonlocal steps
        if over_time():
            return False
        steps += 1
        if steps > max_steps:
            return False
        if len(path) == n:
            cur = path[-1]
            for nxt, _ in adj[cur]:
                if int(nxt) == start:
                    return True
            return False
        cur = path[-1]
        cand = [(int(v), float(w)) for v, w in adj[cur] if v not in visited]
        order = np.arange(len(cand), dtype=np.int64)
        rng.shuffle(order)
        for idx in order:
            nxt, _ = cand[int(idx)]
            visited.add(nxt)
            path.append(nxt)
            if recurse():
                return True
            path.pop()
            visited.remove(nxt)
        return False

    if recurse():
        return path + [start]
    return None


def greedy_nn_tour(tsp: TSPGraph, start: int = 0) -> SearchResult:
    """在邻接表上：每步选到未访问邻居的最小边权；最后需存在回到 start 的边。"""
    t0 = time.perf_counter()
    n = tsp.n
    start = normalize_start(start, n)
    adj = tsp.adjacency
    visited = {start}
    path: List[int] = [start]
    current = start
    total = 0.0
    steps = 0

    while len(visited) < n:
        best_nxt: int | None = None
        best_w = float("inf")
        for nxt, w in adj[current]:
            if nxt in visited:
                continue
            if w < best_w:
                best_w = float(w)
                best_nxt = int(nxt)
        if best_nxt is None:
            return SearchResult(
                False,
                list(path),
                total,
                steps,
                time.perf_counter() - t0,
                "greedy_nn: no unvisited neighbor",
                method="greedy_nn",
                optimal=False,
            )
        total += best_w
        visited.add(best_nxt)
        path.append(best_nxt)
        current = best_nxt
        steps += 1

    back_w: float | None = None
    for nxt, w in adj[current]:
        if int(nxt) == start:
            back_w = float(w)
            break
    if back_w is None:
        return SearchResult(
            False,
            list(path),
            total,
            steps,
            time.perf_counter() - t0,
            "greedy_nn: cannot close to start",
            method="greedy_nn",
            optimal=False,
        )
    total += back_w
    tour = path + [start]
    return SearchResult(
        True,
        tour,
        total,
        steps,
        time.perf_counter() - t0,
        "",
        method="greedy_nn",
        optimal=False,
    )


def _two_opt_move(seq: List[int], i: int, j: int) -> List[int]:
    """反转 seq[i+1 : j+1]（含端点），要求 j >= i + 2。"""
    return seq[: i + 1] + list(reversed(seq[i + 1 : j + 1])) + seq[j + 1 :]


def simulated_annealing_tour(
    initial_seq: List[int],
    edge_set: Set[Tuple[int, int]],
    wdir: Dict[Tuple[int, int], float],
    *,
    iterations: int,
    T0: float,
    T_min: float,
    cool: float,
    rng: np.random.Generator,
) -> Tuple[List[int], float, int]:
    """固定 seq[0] 为起点，仅对内部做 2-opt；Metropolis + 几何降温。"""
    seq = list(initial_seq)
    if not tour_edges_valid(seq, edge_set):
        return seq, float("inf"), 0
    cur = closed_tour_cost(seq, wdir)
    best_seq = list(seq)
    best = cur
    proposals = 0
    T = max(T0, 1e-9)
    n = len(seq)
    # 需 i>=1、j<=n-2 且 j>=i+2，故至少 n>=5 才有非空 2-opt
    if n < 5:
        return best_seq, best, 0

    for _ in range(iterations):
        proposals += 1
        i = int(rng.integers(1, n - 3))
        j = int(rng.integers(i + 2, n - 1))
        cand = _two_opt_move(seq, i, j)
        if not tour_edges_valid(cand, edge_set):
            continue
        new_c = closed_tour_cost(cand, wdir)
        delta = new_c - cur
        if delta < 0 or rng.random() < math.exp(-delta / T):
            seq = cand
            cur = new_c
            if cur < best:
                best = cur
                best_seq = list(seq)
        T = max(T * cool, T_min)

    return best_seq, best, proposals


def _feasible_tour_after_deterministic_fail(
    tsp: TSPGraph,
    start: int,
    rng: np.random.Generator,
    cfg: Dict[str, Any],
    wdir: Dict[Tuple[int, int], float],
) -> SearchResult | None:
    """确定性最近邻已失败时：RCL 随机贪心多重启 → 可选回溯搜哈密顿回路。"""
    rt = int(cfg.get("fallback_random_nn_tries", 500))
    rcl = int(cfg.get("fallback_rcl_size", 4))
    alt = randomized_rcl_nn_tour(tsp, start, rng, max_tries=rt, rcl_size=rcl)
    if alt is not None:
        return alt
    if not cfg.get("fallback_backtrack", True):
        return None
    bt_steps = int(cfg.get("fallback_backtrack_max_steps", 3_000_000))
    restarts = max(1, int(cfg.get("fallback_backtrack_restarts", 24)))
    steps_per = max(200_000, bt_steps // restarts)
    bt_tl = cfg.get("fallback_backtrack_time_sec")
    bt_tl_f = float(bt_tl) if bt_tl is not None else None
    hc: List[int] | None = None
    for _ in range(restarts):
        child_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        hc = hamiltonian_cycle_backtrack(
            tsp, start, child_rng, max_steps=steps_per, time_limit_sec=bt_tl_f
        )
        if hc is not None:
            break
    if hc is None:
        return None
    seq_bt = hc[:-1]
    cost_bt = closed_tour_cost(seq_bt, wdir)
    if not math.isfinite(cost_bt):
        return None
    return SearchResult(
        True,
        hc,
        cost_bt,
        len(seq_bt),
        0.0,
        "",
        method="greedy_nn",
        optimal=False,
    )


def greedy_then_sa(
    tsp: TSPGraph,
    start: int = 0,
    *,
    sa_iterations: int = 8000,
    sa_T0: float = 1.0,
    sa_T_min: float = 1e-4,
    sa_cool: float = 0.995,
    sa_seed: int | None = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> SearchResult:
    """可行初解（最近邻 / RCL 随机 / 回溯回路）+ 合法边上 2-opt 模拟退火。"""
    t0 = time.perf_counter()
    policy_cfg = cfg or {}
    start = normalize_start(start, tsp.n)
    rng = np.random.default_rng(sa_seed)
    edge_set = build_undirected_edge_set(tsp)
    wdir = build_directed_weight(tsp)

    g0 = greedy_nn_tour(tsp, start=start)
    init: SearchResult | None = g0 if g0.success else None
    if init is None:
        init = _feasible_tour_after_deterministic_fail(
            tsp, start, rng, policy_cfg, wdir
        )
    if init is None or not init.success or len(init.tour) < tsp.n + 1:
        return SearchResult(
            False,
            g0.tour,
            g0.cost,
            g0.expansions,
            time.perf_counter() - t0,
            g0.message or "heuristic: no feasible tour (增大 k 或换 seed)",
            method="greedy_nn",
            optimal=False,
        )

    g = init
    seq = g.tour[:-1]
    assert len(seq) == tsp.n

    best_seq, best_cost, prop = simulated_annealing_tour(
        seq,
        edge_set,
        wdir,
        iterations=sa_iterations,
        T0=sa_T0,
        T_min=sa_T_min,
        cool=sa_cool,
        rng=rng,
    )

    if not math.isfinite(best_cost):
        return SearchResult(
            False,
            g.tour,
            g.cost,
            g.expansions + prop,
            time.perf_counter() - t0,
            "greedy+sa: SA found no valid state",
            method="greedy_sa",
            optimal=False,
        )

    if best_seq[0] != start:
        # 固定起点 2-opt 下应不变；若不一致则旋转回起点
        idx = best_seq.index(start)
        best_seq = best_seq[idx:] + best_seq[:idx]
    tour = list(best_seq) + [start]
    expansions = g.expansions + prop
    res = SearchResult(
        True,
        tour,
        best_cost,
        expansions,
        time.perf_counter() - t0,
        "sa_refined",
        method="greedy_sa",
        optimal=False,
    )
    chk = validate_closed_tour(tsp, res.tour, start=start, cost=res.cost)
    if not chk.valid:
        return SearchResult(
            False,
            list(res.tour),
            float(chk.computed_cost),
            expansions,
            time.perf_counter() - t0,
            f"heuristic validation failed: {chk.message}",
            method="greedy_sa",
            optimal=False,
        )
    res.tour = chk.normalized_tour
    res.cost = chk.computed_cost
    return res


def search_heuristic_stepwise(
    tsp: TSPGraph,
    start: int,
    cfg: Dict[str, Any],
    max_expansions: Optional[int] = None,
    time_limit_sec: Optional[float] = None,
    *,
    sa_progress_every: int = 40,
) -> Generator[Dict[str, Any], None, Optional[SearchResult]]:
    """与 greedy_then_sa 相同的初解策略；步进重放贪心前缀或展示 Fallback 回路，再 SA。"""
    t0 = time.perf_counter()
    start = normalize_start(start, tsp.n)
    rng = np.random.default_rng(
        int(cfg["fallback_sa_seed"]) if cfg.get("fallback_sa_seed") is not None else None
    )
    edge_set = build_undirected_edge_set(tsp)
    wdir = build_directed_weight(tsp)
    n = tsp.n

    def elapsed() -> float:
        return time.perf_counter() - t0

    def over_time() -> bool:
        return time_limit_sec is not None and elapsed() >= time_limit_sec

    def over_exp(exp: int) -> bool:
        return max_expansions is not None and exp >= max_expansions

    expansions = 0

    def pop_path(path: List[int], *, stage: str, current: int | None = None) -> Dict[str, Any]:
        nonlocal expansions
        expansions += 1
        ev: Dict[str, Any] = {
            "event": EVENT_POP,
            "path": list(path),
            "expansions": expansions,
            "heuristic": True,
            "stage": stage,
        }
        if current is not None:
            ev["current"] = current
        return ev

    g0 = greedy_nn_tour(tsp, start=start)
    init: SearchResult | None = g0 if g0.success else None
    if init is None:
        init = _feasible_tour_after_deterministic_fail(tsp, start, rng, cfg, wdir)

    if init is None or not init.success:
        if g0.tour and len(g0.tour) >= 2:
            cur_last = g0.tour[-1]
            yield pop_path(list(g0.tour), stage="贪心失败", current=cur_last)
        res = SearchResult(
            False,
            list(g0.tour),
            g0.cost,
            expansions,
            elapsed(),
            g0.message or "heuristic: no feasible tour",
            method="greedy_nn",
            optimal=False,
        )
        yield {"event": EVENT_DONE, "result": res}
        return None

    tour_closed = list(init.tour)
    seq_vertices = tour_closed[:-1]
    init_cost = float(init.cost)

    if g0.success:
        for i in range(1, n):
            if over_time():
                res = SearchResult(
                    False,
                    seq_vertices[: i + 1],
                    0.0,
                    expansions,
                    elapsed(),
                    "time limit",
                    method="greedy_nn",
                    optimal=False,
                )
                yield {"event": EVENT_DONE, "result": res}
                return None
            if over_exp(expansions + 1):
                res = SearchResult(
                    False,
                    seq_vertices[: i + 1],
                    0.0,
                    expansions,
                    elapsed(),
                    "expansion limit exceeded",
                    method="greedy_nn",
                    optimal=False,
                )
                yield {"event": EVENT_DONE, "result": res}
                return None
            prefix = seq_vertices[: i + 1]
            yield pop_path(prefix, stage="贪心", current=prefix[-1])
    else:
        if over_time():
            res = SearchResult(
                False,
                tour_closed,
                init_cost,
                expansions,
                elapsed(),
                "time limit",
                method="greedy_nn",
                optimal=False,
            )
            yield {"event": EVENT_DONE, "result": res}
            return None
        if over_exp(expansions + 1):
            res = SearchResult(
                False,
                tour_closed,
                init_cost,
                expansions,
                elapsed(),
                "expansion limit exceeded",
                method="greedy_nn",
                optimal=False,
            )
            yield {"event": EVENT_DONE, "result": res}
            return None
        yield pop_path(tour_closed, stage="RCL/回溯初解")

    yield {
        "event": EVENT_GOAL,
        "tour": tour_closed,
        "cost": init_cost,
        "g": init_cost,
        "h": 0.0,
        "f": init_cost,
        "expansions": expansions,
        "heuristic": True,
        "stage": "初解闭合",
    }

    sa_iters = int(cfg.get("fallback_sa_iterations", 8000))
    sa_T0 = float(cfg.get("fallback_sa_T0", 1.0))
    sa_T_min = float(cfg.get("fallback_sa_T_min", 1e-4))
    sa_cool = float(cfg.get("fallback_sa_cool", 0.995))

    if n < 5:
        res = SearchResult(
            True,
            tour_closed,
            init_cost,
            expansions,
            elapsed(),
            "",
            method="greedy_sa",
            optimal=False,
        )
        yield {"event": EVENT_DONE, "result": res}
        return res

    cur_seq = list(seq_vertices)
    if not tour_edges_valid(cur_seq, edge_set):
        res = SearchResult(
            False,
            tour_closed,
            init_cost,
            expansions,
            elapsed(),
            "heuristic stepwise: invalid tour",
            method="greedy_nn",
            optimal=False,
        )
        yield {"event": EVENT_DONE, "result": res}
        return None

    cur_cost = closed_tour_cost(cur_seq, wdir)
    best_seq = list(cur_seq)
    best_cost = cur_cost
    T = max(sa_T0, 1e-9)
    progress_every = max(1, int(sa_progress_every))

    if not over_time() and not over_exp(expansions + 1):
        yield pop_path(list(best_seq) + [start], stage="模拟退火(初)")

    for it in range(sa_iters):
        if over_time():
            break
        if over_exp(expansions):
            break
        proposals = it + 1
        i = int(rng.integers(1, n - 3))
        j = int(rng.integers(i + 2, n - 1))
        cand = _two_opt_move(cur_seq, i, j)
        if not tour_edges_valid(cand, edge_set):
            T = max(T * sa_cool, sa_T_min)
            continue
        new_c = closed_tour_cost(cand, wdir)
        delta = new_c - cur_cost
        if delta < 0 or rng.random() < math.exp(-delta / T):
            cur_seq = cand
            cur_cost = new_c
            if cur_cost < best_cost:
                best_cost = cur_cost
                best_seq = list(cur_seq)
        T = max(T * sa_cool, sa_T_min)

        if proposals % progress_every == 0 or it == sa_iters - 1:
            if over_exp(expansions + 1):
                break
            bt = list(best_seq) + [start]
            yield pop_path(bt, stage="模拟退火")

    if best_seq[0] != start:
        idx = best_seq.index(start)
        best_seq = best_seq[idx:] + best_seq[:idx]
    final_tour = list(best_seq) + [start]
    res = SearchResult(
        True,
        final_tour,
        best_cost,
        expansions,
        elapsed(),
        "sa_refined",
        method="greedy_sa",
        optimal=False,
    )
    chk = validate_closed_tour(tsp, res.tour, start=start, cost=res.cost)
    if not chk.valid:
        res_bad = SearchResult(
            False,
            list(res.tour),
            float(chk.computed_cost),
            expansions,
            elapsed(),
            f"heuristic validation failed: {chk.message}",
            method="greedy_sa",
            optimal=False,
        )
        yield {"event": EVENT_DONE, "result": res_bad}
        return None
    res.tour = chk.normalized_tour
    res.cost = chk.computed_cost
    yield {
        "event": EVENT_GOAL,
        "tour": list(res.tour),
        "cost": res.cost,
        "g": res.cost,
        "h": 0.0,
        "f": res.cost,
        "expansions": expansions,
        "heuristic": True,
        "stage": "SA 最优",
    }
    yield {"event": EVENT_DONE, "result": res}
    return res
