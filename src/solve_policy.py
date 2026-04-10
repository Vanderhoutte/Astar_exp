"""按城市规模选择精确 A*、加权 A* 或贪心+模拟退火。"""

from __future__ import annotations

import itertools
import time
from typing import Any, Dict, Literal

from .astar_tsp_solver import AStarTSPSolver, SearchResult
from .map_generator import TSPGraph
from .monster_solver import solve_tsp_monster_bnb
from .solution_verifier import (
    exact_optimal_tour_small_n,
    normalize_start,
    validate_closed_tour,
)
from .tsp_heuristic_solvers import greedy_then_sa

SolverMode = Literal["exact", "weighted", "heuristic"]


def state_space_upper_bound(n: int) -> int:
    """位掩码×当前点×phase 的保守量级上界（非扩展次数预测）。"""
    return 2 * n * (2**n)


def _weighted_available(n: int, cfg: Dict[str, Any]) -> bool:
    eps = cfg.get("weighted_astar_epsilon")
    wmax = cfg.get("weighted_max_n")
    return (
        eps is not None
        and float(eps) > 1.0
        and wmax is not None
        and n <= int(wmax)
    )


def _edge_weights(tsp: TSPGraph) -> Dict[tuple[int, int], float]:
    w: Dict[tuple[int, int], float] = {}
    for u in range(tsp.n):
        for v_raw, ww in tsp.adjacency[u]:
            w[(u, int(v_raw))] = float(ww)
    return w


def _solve_tsp_bruteforce(
    tsp: TSPGraph, start: int, *, time_limit_sec: float | None = None
) -> SearchResult:
    """纯暴力：枚举除起点外所有排列，检查闭合回路并取最小代价。"""
    t0 = time.perf_counter()
    n = tsp.n
    if n <= 1:
        return SearchResult(
            False,
            [],
            0.0,
            0,
            0.0,
            "bruteforce: n must be >= 2",
            method="bruteforce",
            optimal=False,
        )

    w = _edge_weights(tsp)
    verts = [v for v in range(n) if v != start]
    checked = 0
    best = float("inf")
    best_tour: list[int] = []

    for perm in itertools.permutations(verts):
        checked += 1
        if time_limit_sec is not None and (time.perf_counter() - t0) >= time_limit_sec:
            return SearchResult(
                False,
                best_tour if best_tour else [],
                best if best_tour else 0.0,
                checked,
                time.perf_counter() - t0,
                "bruteforce: time limit exceeded",
                method="bruteforce",
                optimal=False,
            )

        prev = start
        total = 0.0
        feasible = True
        for v in perm:
            ww = w.get((prev, v))
            if ww is None:
                feasible = False
                break
            total += ww
            prev = v
        if not feasible:
            continue
        back = w.get((prev, start))
        if back is None:
            continue
        total += back
        if total < best:
            best = total
            best_tour = [start, *perm, start]

    if not best_tour:
        return SearchResult(
            False,
            [],
            0.0,
            checked,
            time.perf_counter() - t0,
            "bruteforce: no Hamiltonian cycle exists on graph",
            method="bruteforce",
            optimal=True,
        )
    return SearchResult(
        True,
        best_tour,
        best,
        checked,
        time.perf_counter() - t0,
        "",
        method="bruteforce",
        optimal=True,
    )


def _solve_tsp_held_karp(
    tsp: TSPGraph,
    start: int,
    *,
    time_limit_sec: float | None = None,
) -> SearchResult:
    """更强（非 A*）精确法：Held-Karp DP。"""
    cert = exact_optimal_tour_small_n(tsp, start=start, time_limit_sec=time_limit_sec)
    if not cert.certified:
        return SearchResult(
            False,
            [],
            0.0,
            0,
            0.0,
            cert.message or "held-karp not certified",
            method="held_karp_exact",
            optimal=False,
        )
    if cert.optimal_cost is None:
        return SearchResult(
            False,
            [],
            0.0,
            0,
            0.0,
            cert.message or "no Hamiltonian cycle exists on graph",
            method="held_karp_exact",
            optimal=True,
        )
    return SearchResult(
        True,
        list(cert.optimal_tour),
        float(cert.optimal_cost),
        0,
        0.0,
        "",
        method="held_karp_exact",
        optimal=True,
    )


def _solve_tsp_monster(
    tsp: TSPGraph,
    start: int,
    cfg: Dict[str, Any],
    *,
    time_limit_sec: float | None = None,
) -> SearchResult:
    """
    Monster 模式：分支定界（精确，超时可返回 best-so-far）。
    先用贪心+SA 生成可行上界以加速剪枝。
    """
    warm: SearchResult | None = None
    seed = cfg.get("fallback_sa_seed")
    seed_i = int(seed) if seed is not None else None
    try:
        warm = greedy_then_sa(
            tsp,
            start=start,
            sa_iterations=int(cfg.get("fallback_sa_iterations", 8000)),
            sa_T0=float(cfg.get("fallback_sa_T0", 1.0)),
            sa_T_min=float(cfg.get("fallback_sa_T_min", 1e-4)),
            sa_cool=float(cfg.get("fallback_sa_cool", 0.995)),
            sa_seed=seed_i,
            cfg=cfg,
        )
    except Exception:
        warm = None
    init_tour = list(warm.tour) if warm and warm.success and warm.tour else None
    init_cost = float(warm.cost) if warm and warm.success else None
    return solve_tsp_monster_bnb(
        tsp,
        start=start,
        initial_tour=init_tour,
        initial_cost=init_cost,
        time_limit_sec=time_limit_sec,
    )


def _solve_tsp_stronger_non_astar(
    tsp: TSPGraph,
    start: int,
    cfg: Dict[str, Any],
) -> SearchResult:
    """
    非 A* 的“更强链路”：
    1) 小规模优先 Held-Karp 精确；
    2) 超时/未认证/超阈值时回退 greedy+SA，尽量给出可行解。
    """
    hk_max_n = int(cfg.get("stronger_exact_max_n", 18))
    hk_tl_raw = cfg.get("stronger_exact_time_limit_sec")
    hk_tl = float(hk_tl_raw) if hk_tl_raw is not None else None
    if tsp.n <= hk_max_n:
        hk = _solve_tsp_held_karp(tsp, start, time_limit_sec=hk_tl)
        if hk.success:
            return hk
        if "not certified" in hk.message or "time limit exceeded" in hk.message:
            seed = cfg.get("fallback_sa_seed")
            seed_i = int(seed) if seed is not None else None
            res = greedy_then_sa(
                tsp,
                start=start,
                sa_iterations=int(cfg.get("fallback_sa_iterations", 8000)),
                sa_T0=float(cfg.get("fallback_sa_T0", 1.0)),
                sa_T_min=float(cfg.get("fallback_sa_T_min", 1e-4)),
                sa_cool=float(cfg.get("fallback_sa_cool", 0.995)),
                sa_seed=seed_i,
                cfg=cfg,
            )
            if res.success:
                res.message = (
                    ((res.message + "; ") if res.message else "")
                    + "stronger fallback: held-karp timeout"
                )
            return res
        # 已证明无解等强结论，直接返回。
        return hk

    seed = cfg.get("fallback_sa_seed")
    seed_i = int(seed) if seed is not None else None
    res = greedy_then_sa(
        tsp,
        start=start,
        sa_iterations=int(cfg.get("fallback_sa_iterations", 8000)),
        sa_T0=float(cfg.get("fallback_sa_T0", 1.0)),
        sa_T_min=float(cfg.get("fallback_sa_T_min", 1e-4)),
        sa_cool=float(cfg.get("fallback_sa_cool", 0.995)),
        sa_seed=seed_i,
        cfg=cfg,
    )
    if res.success:
        res.message = (
            (res.message + "; ") if res.message else ""
        ) + f"stronger fallback: n={tsp.n} > stronger_exact_max_n={hk_max_n}"
    return res


def recommend_solver_mode(n: int, cfg: Dict[str, Any]) -> SolverMode:
    if not cfg.get("auto_solver", True):
        return "exact"
    if cfg.get("force_exact_astar"):
        return "exact"
    amax = int(cfg.get("astar_max_n", 20))
    if n <= amax:
        return "exact"
    if _weighted_available(n, cfg):
        base_mode: SolverMode = "weighted"
    else:
        base_mode = "heuristic"

    if cfg.get("force_stronger_solver", False):
        # 仅用于显示“规模策略”，具体执行在 solve_tsp_auto 中走非 A* 路径。
        return base_mode
    return base_mode


def solve_tsp_auto(
    tsp: TSPGraph,
    heuristic: str,
    start: int,
    cfg: Dict[str, Any],
    max_expansions: int | None = None,
    time_limit_sec: float | None = None,
) -> SearchResult:
    start = normalize_start(start, tsp.n)
    if cfg.get("force_monster_solver", False):
        mmax = int(cfg.get("monster_max_n", 24))
        mtl_raw = cfg.get("monster_time_limit_sec")
        mtl = float(mtl_raw) if mtl_raw is not None else None
        if tsp.n > mmax:
            res = _solve_tsp_stronger_non_astar(tsp, start, cfg)
            res.message = (
                ((res.message + "; ") if res.message else "")
                + f"monster fallback: n={tsp.n} > monster_max_n={mmax}"
            )
        else:
            res = _solve_tsp_monster(
                tsp,
                start,
                cfg,
                time_limit_sec=mtl if mtl is not None else time_limit_sec,
            )
            # Monster 若没给出强结论（常见于超时无可行解），自动回退更强链路避免“明明有解却失败”。
            proved_no_solution = (
                (not res.success)
                and (res.method == "monster_bnb_exact")
                and ("no Hamiltonian cycle" in (res.message or ""))
            )
            if (not res.success) and (not proved_no_solution):
                fb = _solve_tsp_stronger_non_astar(tsp, start, cfg)
                if fb.success or "no Hamiltonian cycle" in (fb.message or ""):
                    fb.message = (
                        ((fb.message + "; ") if fb.message else "")
                        + f"monster fallback after: {res.message}"
                    )
                    res = fb
        mode: SolverMode | None = None
    elif cfg.get("force_bruteforce", False):
        bf_max_n = int(cfg.get("bruteforce_max_n", 11))
        if tsp.n > bf_max_n:
            return SearchResult(
                False,
                [],
                0.0,
                0,
                0.0,
                f"bruteforce disabled for n={tsp.n} > bruteforce_max_n={bf_max_n}",
                method="bruteforce",
                optimal=False,
            )
        bf_tl_raw = cfg.get("bruteforce_time_limit_sec")
        bf_tl = float(bf_tl_raw) if bf_tl_raw is not None else None
        res = _solve_tsp_bruteforce(tsp, start, time_limit_sec=bf_tl)
        # 暴力完整跑完时已是最优；后续验证链路保留用于一致性与校验。
        mode: SolverMode | None = None
    elif cfg.get("force_stronger_solver", False):
        res = _solve_tsp_stronger_non_astar(tsp, start, cfg)
        mode = None
    else:
        mode = recommend_solver_mode(tsp.n, cfg)

    verify_exact_max_n = int(cfg.get("verify_exact_max_n", 14))
    verify_exact_time_sec = cfg.get("verify_exact_time_sec")
    verify_exact_time = float(verify_exact_time_sec) if verify_exact_time_sec is not None else None

    if mode is not None:
        if mode == "exact":
            solver = AStarTSPSolver(tsp, heuristic=heuristic, start=start)
            res = solver.search(
                max_expansions=max_expansions,
                time_limit_sec=time_limit_sec,
                epsilon=1.0,
            )
        elif mode == "weighted":
            eps = float(cfg["weighted_astar_epsilon"])
            solver = AStarTSPSolver(tsp, heuristic=heuristic, start=start)
            res = solver.search(
                max_expansions=max_expansions,
                time_limit_sec=time_limit_sec,
                epsilon=eps,
            )
        else:
            seed = cfg.get("fallback_sa_seed")
            seed_i = int(seed) if seed is not None else None
            res = greedy_then_sa(
                tsp,
                start=start,
                sa_iterations=int(cfg.get("fallback_sa_iterations", 8000)),
                sa_T0=float(cfg.get("fallback_sa_T0", 1.0)),
                sa_T_min=float(cfg.get("fallback_sa_T_min", 1e-4)),
                sa_cool=float(cfg.get("fallback_sa_cool", 0.995)),
                sa_seed=seed_i,
                cfg=cfg,
            )

    # 严格合法性校验：success=True 必须是合法闭合回路
    if res.success:
        chk = validate_closed_tour(tsp, res.tour, start=start, cost=res.cost)
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

    # 小规模可证最优认证（对所有模式生效）
    if tsp.n <= verify_exact_max_n:
        cert = exact_optimal_tour_small_n(
            tsp, start=start, time_limit_sec=verify_exact_time
        )
        if cert.certified:
            if cert.optimal_cost is None:
                # 证明图上无回路：任何 success 都视为错误并转失败
                if res.success:
                    return SearchResult(
                        False,
                        [],
                        0.0,
                        res.expansions,
                        res.elapsed_sec,
                        "verifier contradiction: solver returned tour but exact verifier found no cycle",
                        method=res.method,
                        optimal=False,
                    )
                return SearchResult(
                    False,
                    [],
                    0.0,
                    res.expansions,
                    res.elapsed_sec,
                    "verified: no Hamiltonian cycle exists",
                    method=res.method,
                    optimal=False,
                )
            if res.success:
                if abs(res.cost - cert.optimal_cost) <= 1e-8:
                    res.optimal = True
                else:
                    res.optimal = False
                    res.message = (
                        (res.message + "; ") if res.message else ""
                    ) + f"verified suboptimal: gap={res.cost - cert.optimal_cost:.6g}"
        else:
            if res.success:
                res.message = ((res.message + "; ") if res.message else "") + cert.message
    elif res.success:
        res.message = (
            ((res.message + "; ") if res.message else "")
            + f"optimality uncertified: n={tsp.n} > verify_exact_max_n={verify_exact_max_n}"
        )
    return res
