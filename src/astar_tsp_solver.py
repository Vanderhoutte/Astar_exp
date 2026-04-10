"""通用 A* 闭合 TSP 求解器（bitmask 状态）。"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, List, Optional, TextIO, Tuple, Union

from .heuristics import HeuristicFn, get_heuristic, h_euclidean
from .map_generator import TSPGraph
from .solution_verifier import normalize_start, validate_closed_tour

State = Tuple[int, int, int]


@dataclass
class SearchResult:
    success: bool
    tour: List[int]
    cost: float
    expansions: int
    elapsed_sec: float
    message: str = ""
    method: str = "astar_exact"
    optimal: bool = True


AStarResult = SearchResult


def _reconstruct_tour(
    came_from: Dict[State, State],
    full_mask: int,
    start: int,
) -> List[int]:
    gstate: State = (full_mask, start, 1)
    if gstate not in came_from:
        return []
    pm, pc, pph = came_from[gstate]
    assert pph == 0
    path_rev: List[int] = []
    m, c, ph = pm, pc, pph
    while True:
        path_rev.append(c)
        key = (m, c, ph)
        if key not in came_from:
            break
        m, c, ph = came_from[key]
    seq = list(reversed(path_rev))
    return seq + [start]


def _reconstruct_prefix(
    came_from: Dict[State, State],
    mask: int,
    current: int,
    phase: int,
) -> List[int]:
    """从 came_from 回溯当前 POP 状态对应的路径前缀（城市序号，含当前）。"""
    path_rev: List[int] = []
    m, c, ph = mask, current, phase
    while True:
        path_rev.append(c)
        key: State = (m, c, ph)
        if key not in came_from:
            break
        m, c, ph = came_from[key]
    return list(reversed(path_rev))


EVENT_POP = "pop"
EVENT_PUSH = "push"
EVENT_GOAL = "goal"
EVENT_DONE = "done"


class AStarTSPSolver:
    def __init__(
        self,
        tsp: TSPGraph,
        heuristic: Union[str, HeuristicFn, None] = "euclidean",
        start: int = 0,
    ) -> None:
        self.tsp = tsp
        self._heuristic_arg: Union[str, HeuristicFn, None] = heuristic
        self.start = normalize_start(start, tsp.n)

    def _h_fn(self) -> HeuristicFn:
        if self._heuristic_arg is None:
            return h_euclidean
        if isinstance(self._heuristic_arg, str):
            return get_heuristic(self._heuristic_arg)
        return self._heuristic_arg

    def search(
        self,
        max_expansions: int | None = None,
        time_limit_sec: float | None = None,
        epsilon: float = 1.0,
        cancel_check: Optional[Callable[[], bool]] = None,
        *,
        progress_every_expansions: int | None = None,
        progress_every_sec: float | None = None,
        progress_stream: Optional[TextIO] = None,
        progress_prefix: str = "",
    ) -> SearchResult:
        h_fn = self._h_fn()
        tsp = self.tsp
        start = normalize_start(self.start, tsp.n)
        t0 = time.perf_counter()
        n = tsp.n
        full_mask = (1 << n) - 1
        adj = tsp.adjacency

        def elapsed() -> float:
            return time.perf_counter() - t0

        def over_time() -> bool:
            return time_limit_sec is not None and elapsed() >= time_limit_sec

        def cancelled() -> bool:
            return cancel_check is not None and bool(cancel_check())

        prog_n = (
            int(progress_every_expansions)
            if progress_every_expansions is not None and int(progress_every_expansions) > 0
            else 0
        )
        prog_sec = (
            float(progress_every_sec)
            if progress_every_sec is not None and float(progress_every_sec) > 0
            else 0.0
        )
        prog_out = progress_stream
        last_prog_t = t0

        def emit_progress(note: str) -> None:
            nonlocal last_prog_t
            if prog_out is None:
                return
            last_prog_t = time.perf_counter()
            pf = progress_prefix or "A*"
            prog_out.write(
                f"[{pf}] n={n} start={start} expansions={expansions} "
                f"elapsed={elapsed():.2f}s open_heap={len(open_heap)} {note}\n"
            )
            prog_out.flush()

        counter = 0
        start_state: State = (1 << start, start, 0)
        g_best: Dict[State, float] = {start_state: 0.0}
        came_from: Dict[State, State] = {}

        h0 = h_fn(1 << start, start, tsp, start)
        eps = float(epsilon)
        f0 = 0.0 + eps * h0
        open_heap: List[Tuple[float, float, int, int, int, int, float]] = []
        heapq.heappush(open_heap, (f0, h0, counter, 1 << start, start, 0, 0.0))
        counter += 1

        expansions = 0
        mth = "astar_exact" if eps <= 1.0 + 1e-12 else "astar_weighted"
        # 仅凭 epsilon=1 无法严格证明最优（启发可能不可采纳）；保守置 False。
        opt = False

        while open_heap:
            if cancelled():
                return SearchResult(
                    False,
                    [],
                    0.0,
                    expansions,
                    elapsed(),
                    "search cancelled",
                    method=mth,
                    optimal=opt,
                )
            if over_time():
                return SearchResult(
                    False,
                    [],
                    0.0,
                    expansions,
                    elapsed(),
                    "time limit exceeded",
                    method=mth,
                    optimal=opt,
                )
            f, h, _, mask, current, phase, g = heapq.heappop(open_heap)
            mask, current, phase = int(mask), int(current), int(phase)
            state: State = (mask, current, phase)
            if g_best.get(state, float("inf")) < g - 1e-12:
                continue
            expansions += 1
            if prog_out is not None:
                if prog_n and expansions % prog_n == 0:
                    emit_progress(f"(every {prog_n} expansions)")
                elif prog_sec and (time.perf_counter() - last_prog_t) >= prog_sec:
                    emit_progress(f"(every {prog_sec:g}s)")
            if max_expansions is not None and expansions > max_expansions:
                return SearchResult(
                    False,
                    [],
                    0.0,
                    expansions,
                    elapsed(),
                    "expansion limit exceeded",
                    method=mth,
                    optimal=opt,
                )

            if phase == 1 and mask == full_mask and current == start:
                tour = _reconstruct_tour(came_from, full_mask, start)
                chk = validate_closed_tour(tsp, tour, start=start, cost=g)
                if not chk.valid:
                    return SearchResult(
                        False,
                        list(tour),
                        float(chk.computed_cost),
                        expansions,
                        elapsed(),
                        f"internal validation failed: {chk.message}",
                        method=mth,
                        optimal=False,
                    )
                return SearchResult(
                    True,
                    chk.normalized_tour,
                    chk.computed_cost,
                    expansions,
                    elapsed(),
                    "",
                    method=mth,
                    optimal=opt,
                )

            if phase == 1:
                continue

            if mask == full_mask:
                for nxt_raw, w in adj[current]:
                    nxt = int(nxt_raw)
                    if nxt != start:
                        continue
                    new_g = g + w
                    gstate: State = (full_mask, start, 1)
                    if new_g >= g_best.get(gstate, float("inf")):
                        continue
                    g_best[gstate] = new_g
                    came_from[gstate] = (mask, current, 0)
                    heapq.heappush(
                        open_heap, (new_g, 0.0, counter, full_mask, start, 1, new_g)
                    )
                    counter += 1

            for nxt_raw, w in adj[current]:
                nxt = int(nxt_raw)
                if nxt == start:
                    continue
                if (mask >> nxt) & 1:
                    continue
                new_mask = mask | (1 << nxt)
                new_g = g + w
                nstate: State = (new_mask, nxt, 0)
                if new_g >= g_best.get(nstate, float("inf")):
                    continue
                g_best[nstate] = new_g
                came_from[nstate] = (mask, current, 0)
                hn = h_fn(new_mask, nxt, tsp, start)
                fn = new_g + eps * hn
                heapq.heappush(open_heap, (fn, hn, counter, new_mask, nxt, 0, new_g))
                counter += 1

        return SearchResult(
            False,
            [],
            0.0,
            expansions,
            elapsed(),
            "no solution (open empty)",
            method=mth,
            optimal=opt,
        )

    def search_stepwise(
        self,
        max_expansions: int | None = None,
        time_limit_sec: float | None = None,
        epsilon: float = 1.0,
        cancel_check: Optional[Callable[[], bool]] = None,
        *,
        progress_every_expansions: int | None = None,
        progress_every_sec: float | None = None,
        progress_stream: Optional[TextIO] = None,
        progress_prefix: str = "",
    ) -> Generator[Dict[str, Any], None, Optional[SearchResult]]:
        h_fn = self._h_fn()
        tsp = self.tsp
        start = normalize_start(self.start, tsp.n)
        t0 = time.perf_counter()
        n = tsp.n
        full_mask = (1 << n) - 1
        adj = tsp.adjacency

        def elapsed() -> float:
            return time.perf_counter() - t0

        def cancelled() -> bool:
            return cancel_check is not None and bool(cancel_check())

        prog_n = (
            int(progress_every_expansions)
            if progress_every_expansions is not None and int(progress_every_expansions) > 0
            else 0
        )
        prog_sec = (
            float(progress_every_sec)
            if progress_every_sec is not None and float(progress_every_sec) > 0
            else 0.0
        )
        prog_out = progress_stream
        last_prog_t = t0

        def emit_progress(note: str) -> None:
            nonlocal last_prog_t
            if prog_out is None:
                return
            last_prog_t = time.perf_counter()
            pf = progress_prefix or "A*"
            prog_out.write(
                f"[{pf}] n={n} start={start} expansions={expansions} "
                f"elapsed={elapsed():.2f}s open_heap={len(open_heap)} {note}\n"
            )
            prog_out.flush()

        counter = 0
        start_state: State = (1 << start, start, 0)
        g_best: Dict[State, float] = {start_state: 0.0}
        came_from: Dict[State, State] = {}

        h0 = h_fn(1 << start, start, tsp, start)
        eps = float(epsilon)
        f0 = 0.0 + eps * h0
        open_heap: List[Tuple[float, float, int, int, int, int, float]] = []
        heapq.heappush(open_heap, (f0, h0, counter, 1 << start, start, 0, 0.0))
        counter += 1

        expansions = 0
        mth = "astar_exact" if eps <= 1.0 + 1e-12 else "astar_weighted"
        # 仅凭 epsilon=1 无法严格证明最优（启发可能不可采纳）；保守置 False。
        opt = False

        while open_heap:
            if cancelled():
                yield {
                    "event": EVENT_DONE,
                    "result": SearchResult(
                        False,
                        [],
                        0.0,
                        expansions,
                        elapsed(),
                        "search cancelled",
                        method=mth,
                        optimal=opt,
                    ),
                }
                return None
            if time_limit_sec is not None and elapsed() >= time_limit_sec:
                yield {
                    "event": EVENT_DONE,
                    "result": SearchResult(
                        False,
                        [],
                        0.0,
                        expansions,
                        elapsed(),
                        "time limit",
                        method=mth,
                        optimal=opt,
                    ),
                }
                return None
            f, h, _, mask, current, phase, g = heapq.heappop(open_heap)
            mask, current, phase = int(mask), int(current), int(phase)
            state: State = (mask, current, phase)
            if g_best.get(state, float("inf")) < g - 1e-12:
                continue
            expansions += 1
            if prog_out is not None:
                if prog_n and expansions % prog_n == 0:
                    emit_progress(f"(every {prog_n} expansions)")
                elif prog_sec and (time.perf_counter() - last_prog_t) >= prog_sec:
                    emit_progress(f"(every {prog_sec:g}s)")
            if max_expansions is not None and expansions > max_expansions:
                yield {
                    "event": EVENT_DONE,
                    "result": SearchResult(
                        False,
                        [],
                        0.0,
                        expansions,
                        elapsed(),
                        "expansion limit",
                        method=mth,
                        optimal=opt,
                    ),
                }
                return None

            if cancelled():
                yield {
                    "event": EVENT_DONE,
                    "result": SearchResult(
                        False,
                        [],
                        0.0,
                        expansions,
                        elapsed(),
                        "search cancelled",
                        method=mth,
                        optimal=opt,
                    ),
                }
                return None

            yield {
                "event": EVENT_POP,
                "mask": mask,
                "current": current,
                "phase": phase,
                "g": g,
                "h": h,
                "f": f,
                "expansions": expansions,
                "path": _reconstruct_prefix(came_from, mask, current, phase),
            }

            if phase == 1 and mask == full_mask and current == start:
                tour = _reconstruct_tour(came_from, full_mask, start)
                chk = validate_closed_tour(tsp, tour, start=start, cost=g)
                if not chk.valid:
                    res_bad = SearchResult(
                        False,
                        list(tour),
                        float(chk.computed_cost),
                        expansions,
                        elapsed(),
                        f"internal validation failed: {chk.message}",
                        method=mth,
                        optimal=False,
                    )
                    yield {"event": EVENT_DONE, "result": res_bad}
                    return None
                res = SearchResult(
                    True,
                    chk.normalized_tour,
                    chk.computed_cost,
                    expansions,
                    elapsed(),
                    "",
                    method=mth,
                    optimal=opt,
                )
                yield {
                    "event": EVENT_GOAL,
                    "tour": chk.normalized_tour,
                    "cost": chk.computed_cost,
                    "g": chk.computed_cost,
                    "h": 0.0,
                    "f": chk.computed_cost,
                    "expansions": expansions,
                }
                yield {"event": EVENT_DONE, "result": res}
                return res

            if phase == 1:
                continue

            if mask == full_mask:
                for nxt_raw, w in adj[current]:
                    if cancelled():
                        yield {
                            "event": EVENT_DONE,
                            "result": SearchResult(
                                False,
                                [],
                                0.0,
                                expansions,
                                elapsed(),
                                "search cancelled",
                                method=mth,
                                optimal=opt,
                            ),
                        }
                        return None
                    nxt = int(nxt_raw)
                    if nxt != start:
                        continue
                    new_g = g + w
                    gstate: State = (full_mask, start, 1)
                    if new_g >= g_best.get(gstate, float("inf")):
                        continue
                    g_best[gstate] = new_g
                    came_from[gstate] = (mask, current, 0)
                    heapq.heappush(
                        open_heap, (new_g, 0.0, counter, full_mask, start, 1, new_g)
                    )
                    counter += 1
                    yield {
                        "event": EVENT_PUSH,
                        "to": start,
                        "from": current,
                        "g": new_g,
                        "h": 0.0,
                        "f": new_g,
                        "phase": 1,
                    }

            for nxt_raw, w in adj[current]:
                if cancelled():
                    yield {
                        "event": EVENT_DONE,
                        "result": SearchResult(
                            False,
                            [],
                            0.0,
                            expansions,
                            elapsed(),
                            "search cancelled",
                            method=mth,
                            optimal=opt,
                        ),
                    }
                    return None
                nxt = int(nxt_raw)
                if nxt == start:
                    continue
                if (mask >> nxt) & 1:
                    continue
                new_mask = mask | (1 << nxt)
                new_g = g + w
                nstate: State = (new_mask, nxt, 0)
                if new_g >= g_best.get(nstate, float("inf")):
                    continue
                g_best[nstate] = new_g
                came_from[nstate] = (mask, current, 0)
                hn = h_fn(new_mask, nxt, tsp, start)
                fn = new_g + eps * hn
                heapq.heappush(open_heap, (fn, hn, counter, new_mask, nxt, 0, new_g))
                counter += 1
                yield {
                    "event": EVENT_PUSH,
                    "to": nxt,
                    "from": current,
                    "g": new_g,
                    "h": hn,
                    "f": fn,
                    "phase": 0,
                }

        res = SearchResult(
            False,
            [],
            0.0,
            expansions,
            elapsed(),
            "no solution",
            method=mth,
            optimal=opt,
        )
        yield {"event": EVENT_DONE, "result": res}
        return None


def run_stepwise_to_end(
    gen: Generator[Dict[str, Any], None, Optional[SearchResult]],
) -> SearchResult:
    result: Optional[SearchResult] = None
    for ev in gen:
        if ev.get("event") == EVENT_DONE:
            r = ev.get("result")
            if isinstance(r, SearchResult):
                result = r
    if result is None:
        return SearchResult(
            False, [], 0.0, 0, 0.0, "no events", method="astar_exact", optimal=True
        )
    return result
