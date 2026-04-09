"""通用 A* 闭合 TSP 求解器（bitmask 状态）。"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

from .heuristics import HeuristicFn, get_heuristic, h_euclidean
from .map_generator import TSPGraph

State = Tuple[int, int, int]


@dataclass
class SearchResult:
    success: bool
    tour: List[int]
    cost: float
    expansions: int
    elapsed_sec: float
    message: str = ""


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
        self.start = start

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
    ) -> SearchResult:
        h_fn = self._h_fn()
        tsp = self.tsp
        start = self.start
        t0 = time.perf_counter()
        n = tsp.n
        full_mask = (1 << n) - 1
        adj = tsp.adjacency

        def elapsed() -> float:
            return time.perf_counter() - t0

        def over_time() -> bool:
            return time_limit_sec is not None and elapsed() >= time_limit_sec

        counter = 0
        start_state: State = (1 << start, start, 0)
        g_best: Dict[State, float] = {start_state: 0.0}
        came_from: Dict[State, State] = {}

        h0 = h_fn(1 << start, start, tsp, start)
        f0 = h0
        open_heap: List[Tuple[float, float, int, int, int, int, float]] = []
        heapq.heappush(open_heap, (f0, h0, counter, 1 << start, start, 0, 0.0))
        counter += 1

        expansions = 0

        while open_heap:
            if over_time():
                return SearchResult(
                    False, [], 0.0, expansions, elapsed(), "time limit exceeded"
                )
            f, h, _, mask, current, phase, g = heapq.heappop(open_heap)
            state: State = (mask, current, phase)
            if g_best.get(state, float("inf")) < g - 1e-12:
                continue
            expansions += 1
            if max_expansions is not None and expansions > max_expansions:
                return SearchResult(
                    False, [], 0.0, expansions, elapsed(), "expansion limit exceeded"
                )

            if phase == 1 and mask == full_mask and current == start:
                tour = _reconstruct_tour(came_from, full_mask, start)
                return SearchResult(True, tour, g, expansions, elapsed(), "")

            if phase == 1:
                continue

            if mask == full_mask:
                for nxt, w in adj[current]:
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

            for nxt, w in adj[current]:
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
                fn = new_g + hn
                heapq.heappush(open_heap, (fn, hn, counter, new_mask, nxt, 0, new_g))
                counter += 1

        return SearchResult(
            False, [], 0.0, expansions, elapsed(), "no solution (open empty)"
        )

    def search_stepwise(
        self,
        max_expansions: int | None = None,
        time_limit_sec: float | None = None,
    ) -> Generator[Dict[str, Any], None, Optional[SearchResult]]:
        h_fn = self._h_fn()
        tsp = self.tsp
        start = self.start
        t0 = time.perf_counter()
        n = tsp.n
        full_mask = (1 << n) - 1
        adj = tsp.adjacency

        def elapsed() -> float:
            return time.perf_counter() - t0

        counter = 0
        start_state: State = (1 << start, start, 0)
        g_best: Dict[State, float] = {start_state: 0.0}
        came_from: Dict[State, State] = {}

        h0 = h_fn(1 << start, start, tsp, start)
        open_heap: List[Tuple[float, float, int, int, int, int, float]] = []
        heapq.heappush(open_heap, (h0, h0, counter, 1 << start, start, 0, 0.0))
        counter += 1

        expansions = 0

        while open_heap:
            if time_limit_sec is not None and elapsed() >= time_limit_sec:
                yield {
                    "event": EVENT_DONE,
                    "result": SearchResult(False, [], 0.0, expansions, elapsed(), "time limit"),
                }
                return None
            f, h, _, mask, current, phase, g = heapq.heappop(open_heap)
            state: State = (mask, current, phase)
            if g_best.get(state, float("inf")) < g - 1e-12:
                continue
            expansions += 1
            if max_expansions is not None and expansions > max_expansions:
                yield {
                    "event": EVENT_DONE,
                    "result": SearchResult(
                        False, [], 0.0, expansions, elapsed(), "expansion limit"
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
            }

            if phase == 1 and mask == full_mask and current == start:
                tour = _reconstruct_tour(came_from, full_mask, start)
                res = SearchResult(True, tour, g, expansions, elapsed(), "")
                yield {
                    "event": EVENT_GOAL,
                    "tour": tour,
                    "cost": g,
                    "g": g,
                    "h": 0.0,
                    "f": g,
                    "expansions": expansions,
                }
                yield {"event": EVENT_DONE, "result": res}
                return res

            if phase == 1:
                continue

            if mask == full_mask:
                for nxt, w in adj[current]:
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

            for nxt, w in adj[current]:
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
                fn = new_g + hn
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

        res = SearchResult(False, [], 0.0, expansions, elapsed(), "no solution")
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
        return SearchResult(False, [], 0.0, 0, 0.0, "no events")
    return result
