"""Heuristic functions h(mask, current) for A* on TSP states."""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from .map_generator import TSPGraph

MetricFn = Callable[[np.ndarray, np.ndarray], float]


def dist_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def dist_l1(a: np.ndarray, b: np.ndarray) -> float:
    return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))


def dist_cheb(a: np.ndarray, b: np.ndarray) -> float:
    return float(max(abs(a[0] - b[0]), abs(a[1] - b[1])))


def _mst_cost(points_subset: np.ndarray, dist_fn: MetricFn) -> float:
    """Prim's algorithm; points_subset shape (m, 2)."""
    m = points_subset.shape[0]
    if m <= 1:
        return 0.0
    in_tree = np.zeros(m, dtype=bool)
    min_dist = np.array([dist_fn(points_subset[i], points_subset[0]) for i in range(m)])
    min_dist[0] = np.inf
    in_tree[0] = True
    total = 0.0
    for _ in range(m - 1):
        j = int(np.argmin(min_dist))
        total += float(min_dist[j])
        in_tree[j] = True
        min_dist[j] = np.inf
        for k in range(m):
            if not in_tree[k]:
                d = dist_fn(points_subset[j], points_subset[k])
                if d < min_dist[k]:
                    min_dist[k] = d
    return total


def _h_from_metric(
    mask: int,
    current: int,
    tsp: TSPGraph,
    start: int,
    dist_fn: MetricFn,
) -> float:
    n = tsp.n
    full = (1 << n) - 1
    pts = tsp.points
    if mask == full:
        return dist_fn(pts[current], pts[start])

    unvisited = [i for i in range(n) if not (mask >> i) & 1]
    idxs = unvisited + [current]
    sub = pts[np.array(idxs, dtype=int)]
    return _mst_cost(sub, dist_fn)


def h_euclidean(mask: int, current: int, tsp: TSPGraph, start: int = 0) -> float:
    return _h_from_metric(mask, current, tsp, start, dist_l2)


def h_manhattan(mask: int, current: int, tsp: TSPGraph, start: int = 0) -> float:
    """L1 MST + L1 return leg; may be inadmissible vs Euclidean edge weights (often larger)."""
    return _h_from_metric(mask, current, tsp, start, dist_l1)


def h_custom(mask: int, current: int, tsp: TSPGraph, start: int = 0) -> float:
    """L∞ MST + return leg (same scaffold as Euclidean). d∞≤d₂ per pair in ℝ²; graph uses Euclidean k-NN edges — admissibility vs true cost is non-trivial."""
    return _h_from_metric(mask, current, tsp, start, dist_cheb)


HeuristicFn = Callable[[int, int, TSPGraph, int], float]

HEURISTICS: Dict[str, HeuristicFn] = {
    "euclidean": h_euclidean,
    "manhattan": h_manhattan,
    "custom": h_custom,
}


def get_heuristic(name: str) -> HeuristicFn:
    key = name.lower().strip()
    if key not in HEURISTICS:
        raise KeyError(f"Unknown heuristic {name!r}; choose from {list(HEURISTICS)}")
    return HEURISTICS[key]
