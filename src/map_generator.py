"""TSP 地图：随机 k-NN 图、连通修补、CSV 读写。"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np


@dataclass
class TSPGraph:
    """无向带权图：adjacency[i] = list of (j, weight)."""

    n: int
    points: np.ndarray  # shape (n, 2), float64
    adjacency: List[List[Tuple[int, float]]]


def euclidean_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def build_knn_graph(points: np.ndarray, k: int) -> List[List[Tuple[int, float]]]:
    """每点连 k 个最近邻，对称化；边权为欧氏距离。"""
    n = points.shape[0]
    k = min(max(1, k), n - 1)
    edges: set[Tuple[int, int]] = set()

    dist_mat = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(dist_mat, np.inf)

    for i in range(n):
        order = np.argsort(dist_mat[i])[:k]
        for j in order:
            if j == i:
                continue
            a, b = (i, j) if i < j else (j, i)
            edges.add((a, b))

    adjacency: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    for i, j in edges:
        w = euclidean_dist(points[i], points[j])
        adjacency[i].append((j, w))
        adjacency[j].append((i, w))

    for i in range(n):
        adjacency[i].sort(key=lambda t: t[0])

    return adjacency


def random_tsp_instance(
    n: int,
    k_neighbors: int,
    seed: int | None = None,
    bounds: Tuple[float, float] = (0.0, 1.0),
) -> TSPGraph:
    rng = np.random.default_rng(seed)
    low, high = bounds
    points = rng.uniform(low, high, size=(n, 2)).astype(np.float64)
    adjacency = build_knn_graph(points, k_neighbors)
    return TSPGraph(n=n, points=points, adjacency=adjacency)


def generate_random_map(
    n: int,
    k_neighbors: int,
    seed: int | None = None,
    bounds: Tuple[float, float] = (0.0, 1.0),
) -> TSPGraph:
    """与 random_tsp_instance 相同，便于在报告中称为「地图生成」。"""
    return random_tsp_instance(n, k_neighbors, seed=seed, bounds=bounds)


def ensure_connected_fallback(tsp: TSPGraph, rng: np.random.Generator | None = None) -> TSPGraph:
    """若图不连通，按全体点欧氏 MST 补边直至连通。"""
    rng = rng or np.random.default_rng()
    n = tsp.n
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j, w in tsp.adjacency[i]:
            if i < j:
                union(i, j)

    if all(find(i) == find(0) for i in range(n)):
        return tsp

    dist_mat = np.sqrt(((tsp.points[:, None, :] - tsp.points[None, :, :]) ** 2).sum(axis=2))
    edges: List[Tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((dist_mat[i, j], i, j))
    edges.sort(key=lambda t: t[0])

    adj = [list(row) for row in tsp.adjacency]
    seen: set[Tuple[int, int]] = {(i, j) if i < j else (j, i) for i in range(n) for j, _ in adj[i]}

    for w, i, j in edges:
        if find(i) == find(j):
            continue
        union(i, j)
        a, b = (i, j) if i < j else (j, i)
        if (a, b) not in seen:
            seen.add((a, b))
            adj[i].append((j, float(w)))
            adj[j].append((i, float(w)))
        if all(find(x) == find(0) for x in range(n)):
            break

    for i in range(n):
        adj[i].sort(key=lambda t: t[0])
    return TSPGraph(n=n, points=tsp.points.copy(), adjacency=adj)


def save_instance(tsp: TSPGraph, out_dir: str | Path, base_name: str) -> tuple[Path, Path]:
    """写入 points.csv 与 edges.csv。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    points_path = out_dir / f"{base_name}_points.csv"
    edges_path = out_dir / f"{base_name}_edges.csv"

    with points_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "x", "y"])
        for i in range(tsp.n):
            w.writerow([i, float(tsp.points[i, 0]), float(tsp.points[i, 1])])

    seen: set[tuple[int, int]] = set()
    with edges_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["u", "v", "weight"])
        for u in range(tsp.n):
            for v, wt in tsp.adjacency[u]:
                a, b = (u, v) if u < v else (v, u)
                if (a, b) in seen:
                    continue
                seen.add((a, b))
                w.writerow([a, b, float(wt)])

    return points_path, edges_path


def load_instance(points_csv: str | Path, edges_csv: str | Path) -> TSPGraph:
    points_csv = Path(points_csv)
    edges_csv = Path(edges_csv)
    with points_csv.open(newline="", encoding="utf-8") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r["id"]))
    old_ids = [int(r["id"]) for r in rows]
    remap = {oid: i for i, oid in enumerate(old_ids)}
    n = len(rows)
    points = np.array([[float(r["x"]), float(r["y"])] for r in rows], dtype=np.float64)

    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    seen: set[tuple[int, int]] = set()
    with edges_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            u, v = remap[int(row["u"])], remap[int(row["v"])]
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            w = float(row["weight"])
            adjacency[u].append((v, w))
            adjacency[v].append((u, w))
    for i in range(n):
        adjacency[i] = [
            (j, euclidean_dist(points[i], points[j])) for j, _ in adjacency[i]
        ]
        adjacency[i].sort(key=lambda t: t[0])
    return TSPGraph(n=n, points=points, adjacency=adjacency)
