from __future__ import annotations

from main import _benchmark_scale

REPORT_KEYS = (
    "城市规模",
    "最优路径长度",
    "最差路径长度",
    "平均路径长度",
    "总迭代次数",
    "平均运行时间_s",
)


def test_benchmark_scale_has_report_columns() -> None:
    cfg = {
        "auto_solver": True,
        "astar_max_n": 20,
        "tables_start": 0,
        "verify_exact_max_n": 14,
    }
    row = _benchmark_scale(
        6,
        4,
        "euclidean",
        repeats=2,
        base_seed=12345,
        max_expansions=500_000,
        time_limit_sec=30.0,
        cfg=cfg,
    )
    for k in REPORT_KEYS:
        assert k in row
    assert row["城市规模"] == 6
    if row["成功次数"] > 0:
        assert isinstance(row["最优路径长度"], float)
        assert isinstance(row["最差路径长度"], float)
        assert isinstance(row["平均路径长度"], float)
        assert isinstance(row["总迭代次数"], int)
        assert isinstance(row["平均运行时间_s"], float)
        assert row["最优路径长度"] <= row["最差路径长度"]
