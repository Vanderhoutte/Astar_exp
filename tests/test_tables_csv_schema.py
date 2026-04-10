from __future__ import annotations

import csv
from pathlib import Path

from main import _write_table_csv


def test_write_table_csv_contains_report_headers(tmp_path: Path) -> None:
    rows = [
        {
            "城市规模": 10,
            "k近邻": 4,
            "启发": "euclidean",
            "重复次数": 1,
            "成功次数": 1,
            "失败次数": 0,
            "状态空间上界": 20480,
            "求解方法": "astar_exact",
            "是否最优": "是",
            "最优路径长度": 1.0,
            "最差路径长度": 1.0,
            "平均路径长度": 1.0,
            "总迭代次数": 100,
            "平均运行时间_s": 0.01,
        }
    ]
    out = _write_table_csv(rows, tmp_path / "t.csv")
    text = out.read_text(encoding="utf-8-sig")
    header = text.splitlines()[0]
    for name in (
        "城市规模",
        "最优路径长度",
        "最差路径长度",
        "平均路径长度",
        "总迭代次数",
        "平均运行时间_s",
    ):
        assert name in header
    with out.open(newline="", encoding="utf-8-sig") as f:
        r = list(csv.DictReader(f))
    assert len(r) == 1
    assert r[0]["城市规模"] == "10"
