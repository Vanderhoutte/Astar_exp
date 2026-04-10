from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_gui_mindmap_drawio_exists() -> None:
    p = ROOT / "docs" / "gui_mindmap.drawio"
    assert p.is_file()
    assert "mxfile" in p.read_text(encoding="utf-8", errors="ignore")
