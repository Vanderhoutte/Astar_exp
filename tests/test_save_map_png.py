from __future__ import annotations

import numpy as np

from src.map_generator import ensure_connected_fallback, generate_random_map
from src.visualization import save_map_png


def test_save_map_png_writes_nonempty_file(tmp_path) -> None:
    g = generate_random_map(8, 4, seed=99)
    g = ensure_connected_fallback(g, np.random.default_rng(99))
    path = tmp_path / "m.png"
    save_map_png(g, path, title="test")
    assert path.is_file()
    assert path.stat().st_size > 500
