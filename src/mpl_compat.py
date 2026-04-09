"""
Silence Matplotlib's Axes3D UserWarning when Debian/Ubuntu's system
``python3-matplotlib`` (``/usr/lib/python3/dist-packages/mpl_toolkits``) is mixed
with a newer pip ``matplotlib`` in ``~/.local``: the old ``mplot3d`` cannot
import removed symbols from the new core library. 2D plotting still works; this
only hides the known noisy warning. For a real fix, see README「故障排除」.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings(
    "ignore",
    message="Unable to import Axes3D",
    category=UserWarning,
)
