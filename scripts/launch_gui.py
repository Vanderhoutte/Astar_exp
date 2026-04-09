#!/usr/bin/env python3
"""启动 GUI。支持：python scripts/launch_gui.py [--config FILE]"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    rest = list(sys.argv[1:])
    if len(rest) >= 2 and rest[0] == "--config":
        cmd = [sys.executable, str(ROOT / "main.py"), "--config", rest[1], "gui", *rest[2:]]
    else:
        cmd = [sys.executable, str(ROOT / "main.py"), "gui", *rest]
    raise SystemExit(subprocess.call(cmd))
