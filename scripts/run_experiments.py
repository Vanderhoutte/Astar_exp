#!/usr/bin/env python3
"""批量填表（转发至 python main.py tables …）。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    rest = list(sys.argv[1:])
    if len(rest) >= 2 and rest[0] == "--config":
        cmd = [sys.executable, str(ROOT / "main.py"), "--config", rest[1], "tables", *rest[2:]]
    else:
        cmd = [sys.executable, str(ROOT / "main.py"), "tables", *rest]
    raise SystemExit(subprocess.call(cmd))
