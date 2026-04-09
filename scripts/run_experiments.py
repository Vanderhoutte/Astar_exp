#!/usr/bin/env python3
"""批量填表（转发至 python main.py tables …）。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    cmd = [sys.executable, str(ROOT / "main.py"), "tables", *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd))
