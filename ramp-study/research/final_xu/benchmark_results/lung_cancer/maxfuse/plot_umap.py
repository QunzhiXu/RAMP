#!/usr/bin/env python3
"""Draw all three lung-cancer MaxFuse hub UMAPs and their summary."""
from pathlib import Path
import os
import sys

HERE = Path(__file__).resolve().parent
COMMON = HERE.parents[1] / "_maxfuse_benchmark.py"
PYTHON = Path(os.environ.get("MAXFUSE_PYTHON", HERE.parents[2] / "algorithm" / "maxfuse" / ".venv" / "bin" / "python"))
if not PYTHON.is_file():
    raise FileNotFoundError(f"Dedicated MaxFuse Python not found: {PYTHON}")
os.execv(str(PYTHON), [str(PYTHON), str(COMMON), "umap", "--config", str(HERE / "config.json"), *sys.argv[1:]])
