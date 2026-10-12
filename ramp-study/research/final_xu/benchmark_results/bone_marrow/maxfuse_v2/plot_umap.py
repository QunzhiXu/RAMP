#!/usr/bin/env python3
"""Plot the isolated bone-marrow MaxFuse v2 hubs after alignment completes."""
from pathlib import Path
import json
import os
import sys

HERE = Path(__file__).resolve().parent
COMMON = HERE.parents[1] / "_maxfuse_benchmark.py"
PYTHON = Path(
    os.environ.get(
        "MAXFUSE_PYTHON",
        HERE.parents[2] / "algorithm" / "maxfuse" / ".venv" / "bin" / "python",
    )
)
if not PYTHON.is_file():
    raise FileNotFoundError(f"Dedicated MaxFuse Python not found: {PYTHON}")
if any(arg == "--config" or arg.startswith("--config=") for arg in sys.argv[1:]):
    raise ValueError("This scoped wrapper owns --config; do not override it")
payload = json.loads((HERE / "config.json").read_text())
if payload.get("benchmark_slug") != "bone_marrow" or payload.get("result_subdir") != HERE.name:
    raise AssertionError("Scoped wrapper/config benchmark contract changed")
os.execv(
    str(PYTHON),
    [str(PYTHON), str(COMMON), "umap", *sys.argv[1:], "--config", str(HERE / "config.json")],
)
