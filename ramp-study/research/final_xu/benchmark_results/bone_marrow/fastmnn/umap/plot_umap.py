#!/usr/bin/env python3
"""Draw UMAP directly from the saved bone-marrow fastMNN embedding."""

from pathlib import Path
import os
import sys

HERE = Path(__file__).resolve().parent
METHOD = HERE.parent
RUNNER = METHOD.parents[1] / "fastmnn_benchmark_runner.py"
os.execv(
    sys.executable,
    [sys.executable, str(RUNNER), "umap", "--config", str(METHOD / "benchmark_config.json")],
)
