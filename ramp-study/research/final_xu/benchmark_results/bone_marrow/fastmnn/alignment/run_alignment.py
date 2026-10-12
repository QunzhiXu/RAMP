#!/usr/bin/env python3
"""Run the authenticated bone-marrow fastMNN alignment."""

from pathlib import Path
import os
import sys

HERE = Path(__file__).resolve().parent
METHOD = HERE.parent
RUNNER = METHOD.parents[1] / "fastmnn_benchmark_runner.py"
os.execv(
    sys.executable,
    [sys.executable, str(RUNNER), "align", "--config", str(METHOD / "benchmark_config.json")],
)
