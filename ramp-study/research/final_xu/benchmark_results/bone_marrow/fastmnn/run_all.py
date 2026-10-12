#!/usr/bin/env python3
"""Build, align, plot, and audit the bone-marrow fastMNN benchmark."""

from pathlib import Path
import os
import sys

HERE = Path(__file__).resolve().parent
RUNNER = HERE.parents[1] / "fastmnn_benchmark_runner.py"
os.execv(
    sys.executable,
    [sys.executable, str(RUNNER), "all", "--config", str(HERE / "benchmark_config.json")],
)
