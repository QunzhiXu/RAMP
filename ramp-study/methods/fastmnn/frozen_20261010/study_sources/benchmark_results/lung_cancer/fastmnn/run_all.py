#!/usr/bin/env python3
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
RUNNER = HERE.parents[1] / "fastmnn_benchmark_runner.py"
subprocess.run(
    [sys.executable, str(RUNNER), "all", "--config", str(HERE / "benchmark_config.json")],
    check=True,
)
