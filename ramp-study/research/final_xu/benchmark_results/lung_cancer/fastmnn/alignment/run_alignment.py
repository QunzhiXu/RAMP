#!/usr/bin/env python3
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
METHOD = HERE.parent
RUNNER = METHOD.parents[1] / "fastmnn_benchmark_runner.py"
subprocess.run(
    [sys.executable, str(RUNNER), "align", "--config", str(METHOD / "benchmark_config.json")],
    check=True,
)
