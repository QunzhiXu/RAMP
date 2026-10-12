#!/usr/bin/env python3
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
subprocess.run([sys.executable, str(HERE.parents[1] / "harmony_benchmark_runner.py"), "--config", str(HERE / "config.json"), "--variant", "both"], check=True)
