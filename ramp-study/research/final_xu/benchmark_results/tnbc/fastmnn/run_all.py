#!/usr/bin/env python3
"""Run one TNBC fastMNN benchmark stage through the shared orchestrator."""

from pathlib import Path
import os
import sys


HERE = Path(__file__).resolve().parent
RUNNER = HERE.parents[1] / "fastmnn_benchmark_runner.py"
command = sys.argv[1] if len(sys.argv) > 1 else "all"
if command not in {"preprocess", "align", "umap", "audit", "all"}:
    raise SystemExit("Usage: run_all.py [preprocess|align|umap|audit|all]")
if len(sys.argv) > 2:
    raise SystemExit("Usage: run_all.py [preprocess|align|umap|audit|all]")
os.execv(
    sys.executable,
    [
        sys.executable,
        str(RUNNER),
        command,
        "--config",
        str(HERE / "benchmark_config.json"),
    ],
)
