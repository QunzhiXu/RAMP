#!/usr/bin/env python3
"""Run or audit the canonical melanoma fastMNN benchmark."""

from __future__ import annotations

import os
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
CENTRAL_RUNNER = HERE.parents[1] / "fastmnn_benchmark_runner.py"
CONFIG = HERE / "config.json"
COMMANDS = {"preprocess", "align", "umap", "audit", "all"}


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "all"
    if command not in COMMANDS or len(sys.argv) > 2:
        choices = "|".join(sorted(COMMANDS))
        raise SystemExit(f"Usage: {Path(__file__).name} [{choices}]")
    if not CENTRAL_RUNNER.is_file():
        raise FileNotFoundError(CENTRAL_RUNNER)
    os.execv(
        sys.executable,
        [
            sys.executable,
            str(CENTRAL_RUNNER),
            command,
            "--config",
            str(CONFIG),
        ],
    )


if __name__ == "__main__":
    main()
