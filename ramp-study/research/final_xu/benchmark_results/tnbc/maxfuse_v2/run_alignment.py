#!/usr/bin/env python3
"""Run or validate TNBC MaxFuse v2 without touching historical MaxFuse output."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
FINAL_XU = HERE.parents[2]
COMMON = FINAL_XU / "benchmark_results" / "_maxfuse_benchmark.py"
CONFIG = HERE / "config.json"
DEFAULT_PYTHON = FINAL_XU / "algorithm" / "maxfuse" / ".venv" / "bin" / "python"


def main() -> None:
    if any(arg == "--config" or arg.startswith("--config=") for arg in sys.argv[1:]):
        raise ValueError("This scoped wrapper owns --config; do not override it")
    payload = json.loads(CONFIG.read_text())
    if payload.get("benchmark_slug") != "tnbc":
        raise AssertionError("Scoped wrapper requires benchmark_slug=tnbc")
    if payload.get("result_subdir") != HERE.name:
        raise AssertionError(
            f"Scoped wrapper requires result_subdir={HERE.name!r}; "
            f"observed {payload.get('result_subdir')!r}"
        )

    # Preserve the venv launcher path.  Resolving its ``python`` symlink would
    # execute the base interpreter directly and lose the venv site-packages.
    python = Path(os.environ.get("MAXFUSE_PYTHON", DEFAULT_PYTHON)).expanduser()
    if not python.is_file():
        raise FileNotFoundError(f"Dedicated MaxFuse Python not found: {python}")
    os.execv(
        str(python),
        [str(python), str(COMMON), *sys.argv[1:], "--config", str(CONFIG)],
    )


if __name__ == "__main__":
    main()
