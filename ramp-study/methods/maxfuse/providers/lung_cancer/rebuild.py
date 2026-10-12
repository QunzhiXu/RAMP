"""Explicit source extraction then method scaling; never fit MaxFuse."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from lung_support import HERE, create_json, sha, reserve


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("raw-root", "benchmark-root", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    args = p.parse_args(argv)
    raw, benchmark = args.raw_root.resolve(strict=True), args.benchmark_root.resolve(strict=True)
    root = reserve(args.output, (raw, benchmark, HERE / "source"))
    commands = [
        [sys.executable, "-B", str(HERE / "rebuild_source_active.py"), "--raw-root", str(raw),
         "--benchmark-root", str(benchmark), "--output", str(root / "source_active")],
        [sys.executable, "-B", str(HERE / "build_inputs.py"), "--benchmark-root", str(benchmark),
         "--source-active", str(root / "source_active"), "--output", str(root / "method_inputs")],
    ]
    create_json(root / "REQUEST.json", {"commands": commands, "launcher_sha256": sha(Path(__file__)),
                "new_maxfuse_fit_performed": False})
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        env[k] = "1"
    try:
        for index, cmd in enumerate(commands):
            with (root / f"stage_{index}.stdout.log").open("x") as stdout, (root / f"stage_{index}.stderr.log").open("x") as stderr:
                result = subprocess.run(cmd, cwd=root, env=env, stdout=stdout, stderr=stderr)
            if result.returncode:
                raise RuntimeError(f"Stage {index} failed; output/logs retained")
            print("PASS", ("source extraction", "method scaling")[index], flush=True)
        create_json(root / "COMPLETE.json", {"status": "complete", "cells": 60000,
            "stage_manifests": {n: sha(root / n / "manifest.json") for n in ("source_active", "method_inputs")}})
    except BaseException as error:
        create_json(root / "FAILED.json", {"error": repr(error), "partial_output_retained": True})
        raise
    print(root, flush=True)


if __name__ == "__main__":
    main()

