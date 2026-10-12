#!/usr/bin/env python3
"""Run the frozen BM benchmark, writing only to a new run/stage directory."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "scripts"))
from paths import check_new_output, new_output


def commands(run_dir, raw_root, stage):
    scripts = HERE / "scripts"
    plan = {
        "data": ["build_benchmark.py", "--raw-root", str(raw_root),
                 "--output-dir", str(run_dir / "data")],
        "relabel": ["build_relabel.py", "--benchmark-root", str(run_dir / "data"),
                    "--output-dir", str(run_dir / "relabel")],
        "alignment": ["run_alignment.py", "--relabel-root", str(run_dir / "relabel"),
                      "--output-dir", str(run_dir / "alignment")],
        "umap": ["plot_umap.py", "--alignment-root", str(run_dir / "alignment"),
                 "--output-dir", str(run_dir / "umap")],
    }
    selected = list(plan) if stage == "all" else [stage]
    return [(name, [sys.executable, "-B", str(scripts / plan[name][0]),
                    *plan[name][1:]]) for name in selected]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, help="Provider BM root; required for data/all")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=["all", "data", "relabel", "alignment", "umap"], default="all")
    args = parser.parse_args()
    if args.stage in {"data", "all"} and args.raw_root is None:
        parser.error("--raw-root is required for data/all")
    raw_root = args.raw_root.expanduser().resolve(strict=True) if args.raw_root else None
    # An existing run is allowed only to add a missing individual stage.
    run_dir = args.run_dir.expanduser()
    if run_dir.is_symlink():
        parser.error("A symlink cannot be used as the run directory")
    if args.stage == "all":
        run_dir = new_output(run_dir, [raw_root])
    else:
        run_dir = run_dir.resolve()
    plan = commands(run_dir, raw_root, args.stage)
    for name, _ in plan:
        # The stage subprocess, not this preflight, claims its own output.
        check_new_output(run_dir / name, [raw_root] if raw_root else [])
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for name, command in plan:
        print(f"\n[{name}]", flush=True)
        subprocess.run(command, check=True, env=env)


if __name__ == "__main__":
    main()
