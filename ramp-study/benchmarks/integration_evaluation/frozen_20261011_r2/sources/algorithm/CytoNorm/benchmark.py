#!/usr/bin/env python3
"""Read-only adapter for all four frozen final_xu benchmark_data cohorts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
DATA = PROJECT / "benchmark_data"
DEFAULT_OUTPUT = PROJECT / "algorithm_test" / "cytonorm_validation" / "results"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def configure(output):
    output = Path(output).resolve()
    if not output.is_relative_to((PROJECT / "algorithm_test").resolve()):
        raise ValueError("All run outputs must be under algorithm_test")
    output.mkdir(parents=True, exist_ok=True)
    for key, subdir in {"MPLCONFIGDIR": "cache/matplotlib", "NUMBA_CACHE_DIR": "cache/numba",
                        "XDG_CACHE_HOME": "cache/xdg", "RAMP_EVALUATION_CACHE": "cache/evaluation",
                        "JOBLIB_TEMP_FOLDER": "tmp"}.items():
        path = output / subdir
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
        os.environ[key] = "1"
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    return output


def load_source(case, cfg):
    import numpy as np
    import pandas as pd

    root = DATA / case
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    markers = manifest["markers_used_for_initial_alignment"] if case == "bone_marrow" else manifest["markers"]
    paths = [manifest_path, root / "marker_manifest.csv", root / "selected_cells.csv.gz"]
    parts = []
    for slug, dataset in zip(cfg["directories"], cfg["datasets"], strict=True):
        p = root / slug / f"alignment_input_{cfg['markers']}.csv.gz"
        paths.append(p)
        columns = ["cell_id", "dataset", "unit_key", *markers]
        part = pd.read_csv(p, usecols=columns, dtype={k: str for k in columns[:3]}, float_precision="round_trip")[columns]
        if part.dataset.unique().tolist() != [dataset]:
            raise ValueError(f"Wrong dataset in {p}")
        parts.append(part)
    table = pd.concat(parts, ignore_index=True)
    x = table[markers].to_numpy(dtype=np.float64)
    if x.shape != (cfg["cells"], cfg["markers"]) or not np.isfinite(x).all():
        raise ValueError(f"{case}: invalid matrix shape/values")
    if table.cell_id.duplicated().any() or table.isna().any().any() or table.unit_key.nunique() != cfg["units"]:
        raise ValueError(f"{case}: row identity / unit coverage failed")
    if table.groupby("unit_key").dataset.nunique().max() != 1:
        raise ValueError("Unit identifier crosses datasets")
    selection = pd.read_csv(root / "selected_cells.csv.gz", usecols=["cell_id"], dtype=str)
    if selection.cell_id.tolist() != table.cell_id.tolist():
        raise ValueError(f"{case}: frozen selection IDs/order changed")
    if case == "bone_marrow":
        if "CD19" in markers:
            raise ValueError("Held-out CD19 leakage")
    else:
        paths += [root / "input/raw.npy", root / "input/fit_metadata.csv.gz", root / "input/markers.txt"]
        canonical = np.load(root / "input/raw.npy", allow_pickle=False)
        fit_meta = pd.read_csv(root / "input/fit_metadata.csv.gz", dtype=str)
        if list(fit_meta) != ["dataset", "unit_key"] or not fit_meta.equals(table[["dataset", "unit_key"]]):
            raise ValueError("Canonical fitting metadata/order differs")
        if (root / "input/markers.txt").read_text().splitlines() != markers or not np.array_equal(canonical, x.astype(canonical.dtype)):
            raise ValueError("Per-platform values differ from canonical input/raw.npy")
        # CSV exports print short decimal forms of float32. Use the canonical
        # binary values after checking the export round trip and row identity.
        table[markers] = canonical.astype(np.float64)
    records = []
    for p in paths:
        actual = sha(p)
        known = manifest.get("files_sha256", {}).get(str(p.relative_to(root)))
        if known is not None and actual != known:
            raise ValueError(f"Frozen source checksum mismatch: {p}")
        records.append({"path": str(p), "sha256": actual})
    return table, markers, records


def transformed(table, markers, cfg):
    import numpy as np
    result = table.copy()
    audit = []
    for dataset, spec in zip(cfg["datasets"], cfg["transforms"], strict=True):
        rows = table.dataset == dataset
        x = table.loc[rows, markers].to_numpy(dtype=np.float64)
        if spec["operator"] == "identity":
            y = x.copy()
        elif spec["operator"] == "asinh":
            y = np.arcsinh(x / spec["cofactor"])
        elif spec["operator"] == "log1p":
            if np.any(x < 0):
                raise ValueError("log1p count preprocessing requires nonnegative counts")
            y = np.log1p(x)
        else:
            raise ValueError("Unknown preprocessing operator")
        if not np.isfinite(y).all():
            raise ValueError("Nonfinite preprocessing")
        result.loc[rows, markers] = y
        for j, marker in enumerate(markers):
            order = np.argsort(x[:, j], kind="stable")
            dx, dy = np.diff(x[order, j]), np.diff(y[order, j])
            inversions = int(np.sum((dx > 0) & (dy < 0)))
            if inversions:
                raise ValueError("Preprocessing reversed ranks")
            audit.append({"dataset": dataset, "marker": marker, "operator": spec["operator"],
                          "cells": len(x), "input_min": float(x[:, j].min()), "input_max": float(x[:, j].max()),
                          "prepared_min": float(y[:, j].min()), "prepared_max": float(y[:, j].max()),
                          "adjacent_rank_inversions": inversions, "new_ties": int(np.sum((dx > 0) & (dy == 0)))})
    return result, audit


def prepare(case, output, design):
    import pandas as pd
    table, markers, sources = load_source(case, design["cases"][case])
    cfg = design["cases"][case]
    prepared, audit = transformed(table, markers, cfg)
    provenance = {"case": case, "design": design, "sources": sources,
                  "adapter_sha256": sha(__file__), "labels_read": False,
                  "scope": "benchmark_data-only adaptation; not identical to historical cyCombine preprocessing"}
    key = fingerprint(provenance)
    target = output / case / "prepared"
    if target.exists():
        prior = json.loads((target / "preprocessing.json").read_text())
        if prior["fingerprint"] != key:
            raise ValueError("Prepared inputs/configuration changed; choose a fresh --output directory")
        for rel, h in prior["outputs"].items():
            if sha(target / rel) != h:
                raise ValueError(f"Prepared file changed: {rel}")
        return target
    target.mkdir(parents=True)
    prepared.to_csv(target / "analysis_ready.csv.gz", index=False, float_format="%.17g")
    (target / "markers.txt").write_text("\n".join(markers) + "\n")
    table[["cell_id", "dataset", "unit_key"]].to_csv(target / "row_metadata.csv.gz", index=False)
    pd.DataFrame(audit).to_csv(target / "preprocessing_rank_audit.csv", index=False)
    write_json(target / "preprocessing.json", {"fingerprint": key, **provenance,
        "outputs": {p.name: sha(p) for p in target.iterdir() if p.is_file()}})
    return target


def verify_complete(path):
    record = json.loads((path / "COMPLETE.json").read_text())
    for rel, value in record["outputs"].items():
        if sha(path / rel) != value:
            raise ValueError(f"Completed output changed: {path / rel}")
    return record


def fit(case, output, design):
    prepared = prepare(case, output, design)
    root = output / case
    cfg = {k: v for k, v in design.items() if k != "cases"}
    cfg.update(input_sha256=sha(prepared / "analysis_ready.csv.gz"), markers_sha256=sha(prepared / "markers.txt"),
               preprocessing_file=str(prepared / "preprocessing.json"), preprocessing_sha256=sha(prepared / "preprocessing.json"),
               code_hashes={p.name: sha(p) for p in HERE.glob("*.R")},
               upstream_lock_sha256=sha(HERE / "upstream_lock.json"),
               runtime_lock_sha256=sha(HERE / ".Rlib/runtime_lock.csv"))
    config_path = root / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != cfg:
        raise ValueError("Fit inputs/code/runtime changed; use a fresh --output directory")
    if not config_path.exists():
        write_json(config_path, cfg)
    dest = root / "alignment"
    if dest.exists():
        verify_complete(dest)
        print(case, "alignment already complete and verified", flush=True)
        return
    logdir = output / "logs"
    logdir.mkdir(exist_ok=True)
    # R rejects a temporary directory whose path contains spaces. Only this
    # temporary link is outside algorithm_test; the actual files remain inside.
    link = Path("/private/tmp" if Path("/private/tmp").exists() else "/tmp") / f"cytonorm-r-{os.getpid()}"
    link.symlink_to((output / "tmp").resolve(), target_is_directory=True)
    try:
        env = dict(os.environ, TMPDIR=str(link))
        with (logdir / f"{case}_fit.log").open("w") as log:
            subprocess.run(["Rscript", "--vanilla", str(HERE / "run_cytonorm.R"),
                            "--input", str(prepared / "analysis_ready.csv.gz"), "--markers", str(prepared / "markers.txt"),
                            "--config", str(config_path), "--output", str(dest)],
                           check=True, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=root)
    finally:
        link.unlink()
    verify_complete(dest)
    for r in json.loads((prepared / "preprocessing.json").read_text())["sources"]:
        if sha(r["path"]) != r["sha256"]:
            raise ValueError("Benchmark input changed during fitting")
    print(case, "FIT COMPLETE", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["prepare", "fit", "evaluate", "all"])
    p.add_argument("--case", choices=["all", "bone_marrow", "tnbc", "melanoma", "lung_cancer"], default="all")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--resamples", type=int, choices=[1, 20], default=20,
                   help="1 is an integration check, 20 is the frozen full metric protocol")
    args = p.parse_args()
    output = configure(args.output)
    design = json.loads((HERE / "config.json").read_text())
    cases = list(design["cases"]) if args.case == "all" else [args.case]
    for case in cases:
        if args.stage == "prepare":
            prepare(case, output, design)
            print(case, "PREPARED", flush=True)
        elif args.stage in ["fit", "all"]:
            fit(case, output, design)
    if args.stage in ["evaluate", "all"]:
        # All four fits must be frozen before any post-hoc label is opened.
        for case in design["cases"]:
            verify_complete(output / case / "alignment")
        from evaluate import evaluate
        for case in cases:
            evaluate(case, output, design, args.resamples)


if __name__ == "__main__":
    main()
