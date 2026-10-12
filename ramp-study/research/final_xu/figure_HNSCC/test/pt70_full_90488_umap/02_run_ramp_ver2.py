#!/usr/bin/env python3
"""Run canonical RAMP 2.3.0 on the frozen PT70 test input."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
FINAL_XU = HERE.parents[2]
INPUT = HERE / "input"
OUT = HERE / "alignment"
CACHE = HERE / ".cache"

for variable, child in (
    ("MPLCONFIGDIR", "matplotlib"),
    ("NUMBA_CACHE_DIR", "numba"),
    ("XDG_CACHE_HOME", "xdg"),
    ("JOBLIB_TEMP_FOLDER", "joblib"),
    ("TMPDIR", "tmp"),
):
    path = CACHE / child
    path.mkdir(parents=True, exist_ok=True)
    os.environ[variable] = str(path)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd

sys.path.insert(0, str(FINAL_XU / "algorithm"))
from ramp import __version__ as RAMP_VERSION  # noqa: E402
from ramp import fit_transform  # noqa: E402


DATASETS = ["GSE212797_ADT", "Haist_CODEX"]
FAMILY = "hnscc_pt70_primary_tumor_full_umap_test"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = np.load(INPUT / "raw_shared_markers.float64.npy", allow_pickle=False)
    metadata = pd.read_csv(
        INPUT / "cell_metadata.csv.gz",
        dtype={"cell_uid": str, "dataset": str, "donor": str, "unit_key": str},
        low_memory=False,
    )
    markers = [line.strip() for line in (INPUT / "markers.txt").read_text().splitlines() if line.strip()]
    if RAMP_VERSION != "2.3.0":
        raise AssertionError(f"Expected canonical RAMP 2.3.0, observed {RAMP_VERSION}")
    if raw.shape != (len(metadata), len(markers)):
        raise AssertionError("Raw matrix, metadata, and marker dimensions disagree")

    result = fit_transform(
        raw,
        metadata[["dataset", "unit_key"]],
        markers,
        datasets=DATASETS,
        family=FAMILY,
    )
    if result.strict_rank_inversions != 0:
        raise AssertionError("RAMP rank-preservation audit failed")

    baseline_path = OUT / "baseline_kmeanslow.float32.npy"
    aligned_path = OUT / "aligned_ramp_ver2.float32.npy"
    np.save(baseline_path, result.baseline.astype(np.float32))
    np.save(aligned_path, result.aligned.astype(np.float32))
    result.coupling_audit.to_csv(OUT / "coupling_audit.csv", index=False)
    result.strength_audit.to_csv(OUT / "strength_audit.csv", index=False)
    result.evidence_audit.to_csv(OUT / "evidence_audit.csv", index=False)
    result.reference_weight_audit.to_csv(OUT / "reference_weight_audit.csv", index=False)
    result.profile_audit.to_csv(OUT / "profile_audit.csv", index=False)
    result.algorithm_audit.to_csv(OUT / "algorithm_audit.csv", index=False)

    manifest = {
        "algorithm": "RAMP",
        "canonical_version": RAMP_VERSION,
        "implementation": str(FINAL_XU / "algorithm/ramp/ramp_ver2.py"),
        "family": FAMILY,
        "datasets": DATASETS,
        "cells": int(len(metadata)),
        "dataset_cells": metadata["dataset"].value_counts(sort=False).astype(int).to_dict(),
        "units": metadata.groupby("dataset", sort=False)["unit_key"].nunique().astype(int).to_dict(),
        "shared_markers": markers,
        "fit_metadata_columns": ["dataset", "unit_key"],
        "labels_available_to_fit": False,
        "coordinates_available_to_fit": False,
        "outcomes_available_to_fit": False,
        "strict_rank_inversions": int(result.strict_rank_inversions),
        "sha256": {
            "baseline": sha256(baseline_path),
            "aligned": sha256(aligned_path),
        },
    }
    (OUT / "alignment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
