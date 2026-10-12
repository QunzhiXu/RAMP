#!/usr/bin/env python3
"""Run canonical RAMP 2.3.0 on the frozen 88,031-cell BM benchmark."""

from __future__ import annotations

import gzip
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from paths import new_output


STUDY_ROOT = Path(__file__).resolve().parents[3]
ALGORITHM_RUNNER = STUDY_ROOT / "vendor" / "ramp" / "run_ramp.py"
HERE = SOURCE = INPUT_DIR = DATA_DIR = None

DATASETS = ["BM CODEX", "BM CyTOF", "Triana AbSeq"]
MARKERS = [
    "CD117",
    "CD11B",
    "CD11C",
    "CD123",
    "CD14",
    "CD3",
    "CD33",
    "CD34",
    "CD38",
    "CD4",
    "CD45",
    "CD45RA",
    "CD61",
    "CD8",
]
FAMILY = "bm_prediction_story_observed14"
EXPECTED_DATASET_CELLS = {
    "BM CODEX": 30_000,
    "BM CyTOF": 30_000,
    "Triana AbSeq": 28_031,
}
EXPECTED_TOTAL = 88_031


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv_gz(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(filename=str(path), mode="wb", mtime=0) as stream:
        frame.to_csv(stream, index=False)


def main() -> None:
    global HERE, SOURCE, INPUT_DIR, DATA_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relabel-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    relabel_root = args.relabel_root.resolve(strict=True)
    SOURCE = relabel_root / "data" / "bone_marrow_88031_relabeled.csv.gz"
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    HERE = new_output(args.output_dir, [relabel_root, ALGORITHM_RUNNER.parent])
    INPUT_DIR, DATA_DIR = HERE / "input", HERE / "data"
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    source = pd.read_csv(SOURCE, low_memory=False)
    required = ["cell_id", "dataset", "donor", "unit_key", "celltype", *MARKERS]
    missing = [column for column in required if column not in source]
    if missing:
        raise ValueError(f"Missing benchmark columns: {missing}")
    if len(source) != EXPECTED_TOTAL:
        raise AssertionError((len(source), EXPECTED_TOTAL))
    observed_order = list(pd.unique(source["dataset"].astype(str)))
    if observed_order != DATASETS:
        raise AssertionError(f"Dataset order changed: {observed_order}")
    observed_counts = source.groupby("dataset", sort=False).size().to_dict()
    if observed_counts != EXPECTED_DATASET_CELLS:
        raise AssertionError(f"Dataset counts changed: {observed_counts}")
    if source["cell_id"].astype(str).duplicated().any():
        raise AssertionError("Duplicate cell IDs")

    # Preserve the highest-available source values at float64 precision. The
    # canonical RAMP runner controls its own output dtype.
    raw = source[MARKERS].to_numpy(np.float64)
    if raw.shape != (EXPECTED_TOTAL, len(MARKERS)) or not np.isfinite(raw).all():
        raise AssertionError("Invalid source-scale marker matrix")

    # RAMP receives only its two required metadata fields. Provider labels are
    # kept in a separate post-hoc file and cannot leak into fitting.
    fit_metadata = source[["dataset", "unit_key"]].copy()
    display_columns = [
        "cell_id",
        "dataset",
        "donor",
        "unit_key",
        "provider_label_l1",
        "provider_label_l2",
        "provider_label_coarse",
        "celltype",
        "celltype_relabel_version",
    ]
    display_metadata = source[display_columns].copy()

    raw_path = INPUT_DIR / "raw.npy"
    fit_metadata_path = INPUT_DIR / "fit_metadata.csv.gz"
    display_metadata_path = INPUT_DIR / "cell_metadata.csv.gz"
    markers_path = INPUT_DIR / "markers.txt"
    np.save(raw_path, raw)
    write_csv_gz(fit_metadata, fit_metadata_path)
    write_csv_gz(display_metadata, display_metadata_path)
    markers_path.write_text("\n".join(MARKERS) + "\n")

    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.setdefault("LOKY_MAX_CPU_COUNT", "1")
    command = [
        sys.executable,
        str(ALGORITHM_RUNNER),
        "--raw",
        str(raw_path),
        "--metadata",
        str(fit_metadata_path),
        "--markers",
        str(markers_path),
        "--datasets",
        *DATASETS,
        "--family",
        FAMILY,
        "--output",
        str(DATA_DIR),
    ]
    print("Running canonical RAMP 2.3.0 on 88,031 cells x 14 markers", flush=True)
    subprocess.run(command, check=True, cwd=HERE, env=environment)

    aligned = np.load(DATA_DIR / "ramp_aligned.npy")
    baseline = np.load(DATA_DIR / "kmeanslow_global.npy")
    if aligned.shape != raw.shape or baseline.shape != raw.shape:
        raise AssertionError("RAMP output shape mismatch")
    if not np.isfinite(aligned).all() or not np.isfinite(baseline).all():
        raise AssertionError("Non-finite RAMP output")

    ramp_manifest_path = DATA_DIR / "ramp_manifest.json"
    ramp_manifest = json.loads(ramp_manifest_path.read_text())
    if ramp_manifest["algorithm_version"] != "2.3.0":
        raise AssertionError(ramp_manifest["algorithm_version"])
    if ramp_manifest["strict_rank_inversions"] != 0:
        raise AssertionError("RAMP reported strict rank inversions")
    if ramp_manifest["metadata_columns_read_for_fitting"] != ["dataset", "unit_key"]:
        raise AssertionError("Unexpected fitting metadata")

    aligned_frame = display_metadata.copy()
    baseline_frame = display_metadata.copy()
    for index, marker in enumerate(MARKERS):
        aligned_frame[marker] = aligned[:, index]
        baseline_frame[marker] = baseline[:, index]
    aligned_csv = DATA_DIR / "ramp_aligned.csv.gz"
    baseline_csv = DATA_DIR / "kmeanslow_global.csv.gz"
    write_csv_gz(aligned_frame, aligned_csv)
    write_csv_gz(baseline_frame, baseline_csv)

    check = pd.read_csv(aligned_csv, low_memory=False)
    if len(check) != EXPECTED_TOTAL:
        raise AssertionError("Aligned CSV row count changed")
    if not np.array_equal(check["cell_id"].astype(str), source["cell_id"].astype(str)):
        raise AssertionError("Aligned CSV cell order changed")
    if not np.allclose(
        check[MARKERS].to_numpy(np.float32), aligned, rtol=0.0, atol=0.0
    ):
        raise AssertionError("Aligned CSV values differ from aligned NPY")

    manifest = {
        "analysis": "bone_marrow_ramp_alignment",
        "cells": EXPECTED_TOTAL,
        "markers": MARKERS,
        "datasets": DATASETS,
        "dataset_cells": EXPECTED_DATASET_CELLS,
        "family": FAMILY,
        "source_scale_input": True,
        "external_preprocessing_before_ramp": False,
        "ramp_internal_preprocessing": "KMEANSLOW",
        "labels_used_for_fitting": False,
        "canonical_algorithm_version": ramp_manifest["algorithm_version"],
        "canonical_algorithm_source": ramp_manifest["algorithm_source"],
        "strict_rank_inversions": ramp_manifest["strict_rank_inversions"],
        "source": {"path": str(SOURCE), "sha256": sha256(SOURCE)},
        "runner": {"path": str(ALGORITHM_RUNNER), "sha256": sha256(ALGORITHM_RUNNER)},
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
        "inputs": {
            "raw.npy": sha256(raw_path),
            "fit_metadata.csv.gz": sha256(fit_metadata_path),
            "cell_metadata.csv.gz": sha256(display_metadata_path),
            "markers.txt": sha256(markers_path),
        },
        "outputs": {
            "ramp_aligned.npy": sha256(DATA_DIR / "ramp_aligned.npy"),
            "ramp_aligned.csv.gz": sha256(aligned_csv),
            "kmeanslow_global.npy": sha256(DATA_DIR / "kmeanslow_global.npy"),
            "kmeanslow_global.csv.gz": sha256(baseline_csv),
            "ramp_manifest.json": sha256(ramp_manifest_path),
        },
    }
    (HERE / "alignment_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"Saved aligned data to {aligned_csv}", flush=True)


if __name__ == "__main__":
    main()
