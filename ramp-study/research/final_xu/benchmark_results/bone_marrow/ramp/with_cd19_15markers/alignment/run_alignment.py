#!/usr/bin/env python3
"""Run canonical RAMP 2.3.0 on BM with observed CD19 included."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
FINAL_XU = HERE.parents[4]
BENCHMARK_ROOT = FINAL_XU / "benchmark_data" / "bone_marrow"
RELABEL_METADATA = BENCHMARK_ROOT / "relabel" / "relabeled_metadata.csv.gz"
ALGORITHM_RUNNER = FINAL_XU / "algorithm" / "ramp" / "run_ramp.py"
INPUT_DIR = HERE / "input"
DATA_DIR = HERE / "data"

DATASET_FILES = [
    ("BM CODEX", "bm_codex"),
    ("BM CyTOF", "bm_cytof"),
    ("Triana AbSeq", "triana_abseq"),
]
DATASETS = [dataset for dataset, _ in DATASET_FILES]
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
    "CD19",
]
FAMILY = "bm_prediction_story_wrong_observed15"
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


def load_source() -> tuple[pd.DataFrame, dict[str, str]]:
    parts: list[pd.DataFrame] = []
    hashes: dict[str, str] = {}
    for expected_dataset, slug in DATASET_FILES:
        path = BENCHMARK_ROOT / slug / "source_input_15.csv.gz"
        frame = pd.read_csv(path, low_memory=False)
        required = ["cell_id", "dataset", "donor", "unit_key", *MARKERS]
        missing = [column for column in required if column not in frame]
        if missing:
            raise ValueError(f"{expected_dataset} missing columns: {missing}")
        if set(frame["dataset"].astype(str)) != {expected_dataset}:
            raise AssertionError(f"Unexpected dataset values in {path}")
        if len(frame) != EXPECTED_DATASET_CELLS[expected_dataset]:
            raise AssertionError(f"Unexpected cell count in {path}")
        parts.append(frame[required].copy())
        hashes[str(path)] = sha256(path)
    return pd.concat(parts, ignore_index=True), hashes


def main() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    source, source_hashes = load_source()
    labels = pd.read_csv(RELABEL_METADATA, low_memory=False)
    if len(source) != EXPECTED_TOTAL or len(labels) != EXPECTED_TOTAL:
        raise AssertionError("Frozen BM total changed")
    if not source["cell_id"].astype(str).equals(labels["cell_id"].astype(str)):
        raise AssertionError("Relabel metadata cell order differs from 15-marker input")
    for column in ("dataset", "donor", "unit_key"):
        if not source[column].astype(str).equals(labels[column].astype(str)):
            raise AssertionError(f"{column} differs between matrix and labels")
    if list(pd.unique(source["dataset"].astype(str))) != DATASETS:
        raise AssertionError("Dataset order changed")
    if source["cell_id"].astype(str).duplicated().any():
        raise AssertionError("Duplicate cell IDs")

    raw = source[MARKERS].to_numpy(np.float64)
    if raw.shape != (EXPECTED_TOTAL, len(MARKERS)) or not np.isfinite(raw).all():
        raise AssertionError("Invalid source-scale 15-marker matrix")

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
    display_metadata = labels[display_columns].copy()

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
    print(
        "Running canonical RAMP 2.3.0 on 88,031 cells x 15 observed markers "
        "(including failed CODEX CD19)",
        flush=True,
    )
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
    aligned_csv = DATA_DIR / "ramp_aligned_15markers_with_cd19.csv.gz"
    baseline_csv = DATA_DIR / "kmeanslow_global_15markers_with_cd19.csv.gz"
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
        "analysis": "bone_marrow_ramp_alignment_15_observed_markers_with_cd19",
        "cells": EXPECTED_TOTAL,
        "markers": MARKERS,
        "datasets": DATASETS,
        "dataset_cells": EXPECTED_DATASET_CELLS,
        "family": FAMILY,
        "source_scale_input": True,
        "external_preprocessing_before_ramp": False,
        "ramp_internal_preprocessing": "KMEANSLOW",
        "labels_used_for_fitting": False,
        "observed_cd19_included": True,
        "codex_cd19_status": "known failed assay channel; deliberately included for this separate comparison",
        "canonical_algorithm_version": ramp_manifest["algorithm_version"],
        "canonical_algorithm_source": ramp_manifest["algorithm_source"],
        "strict_rank_inversions": ramp_manifest["strict_rank_inversions"],
        "source_files_sha256": source_hashes,
        "relabel_metadata": {
            "path": str(RELABEL_METADATA),
            "sha256": sha256(RELABEL_METADATA),
        },
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
            aligned_csv.name: sha256(aligned_csv),
            "kmeanslow_global.npy": sha256(DATA_DIR / "kmeanslow_global.npy"),
            baseline_csv.name: sha256(baseline_csv),
            "ramp_manifest.json": sha256(ramp_manifest_path),
        },
    }
    (HERE / "alignment_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"Saved 15-marker aligned data to {aligned_csv}", flush=True)


if __name__ == "__main__":
    main()
