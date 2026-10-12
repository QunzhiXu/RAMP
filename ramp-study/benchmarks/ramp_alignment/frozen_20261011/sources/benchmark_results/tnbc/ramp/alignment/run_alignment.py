#!/usr/bin/env python3
"""Run canonical RAMP 2.3.0 on the frozen TNBC benchmark v2."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
FINAL_XU = HERE.parents[3]
BENCHMARK_ROOT = FINAL_XU / "benchmark_data" / "tnbc"
RAW_SOURCE = BENCHMARK_ROOT / "input" / "raw.npy"
FIT_METADATA_SOURCE = BENCHMARK_ROOT / "input" / "fit_metadata.csv.gz"
LOCKED_METADATA_SOURCE = BENCHMARK_ROOT / "input" / "locked_metadata.csv.gz"
RELABEL_METADATA_SOURCE = BENCHMARK_ROOT / "relabel" / "relabeled_metadata.csv.gz"
MARKERS_SOURCE = BENCHMARK_ROOT / "input" / "markers.txt"
BENCHMARK_MANIFEST_SOURCE = BENCHMARK_ROOT / "manifest.json"
ALGORITHM_RUNNER = FINAL_XU / "algorithm" / "ramp" / "run_ramp.py"
INPUT_DIR = HERE / "input"
DATA_DIR = HERE / "data"

DATASETS = ["Tietscher IMC", "Meyer IMC", "Shiao CODEX"]
MARKERS = [
    "CD11C", "CD20", "CD3", "CD31", "CD4", "CD68", "CD8",
    "HLA_DR", "KI67", "PANCK", "SMA",
]
EXPECTED_CELLS = {
    "Tietscher IMC": 20_000,
    "Meyer IMC": 22_701,
    "Shiao CODEX": 20_000,
}
EXPECTED_UNITS = {
    "Tietscher IMC": 1,
    "Meyer IMC": 5,
    "Shiao CODEX": 5,
}
EXPECTED_TOTAL = 62_701
FAMILY = (
    "tnbc_tietscher1_meyer5_coverage_guided_seed20260960_"
    "shiao5_uniform4k_seed20260827"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv_gz(
    frame: pd.DataFrame,
    path: Path,
    *,
    float_format: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="", fileobj=raw_handle, mode="wb", compresslevel=6, mtime=0
        ) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                frame.to_csv(text, index=False, float_format=float_format)


def audit_dictionary(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    return dict(zip(frame["audit_key"], frame["audit_value"]))


def main() -> None:
    for path in (
        RAW_SOURCE,
        FIT_METADATA_SOURCE,
        LOCKED_METADATA_SOURCE,
        RELABEL_METADATA_SOURCE,
        MARKERS_SOURCE,
        BENCHMARK_MANIFEST_SOURCE,
        ALGORITHM_RUNNER,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    raw_source = np.load(RAW_SOURCE, allow_pickle=False)
    fit_source = pd.read_csv(FIT_METADATA_SOURCE, low_memory=False)
    benchmark_manifest = json.loads(BENCHMARK_MANIFEST_SOURCE.read_text())
    if benchmark_manifest.get("package") != "tnbc_three_platform_benchmark_v2":
        raise AssertionError("Expected TNBC benchmark v2")
    if benchmark_manifest.get("cells") != EXPECTED_TOTAL:
        raise AssertionError("TNBC benchmark manifest cell count changed")
    meyer_sampling = benchmark_manifest.get("meyer_sampling", {})
    if meyer_sampling.get("seed") != 20260960:
        raise AssertionError("Expected coverage-informed Meyer seed 20260960")
    if meyer_sampling.get("selection_mode") != "coverage_guided":
        raise AssertionError("Expected coverage-informed Meyer selection")
    markers = [
        value.strip() for value in MARKERS_SOURCE.read_text().splitlines()
        if value.strip()
    ]
    if markers != MARKERS or raw_source.shape != (EXPECTED_TOTAL, len(MARKERS)):
        raise AssertionError("Canonical TNBC matrix/marker contract changed")
    if fit_source.columns.tolist() != ["dataset", "unit_key"]:
        raise AssertionError(f"Fit metadata leaked extra columns: {fit_source.columns.tolist()}")
    if len(fit_source) != EXPECTED_TOTAL:
        raise AssertionError("TNBC fit-metadata row count changed")
    if list(pd.unique(fit_source["dataset"].astype(str))) != DATASETS:
        raise AssertionError("Dataset order changed")
    if fit_source.groupby("dataset", sort=False).size().to_dict() != EXPECTED_CELLS:
        raise AssertionError("Dataset cell counts changed")
    observed_units = (
        fit_source.groupby("dataset", sort=False)["unit_key"].nunique().to_dict()
    )
    if observed_units != EXPECTED_UNITS:
        raise AssertionError(f"Dataset unit counts changed: {observed_units}")
    if not np.isfinite(raw_source).all() or np.any(raw_source < 0):
        raise AssertionError("Invalid source-scale TNBC matrix")

    raw_path = INPUT_DIR / "raw.npy"
    fit_path = INPUT_DIR / "fit_metadata.csv.gz"
    display_path = INPUT_DIR / "cell_metadata.csv.gz"
    markers_path = INPUT_DIR / "markers.txt"
    shutil.copyfile(RAW_SOURCE, raw_path)
    shutil.copyfile(FIT_METADATA_SOURCE, fit_path)
    shutil.copyfile(MARKERS_SOURCE, markers_path)

    environment = os.environ.copy()
    command = [
        sys.executable,
        str(ALGORITHM_RUNNER),
        "--raw", str(raw_path),
        "--metadata", str(fit_path),
        "--markers", str(markers_path),
        "--datasets", *DATASETS,
        "--family", FAMILY,
        "--output", str(DATA_DIR),
    ]
    print(
        f"Running canonical RAMP 2.3.0 on {EXPECTED_TOTAL:,} cells x {len(MARKERS)} markers",
        flush=True,
    )
    subprocess.run(command, check=True, cwd=HERE, env=environment)

    aligned = np.load(DATA_DIR / "ramp_aligned.npy", allow_pickle=False)
    baseline = np.load(DATA_DIR / "kmeanslow_global.npy", allow_pickle=False)
    if aligned.shape != raw_source.shape or baseline.shape != raw_source.shape:
        raise AssertionError("RAMP output shape changed")
    if not np.isfinite(aligned).all() or not np.isfinite(baseline).all():
        raise AssertionError("RAMP produced non-finite output")

    ramp_manifest_path = DATA_DIR / "ramp_manifest.json"
    ramp_manifest = json.loads(ramp_manifest_path.read_text())
    if ramp_manifest["algorithm_version"] != "2.3.0":
        raise AssertionError(ramp_manifest["algorithm_version"])
    if ramp_manifest["strict_rank_inversions"] != 0:
        raise AssertionError("RAMP reported a strict rank inversion")
    if ramp_manifest["metadata_columns_read_for_fitting"] != ["dataset", "unit_key"]:
        raise AssertionError("Unexpected fitting metadata")
    if ramp_manifest.get("celltype_labels_read_for_fitting") is not False:
        raise AssertionError("Cell-type labels were exposed to RAMP fitting")
    ramp_inputs = ramp_manifest.get("inputs", {})
    if Path(ramp_inputs.get("metadata", "")).resolve() != fit_path.resolve():
        raise AssertionError("RAMP manifest points to unexpected fitting metadata")
    if ramp_inputs.get("metadata_sha256") != sha256(fit_path):
        raise AssertionError("RAMP fitting-metadata hash changed")
    algorithm_audit = audit_dictionary(DATA_DIR / "algorithm_audit.csv")
    if algorithm_audit.get("mapping_operator_calibration") != "bounded_profile_mapping":
        raise AssertionError("Calibration did not use bounded mapping")
    if algorithm_audit.get("mapping_operator_final") != "bounded_profile_mapping":
        raise AssertionError("Final application did not use bounded mapping")
    if int(algorithm_audit.get("cell_marker_values_over_maximum", "-1")) != 0:
        raise AssertionError("RAMP exceeded the maximum displacement")
    if int(algorithm_audit.get("graph_k_nonself", "-1")) != 15:
        raise AssertionError("RAMP KNN contract changed")

    # Preserve the canonical in-memory precision for deterministic downstream
    # UMAP. The generic CLI stores compact float32 matrices; UMAP is sensitive
    # enough that this harmless round-off can alter the embedding. This second
    # label-free call uses the same public API and must agree with the compact
    # CLI outputs after float32 casting.
    algorithm_root = FINAL_XU / "algorithm"
    if str(algorithm_root) not in sys.path:
        sys.path.insert(0, str(algorithm_root))
    from ramp import fit_transform as canonical_fit_transform

    full_precision = canonical_fit_transform(
        raw_source,
        fit_source,
        MARKERS,
        datasets=DATASETS,
        family=FAMILY,
    )
    aligned_float64 = np.asarray(full_precision.aligned, dtype=np.float64)
    baseline_float64 = np.asarray(full_precision.baseline, dtype=np.float64)
    if full_precision.strict_rank_inversions != 0:
        raise AssertionError("Full-precision RAMP reported a strict rank inversion")
    if not np.array_equal(aligned_float64.astype(np.float32), aligned):
        raise AssertionError("Full-precision/compact aligned matrices disagree")
    if not np.array_equal(baseline_float64.astype(np.float32), baseline):
        raise AssertionError("Full-precision/compact baseline matrices disagree")
    aligned_float64_path = DATA_DIR / "ramp_aligned_float64.npy"
    baseline_float64_path = DATA_DIR / "kmeanslow_global_float64.npy"
    np.save(aligned_float64_path, aligned_float64)
    np.save(baseline_float64_path, baseline_float64)

    # Display labels are deliberately opened only after RAMP fitting has ended.
    locked = pd.read_csv(LOCKED_METADATA_SOURCE, low_memory=False)
    labels = pd.read_csv(RELABEL_METADATA_SOURCE, low_memory=False)
    if len(locked) != EXPECTED_TOTAL or len(labels) != EXPECTED_TOTAL:
        raise AssertionError("TNBC display-metadata row count changed")
    if not fit_source["unit_key"].astype(str).equals(locked["unit_key"].astype(str)):
        raise AssertionError("Fit metadata row order differs from locked metadata")
    for column in ("cell_id", "source_cell_id", "dataset", "donor", "unit_key"):
        if not locked[column].astype(str).equals(labels[column].astype(str)):
            raise AssertionError(f"Relabel metadata {column} order changed")
    if labels["cell_id"].astype(str).duplicated().any():
        raise AssertionError("Duplicate TNBC cell IDs")
    if labels["celltype"].isin(
        ["SMA-high CAF", "SMA-low Stroma", "Stromal / immune mix"]
    ).any():
        raise AssertionError("Exploratory SMA-derived label entered the benchmark")
    display = labels[
        [
            "cell_id", "source_cell_id", "dataset", "donor", "unit_key",
            "provider_label_l1", "provider_label_l2", "celltype",
            "celltype_relabel_version",
        ]
    ].copy()
    write_csv_gz(display, display_path)

    aligned_frame = display.copy()
    baseline_frame = display.copy()
    for index, marker in enumerate(MARKERS):
        aligned_frame[marker] = aligned[:, index]
        baseline_frame[marker] = baseline[:, index]
    aligned_csv = DATA_DIR / "ramp_aligned.csv.gz"
    baseline_csv = DATA_DIR / "kmeanslow_global.csv.gz"
    write_csv_gz(aligned_frame, aligned_csv, float_format="%.9g")
    write_csv_gz(baseline_frame, baseline_csv, float_format="%.9g")
    check = pd.read_csv(aligned_csv, low_memory=False, float_precision="round_trip")
    if not np.array_equal(check["cell_id"].astype(str), display["cell_id"].astype(str)):
        raise AssertionError("Aligned CSV cell order changed")
    if not np.array_equal(check[MARKERS].to_numpy(np.float32), aligned):
        raise AssertionError("Aligned CSV values differ from aligned NPY")

    manifest = {
        "analysis": "tnbc_ramp_alignment",
        "cells": EXPECTED_TOTAL,
        "markers": MARKERS,
        "datasets": DATASETS,
        "dataset_cells": EXPECTED_CELLS,
        "dataset_units": EXPECTED_UNITS,
        "family": FAMILY,
        "source_scale_input": True,
        "external_preprocessing_before_ramp": False,
        "ramp_internal_preprocessing": "KMEANSLOW",
        "labels_used_for_fitting": False,
        "canonical_algorithm_version": ramp_manifest["algorithm_version"],
        "canonical_algorithm_source": ramp_manifest["algorithm_source"],
        "strict_rank_inversions": ramp_manifest["strict_rank_inversions"],
        "sources": {
            "raw": {"path": str(RAW_SOURCE), "sha256": sha256(RAW_SOURCE)},
            "fit_metadata": {"path": str(FIT_METADATA_SOURCE), "sha256": sha256(FIT_METADATA_SOURCE)},
            "relabel_metadata": {"path": str(RELABEL_METADATA_SOURCE), "sha256": sha256(RELABEL_METADATA_SOURCE)},
            "markers": {"path": str(MARKERS_SOURCE), "sha256": sha256(MARKERS_SOURCE)},
            "benchmark_manifest": {
                "path": str(BENCHMARK_MANIFEST_SOURCE),
                "sha256": sha256(BENCHMARK_MANIFEST_SOURCE),
            },
        },
        "runner": {"path": str(ALGORITHM_RUNNER), "sha256": sha256(ALGORITHM_RUNNER)},
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
        "inputs": {
            "raw.npy": sha256(raw_path),
            "fit_metadata.csv.gz": sha256(fit_path),
            "cell_metadata.csv.gz": sha256(display_path),
            "markers.txt": sha256(markers_path),
        },
        "outputs": {
            "ramp_aligned.npy": sha256(DATA_DIR / "ramp_aligned.npy"),
            "ramp_aligned.csv.gz": sha256(aligned_csv),
            "kmeanslow_global.npy": sha256(DATA_DIR / "kmeanslow_global.npy"),
            "ramp_aligned_float64.npy": sha256(aligned_float64_path),
            "kmeanslow_global_float64.npy": sha256(baseline_float64_path),
            "kmeanslow_global.csv.gz": sha256(baseline_csv),
            "ramp_manifest.json": sha256(ramp_manifest_path),
            "algorithm_audit.csv": sha256(DATA_DIR / "algorithm_audit.csv"),
        },
    }
    (HERE / "alignment_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"Saved aligned data to {aligned_csv}", flush=True)


if __name__ == "__main__":
    main()
