#!/usr/bin/env python3
"""Run canonical RAMP 2.3.0 on the frozen 60,000-cell melanoma benchmark."""

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
BENCHMARK_ROOT = FINAL_XU / "benchmark_data" / "melanoma"
RAW_SOURCE = BENCHMARK_ROOT / "input" / "raw.npy"
FIT_METADATA_SOURCE = BENCHMARK_ROOT / "input" / "fit_metadata.csv.gz"
LOCKED_METADATA_SOURCE = BENCHMARK_ROOT / "input" / "locked_metadata.csv.gz"
RELABEL_METADATA_SOURCE = BENCHMARK_ROOT / "relabel" / "relabeled_metadata.csv.gz"
MARKERS_SOURCE = BENCHMARK_ROOT / "input" / "markers.txt"
BENCHMARK_MANIFEST_SOURCE = BENCHMARK_ROOT / "manifest.json"
RELABEL_MANIFEST_SOURCE = BENCHMARK_ROOT / "relabel" / "manifest.json"
ALGORITHM_RUNNER = FINAL_XU / "algorithm" / "ramp" / "run_ramp.py"
INPUT_DIR = HERE / "input"
DATA_DIR = HERE / "data"

DATASETS = [
    "MEL_Hickey2023_CODEX",
    "MEL_Hoch2021_IMC",
    "MEL_Ptacek2021_MIBI",
]
MARKERS = [
    "CD163", "CD3", "CD31", "CD68", "CD8", "HLA_DR",
    "LAG3", "PD1", "PDL1", "SMA", "SOX10", "VIMENTIN",
]
EXPECTED_CELLS = {dataset: 20_000 for dataset in DATASETS}
EXPECTED_UNITS = {
    DATASETS[0]: 6,
    DATASETS[1]: 17,
    DATASETS[2]: 23,
}
EXPECTED_TOTAL = 60_000
FAMILY = "melanoma_metadata_stage3_20k_each_20260827"
EXPECTED_ALIGNED_SHA256 = "c5af27981cc3b07e47065c0fb0b419a2f1947b05902a3928470bd8fc0a403606"
EXPECTED_BASELINE_SHA256 = "32e9e6441c6c051bb16889c2d2cded59d364ce9c2a0cfad7df0c0e7af382dbb0"
ACCEPTED_TEST_ROOT = FINAL_XU / "test" / "melanoma_metadata_matched_20k_20260827"


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


def assert_dataset_blocks(frame: pd.DataFrame) -> None:
    observed = frame["dataset"].astype(str).to_numpy()
    expected = np.concatenate(
        [np.repeat(dataset, EXPECTED_CELLS[dataset]) for dataset in DATASETS]
    )
    if not np.array_equal(observed, expected):
        raise AssertionError("Dataset rows are not in the frozen contiguous order")


def main() -> None:
    for path in (
        RAW_SOURCE,
        FIT_METADATA_SOURCE,
        LOCKED_METADATA_SOURCE,
        RELABEL_METADATA_SOURCE,
        MARKERS_SOURCE,
        BENCHMARK_MANIFEST_SOURCE,
        RELABEL_MANIFEST_SOURCE,
        ALGORITHM_RUNNER,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    benchmark_manifest = json.loads(BENCHMARK_MANIFEST_SOURCE.read_text())
    if benchmark_manifest.get("package") != "melanoma_three_platform_benchmark_v2_60000":
        raise AssertionError("Unexpected melanoma benchmark package")
    for relative, expected in benchmark_manifest["files_sha256"].items():
        if sha256(BENCHMARK_ROOT / relative) != expected:
            raise AssertionError(f"Benchmark source hash changed: {relative}")

    raw_source = np.load(RAW_SOURCE, allow_pickle=False)
    fit_source = pd.read_csv(FIT_METADATA_SOURCE, low_memory=False)
    markers = [
        value.strip() for value in MARKERS_SOURCE.read_text().splitlines()
        if value.strip()
    ]
    if raw_source.shape != (EXPECTED_TOTAL, len(MARKERS)) or raw_source.dtype != np.float32:
        raise AssertionError("Canonical melanoma matrix shape or dtype changed")
    if markers != MARKERS or not np.isfinite(raw_source).all():
        raise AssertionError("Canonical melanoma marker contract changed")
    if fit_source.columns.tolist() != ["dataset", "unit_key"]:
        raise AssertionError(f"Fit metadata leaked extra columns: {fit_source.columns.tolist()}")
    if len(fit_source) != EXPECTED_TOTAL:
        raise AssertionError("Fit-metadata row count changed")
    assert_dataset_blocks(fit_source)
    if fit_source.groupby("dataset", sort=False).size().to_dict() != EXPECTED_CELLS:
        raise AssertionError("Dataset cell counts changed")
    observed_units = (
        fit_source.groupby("dataset", sort=False)["unit_key"].nunique().to_dict()
    )
    if observed_units != EXPECTED_UNITS:
        raise AssertionError(f"Dataset unit counts changed: {observed_units}")

    raw_path = INPUT_DIR / "raw.npy"
    fit_path = INPUT_DIR / "fit_metadata.csv.gz"
    markers_path = INPUT_DIR / "markers.txt"
    display_path = INPUT_DIR / "cell_metadata.csv.gz"
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
        f"Running canonical RAMP 2.3.0 on {EXPECTED_TOTAL:,} cells x "
        f"{len(MARKERS)} markers",
        flush=True,
    )
    subprocess.run(command, check=True, cwd=HERE, env=environment)

    aligned_path = DATA_DIR / "ramp_aligned.npy"
    baseline_path = DATA_DIR / "kmeanslow_global.npy"
    aligned = np.load(aligned_path, allow_pickle=False)
    baseline = np.load(baseline_path, allow_pickle=False)
    if aligned.shape != raw_source.shape or baseline.shape != raw_source.shape:
        raise AssertionError("RAMP output shape changed")
    if aligned.dtype != np.float32 or baseline.dtype != np.float32:
        raise AssertionError("Canonical compact RAMP output dtype changed")
    if not np.isfinite(aligned).all() or not np.isfinite(baseline).all():
        raise AssertionError("RAMP produced non-finite output")

    ramp_manifest_path = DATA_DIR / "ramp_manifest.json"
    ramp_manifest = json.loads(ramp_manifest_path.read_text())
    if ramp_manifest.get("algorithm_version") != "2.3.0":
        raise AssertionError(ramp_manifest.get("algorithm_version"))
    if ramp_manifest.get("strict_rank_inversions") != 0:
        raise AssertionError("RAMP reported a strict rank inversion")
    if ramp_manifest.get("metadata_columns_read_for_fitting") != ["dataset", "unit_key"]:
        raise AssertionError("Unexpected fitting metadata")
    if ramp_manifest.get("celltype_labels_read_for_fitting") is not False:
        raise AssertionError("Cell-type labels were exposed during RAMP fitting")
    ramp_inputs = ramp_manifest.get("inputs", {})
    if Path(ramp_inputs.get("metadata", "")).resolve() != fit_path.resolve():
        raise AssertionError("RAMP manifest points to unexpected fitting metadata")
    if ramp_inputs.get("metadata_sha256") != sha256(fit_path):
        raise AssertionError("RAMP fit-metadata hash changed")

    algorithm_audit = audit_dictionary(DATA_DIR / "algorithm_audit.csv")
    if algorithm_audit.get("mapping_operator_calibration") != "bounded_profile_mapping":
        raise AssertionError("Calibration did not use bounded mapping")
    if algorithm_audit.get("mapping_operator_final") != "bounded_profile_mapping":
        raise AssertionError("Final application did not use bounded mapping")
    if int(algorithm_audit.get("cell_marker_values_over_maximum", "-1")) != 0:
        raise AssertionError("RAMP exceeded the maximum displacement")
    if int(algorithm_audit.get("graph_k_nonself", "-1")) != 15:
        raise AssertionError("RAMP KNN contract changed")

    if sha256(aligned_path) != EXPECTED_ALIGNED_SHA256:
        raise AssertionError("Formal aligned NPY hash differs from accepted melanoma run")
    if sha256(baseline_path) != EXPECTED_BASELINE_SHA256:
        raise AssertionError("Formal KMEANSLOW NPY hash differs from accepted melanoma run")
    accepted_comparison: dict[str, dict[str, str | float]] = {}
    for name, accepted_path, formal in (
        ("aligned", ACCEPTED_TEST_ROOT / "alignment" / "ramp_aligned.npy", aligned),
        ("baseline", ACCEPTED_TEST_ROOT / "alignment" / "kmeanslow_global.npy", baseline),
    ):
        if accepted_path.is_file():
            accepted = np.load(accepted_path, allow_pickle=False)
            maximum_error = float(np.max(np.abs(accepted.astype(np.float64) - formal)))
            if maximum_error != 0.0:
                raise AssertionError(f"Formal {name} differs from accepted test: {maximum_error}")
            accepted_comparison[name] = {
                "path": str(accepted_path),
                "sha256": sha256(accepted_path),
                "maximum_absolute_error": maximum_error,
            }

    # Provider labels are opened only after the RAMP subprocess and all
    # algorithm-only checks have completed.
    locked = pd.read_csv(LOCKED_METADATA_SOURCE, low_memory=False)
    labels = pd.read_csv(RELABEL_METADATA_SOURCE, low_memory=False)
    if len(locked) != EXPECTED_TOTAL or len(labels) != EXPECTED_TOTAL:
        raise AssertionError("Display metadata row count changed")
    assert_dataset_blocks(locked)
    assert_dataset_blocks(labels)
    if not fit_source.astype(str).equals(locked[["dataset", "unit_key"]].astype(str)):
        raise AssertionError("Fit metadata differs from locked metadata row order")
    for column in ("row_id", "cell_id", "source_cell_id", "dataset", "donor", "unit_key"):
        if not locked[column].astype(str).equals(labels[column].astype(str)):
            raise AssertionError(f"Relabel metadata {column} order changed")
    if labels["cell_id"].astype(str).duplicated().any():
        raise AssertionError("Duplicate melanoma cell IDs")
    display_columns = [
        "row_id", "cell_id", "source_cell_id", "source_index_zero_based",
        "dataset", "technology", "donor", "patient_id", "specimen_id",
        "acquisition_id", "unit_key", "celltype", "celltype_joint_evaluation",
        "joint_evaluation", "is_nonjoint", "celltype_relabel_version",
    ]
    display = labels[display_columns].copy()
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
    for path, matrix in ((aligned_csv, aligned), (baseline_csv, baseline)):
        check = pd.read_csv(path, low_memory=False, float_precision="round_trip")
        if not np.array_equal(check["row_id"].to_numpy(np.int64), display["row_id"]):
            raise AssertionError(f"{path.name}: row order changed")
        if not np.array_equal(check[MARKERS].to_numpy(np.float32), matrix):
            raise AssertionError(f"{path.name}: marker values differ from NPY")

    outputs = {
        path.name: sha256(path)
        for path in sorted(DATA_DIR.iterdir())
        if path.is_file()
    }
    relabel_manifest = json.loads(RELABEL_MANIFEST_SOURCE.read_text())
    manifest = {
        "analysis": "melanoma_ramp_alignment",
        "cells": EXPECTED_TOTAL,
        "markers": MARKERS,
        "datasets": DATASETS,
        "dataset_cells": EXPECTED_CELLS,
        "dataset_units": EXPECTED_UNITS,
        "family": FAMILY,
        "input_precision": "frozen float32 source matrix",
        "source_scale_input": True,
        "external_preprocessing_before_ramp": False,
        "ramp_internal_preprocessing": "KMEANSLOW",
        "labels_used_for_fitting": False,
        "labels_loaded_after_fitting": True,
        "canonical_algorithm_version": ramp_manifest["algorithm_version"],
        "canonical_algorithm_source": ramp_manifest["algorithm_source"],
        "strict_rank_inversions": ramp_manifest["strict_rank_inversions"],
        "accepted_test_comparison": accepted_comparison,
        "sources": {
            "benchmark_manifest": {
                "path": str(BENCHMARK_MANIFEST_SOURCE),
                "sha256": sha256(BENCHMARK_MANIFEST_SOURCE),
            },
            "raw": {"path": str(RAW_SOURCE), "sha256": sha256(RAW_SOURCE)},
            "fit_metadata": {
                "path": str(FIT_METADATA_SOURCE),
                "sha256": sha256(FIT_METADATA_SOURCE),
            },
            "locked_metadata": {
                "path": str(LOCKED_METADATA_SOURCE),
                "sha256": sha256(LOCKED_METADATA_SOURCE),
            },
            "relabel_metadata": {
                "path": str(RELABEL_METADATA_SOURCE),
                "sha256": sha256(RELABEL_METADATA_SOURCE),
                "manifest_parent_sha256": relabel_manifest["source"]["parent_manifest_sha256"],
            },
            "markers": {"path": str(MARKERS_SOURCE), "sha256": sha256(MARKERS_SOURCE)},
        },
        "runner": {"path": str(ALGORITHM_RUNNER), "sha256": sha256(ALGORITHM_RUNNER)},
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
        "inputs": {
            "raw.npy": sha256(raw_path),
            "fit_metadata.csv.gz": sha256(fit_path),
            "markers.txt": sha256(markers_path),
            "cell_metadata.csv.gz": sha256(display_path),
        },
        "outputs": outputs,
    }
    (HERE / "alignment_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"Saved formal melanoma alignment under {HERE}", flush=True)


if __name__ == "__main__":
    main()
