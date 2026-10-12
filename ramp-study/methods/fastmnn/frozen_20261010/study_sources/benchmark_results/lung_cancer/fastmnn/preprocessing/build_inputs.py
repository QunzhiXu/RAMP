#!/usr/bin/env python3
"""Publish the audited, pre-correction lung protein input for fastMNN.

The source cache was built directly from the three provider objects by the
audited lung preprocessing code.  It contains no cyCombine-corrected values;
we reuse the exact pre-correction bytes so comparator differences cannot be
caused by another CSV round-trip or by a newly chosen transform.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
METHOD = HERE.parent
FINAL_XU = HERE.parents[3]
SOURCE = FINAL_XU / "benchmark_results" / "lung_cancer" / "cycombine" / "preprocessing"
BENCHMARK = FINAL_XU / "benchmark_data" / "lung_cancer"
DATA = HERE / "data"
ALIGNMENT = METHOD / "alignment"

EXPECTED_SOURCE_MANIFEST_SHA256 = "3b526b6babb8feb40825770cfcbaba2a5e0c05a49f15d3d351c0e3b37afcf6c4"
EXPECTED_SOURCE_BUILDER_SHA256 = "739c72d1d1a383fb566cd03611bb19750c2d478380ff0344ccbe5782fd23530e"
EXPECTED_COMBINED_SHA256 = "68e346f90b286665797b3c194acc68be467ebc76e6a94ea1e95ff0ceb06368a1"
EXPECTED_SELECTION_SHA256 = "e8a4c1d1ec91916607ddc1501dd1ce3d05487a3baab79988c217028527653de1"
MARKERS = ["CD3", "CD4", "CD8", "CD31", "HLA_DR", "KI67", "VIMENTIN", "EPI"]
DATASETS = ["Senosain CyTOF", "Cords IMC", "Monkman CODEX"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def checked_copy(source: Path, target: Path, expected: str | None = None) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    observed = sha256(source)
    if expected is not None and observed != expected:
        raise RuntimeError(f"Source hash changed: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256(target) != observed:
            raise RuntimeError(f"Refusing to overwrite a different artifact: {target}")
        return
    shutil.copyfile(source, target)
    if sha256(target) != observed:
        raise RuntimeError(f"Copied bytes failed verification: {target}")


def algorithm_parameters() -> dict[str, object]:
    return {
        "seed": 42,
        "rng_kind": "Mersenne-Twister",
        "normal_kind": "Inversion",
        "sample_kind": "Rejection",
        "k": 20,
        "prop_k": None,
        "cos_norm": True,
        "ndist": 3,
        "requested_d": 50,
        "d_policy": "request 50; batchelor realizes min(50, n_markers - 1, n_cells - 1)",
        "deferred": True,
        "weights": None,
        "get_variance": True,
        "merge_order": None,
        "auto_merge": False,
        "min_batch_skip": 0,
        "subset_row": None,
        "correct_all": False,
        "pca_solver": "IrlbaParam",
        "neighbor_solver": "KmknnParam",
        "parallel_solver": "SerialParam",
    }


def main() -> None:
    source_manifest_path = SOURCE / "preprocessing_manifest.json"
    source_builder = SOURCE / "build_inputs.py"
    if sha256(source_manifest_path) != EXPECTED_SOURCE_MANIFEST_SHA256:
        raise RuntimeError("Audited source preprocessing manifest changed")
    if sha256(source_builder) != EXPECTED_SOURCE_BUILDER_SHA256:
        raise RuntimeError("Audited source preprocessing code changed")
    source_manifest = json.loads(source_manifest_path.read_text())
    if [row["dataset"] for row in source_manifest["datasets"]] != DATASETS:
        raise RuntimeError("Dataset order changed in audited source preprocessing")
    selection = BENCHMARK / "selected_cells.csv.gz"
    if sha256(selection) != EXPECTED_SELECTION_SHA256:
        raise RuntimeError("Frozen cell-selection manifest changed")

    combined_source = SOURCE / "data" / "combined_analysis_ready.csv.gz"
    checked_copy(combined_source, DATA / "combined_analysis_ready.csv.gz", EXPECTED_COMBINED_SHA256)
    for row in source_manifest["datasets"]:
        matrix_source = SOURCE / row["input_matrix_file"]
        checked_copy(matrix_source, HERE / row["input_matrix_file"], row["input_matrix_sha256"])
    checked_copy(SOURCE / "markers.txt", HERE / "markers.txt", source_manifest["correction_marker_sha256"])
    checked_copy(
        SOURCE / "marker_harmonization_manifest.csv",
        HERE / "marker_harmonization_manifest.csv",
        source_manifest["marker_harmonization_manifest_sha256"],
    )
    checked_copy(SOURCE / "input_scale_audit.csv", HERE / "input_scale_audit.csv")
    checked_copy(source_builder, HERE / "source_preprocessing_builder.py", EXPECTED_SOURCE_BUILDER_SHA256)

    combined = pd.read_csv(DATA / "combined_analysis_ready.csv.gz", low_memory=False)
    expected_columns = ["cell_id", "dataset", "donor", "unit_key", *MARKERS]
    if combined.columns.tolist() != expected_columns or len(combined) != 60000:
        raise RuntimeError("Analysis-ready lung matrix contract changed")
    if not combined["cell_id"].is_unique or combined[MARKERS].isna().any().any():
        raise RuntimeError("Analysis-ready lung matrix has invalid IDs or values")
    observed = combined.groupby("dataset", sort=False).size().to_dict()
    if observed != {dataset: 20000 for dataset in DATASETS}:
        raise RuntimeError(f"Dataset counts/order changed: {observed}")
    selected = pd.read_csv(selection, usecols=["cell_id", "dataset", "donor", "unit_key"], low_memory=False)
    if not combined[["cell_id", "dataset", "donor", "unit_key"]].astype(str).equals(
        selected[["cell_id", "dataset", "donor", "unit_key"]].astype(str)
    ):
        raise RuntimeError("Method input differs from the frozen cell/order lock")
    zero_l2 = combined[MARKERS].pow(2).sum(axis=1).eq(0)
    zero_l2_by_dataset = {
        dataset: int(zero_l2[combined["dataset"].eq(dataset)].sum())
        for dataset in DATASETS
    }

    script_hash = sha256(Path(__file__))
    dataset_entries = []
    for source_entry in source_manifest["datasets"]:
        entry = dict(source_entry)
        entry["analysis_ready_matrix_file"] = entry.pop("input_matrix_file")
        entry["analysis_ready_matrix_sha256"] = entry.pop("input_matrix_sha256")
        entry["analysis_ready_input_scale"] = entry.pop("cycombine_input_scale")
        entry["processing_code_file"] = "source_preprocessing_builder.py"
        entry["processing_code_sha256"] = EXPECTED_SOURCE_BUILDER_SHA256
        entry["universal_positive_median_arcsinh_not_used"] = True
        entry["additional_marker_wise_zscore_not_used"] = True
        entry["provider_analysis_layer_preserved"] = True
        dataset_entries.append(entry)

    marker_path = HERE / "markers.txt"
    harmonization_path = HERE / "marker_harmonization_manifest.csv"
    manifest = {
        "schema_version": "final_xu.fastmnn_preprocessing.v1",
        "combined_input_file": "data/combined_analysis_ready.csv.gz",
        "combined_input_sha256": sha256(DATA / "combined_analysis_ready.csv.gz"),
        "shared_marker_policy": "Exact ordered eight-protein intersection frozen before any comparator fitting; no labels, outcomes, or spatial fields used.",
        "shared_marker_selection_used_celltype_or_outcome_labels": False,
        "merge_order_selected_using_posthoc_labels_or_metrics": False,
        "canonical_parameters_selected_using_posthoc_labels_or_metrics": False,
        "correction_marker_file": "markers.txt",
        "correction_marker_sha256": sha256(marker_path),
        "marker_harmonization_manifest_file": "marker_harmonization_manifest.csv",
        "marker_harmonization_manifest_sha256": sha256(harmonization_path),
        "cell_selection_manifest_file": "../../../../benchmark_data/lung_cancer/selected_cells.csv.gz",
        "cell_selection_manifest_sha256": sha256(selection),
        "cell_selection_id_column": "cell_id",
        "dataset_order_is_progressive_merge_order": True,
        "zero_l2_cell_count": int(zero_l2.sum()),
        "zero_l2_cell_count_by_dataset": zero_l2_by_dataset,
        "zero_l2_cell_policy": "Retain unchanged; official batchelor cosineNorm uses pmax(1e-8, L2). No deletion or epsilon is applied upstream.",
        "datasets": dataset_entries,
        "method_owned_publisher_file": "build_inputs.py",
        "method_owned_publisher_sha256": script_hash,
        "upstream_source_preprocessing_manifest_file": str(source_manifest_path.resolve()),
        "upstream_source_preprocessing_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
        "upstream_contains_corrected_method_output": False,
        "note": "The copied cache is the audited pre-correction provider/source-supported expression matrix; no cyCombine correction is read or reused.",
    }
    manifest_path = HERE / "preprocessing_manifest.json"
    write_json(manifest, manifest_path)

    ALIGNMENT.mkdir(parents=True, exist_ok=True)
    algorithm_config = {
        "schema_version": "final_xu.fastmnn_repro.v1",
        "method_label": "fastMNN (batchelor 1.18.1; corrected embedding)",
        "cell_id_column": "cell_id",
        "batch_column": "dataset",
        "sample_column": "unit_key",
        "batch_levels": DATASETS,
        "input_scale": {
            "status": "analysis_ready",
            "name": "source-supported platform-specific compressed protein scale",
            "description": "Senosain asinh(x/5), provider Cords c_counts_asinh, and provider Monkman analysis layer; no universal transform or marker z-score is added by fastMNN.",
            "preprocessing_record_file": "../preprocessing/preprocessing_manifest.json",
            "preprocessing_record_sha256": sha256(manifest_path),
            "input_sha256": sha256(DATA / "combined_analysis_ready.csv.gz"),
            "marker_file_sha256": sha256(marker_path),
            "algorithm_internal_cosine_normalization": True,
            "algorithm_applies_other_expression_transform": False,
        },
        "algorithm_parameters": algorithm_parameters(),
    }
    write_json(algorithm_config, ALIGNMENT / "config.json")
    build_manifest = {
        "schema_version": "final_xu.lung_cancer.fastmnn_input_build.v1",
        "status": "complete",
        "labels_or_outcomes_read": False,
        "cells": 60000,
        "markers": MARKERS,
        "dataset_order": DATASETS,
        "source_preprocessing_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
        "source_preprocessing_builder_sha256": EXPECTED_SOURCE_BUILDER_SHA256,
        "combined_input_sha256": sha256(DATA / "combined_analysis_ready.csv.gz"),
        "preprocessing_manifest_sha256": sha256(manifest_path),
        "algorithm_config_sha256": sha256(ALIGNMENT / "config.json"),
        "publisher_sha256": script_hash,
    }
    build_manifest_path = HERE / "preprocessing_build_manifest.json"
    write_json(build_manifest, build_manifest_path)
    write_json(
        {
            "status": "complete",
            "preprocessing_manifest_sha256": sha256(manifest_path),
            "preprocessing_build_manifest_sha256": sha256(build_manifest_path),
            "algorithm_config_sha256": sha256(ALIGNMENT / "config.json"),
        },
        HERE / "COMPLETE.json",
    )
    print("Built lung-cancer fastMNN input: 60,000 cells x 8 markers", flush=True)


if __name__ == "__main__":
    main()
