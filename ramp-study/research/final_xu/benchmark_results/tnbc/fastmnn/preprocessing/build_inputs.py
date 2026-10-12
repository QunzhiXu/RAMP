#!/usr/bin/env python3
"""Build the frozen TNBC analysis-ready input for canonical fastMNN.

This builder deliberately reuses only the immutable *pre-correction* table
that was already audited against the three provider datasets.  It never opens
cyCombine alignment outputs.  The copied numeric table is independently
checked against ``asinh(canonical_source_scale / 1)`` before publication.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
FASTMNN_DIR = HERE.parent
TNBC_RESULTS = FASTMNN_DIR.parent
FINAL_XU = HERE.parents[3]
BENCHMARK = FINAL_XU / "benchmark_data" / "tnbc"

# This is an immutable, pre-correction analysis-ready artifact.  Nothing under
# cycombine/alignment is read by this builder.
PRECORRECTION_ROOT = TNBC_RESULTS / "cycombine" / "preprocessing"
PRECORRECTION_DATA = PRECORRECTION_ROOT / "data"

CANONICAL_RAW = BENCHMARK / "input" / "raw.npy"
CANONICAL_MARKERS = BENCHMARK / "input" / "markers.txt"
SELECTED_CELLS = BENCHMARK / "selected_cells.csv.gz"
MARKER_MANIFEST = BENCHMARK / "marker_manifest.csv"
SOURCE_PREPROCESSING_RECORD = PRECORRECTION_ROOT / "preprocessing_manifest.json"
SOURCE_BUILD_RECIPE = PRECORRECTION_ROOT / "build_inputs.R"
ALGORITHM_CONFIG = FASTMNN_DIR / "alignment" / "config.json"

DATASET_ORDER = ["Tietscher IMC", "Meyer IMC", "Shiao CODEX"]
DATASET_SLUGS = {
    "Tietscher IMC": "tietscher_imc",
    "Meyer IMC": "meyer_imc",
    "Shiao CODEX": "shiao_codex",
}
DATASET_COUNTS = {
    "Tietscher IMC": 20_000,
    "Meyer IMC": 22_701,
    "Shiao CODEX": 20_000,
}
MARKERS = [
    "CD11C",
    "CD20",
    "CD3",
    "CD31",
    "CD4",
    "CD68",
    "CD8",
    "HLA_DR",
    "KI67",
    "PANCK",
    "SMA",
]
EXPECTED_COLUMNS = ["cell_id", "dataset", "donor", "unit_key", *MARKERS]

EXPECTED_SOURCE_HASHES = {
    "combined_analysis_ready.csv.gz":
        "c37eb7dced3bccad11921f0346450bffa86a242d2e1fb86a4fde1d4ad8f93daa",
    "tietscher_imc_analysis_ready.csv.gz":
        "7c76e78c912d796ec02602f24fedc998a553df6f1f9eef49a35b325ed9bb8a83",
    "meyer_imc_analysis_ready.csv.gz":
        "1409f025b86277dd352e6b43d8bca75050aa48af04fd43fa02430cd56bde97c3",
    "shiao_codex_analysis_ready.csv.gz":
        "5b093264e1dbe6d3db884e5ade57698af9b7c78cc3a9e45e1a3f26daf2ed9848",
    "markers.txt":
        "39bb80c51c6b7b246d915f2f524def088b056c51a9a083e2db44ea7fe79cce77",
    "selected_cells.csv.gz":
        "55ea28f02d14618b307f0309a7debda77381ec9769b9876c442d58c9a794dd1e",
    "marker_manifest.csv":
        "b474027cde63739b57e0f3a1fc3b1f308d15364f3c0f8955983046454b211751",
    "source_preprocessing_manifest.json":
        "0ed1a75e49c082e706314e165728778455686247e94e735d8fd7faf63fd3e72c",
    "source_build_recipe":
        "02bcc2918107fef85253a951cf61dae3f8e294774b7e4a0b5f35fdddf3c21fe3",
    "canonical_raw.npy":
        "f4b8b9e7e20fe80c005fc3fef549b6a2f5e60b0b03c1d016f8449b0625ff5c10",
}

PROVIDER_SOURCES = {
    "Tietscher IMC": {
        "path": FINAL_XU / "raw_data" / "tnbc" / "Tietscher_2023"
        / "Protein_panel_SingleCellExperiment_object.rds",
        "sha256": "660634f9f96e64ab6c3158510ebe6631b4943218aaafd0993539bff6661e9928",
    },
    "Meyer IMC": {
        "path": FINAL_XU / "raw_data" / "tnbc" / "Meyer_2025"
        / "processed_r_objects" / "sce_ALL.rds",
        "sha256": "cbe2a4e3f857f08e8516c045dcea9bd098c1e35e8bb45438ab6962650b6e5974",
    },
    "Shiao CODEX": {
        "path": FINAL_XU / "raw_data" / "tnbc" / "Shiao_2024"
        / "TNBC_immune_CODEX_full_panel_counts.csv",
        "sha256": "0697184090d5373c4d9ee7a052863504ff6984e9a88674c7158bdd3713361c96",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256(path)
    if observed != expected:
        raise AssertionError(
            f"{label} hash changed: expected {expected}, observed {observed}"
        )


def write_json(value: object, path: Path) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def source_files() -> dict[str, Path]:
    files = {
        "combined_analysis_ready.csv.gz":
            PRECORRECTION_DATA / "combined_analysis_ready.csv.gz",
        "tietscher_imc_analysis_ready.csv.gz":
            PRECORRECTION_DATA / "tietscher_imc_analysis_ready.csv.gz",
        "meyer_imc_analysis_ready.csv.gz":
            PRECORRECTION_DATA / "meyer_imc_analysis_ready.csv.gz",
        "shiao_codex_analysis_ready.csv.gz":
            PRECORRECTION_DATA / "shiao_codex_analysis_ready.csv.gz",
    }
    for path in files.values():
        if "alignment" in path.parts or "corrected_expression" in path.name:
            raise AssertionError("A correction/alignment artifact entered preprocessing")
    return files


def validate_frozen_sources() -> None:
    for name, path in source_files().items():
        require_hash(path, EXPECTED_SOURCE_HASHES[name], f"pre-correction {name}")
    require_hash(
        CANONICAL_MARKERS, EXPECTED_SOURCE_HASHES["markers.txt"], "canonical markers"
    )
    require_hash(
        SELECTED_CELLS,
        EXPECTED_SOURCE_HASHES["selected_cells.csv.gz"],
        "canonical selected cells",
    )
    require_hash(
        MARKER_MANIFEST,
        EXPECTED_SOURCE_HASHES["marker_manifest.csv"],
        "canonical marker manifest",
    )
    require_hash(
        SOURCE_PREPROCESSING_RECORD,
        EXPECTED_SOURCE_HASHES["source_preprocessing_manifest.json"],
        "source preprocessing record",
    )
    require_hash(
        SOURCE_BUILD_RECIPE,
        EXPECTED_SOURCE_HASHES["source_build_recipe"],
        "source preprocessing recipe",
    )
    require_hash(
        CANONICAL_RAW,
        EXPECTED_SOURCE_HASHES["canonical_raw.npy"],
        "canonical source-scale matrix",
    )
    for dataset, record in PROVIDER_SOURCES.items():
        require_hash(record["path"], record["sha256"], f"{dataset} provider source")

    markers = [line.strip() for line in CANONICAL_MARKERS.read_text().splitlines() if line.strip()]
    if markers != MARKERS:
        raise AssertionError(f"Marker order changed: {markers}")


def validate_combined(path: Path) -> dict[str, object]:
    table = pd.read_csv(
        path,
        dtype={
            "cell_id": str,
            "dataset": str,
            "donor": str,
            "unit_key": str,
        },
        low_memory=False,
    )
    if table.columns.tolist() != EXPECTED_COLUMNS:
        raise AssertionError(f"Combined columns changed: {table.columns.tolist()}")
    if len(table) != sum(DATASET_COUNTS.values()) or not table["cell_id"].is_unique:
        raise AssertionError("Combined row count or cell-ID uniqueness changed")

    selected = pd.read_csv(
        SELECTED_CELLS,
        usecols=["cell_id", "dataset", "donor", "unit_key"],
        dtype=str,
        low_memory=False,
    )
    if not table[["cell_id", "dataset", "donor", "unit_key"]].equals(selected):
        raise AssertionError("Combined identities/order differ from selected_cells.csv.gz")
    observed_order = table["dataset"].drop_duplicates().tolist()
    if observed_order != DATASET_ORDER:
        raise AssertionError(f"Dataset block order changed: {observed_order}")
    observed_counts = table["dataset"].value_counts(sort=False).to_dict()
    if observed_counts != DATASET_COUNTS:
        raise AssertionError(f"Dataset counts changed: {observed_counts}")

    matrix = table[MARKERS].to_numpy(dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise AssertionError("Analysis-ready marker matrix contains non-finite values")
    canonical_raw = np.load(CANONICAL_RAW, allow_pickle=False)
    expected = np.arcsinh(np.asarray(canonical_raw, dtype=np.float64))
    if matrix.shape != expected.shape:
        raise AssertionError("Analysis-ready matrix shape changed")
    maximum_absolute_error = float(np.max(np.abs(matrix - expected)))
    if maximum_absolute_error > 5e-14:
        raise AssertionError(
            "Analysis-ready values are not asinh(canonical source scale / 1): "
            f"maximum error {maximum_absolute_error}"
        )
    l2 = np.linalg.norm(matrix, axis=1)
    if np.any(~np.isfinite(l2) | (l2 <= 0)):
        raise AssertionError("Analysis-ready matrix contains zero-L2 cells")
    variances = np.var(matrix, axis=0, ddof=1)
    if np.any(~np.isfinite(variances) | (variances <= 0)):
        raise AssertionError("Analysis-ready matrix contains a constant marker")

    return {
        "cells": int(len(table)),
        "markers": int(len(MARKERS)),
        "maximum_absolute_error_vs_asinh_canonical_source_scale": maximum_absolute_error,
        "zero_l2_cells": int(np.sum(l2 <= 0)),
        "all_marker_variances_positive": True,
        "cell_ids_and_row_order_match_selected_cells": True,
        "dataset_order": DATASET_ORDER,
        "dataset_counts": DATASET_COUNTS,
    }


def validate_per_dataset(data_dir: Path, combined_path: Path) -> None:
    combined = pd.read_csv(combined_path, dtype={"cell_id": str}, low_memory=False)
    for dataset in DATASET_ORDER:
        slug = DATASET_SLUGS[dataset]
        path = data_dir / f"{slug}_analysis_ready.csv.gz"
        expected_hash = EXPECTED_SOURCE_HASHES[f"{slug}_analysis_ready.csv.gz"]
        require_hash(path, expected_hash, f"copied {dataset} analysis-ready input")
        current = pd.read_csv(path, dtype={"cell_id": str}, low_memory=False)
        expected = combined.loc[combined["dataset"] == dataset].reset_index(drop=True)
        if current.columns.tolist() != EXPECTED_COLUMNS or len(current) != DATASET_COUNTS[dataset]:
            raise AssertionError(f"{dataset} analysis-ready structure changed")
        if not current["cell_id"].equals(expected["cell_id"]):
            raise AssertionError(f"{dataset} cell order differs from the combined table")
        error = float(
            np.max(
                np.abs(
                    current[MARKERS].to_numpy(np.float64)
                    - expected[MARKERS].to_numpy(np.float64)
                )
            )
        )
        if error != 0.0:
            raise AssertionError(f"{dataset} and combined values differ: {error}")


def dataset_records(builder_hash: str) -> list[dict[str, object]]:
    shared = {
        "processing_code_file": "build_inputs.py",
        "processing_code_sha256": builder_hash,
        "universal_positive_median_arcsinh_not_used": True,
        "marker_wise_zscore_not_used": True,
        "double_transform_checked": True,
    }
    return [
        {
            "dataset": "Tietscher IMC",
            "technology": "Imaging mass cytometry (IMC)",
            "analysis_ready_matrix_file": "data/tietscher_imc_analysis_ready.csv.gz",
            "analysis_ready_matrix_sha256": EXPECTED_SOURCE_HASHES[
                "tietscher_imc_analysis_ready.csv.gz"
            ],
            "source_file": str(PROVIDER_SOURCES["Tietscher IMC"]["path"]),
            "source_file_sha256": PROVIDER_SOURCES["Tietscher IMC"]["sha256"],
            "source_state": "provider_analysis_scale",
            "assay_or_layer": (
                "Protein_panel_SingleCellExperiment_object.rds exprs assay for the "
                "frozen cells and source channels"
            ),
            "processing_steps": [
                "Retain the exact 20,000 frozen BB165 cells and canonical row order.",
                "Use the provider exprs assay after verifying exprs equals asinh(counts/1).",
                "Apply no second transform, marker z-score, clipping, or scaling.",
            ],
            "transform_formula": "provider exprs = asinh(counts / 1); no additional transform",
            "transform_scope": "element-wise provider protein expression before frozen-cell selection",
            "processing_reference": (
                "Tietscher et al. 2023 IMC analysis methods; provider assay identity "
                "verified against the frozen source-scale matrix"
            ),
            **shared,
        },
        {
            "dataset": "Meyer IMC",
            "technology": "Imaging mass cytometry (IMC)",
            "analysis_ready_matrix_file": "data/meyer_imc_analysis_ready.csv.gz",
            "analysis_ready_matrix_sha256": EXPECTED_SOURCE_HASHES[
                "meyer_imc_analysis_ready.csv.gz"
            ],
            "source_file": str(PROVIDER_SOURCES["Meyer IMC"]["path"]),
            "source_file_sha256": PROVIDER_SOURCES["Meyer IMC"]["sha256"],
            "source_state": "provider_analysis_scale",
            "assay_or_layer": (
                "processed_r_objects/sce_ALL.rds exprs assay for the frozen cells "
                "and source channels"
            ),
            "processing_steps": [
                "Retain all 22,701 cells from the frozen five-patient coverage-informed set.",
                "Use the provider exprs assay after verifying exprs equals asinh(counts/1).",
                "Apply no min-max scale, second transform, marker z-score, or clipping.",
            ],
            "transform_formula": "provider exprs = asinh(counts / 1); no additional transform",
            "transform_scope": "element-wise mean-pixel intensity before frozen-cell selection",
            "processing_reference": (
                "Meyer et al. 2025 Data processing: mean pixel intensities were "
                "arcsinh-transformed with cofactor 1"
            ),
            **shared,
        },
        {
            "dataset": "Shiao CODEX",
            "technology": "CODEX multiplexed fluorescence imaging",
            "analysis_ready_matrix_file": "data/shiao_codex_analysis_ready.csv.gz",
            "analysis_ready_matrix_sha256": EXPECTED_SOURCE_HASHES[
                "shiao_codex_analysis_ready.csv.gz"
            ],
            "source_file": str(PROVIDER_SOURCES["Shiao CODEX"]["path"]),
            "source_file_sha256": PROVIDER_SOURCES["Shiao CODEX"]["sha256"],
            "source_state": "benchmark_transformed_segmented_cell_intensity",
            "assay_or_layer": (
                "Frozen arithmetic mean of provider nuclei_mean and membrane_mean "
                "fields for each shared marker"
            ),
            "processing_steps": [
                "Retain the exact 20,000 frozen cells and canonical row order.",
                "Use the benchmark-locked nuclei/membrane arithmetic mean without compartment reselection.",
                "Apply asinh(x/1) once as a predeclared comparator adaptation.",
                "Apply no marker z-score, data-derived cofactor, clipping, or second transform.",
            ],
            "transform_formula": "asinh(mean(nuclei_mean, membrane_mean) / 1)",
            "transform_scope": "element-wise on all eleven frozen compartment averages",
            "processing_reference": (
                "Shiao et al. 2024 CODEX image processing methods document corrected "
                "segmented-cell channel averages but no nonlinear expression transform; "
                "cofactor-1 asinh is a frozen benchmark adaptation, not a provider claim"
            ),
            **shared,
        },
    ]


def build_preprocessing_manifest(builder_hash: str) -> dict[str, object]:
    return {
        "schema_version": "final_xu.fastmnn_preprocessing.v1",
        "combined_input_file": "data/combined_analysis_ready.csv.gz",
        "combined_input_sha256": EXPECTED_SOURCE_HASHES[
            "combined_analysis_ready.csv.gz"
        ],
        "shared_marker_policy": (
            "Exact ordered intersection of eleven harmonized protein markers frozen "
            "before method fitting; no label, outcome, spatial field, alignment, or UMAP "
            "output enters preprocessing."
        ),
        "shared_marker_selection_used_celltype_or_outcome_labels": False,
        "merge_order_selected_using_posthoc_labels_or_metrics": False,
        "canonical_parameters_selected_using_posthoc_labels_or_metrics": False,
        "correction_marker_file": "markers.txt",
        "correction_marker_sha256": EXPECTED_SOURCE_HASHES["markers.txt"],
        "marker_harmonization_manifest_file": "marker_harmonization_manifest.csv",
        "marker_harmonization_manifest_sha256": EXPECTED_SOURCE_HASHES[
            "marker_manifest.csv"
        ],
        "cell_selection_manifest_file": "../../../../benchmark_data/tnbc/selected_cells.csv.gz",
        "cell_selection_manifest_sha256": EXPECTED_SOURCE_HASHES[
            "selected_cells.csv.gz"
        ],
        "cell_selection_id_column": "cell_id",
        "dataset_order_is_progressive_merge_order": True,
        "dataset_order": DATASET_ORDER,
        "zero_l2_cell_count": 0,
        "zero_l2_cell_count_by_dataset": {
            dataset: 0 for dataset in DATASET_ORDER
        },
        "input_scale_name": "cofactor1_asinh_analysis_ready",
        "combined_transform_formula": "asinh(canonical source-scale marker value / 1)",
        "fastmnn_algorithm_layer_expression_transform": "none",
        "fastmnn_algorithm_internal_cosine_normalization": True,
        "precorrection_artifact_reuse": {
            "path": str(PRECORRECTION_DATA / "combined_analysis_ready.csv.gz"),
            "sha256": EXPECTED_SOURCE_HASHES["combined_analysis_ready.csv.gz"],
            "purpose": (
                "Immutable byte source for an already audited pre-correction analysis-ready matrix; "
                "the fastMNN-owned copy is independently checked against asinh(canonical raw)."
            ),
            "cycombine_alignment_or_corrected_expression_read": False,
            "cycombine_correction_applied": False,
        },
        "source_preprocessing_record": {
            "path": str(SOURCE_PREPROCESSING_RECORD),
            "sha256": EXPECTED_SOURCE_HASHES["source_preprocessing_manifest.json"],
        },
        "datasets": dataset_records(builder_hash),
    }


def build_algorithm_config(preprocessing_manifest_hash: str) -> dict[str, object]:
    """Return the exact algorithm/fastmnn v1 config authenticated by this build."""
    return {
        "schema_version": "final_xu.fastmnn_repro.v1",
        "method_label": "fastMNN (batchelor 1.18.1; corrected embedding)",
        "cell_id_column": "cell_id",
        "batch_column": "dataset",
        "sample_column": "unit_key",
        "batch_levels": DATASET_ORDER,
        "input_scale": {
            "status": "analysis_ready",
            "name": "cofactor1_asinh_analysis_ready",
            "description": (
                "Tietscher and Meyer provider exprs scales verified as asinh(counts/1); "
                "Shiao frozen compartment means receive one predeclared asinh(x/1) "
                "benchmark adaptation. No marker-wise z-score or cyCombine correction."
            ),
            "preprocessing_record_file": "../preprocessing/preprocessing_manifest.json",
            "preprocessing_record_sha256": preprocessing_manifest_hash,
            "input_sha256": EXPECTED_SOURCE_HASHES["combined_analysis_ready.csv.gz"],
            "marker_file_sha256": EXPECTED_SOURCE_HASHES["markers.txt"],
            "algorithm_internal_cosine_normalization": True,
            "algorithm_applies_other_expression_transform": False,
        },
        "algorithm_parameters": {
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
        },
    }


def validate_published() -> dict[str, object]:
    data_dir = HERE / "data"
    combined = data_dir / "combined_analysis_ready.csv.gz"
    require_hash(
        combined,
        EXPECTED_SOURCE_HASHES["combined_analysis_ready.csv.gz"],
        "fastMNN combined input",
    )
    require_hash(HERE / "markers.txt", EXPECTED_SOURCE_HASHES["markers.txt"], "fastMNN markers")
    require_hash(
        HERE / "marker_harmonization_manifest.csv",
        EXPECTED_SOURCE_HASHES["marker_manifest.csv"],
        "fastMNN marker harmonization manifest",
    )
    audit = validate_combined(combined)
    validate_per_dataset(data_dir, combined)
    record = json.loads((HERE / "preprocessing_manifest.json").read_text(encoding="utf-8"))
    required_false = [
        "shared_marker_selection_used_celltype_or_outcome_labels",
        "merge_order_selected_using_posthoc_labels_or_metrics",
        "canonical_parameters_selected_using_posthoc_labels_or_metrics",
    ]
    if record.get("schema_version") != "final_xu.fastmnn_preprocessing.v1":
        raise AssertionError("Published preprocessing manifest has the wrong schema")
    if any(record.get(field) is not False for field in required_false):
        raise AssertionError("Published preprocessing manifest lacks label-free declarations")
    if record.get("dataset_order") != DATASET_ORDER:
        raise AssertionError("Published preprocessing manifest dataset order changed")
    if record.get("zero_l2_cell_count") != 0:
        raise AssertionError("Published preprocessing manifest zero-L2 count changed")
    if record.get("zero_l2_cell_count_by_dataset") != {
        dataset: 0 for dataset in DATASET_ORDER
    }:
        raise AssertionError("Published preprocessing per-dataset zero-L2 counts changed")
    algorithm_config = json.loads(ALGORITHM_CONFIG.read_text(encoding="utf-8"))
    if algorithm_config.get("schema_version") != "final_xu.fastmnn_repro.v1":
        raise AssertionError("Published algorithm config has the wrong schema")
    if algorithm_config.get("batch_levels") != DATASET_ORDER:
        raise AssertionError("Published algorithm config dataset order changed")
    scale = algorithm_config.get("input_scale", {})
    if scale.get("input_sha256") != EXPECTED_SOURCE_HASHES[
        "combined_analysis_ready.csv.gz"
    ]:
        raise AssertionError("Published algorithm config input hash changed")
    if scale.get("marker_file_sha256") != EXPECTED_SOURCE_HASHES["markers.txt"]:
        raise AssertionError("Published algorithm config marker hash changed")
    if scale.get("preprocessing_record_sha256") != sha256(
        HERE / "preprocessing_manifest.json"
    ):
        raise AssertionError("Published algorithm config preprocessing hash changed")
    return audit


def main() -> None:
    validate_frozen_sources()
    final_files = [
        HERE / "data",
        HERE / "markers.txt",
        HERE / "marker_harmonization_manifest.csv",
        HERE / "preprocessing_manifest.json",
        HERE / "preprocessing_build_manifest.json",
        HERE / "COMPLETE.json",
        ALGORITHM_CONFIG,
    ]
    if any(path.exists() for path in final_files):
        if not all(path.exists() for path in final_files):
            raise FileExistsError("Refusing to overwrite an incomplete fastMNN preprocessing build")
        audit = validate_published()
        print(
            f"PASS: reused validated TNBC fastMNN input; {audit['cells']} cells x "
            f"{audit['markers']} markers"
        )
        return

    builder_hash = sha256(Path(__file__).resolve())
    stage = Path(tempfile.mkdtemp(prefix=".fastmnn-preprocessing-", dir=HERE))
    try:
        stage_data = stage / "data"
        stage_data.mkdir()
        for name, source in source_files().items():
            shutil.copyfile(source, stage_data / name)
        shutil.copyfile(CANONICAL_MARKERS, stage / "markers.txt")
        shutil.copyfile(MARKER_MANIFEST, stage / "marker_harmonization_manifest.csv")

        audit = validate_combined(stage_data / "combined_analysis_ready.csv.gz")
        validate_per_dataset(stage_data, stage_data / "combined_analysis_ready.csv.gz")
        preprocessing_record = build_preprocessing_manifest(builder_hash)
        write_json(preprocessing_record, stage / "preprocessing_manifest.json")
        preprocessing_manifest_hash = sha256(stage / "preprocessing_manifest.json")
        write_json(
            build_algorithm_config(preprocessing_manifest_hash),
            stage / "algorithm_config.json",
        )

        output_hashes = {
            path.name: sha256(path)
            for path in sorted(stage_data.iterdir())
            if path.is_file()
        }
        output_hashes.update(
            {
                "markers.txt": sha256(stage / "markers.txt"),
                "marker_harmonization_manifest.csv": sha256(
                    stage / "marker_harmonization_manifest.csv"
                ),
                "preprocessing_manifest.json": sha256(
                    stage / "preprocessing_manifest.json"
                ),
                "alignment/config.json": sha256(stage / "algorithm_config.json"),
            }
        )
        build_manifest = {
            "schema_version": "final_xu.tnbc.fastmnn_input_build.v1",
            "status": "complete",
            "method": "fastMNN (batchelor 1.18.1; corrected embedding)",
            "labels_read_during_preprocessing": False,
            "alignment_or_corrected_expression_read": False,
            "cycombine_correction_applied": False,
            "precorrection_bytes_copied_without_reserialization": True,
            "dataset_order": DATASET_ORDER,
            "dataset_counts": DATASET_COUNTS,
            "markers": MARKERS,
            "analysis_ready_formula": "asinh(canonical source-scale marker value / 1)",
            "numeric_verification": audit,
            "sources": {
                "canonical_raw": {
                    "path": str(CANONICAL_RAW),
                    "sha256": EXPECTED_SOURCE_HASHES["canonical_raw.npy"],
                },
                "selected_cells": {
                    "path": str(SELECTED_CELLS),
                    "sha256": EXPECTED_SOURCE_HASHES["selected_cells.csv.gz"],
                },
                "markers": {
                    "path": str(CANONICAL_MARKERS),
                    "sha256": EXPECTED_SOURCE_HASHES["markers.txt"],
                },
                "marker_manifest": {
                    "path": str(MARKER_MANIFEST),
                    "sha256": EXPECTED_SOURCE_HASHES["marker_manifest.csv"],
                },
                "precorrection_combined": {
                    "path": str(PRECORRECTION_DATA / "combined_analysis_ready.csv.gz"),
                    "sha256": EXPECTED_SOURCE_HASHES[
                        "combined_analysis_ready.csv.gz"
                    ],
                },
                "source_preprocessing_record": {
                    "path": str(SOURCE_PREPROCESSING_RECORD),
                    "sha256": EXPECTED_SOURCE_HASHES[
                        "source_preprocessing_manifest.json"
                    ],
                },
            },
            "outputs": output_hashes,
            "script": {"path": str(Path(__file__).resolve()), "sha256": builder_hash},
            "runtime": {
                "Python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
        }
        write_json(build_manifest, stage / "preprocessing_build_manifest.json")
        write_json(
            {
                "schema_version": "final_xu.tnbc.fastmnn_input_complete.v1",
                "status": "complete",
                "preprocessing_manifest_sha256": sha256(
                    stage / "preprocessing_manifest.json"
                ),
                "algorithm_config_sha256": sha256(stage / "algorithm_config.json"),
                "preprocessing_build_manifest_sha256": sha256(
                    stage / "preprocessing_build_manifest.json"
                ),
            },
            stage / "COMPLETE.json",
        )

        os.replace(stage_data, HERE / "data")
        for name in [
            "markers.txt",
            "marker_harmonization_manifest.csv",
            "preprocessing_manifest.json",
            "preprocessing_build_manifest.json",
            "COMPLETE.json",
        ]:
            os.replace(stage / name, HERE / name)
        ALGORITHM_CONFIG.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage / "algorithm_config.json", ALGORITHM_CONFIG)
        stage.rmdir()
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise

    audit = validate_published()
    print(
        f"PASS: built TNBC fastMNN input; {audit['cells']} cells x "
        f"{audit['markers']} markers; max formula error "
        f"{audit['maximum_absolute_error_vs_asinh_canonical_source_scale']:.3g}"
    )


if __name__ == "__main__":
    main()
