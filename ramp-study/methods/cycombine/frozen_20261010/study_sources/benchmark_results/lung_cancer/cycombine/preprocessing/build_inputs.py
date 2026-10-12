#!/usr/bin/env python3
"""Build source-faithful cyCombine inputs for frozen lung-cancer cells.

Only provider expression sources and the canonical benchmark identity/order
locks are read. Labels, coordinates, clinical variables, outcomes, RAMP, and
other method outputs are deliberately unavailable to this build.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import platform
import subprocess
import tempfile
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
METHOD_DIR = HERE.parent
FINAL_XU = METHOD_DIR.parents[2]
BENCHMARK = FINAL_XU / "benchmark_data" / "lung_cancer"
RAW_ROOT = FINAL_XU / "raw_data" / "lung_cancer"
DATA_DIR = HERE / "data"

DATASETS = ["Senosain CyTOF", "Cords IMC", "Monkman CODEX"]
SLUGS = {
    "Senosain CyTOF": "senosain_cytof",
    "Cords IMC": "cords_imc",
    "Monkman CODEX": "monkman_codex",
}
MARKERS = ["CD3", "CD4", "CD8", "CD31", "HLA_DR", "KI67", "VIMENTIN", "EPI"]
ID_COLUMNS = ["cell_id", "dataset", "donor", "unit_key"]
EXPECTED_PER_DATASET = 20_000
EXPECTED_TOTAL = 60_000

SENOSAIN_NATIVE = {
    "CD3": "CD3", "CD4": "CD4", "CD8": "CD8", "CD31": "CD31",
    "HLA_DR": "HLA_DR", "KI67": "KI67", "VIMENTIN": "VIMENTIN",
    "EPI": "CYTOKERATIN",
}
CORDS_NATIVE = {
    "CD3": "CD3", "CD4": "CD4", "CD8": "CD8a", "CD31": "vWF + CD31",
    "HLA_DR": "HLA-DR", "KI67": "Ki-67", "VIMENTIN": "Vimentin",
    "EPI": "Pan Cytokeratin + Keratin Epithelial",
}
MONKMAN_NATIVE = {
    "CD3": "CD3e", "CD4": "CD4", "CD8": "CD8", "CD31": "CD31",
    "HLA_DR": "HLA-DR", "KI67": "Ki67", "VIMENTIN": "Vimentin",
    "EPI": "PanCK",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def buffer_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(values, dtype=np.float64).tobytes()
    ).hexdigest()


def source_record(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "bytes": int(stat.st_size),
        "sha256": sha256(path),
    }


def deterministic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(
                filename="", fileobj=raw, mode="wb", compresslevel=6, mtime=0
            ) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                    frame.to_csv(text, index=False, float_format="%.17g")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def deterministic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def canonical_locks() -> tuple[pd.DataFrame, np.ndarray, dict[str, pd.DataFrame], dict[str, object]]:
    selected_path = BENCHMARK / "selected_cells.csv.gz"
    raw_path = BENCHMARK / "input" / "raw.npy"
    marker_path = BENCHMARK / "input" / "markers.txt"
    benchmark_manifest_path = BENCHMARK / "manifest.json"
    benchmark_manifest = json.loads(benchmark_manifest_path.read_text())
    selected = pd.read_csv(
        selected_path,
        dtype={
            "cell_id": str, "source_cell_id": str, "dataset": str,
            "donor": str, "unit_key": str,
        },
        low_memory=False,
    )
    required = {
        "row_id", "cell_id", "source_cell_id", "source_row_zero_based",
        "dataset", "donor", "unit_key",
    }
    if not required.issubset(selected.columns):
        raise AssertionError("selected_cells.csv.gz lacks canonical identity columns")
    if len(selected) != EXPECTED_TOTAL:
        raise AssertionError("Frozen lung cell count changed")
    if not np.array_equal(selected["row_id"].to_numpy(), np.arange(EXPECTED_TOTAL)):
        raise AssertionError("Frozen lung row_id order changed")
    if selected["cell_id"].duplicated().any() or selected["source_cell_id"].duplicated().any():
        raise AssertionError("Canonical lung cell identifiers are not globally unique")
    block_order = selected.loc[
        selected["dataset"].ne(selected["dataset"].shift(fill_value="")), "dataset"
    ].tolist()
    if block_order != DATASETS:
        raise AssertionError(f"Frozen dataset block order changed: {block_order}")
    counts = selected["dataset"].value_counts(sort=False).to_dict()
    if counts != {dataset: EXPECTED_PER_DATASET for dataset in DATASETS}:
        raise AssertionError(f"Frozen dataset counts changed: {counts}")

    markers = [line for line in marker_path.read_text().splitlines() if line]
    raw = np.load(raw_path, mmap_mode="r", allow_pickle=False)
    if markers != MARKERS or raw.shape != (EXPECTED_TOTAL, len(MARKERS)):
        raise AssertionError("Canonical lung marker matrix contract changed")
    if raw.dtype != np.float64 or not np.isfinite(raw).all():
        raise AssertionError("Canonical lung raw matrix is not finite float64")

    files = benchmark_manifest.get("files_sha256", {})
    for relative in (
        "selected_cells.csv.gz", "input/raw.npy", "input/markers.txt",
        "senosain_cytof/alignment_input_8.csv.gz",
        "cords_imc/alignment_input_8.csv.gz",
        "monkman_codex/alignment_input_8.csv.gz",
    ):
        path = BENCHMARK / relative
        if files.get(relative) != sha256(path):
            raise AssertionError(f"Benchmark manifest hash mismatch: {relative}")

    blocks: dict[str, pd.DataFrame] = {}
    inspectable: dict[str, object] = {}
    for dataset in DATASETS:
        slug = SLUGS[dataset]
        block = selected.loc[selected["dataset"].eq(dataset)].reset_index(drop=True)
        inspectable_path = BENCHMARK / slug / "alignment_input_8.csv.gz"
        frame = pd.read_csv(
            inspectable_path,
            dtype={"cell_id": str, "dataset": str, "donor": str, "unit_key": str},
            low_memory=False,
        )
        if list(frame.columns) != ["row_id", *ID_COLUMNS, *MARKERS]:
            raise AssertionError(f"{dataset}: inspectable input columns changed")
        if not np.array_equal(frame["cell_id"].to_numpy(), block["cell_id"].to_numpy()):
            raise AssertionError(f"{dataset}: inspectable cell order differs")
        rows = block["row_id"].to_numpy(dtype=np.int64)
        csv_roundtrip_error = float(np.max(np.abs(
            frame[MARKERS].to_numpy(np.float64) - np.asarray(raw[rows])
        )))
        if csv_roundtrip_error > 1e-12:
            raise AssertionError(
                f"{dataset}: inspectable matrix differs from canonical raw.npy "
                f"beyond numeric CSV round-trip tolerance: {csv_roundtrip_error}"
            )
        blocks[dataset] = block
        inspectable[slug] = {
            **source_record(inspectable_path),
            "maximum_float64_csv_roundtrip_error": csv_roundtrip_error,
        }

    provenance = {
        "benchmark_manifest": source_record(benchmark_manifest_path),
        "selected_cells": source_record(selected_path),
        "raw_shared": source_record(raw_path),
        "shared_markers": source_record(marker_path),
        "inspectable_inputs": inspectable,
    }
    return selected, np.asarray(raw), blocks, provenance


def ready_frame(block: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(block), len(MARKERS)) or not np.isfinite(values).all():
        raise AssertionError("Analysis-ready expression has invalid shape or values")
    frame = block[ID_COLUMNS].copy()
    for index, marker in enumerate(MARKERS):
        frame[marker] = values[:, index]
    return frame


def build_senosain(
    block: pd.DataFrame, expected_raw: np.ndarray
) -> tuple[pd.DataFrame, dict[str, object]]:
    source = RAW_ROOT / "Senosain_2025_CyTOF_and_omics" / "annotation"
    matrix_path = source / "senosain_140k_raw35.npy"
    marker_path = source / "senosain_140k_markers35.csv"
    metadata_path = source / "senosain_140k_cell_metadata.csv.gz"
    matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
    marker_table = pd.read_csv(marker_path, low_memory=False).sort_values(
        "raw35_column_zero_based"
    )
    native_markers = marker_table["marker"].astype(str).tolist()
    if matrix.shape != (140_000, 35) or len(native_markers) != 35:
        raise AssertionError(f"Senosain source shape changed: {matrix.shape}")
    positions_numeric = pd.to_numeric(block["source_row_zero_based"], errors="raise")
    positions = positions_numeric.to_numpy(np.int64)
    if not np.array_equal(positions_numeric.to_numpy(np.float64), positions.astype(np.float64)):
        raise AssertionError("Senosain source row positions are not integral")
    if positions.min() < 0 or positions.max() >= len(matrix) or len(np.unique(positions)) != len(block):
        raise AssertionError("Senosain source row positions are invalid")
    identities = pd.read_csv(
        metadata_path,
        usecols=["sampled_row_zero_based", "source_cell_id"],
        dtype={"source_cell_id": str},
    ).sort_values("sampled_row_zero_based")
    if not np.array_equal(identities["sampled_row_zero_based"].to_numpy(), np.arange(140_000)):
        raise AssertionError("Senosain source identity row order changed")
    if not np.array_equal(
        identities.iloc[positions]["source_cell_id"].to_numpy(),
        block["source_cell_id"].to_numpy(),
    ):
        raise AssertionError("Senosain provider identities differ from frozen cells")
    native_index = {marker: index for index, marker in enumerate(native_markers)}
    raw_values = np.column_stack([
        np.asarray(matrix[positions, native_index[SENOSAIN_NATIVE[marker]]], dtype=np.float64)
        for marker in MARKERS
    ])
    raw_error = float(np.max(np.abs(raw_values - expected_raw)))
    if raw_error != 0.0:
        raise AssertionError(f"Senosain canonical raw reconstruction error: {raw_error}")
    transformed = np.arcsinh(raw_values / 5.0)
    return ready_frame(block, transformed), {
        "source_files": {
            "raw35": source_record(matrix_path),
            "marker_map": source_record(marker_path),
            "identity_lock": source_record(metadata_path),
        },
        "raw_reconstruction_maximum_absolute_error": raw_error,
        "source_buffer_sha256_float64": buffer_sha256(raw_values),
        "analysis_ready_buffer_sha256_float64": buffer_sha256(transformed),
    }


CORDS_EXPORT_R = r'''
suppressPackageStartupMessages({
  library(SingleCellExperiment)
  library(data.table)
})
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args) == 4L)
sce_path <- args[[1L]]
identity_path <- args[[2L]]
ready_path <- args[[3L]]
raw_path <- args[[4L]]
identity <- fread(identity_path, colClasses=list(character="source_cell_id"))
stopifnot(names(identity) == "source_cell_id", nrow(identity) == 20000L)
sce <- readRDS(sce_path)
positions <- match(identity$source_cell_id, colnames(sce))
stopifnot(!anyNA(positions), uniqueN(positions) == 20000L)
native <- c(
  "CD3", "CD4", "CD8a", "vWF + CD31", "HLA-DR", "Ki-67",
  "Vimentin", "Pan Cytokeratin + Keratin Epithelial"
)
canonical <- c("CD3", "CD4", "CD8", "CD31", "HLA_DR", "KI67", "VIMENTIN", "EPI")
censored <- assay(sce, "c_counts")[native, positions, drop=FALSE]
ready <- assay(sce, "c_counts_asinh")[native, positions, drop=FALSE]
stopifnot(max(abs(asinh(censored) - ready)) < 1e-12)
raw <- assay(sce, "counts")[native, positions, drop=FALSE]
ready_frame <- as.data.frame(t(ready), check.names=FALSE)
raw_frame <- as.data.frame(t(raw), check.names=FALSE)
colnames(ready_frame) <- canonical
colnames(raw_frame) <- canonical
fwrite(cbind(data.frame(source_cell_id=identity$source_cell_id), ready_frame), ready_path, compress="gzip")
fwrite(cbind(data.frame(source_cell_id=identity$source_cell_id), raw_frame), raw_path, compress="gzip")
'''


def build_cords(
    block: pd.DataFrame, expected_raw: np.ndarray
) -> tuple[pd.DataFrame, dict[str, object]]:
    sce_path = RAW_ROOT / "Cords2024" / "SingleCellExperiment Objects" / "sce_all_annotated.rds"
    with tempfile.TemporaryDirectory(prefix=".cords-extract-", dir=HERE) as temp_name:
        temp = Path(temp_name)
        identity_path = temp / "identity.csv.gz"
        ready_path = temp / "ready.csv.gz"
        raw_path = temp / "raw.csv.gz"
        deterministic_csv(block[["source_cell_id"]].astype(str), identity_path)
        completed = subprocess.run(
            [
                "Rscript", "--vanilla", "-e", CORDS_EXPORT_R,
                str(sce_path), str(identity_path), str(ready_path), str(raw_path),
            ],
            check=True,
            cwd=HERE,
            text=True,
            capture_output=True,
        )
        if completed.stdout.strip():
            print(completed.stdout.strip(), flush=True)
        if completed.stderr.strip():
            print(completed.stderr.strip(), flush=True)
        ready = pd.read_csv(ready_path, dtype={"source_cell_id": str}, low_memory=False)
        raw = pd.read_csv(raw_path, dtype={"source_cell_id": str}, low_memory=False)
    expected_ids = block["source_cell_id"].to_numpy()
    if not (
        np.array_equal(ready["source_cell_id"].to_numpy(), expected_ids)
        and np.array_equal(raw["source_cell_id"].to_numpy(), expected_ids)
    ):
        raise AssertionError("Cords provider identities differ from frozen cells")
    ready_values = ready[MARKERS].to_numpy(np.float64)
    raw_values = raw[MARKERS].to_numpy(np.float64)
    raw_error = float(np.max(np.abs(raw_values - expected_raw)))
    if raw_error > 1e-8:
        raise AssertionError(f"Cords canonical raw reconstruction error: {raw_error}")
    return ready_frame(block, ready_values), {
        "source_files": {"annotated_sce": source_record(sce_path)},
        "raw_reconstruction_maximum_absolute_error": raw_error,
        "source_buffer_sha256_float64": buffer_sha256(raw_values),
        "analysis_ready_buffer_sha256_float64": buffer_sha256(ready_values),
        "provider_layer_identity_check": "max(abs(asinh(c_counts) - c_counts_asinh)) < 1e-12",
    }


def backed_rows(matrix: object, positions: np.ndarray, columns: np.ndarray) -> np.ndarray:
    order = np.argsort(positions)
    sorted_positions = positions[order]
    if len(np.unique(sorted_positions)) != len(sorted_positions):
        raise AssertionError("Monkman H5AD positions are not unique")
    # Backed AnnData/HDF5 matrices do not reliably support two simultaneous
    # advanced-index arrays. Slice rows first, then select marker columns in
    # memory so this is unambiguously a cells-by-markers extraction.
    sorted_values = np.asarray(matrix[sorted_positions, :], dtype=np.float64)[:, columns]
    expected_shape = (len(sorted_positions), len(columns))
    if sorted_values.shape != expected_shape:
        raise AssertionError(
            f"Backed matrix extraction shape mismatch: {sorted_values.shape} != {expected_shape}"
        )
    values = np.empty_like(sorted_values)
    values[order] = sorted_values
    return values


def build_monkman(
    block: pd.DataFrame, expected_raw: np.ndarray
) -> tuple[pd.DataFrame, dict[str, object]]:
    source = RAW_ROOT / "Monkman2024"
    h5ad_path = source / "anndata_4301_v4_annotated.h5ad"
    csv_path = source / "4301_cells.csv"
    obj = ad.read_h5ad(h5ad_path, backed="r")
    try:
        if obj.shape != (210_945, 27) or "Object ID" not in obj.obs:
            raise AssertionError(f"Monkman H5AD contract changed: {obj.shape}")
        object_ids = pd.Index(obj.obs["Object ID"].astype(str))
        if object_ids.has_duplicates:
            raise AssertionError("Monkman H5AD Object ID is not unique")
        expected_ids = block["source_cell_id"].astype(str)
        positions = object_ids.get_indexer(expected_ids)
        if np.any(positions < 0) or len(np.unique(positions)) != len(block):
            raise AssertionError("Monkman provider identities differ from frozen cells")
        native_markers = [str(value) for value in obj.var_names]
        columns = np.asarray(
            [native_markers.index(MONKMAN_NATIVE[marker]) for marker in MARKERS],
            dtype=np.int64,
        )
        ready_values = backed_rows(obj.X, positions, columns)
    finally:
        obj.file.close()

    raw_columns = ["Object ID", *dict.fromkeys(MONKMAN_NATIVE.values())]
    raw = pd.read_csv(csv_path, usecols=raw_columns, dtype={"Object ID": str}, low_memory=False)
    if raw["Object ID"].duplicated().any():
        raise AssertionError("Monkman raw Object ID is not unique")
    raw = raw.set_index("Object ID").reindex(expected_ids)
    if raw.isna().any().any():
        raise AssertionError("Frozen Monkman cells are absent from raw source CSV")
    raw_values = np.column_stack([
        raw[MONKMAN_NATIVE[marker]].to_numpy(np.float64) for marker in MARKERS
    ])
    raw_error = float(np.max(np.abs(raw_values - expected_raw)))
    if raw_error != 0.0:
        raise AssertionError(f"Monkman canonical raw reconstruction error: {raw_error}")
    return ready_frame(block, ready_values), {
        "source_files": {
            "annotated_h5ad": source_record(h5ad_path),
            "raw_cell_csv": source_record(csv_path),
        },
        "raw_reconstruction_maximum_absolute_error": raw_error,
        "source_buffer_sha256_float64": buffer_sha256(raw_values),
        "analysis_ready_buffer_sha256_float64": buffer_sha256(ready_values),
    }


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    selected, canonical_raw, blocks, canonical_provenance = canonical_locks()
    records: dict[str, dict[str, object]] = {}
    tables: dict[str, pd.DataFrame] = {}
    for dataset, builder in (
        ("Senosain CyTOF", build_senosain),
        ("Cords IMC", build_cords),
        ("Monkman CODEX", build_monkman),
    ):
        block = blocks[dataset]
        expected = canonical_raw[block["row_id"].to_numpy(np.int64)]
        tables[dataset], records[dataset] = builder(block, expected)

    output_paths = {
        "Senosain CyTOF": DATA_DIR / "senosain_cytof_analysis_ready.csv.gz",
        "Cords IMC": DATA_DIR / "cords_imc_analysis_ready.csv.gz",
        "Monkman CODEX": DATA_DIR / "monkman_codex_analysis_ready.csv.gz",
    }
    for dataset in DATASETS:
        deterministic_csv(tables[dataset], output_paths[dataset])
    combined = pd.concat([tables[dataset] for dataset in DATASETS], ignore_index=True)
    if len(combined) != EXPECTED_TOTAL or not combined["cell_id"].is_unique:
        raise AssertionError("Combined analysis-ready matrix violates frozen cell contract")
    if not np.array_equal(combined["cell_id"].to_numpy(), selected["cell_id"].to_numpy()):
        raise AssertionError("Combined analysis-ready cell order differs from selected_cells")
    for column in ("dataset", "donor", "unit_key"):
        if not np.array_equal(combined[column].astype(str).to_numpy(), selected[column].astype(str).to_numpy()):
            raise AssertionError(f"Combined analysis-ready {column} differs from selected_cells")
    combined_path = DATA_DIR / "combined_analysis_ready.csv.gz"
    deterministic_csv(combined, combined_path)

    audit_rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        for marker in MARKERS:
            values = tables[dataset][marker].to_numpy(np.float64)
            audit_rows.append({
                "dataset": dataset, "marker": marker, "cells": len(values),
                "minimum": float(values.min()), "median": float(np.median(values)),
                "mean": float(values.mean()), "maximum": float(values.max()),
                "standard_deviation": float(values.std(ddof=1)),
                "exact_zero_sum": bool(values.sum() == 0),
            })
    scale_audit_path = HERE / "input_scale_audit.csv"
    pd.DataFrame(audit_rows).to_csv(scale_audit_path, index=False)

    marker_file = HERE / "markers.txt"
    harmonization_file = HERE / "marker_harmonization_manifest.csv"
    selection_file = BENCHMARK / "selected_cells.csv.gz"
    if marker_file.read_text().splitlines() != MARKERS:
        raise AssertionError("Local correction marker file changed")
    script_hash = sha256(Path(__file__))
    dataset_entries = [
        {
            "dataset": "Senosain CyTOF",
            "technology": "CyTOF",
            "input_matrix_file": "data/senosain_cytof_analysis_ready.csv.gz",
            "input_matrix_sha256": sha256(output_paths["Senosain CyTOF"]),
            "source_state": "technically_normalized_untransformed",
            "assay_or_layer": "senosain_140k_raw35.npy extracted from provider normalized_denoised FCS events",
            "processing_steps": [
                "Retain the exact 20,000 frozen Senosain cells and row order.",
                "Extract the eight shared biological markers from the provider-normalized and denoised 35-marker FCS view.",
                "Apply asinh(x/5) exactly once; apply no KMEANSLOW, z-score, or data-derived cofactor.",
            ],
            "cycombine_input_scale": "asinh protein intensity with cofactor 5",
            "transform_formula": "asinh(x / 5)",
            "transform_scope": "Element-wise on all eight shared markers for every frozen Senosain cell; selection order does not affect the transform.",
            "processing_reference": "Senosain et al. 2023 Methods: bead-normalized/denoised CyTOF and raw intensity 10 mapping to arcsinh 1.44, which identifies cofactor 5.",
            "processing_code_file": "build_inputs.py",
            "processing_code_sha256": script_hash,
            "double_transform_checked": True,
        },
        {
            "dataset": "Cords IMC",
            "technology": "IMC",
            "input_matrix_file": "data/cords_imc_analysis_ready.csv.gz",
            "input_matrix_sha256": sha256(output_paths["Cords IMC"]),
            "source_state": "normalized_and_nonlinearly_transformed",
            "assay_or_layer": "sce_all_annotated.rds assay c_counts_asinh",
            "processing_steps": [
                "Retain the exact 20,000 frozen Cords cells and row order by matching source_cell_id to SCE column names.",
                "Extract provider c_counts_asinh values for the eight harmonized shared markers.",
                "Use the provider full-cohort global p99 censoring and asinh cofactor-1 layer directly; do not re-estimate clipping on the frozen subset.",
            ],
            "cycombine_input_scale": "Provider globally p99-censored IMC counts after asinh cofactor 1",
            "transform_formula": "Provider c_counts_asinh = asinh(provider c_counts / 1); no additional transform",
            "transform_scope": "Global provider dataset before frozen-cell selection, as stored in the annotated SCE.",
            "processing_reference": "Cords et al. 2024 Segmentation pipeline: raw counts censored at the 99th percentile and arcsinh transformed with cofactor 1; provider c_counts_asinh assay.",
            "processing_code_file": "build_inputs.py",
            "processing_code_sha256": script_hash,
            "double_transform_checked": True,
        },
        {
            "dataset": "Monkman CODEX",
            "technology": "CODEX multiplexed fluorescence imaging",
            "input_matrix_file": "data/monkman_codex_analysis_ready.csv.gz",
            "input_matrix_sha256": sha256(output_paths["Monkman CODEX"]),
            "source_state": "provider_analysis_scale",
            "assay_or_layer": "anndata_4301_v4_annotated.h5ad X provider analysis layer",
            "processing_steps": [
                "Retain the exact 20,000 frozen Monkman cells and row order by matching source_cell_id to H5AD obs Object ID.",
                "Extract the eight harmonized shared-marker columns from provider H5AD X.",
                "Preserve source-study arcsinh cofactor 150, marker-wise scaling, and cell-wise scaling; do not repeat or recompute them on eight markers.",
            ],
            "cycombine_input_scale": "Provider CODEX analysis scale after asinh cofactor 150, marker-wise scaling, and cell-wise scaling",
            "transform_formula": "Provider H5AD X passthrough; source workflow arcsinh(x/150), marker scaling, then cell scaling",
            "transform_scope": "Computed by the source study on its complete retained marker panel before frozen-cell selection and shared-marker subsetting.",
            "processing_reference": "Monkman et al. 2024 Cell classification Methods and provider annotated H5AD X.",
            "processing_code_file": "build_inputs.py",
            "processing_code_sha256": script_hash,
            "double_transform_checked": True,
        },
    ]
    preprocessing_manifest = {
        "schema_version": "final_xu.cycombine_preprocessing.v2",
        "combined_input_sha256": sha256(combined_path),
        "shared_marker_policy": "Exact ordered eight-protein intersection frozen before method fitting; no non-shared marker, label, outcome, or spatial variable enters correction.",
        "correction_marker_file": "markers.txt",
        "correction_marker_sha256": sha256(marker_file),
        "marker_harmonization_manifest_file": "marker_harmonization_manifest.csv",
        "marker_harmonization_manifest_sha256": sha256(harmonization_file),
        "cell_selection_manifest_file": "../../../../benchmark_data/lung_cancer/selected_cells.csv.gz",
        "cell_selection_manifest_sha256": sha256(selection_file),
        "cell_selection_id_column": "cell_id",
        "datasets": dataset_entries,
    }
    preprocessing_manifest_path = HERE / "preprocessing_manifest.json"
    deterministic_json(preprocessing_manifest, preprocessing_manifest_path)

    paper_paths = [
        RAW_ROOT / "Senosain_2025_CyTOF_and_omics" / "Senosain_2023.pdf",
        RAW_ROOT / "Cords2024" / "Paper_Cords_2024.pdf",
        RAW_ROOT / "Monkman2024" / "Paper_Monkman2024.pdf",
    ]
    build_manifest = {
        "schema_version": "final_xu.lung_cancer.cycombine_input_build.v1",
        "status": "complete",
        "method": "cyCombine v0.2.5 canonical comparator",
        "labels_or_outcomes_read": False,
        "locked_cells": EXPECTED_TOTAL,
        "dataset_order": DATASETS,
        "dataset_cell_counts": {dataset: EXPECTED_PER_DATASET for dataset in DATASETS},
        "markers": MARKERS,
        "canonical_inputs": canonical_provenance,
        "datasets": [dict(dataset=dataset, **records[dataset]) for dataset in DATASETS],
        "papers": [source_record(path) for path in paper_paths],
        "outputs": {
            **{
                output_paths[dataset].name: source_record(output_paths[dataset])
                for dataset in DATASETS
            },
            combined_path.name: source_record(combined_path),
            scale_audit_path.name: source_record(scale_audit_path),
        },
        "preprocessing_manifest": source_record(preprocessing_manifest_path),
        "script": source_record(Path(__file__)),
        "runtime": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "anndata": version("anndata"),
        },
    }
    deterministic_json(build_manifest, HERE / "preprocessing_build_manifest.json")
    print(
        f"Built lung-cancer cyCombine input: {len(combined):,} cells x "
        f"{len(MARKERS)} markers; output: {combined_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
