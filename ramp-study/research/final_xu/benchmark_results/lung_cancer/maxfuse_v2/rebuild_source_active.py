#!/usr/bin/env python3
"""Rebuild frozen lung full active panels directly from provider sources.

The recipe is independent of ``test0827`` and of all prior alignment, UMAP,
and relabel outputs.  It reads only provider protein measurements and the
canonical identity/row locks in ``benchmark_data/lung_cancer``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import anndata as ad
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
FINAL_XU = HERE.parents[2]
BENCHMARK = FINAL_XU / "benchmark_data" / "lung_cancer"
RAW_ROOT = FINAL_XU / "raw_data" / "lung_cancer"
OUTPUT = HERE / "source_active"
EXPECTED_CELLS = 20_000
SHARED = ["CD3", "CD4", "CD8", "CD31", "HLA_DR", "KI67", "VIMENTIN", "EPI"]
DATASETS = [
    ("Senosain CyTOF", "senosain_cytof"),
    ("Cords IMC", "cords_imc"),
    ("Monkman CODEX", "monkman_codex"),
]
SENOSAIN_SHARED_NATIVE = {
    "CD3": "CD3", "CD4": "CD4", "CD8": "CD8", "CD31": "CD31",
    "HLA_DR": "HLA_DR", "KI67": "KI67", "VIMENTIN": "VIMENTIN",
    "EPI": "CYTOKERATIN",
}
CORDS_SHARED_NATIVE = {
    "CD3": "CD3", "CD4": "CD4", "CD8": "CD8a", "CD31": "vWF + CD31",
    "HLA_DR": "HLA-DR", "KI67": "Ki-67", "VIMENTIN": "Vimentin",
    "EPI": "Pan Cytokeratin + Keratin Epithelial",
}
MONKMAN_SHARED_NATIVE = {
    "CD3": "CD3e", "CD4": "CD4", "CD8": "CD8", "CD31": "CD31",
    "HLA_DR": "HLA-DR", "KI67": "Ki67", "VIMENTIN": "Vimentin",
    "EPI": "PanCK",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def buffer_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(values, dtype=np.float64).tobytes()
    ).hexdigest()


def deterministic_csv(frame: pd.DataFrame, path: Path) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
                frame.to_csv(text, index=False)


def source_identity(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "bytes": int(stat.st_size),
        "sha256": sha256(path),
    }


def publish(stage: Path, output: Path) -> None:
    if not output.exists():
        stage.rename(output)
        return
    backup = output.with_name(f".{output.name}.old-{next(tempfile._get_candidate_names())}")
    output.rename(backup)
    try:
        stage.rename(output)
    except BaseException:
        backup.rename(output)
        raise
    shutil.rmtree(backup)


def read_locks() -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray], dict[str, object]]:
    selected_path = BENCHMARK / "selected_cells.csv.gz"
    raw_path = BENCHMARK / "input" / "raw.npy"
    markers_path = BENCHMARK / "input" / "markers.txt"
    columns = [
        "row_id", "cell_id", "source_cell_id", "source_row_zero_based",
        "dataset", "donor", "unit_key",
    ]
    selected = pd.read_csv(
        selected_path,
        usecols=columns,
        dtype={"cell_id": str, "source_cell_id": str, "dataset": str},
        low_memory=False,
    )
    frozen = np.load(raw_path, mmap_mode="r", allow_pickle=False)
    markers = [line for line in markers_path.read_text().splitlines() if line]
    if markers != SHARED or frozen.shape != (60_000, len(SHARED)):
        raise AssertionError("lung canonical shared-marker contract changed")
    if not np.array_equal(selected["row_id"].to_numpy(), np.arange(60_000)):
        raise AssertionError("lung canonical row lock changed")
    if selected["cell_id"].duplicated().any() or selected["source_cell_id"].duplicated().any():
        raise AssertionError("lung canonical cell IDs are not unique")
    order = selected.loc[
        selected["dataset"].ne(selected["dataset"].shift(fill_value="")), "dataset"
    ].tolist()
    if order != [dataset for dataset, _ in DATASETS]:
        raise AssertionError(f"lung dataset block order changed: {order}")

    blocks: dict[str, pd.DataFrame] = {}
    shared: dict[str, np.ndarray] = {}
    for dataset, slug in DATASETS:
        block = selected.loc[selected["dataset"].eq(dataset)].reset_index(drop=True)
        if len(block) != EXPECTED_CELLS:
            raise AssertionError(f"{dataset}: canonical cell count changed")
        inspectable_path = BENCHMARK / slug / "alignment_input_8.csv.gz"
        inspectable_ids = pd.read_csv(
            inspectable_path, usecols=["cell_id"], dtype=str
        )["cell_id"].to_numpy()
        if not np.array_equal(inspectable_ids, block["cell_id"].to_numpy()):
            raise AssertionError(f"{dataset}: inspectable/canonical row order differs")
        rows = block["row_id"].to_numpy(dtype=int)
        blocks[dataset] = block
        shared[dataset] = np.asarray(frozen[rows], dtype=np.float64)
    audit = {
        "selected_cells": source_identity(selected_path),
        "raw_shared": source_identity(raw_path),
        "shared_markers": source_identity(markers_path),
        "inspectable_row_locks": {
            slug: source_identity(BENCHMARK / slug / "alignment_input_8.csv.gz")
            for _, slug in DATASETS
        },
    }
    return blocks, shared, audit


def published_markers(
    native_markers: list[str], shared_native: dict[str, str]
) -> list[str]:
    native_to_shared = {native: canonical for canonical, native in shared_native.items()}
    markers = [native_to_shared.get(marker, marker) for marker in native_markers]
    if len(set(markers)) != len(markers):
        raise AssertionError("canonicalized active marker names collide")
    return markers


def save_dataset(
    *,
    dataset: str,
    slug: str,
    values: np.ndarray,
    native_markers: list[str],
    shared_native: dict[str, str],
    observed_source_ids: np.ndarray,
    canonical: pd.DataFrame,
    canonical_raw_reconstructed: np.ndarray,
    expected_canonical_raw: np.ndarray,
    source_scale: str,
    excluded_features: list[str],
    source_files: dict[str, dict[str, object]],
    stage: Path,
) -> dict[str, object]:
    values = np.asarray(values, dtype=np.float64)
    markers = published_markers(native_markers, shared_native)
    if values.shape != (EXPECTED_CELLS, len(markers)):
        raise AssertionError(f"{dataset}: source-active shape changed: {values.shape}")
    if not np.isfinite(values).all() or np.any(values.std(axis=0, ddof=1) < 1e-10):
        raise AssertionError(f"{dataset}: invalid or constant source feature")
    observed_source_ids = np.asarray(observed_source_ids, dtype=str)
    expected_source_ids = canonical["source_cell_id"].astype(str).to_numpy()
    if not np.array_equal(observed_source_ids, expected_source_ids):
        mismatch = int(np.flatnonzero(observed_source_ids != expected_source_ids)[0])
        raise AssertionError(f"{dataset}: source identity mismatch at row {mismatch}")
    canonical_raw_reconstructed = np.asarray(canonical_raw_reconstructed, dtype=np.float64)
    maximum_error = float(
        np.max(np.abs(canonical_raw_reconstructed - expected_canonical_raw))
    )
    if maximum_error > 1e-8:
        raise AssertionError(
            f"{dataset}: provider/canonical raw shared reconstruction error {maximum_error}"
        )
    if any(marker not in markers for marker in SHARED):
        raise AssertionError(f"{dataset}: source panel lacks a shared marker")

    destination = stage / slug
    destination.mkdir()
    np.save(destination / "active_source_float64.npy", values, allow_pickle=False)
    (destination / "active_markers.txt").write_text(
        "".join(f"{marker}\n" for marker in markers)
    )
    deterministic_csv(
        canonical[["cell_id", "source_cell_id"]].astype(str),
        destination / "cell_ids.csv.gz",
    )
    outputs = {
        path.name: sha256(path)
        for path in destination.iterdir()
        if path.is_file()
    }
    return {
        "dataset": dataset,
        "slug": slug,
        "cells": EXPECTED_CELLS,
        "active_features": len(markers),
        "markers": markers,
        "native_markers": native_markers,
        "source_scale": source_scale,
        "excluded_source_features": excluded_features,
        "maximum_canonical_raw_shared_reconstruction_error": maximum_error,
        "source_buffer_sha256_float64": buffer_sha256(values),
        "source_files": source_files,
        "outputs": outputs,
    }


def build_senosain(
    canonical: pd.DataFrame, expected_raw: np.ndarray, stage: Path
) -> dict[str, object]:
    dataset, slug = DATASETS[0]
    source = RAW_ROOT / "Senosain_2025_CyTOF_and_omics" / "annotation"
    matrix_path = source / "senosain_140k_raw35.npy"
    marker_path = source / "senosain_140k_markers35.csv"
    metadata_path = source / "senosain_140k_cell_metadata.csv.gz"
    matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
    marker_table = pd.read_csv(marker_path, low_memory=False).sort_values(
        "raw35_column_zero_based"
    )
    if not np.array_equal(
        marker_table["raw35_column_zero_based"].to_numpy(dtype=int),
        np.arange(len(marker_table)),
    ):
        raise AssertionError("Senosain marker indices are not contiguous")
    native_all = marker_table["marker"].astype(str).tolist()
    if matrix.shape != (140_000, 35) or len(native_all) != 35:
        raise AssertionError(f"Senosain source shape changed: {matrix.shape}")
    numeric = pd.to_numeric(canonical["source_row_zero_based"], errors="raise")
    rows = numeric.to_numpy(dtype=np.int64)
    if not np.array_equal(numeric.to_numpy(np.float64), rows.astype(np.float64)):
        raise AssertionError("Senosain row locks are not integral")
    if rows.min() < 0 or rows.max() >= len(matrix) or len(np.unique(rows)) != EXPECTED_CELLS:
        raise AssertionError("Senosain row locks are invalid")
    source_ids = pd.read_csv(
        metadata_path,
        usecols=["sampled_row_zero_based", "source_cell_id"],
        dtype={"source_cell_id": str},
    ).sort_values("sampled_row_zero_based")
    if not np.array_equal(
        source_ids["sampled_row_zero_based"].to_numpy(), np.arange(140_000)
    ):
        raise AssertionError("Senosain 140k identity table row order changed")
    observed_ids = source_ids.iloc[rows]["source_cell_id"].to_numpy()
    selected_all = np.asarray(matrix[rows], dtype=np.float64)
    keep = [index for index, marker in enumerate(native_all) if marker != "CD90"]
    native_markers = [native_all[index] for index in keep]
    if len(native_markers) != 34:
        raise AssertionError("Senosain active panel must contain 34 proteins")
    native_index = {marker: index for index, marker in enumerate(native_all)}
    reconstructed = np.column_stack([
        selected_all[:, native_index[SENOSAIN_SHARED_NATIVE[marker]]]
        for marker in SHARED
    ])
    return save_dataset(
        dataset=dataset,
        slug=slug,
        values=selected_all[:, keep],
        native_markers=native_markers,
        shared_native=SENOSAIN_SHARED_NATIVE,
        observed_source_ids=observed_ids,
        canonical=canonical,
        canonical_raw_reconstructed=reconstructed,
        expected_canonical_raw=expected_raw,
        source_scale="provider-normalized and denoised FCS event intensities",
        excluded_features=["CD90: not uniformly measured across the provider cohort"],
        source_files={
            "raw35": source_identity(matrix_path),
            "marker_map": source_identity(marker_path),
            "identity_lock": source_identity(metadata_path),
        },
        stage=stage,
    )


CORDS_EXPORT_R = r'''
suppressPackageStartupMessages({
  library(SingleCellExperiment)
  library(data.table)
})
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args) == 4L)
sce_path <- args[[1L]]
identity_path <- args[[2L]]
active_path <- args[[3L]]
raw_shared_path <- args[[4L]]
identity <- fread(identity_path, colClasses=list(character="source_cell_id"))
stopifnot(names(identity) == "source_cell_id", nrow(identity) == 20000L)
sce <- readRDS(sce_path)
positions <- match(identity$source_cell_id, colnames(sce))
stopifnot(!anyNA(positions), uniqueN(positions) == 20000L)
features <- setdiff(rownames(sce), c("Iridium_191", "Iridium_193"))
stopifnot(length(features) == 41L)
active_native <- assay(sce, "c_counts_asinh")[features, positions, drop=FALSE]
censored <- assay(sce, "c_counts")[features, positions, drop=FALSE]
stopifnot(max(abs(asinh(censored) - active_native)) < 1e-12)
active_frame <- as.data.frame(t(active_native), check.names=FALSE)
colnames(active_frame) <- features
active_frame <- cbind(data.frame(source_cell_id=identity$source_cell_id), active_frame)
fwrite(active_frame, active_path, compress="gzip")
shared_native <- c(
  "CD3", "CD4", "CD8a", "vWF + CD31", "HLA-DR", "Ki-67",
  "Vimentin", "Pan Cytokeratin + Keratin Epithelial"
)
shared_names <- c("CD3", "CD4", "CD8", "CD31", "HLA_DR", "KI67", "VIMENTIN", "EPI")
raw_shared <- as.data.frame(t(assay(sce, "counts")[shared_native, positions, drop=FALSE]))
colnames(raw_shared) <- shared_names
raw_shared <- cbind(data.frame(source_cell_id=identity$source_cell_id), raw_shared)
fwrite(raw_shared, raw_shared_path, compress="gzip")
'''


def build_cords(
    canonical: pd.DataFrame, expected_raw: np.ndarray, stage: Path
) -> dict[str, object]:
    dataset, slug = DATASETS[1]
    path = RAW_ROOT / "Cords2024" / "SingleCellExperiment Objects" / "sce_all_annotated.rds"
    identity_path = stage / ".cords_identity.csv.gz"
    active_path = stage / ".cords_active.csv.gz"
    raw_path = stage / ".cords_raw_shared.csv.gz"
    deterministic_csv(canonical[["source_cell_id"]].astype(str), identity_path)
    completed = subprocess.run(
        ["Rscript", "-e", CORDS_EXPORT_R, str(path), str(identity_path), str(active_path), str(raw_path)],
        check=True,
        cwd=HERE,
        text=True,
        capture_output=True,
    )
    if completed.stdout.strip():
        print(completed.stdout.strip(), flush=True)
    if completed.stderr.strip():
        print(completed.stderr.strip(), flush=True)
    exported = pd.read_csv(active_path, dtype={"source_cell_id": str}, low_memory=False)
    raw_shared = pd.read_csv(raw_path, dtype={"source_cell_id": str}, low_memory=False)
    for temporary in (identity_path, active_path, raw_path):
        temporary.unlink()
    if not np.array_equal(
        exported["source_cell_id"].to_numpy(), raw_shared["source_cell_id"].to_numpy()
    ):
        raise AssertionError("Cords exported source ID order differs")
    native_markers = exported.columns[1:].astype(str).tolist()
    if len(native_markers) != 41:
        raise AssertionError("Cords active panel must contain 41 proteins")
    return save_dataset(
        dataset=dataset,
        slug=slug,
        values=exported.iloc[:, 1:].to_numpy(np.float64),
        native_markers=native_markers,
        shared_native=CORDS_SHARED_NATIVE,
        observed_source_ids=exported["source_cell_id"].to_numpy(),
        canonical=canonical,
        canonical_raw_reconstructed=raw_shared[SHARED].to_numpy(np.float64),
        expected_canonical_raw=expected_raw,
        source_scale="provider c_counts_asinh: global p99 censoring then asinh cofactor 1",
        excluded_features=["Iridium_191: DNA", "Iridium_193: DNA"],
        source_files={"annotated_sce": source_identity(path)},
        stage=stage,
    )


def backed_rows(matrix: object, positions: np.ndarray) -> np.ndarray:
    order = np.argsort(positions)
    sorted_positions = positions[order]
    if len(np.unique(sorted_positions)) != len(sorted_positions):
        raise AssertionError("Monkman H5AD positions are not unique")
    sorted_values = np.asarray(matrix[sorted_positions, :], dtype=np.float64)
    values = np.empty_like(sorted_values)
    values[order] = sorted_values
    return values


def build_monkman(
    canonical: pd.DataFrame, expected_raw: np.ndarray, stage: Path
) -> dict[str, object]:
    dataset, slug = DATASETS[2]
    source = RAW_ROOT / "Monkman2024"
    h5ad_path = source / "anndata_4301_v4_annotated.h5ad"
    csv_path = source / "4301_cells.csv"
    obj = ad.read_h5ad(h5ad_path, backed="r")
    try:
        if obj.shape != (210_945, 27) or "Object ID" not in obj.obs:
            raise AssertionError(f"Monkman H5AD contract changed: {obj.shape}")
        object_ids = pd.Index(obj.obs["Object ID"].astype(str))
        if object_ids.has_duplicates:
            raise AssertionError("Monkman Object ID is not unique")
        expected_ids = canonical["source_cell_id"].astype(str)
        positions = object_ids.get_indexer(expected_ids)
        if np.any(positions < 0) or len(np.unique(positions)) != EXPECTED_CELLS:
            raise AssertionError("Monkman canonical IDs do not match H5AD")
        native_all = [str(value) for value in obj.var_names]
        selected_all = backed_rows(obj.X, positions)
    finally:
        obj.file.close()
    if native_all.count("nsize") != 1:
        raise AssertionError("Monkman H5AD lacks expected nsize feature")
    keep = [index for index, marker in enumerate(native_all) if marker != "nsize"]
    native_markers = [native_all[index] for index in keep]
    if len(native_markers) != 26:
        raise AssertionError("Monkman active panel must contain 26 proteins")

    raw_columns = ["Object ID", *dict.fromkeys(MONKMAN_SHARED_NATIVE.values())]
    raw = pd.read_csv(
        csv_path, usecols=raw_columns, dtype={"Object ID": str}, low_memory=False
    )
    if raw["Object ID"].duplicated().any():
        raise AssertionError("Monkman raw Object ID is not unique")
    raw = raw.set_index("Object ID").reindex(expected_ids)
    if raw.isna().any().any():
        raise AssertionError("Monkman canonical IDs are missing from raw CSV")
    reconstructed = np.column_stack([
        raw[MONKMAN_SHARED_NATIVE[marker]].to_numpy(np.float64) for marker in SHARED
    ])
    return save_dataset(
        dataset=dataset,
        slug=slug,
        values=selected_all[:, keep],
        native_markers=native_markers,
        shared_native=MONKMAN_SHARED_NATIVE,
        observed_source_ids=expected_ids.to_numpy(),
        canonical=canonical,
        canonical_raw_reconstructed=reconstructed,
        expected_canonical_raw=expected_raw,
        source_scale="provider H5AD X after asinh c150, marker-wise and cell-wise scaling",
        excluded_features=["nsize: nuclear morphology"],
        source_files={
            "annotated_h5ad": source_identity(h5ad_path),
            "raw_cell_csv": source_identity(csv_path),
        },
        stage=stage,
    )


def verified_existing() -> bool:
    manifest_path = OUTPUT / "manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("recipe_sha256") != sha256(Path(__file__)):
        return False
    for relative, expected in manifest.get("outputs", {}).items():
        path = OUTPUT / relative
        if not path.is_file() or sha256(path) != expected:
            return False

    def identities_match(value: object) -> bool:
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                source = Path(str(value["path"]))
                if not source.is_file() or sha256(source) != value["sha256"]:
                    return False
            return all(identities_match(child) for child in value.values())
        if isinstance(value, list):
            return all(identities_match(child) for child in value)
        return True

    return identities_match(manifest.get("canonical_inputs", {})) and identities_match(
        manifest.get("datasets", [])
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="atomically rebuild verified derived files")
    args = parser.parse_args()
    if not args.force and verified_existing():
        print(f"verified existing source-active package: {OUTPUT}")
        return
    stage = Path(tempfile.mkdtemp(prefix=".source_active.stage-", dir=HERE))
    try:
        blocks, shared, canonical_audit = read_locks()
        records = [
            build_senosain(blocks[DATASETS[0][0]], shared[DATASETS[0][0]], stage),
            build_cords(blocks[DATASETS[1][0]], shared[DATASETS[1][0]], stage),
            build_monkman(blocks[DATASETS[2][0]], shared[DATASETS[2][0]], stage),
        ]
        outputs = {
            str(path.relative_to(stage)): sha256(path)
            for path in sorted(stage.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "schema_version": "final_xu.maxfuse_source_active.v2",
            "status": "complete",
            "benchmark": "lung_cancer",
            "selection": "exact canonical frozen cells and row order; no resampling",
            "labels_or_outcomes_read": False,
            "shared_markers": SHARED,
            "canonical_inputs": canonical_audit,
            "datasets": records,
            "recipe": str(Path(__file__).resolve()),
            "recipe_sha256": sha256(Path(__file__)),
            "outputs": outputs,
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        publish(stage, OUTPUT)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(f"rebuilt source-active package: {OUTPUT}")


if __name__ == "__main__":
    main()
