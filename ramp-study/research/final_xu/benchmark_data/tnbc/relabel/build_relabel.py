#!/usr/bin/env python3
"""Reproduce the provider-based broad TNBC cell-type harmonization.

Canonical benchmark inputs in the parent directory are read-only. This script
uses only provider cell type/subtype fields to generate post-hoc display labels;
marker expression, RAMP outputs, UMAP coordinates, outcomes, and the exploratory
SMA-based Stroma split are never used.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path


sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
BENCHMARK_ROOT = HERE.parent
DATA_DIR = HERE / "data"
VERSION = "tnbc_provider_broad_v2_2026-09-02"
EXPECTED_TOTAL = 62_701
DATASETS = (
    ("Tietscher IMC", "tietscher_imc", 20_000),
    ("Meyer IMC", "meyer_imc", 22_701),
    ("Shiao CODEX", "shiao_codex", 20_000),
)
MARKERS = [
    "CD11C", "CD20", "CD3", "CD31", "CD4", "CD68", "CD8",
    "HLA_DR", "KI67", "PANCK", "SMA",
]
LABEL_ORDER = (
    "Non-joint",
    "B lineage",
    "CD4 T",
    "CD8 T",
    "Endothelial",
    "Myeloid",
    "NK",
    "Stroma",
    "Tumor / epithelial",
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


def relabel_tietscher(celltype: np.ndarray, subtype: np.ndarray) -> np.ndarray:
    result = np.full(len(celltype), "", dtype=object)
    direct = {
        "tumor": "Tumor / epithelial",
        "myeloid": "Myeloid",
        "neutrophil": "Myeloid",
        "pDC": "Myeloid",
        "aDC": "Myeloid",
        "fibroblast": "Stroma",
        "stromal_undefined": "Stroma",
        "endothelial": "Endothelial",
        "B_cell": "B lineage",
        "plasma_cell": "B lineage",
    }
    for source, target in direct.items():
        result[celltype == source] = target
    t_cells = celltype == "T_NK"
    result[t_cells & np.isin(subtype, ["CD4", "Treg", "CD4_PD1"])] = "CD4 T"
    result[t_cells & np.isin(subtype, ["CD8", "CD8_PD1"])] = "CD8 T"
    result[t_cells & (subtype == "NK")] = "NK"
    return result.astype(str)


def relabel_meyer(celltype: np.ndarray, subtype: np.ndarray) -> np.ndarray:
    del subtype
    result = np.full(len(celltype), "", dtype=object)
    result[np.char.startswith(celltype.astype(str), "tumor_")] = "Tumor / epithelial"
    direct = {
        "Neutrophil": "Myeloid",
        "HLA_DR": "Myeloid",
        "Monocyte": "Myeloid",
        "Endothelial": "Endothelial",
        "CD8_T_cell": "CD8 T",
        "Stromal": "Stroma",
        "CD4_T_cell": "CD4 T",
        "B_cell": "B lineage",
        "Immune_low": "Non-joint",
    }
    for source, target in direct.items():
        result[celltype == source] = target
    return result.astype(str)


def relabel_shiao(celltype: np.ndarray, subtype: np.ndarray) -> np.ndarray:
    del subtype
    mapping = {
        "Epithelial": "Tumor / epithelial",
        "CD8T": "CD8 T",
        "Stromal": "Stroma",
        "Mac": "Myeloid",
        "CD4T": "CD4 T",
        "Plasma": "B lineage",
        "Bcell": "B lineage",
        "Endothelial": "Endothelial",
    }
    return np.asarray([mapping.get(value, "") for value in celltype], dtype=str)


def derive_labels(dataset: str, metadata: pd.DataFrame) -> np.ndarray:
    celltype = metadata["provider_celltype"].fillna("").astype(str).to_numpy()
    subtype = metadata["provider_cell_subtype"].fillna("").astype(str).to_numpy()
    if dataset == "Tietscher IMC":
        result = relabel_tietscher(celltype, subtype)
    elif dataset == "Meyer IMC":
        result = relabel_meyer(celltype, subtype)
    elif dataset == "Shiao CODEX":
        result = relabel_shiao(celltype, subtype)
    else:
        raise ValueError(dataset)
    missing = sorted(set(celltype[result == ""]))
    if missing:
        raise AssertionError(f"{dataset}: unmapped provider cell types {missing}")
    if not set(result).issubset(LABEL_ORDER):
        raise AssertionError(f"{dataset}: unexpected relabel output")
    return result


def build(stage: Path) -> dict[str, object]:
    temporary_data = stage / "data"
    combined_parts = []
    metadata_parts = []
    audit_parts = []
    source_files = []

    for dataset, slug, expected_cells in DATASETS:
        alignment_path = BENCHMARK_ROOT / slug / "alignment_input_11.csv.gz"
        metadata_path = BENCHMARK_ROOT / slug / "locked_metadata.csv.gz"
        alignment = pd.read_csv(
            alignment_path, low_memory=False, float_precision="round_trip"
        )
        metadata = pd.read_csv(metadata_path, low_memory=False)
        if len(alignment) != expected_cells or len(metadata) != expected_cells:
            raise AssertionError(f"{dataset}: row count changed")
        for column in ("cell_id", "dataset", "donor", "unit_key"):
            if not alignment[column].astype(str).equals(metadata[column].astype(str)):
                raise AssertionError(f"{dataset}: {column} row order changed")
        if alignment[MARKERS].isna().any().any():
            raise AssertionError(f"{dataset}: non-finite marker value")

        labels = derive_labels(dataset, metadata)
        frozen = metadata["celltype"].astype(str).to_numpy()
        display = metadata["display_label"].astype(str).to_numpy()
        if not np.array_equal(labels, frozen) or not np.array_equal(labels, display):
            raise AssertionError(f"{dataset}: explicit mapping differs from frozen broad label")

        relabeled = alignment.copy()
        relabeled["provider_label_l1"] = metadata["provider_celltype"].astype(str).to_numpy()
        relabeled["provider_label_l2"] = metadata["provider_cell_subtype"].astype(str).to_numpy()
        relabeled["celltype"] = labels
        relabeled["celltype_relabel_version"] = VERSION
        output_path = temporary_data / f"{slug}_relabeled.csv.gz"
        write_csv_gz(relabeled, output_path, float_format="%.17g")
        check = pd.read_csv(output_path, low_memory=False, float_precision="round_trip")
        if not alignment.equals(check[alignment.columns]):
            raise AssertionError(f"{dataset}: alignment values changed during relabeling")

        metadata_output = relabeled[
            [
                "cell_id", "source_cell_id", "dataset", "donor", "unit_key",
                "provider_label_l1", "provider_label_l2", "celltype",
                "celltype_relabel_version",
            ]
        ].copy()
        combined_parts.append(relabeled)
        metadata_parts.append(metadata_output)
        audit_parts.append(metadata_output)
        source_files.append({
            "dataset": dataset,
            "alignment_input": str(alignment_path.relative_to(BENCHMARK_ROOT)),
            "alignment_input_sha256": sha256(alignment_path),
            "locked_metadata": str(metadata_path.relative_to(BENCHMARK_ROOT)),
            "locked_metadata_sha256": sha256(metadata_path),
            "relabeled_output": str((DATA_DIR / output_path.name).relative_to(BENCHMARK_ROOT)),
            "relabeled_output_sha256": sha256(output_path),
        })

    combined = pd.concat(combined_parts, ignore_index=True)
    relabeled_metadata = pd.concat(metadata_parts, ignore_index=True)
    audit = pd.concat(audit_parts, ignore_index=True)
    if len(combined) != EXPECTED_TOTAL or combined["cell_id"].duplicated().any():
        raise AssertionError("Combined TNBC relabel output is invalid")
    if any(value in set(combined["celltype"]) for value in (
        "SMA-high CAF", "SMA-low Stroma", "Stromal / immune mix"
    )):
        raise AssertionError("Exploratory SMA-derived label leaked into benchmark")

    combined_path = temporary_data / "tnbc_62701_relabeled.csv.gz"
    metadata_path = stage / "relabeled_metadata.csv.gz"
    write_csv_gz(combined, combined_path, float_format="%.17g")
    write_csv_gz(relabeled_metadata, metadata_path)

    mapping = (
        audit.groupby(
            ["dataset", "provider_label_l1", "provider_label_l2", "celltype"],
            dropna=False, sort=False,
        )
        .size().rename("n_cells").reset_index()
        .sort_values(["dataset", "celltype", "provider_label_l1", "provider_label_l2"])
    )
    mapping.to_csv(stage / "relabel_mapping_audit.csv", index=False)
    counts = (
        audit.groupby(["celltype", "dataset"], observed=False)
        .size().unstack(fill_value=0).reindex(LABEL_ORDER, fill_value=0)
    )
    counts["Total"] = counts.sum(axis=1)
    counts.to_csv(stage / "relabel_counts.csv")

    outputs = sorted(path for path in stage.rglob("*") if path.is_file())
    manifest = {
        "version": VERSION,
        "purpose": "Provider-label-based broad TNBC harmonization",
        "cells": EXPECTED_TOTAL,
        "labels": list(LABEL_ORDER),
        "label_count": len(LABEL_ORDER),
        "provider_fields_used": ["provider_celltype", "provider_cell_subtype"],
        "marker_expression_used_for_relabeling": False,
        "ramp_or_umap_used_for_relabeling": False,
        "outcome_or_spatial_fields_used_for_relabeling": False,
        "exploratory_sma_stroma_relabel_included": False,
        "canonical_parent_inputs_modified": False,
        "source_files": source_files,
        "recipe": {
            "path": "relabel/build_relabel.py",
            "sha256": sha256(Path(__file__)),
        },
        "software_versions": {
            "Python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "output_sha256": {
            str(path.relative_to(stage)): sha256(path) for path in outputs
        },
    }
    return manifest


def publish(stage: Path, manifest: dict[str, object]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for path in (stage / "data").iterdir():
        os.replace(path, DATA_DIR / path.name)
    stale_combined = DATA_DIR / "tnbc_65211_relabeled.csv.gz"
    if stale_combined.exists():
        stale_combined.unlink()
    for name in (
        "relabeled_metadata.csv.gz", "relabel_mapping_audit.csv", "relabel_counts.csv"
    ):
        os.replace(stage / name, HERE / name)
    (HERE / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    stage = Path(tempfile.mkdtemp(prefix=".tnbc-relabel-build-", dir=HERE))
    try:
        manifest = build(stage)
        publish(stage, manifest)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    counts = pd.read_csv(HERE / "relabel_counts.csv")
    print(counts.to_string(index=False))
    print(f"\nSaved {EXPECTED_TOTAL:,} cells with {len(LABEL_ORDER)} broad labels to {DATA_DIR}")


if __name__ == "__main__":
    main()
