#!/usr/bin/env python3
"""Build deterministic post-hoc labels for the 60,000-cell lung benchmark."""

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
ROOT = HERE.parent
SOURCE = ROOT / "input/locked_metadata.csv.gz"
PARENT_MANIFEST = ROOT / "manifest.json"
DATASETS = ["Senosain CyTOF", "Cords IMC", "Monkman CODEX"]
SLUGS = {
    "Senosain CyTOF": "senosain_cytof",
    "Cords IMC": "cords_imc",
    "Monkman CODEX": "monkman_codex",
}
EXPECTED_CELLS = {dataset: 20_000 for dataset in DATASETS}
EXPECTED_UNITS = {dataset: 10 for dataset in DATASETS}
VERSION = "lung_broad_display_v1_60000"
FINE_LABEL_ORDER = [
    "Tumor/Epithelial", "CD4 T", "CD8 T", "B-like",
    "Myeloid", "Endothelial", "Mix",
]
JOINT_LABEL_ORDER = [
    "Tumor/Epithelial", "CD4 T", "CD8 T", "B-like",
    "Myeloid", "Endothelial", "Non-joint",
]
PALETTE = {
    "Tumor/Epithelial": "#D73027",
    "CD4 T": "#E78AC3",
    "CD8 T": "#D95F02",
    "B-like": "#1F78B4",
    "Myeloid": "#E6AB02",
    "Endothelial": "#17BECF",
    "Mix": "#9CA3AF",
}
DRAW_ORDER = [
    "Tumor/Epithelial", "Mix", "B-like", "CD4 T",
    "CD8 T", "Endothelial", "Myeloid",
]
MANAGED = [
    "data", "relabeled_metadata.csv.gz", "relabel_mapping_audit.csv",
    "relabel_counts.csv", "joint_relabel_counts.csv", "display_palette.csv",
    "manifest.json",
]
STALE_MARKER = HERE / "PARENT_REBUILD_REQUIRED.txt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv_gz(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="", fileobj=raw_handle, mode="wb", compresslevel=6, mtime=0
        ) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                frame.to_csv(text, index=False)


def dataset_block_order(series: pd.Series) -> list[str]:
    values = series.astype(str).reset_index(drop=True)
    starts = values.ne(values.shift(fill_value="")).to_numpy()
    return values.loc[starts].tolist()


def relabel_senosain(frame: pd.DataFrame) -> np.ndarray:
    allowed = set(FINE_LABEL_ORDER)
    result = frame["annotation_label_original"].astype(str).to_numpy(copy=True)
    unexpected = sorted(set(result) - allowed)
    if unexpected:
        raise AssertionError(f"Unexpected Senosain project labels: {unexpected}")
    refined = frame["c06_s01_reassigned_to_mix"].astype(bool).to_numpy()
    if int(refined.sum()) != 358:
        raise AssertionError("Expected exactly 358 C06/S01 display refinements")
    if not np.all(result[refined] == "CD4 T"):
        raise AssertionError("C06/S01 refinement no longer starts from CD4 T")
    result[refined] = "Mix"
    return result.astype(str)


def relabel_cords(frame: pd.DataFrame) -> np.ndarray:
    mapping = {
        "Bcell": "B-like",
        "Blood": "Endothelial",
        "CD4": "CD4 T",
        "CD8": "CD8 T",
        "HEV": "Endothelial",
        "Lymphatic": "Endothelial",
        "Myeloid": "Myeloid",
        "Neutrophil": "Myeloid",
        "normal": "Tumor/Epithelial",
        "hypoxic": "Tumor/Epithelial",
        "Collagen_CAF": "Mix",
        "IDO_CAF": "Mix",
        "Other": "Mix",
        "PDPN_CAF": "Mix",
        "SMA_CAF": "Mix",
        "dCAF": "Mix",
        "hypoxic_CAF": "Mix",
        "iCAF": "Mix",
        "mCAF": "Mix",
        "tpCAF": "Mix",
        "vCAF": "Mix",
    }
    source = frame["provider_cell_type"].astype(str)
    result = source.map(mapping)
    if result.isna().any():
        raise AssertionError(
            f"Unexpected Cords provider labels: {sorted(source.loc[result.isna()].unique())}"
        )
    return result.to_numpy(str)


def relabel_monkman(frame: pd.DataFrame) -> np.ndarray:
    mapping = {
        "Tumour": "Tumor/Epithelial",
        "HLADR Tumour": "Tumor/Epithelial",
        "CD44 Tumour": "Tumor/Epithelial",
        "Proliferating Tumour": "Tumor/Epithelial",
        "B Cells": "B-like",
        "CD4 Cells": "CD4 T",
        "Effector CD4": "CD4 T",
        "Treg": "CD4 T",
        "CD8 Cells": "CD8 T",
        "Blood Vessels": "Endothelial",
        "Lymphatics": "Endothelial",
        "Monocytes": "Myeloid",
        "Granulocytes": "Myeloid",
        "Macrophages": "Myeloid",
        "Mast Cells": "Myeloid",
        "Stroma": "Mix",
        "Unclassified": "Mix",
        "Proliferating Lymphocytes": "Mix",
        "Lymphocytes": "Mix",
        "CCR7+ CD8/CD4 Cells": "Mix",
        "Artifact": "Mix",
    }
    source = frame["provider_cell_type"].astype(str)
    result = source.map(mapping)
    if result.isna().any():
        raise AssertionError(
            f"Unexpected Monkman provider labels: {sorted(source.loc[result.isna()].unique())}"
        )
    return result.to_numpy(str)


def derive_joint(fine: np.ndarray) -> np.ndarray:
    return np.where(fine == "Mix", "Non-joint", fine).astype(str)


def palette_frame() -> pd.DataFrame:
    draw_rank = {label: rank for rank, label in enumerate(DRAW_ORDER)}
    rows = []
    for legend_rank, label in enumerate(FINE_LABEL_ORDER):
        rows.append({
            "celltype": label,
            "color": PALETTE[label],
            "legend_order": legend_rank,
            "draw_order_bottom_to_top": draw_rank[label],
            "point_size": 0.70 if label != "Mix" else 0.62,
            "alpha": 0.76 if label != "Mix" else 0.36,
            "zorder": draw_rank[label] + 1,
            "joint_evaluation_label": "Non-joint" if label == "Mix" else label,
        })
    return pd.DataFrame(rows)


def load_source() -> tuple[pd.DataFrame, dict]:
    if not SOURCE.is_file() or not PARENT_MANIFEST.is_file():
        raise FileNotFoundError("Parent lung benchmark must be built first")
    parent = json.loads(PARENT_MANIFEST.read_text())
    if parent.get("package") != "lung_three_platform_benchmark_v1_60000":
        raise AssertionError("Unexpected parent benchmark package")
    expected = parent["files_sha256"]["input/locked_metadata.csv.gz"]
    if sha256(SOURCE) != expected:
        raise AssertionError("Parent locked metadata hash differs from manifest")
    frame = pd.read_csv(SOURCE, dtype={"donor_id": str, "donor": str}, low_memory=False)
    if len(frame) != 60_000:
        raise AssertionError("Expected 60,000 locked rows")
    if dataset_block_order(frame["dataset"]) != DATASETS:
        raise AssertionError("Dataset block order changed")
    if frame.groupby("dataset", sort=False).size().astype(int).to_dict() != EXPECTED_CELLS:
        raise AssertionError("Dataset cell counts changed")
    if (
        frame.groupby("dataset", sort=False)["unit_key"].nunique().astype(int).to_dict()
        != EXPECTED_UNITS
    ):
        raise AssertionError("Dataset unit counts changed")
    if not np.array_equal(frame["row_id"].to_numpy(), np.arange(len(frame))):
        raise AssertionError("row_id changed")
    return frame, parent


def build(stage: Path, source: pd.DataFrame, parent: dict) -> None:
    data_dir = stage / "data"
    data_dir.mkdir(parents=True)
    parts = []
    for dataset in DATASETS:
        block = source.loc[source["dataset"].astype(str).eq(dataset)].copy()
        if dataset == "Senosain CyTOF":
            fine = relabel_senosain(block)
        elif dataset == "Cords IMC":
            fine = relabel_cords(block)
        else:
            fine = relabel_monkman(block)
        frozen = block["display_label"].astype(str).to_numpy()
        if not np.array_equal(fine, frozen):
            mismatch = np.flatnonzero(fine != frozen)[:10]
            raise AssertionError(
                f"{dataset}: recomputed display labels differ at {mismatch.tolist()}"
            )
        joint = derive_joint(fine)
        frozen_joint = block["celltype_joint_evaluation"].astype(str).to_numpy()
        if not np.array_equal(joint, frozen_joint):
            raise AssertionError(f"{dataset}: joint evaluation labels differ")
        block["celltype"] = fine
        block["celltype_joint_evaluation"] = joint
        block["joint_evaluation"] = joint != "Non-joint"
        block["is_mix_display"] = fine == "Mix"
        block["celltype_relabel_version"] = VERSION
        parts.append(block)
        write_csv_gz(
            block,
            data_dir / f"{SLUGS[dataset]}_relabeled_metadata.csv.gz",
        )

    combined = pd.concat(parts, ignore_index=True)
    if not np.array_equal(combined["row_id"].to_numpy(), np.arange(len(combined))):
        raise AssertionError("Combined relabel row order changed")
    write_csv_gz(combined, data_dir / "lung_cancer_60000_relabeled_metadata.csv.gz")
    write_csv_gz(combined, stage / "relabeled_metadata.csv.gz")

    mapping = (
        combined.groupby(
            [
                "dataset", "annotation_label_original", "provider_cell_category",
                "provider_cell_type", "provider_cell_subtype", "celltype",
                "celltype_joint_evaluation",
            ],
            sort=False,
            dropna=False,
        )
        .size().rename("cells").reset_index()
    )
    mapping.to_csv(stage / "relabel_mapping_audit.csv", index=False)
    (
        combined.groupby(["dataset", "celltype"], sort=False)
        .size().rename("cells").reset_index()
        .to_csv(stage / "relabel_counts.csv", index=False)
    )
    (
        combined.groupby(["dataset", "celltype_joint_evaluation"], sort=False)
        .size().rename("cells").reset_index()
        .to_csv(stage / "joint_relabel_counts.csv", index=False)
    )
    palette_frame().to_csv(stage / "display_palette.csv", index=False)

    outputs = sorted(path for path in stage.rglob("*") if path.is_file())
    manifest = {
        "version": VERSION,
        "purpose": "post-hoc display labels for canonical lung 60,000-cell benchmark",
        "cells": 60_000,
        "datasets": DATASETS,
        "dataset_cells": EXPECTED_CELLS,
        "dataset_units": EXPECTED_UNITS,
        "display_labels": FINE_LABEL_ORDER,
        "joint_labels": JOINT_LABEL_ORDER,
        "display_draw_order_bottom_to_top": DRAW_ORDER,
        "mix_policy": (
            "Mix is a display catch-all and maps to Non-joint for evaluation; "
            "358 Senosain C06/S01 cells are alignment-inspection-informed display refinements"
        ),
        "expression_matrix_read": False,
        "ramp_umap_metrics_or_outcomes_read": False,
        "labels_used_for_alignment": False,
        "source": {
            "path": str(SOURCE.resolve()),
            "sha256": sha256(SOURCE),
            "parent_manifest": str(PARENT_MANIFEST.resolve()),
            "parent_manifest_sha256": sha256(PARENT_MANIFEST),
            "parent_package": parent["package"],
        },
        "recipe": {
            "build_relabel.py": sha256(Path(__file__).resolve()),
            "README.md": sha256(HERE / "README.md"),
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
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def publish(stage: Path) -> None:
    for name in MANAGED:
        if name == "manifest.json":
            continue
        source = stage / name
        destination = HERE / name
        if destination.is_dir():
            shutil.rmtree(destination)
        elif destination.exists():
            destination.unlink()
        os.replace(source, destination)
    destination = HERE / "manifest.json"
    if destination.exists():
        destination.unlink()
    os.replace(stage / "manifest.json", destination)
    if STALE_MARKER.exists():
        STALE_MARKER.unlink()


def main() -> None:
    source, parent = load_source()
    with tempfile.TemporaryDirectory(
        prefix=".lung_relabel_stage_", dir=ROOT
    ) as temporary:
        stage = Path(temporary) / "relabel"
        stage.mkdir()
        build(stage, source, parent)
        publish(stage)
    print("Built lung relabel package: 60,000 rows", flush=True)


if __name__ == "__main__":
    main()
