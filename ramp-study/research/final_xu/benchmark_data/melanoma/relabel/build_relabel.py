#!/usr/bin/env python3
"""Build post-hoc labels for the canonical 60,000-cell melanoma benchmark."""

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
SOURCE = ROOT / "input" / "locked_metadata.csv.gz"
PARENT_MANIFEST = ROOT / "manifest.json"
DATASETS = [
    "MEL_Hickey2023_CODEX",
    "MEL_Hoch2021_IMC",
    "MEL_Ptacek2021_MIBI",
]
SLUGS = {
    DATASETS[0]: "hickey_codex",
    DATASETS[1]: "hoch_imc",
    DATASETS[2]: "ptacek_mibi",
}
EXPECTED_CELLS = {dataset: 20_000 for dataset in DATASETS}
EXPECTED_UNITS = {DATASETS[0]: 6, DATASETS[1]: 17, DATASETS[2]: 23}
VERSION = "melanoma_provider_fine_v2_60000"
FINE_LABEL_ORDER = [
    "Non-joint", "Epithelial (non-joint)", "CD4 T", "CD8 T",
    "Other T / CD8−", "NK", "Endothelial", "Myeloid", "Stroma", "Tumor",
]
JOINT_LABEL_ORDER = [
    "Non-joint", "CD8− T", "CD8 T", "Endothelial", "Myeloid", "Stroma", "Tumor",
]
PALETTE = {
    "Non-joint": "#8C939B",
    "Epithelial (non-joint)": "#F2C14E",
    "CD4 T": "#00A6D6",
    "CD8 T": "#1F4E79",
    "Other T / CD8−": "#6F8FC5",
    "NK": "#008B4F",
    "Endothelial": "#7FBF3F",
    "Myeloid": "#E68613",
    "Stroma": "#9467BD",
    "Tumor": "#D62728",
}
DRAW_ORDER = [
    "Tumor", "Non-joint", "Stroma", "Endothelial", "Myeloid",
    "Other T / CD8−", "CD4 T", "CD8 T", "NK", "Epithelial (non-joint)",
]
MANAGED = [
    "data", "relabeled_metadata.csv.gz", "relabel_mapping_audit.csv",
    "relabel_counts.csv", "joint_relabel_counts.csv", "display_palette.csv",
    "manifest.json",
]
PARENT_STALE_MARKER = HERE / "PARENT_REBUILD_REQUIRED.txt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
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
    """Return consecutive dataset blocks, not merely first-seen categories."""
    values = series.astype(str).reset_index(drop=True)
    starts = values.ne(values.shift(fill_value="")).to_numpy()
    return values.loc[starts].tolist()


def relabel_hickey(source: pd.Series) -> np.ndarray:
    values = source.fillna("").astype(str).to_numpy(dtype=str)
    result = np.full(len(values), "Non-joint", dtype=object)
    result[np.isin(values, ["CD4+ T", "CD4+ Treg"])] = "CD4 T"
    result[np.char.find(values, "CD8") >= 0] = "CD8 T"
    result[values == "Endothelial"] = "Endothelial"
    result[np.isin(values, [
        "DC", "CD163+ CD206+ Macrophage", "CD68+ Macrophage", "Neutrophil",
    ])] = "Myeloid"
    result[np.isin(values, [
        "Stromal", "CD73+ Stromal", "FAP+ Fibroblast", "PDPN+ Stromal",
        "Smooth Muscle",
    ])] = "Stroma"
    result[np.char.find(values, "Tumor") >= 0] = "Tumor"
    result[values == "Epithelial"] = "Epithelial (non-joint)"
    result[values == "NK"] = "NK"
    return result.astype(str)


def relabel_hoch(source: pd.Series) -> np.ndarray:
    mapping = {
        "CD8+ T cell": "CD8 T",
        "CD8- T cell": "Other T / CD8−",
        "Vasculature": "Endothelial",
        "Macrophage": "Myeloid",
        "Neutrophil": "Myeloid",
        "Stroma": "Stroma",
        "Tumor": "Tumor",
    }
    return source.fillna("").astype(str).map(mapping).fillna("Non-joint").to_numpy(str)


def validate_ptacek(frame: pd.DataFrame) -> np.ndarray:
    broad = frame["provider_celltype_original"].fillna("").astype(str).to_numpy(str)
    frozen = frame["celltype"].astype(str).to_numpy(str)
    expected_broad = {
        "tumor": {"Tumor"},
        "myeloid": {"Myeloid"},
        "stroma": {"Stroma"},
        "T_cell": {"CD4 T", "CD8 T", "Other T / CD8−"},
        "B_cell": {"Non-joint"},
        "other": {"Non-joint"},
        "unknown": {"Non-joint"},
    }
    if not set(broad).issubset(expected_broad):
        raise AssertionError(f"unexpected Ptacek broad labels: {sorted(set(broad))}")
    for broad_label, allowed in expected_broad.items():
        observed = set(frozen[broad == broad_label])
        if not observed.issubset(allowed):
            raise AssertionError(
                f"Ptacek {broad_label} has incompatible frozen fine labels: {observed}"
            )
    return frozen


def derive_joint(fine: np.ndarray) -> np.ndarray:
    result = np.full(len(fine), "Non-joint", dtype=object)
    result[np.isin(fine, ["CD4 T", "Other T / CD8−"])] = "CD8− T"
    result[fine == "CD8 T"] = "CD8 T"
    for label in ("Endothelial", "Myeloid", "Stroma", "Tumor"):
        result[fine == label] = label
    return result.astype(str)


def palette_frame() -> pd.DataFrame:
    draw_rank = {label: rank for rank, label in enumerate(DRAW_ORDER)}
    rows = []
    for legend_rank, label in enumerate(FINE_LABEL_ORDER):
        if label == "Tumor":
            size, alpha, zorder = 0.52, 0.42, 1
        elif label == "Non-joint":
            size, alpha, zorder = 0.62, 0.28, 2
        elif label == "Epithelial (non-joint)":
            size, alpha, zorder = 0.76, 0.76, 20
        else:
            size, alpha, zorder = 0.76, 0.76, 3 + draw_rank[label]
        rows.append({
            "celltype": label,
            "color": PALETTE[label],
            "legend_order": legend_rank,
            "draw_order_bottom_to_top": draw_rank[label],
            "point_size": size,
            "alpha": alpha,
            "zorder": zorder,
        })
    return pd.DataFrame(rows)


def load_source() -> pd.DataFrame:
    if not SOURCE.is_file() or not PARENT_MANIFEST.is_file():
        raise FileNotFoundError("parent benchmark must be built first")
    parent = json.loads(PARENT_MANIFEST.read_text())
    if parent.get("package") != "melanoma_three_platform_benchmark_v2_60000":
        raise AssertionError("unexpected parent benchmark package")
    expected = parent["files_sha256"]["input/locked_metadata.csv.gz"]
    if sha256(SOURCE) != expected:
        raise AssertionError("parent locked metadata hash differs from manifest")
    frame = pd.read_csv(SOURCE, low_memory=False)
    if len(frame) != 60_000:
        raise AssertionError("expected 60,000 locked rows")
    if dataset_block_order(frame["dataset"]) != DATASETS:
        raise AssertionError("dataset blocks are not contiguous in canonical order")
    if frame.groupby("dataset", sort=False).size().astype(int).to_dict() != EXPECTED_CELLS:
        raise AssertionError("dataset cell counts changed")
    if frame.groupby("dataset", sort=False)["unit_key"].nunique().astype(int).to_dict() != EXPECTED_UNITS:
        raise AssertionError("dataset unit counts changed")
    if not np.array_equal(frame["row_id"].to_numpy(), np.arange(len(frame))):
        raise AssertionError("row order changed")
    return frame


def build(stage: Path, source: pd.DataFrame) -> None:
    data_dir = stage / "data"
    data_dir.mkdir(parents=True)
    parts = []
    for dataset in DATASETS:
        block = source.loc[source["dataset"].astype(str).eq(dataset)].copy()
        frozen = block["celltype"].astype(str).to_numpy(str)
        if dataset == DATASETS[0]:
            fine = relabel_hickey(block["provider_celltype_original"])
        elif dataset == DATASETS[1]:
            fine = relabel_hoch(block["provider_celltype_original"])
        else:
            fine = validate_ptacek(block)
        if not np.array_equal(fine, frozen):
            raise AssertionError(f"{dataset}: recomputed and frozen fine labels differ")
        joint = derive_joint(fine)
        if not np.array_equal(joint, block["celltype_joint_evaluation"].astype(str).to_numpy(str)):
            raise AssertionError(f"{dataset}: recomputed and frozen joint labels differ")
        answer = block.copy()
        answer["celltype"] = fine
        answer["celltype_joint_evaluation"] = joint
        answer["joint_evaluation"] = joint != "Non-joint"
        answer["is_nonjoint"] = np.isin(
            fine, ["Non-joint", "Epithelial (non-joint)"]
        )
        answer["celltype_relabel_version"] = VERSION
        parts.append(answer)
        write_csv_gz(
            answer,
            data_dir / f"{SLUGS[dataset]}_relabeled_metadata.csv.gz",
        )
    combined = pd.concat(parts, ignore_index=True)
    if not np.array_equal(combined["row_id"].to_numpy(), np.arange(len(combined))):
        raise AssertionError("combined relabel order changed")
    write_csv_gz(combined, data_dir / "melanoma_60000_relabeled_metadata.csv.gz")
    write_csv_gz(combined, stage / "relabeled_metadata.csv.gz")
    mapping = (
        combined.groupby(
            [
                "dataset", "provider_celltype_original", "celltype",
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
    outputs = [path for path in stage.rglob("*") if path.is_file()]
    manifest = {
        "version": VERSION,
        "purpose": "post-hoc provider-based labels for canonical melanoma 60,000-cell benchmark",
        "cells": 60_000,
        "datasets": DATASETS,
        "dataset_cells": EXPECTED_CELLS,
        "dataset_units": EXPECTED_UNITS,
        "fine_labels": FINE_LABEL_ORDER,
        "joint_labels": JOINT_LABEL_ORDER,
        "display_draw_order_bottom_to_top": DRAW_ORDER,
        "epithelial_color": PALETTE["Epithelial (non-joint)"],
        "expression_matrix_read": False,
        "ramp_or_umap_used_for_relabeling": False,
        "response_survival_timepoint_or_coordinates_used": False,
        "ptacek_tcell_policy": "upstream provider CD4/CD8 gates frozen before alignment; compatibility revalidated here",
        "ptacek_fine_label_recomputed_from_gates_here": False,
        "source": {
            "path": str(SOURCE),
            "sha256": sha256(SOURCE),
            "parent_manifest": str(PARENT_MANIFEST),
            "parent_manifest_sha256": sha256(PARENT_MANIFEST),
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
            str(path.relative_to(stage)): sha256(path) for path in sorted(outputs)
        },
    }
    (stage / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def validate(stage: Path) -> None:
    combined = pd.read_csv(stage / "relabeled_metadata.csv.gz", low_memory=False)
    if len(combined) != 60_000 or combined["cell_id"].astype(str).duplicated().any():
        raise AssertionError("invalid combined relabel output")
    manifest = json.loads((stage / "manifest.json").read_text())
    for relative, expected in manifest["output_sha256"].items():
        if sha256(stage / relative) != expected:
            raise AssertionError(f"relabel staged hash differs: {relative}")


def replace_directory(source: Path, target: Path) -> None:
    """Replace one managed directory while retaining the old copy on failure."""
    replacement = Path(tempfile.mkdtemp(prefix=f".{target.name}.new.", dir=target.parent))
    backup = target.parent / f".{target.name}.old.{os.getpid()}"
    try:
        shutil.copytree(source, replacement, dirs_exist_ok=True)
        if backup.exists():
            shutil.rmtree(backup)
        if target.exists():
            os.replace(target, backup)
        try:
            os.replace(replacement, target)
        except Exception:
            if backup.exists() and not target.exists():
                os.replace(backup, target)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if replacement.exists():
            shutil.rmtree(replacement)


def replace_file(source: Path, target: Path) -> None:
    """Publish a managed file with an atomic same-filesystem rename."""
    temporary = target.parent / f".{target.name}.new.{os.getpid()}"
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def publish(stage: Path) -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    replace_directory(stage / "data", HERE / "data")
    for name in MANAGED:
        if name in {"data", "manifest.json"}:
            continue
        replace_file(stage / name, HERE / name)
    replace_file(stage / "manifest.json", HERE / "manifest.json")


def main() -> None:
    source = load_source()
    with tempfile.TemporaryDirectory(
        prefix=".melanoma_relabel_", dir=ROOT
    ) as temporary:
        stage = Path(temporary) / "package"
        stage.mkdir(parents=True)
        build(stage, source)
        validate(stage)
        publish(stage)
        validate(HERE)
        if PARENT_STALE_MARKER.exists():
            PARENT_STALE_MARKER.unlink()
    print("Published melanoma 60,000-cell relabel package", flush=True)


if __name__ == "__main__":
    main()
