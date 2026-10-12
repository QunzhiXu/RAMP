#!/usr/bin/env python3
"""Reproduce the provider-based melanoma benchmark-core relabeling.

Only frozen provider annotation fields and Ptacek provider gates are used to
derive labels. Expression values, RAMP outputs, UMAP coordinates, spatial
coordinates, treatment/response, and timepoint fields are never read.
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
BENCHMARK_ROOT = HERE.parents[1]
FINAL_XU = HERE.parents[3]
PROJECT_ROOT = HERE.parents[4]
DATA_DIR = HERE / "data"
PROVIDER_SOURCE = (
    PROJECT_ROOT / "final" / "melanoma" / "relabel"
    / "cell_metadata_relabeled.csv.gz"
)
FROZEN_ROW_SOURCE = (
    FINAL_XU / "test0827" / "melanoma" / "benchmark_core"
    / "input" / "metadata.csv.gz"
)
VERSION = "melanoma_provider_fine_v1_2026-08-27"
EXPECTED_TOTAL = 61_221
DATASETS = (
    ("MEL_Hickey2023_CODEX", "hickey_codex", 20_000, 2),
    ("MEL_Hoch2021_IMC", "hoch_imc", 20_000, 4),
    ("MEL_Ptacek2021_MIBI", "ptacek_mibi", 21_221, 20),
)

FINE_LABEL_ORDER = (
    "Non-joint",
    "Epithelial (non-joint)",
    "CD4 T",
    "CD8 T",
    "Other T / CD8−",
    "NK",
    "Endothelial",
    "Myeloid",
    "Stroma",
    "Tumor",
)
JOINT_LABEL_ORDER = (
    "Non-joint",
    "CD8− T",
    "CD8 T",
    "Endothelial",
    "Myeloid",
    "Stroma",
    "Tumor",
)

EXPECTED_FINE_COUNTS = {
    "MEL_Hickey2023_CODEX": {
        "Non-joint": 501,
        "Epithelial (non-joint)": 3_642,
        "CD4 T": 1_436,
        "CD8 T": 476,
        "Other T / CD8−": 0,
        "NK": 199,
        "Endothelial": 856,
        "Myeloid": 1_927,
        "Stroma": 4_072,
        "Tumor": 6_891,
    },
    "MEL_Hoch2021_IMC": {
        "Non-joint": 823,
        "Epithelial (non-joint)": 0,
        "CD4 T": 0,
        "CD8 T": 2_076,
        "Other T / CD8−": 3_302,
        "NK": 0,
        "Endothelial": 463,
        "Myeloid": 1_992,
        "Stroma": 880,
        "Tumor": 10_464,
    },
    "MEL_Ptacek2021_MIBI": {
        "Non-joint": 3_001,
        "Epithelial (non-joint)": 0,
        "CD4 T": 226,
        "CD8 T": 390,
        "Other T / CD8−": 144,
        "NK": 0,
        "Endothelial": 0,
        "Myeloid": 3_166,
        "Stroma": 1_671,
        "Tumor": 12_623,
    },
}

HICKEY_SOURCE_LABELS = {
    "B", "Bcatenin+ Tumor", "CD163+ CD206+ Macrophage", "CD4+ T",
    "CD4+ Treg", "CD56+ Tumor", "CD68+ Macrophage", "CD73+ Stromal",
    "CD8+ T", "DC", "EZH2+ Ki67+ Tumor", "EZH2+ Tumor", "Endothelial",
    "Epithelial", "FAP+ Fibroblast", "FAP+ Tumor", "GZMB+ CD8+ T",
    "GZMB+ PD1+ CD8+ T", "GZMB+ TCF1/7+ CD8+ T",
    "GZMB+ TCF1/7+ PD1+ CD8+ T", "LDHA+ Tumor",
    "MHCI+ PDL1+ Tumor", "MHCI+ Tumor", "MelanA+ Tumor", "NK", "Nerve",
    "Neutrophil", "PD1+ CD8+ T", "PD1+ TCF1/7+ CD8+ T",
    "PDL1+ CD56+ Tumor", "PDPN+ Stromal", "PTEN+ Tumor",
    "SOX10/S100+ Tumor", "Smooth Muscle", "Stromal", "TCF1/7+ CD8+ T",
    "Tumor",
}
HOCH_SOURCE_LABELS = {
    "CD38", "CD8+ T cell", "CD8- T cell", "HLA-DR", "Macrophage",
    "Neutrophil", "Stroma", "Tumor", "Vasculature", "unknown",
}
PTACEK_SOURCE_LABELS = {
    "B_cell", "T_cell", "myeloid", "other", "stroma", "tumor", "unknown",
}

PALETTE = {
    "Non-joint": "#8C939B",
    "Epithelial (non-joint)": "#C77C9D",
    "CD4 T": "#00A6D6",
    "CD8 T": "#1F4E79",
    "Other T / CD8−": "#6F8FC5",
    "NK": "#008B4F",
    "Endothelial": "#7FBF3F",
    "Myeloid": "#E68613",
    "Stroma": "#9467BD",
    "Tumor": "#D62728",
}
DRAW_ORDER_BOTTOM_TO_TOP = (
    "Tumor",
    "Non-joint",
    "Stroma",
    "Endothelial",
    "Myeloid",
    "Other T / CD8−",
    "CD4 T",
    "CD8 T",
    "NK",
    "Epithelial (non-joint)",
)


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


def true_mask(series: pd.Series) -> np.ndarray:
    return series.astype(str).str.lower().isin({"true", "1", "1.0"}).to_numpy()


def relabel_hickey(frame: pd.DataFrame) -> np.ndarray:
    source = frame["provider_celltype_original"].fillna("").astype(str).to_numpy(dtype=str)
    if set(source) != HICKEY_SOURCE_LABELS:
        raise AssertionError("Hickey provider-label vocabulary changed")
    result = np.full(len(frame), "Non-joint", dtype=object)
    result[np.isin(source, ["CD4+ T", "CD4+ Treg"])] = "CD4 T"
    result[np.char.find(source, "CD8") >= 0] = "CD8 T"
    result[source == "Endothelial"] = "Endothelial"
    result[np.isin(source, [
        "DC", "CD163+ CD206+ Macrophage", "CD68+ Macrophage", "Neutrophil",
    ])] = "Myeloid"
    result[np.isin(source, [
        "Stromal", "CD73+ Stromal", "FAP+ Fibroblast", "PDPN+ Stromal",
        "Smooth Muscle",
    ])] = "Stroma"
    result[np.char.find(source, "Tumor") >= 0] = "Tumor"
    result[source == "Epithelial"] = "Epithelial (non-joint)"
    result[source == "NK"] = "NK"
    return result.astype(str)


def relabel_hoch(frame: pd.DataFrame) -> np.ndarray:
    source = frame["provider_celltype_original"].fillna("").astype(str).to_numpy(dtype=str)
    if set(source) != HOCH_SOURCE_LABELS:
        raise AssertionError("Hoch provider-label vocabulary changed")
    mapping = {
        "CD8+ T cell": "CD8 T",
        "CD8- T cell": "Other T / CD8−",
        "Vasculature": "Endothelial",
        "Macrophage": "Myeloid",
        "Neutrophil": "Myeloid",
        "Stroma": "Stroma",
        "Tumor": "Tumor",
    }
    return np.asarray([mapping.get(value, "Non-joint") for value in source], dtype=str)


def relabel_ptacek(frame: pd.DataFrame) -> np.ndarray:
    source = frame["provider_celltype_original"].fillna("").astype(str).to_numpy(dtype=str)
    if set(source) != PTACEK_SOURCE_LABELS:
        raise AssertionError("Ptacek provider-label vocabulary changed")
    result = np.full(len(frame), "Non-joint", dtype=object)
    result[source == "tumor"] = "Tumor"
    result[source == "myeloid"] = "Myeloid"
    result[source == "stroma"] = "Stroma"
    t_cell = true_mask(frame["provider_t_cell_gate"])
    cd4 = true_mask(frame["provider_cd4_t_gate"])
    cd8 = true_mask(frame["provider_cd8_t_gate"])
    b_cell = true_mask(frame["provider_b_cell_gate"])
    if (int(t_cell.sum()), int(cd4.sum()), int(cd8.sum()), int(b_cell.sum())) != (
        760, 226, 390, 12
    ):
        raise AssertionError("Ptacek provider gate counts changed")
    if np.any(cd4 & cd8) or np.any((cd4 | cd8) & ~t_cell):
        raise AssertionError("Unexpected Ptacek provider T-cell gates")
    if not np.array_equal(t_cell, source == "T_cell"):
        raise AssertionError("Ptacek T-cell gate/source label mismatch")
    if not np.array_equal(b_cell, source == "B_cell"):
        raise AssertionError("Ptacek B-cell gate/source label mismatch")
    result[t_cell & ~cd4 & ~cd8] = "Other T / CD8−"
    result[cd4] = "CD4 T"
    result[cd8] = "CD8 T"
    return result.astype(str)


def derive_fine_labels(dataset: str, frame: pd.DataFrame) -> np.ndarray:
    if dataset == "MEL_Hickey2023_CODEX":
        result = relabel_hickey(frame)
    elif dataset == "MEL_Hoch2021_IMC":
        result = relabel_hoch(frame)
    elif dataset == "MEL_Ptacek2021_MIBI":
        result = relabel_ptacek(frame)
    else:
        raise ValueError(dataset)
    if not set(result).issubset(FINE_LABEL_ORDER):
        raise AssertionError(f"Unexpected fine labels for {dataset}")
    return result


def derive_joint_labels(fine: np.ndarray) -> np.ndarray:
    result = np.full(len(fine), "Non-joint", dtype=object)
    result[np.isin(fine, ["CD4 T", "Other T / CD8−"])] = "CD8− T"
    result[fine == "CD8 T"] = "CD8 T"
    for label in ("Endothelial", "Myeloid", "Stroma", "Tumor"):
        result[fine == label] = label
    return result.astype(str)


def palette_frame() -> pd.DataFrame:
    draw_rank = {label: rank for rank, label in enumerate(DRAW_ORDER_BOTTOM_TO_TOP)}
    rows = []
    for legend_rank, label in enumerate(FINE_LABEL_ORDER):
        if label == "Tumor":
            point_size, alpha, zorder = 0.52, 0.42, 1
        elif label == "Non-joint":
            point_size, alpha, zorder = 0.62, 0.28, 2
        elif label == "Epithelial (non-joint)":
            point_size, alpha, zorder = 0.76, 0.76, 20
        else:
            point_size, alpha, zorder = 0.76, 0.76, 3 + draw_rank[label]
        rows.append({
            "celltype": label,
            "color": PALETTE[label],
            "legend_order": legend_rank,
            "draw_order_bottom_to_top": draw_rank[label],
            "point_size": point_size,
            "alpha": alpha,
            "zorder": zorder,
        })
    return pd.DataFrame(rows)


def build(stage: Path) -> dict[str, object]:
    if not PROVIDER_SOURCE.is_file() or not FROZEN_ROW_SOURCE.is_file():
        raise FileNotFoundError("Frozen melanoma metadata source is missing")
    source = pd.read_csv(PROVIDER_SOURCE, low_memory=False)
    row_lock = pd.read_csv(FROZEN_ROW_SOURCE, low_memory=False)
    if len(source) != EXPECTED_TOTAL or len(row_lock) != EXPECTED_TOTAL:
        raise AssertionError("Melanoma benchmark row count changed")
    if not np.array_equal(source["cell_order"].to_numpy(), np.arange(EXPECTED_TOTAL)):
        raise AssertionError("Provider source cell order changed")
    if not np.array_equal(row_lock["row_id"].to_numpy(), np.arange(EXPECTED_TOTAL)):
        raise AssertionError("test0827 row IDs changed")
    expected_dataset_order = [dataset for dataset, _, _, _ in DATASETS]
    if list(pd.unique(source["dataset"].astype(str))) != expected_dataset_order:
        raise AssertionError("Melanoma dataset block order changed")
    for left, right in (
        (source["source_cell_id"], row_lock["source_cell_id"]),
        (source["dataset"], row_lock["dataset"]),
        (source["unit_key"], row_lock["unit_key"]),
        (source["patient_id"], row_lock["donor"]),
    ):
        if not left.astype(str).equals(right.astype(str)):
            raise AssertionError("Frozen provider/test0827 row order differs")
    if source["source_cell_id"].astype(str).duplicated().any():
        raise AssertionError("Duplicate melanoma source cell IDs")
    gate_columns = [
        "provider_t_cell_gate", "provider_cd4_t_gate",
        "provider_cd8_t_gate", "provider_b_cell_gate",
    ]
    non_ptacek = ~source["dataset"].astype(str).eq("MEL_Ptacek2021_MIBI")
    if not source.loc[non_ptacek, gate_columns].isna().all().all():
        raise AssertionError("Unexpected provider gates outside Ptacek")
    if source.loc[~non_ptacek, gate_columns].isna().any().any():
        raise AssertionError("Missing Ptacek provider gates")

    temporary_data = stage / "data"
    combined_parts = []
    audit_parts = []
    source_files = []
    for dataset, slug, expected_cells, expected_units in DATASETS:
        selected = source["dataset"].astype(str).eq(dataset).to_numpy()
        block = source.loc[selected].copy().reset_index(drop=True)
        if len(block) != expected_cells or block["unit_key"].astype(str).nunique() != expected_units:
            raise AssertionError(f"{dataset}: frozen cohort changed")
        fine = derive_fine_labels(dataset, block)
        joint = derive_joint_labels(fine)
        if not np.array_equal(fine, block["celltype_display"].astype(str).to_numpy()):
            raise AssertionError(f"{dataset}: fine mapping differs from frozen labels")
        if not np.array_equal(
            joint, block["celltype_joint_evaluation"].astype(str).to_numpy()
        ):
            raise AssertionError(f"{dataset}: joint mapping differs from frozen labels")
        if not np.array_equal(
            joint != "Non-joint", true_mask(block["is_joint_evaluation"])
        ):
            raise AssertionError(f"{dataset}: joint-evaluation flag changed")
        if not np.array_equal(
            np.isin(fine, ["Non-joint", "Epithelial (non-joint)"]),
            true_mask(block["is_nonjoint_display"]),
        ):
            raise AssertionError(f"{dataset}: non-joint display flag changed")
        observed = pd.Series(fine).value_counts().reindex(FINE_LABEL_ORDER, fill_value=0)
        if observed.to_dict() != EXPECTED_FINE_COUNTS[dataset]:
            raise AssertionError(f"{dataset}: fine-label counts changed")

        prefix = {
            "MEL_Hickey2023_CODEX": "hickey_codex",
            "MEL_Hoch2021_IMC": "hoch_imc",
            "MEL_Ptacek2021_MIBI": "ptacek_mibi",
        }[dataset]
        answer = pd.DataFrame({
            "cell_id": prefix + "::" + block["source_cell_id"].astype(str),
            "source_cell_id": block["source_cell_id"].astype(str),
            "source_index_zero_based": block["source_index_zero_based"],
            "dataset": block["dataset"].astype(str),
            "technology": block["technology"].astype(str),
            "donor": block["patient_id"].astype(str),
            "sample_id": block["sample_id"].astype(str),
            "unit_key": block["unit_key"].astype(str),
            "provider_celltype_original": block["provider_celltype_original"].astype(str),
            "provider_celltype_common": block["provider_celltype_common"].astype(str),
            "provider_celltype_coarse": block["provider_celltype_coarse"].astype(str),
            "provider_t_cell_gate": block["provider_t_cell_gate"],
            "provider_cd4_t_gate": block["provider_cd4_t_gate"],
            "provider_cd8_t_gate": block["provider_cd8_t_gate"],
            "provider_b_cell_gate": block["provider_b_cell_gate"],
            "celltype": fine,
            "celltype_joint_evaluation": joint,
            "joint_evaluation": joint != "Non-joint",
            "is_nonjoint": np.isin(
                fine, ["Non-joint", "Epithelial (non-joint)"]
            ),
            "celltype_relabel_version": VERSION,
        })
        output_path = temporary_data / f"{slug}_relabeled_metadata.csv.gz"
        write_csv_gz(answer, output_path)
        check = pd.read_csv(output_path, low_memory=False)
        if not answer.astype(str).equals(check.astype(str)):
            raise AssertionError(f"{dataset}: relabel output did not round-trip")
        combined_parts.append(answer)
        audit_parts.append(answer[[
            "dataset", "provider_celltype_original", "provider_t_cell_gate",
            "provider_cd4_t_gate", "provider_cd8_t_gate", "provider_b_cell_gate",
            "celltype", "celltype_joint_evaluation", "joint_evaluation",
        ]])
        source_files.append({
            "dataset": dataset,
            "cells": expected_cells,
            "units": expected_units,
            "relabeled_output": str((DATA_DIR / output_path.name).relative_to(BENCHMARK_ROOT)),
            "relabeled_output_sha256": sha256(output_path),
        })

    combined = pd.concat(combined_parts, ignore_index=True)
    audit = pd.concat(audit_parts, ignore_index=True)
    if len(combined) != EXPECTED_TOTAL or combined["cell_id"].duplicated().any():
        raise AssertionError("Combined melanoma relabel output is invalid")
    if not np.array_equal(
        combined["celltype"].astype(str).to_numpy(),
        row_lock["provider_display_label"].astype(str).to_numpy(),
    ):
        raise AssertionError("Formal fine labels differ from test0827 row lock")
    if not np.array_equal(
        combined["celltype_joint_evaluation"].astype(str).to_numpy(),
        row_lock["provider_label"].astype(str).to_numpy(),
    ):
        raise AssertionError("Formal joint labels differ from test0827 row lock")
    if not np.array_equal(
        combined["joint_evaluation"].to_numpy(dtype=bool),
        true_mask(row_lock["joint_evaluation"]),
    ):
        raise AssertionError("Formal joint flags differ from test0827 row lock")
    combined_path = temporary_data / "melanoma_61221_relabeled_metadata.csv.gz"
    lightweight_path = stage / "relabeled_metadata.csv.gz"
    write_csv_gz(combined, combined_path)
    write_csv_gz(combined, lightweight_path)

    mapping = (
        audit.groupby(
            [
                "dataset", "provider_celltype_original", "provider_t_cell_gate",
                "provider_cd4_t_gate", "provider_cd8_t_gate", "provider_b_cell_gate",
                "celltype", "celltype_joint_evaluation", "joint_evaluation",
            ],
            dropna=False,
            sort=False,
        )
        .size().rename("n_cells").reset_index()
        .sort_values(["dataset", "celltype", "provider_celltype_original"])
    )
    if len(mapping) != 56 or int(mapping["n_cells"].sum()) != EXPECTED_TOTAL:
        raise AssertionError("Provider mapping audit contract changed")
    mapping.to_csv(stage / "relabel_mapping_audit.csv", index=False)
    fine_counts = (
        audit.groupby(["celltype", "dataset"], observed=False)
        .size().unstack(fill_value=0).reindex(FINE_LABEL_ORDER, fill_value=0)
    )
    fine_counts["Total"] = fine_counts.sum(axis=1)
    fine_counts.to_csv(stage / "relabel_counts.csv")
    joint_counts = (
        audit.groupby(["celltype_joint_evaluation", "dataset"], observed=False)
        .size().unstack(fill_value=0).reindex(JOINT_LABEL_ORDER, fill_value=0)
    )
    joint_counts["Total"] = joint_counts.sum(axis=1)
    joint_counts.to_csv(stage / "joint_relabel_counts.csv")
    palette_frame().to_csv(stage / "display_palette.csv", index=False)

    outputs = sorted(path for path in stage.rglob("*") if path.is_file())
    return {
        "version": VERSION,
        "purpose": "Provider-based fine and joint melanoma benchmark-core labels",
        "cells": EXPECTED_TOTAL,
        "datasets": [dataset for dataset, _, _, _ in DATASETS],
        "dataset_cells": {dataset: cells for dataset, _, cells, _ in DATASETS},
        "dataset_units": {dataset: units for dataset, _, _, units in DATASETS},
        "fine_labels": list(FINE_LABEL_ORDER),
        "joint_labels": list(JOINT_LABEL_ORDER),
        "provider_fields_used": [
            "provider_celltype_original", "provider_t_cell_gate",
            "provider_cd4_t_gate", "provider_cd8_t_gate", "provider_b_cell_gate",
        ],
        "expression_matrix_read": False,
        "ramp_or_umap_used_for_relabeling": False,
        "response_or_timepoint_used_for_relabeling": False,
        "spatial_coordinates_used_for_relabeling": False,
        "frozen_old_labels_used_only_for_exact_validation": True,
        "canonical_parent_inputs_modified": False,
        "display_draw_order_bottom_to_top": list(DRAW_ORDER_BOTTOM_TO_TOP),
        "source_files": {
            "provider_metadata": {
                "path": str(PROVIDER_SOURCE),
                "sha256": sha256(PROVIDER_SOURCE),
            },
            "test0827_row_lock": {
                "path": str(FROZEN_ROW_SOURCE),
                "sha256": sha256(FROZEN_ROW_SOURCE),
            },
        },
        "dataset_outputs": source_files,
        "recipe": {
            "path": str(Path(__file__).relative_to(BENCHMARK_ROOT)),
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


def publish(stage: Path, manifest: dict[str, object]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for path in (stage / "data").iterdir():
        os.replace(path, DATA_DIR / path.name)
    for name in (
        "relabeled_metadata.csv.gz", "relabel_mapping_audit.csv",
        "relabel_counts.csv", "joint_relabel_counts.csv", "display_palette.csv",
    ):
        os.replace(stage / name, HERE / name)
    (HERE / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    stage = Path(tempfile.mkdtemp(prefix=".melanoma-relabel-build-", dir=HERE))
    try:
        manifest = build(stage)
        publish(stage, manifest)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    counts = pd.read_csv(HERE / "relabel_counts.csv")
    print(counts.to_string(index=False))
    print(f"\nSaved {EXPECTED_TOTAL:,} cells with {len(FINE_LABEL_ORDER)} fine labels")


if __name__ == "__main__":
    main()
