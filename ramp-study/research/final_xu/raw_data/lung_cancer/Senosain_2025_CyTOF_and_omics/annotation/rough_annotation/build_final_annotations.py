#!/usr/bin/env python3
"""Build the frozen 140,000-cell final Senosain annotation.

This script joins the original all-34-marker Leiden assignments to the reviewed
cluster/subcluster annotation map.  C06 is resolved with its frozen local
subclusters; all other cells use their original C00-C17 cluster.  It never
changes the expression matrices, original Leiden assignments, or C06
subclustering.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ANNOTATION_VERSION = "senosain-140k-final-annotation-v1"
MAPPING_REVISION = "2026-08-27-c06-s01-conservative-mix"
EXPECTED_CELLS = 140_000
EXPECTED_C06_CELLS = 9_778
EXPECTED_C06_S01_CELLS = 3_571

HERE = Path(__file__).resolve().parent
ANNOTATION_ROOT = HERE.parent
ASSIGNMENTS_PATH = ANNOTATION_ROOT / "leiden_all34" / "cluster_assignments.csv.gz"
C06_ASSIGNMENTS_PATH = HERE / "c06_recluster" / "c06_subcluster_assignments.csv.gz"
MAP_PATH = HERE / "final_cluster_subcluster_annotation_map.csv"

CELL_OUTPUT = HERE / "senosain_140k_final_cell_annotations.csv.gz"
CLUSTER_SUMMARY_OUTPUT = HERE / "final_annotation_summary.csv"
CELL_TYPE_SUMMARY_OUTPUT = HERE / "final_cell_type_summary.csv"
MANIFEST_OUTPUT = HERE / "final_annotation_manifest.json"
PLOT_STEM = HERE / "senosain_140k_final_annotation_umap"

CELL_TYPE_ORDER = [
    "CD4 T",
    "CD8 T",
    "Myeloid",
    "B-like",
    "Tumor/Epithelial",
    "Endothelial",
    "Mix",
]

COLORS = {
    "CD4 T": "#3B82F6",
    "CD8 T": "#06B6D4",
    "Myeloid": "#EAB308",
    "B-like": "#8B5CF6",
    "Tumor/Epithelial": "#DC2626",
    "Endothelial": "#059669",
    "Mix": "#9CA3AF",
}

PLOT_LABELS = {
    "C00": "CD4 T",
    "C01": "Tumor/Epithelial",
    "C02": "Myeloid (medium)",
    "C03": "CD8 T",
    "C04": "Mix / mesenchymal-leaning",
    "C05": "Tumor/Epithelial",
    "C06": "All subclusters Mix",
    "C07": "Mix",
    "C08": "Mix",
    "C09": "Myeloid",
    "C10": "B-like (provisional)",
    "C11": "Endothelial",
    "C12": "Mix",
    "C13": "Mix",
    "C14": "Mix / QC exclude",
    "C15": "Mix",
    "C16": "Mix / QC suspect",
    "C17": "Tumor/Epithelial (medium-high)",
}

LABEL_OFFSETS = {
    "C04": (1.2, -0.75),
    "C07": (-0.8, 0.75),
    "C08": (0.7, 0.5),
    "C10": (-0.55, -0.45),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_and_validate_inputs() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    assignments = pd.read_csv(ASSIGNMENTS_PATH, low_memory=False)
    required = {
        "sampled_row_zero_based",
        "source_cell_id",
        "patient_id",
        "unit_key",
        "sample_rank_within_patient",
        "cluster",
        "UMAP1",
        "UMAP2",
    }
    missing = required - set(assignments.columns)
    if missing:
        raise ValueError(f"Original assignments lack columns: {sorted(missing)}")
    if len(assignments) != EXPECTED_CELLS:
        raise ValueError(f"Expected {EXPECTED_CELLS:,} cells; found {len(assignments):,}")
    if not np.array_equal(
        assignments["sampled_row_zero_based"].to_numpy(),
        np.arange(EXPECTED_CELLS),
    ):
        raise ValueError("Original assignments are no longer matrix-row aligned")
    if not assignments["source_cell_id"].astype(str).is_unique:
        raise ValueError("source_cell_id is not unique")
    expected_clusters = {f"C{i:02d}" for i in range(18)}
    observed_clusters = set(assignments["cluster"].astype(str))
    if observed_clusters != expected_clusters:
        raise ValueError(
            f"Expected C00-C17; observed {sorted(observed_clusters)}"
        )

    c06 = pd.read_csv(C06_ASSIGNMENTS_PATH, low_memory=False)
    required_c06 = {"sampled_row_zero_based", "parent_cluster", "subcluster"}
    missing_c06 = required_c06 - set(c06.columns)
    if missing_c06:
        raise ValueError(f"C06 assignments lack columns: {sorted(missing_c06)}")
    if len(c06) != EXPECTED_C06_CELLS:
        raise ValueError(
            f"Expected {EXPECTED_C06_CELLS:,} C06 cells; found {len(c06):,}"
        )
    if not c06["sampled_row_zero_based"].is_unique:
        raise ValueError("C06 sampled rows are not unique")
    if set(c06["parent_cluster"].astype(str)) != {"C06"}:
        raise ValueError("C06 subcluster table contains another parent cluster")
    if set(c06["subcluster"].astype(str)) != {"S00", "S01", "S02", "S03"}:
        raise ValueError("Expected C06 subclusters S00-S03")
    frozen_c06_rows = set(
        assignments.loc[
            assignments["cluster"].eq("C06"), "sampled_row_zero_based"
        ].astype(int)
    )
    if set(c06["sampled_row_zero_based"].astype(int)) != frozen_c06_rows:
        raise ValueError("C06 local assignments do not match the frozen C06 cells")

    mapping = pd.read_csv(MAP_PATH, keep_default_na=False)
    required_map = {
        "parent_cluster",
        "subcluster",
        "final_cell_type",
        "annotation_detail",
        "annotation_confidence",
        "qc_recommendation",
        "lineage_reference_use",
        "rationale_short",
    }
    missing_map = required_map - set(mapping.columns)
    if missing_map:
        raise ValueError(f"Final mapping lacks columns: {sorted(missing_map)}")
    if mapping.duplicated(["parent_cluster", "subcluster"]).any():
        raise ValueError("Final mapping has duplicate parent/subcluster keys")
    if set(mapping["final_cell_type"]) - set(CELL_TYPE_ORDER):
        raise ValueError("Final mapping contains an unexpected cell type")
    non_c06 = mapping.loc[mapping["parent_cluster"].ne("C06")]
    if set(non_c06["parent_cluster"]) != expected_clusters - {"C06"}:
        raise ValueError("Final mapping does not cover every non-C06 cluster once")
    c06_map = mapping.loc[mapping["parent_cluster"].eq("C06")]
    if set(c06_map["subcluster"]) != {"S00", "S01", "S02", "S03"}:
        raise ValueError("Final mapping does not cover every C06 subcluster")
    c06_s01_map = c06_map.loc[c06_map["subcluster"].eq("S01")].iloc[0]
    if (
        c06_s01_map["final_cell_type"] != "Mix"
        or c06_s01_map["annotation_detail"]
        != "KI67-high proliferating CD4-like/epithelial-mixed cells"
    ):
        raise ValueError("C06/S01 must use the reviewed conservative Mix mapping")
    return assignments, c06, mapping


def build_cell_annotations(
    assignments: pd.DataFrame,
    c06: pd.DataFrame,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    answer = assignments.copy()
    answer = answer.rename(columns={"cluster": "parent_cluster"})
    c06_lookup = c06.set_index("sampled_row_zero_based")["subcluster"]
    answer["subcluster"] = answer["sampled_row_zero_based"].map(c06_lookup)
    answer["subcluster_key"] = answer["subcluster"].fillna("")

    mapping_for_join = mapping.rename(columns={"subcluster": "subcluster_key"})
    answer = answer.merge(
        mapping_for_join,
        on=["parent_cluster", "subcluster_key"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    answer = answer.sort_values("sampled_row_zero_based", kind="stable").reset_index(
        drop=True
    )
    if answer["final_cell_type"].isna().any():
        bad = answer.loc[
            answer["final_cell_type"].isna(), ["parent_cluster", "subcluster_key"]
        ].drop_duplicates()
        raise ValueError(f"Unmapped annotation keys:\n{bad}")
    if not np.array_equal(
        answer["sampled_row_zero_based"].to_numpy(), np.arange(EXPECTED_CELLS)
    ):
        raise ValueError("Final merge changed matrix-row order")

    answer.insert(0, "annotation_version", ANNOTATION_VERSION)
    answer["subcluster"] = answer["subcluster"].fillna("")
    answer = answer.drop(columns="subcluster_key")
    ordered = [
        "annotation_version",
        "sampled_row_zero_based",
        "source_cell_id",
        "patient_id",
        "unit_key",
        "sample_rank_within_patient",
        "parent_cluster",
        "subcluster",
        "final_cell_type",
        "annotation_detail",
        "annotation_confidence",
        "qc_recommendation",
        "lineage_reference_use",
        "rationale_short",
        "UMAP1",
        "UMAP2",
        "leiden_raw_label",
    ]
    return answer[ordered]


def write_summaries(cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cluster_summary = (
        cells.groupby(
            [
                "parent_cluster",
                "subcluster",
                "final_cell_type",
                "annotation_detail",
                "annotation_confidence",
                "qc_recommendation",
                "lineage_reference_use",
            ],
            dropna=False,
            sort=True,
        )
        .agg(
            cells=("sampled_row_zero_based", "size"),
            patients_with_cells=("patient_id", "nunique"),
        )
        .reset_index()
    )
    cluster_summary["fraction_of_140k"] = cluster_summary["cells"] / len(cells)
    cluster_summary.to_csv(CLUSTER_SUMMARY_OUTPUT, index=False)

    cell_type_summary = (
        cells.groupby("final_cell_type", sort=False)
        .agg(
            cells=("sampled_row_zero_based", "size"),
            patients_with_cells=("patient_id", "nunique"),
        )
        .reindex(CELL_TYPE_ORDER)
        .dropna(how="all")
        .reset_index()
    )
    cell_type_summary["cells"] = cell_type_summary["cells"].astype(int)
    cell_type_summary["patients_with_cells"] = cell_type_summary[
        "patients_with_cells"
    ].astype(int)
    cell_type_summary["fraction_of_140k"] = cell_type_summary["cells"] / len(cells)
    cell_type_summary.to_csv(CELL_TYPE_SUMMARY_OUTPUT, index=False)
    return cluster_summary, cell_type_summary


def setup_plotting():
    os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".matplotlib_cache"))
    os.environ.setdefault("XDG_CACHE_HOME", str(HERE / ".xdg_cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "savefig.dpi": 300,
        }
    )
    return plt


def write_umap(cells: pd.DataFrame) -> None:
    plt = setup_plotting()
    from matplotlib import patheffects

    fig, axis = plt.subplots(figsize=(14.8, 9.2), constrained_layout=True)
    for cell_type in CELL_TYPE_ORDER:
        rows = cells["final_cell_type"].eq(cell_type)
        if not rows.any():
            continue
        axis.scatter(
            cells.loc[rows, "UMAP1"],
            cells.loc[rows, "UMAP2"],
            s=0.35,
            alpha=0.52,
            color=COLORS[cell_type],
            linewidths=0,
            rasterized=True,
            label=f"{cell_type}  n={int(rows.sum()):,}",
        )

    for cluster, plot_label in PLOT_LABELS.items():
        rows = cells["parent_cluster"].eq(cluster)
        x = float(cells.loc[rows, "UMAP1"].median())
        y = float(cells.loc[rows, "UMAP2"].median())
        dx, dy = LABEL_OFFSETS.get(cluster, (0.0, 0.0))
        label = axis.text(
            x + dx,
            y + dy,
            f"{cluster}\n{plot_label}",
            ha="center",
            va="center",
            fontsize=7.5,
            fontweight="bold",
            color="black",
        )
        label.set_path_effects(
            [patheffects.withStroke(linewidth=2.5, foreground="white")]
        )

    axis.set_title(
        "Senosain CyTOF · final reviewed annotation",
        fontsize=18,
        fontweight="bold",
    )
    axis.set_xlabel("UMAP 1")
    axis.set_ylabel("UMAP 2")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.spines[:].set_visible(False)
    axis.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        markerscale=6,
        fontsize=10,
        title="Cell type",
        title_fontsize=11,
    )
    axis.text(
        0.01,
        0.012,
        "C06: all S00–S03 subclusters → Mix; S01 is CD4-like/epithelial-mixed.  "
        "C10 B-like is provisional; C02 is medium-confidence Myeloid; "
        "C17 is medium-high Tumor/Epithelial.\n"
        "QC: exclude C06-S03 and C14; C16 is excluded pending QC review.",
        transform=axis.transAxes,
        fontsize=8.2,
        color="#444444",
    )
    for suffix in ("png", "pdf"):
        fig.savefig(
            PLOT_STEM.with_suffix(f".{suffix}"),
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def main() -> None:
    assignments, c06, mapping = load_and_validate_inputs()
    cells = build_cell_annotations(assignments, c06, mapping)
    c06_s01 = cells["parent_cluster"].eq("C06") & cells["subcluster"].eq("S01")
    if int(c06_s01.sum()) != EXPECTED_C06_S01_CELLS:
        raise AssertionError(
            f"Expected {EXPECTED_C06_S01_CELLS:,} C06/S01 cells; "
            f"found {int(c06_s01.sum()):,}"
        )
    if set(cells.loc[c06_s01, "final_cell_type"].astype(str)) != {"Mix"}:
        raise AssertionError("C06/S01 is not mapped exclusively to Mix")
    cells.to_csv(
        CELL_OUTPUT,
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )
    cluster_summary, cell_type_summary = write_summaries(cells)
    write_umap(cells)

    manifest = {
        "schema_version": ANNOTATION_VERSION,
        "mapping_revision": MAPPING_REVISION,
        "description": (
            "Reviewed per-cell annotation joined to the frozen 140k Leiden and "
            "C06 subcluster assignments"
        ),
        "cells": int(len(cells)),
        "patients": int(cells["patient_id"].nunique()),
        "input_sha256": {
            ASSIGNMENTS_PATH.name: sha256(ASSIGNMENTS_PATH),
            C06_ASSIGNMENTS_PATH.name: sha256(C06_ASSIGNMENTS_PATH),
            MAP_PATH.name: sha256(MAP_PATH),
            Path(__file__).name: sha256(Path(__file__).resolve()),
        },
        "cell_type_counts": {
            str(row.final_cell_type): int(row.cells)
            for row in cell_type_summary.itertuples(index=False)
        },
        "qc_recommendation_counts": {
            str(key): int(value)
            for key, value in cells["qc_recommendation"]
            .value_counts(sort=False)
            .items()
        },
        "outputs": [
            CELL_OUTPUT.name,
            CLUSTER_SUMMARY_OUTPUT.name,
            CELL_TYPE_SUMMARY_OUTPUT.name,
            PLOT_STEM.with_suffix(".png").name,
            PLOT_STEM.with_suffix(".pdf").name,
            MANIFEST_OUTPUT.name,
        ],
        "important_interpretation": {
            "C06_S01": (
                "Mix; KI67-high proliferating CD4-like/epithelial-mixed, "
                "not a pure CD4 T-cell call"
            ),
            "C06_S02": "Mix; epithelial-leaning but not called tumor",
            "C04": "Mix; mesenchymal-leaning but not called Fmes",
            "C10": "B-like is provisional because CD19/CD20 are unavailable",
            "qc_exclude": ["C06-S03", "C14"],
            "qc_exclude_pending_review": ["C16"],
        },
    }
    MANIFEST_OUTPUT.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {CELL_OUTPUT}")
    print(cell_type_summary.to_string(index=False))
    print("QC recommendations:")
    print(cells["qc_recommendation"].value_counts().to_string())


if __name__ == "__main__":
    main()
