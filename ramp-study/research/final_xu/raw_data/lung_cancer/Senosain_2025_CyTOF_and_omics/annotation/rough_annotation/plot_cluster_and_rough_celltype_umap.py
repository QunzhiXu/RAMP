#!/usr/bin/env python3
"""Plot frozen Senosain UMAP with Leiden IDs and rough cell-type calls.

This script never recomputes PCA, neighbors, Leiden, or UMAP.  It joins the
frozen per-cell cluster assignments to ``rough_cluster_annotations.csv`` and
creates a side-by-side visual audit of the manual first-pass annotations.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ANNOTATION_ROOT = HERE.parent
ASSIGNMENTS_PATH = ANNOTATION_ROOT / "leiden_all34" / "cluster_assignments.csv.gz"
ROUGH_LABELS_PATH = HERE / "rough_cluster_annotations.csv"
OUTPUT_STEM = HERE / "senosain_140k_cluster_and_rough_celltype_umap"

PLOT_GROUP = {
    "Immune": "Immune",
    "Epithelial cancer": "Epithelial cancer",
    "Fibroblast/mesenchymal": "Fibroblast/mesenchymal",
    "Endothelial": "Endothelial",
    "Mixed / unresolved": "Mixed / unresolved / QC",
    "Unassigned": "Mixed / unresolved / QC",
    "Endothelial-like / mixed": "Mixed / unresolved / QC",
    "Quality-control / mixed": "Mixed / unresolved / QC",
    "Mixed / epithelial-leaning": "Mixed / unresolved / QC",
    "Unassigned / epithelial-leaning": "Mixed / unresolved / QC",
}

GROUP_COLORS = {
    "Immune": "#2878B5",
    "Epithelial cancer": "#D73027",
    "Fibroblast/mesenchymal": "#E69F00",
    "Endothelial": "#009E73",
    "Mixed / unresolved / QC": "#8A8A8A",
}

# Short display names are used only inside the UMAP.  The full labels from the
# mapping CSV are printed in the right-hand key.
SHORT_LABELS = {
    "C00": "CD4 T",
    "C01": "EPCAM+ CK7+ cancer",
    "C02": "CD56+ HLA-DRhi myeloid",
    "C03": "CD8 T",
    "C04": "Apoptotic Fmes-like",
    "C05": "Activated cancer",
    "C06": "KI67hi mixed",
    "C07": "H3hi lineage-low",
    "C08": "HLA-DR+ APC-like",
    "C09": "Activated myeloid",
    "C10": "CD24hi immune (B-like?)",
    "C11": "Endothelial",
    "C12": "pSTAT6hi unassigned",
    "C13": "Endothelial-like / mixed",
    "C14": "Doublet / aggregate",
    "C15": "HER2hi epithelial-immune mix",
    "C16": "MDM2hi pSTAT3hi unassigned",
    "C17": "panCKhi METhi cancer",
}

# Fixed text positions reduce collisions in the crowded middle of the embedding.
# They affect labels only and never alter a cell coordinate.
LABEL_POSITIONS = {
    "C00": (12.00, 7.22),
    "C01": (12.20, -1.35),
    "C02": (9.74, 1.72),
    "C03": (14.68, 6.98),
    "C04": (14.50, 1.05),
    "C05": (10.20, -2.20),
    "C06": (12.28, 4.92),
    "C07": (14.65, 1.82),
    "C08": (13.25, 3.36),
    "C09": (8.83, 3.25),
    "C10": (11.65, 3.15),
    "C11": (9.90, 1.07),
    "C12": (12.90, 1.73),
    "C13": (10.08, 4.09),
    "C14": (7.98, 0.22),
    "C15": (11.12, -0.34),
    "C16": (16.08, 1.48),
    "C17": (13.18, 0.30),
}


def validate_inputs(assignments: pd.DataFrame, labels: pd.DataFrame) -> list[str]:
    required_assignment_columns = {"cluster", "UMAP1", "UMAP2"}
    required_label_columns = {
        "cluster",
        "rough_label",
        "broad_compartment",
        "confidence",
    }
    if not required_assignment_columns.issubset(assignments.columns):
        missing = required_assignment_columns - set(assignments.columns)
        raise ValueError(f"Assignment table is missing columns: {sorted(missing)}")
    if not required_label_columns.issubset(labels.columns):
        missing = required_label_columns - set(labels.columns)
        raise ValueError(f"Rough-label table is missing columns: {sorted(missing)}")
    if labels["cluster"].duplicated().any():
        raise ValueError("Rough-label table has duplicate cluster IDs")

    cluster_order = sorted(assignments["cluster"].astype(str).unique())
    expected = [f"C{i:02d}" for i in range(18)]
    if cluster_order != expected:
        raise ValueError(f"Expected frozen clusters C00-C17; got {cluster_order}")
    if sorted(labels["cluster"].astype(str)) != expected:
        raise ValueError("Rough-label table does not cover C00-C17 exactly once")
    if set(labels["broad_compartment"]) - set(PLOT_GROUP):
        unknown = set(labels["broad_compartment"]) - set(PLOT_GROUP)
        raise ValueError(f"No display group defined for: {sorted(unknown)}")
    if set(SHORT_LABELS) != set(expected):
        raise ValueError("Short-label key does not cover C00-C17 exactly")
    coordinates = assignments[["UMAP1", "UMAP2"]].to_numpy(float)
    if not np.isfinite(coordinates).all():
        raise ValueError("Frozen UMAP coordinates contain non-finite values")
    return expected


def add_cluster_centers(axis, frame: pd.DataFrame, cluster_order: list[str]) -> None:
    from matplotlib import patheffects

    for cluster in cluster_order:
        rows = frame["cluster"].eq(cluster)
        x, y = frame.loc[rows, ["UMAP1", "UMAP2"]].median().to_numpy(float)
        label = axis.text(
            x,
            y,
            cluster,
            ha="center",
            va="center",
            fontsize=9.2,
            fontweight="bold",
            color="black",
            zorder=10,
        )
        label.set_path_effects(
            [patheffects.withStroke(linewidth=2.8, foreground="white")]
        )


def style_umap_axis(axis, title: str) -> None:
    axis.set_title(title, fontsize=15, fontweight="bold", pad=13)
    axis.set_xlabel("UMAP 1", fontsize=11)
    axis.set_ylabel("UMAP 2", fontsize=11)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.spines[:].set_visible(False)
    axis.set_aspect("equal", adjustable="box")


def main() -> None:
    mpl_cache = HERE / ".matplotlib_cache"
    xdg_cache = HERE / ".xdg_cache"
    mpl_cache.mkdir(exist_ok=True)
    xdg_cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    assignments = pd.read_csv(
        ASSIGNMENTS_PATH,
        usecols=["cluster", "UMAP1", "UMAP2"],
    )
    labels = pd.read_csv(ROUGH_LABELS_PATH)
    cluster_order = validate_inputs(assignments, labels)
    label_index = labels.set_index("cluster").loc[cluster_order]
    assignments["plot_group"] = assignments["cluster"].map(
        label_index["broad_compartment"].map(PLOT_GROUP)
    )
    if assignments["plot_group"].isna().any():
        raise AssertionError("Some cells did not receive a display group")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "savefig.dpi": 300,
        }
    )
    fig = plt.figure(figsize=(25.5, 10.5), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.82])
    cluster_axis = fig.add_subplot(grid[0, 0])
    type_axis = fig.add_subplot(grid[0, 1])
    key_axis = fig.add_subplot(grid[0, 2])

    cluster_palette = plt.get_cmap("tab20")
    cluster_colors = {
        cluster: cluster_palette(i) for i, cluster in enumerate(cluster_order)
    }
    for cluster in cluster_order:
        rows = assignments["cluster"].eq(cluster)
        cluster_axis.scatter(
            assignments.loc[rows, "UMAP1"],
            assignments.loc[rows, "UMAP2"],
            s=0.32,
            alpha=0.62,
            linewidths=0,
            rasterized=True,
            color=cluster_colors[cluster],
        )
    add_cluster_centers(cluster_axis, assignments, cluster_order)
    style_umap_axis(cluster_axis, "A  Leiden clusters (all 34 uniform markers)")

    plot_order = [
        "Mixed / unresolved / QC",
        "Fibroblast/mesenchymal",
        "Endothelial",
        "Immune",
        "Epithelial cancer",
    ]
    for group in plot_order:
        rows = assignments["plot_group"].eq(group)
        type_axis.scatter(
            assignments.loc[rows, "UMAP1"],
            assignments.loc[rows, "UMAP2"],
            s=0.32,
            alpha=0.56,
            linewidths=0,
            rasterized=True,
            color=GROUP_COLORS[group],
        )

    for cluster in cluster_order:
        rows = assignments["cluster"].eq(cluster)
        x, y = assignments.loc[rows, ["UMAP1", "UMAP2"]].median().to_numpy(float)
        label_x, label_y = LABEL_POSITIONS[cluster]
        type_axis.annotate(
            f"{cluster}  {SHORT_LABELS[cluster]}",
            xy=(x, y),
            xytext=(label_x, label_y),
            textcoords="data",
            ha="center",
            va="center",
            fontsize=7.7,
            fontweight="bold",
            color="black",
            bbox={
                "boxstyle": "round,pad=0.20",
                "facecolor": "white",
                "edgecolor": GROUP_COLORS[PLOT_GROUP[label_index.loc[cluster, "broad_compartment"]]],
                "linewidth": 0.8,
                "alpha": 0.88,
            },
            arrowprops={
                "arrowstyle": "-",
                "color": "#555555",
                "linewidth": 0.55,
                "shrinkA": 1.0,
                "shrinkB": 1.0,
            },
            zorder=10,
        )
    style_umap_axis(type_axis, "B  First-pass rough cell-type annotation")

    key_axis.axis("off")
    key_axis.set_xlim(0, 1)
    key_axis.set_ylim(0, 1)
    key_axis.text(
        0.02,
        0.975,
        "Cluster → complete rough annotation",
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
    )
    key_axis.text(
        0.02,
        0.935,
        "Confidence is shown in parentheses",
        ha="left",
        va="top",
        fontsize=9.2,
        color="#555555",
    )
    group_key_positions = [
        ("Immune", 0.025, 0.892),
        ("Epithelial cancer", 0.50, 0.892),
        ("Fibroblast/mesenchymal", 0.025, 0.857),
        ("Endothelial", 0.50, 0.857),
        ("Mixed / unresolved / QC", 0.025, 0.822),
    ]
    for group, x, y in group_key_positions:
        key_axis.scatter(
            [x], [y], s=42, color=GROUP_COLORS[group], linewidths=0, clip_on=False
        )
        key_axis.text(
            x + 0.028,
            y,
            group,
            ha="left",
            va="center",
            fontsize=8.4,
            color="#333333",
        )

    y_positions = np.linspace(0.770, 0.040, len(cluster_order))
    for cluster, y in zip(cluster_order, y_positions, strict=True):
        row = label_index.loc[cluster]
        group = PLOT_GROUP[row["broad_compartment"]]
        key_axis.scatter(
            [0.025],
            [y],
            s=38,
            color=GROUP_COLORS[group],
            linewidths=0,
            clip_on=False,
        )
        key_axis.text(
            0.055,
            y,
            cluster,
            ha="left",
            va="center",
            fontsize=9.2,
            fontweight="bold",
        )
        key_axis.text(
            0.15,
            y,
            f"{row['rough_label']}  ({row['confidence']})",
            ha="left",
            va="center",
            fontsize=8.55,
        )

    fig.suptitle(
        "Senosain CyTOF · frozen 140,000-cell UMAP",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.012,
        "Rough labels are manual first-pass interpretations, not provider ground truth; mixed/unresolved clusters remain explicitly flagged.",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#444444",
    )

    for suffix in ("png", "pdf"):
        output = OUTPUT_STEM.with_suffix(f".{suffix}")
        fig.savefig(output, bbox_inches="tight", facecolor="white")
        print(output)
    plt.close(fig)


if __name__ == "__main__":
    main()
