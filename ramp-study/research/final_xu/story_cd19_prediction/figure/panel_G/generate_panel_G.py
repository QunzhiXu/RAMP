#!/usr/bin/env python3
"""Generate the complete Panel G from the frozen matched UMAP coordinates.

The two views contain exactly the same 88,031 cells and differ only in whether
the aligned matrix used the 14 shared markers alone or those markers plus the
predicted CD19 channel. Provider labels are used only for post-hoc coloring.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
STORY = HERE.parents[1]
CACHE = HERE / ".cache"
CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd


COORDINATES = STORY / "umap" / "data" / "umap_coordinates.npz"
METADATA = STORY / "alignment" / "output" / "cell_metadata.csv.gz"
METRICS = STORY / "umap" / "data" / "metrics.csv"
CONDITIONS = ["without_cd19", "with_predicted_cd19"]
CONDITION_TITLES = {
    "without_cd19": "14 shared markers (without CD19)",
    "with_predicted_cd19": "14 shared markers + predicted CD19",
}
LABEL_ORDER = [
    "Mature / CODEX B",
    "Pre / immature B",
    "Plasma cell",
    "CD4 T",
    "CD8 T",
    "Other T / NKT",
    "NK",
    "Monocyte",
    "Macrophage",
    "Dendritic",
    "Other myeloid",
    "Promyelocytic progenitors",
    "HSPC / progenitor",
    "CODEX erythroid",
    "Triana erythroid progenitor",
    "Megakaryocytic",
    "Endothelial",
    "Stromal / VSMC",
    "Unclassified / technical",
]
COLORS = {
    "Mature / CODEX B": "#D81B60",
    "Pre / immature B": "#F06292",
    "Plasma cell": "#8E24AA",
    "CD4 T": "#1769AA",
    "CD8 T": "#00A6D6",
    "Other T / NKT": "#6A5ACD",
    "NK": "#00A087",
    "Monocyte": "#F39C12",
    "Macrophage": "#B45F06",
    "Dendritic": "#7CB342",
    "Other myeloid": "#A66E3F",
    "Promyelocytic progenitors": "#FF7043",
    "HSPC / progenitor": "#7E57C2",
    "CODEX erythroid": "#C2185B",
    "Triana erythroid progenitor": "#EC407A",
    "Megakaryocytic": "#8D6E63",
    "Endothelial": "#43A047",
    "Stromal / VSMC": "#00897B",
    "Unclassified / technical": "#697386",
}


def load() -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    archive = np.load(COORDINATES)
    coordinates = {condition: archive[condition].astype(float) for condition in CONDITIONS}
    metadata = pd.read_csv(METADATA, low_memory=False)
    metrics = pd.read_csv(METRICS)
    if len(metadata) != 88_031:
        raise RuntimeError(f"Expected 88,031 cells, found {len(metadata):,}")
    for condition, xy in coordinates.items():
        if xy.shape != (len(metadata), 2) or not np.isfinite(xy).all():
            raise RuntimeError(f"Invalid coordinates for {condition}: {xy.shape}")
    if set(CONDITIONS) - set(metrics["condition"]):
        raise RuntimeError("Required conditions are absent from the frozen metrics")
    return coordinates, metadata, metrics.set_index("condition")


def draw_umap(ax: plt.Axes, xy: np.ndarray, labels: np.ndarray) -> None:
    active = [label for label in LABEL_ORDER if np.any(labels == label)]
    # Put unclassified cells beneath all biological labels.
    order = ["Unclassified / technical"] + [
        label for label in active if label != "Unclassified / technical"
    ]
    for label in order:
        selected = labels == label
        ax.scatter(
            xy[selected, 0],
            xy[selected, 1],
            s=0.48,
            color=COLORS[label],
            alpha=0.18 if label == "Unclassified / technical" else 0.68,
            linewidths=0,
            rasterized=True,
        )
    ax.set_xticks([])
    ax.set_yticks([])
    # UMAP axes have arbitrary orientation and units; filling the assigned
    # rectangular panel avoids wasting horizontal space in the composite.
    ax.set_aspect("auto")
    for spine in ax.spines.values():
        spine.set_visible(False)


def highlight_separation(ax: plt.Axes, condition: str) -> None:
    """Add restrained, data-coordinate callouts to the complete UMAP."""
    if condition == "without_cd19":
        ax.add_patch(
            Ellipse(
                (2.7, 12.0),
                width=4.5,
                height=4.1,
                fill=False,
                edgecolor="#202124",
                linewidth=1.7,
                linestyle=(0, (4, 2.5)),
                zorder=8,
            )
        )
        ax.annotate(
            "Mature B + NK mixed",
            xy=(3.5, 12.7),
            xytext=(6.4, 14.0),
            ha="center",
            va="center",
            fontsize=9.3,
            fontweight="bold",
            color="#202124",
            arrowprops=dict(arrowstyle="->", color="#202124", lw=1.15),
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.9),
            zorder=9,
        )
        return

    ax.add_patch(
        Ellipse(
            (3.35, 3.55),
            width=3.9,
            height=4.1,
            fill=False,
            edgecolor=COLORS["Mature / CODEX B"],
            linewidth=1.9,
            zorder=8,
        )
    )
    ax.add_patch(
        Ellipse(
            (13.0, 10.1),
            width=2.25,
            height=2.45,
            fill=False,
            edgecolor=COLORS["NK"],
            linewidth=1.9,
            zorder=8,
        )
    )
    ax.annotate(
        "Mature B",
        xy=(3.0, 3.4),
        xytext=(-1.6, 0.3),
        ha="center",
        va="center",
        fontsize=9.3,
        fontweight="bold",
        color=COLORS["Mature / CODEX B"],
        arrowprops=dict(arrowstyle="->", color=COLORS["Mature / CODEX B"], lw=1.2),
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.9),
        zorder=9,
    )
    ax.annotate(
        "NK",
        xy=(13.0, 10.1),
        xytext=(11.2, 13.5),
        ha="center",
        va="center",
        fontsize=9.3,
        fontweight="bold",
        color=COLORS["NK"],
        arrowprops=dict(arrowstyle="->", color=COLORS["NK"], lw=1.2),
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.9),
        zorder=9,
    )


def main() -> None:
    coordinates, metadata, metrics = load()
    labels = metadata["celltype"].astype(str).to_numpy()

    active = [label for label in LABEL_ORDER if np.any(labels == label)]

    fig_width = 10.5
    # A slightly taller canvas and near-edge axes let the two UMAPs fill the
    # full bottom-row height used by Panel F in the assembled figure.
    fig_height = 4.90
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=220)
    plot_bottom = 0.012
    plot_height = 0.778
    axis_widths = [0.365, 0.375]
    central_gap = 0.004
    group_left = 0.250
    axes = [
        fig.add_axes([group_left, plot_bottom, axis_widths[0], plot_height]),
        fig.add_axes(
            [group_left + axis_widths[0] + central_gap, plot_bottom, axis_widths[1], plot_height]
        ),
    ]
    fig.suptitle(
        "G.  Predicted CD19 restores Mature B–NK separation",
        x=0.03,
        y=0.985,
        ha="left",
        va="top",
        fontsize=15.5,
        fontweight="bold",
    )

    for ax, condition in zip(axes, CONDITIONS):
        draw_umap(ax, coordinates[condition], labels)
        highlight_separation(ax, condition)
        ax.set_title(CONDITION_TITLES[condition], fontsize=11.5, fontweight="bold", pad=7)

    before = float(metrics.loc["without_cd19", "matureB_vs_NK_silhouette_umap"])
    after = float(metrics.loc["with_predicted_cd19", "matureB_vs_NK_silhouette_umap"])
    fig.text(
        0.67,
        0.890,
        f"Mature B–NK silhouette: {before:.3f}  →  {after:.3f}",
        ha="center",
        va="center",
        fontsize=10.0,
        fontweight="bold",
        color="#2F3133",
    )

    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=COLORS[label], markersize=7.0)
        for label in active
    ]
    fig.legend(
        handles,
        active,
        loc="center right",
        bbox_to_anchor=(0.244, 0.405),
        ncol=1,
        frameon=False,
        fontsize=10.2,
        labelspacing=0.16,
        handletextpad=0.34,
    )
    png = HERE / "panel_G_cd19_umap_comparison.png"
    pdf = HERE / "panel_G_cd19_umap_comparison.pdf"
    fig.savefig(png, dpi=320, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)

    audit = {
        "n_cells": int(len(metadata)),
        "conditions": CONDITIONS,
        "coordinates": str(COORDINATES),
        "metadata": str(METADATA),
        "labels_used_for_umap_fitting": False,
        "displayed_celltypes": active,
        "matureB_vs_NK_silhouette": {
            condition: float(metrics.loc[condition, "matureB_vs_NK_silhouette_umap"])
            for condition in CONDITIONS
        },
    }
    (HERE / "panel_G_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    print(f"Saved {png}")
    print(f"Saved {pdf}")


if __name__ == "__main__":
    main()
