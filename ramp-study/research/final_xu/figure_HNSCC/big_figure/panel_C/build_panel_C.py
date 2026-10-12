#!/usr/bin/env python3
"""Build publication-formatted Panel C from the frozen HNSCC UMAP coordinates."""

from __future__ import annotations

import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
HNSCC_ROOT = HERE.parent.parent
COORDINATES = HNSCC_ROOT / "alignment" / "umap" / "full_94488_umap_coordinates.csv.gz"
CACHE = HERE / ".cache"
for child in ("matplotlib", "xdg"):
    (CACHE / child).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE / "xdg"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


INK = "#182027"
BLUE = "#0072B2"
GOLD = "#E69F00"
PLOT_SEED = 20260902

DATASET_COLORS = {"GSE212797_ADT": BLUE, "Haist_CODEX": GOLD}
DATASET_NAMES = {
    "GSE212797_ADT": "Rahim CITE-seq",
    "Haist_CODEX": "Haist CODEX",
}
DISPLAY_ORDER = [
    "Tumor / epithelial / stroma",
    "Myeloid",
    "Other",
    "Vascular",
    "NK / gdT",
    "Plasma",
    "B",
    "CD8 T",
    "CD4 T",
]
TYPE_COLORS = {
    "CD8 T": "#D55E00",
    "CD4 T": "#CC79A7",
    "B": "#0072B2",
    "Plasma": "#56B4E9",
    "NK / gdT": "#009E73",
    "Myeloid": "#E5C100",
    "Tumor / epithelial / stroma": "#B2182B",
    "Vascular": "#17BECF",
    "Other": "#BDBDBD",
}


def clean_umap_axis(axis: plt.Axes) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_aspect("equal", adjustable="box")


def main() -> None:
    if not COORDINATES.exists():
        raise FileNotFoundError(f"Missing frozen UMAP coordinates: {COORDINATES}")
    data = pd.read_csv(COORDINATES, low_memory=False)
    if len(data) != 94_488:
        raise AssertionError(f"Expected 94,488 cells, found {len(data):,}")
    if set(data["dataset"].astype(str)) != set(DATASET_COLORS):
        raise AssertionError("Unexpected HNSCC datasets")
    if set(data["plot_label"].astype(str)) != set(DISPLAY_ORDER):
        raise AssertionError("Unexpected harmonized display labels")

    rng = np.random.default_rng(PLOT_SEED)
    dataset_order = rng.permutation(len(data))

    fig = plt.figure(figsize=(10.8, 4.6), facecolor="white")
    fig.text(0.018, 0.970, "C", ha="left", va="top",
             fontsize=24, weight="bold", color=INK)
    fig.text(
        0.072,
        0.943,
        "RAMP ALIGNMENT OF HNSCC CITE-seq AND CODEX",
        ha="left",
        va="top",
        fontsize=14.3,
        weight="bold",
        color=INK,
    )

    # Keep the two views visually paired; they use identical coordinates and
    # need only a hairline gap rather than two separate panel margins.
    # Match the axes aspect ratio to the shared data limits while retaining a
    # safe title band above both UMAPs.
    left = fig.add_axes([0.015, 0.055, 0.4825, 0.800])
    right = fig.add_axes([0.5025, 0.055, 0.4825, 0.800])

    left.scatter(
        data["UMAP1"].to_numpy()[dataset_order],
        data["UMAP2"].to_numpy()[dataset_order],
        c=data["dataset"].map(DATASET_COLORS).to_numpy()[dataset_order],
        s=1.0,
        alpha=0.48,
        linewidths=0,
        rasterized=True,
    )
    left.set_title("DATASET", loc="left", fontsize=12.0, weight="bold", color=INK, pad=5)
    dataset_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="", markersize=6.2,
            markerfacecolor=DATASET_COLORS[key], markeredgewidth=0,
            label=f"{DATASET_NAMES[key]}  (n={int(data['dataset'].eq(key).sum()):,})",
        )
        for key in DATASET_COLORS
    ]
    left.legend(
        handles=dataset_handles,
        frameon=False,
        fontsize=8.7,
        loc="upper left",
        borderaxespad=0.1,
        handletextpad=0.5,
    )

    for label_index, label in enumerate(DISPLAY_ORDER):
        rows = np.flatnonzero(data["plot_label"].astype(str).eq(label).to_numpy())
        rows = rows[np.random.default_rng(PLOT_SEED + label_index).permutation(len(rows))]
        right.scatter(
            data["UMAP1"].to_numpy()[rows],
            data["UMAP2"].to_numpy()[rows],
            c=TYPE_COLORS[label],
            s=1.0,
            alpha=0.50,
            linewidths=0,
            rasterized=True,
            label=label,
        )
    right.set_title("CELL TYPE", loc="left", fontsize=12.0, weight="bold", color=INK, pad=5)
    right.legend(
        frameon=False,
        markerscale=4.7,
        fontsize=7.8,
        loc="upper left",
        ncol=2,
        borderaxespad=0.1,
        handletextpad=0.35,
        columnspacing=0.9,
    )

    x_min, x_max = data["UMAP1"].min(), data["UMAP1"].max()
    y_min, y_max = data["UMAP2"].min(), data["UMAP2"].max()
    x_span = x_max - x_min
    y_span = y_max - y_min
    # Reserve an identical upper band in both views for their legends. This
    # moves the point clouds down without changing their relative geometry or
    # making the two panels use different coordinate systems.
    shared_xlim = (x_min - 0.02 * x_span, x_max + 0.02 * x_span)
    shared_ylim = (y_min - 0.02 * y_span, y_max + 0.25 * y_span)
    for axis in (left, right):
        axis.set_xlim(shared_xlim)
        axis.set_ylim(shared_ylim)
        clean_umap_axis(axis)

    png = HERE / "panel_C.png"
    pdf = HERE / "panel_C.pdf"
    fig.savefig(png, dpi=320, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
