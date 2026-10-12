#!/usr/bin/env python3
"""Fit and plot UMAP using every one of the 90,488 aligned test cells."""

from __future__ import annotations

import json
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
INPUT = HERE / "input"
ALIGNMENT = HERE / "alignment"
OUT = HERE / "umap"
CACHE = HERE / ".cache"
os.environ["MPLCONFIGDIR"] = str(CACHE / "matplotlib")
os.environ["NUMBA_CACHE_DIR"] = str(CACHE / "numba")
os.environ["XDG_CACHE_HOME"] = str(CACHE / "xdg")
os.environ["JOBLIB_TEMP_FOLDER"] = str(CACHE / "joblib")
os.environ["TMPDIR"] = str(CACHE / "tmp")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from umap import UMAP


UMAP_SEED = 20260824
PLOT_SEED = 20260902
N_NEIGHBORS = 30
MIN_DIST = 0.25

DATASET_COLORS = {
    "GSE212797_ADT": "#0072B2",
    "Haist_CODEX": "#E69F00",
}
DATASET_NAMES = {
    "GSE212797_ADT": "Rahim CITE-seq",
    "Haist_CODEX": "Haist CODEX (PT)",
}
DISPLAY_ORDER = [
    "Epithelial / tumor",
    "Other",
    "Stromal",
    "Vascular",
    "Myeloid",
    "NK / gdT",
    "Plasma",
    "B",
    "CD4 T",
    "CD8 T",
]
TYPE_COLORS = {
    "CD8 T": "#D55E00",
    "CD4 T": "#CC79A7",
    "B": "#0072B2",
    "Plasma": "#56B4E9",
    "NK / gdT": "#009E73",
    "Myeloid": "#E5C100",
    "Epithelial / tumor": "#B2182B",
    "Stromal": "#8C6BB1",
    "Vascular": "#17BECF",
    "Other": "#BDBDBD",
}


def clean_axis(axis: plt.Axes) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    aligned = np.load(ALIGNMENT / "aligned_ramp_ver2.float32.npy", mmap_mode="r")
    metadata = pd.read_csv(INPUT / "cell_metadata.csv.gz", low_memory=False)
    if aligned.shape[0] != len(metadata) or len(metadata) != 90_488:
        raise AssertionError("The full UMAP input must contain exactly 90,488 aligned cells")

    scaled = StandardScaler().fit_transform(np.asarray(aligned, dtype=np.float64))
    coordinates = UMAP(
        n_neighbors=N_NEIGHBORS,
        min_dist=MIN_DIST,
        metric="euclidean",
        random_state=UMAP_SEED,
        n_jobs=1,
    ).fit_transform(scaled)
    metadata["UMAP1"] = coordinates[:, 0]
    metadata["UMAP2"] = coordinates[:, 1]
    metadata.to_csv(OUT / "full_90488_umap_coordinates.csv.gz", index=False, compression="gzip")

    # Shuffle plotting order only, so neither dataset is forced above the other.
    plot_order = np.random.default_rng(PLOT_SEED).permutation(len(metadata))
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.6))

    dataset_colors = metadata["dataset"].map(DATASET_COLORS).to_numpy()[plot_order]
    axes[0].scatter(
        coordinates[plot_order, 0],
        coordinates[plot_order, 1],
        c=dataset_colors,
        s=1.15,
        alpha=0.48,
        linewidths=0,
        rasterized=True,
    )
    axes[0].set_title("Dataset", loc="left", weight="bold", fontsize=12)
    dataset_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=6,
               markerfacecolor=DATASET_COLORS[key], markeredgewidth=0,
               label=f"{DATASET_NAMES[key]} (n={int(metadata['dataset'].eq(key).sum()):,})")
        for key in DATASET_COLORS
    ]
    axes[0].legend(handles=dataset_handles, frameon=False, fontsize=8.5, loc="upper right")

    for label in DISPLAY_ORDER:
        rows = np.flatnonzero(metadata["display_label"].astype(str).eq(label).to_numpy())
        if len(rows) == 0:
            continue
        rows = rows[np.random.default_rng(PLOT_SEED + DISPLAY_ORDER.index(label)).permutation(len(rows))]
        axes[1].scatter(
            coordinates[rows, 0],
            coordinates[rows, 1],
            c=TYPE_COLORS[label],
            s=1.15,
            alpha=0.50,
            linewidths=0,
            rasterized=True,
            label=label,
        )
    axes[1].set_title("Provider cell type (harmonized for display)", loc="left", weight="bold", fontsize=12)
    axes[1].legend(frameon=False, markerscale=4.5, fontsize=7.5, loc="upper right", ncol=2)

    for axis in axes:
        clean_axis(axis)
    fig.suptitle(
        "HNSCC PT-focused RAMP alignment — all 90,488 cells shown",
        x=0.025,
        y=0.995,
        ha="left",
        fontsize=14,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955), w_pad=0.8)
    png = OUT / "hnscc_pt70_ramp_ver2_full_90488_umap.png"
    pdf = OUT / "hnscc_pt70_ramp_ver2_full_90488_umap.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    counts = (
        metadata.groupby(["dataset", "display_label"], as_index=False)
        .size()
        .rename(columns={"size": "cells"})
    )
    counts.to_csv(OUT / "display_cell_counts.csv", index=False)
    manifest = {
        "matrix": "RAMP 2.3.0 aligned 22-marker matrix",
        "fit_scope": "all 90,488 cells; no UMAP display downsampling",
        "cells": int(len(metadata)),
        "dataset_cells": metadata["dataset"].value_counts(sort=False).astype(int).to_dict(),
        "pre_umap_scaling": "StandardScaler fit on all 90,488 aligned cells",
        "umap": {
            "n_neighbors": N_NEIGHBORS,
            "min_dist": MIN_DIST,
            "metric": "euclidean",
            "random_state": UMAP_SEED,
        },
        "labels_used_for_embedding": False,
        "plot_order_random_state": PLOT_SEED,
        "outputs": {"png": str(png), "pdf": str(pdf)},
    }
    (OUT / "umap_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
