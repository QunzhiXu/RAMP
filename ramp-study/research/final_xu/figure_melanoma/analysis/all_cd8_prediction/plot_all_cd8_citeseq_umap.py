#!/usr/bin/env python3
"""Plot the all-CD8 CITE-seq deployment UMAP."""

from __future__ import annotations

import json
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "citeseq_projection"
TABLE = DATA_DIR / "engelbrecht_all_cd8_projection.csv.gz"
OUT = DATA_DIR / "figures"

os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(OUT / ".xdg-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


COLORS = {
    "CD8ct": "#D73027",
    "CD8ct.CCL4": "#F46D43",
    "CD8ct.MT1E+": "#E78AC3",
    "CD8nv": "#2C7BB6",
    "Other CD8": "#8E8E8E",
}


def display_label(values: pd.Series) -> pd.Series:
    kept = {"CD8ct", "CD8ct.CCL4", "CD8ct.MT1E+", "CD8nv"}
    return values.astype(str).where(values.astype(str).isin(kept), "Other CD8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(TABLE, low_memory=False)
    xy = data[["umap_1", "umap_2"]].to_numpy(float)
    score = data["all_cd8_spatial_state_score"].to_numpy(float)
    labels = display_label(data["clust.name"]).to_numpy(str)

    plt.rcParams.update({"font.family": "Arial", "font.size": 8.5})
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.85), dpi=180)

    layer_order = ["Other CD8", "CD8nv", "CD8ct", "CD8ct.CCL4", "CD8ct.MT1E+"]
    for label in layer_order:
        use = labels == label
        axes[0].scatter(
            xy[use, 0], xy[use, 1], s=4.2, c=COLORS[label],
            alpha=0.82 if label != "Other CD8" else 0.48,
            linewidths=0, rasterized=True,
        )
    axes[0].set_title("A  Provider CD8 states", loc="left", fontsize=10.8, fontweight="bold", pad=6)
    handles = [
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=COLORS[label],
               markeredgecolor="none", label=label, markersize=5.2)
        for label in ["CD8ct", "CD8ct.CCL4", "CD8ct.MT1E+", "CD8nv", "Other CD8"]
    ]
    axes[0].legend(handles=handles, loc="lower left", frameon=False, fontsize=7.2,
                   handletextpad=0.35, borderpad=0.1, labelspacing=0.32)

    order = np.argsort(score, kind="mergesort")
    dots = axes[1].scatter(
        xy[order, 0], xy[order, 1], c=score[order], cmap="magma",
        vmin=0.0, vmax=0.9, s=4.3, alpha=0.92,
        linewidths=0, rasterized=True,
    )
    axes[1].set_title("B  Transferred spatial-state score", loc="left", fontsize=10.8, fontweight="bold", pad=6)
    cbar = fig.colorbar(dots, ax=axes[1], orientation="horizontal", fraction=0.055, pad=0.035, aspect=28, extend="max")
    cbar.set_ticks([0, 0.3, 0.6, 0.9])
    cbar.ax.tick_params(labelsize=6.8, length=2, pad=1)
    cbar.set_label("Predicted tumor-proximal Ki67+ CD8 state", fontsize=7.2, labelpad=1.5)
    cbar.outline.set_linewidth(0.45)

    for ax in axes:
        ax.set(xticks=[], yticks=[])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle(
        "All-CD8 spatial-state projection to Engelbrecht CITE-seq",
        x=0.025, y=0.985, ha="left", fontsize=14.0, fontweight="bold",
    )
    fig.text(
        0.025, 0.925,
        f"{len(data):,} provider-defined CD8 T cells · model trained on all spatial Tc.ae + Tc.naive cells",
        ha="left", va="top", fontsize=8.0, color="#4C4C4C",
    )
    fig.subplots_adjust(left=0.02, right=0.99, top=0.82, bottom=0.07, wspace=0.10)
    png = OUT / "engelbrecht_all_cd8_projection_umap.png"
    pdf = OUT / "engelbrecht_all_cd8_projection_umap.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    manifest = {
        "data": str(TABLE),
        "cells": int(len(data)),
        "patients": int(data["Patient_ID"].astype(str).nunique()),
        "color": "continuous all-CD8 spatial-state probability; magma; display vmax 0.9",
        "score_values_modified": False,
        "display_color_saturation_at": 0.9,
        "png": str(png),
        "pdf": str(pdf),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
