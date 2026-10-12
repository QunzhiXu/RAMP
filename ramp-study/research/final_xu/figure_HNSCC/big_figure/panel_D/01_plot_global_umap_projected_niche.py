#!/usr/bin/env python3
"""Color Rahim tumour CD8 cells by transferred niche score on the full UMAP."""

from __future__ import annotations

import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
HNSCC_ROOT = HERE.parent.parent
COORDINATES = HNSCC_ROOT / "alignment" / "umap" / "full_94488_umap_coordinates.csv.gz"
PROJECTION = (
    HNSCC_ROOT
    / "biomarker_discovery"
    / "projection"
    / "rahim_tumor_cd8_projected_scores.csv.gz"
)
CACHE = HERE / ".cache"
for child in ("matplotlib", "xdg"):
    (CACHE / child).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE / "xdg"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


INK = "#182027"
BACKGROUND = "#C9CFD4"
SCORE_CMAP = LinearSegmentedColormap.from_list(
    "transferred_niche",
    ["#253494", "#2C7FB8", "#7FCDBB", "#FEC44F", "#D7301F"],
)
EXPECTED_ALL = 94_488
EXPECTED_CD8 = 4_788


def main() -> None:
    coordinates = pd.read_csv(COORDINATES, low_memory=False)
    projected = pd.read_csv(PROJECTION, dtype={"donor": str}, low_memory=False)
    if len(coordinates) != EXPECTED_ALL or not coordinates["cell_uid"].is_unique:
        raise AssertionError("The frozen full HNSCC UMAP has changed")
    if len(projected) != EXPECTED_CD8 or projected["cell_uid"].duplicated().any():
        raise AssertionError("The formal Rahim tumour-CD8 projection has changed")

    rahim_cd8 = coordinates.loc[
        coordinates["dataset"].eq("GSE212797_ADT")
        & coordinates["tissue"].eq("tumor")
        & coordinates["provider_cell_type"].eq("CD8 T cell"),
        ["cell_uid", "donor", "UMAP1", "UMAP2"],
    ].copy()
    rahim_cd8["cell_uid"] = rahim_cd8["cell_uid"].str.removeprefix("RAHIM|")
    rahim_cd8["donor"] = rahim_cd8["donor"].astype(str)
    rahim_cd8 = rahim_cd8.merge(
        projected[["cell_uid", "donor", "projected_rank_within_rahim_donor"]],
        on=["cell_uid", "donor"],
        how="left",
        validate="one_to_one",
    )
    if len(rahim_cd8) != EXPECTED_CD8:
        raise AssertionError(f"Expected {EXPECTED_CD8:,} Rahim tumour CD8 cells")
    if rahim_cd8["projected_rank_within_rahim_donor"].isna().any():
        raise AssertionError("Some highlighted cells lack a transferred niche score")

    scores = rahim_cd8["projected_rank_within_rahim_donor"].to_numpy(dtype=float)
    order = np.argsort(scores, kind="stable")

    fig = plt.figure(figsize=(7.2, 5.4), facecolor="white")
    fig.text(0.020, 0.975, "D", ha="left", va="top",
             fontsize=24, weight="bold", color=INK)
    fig.text(
        0.105,
        0.947,
        "SPATIAL-NICHE SCORE PROJECTED TO CITE-seq CD8 T CELLS",
        ha="left",
        va="top",
        fontsize=13.2,
        weight="bold",
        color=INK,
    )
    fig.text(
        0.105,
        0.895,
        "Rahim tumour CD8 T cells (n=4,788); all other aligned cells shown in gray",
        ha="left",
        va="top",
        fontsize=8.8,
        color="#4F5962",
    )

    axis = fig.add_axes([0.065, 0.075, 0.790, 0.785])
    axis.scatter(
        coordinates["UMAP1"],
        coordinates["UMAP2"],
        s=0.72,
        c=BACKGROUND,
        alpha=0.19,
        linewidths=0,
        rasterized=True,
    )
    scatter = axis.scatter(
        rahim_cd8["UMAP1"].to_numpy()[order],
        rahim_cd8["UMAP2"].to_numpy()[order],
        c=scores[order],
        cmap=SCORE_CMAP,
        vmin=0.0,
        vmax=1.0,
        s=5.0,
        alpha=0.92,
        linewidths=0,
        rasterized=True,
    )
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_aspect("equal", adjustable="box")
    for spine in axis.spines.values():
        spine.set_visible(False)

    background_handle = Line2D(
        [0], [0], marker="o", linestyle="", markersize=5.2,
        markerfacecolor=BACKGROUND, markeredgewidth=0,
        label="Other aligned cells",
    )
    axis.legend(
        handles=[background_handle],
        frameon=False,
        loc="lower left",
        fontsize=8.2,
        borderaxespad=0.2,
    )

    color_axis = fig.add_axes([0.885, 0.205, 0.025, 0.535])
    colorbar = fig.colorbar(scatter, cax=color_axis)
    colorbar.set_ticks([0.0, 0.25, 0.50, 0.75, 1.0])
    colorbar.set_label("Transferred niche-score rank", fontsize=9.2, labelpad=8)
    colorbar.ax.tick_params(labelsize=8.3, length=2.5)
    colorbar.outline.set_linewidth(0.6)

    png = HERE / "candidate_global_umap_projected_niche.png"
    pdf = HERE / "candidate_global_umap_projected_niche.pdf"
    fig.savefig(png, dpi=320, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
