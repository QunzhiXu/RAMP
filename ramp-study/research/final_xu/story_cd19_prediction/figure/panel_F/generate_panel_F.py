#!/usr/bin/env python3
"""Generate Panel F: cell-coordinate validation of predicted CODEX CD19.

This is a post-hoc cell-centroid map, not a raw multiplexed tissue image.
Neither coordinates nor provider cell-type labels enter the prediction model.
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
import numpy as np
import pandas as pd


DONOR = "SB67_NBM28_H14_CODEX_Mesmer"
THRESHOLD = 1.4
BACKGROUND_VARIANT = os.environ.get("PANEL_F_BACKGROUND", "yellow").lower()
BACKGROUND_COLORS = {
    "yellow": "#FFD92F",
    "green": "#63BE7B",
}
if BACKGROUND_VARIANT not in BACKGROUND_COLORS:
    raise ValueError(f"Unknown PANEL_F_BACKGROUND={BACKGROUND_VARIANT!r}")
BACKGROUND = BACKGROUND_COLORS[BACKGROUND_VARIANT]
PREDICTED_COLOR = "#E04455"
B_COLOR = "#8E44AD"


def load() -> tuple[pd.DataFrame, np.ndarray]:
    metadata = pd.read_csv(
        STORY / "sampled_data" / "bm_codex" / "locked_metadata.csv.gz"
    )
    prediction_meta = pd.read_csv(
        STORY / "prediction" / "output" / "codex_prediction_metadata.csv.gz",
        usecols=["cell_id"],
    )
    prediction = np.load(
        STORY / "prediction" / "output" / "codex_cd19_prediction.npz"
    )["predicted_kmeanslow"].astype(float)

    if len(prediction_meta) != len(prediction):
        raise RuntimeError("Prediction metadata and values differ in length")
    if not metadata["cell_id"].astype(str).equals(prediction_meta["cell_id"].astype(str)):
        raise RuntimeError("Prediction and CODEX metadata cell orders differ")

    keep = metadata["donor_id"].astype(str).eq(DONOR).to_numpy()
    return metadata.loc[keep].reset_index(drop=True), prediction[keep]


def format_spatial_axis(
    ax: plt.Axes,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main() -> None:
    metadata, predicted = load()
    xy = metadata[["x.coord", "y.coord"]].to_numpy(float)
    provider_b = metadata["cluster_anno_l1"].astype(str).eq("B-Cells").to_numpy()
    predicted_high = predicted > THRESHOLD
    overlap = predicted_high & provider_b

    xpad = 0.015 * float(np.ptp(xy[:, 0]))
    ypad = 0.015 * float(np.ptp(xy[:, 1]))
    xlim = (float(xy[:, 0].min() - xpad), float(xy[:, 0].max() + xpad))
    ylim = (float(xy[:, 1].min() - ypad), float(xy[:, 1].max() + ypad))

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), dpi=220)
    fig.suptitle(
        "F.  Predicted CD19 recovers spatial B-cell locations",
        x=0.03,
        y=0.985,
        ha="left",
        va="top",
        fontsize=15.5,
        fontweight="bold",
    )

    axes[0].scatter(
        xy[~predicted_high, 0],
        xy[~predicted_high, 1],
        s=5.0,
        color=BACKGROUND,
        alpha=0.62,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )
    axes[0].scatter(
        xy[predicted_high, 0],
        xy[predicted_high, 1],
        s=10.5,
        color=PREDICTED_COLOR,
        alpha=0.92,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )
    axes[0].set_title(
        f"Predicted CD19 score > {THRESHOLD:g}  (n = {int(predicted_high.sum()):,})",
        fontsize=12.0,
        fontweight="bold",
        pad=7,
    )

    axes[1].scatter(
        xy[~provider_b, 0],
        xy[~provider_b, 1],
        s=5.0,
        color=BACKGROUND,
        alpha=0.62,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )
    axes[1].scatter(
        xy[provider_b, 0],
        xy[provider_b, 1],
        s=10.5,
        color=B_COLOR,
        alpha=0.92,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )
    axes[1].set_title(
        f"Provider-annotated B cells  (n = {int(provider_b.sum()):,})",
        fontsize=12.0,
        fontweight="bold",
        pad=7,
    )

    for ax in axes:
        format_spatial_axis(ax, xlim, ylim)

    fig.subplots_adjust(left=0.012, right=0.992, top=0.855, bottom=0.012, wspace=0.025)

    suffix = "" if BACKGROUND_VARIANT == "yellow" else f"_{BACKGROUND_VARIANT}_background"
    png = HERE / f"panel_F_cell_coordinate_validation{suffix}.png"
    pdf = HERE / f"panel_F_cell_coordinate_validation{suffix}.pdf"
    fig.savefig(png, dpi=320, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)

    n_high = int(predicted_high.sum())
    n_b = int(provider_b.sum())
    n_overlap = int(overlap.sum())
    audit = {
        "donor": DONOR,
        "n_sampled_cells": int(len(metadata)),
        "predicted_cd19_threshold": THRESHOLD,
        "n_predicted_high": n_high,
        "n_provider_b": n_b,
        "n_overlap": n_overlap,
        "precision_provider_b_among_predicted_high": n_overlap / n_high,
        "recall_of_provider_b": n_overlap / n_b,
        "map_type": "cell-centroid coordinate map; not raw multiplexed image",
        "coordinates_or_labels_used_for_prediction": False,
    }
    audit["background_variant"] = BACKGROUND_VARIANT
    (HERE / f"panel_F_audit{suffix}.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    print(f"Saved {png}")
    print(f"Saved {pdf}")


if __name__ == "__main__":
    main()
