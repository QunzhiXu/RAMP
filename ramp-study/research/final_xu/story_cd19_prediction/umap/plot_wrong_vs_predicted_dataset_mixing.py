#!/usr/bin/env python3
"""Visualize where predicted CD19 changes cross-dataset mixing.

The saved UMAP coordinates are not refit here.  The first row shows all cells;
the lower rows use dataset-balanced subsets of the two cell types whose mixing
improved consistently in both aligned marker space and UMAP space.
"""

from __future__ import annotations

import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIGURES = HERE / "figures"
CACHE = HERE / ".cache" / "dataset_mixing"
for folder in [DATA, FIGURES, CACHE]:
    folder.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


STORY = HERE.parent
METADATA_PATH = STORY / "alignment" / "output" / "cell_metadata.csv.gz"
COORDINATE_PATH = DATA / "umap_coordinates.npz"

DATASETS = ["BM CODEX", "BM CyTOF", "Triana AbSeq"]
DATASET_COLORS = {
    "BM CODEX": "#3E70B2",
    "BM CyTOF": "#55A868",
    "Triana AbSeq": "#E28B35",
}
CONDITIONS = ["with_wrong_observed_cd19", "with_predicted_cd19"]
CONDITION_TITLES = {
    "with_wrong_observed_cd19": "Observed CD19\n(failed CODEX channel)",
    "with_predicted_cd19": "Predicted CD19",
}
FOCUS_CELLTYPES = ["Mature / CODEX B", "CD4 T"]
FOCUS_LABELS = {
    "Mature / CODEX B": "Mature B cells",
    "CD4 T": "CD4 T cells",
}
SEED = 20260820
K = 30


def balanced_rows(metadata: pd.DataFrame, celltype: str) -> np.ndarray:
    """Select the same number of cells from every represented dataset."""

    labels = metadata["celltype"].astype(str).to_numpy()
    datasets = metadata["dataset"].astype(str).to_numpy()
    eligible = np.flatnonzero(labels == celltype)
    present = [dataset for dataset in DATASETS if np.any(datasets[eligible] == dataset)]
    if len(present) < 2:
        raise AssertionError(f"{celltype} is not represented in multiple datasets")
    n_per_dataset = min(int(np.sum(datasets[eligible] == dataset)) for dataset in present)
    rng = np.random.default_rng(SEED + FOCUS_CELLTYPES.index(celltype))
    selected = np.concatenate(
        [
            rng.choice(
                eligible[datasets[eligible] == dataset],
                size=n_per_dataset,
                replace=False,
            )
            for dataset in present
        ]
    )
    return selected[rng.permutation(len(selected))]


def cross_dataset_fraction(
    coordinates: np.ndarray,
    metadata: pd.DataFrame,
    rows: np.ndarray,
) -> float:
    """Fraction of within-cell-type 30-nearest neighbors from another dataset."""

    datasets = metadata["dataset"].astype(str).to_numpy()[rows]
    neighbors = NearestNeighbors(n_neighbors=K + 1, metric="euclidean", n_jobs=-1)
    neighbors.fit(coordinates[rows])
    indices = neighbors.kneighbors(return_distance=False)[:, 1:]
    return float(np.mean(datasets[indices] != datasets[:, None]))


def draw_scatter(
    axis: plt.Axes,
    coordinates: np.ndarray,
    metadata: pd.DataFrame,
    rows: np.ndarray,
    point_size: float,
    alpha: float,
) -> None:
    """Draw with a deterministic interleaved order to avoid dataset layer bias."""

    datasets = metadata["dataset"].astype(str).to_numpy()
    rng = np.random.default_rng(SEED + len(rows))
    rows = rows[rng.permutation(len(rows))]
    axis.scatter(
        coordinates[rows, 0],
        coordinates[rows, 1],
        s=point_size,
        c=[DATASET_COLORS[value] for value in datasets[rows]],
        alpha=alpha,
        linewidths=0,
        rasterized=True,
    )
    axis.set(xticks=[], yticks=[])
    axis.set_aspect("equal", adjustable="box")
    for spine in axis.spines.values():
        spine.set_visible(False)


def main() -> None:
    metadata = pd.read_csv(METADATA_PATH, low_memory=False)
    saved = np.load(COORDINATE_PATH)
    if any(condition not in saved.files for condition in CONDITIONS):
        raise AssertionError(f"Missing required coordinate keys: {saved.files}")
    coordinates = {condition: saved[condition] for condition in CONDITIONS}
    if any(len(values) != len(metadata) for values in coordinates.values()):
        raise AssertionError("UMAP coordinate and metadata row counts differ")

    focus_rows = {celltype: balanced_rows(metadata, celltype) for celltype in FOCUS_CELLTYPES}
    metric_records: list[dict[str, object]] = []
    for celltype, rows in focus_rows.items():
        counts = metadata.iloc[rows]["dataset"].value_counts()
        for condition in CONDITIONS:
            metric_records.append(
                {
                    "condition": condition,
                    "celltype": celltype,
                    "cells": len(rows),
                    "cells_per_dataset": int(counts.min()),
                    "cross_dataset_30nn_umap": cross_dataset_fraction(
                        coordinates[condition], metadata, rows
                    ),
                }
            )
    metrics = pd.DataFrame(metric_records)
    metrics.to_csv(DATA / "wrong_vs_predicted_dataset_mixing.csv", index=False)

    fig, axes = plt.subplots(3, 2, figsize=(10.8, 12.2), dpi=180)
    all_rows = np.arange(len(metadata), dtype=np.int64)
    for column, condition in enumerate(CONDITIONS):
        draw_scatter(
            axes[0, column], coordinates[condition], metadata, all_rows,
            point_size=0.42, alpha=0.55,
        )
        axes[0, column].set_title(
            CONDITION_TITLES[condition], fontsize=13.0, fontweight="bold", pad=8
        )

        for row_index, celltype in enumerate(FOCUS_CELLTYPES, start=1):
            rows = focus_rows[celltype]
            draw_scatter(
                axes[row_index, column], coordinates[condition], metadata, rows,
                point_size=1.25, alpha=0.70,
            )
            value = metrics.loc[
                (metrics["condition"] == condition) & (metrics["celltype"] == celltype),
                "cross_dataset_30nn_umap",
            ].iloc[0]
            axes[row_index, column].text(
                0.02,
                0.98,
                f"Cross-dataset 30-NN = {value:.3f}",
                transform=axes[row_index, column].transAxes,
                ha="left",
                va="top",
                fontsize=9.2,
                fontweight="bold",
                color="#222222",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.88},
            )

    axes[0, 0].set_ylabel("All cells", fontsize=11.5, fontweight="bold")
    for row_index, celltype in enumerate(FOCUS_CELLTYPES, start=1):
        n_per_dataset = int(
            metrics.loc[metrics["celltype"] == celltype, "cells_per_dataset"].iloc[0]
        )
        axes[row_index, 0].set_ylabel(
            f"{FOCUS_LABELS[celltype]}\n{n_per_dataset:,} cells per dataset",
            fontsize=11.0,
            fontweight="bold",
        )

    handles = [
        Line2D(
            [0], [0], marker="o", linestyle="", color=DATASET_COLORS[dataset],
            markersize=6.2, label=dataset,
        )
        for dataset in DATASETS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=3,
        frameon=False,
        fontsize=9.5,
    )
    fig.suptitle(
        "Where predicted CD19 improves cross-dataset mixing",
        x=0.035,
        y=0.992,
        ha="left",
        fontsize=16.0,
        fontweight="bold",
    )
    fig.text(
        0.985,
        0.965,
        "Dataset-balanced focus panels; identical cells compared between conditions",
        ha="right",
        va="top",
        fontsize=8.2,
        color="#555555",
    )
    fig.tight_layout(rect=(0.025, 0.060, 0.995, 0.950), h_pad=0.40, w_pad=0.30)
    fig.savefig(
        FIGURES / "wrong_vs_predicted_cd19_dataset_mixing.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        FIGURES / "wrong_vs_predicted_cd19_dataset_mixing.pdf",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
