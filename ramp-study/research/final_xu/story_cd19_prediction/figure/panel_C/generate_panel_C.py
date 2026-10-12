#!/usr/bin/env python3
"""Generate Panel C: observed versus predicted CODEX CD19 ridgelines.

The prediction arrays are read from the formal RAMP 2.3.0 story output.  Provider
cell-type labels are used only after prediction to identify the three displayed
populations; they were not inputs to RAMP or to the CD19 prediction model.
"""

from __future__ import annotations

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
from scipy.stats import gaussian_kde


PREDICTION_DIR = STORY / "prediction" / "output"
CODEX_DIR = STORY / "sampled_data" / "bm_codex"

GROUPS = [
    ("CODEX B", "B-Cells", "#A55AA1"),
    ("CD4 T", "CD4+ T-Cell", "#36B8B4"),
    ("CD8 T", "CD8+ T-Cell", "#294A76"),
]


def load_values() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    prediction = np.load(PREDICTION_DIR / "codex_cd19_prediction.npz")
    prediction_meta = pd.read_csv(
        PREDICTION_DIR / "codex_prediction_metadata.csv.gz",
        usecols=["cell_id"],
    )
    labels = pd.read_csv(
        CODEX_DIR / "locked_metadata.csv.gz",
        usecols=["cell_id", "cluster_anno_l2"],
    )

    if len(prediction_meta) != len(prediction["observed_kmeanslow"]):
        raise ValueError("Prediction metadata and CD19 arrays have different lengths")
    merged = prediction_meta.merge(
        labels,
        on="cell_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if merged["cluster_anno_l2"].isna().any():
        raise ValueError("Some predicted CODEX cells lack provider cell-type labels")

    observed: dict[str, np.ndarray] = {}
    predicted: dict[str, np.ndarray] = {}
    for display, provider_label, _ in GROUPS:
        keep = merged["cluster_anno_l2"].eq(provider_label).to_numpy()
        observed[display] = prediction["observed_kmeanslow"][keep].astype(float)
        predicted[display] = prediction["predicted_kmeanslow"][keep].astype(float)
    return observed, predicted


def density(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    values = values[np.isfinite(values)]
    if values.size < 2 or np.ptp(values) == 0:
        return np.zeros_like(grid)
    return gaussian_kde(values, bw_method=0.20)(grid)


def draw_ridges(
    ax: plt.Axes,
    values_by_group: dict[str, np.ndarray],
    grid: np.ndarray,
    title: str,
) -> None:
    baselines = np.array([2.0, 1.0, 0.0])
    for baseline, (display, _, color) in zip(baselines, GROUPS):
        values = values_by_group[display]
        y = density(values, grid)
        if y.max() > 0:
            y = 0.78 * y / y.max()
        ax.fill_between(
            grid,
            baseline,
            baseline + y,
            color=color,
            alpha=0.86,
            linewidth=0,
            zorder=2,
        )
        ax.plot(grid, baseline + y, color=color, linewidth=1.35, zorder=3)
        ax.hlines(
            baseline,
            grid.min(),
            grid.max(),
            color="#D8D8D8",
            linewidth=0.75,
            zorder=1,
        )
        ax.plot(
            np.median(values),
            baseline,
            marker="|",
            markersize=8,
            markeredgewidth=1.7,
            color="#1E1E1E",
            zorder=4,
        )

    ax.set_yticks(baselines + 0.18)
    ax.set_yticklabels([g[0] for g in GROUPS], fontsize=10.5, fontweight="bold")
    ax.set_ylim(-0.12, 2.96)
    ax.set_xlim(grid.min(), grid.max())
    ax.tick_params(axis="x", labelsize=9, length=3, color="#555555")
    ax.tick_params(axis="y", length=0, pad=6)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#555555")
    ax.set_title(title, loc="left", fontsize=13.5, fontweight="bold", pad=10)


def main() -> None:
    observed, predicted = load_values()
    pooled = np.concatenate(
        [*observed.values(), *predicted.values()]
    )
    lower, upper = np.quantile(pooled[np.isfinite(pooled)], [0.002, 0.998])
    lower = min(-1.5, float(lower))
    upper = max(3.2, float(upper))
    padding = 0.04 * (upper - lower)
    grid = np.linspace(lower - padding, upper + padding, 500)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(4.65, 7.75),
        dpi=220,
        sharex=True,
    )
    draw_ridges(
        axes[0],
        observed,
        grid,
        "Observed CODEX CD19",
    )
    draw_ridges(
        axes[1],
        predicted,
        grid,
        "Predicted CODEX CD19",
    )
    axes[1].set_xlabel("KMEANSLOW CD19 value", fontsize=10.5, labelpad=7)

    fig.suptitle(
        "C.  CD19 restoration",
        x=0.055,
        y=0.985,
        ha="left",
        va="top",
        fontsize=15.5,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.25, right=0.975, top=0.91, bottom=0.07, hspace=0.28)

    png = HERE / "panel_C_cd19_ridges_before_after.png"
    pdf = HERE / "panel_C_cd19_ridges_before_after.pdf"
    fig.savefig(png, dpi=320, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)

    print(f"Saved {png}")
    print(f"Saved {pdf}")
    for display, _, _ in GROUPS:
        print(
            f"{display}: n={len(observed[display]):,}, "
            f"observed median={np.median(observed[display]):.3f}, "
            f"predicted median={np.median(predicted[display]):.3f}"
        )


if __name__ == "__main__":
    main()
