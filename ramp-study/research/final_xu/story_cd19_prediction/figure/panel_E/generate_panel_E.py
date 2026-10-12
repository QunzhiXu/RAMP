#!/usr/bin/env python3
"""Generate Panel E: permanent-marker-holdout validation before CD19 prediction."""

from __future__ import annotations

import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
CACHE = HERE / ".cache"
CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SUMMARY_FILE = DATA / "heldout_marker_validation_summary.csv"
DONOR_FILE = DATA / "heldout_marker_validation_all_methods_by_donor.csv"
METHOD = "Equal dual-source"
SUMMARY_COLOR = "#3973B7"


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(SUMMARY_FILE)
    donor = pd.read_csv(DONOR_FILE)
    donor = donor.loc[donor["method"].eq(METHOD)].copy()
    expected = set(summary["marker"])
    if set(donor["marker"]) != expected:
        raise RuntimeError("Summary and donor-level marker sets differ")
    if donor["CODEX_donor"].nunique() != 3:
        raise RuntimeError("Expected the frozen three-donor outer validation")

    check = donor.groupby("marker", as_index=False).agg(
        median_donor_spearman=("spearman", "median"),
        mean_high_state_auroc=("high_state_auroc", "mean"),
    )
    merged = summary.merge(check, on="marker", suffixes=("_frozen", "_recomputed"))
    for metric in ("median_donor_spearman", "mean_high_state_auroc"):
        if not np.allclose(
            merged[f"{metric}_frozen"],
            merged[f"{metric}_recomputed"],
            atol=1e-8,
        ):
            raise RuntimeError(f"Frozen summary mismatch for {metric}")
    return summary, donor


def metric_axis(
    ax: plt.Axes,
    summary: pd.DataFrame,
    order: list[str],
    summary_column: str,
    xlabel: str,
    xlim: tuple[float, float],
    aggregate_text: str,
) -> None:
    y = np.arange(len(order))
    summary_values = summary.set_index("marker").loc[order, summary_column].to_numpy(float)
    ax.scatter(
        summary_values,
        y,
        s=62,
        color=SUMMARY_COLOR,
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.55, len(order) - 0.45)
    ax.set_yticks(y, order, fontsize=9.2)
    ax.set_xlabel(xlabel, fontsize=10.3, fontweight="bold", labelpad=6)
    ax.grid(axis="x", color="#D9DEE3", linewidth=0.75, alpha=0.75, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#5D6268")
    ax.tick_params(axis="x", labelsize=9.0)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.text(
        0.98,
        0.025,
        aggregate_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.2,
        fontweight="bold",
        color=SUMMARY_COLOR,
    )


def main() -> None:
    summary, donor = load()
    order = (
        summary.sort_values("median_donor_spearman")["marker"].astype(str).tolist()
    )
    overall_rho = float(summary["median_donor_spearman"].median())
    overall_auc = float(summary["mean_high_state_auroc"].mean())

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.35), dpi=220)
    fig.suptitle(
        "E.  Held-out marker validation",
        x=0.03,
        y=0.985,
        ha="left",
        va="top",
        # Panel E is intrinsically taller than D and is therefore downscaled
        # more in the assembled figure.  Compensate here so the displayed
        # panel-title size matches the other panels after assembly.
        fontsize=18.5,
        fontweight="bold",
    )
    metric_axis(
        axes[0],
        summary,
        order,
        "median_donor_spearman",
        "Within-donor Spearman ρ",
        (0.08, 0.84),
        f"median across markers = {overall_rho:.3f}",
    )
    metric_axis(
        axes[1],
        summary,
        order,
        "mean_high_state_auroc",
        "High-state AUROC",
        (0.48, 0.98),
        f"mean across markers = {overall_auc:.3f}",
    )
    axes[1].axvline(
        0.5,
        color="#777777",
        linewidth=1.0,
        linestyle=(0, (4, 3)),
        zorder=1,
    )
    axes[1].text(
        0.505,
        len(order) - 0.65,
        "chance",
        ha="left",
        va="top",
        fontsize=8.4,
        color="#666666",
    )

    fig.subplots_adjust(left=0.095, right=0.985, top=0.88, bottom=0.12, wspace=0.35)

    png = HERE / "panel_E_heldout_marker_validation.png"
    pdf = HERE / "panel_E_heldout_marker_validation.pdf"
    fig.savefig(png, dpi=320, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    print(f"median across marker-level Spearman summaries: {overall_rho:.6f}")
    print(f"mean across marker-level high-state AUROC summaries: {overall_auc:.6f}")
    print(f"Saved {png}")
    print(f"Saved {pdf}")


if __name__ == "__main__":
    main()
