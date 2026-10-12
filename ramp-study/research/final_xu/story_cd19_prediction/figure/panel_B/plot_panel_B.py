#!/usr/bin/env python3
"""Draw Panel B from the frozen donor-level QC results."""

from __future__ import annotations

import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".cache"
CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE))

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA = HERE / "data"
COLORS = {
    "CD19": "#D81B60",
    "CD61": "#4477AA",
    "CD45RA": "#66A61E",
    "CD33": "#66A61E",
}


def main() -> None:
    scores = pd.read_csv(DATA / "codex_top3_marker_donor_scores.csv")
    summary = pd.read_csv(DATA / "codex_marker_qc_summary.csv")
    manifest = json.loads((DATA / "manifest.json").read_text())
    order = manifest["top3_markers"]
    donors = sorted(scores.unit_key.unique())
    offsets = np.linspace(-0.15, 0.15, len(donors))

    pivot = scores.pivot(index="unit_key", columns="marker", values="qc_alarm").loc[donors, order]
    fig, ax = plt.subplots(figsize=(5.4, 3.05), dpi=220)
    for donor_index, donor in enumerate(donors):
        x = np.arange(len(order), dtype=float) + offsets[donor_index]
        ax.plot(x, pivot.loc[donor].to_numpy(float), color="#A6A6A6", alpha=0.23, lw=0.7, zorder=1)
    for marker_index, marker in enumerate(order):
        values = pivot[marker].to_numpy(float)
        ax.scatter(
            marker_index + offsets, values,
            s=36, color=COLORS.get(marker, "#777777"), edgecolor="white",
            linewidth=0.45, alpha=0.94, zorder=3,
        )
        row = summary.loc[summary.marker.eq(marker)].iloc[0]
        label = f"{int(row.top1_donors)}/12 ranked #1"
        ax.text(
            marker_index, 1.035, label, ha="center", va="bottom",
            fontsize=8.4, fontweight="bold" if marker == "CD19" else "normal",
            color=COLORS.get(marker, "#555555"),
        )

    ax.set_xlim(-0.42, len(order) - 0.58)
    ax.set_ylim(0.50, 1.105)
    ax.set_xticks(range(len(order)), order, fontsize=11, fontweight="bold")
    ax.set_ylabel("Marker anomaly score", fontsize=10.5, fontweight="bold")
    ax.set_title("B.  Label-free marker QC", loc="left", fontsize=12.2, fontweight="bold", pad=9)
    ax.grid(axis="y", color="#D9D9D9", lw=0.65, alpha=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#555555")
    ax.tick_params(axis="y", labelsize=8.8)
    ax.text(
        0.0, -0.19,
        "Each point is one donor; lines connect the same donor across markers.",
        transform=ax.transAxes, ha="left", va="top", fontsize=8.2,
    )
    fig.subplots_adjust(left=0.145, right=0.985, top=0.80, bottom=0.22)
    fig.savefig(HERE / "panel_B_automatic_marker_qc.png", dpi=300, bbox_inches="tight")
    fig.savefig(HERE / "panel_B_automatic_marker_qc.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
