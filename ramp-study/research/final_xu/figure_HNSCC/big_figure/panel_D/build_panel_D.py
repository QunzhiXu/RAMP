#!/usr/bin/env python3
"""Build Panel D: transferred niche-score biomarker discovery in CITE-seq."""

from __future__ import annotations

import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
HNSCC_ROOT = HERE.parent.parent
SCREEN = HNSCC_ROOT / "biomarker_discovery" / "screen"
MARKER_SCREEN = SCREEN / "marker_screen_all_107_all_variants.csv"
DONOR_EFFECTS = SCREEN / "marker_donor_effects_long.csv"
CACHE = HERE / ".cache"
for child in ("matplotlib", "xdg"):
    (CACHE / child).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE / "xdg"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


INK = "#182027"
MUTED = "#59636C"
GRID = "#E4E8EB"
ALL = "#C8CDD1"
FINAL = "#D95F02"
PALE_FINAL = "#FFF0DF"
DONOR_COLORS = {
    "126": "#0072B2",
    "indiv_1": "#D55E00",
    "indiv_2": "#009E73",
}
DONOR_DISPLAY = {
    "126": "Donor 1",
    "indiv_1": "Donor 2",
    "indiv_2": "Donor 3",
}
# Tiny display-only vertical offsets prevent donors with nearly identical
# associations (notably CD39 Donor 1/2) from hiding one another.
DONOR_Y_OFFSETS = {
    "126": 0.045,
    "indiv_1": -0.045,
    "indiv_2": 0.0,
}
FINAL_MARKERS = ["TIGIT--VSTM3", "CD39"]
DISPLAY = {"TIGIT--VSTM3": "TIGIT", "CD39": "CD39"}


def add_flow_box(
    figure: plt.Figure,
    x: float,
    width: float,
    text: str,
    facecolor: str,
    edgecolor: str,
    textcolor: str = INK,
) -> None:
    height = 0.067
    y = 0.805
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        transform=figure.transFigure,
        linewidth=0.9,
        edgecolor=edgecolor,
        facecolor=facecolor,
        clip_on=False,
    )
    figure.add_artist(box)
    figure.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=6.2,
        weight="bold",
        color=textcolor,
        linespacing=0.95,
    )


def main() -> None:
    screen = pd.read_csv(MARKER_SCREEN)
    effects = pd.read_csv(DONOR_EFFECTS, dtype={"donor": str})
    if len(screen) != 107 or screen.marker_raw.duplicated().any():
        raise AssertionError("Expected the frozen 107-protein screen")
    robust = screen.loc[screen.selected_after_redundancy_pruning].copy()
    final = screen.loc[screen.selected_compact_top5pct].copy()
    if len(robust) != 8:
        raise AssertionError(f"Expected eight donor-robust proteins; observed {len(robust)}")
    if set(final.marker_raw) != set(FINAL_MARKERS):
        raise AssertionError("Expected TIGIT and CD39 as the compact program")
    donor = effects.loc[
        effects.variant.eq("paper_clr") & effects.marker_raw.isin(FINAL_MARKERS)
    ].copy()
    if len(donor) != 6 or set(donor.donor) != set(DONOR_COLORS):
        raise AssertionError("Expected TIGIT/CD39 effects in all three Rahim donors")
    if not (donor.association > 0).all():
        raise AssertionError("TIGIT/CD39 must be positively associated in all donors")

    # Match Panel A's native width so A+B and C+D form equal-width rows.
    fig = plt.figure(figsize=(5.575, 4.6), facecolor="white")
    fig.text(0.010, 0.970, "D", ha="left", va="top", fontsize=24, weight="bold", color=INK)
    fig.text(
        0.105,
        0.943,
        "NICHE SCORE IDENTIFIES TIGIT/CD39",
        ha="left",
        va="top",
        fontsize=11.3,
        weight="bold",
        color=INK,
    )

    # A single, readable selection path above the measured-data panels.
    add_flow_box(fig, 0.100, 0.220, "TRANSFERRED\nNICHE SCORE", "#EAF5F2", "#88BCAF")
    add_flow_box(fig, 0.400, 0.230, "107 ADDITIONAL\nPROTEINS", "#F3F5F6", "#C7CDD2")
    add_flow_box(fig, 0.715, 0.205, "TIGIT + CD39", PALE_FINAL, "#E3A064")
    for start, stop in ((0.320, 0.400), (0.630, 0.715)):
        fig.add_artist(
            FancyArrowPatch(
                (start + 0.006, 0.8385),
                (stop - 0.006, 0.8385),
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=11,
                color="#7C858C",
                linewidth=1.15,
                clip_on=False,
            )
        )

    rank_axis = fig.add_axes([0.070, 0.125, 0.465, 0.595])
    # Keep the two evidence panels on exactly the same vertical baseline and height.
    donor_axis = fig.add_axes([0.590, 0.125, 0.370, 0.595])

    # The compact rule is top 5% on the original candidate scale.
    rank_axis.axvspan(0.0, 5.35, color=PALE_FINAL, alpha=0.90, zorder=0)
    rank_axis.axvline(5.35, color=FINAL, linestyle=(0, (4, 3)), linewidth=1.0, zorder=1)

    displayed = screen.robust_median_rank_primary_variants <= 10
    nonfinal = displayed & ~screen.selected_compact_top5pct
    rank_axis.scatter(
        screen.loc[nonfinal, "robust_median_rank_primary_variants"],
        screen.loc[nonfinal, "paper_clr__mean_donor_association"],
        s=19,
        color=ALL,
        alpha=0.72,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )
    rank_axis.scatter(
        final.robust_median_rank_primary_variants,
        final.paper_clr__mean_donor_association,
        s=54,
        color=FINAL,
        edgecolor="white",
        linewidth=0.65,
        zorder=4,
    )

    positions = {
        "TIGIT--VSTM3": (5.15, 0.221),
        "CD39": (5.15, 0.281),
    }
    for marker in FINAL_MARKERS:
        row = final.loc[final.marker_raw.eq(marker)].iloc[0]
        tx, ty = positions[marker]
        rank_axis.annotate(
            DISPLAY[marker],
            xy=(row.robust_median_rank_primary_variants, row.paper_clr__mean_donor_association),
            xytext=(tx, ty),
            fontsize=8.0,
            weight="bold",
            color=INK,
            ha="left",
            va="center",
            arrowprops=dict(arrowstyle="-", color=FINAL, lw=0.9, shrinkA=2, shrinkB=4),
            zorder=5,
        )

    rank_axis.text(
        0.10,
        0.057,
        "TOP 5%",
        ha="left",
        va="bottom",
        fontsize=6.7,
        weight="bold",
        color=FINAL,
    )
    rank_axis.set_xlim(0, 10.25)
    rank_axis.set_ylim(0.045, 0.31)
    rank_axis.set_xlabel("Biomarker rank  (1 = strongest)", fontsize=7.9)
    rank_axis.set_ylabel("Association with niche score", fontsize=7.9)
    rank_axis.set_title("TOP-RANKED CANDIDATES", loc="left", fontsize=8.8, weight="bold", color=INK, pad=5)
    rank_axis.tick_params(axis="both", labelsize=6.8, length=2.5)
    rank_axis.set_yticks([])
    rank_axis.spines[["top", "right"]].set_visible(False)
    rank_axis.spines[["left", "bottom"]].set_color("#7A838A")
    rank_axis.spines[["left", "bottom"]].set_linewidth(0.8)
    y_positions = {"TIGIT--VSTM3": 1.0, "CD39": 0.0}
    for marker in FINAL_MARKERS:
        block = donor.loc[donor.marker_raw.eq(marker)].copy()
        y = y_positions[marker]
        donor_axis.hlines(
            y,
            0,
            block.association.max(),
            color="#CBD1D5",
            linewidth=1.1,
            zorder=1,
        )
        for row in block.itertuples(index=False):
            donor_axis.scatter(
                row.association,
                y + DONOR_Y_OFFSETS[row.donor],
                s=33,
                color=DONOR_COLORS[row.donor],
                edgecolor="white",
                linewidth=0.55,
                zorder=3,
            )
        donor_axis.scatter(
            block.association.mean(),
            y,
            marker="D",
            s=34,
            color=INK,
            edgecolor="white",
            linewidth=0.55,
            zorder=4,
        )

    donor_axis.axvline(0, color="#8B9399", linestyle=(0, (4, 3)), linewidth=0.85)
    donor_axis.set_xlim(-0.012, 0.35)
    donor_axis.set_ylim(-0.58, 1.52)
    donor_axis.set_yticks([1.0, 0.0], ["TIGIT", "CD39"])
    donor_axis.set_xlabel("Association with transferred\nniche score", fontsize=7.0)
    donor_axis.set_title(
        "3-DONOR CONSISTENCY",
        loc="left",
        fontsize=8.8,
        weight="bold",
        color=INK,
        pad=5,
    )
    donor_axis.tick_params(axis="x", labelsize=6.7, length=2.5)
    donor_axis.tick_params(axis="y", labelsize=7.5, length=0, pad=4)
    donor_axis.grid(axis="x", color=GRID, linewidth=0.65, zorder=0)
    donor_axis.spines[["top", "right", "left"]].set_visible(False)
    donor_axis.spines["bottom"].set_color("#7A838A")
    donor_axis.spines["bottom"].set_linewidth(0.8)
    donor_axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=color,
                markeredgewidth=0,
                markersize=5.0,
                label=DONOR_DISPLAY[name],
            )
            for name, color in DONOR_COLORS.items()
        ]
        + [
            Line2D(
                [0],
                [0],
                marker="D",
                linestyle="",
                markerfacecolor=INK,
                markeredgewidth=0,
                markersize=4.5,
                label="Mean",
            )
        ],
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(-0.07, 0.995),
        ncol=2,
        fontsize=5.9,
        handletextpad=0.35,
        columnspacing=0.9,
    )

    png = HERE / "panel_D.png"
    pdf = HERE / "panel_D.pdf"
    fig.savefig(png, dpi=320, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
