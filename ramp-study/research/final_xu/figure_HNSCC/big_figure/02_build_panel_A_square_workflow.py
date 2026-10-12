#!/usr/bin/env python3
"""Build a compact square workflow panel for the HNSCC story."""

from __future__ import annotations

import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUT = HERE / "panels"
CACHE = HERE / ".cache"
for child in ("matplotlib", "xdg"):
    (CACHE / child).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE / "xdg"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


GREEN = "#159A72"
BLUE = "#2775B6"
ORANGE = "#E47700"
RED = "#D84A3F"
INK = "#171717"
ARROW = "#59636E"


def box(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    face: str,
    edge: str,
    fontsize: float = 9.3,
) -> None:
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.022",
            transform=axis.transAxes,
            facecolor=face,
            edgecolor=edge,
            linewidth=2.2,
        )
    )
    axis.text(
        x + width / 2,
        y + height / 2,
        text,
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight="bold",
        color=INK,
        linespacing=1.08,
    )


def arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    text: str | None = None,
    text_xy: tuple[float, float] | None = None,
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            transform=axis.transAxes,
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2.2,
            color=ARROW,
            connectionstyle="arc3,rad=0",
        )
    )
    if text and text_xy:
        axis.text(
            text_xy[0],
            text_xy[1],
            text,
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=9.3,
            weight="bold",
            color=BLUE,
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": "white",
                "edgecolor": BLUE,
                "linewidth": 1.4,
            },
        )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.4, 6.4), facecolor="white")
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    axis.set_axis_off()
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)

    axis.text(0.015, 0.985, "A", ha="left", va="top", fontsize=23, weight="bold")
    axis.text(
        0.095,
        0.965,
        "HNSCC CD8 NICHE–STATE DISCOVERY",
        ha="left",
        va="top",
        fontsize=14.5,
        weight="bold",
        color=INK,
    )

    # Read from top-left to top-right, then snake down and back to the left.
    box(axis, 0.020, 0.705, 0.440, 0.155, "CODEX", "#EAF7F2", GREEN)
    box(
        axis,
        0.540,
        0.690,
        0.440,
        0.185,
        "CAF/MACROPHAGE-RICH CD8 NICHE\nASSOCIATED WITH\nPOORER SURVIVAL",
        "#FCEDEC",
        RED,
        fontsize=9.3,
    )
    box(
        axis,
        0.540,
        0.385,
        0.440,
        0.175,
        "NICHE SCORE\nPROJECTED TO CITE-seq\n107 ADDITIONAL BIOMARKERS",
        "#EAF2FB",
        BLUE,
        fontsize=9.3,
    )
    box(
        axis,
        0.020,
        0.385,
        0.440,
        0.175,
        "HIGH PROJECTED NICHE SCORE\nASSOCIATED WITH\nADDITIONAL CITE-seq BIOMARKERS\nTIGIT/CD39",
        "#FFF3E6",
        ORANGE,
    )
    box(
        axis,
        0.020,
        0.090,
        0.960,
        0.165,
        "RETURNED TIGIT/CD39-HIGH CD8\nPREFERENTIALLY ENRICHED IN ORIGINAL NICHE\nGREATER NICHE EXPOSURE ASSOCIATED WITH\nPOORER OVERALL SURVIVAL",
        "#EDF7F1",
        GREEN,
        fontsize=9.3,
    )

    arrow(axis, (0.468, 0.782), (0.532, 0.782))
    arrow(
        axis,
        (0.763, 0.682),
        (0.763, 0.568),
        text="RAMP TRANSFER\n22 SHARED PROTEINS",
        text_xy=(0.763, 0.625),
    )
    arrow(axis, (0.532, 0.472), (0.468, 0.472))
    arrow(
        axis,
        (0.238, 0.392),
        (0.238, 0.265),
        text="RAMP RETURN",
        text_xy=(0.238, 0.328),
    )

    png = OUT / "panel_A_square_workflow_v12.png"
    pdf = OUT / "panel_A_square_workflow_v12.pdf"
    figure.savefig(png, dpi=320, facecolor="white")
    figure.savefig(pdf, facecolor="white")
    plt.close(figure)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
