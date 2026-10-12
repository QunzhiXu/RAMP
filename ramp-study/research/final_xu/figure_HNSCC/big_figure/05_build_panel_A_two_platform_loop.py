#!/usr/bin/env python3
"""Build a minimal two-platform workflow for HNSCC Panel A."""

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


INK = "#171717"
GRAY = "#59636E"
GREEN = "#159A72"
BLUE = "#2775B6"
ORANGE = "#E47700"
RED = "#D84A3F"


def rounded_box(
    axis: plt.Axes,
    bounds: tuple[float, float, float, float],
    face: str,
    edge: str,
    linewidth: float = 2.2,
    radius: float = 0.022,
) -> None:
    x, y, width, height = bounds
    axis.add_patch(
        FancyBboxPatch(
            (x, y), width, height,
            boxstyle=f"round,pad=0.010,rounding_size={radius}",
            transform=axis.transAxes,
            facecolor=face, edgecolor=edge, linewidth=linewidth,
        )
    )


def label_box(
    axis: plt.Axes,
    bounds: tuple[float, float, float, float],
    text: str,
    face: str,
    edge: str,
    fontsize: float = 9.0,
    color: str = INK,
) -> None:
    rounded_box(axis, bounds, face, edge, linewidth=1.8, radius=0.016)
    x, y, width, height = bounds
    axis.text(
        x + width / 2, y + height / 2, text,
        transform=axis.transAxes, ha="center", va="center",
        fontsize=fontsize, weight="bold", color=color, linespacing=1.05,
    )


def arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = GRAY,
    linewidth: float = 2.2,
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start, end, transform=axis.transAxes,
            arrowstyle="-|>", mutation_scale=18,
            linewidth=linewidth, color=color,
            connectionstyle="arc3,rad=0",
        )
    )


def arrow_label(
    axis: plt.Axes,
    x: float,
    y: float,
    text: str,
    color: str = BLUE,
) -> None:
    axis.text(
        x, y, text, transform=axis.transAxes,
        ha="center", va="center", fontsize=7.8,
        weight="bold", color=color, linespacing=1.02,
        bbox={
            "boxstyle": "round,pad=0.24",
            "facecolor": "white", "edgecolor": color,
            "linewidth": 1.3,
        },
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 6.5), facecolor="white")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.018, 0.985, "A", transform=ax.transAxes,
            ha="left", va="top", fontsize=23, weight="bold", color=INK)
    ax.text(
        0.095, 0.965, "SURVIVAL-FIRST NICHE–STATE TRANSFER",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=13.0, weight="bold", color=INK,
    )

    codex = (0.045, 0.115, 0.390, 0.745)
    cite = (0.565, 0.115, 0.390, 0.745)
    rounded_box(ax, codex, "#F3FAF7", GREEN, linewidth=2.7, radius=0.027)
    rounded_box(ax, cite, "#F3F7FC", BLUE, linewidth=2.7, radius=0.027)

    # Platform headers deliberately dominate the panel.
    rounded_box(ax, (0.072, 0.765, 0.336, 0.068), "#DFF3EA", GREEN, linewidth=0, radius=0.015)
    rounded_box(ax, (0.592, 0.765, 0.336, 0.068), "#E1ECF8", BLUE, linewidth=0, radius=0.015)
    ax.text(0.240, 0.799, "CODEX", transform=ax.transAxes,
            ha="center", va="center", fontsize=14.5, weight="bold", color=GREEN)
    ax.text(0.760, 0.799, "CITE-seq", transform=ax.transAxes,
            ha="center", va="center", fontsize=14.5, weight="bold", color=BLUE)

    # CODEX: survival analysis motivates the spatial niche, then receives the
    # molecular state back for spatial and clinical validation.
    label_box(
        ax, (0.090, 0.625, 0.300, 0.090),
        "SURVIVAL\nANALYSIS", "#FCEDEC", RED, fontsize=9.7, color=RED,
    )
    arrow(ax, (0.240, 0.618), (0.240, 0.555))
    label_box(
        ax, (0.075, 0.425, 0.330, 0.125),
        "CAF/MACROPHAGE-RICH\nCD8 NICHE", "#E8F6F0", GREEN, fontsize=9.2,
    )
    label_box(
        ax, (0.075, 0.185, 0.330, 0.145),
        "VALIDATION\nNICHE ENRICHMENT\n+  POORER SURVIVAL",
        "#FCEDEC", RED, fontsize=8.8,
    )

    # CITE-seq: the transferred niche score exposes two biomarkers unavailable
    # to the spatial panel.
    label_box(
        ax, (0.595, 0.480, 0.330, 0.125),
        "PROJECTED\nNICHE SCORE", "#EAF2FB", BLUE, fontsize=9.7,
    )
    arrow(ax, (0.760, 0.470), (0.760, 0.395))
    ax.text(0.760, 0.432, "IDENTIFY", transform=ax.transAxes,
            ha="center", va="center", fontsize=7.8, weight="bold", color=GRAY)
    label_box(
        ax, (0.595, 0.235, 0.330, 0.150),
        "2 ADDITIONAL BIOMARKERS\nTIGIT  ·  CD39",
        "#FFF3E6", ORANGE, fontsize=9.3, color=ORANGE,
    )

    # Cross-platform loop.  The arrows align with the internal state blocks so
    # the reading order is CODEX top-to-middle, right, down, and back left.
    arrow(ax, (0.412, 0.490), (0.588, 0.540), color=BLUE, linewidth=2.5)
    arrow_label(ax, 0.500, 0.574, "RAMP TRANSFER")

    arrow(ax, (0.588, 0.310), (0.412, 0.260), color=BLUE, linewidth=2.5)
    arrow_label(ax, 0.500, 0.207, "RAMP RETURN")

    # A subtle loop cue keeps the result tied to the survival-first starting
    # point without adding another data panel or causal wording.
    ax.text(
        0.500, 0.070,
        "SPATIAL PHENOTYPE  →  MOLECULAR STATE  →  SPATIAL + CLINICAL VALIDATION",
        transform=ax.transAxes, ha="center", va="center",
        fontsize=7.7, weight="bold", color=GRAY,
    )

    png = OUT / "panel_A_two_platform_loop_v17.png"
    pdf = OUT / "panel_A_two_platform_loop_v17.pdf"
    fig.savefig(png, dpi=320, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
