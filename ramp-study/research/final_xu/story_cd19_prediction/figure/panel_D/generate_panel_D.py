#!/usr/bin/env python3
"""Generate Panel D: concise workflow for label-free CODEX CD19 prediction."""

from __future__ import annotations

import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
CACHE = HERE / ".cache"
CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


INK = "#20252B"
BLUE = "#3E70B2"
GREEN = "#31916E"
ORANGE = "#DF8424"
MAGENTA = "#C51B5D"


def rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    face: str,
    edge: str,
    title: str,
    body: str,
    *,
    title_color: str = INK,
    title_size: float = 12.2,
    body_size: float = 10.2,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.014,rounding_size=0.024",
        facecolor=face,
        edgecolor=edge,
        linewidth=2.0,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height * 0.68,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=title_color,
    )
    ax.text(
        x + width / 2,
        y + height * 0.31,
        body,
        ha="center",
        va="center",
        fontsize=body_size,
        color=INK,
        linespacing=1.22,
    )


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2.0,
            color=INK,
            shrinkA=4,
            shrinkB=4,
        )
    )


def main() -> None:
    fig, ax = plt.subplots(figsize=(9.4, 3.65), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.018,
        0.965,
        "D.  Cross-platform masked-marker prediction",
        ha="left",
        va="top",
        fontsize=15.5,
        fontweight="bold",
        color="#111111",
    )

    rounded_box(
        ax,
        0.025,
        0.55,
        0.255,
        0.255,
        "#EEF4FB",
        BLUE,
        "CODEX",
        "14 observed shared markers\nfailed CD19 withheld",
    )
    rounded_box(
        ax,
        0.025,
        0.205,
        0.255,
        0.255,
        "#F1F8F4",
        GREEN,
        "Reference datasets",
        "same 14 shared markers\nmeasured CD19",
        title_size=11.5,
    )

    rounded_box(
        ax,
        0.395,
        0.315,
        0.275,
        0.365,
        "#FFF7E9",
        ORANGE,
        "Masked-marker model",
        "Learns shared protein-state\nrelationships across platforms",
        title_size=12.3,
        body_size=10.4,
    )
    rounded_box(
        ax,
        0.785,
        0.585,
        0.195,
        0.205,
        "#FCEEF4",
        MAGENTA,
        "Completed CODEX",
        "14 shared markers\n+ predicted CD19",
        title_color=MAGENTA,
        title_size=11.2,
        body_size=9.7,
    )
    rounded_box(
        ax,
        0.785,
        0.205,
        0.195,
        0.225,
        "#F3F0FA",
        "#7355A3",
        "RAMP alignment",
        "Three datasets\nwith completed CD19",
        title_color="#5C4386",
        title_size=11.2,
        body_size=9.2,
    )

    # Keep a visible white gap between every arrow and its source/target box.
    arrow(ax, (0.302, 0.675), (0.373, 0.585))
    arrow(ax, (0.302, 0.333), (0.373, 0.415))
    arrow(ax, (0.692, 0.565), (0.762, 0.675))
    arrow(ax, (0.8825, 0.555), (0.8825, 0.455))

    ax.text(
        0.50,
        0.075,
        "No cell labels, spatial coordinates, or failed CODEX CD19 values are used",
        ha="center",
        va="center",
        fontsize=10.1,
        fontweight="bold",
        color="#3F4348",
    )

    fig.subplots_adjust(left=0.012, right=0.992, top=0.98, bottom=0.035)
    png = HERE / "panel_D_prediction_workflow.png"
    pdf = HERE / "panel_D_prediction_workflow.pdf"
    fig.savefig(png, dpi=320, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    print(f"Saved {png}")
    print(f"Saved {pdf}")


if __name__ == "__main__":
    main()
