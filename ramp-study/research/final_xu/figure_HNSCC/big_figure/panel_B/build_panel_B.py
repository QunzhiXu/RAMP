#!/usr/bin/env python3
"""Build HNSCC Panel B: abstract niche definition plus observed survival.

The Kaplan--Meier coordinates and numbers at risk are read directly from the
frozen outputs in ``figure_HNSCC/survival_motivation/output``.  The left-hand
spatial drawing is deliberately schematic so that later panels can reserve
real CODEX imagery for the returned-state validation.
"""

from __future__ import annotations

import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
HNSCC_ROOT = HERE.parent.parent
SURVIVAL = HNSCC_ROOT / "survival_motivation" / "output"
CACHE = HERE / ".cache"
for child in ("matplotlib", "xdg"):
    (CACHE / child).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE / "xdg"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


INK = "#182027"
MUTED = "#68737D"
LINE = "#CBD4DB"
BLUE = "#2B6CB0"
RED = "#C9433C"
CYAN = "#3BA9D4"
CYAN_EDGE = "#174E66"
GREEN = "#159A72"
PINK = "#D66AA5"
OTHER = "#B8C2CA"
PALE = "#F7F9FA"
PALE_GREEN = "#F1F8F5"
FIGURE_WIDTH = 10.8
FIGURE_HEIGHT = 4.6
AXES_ASPECT_CORRECTION = FIGURE_WIDTH / FIGURE_HEIGHT


def rounded_box(
    ax: plt.Axes,
    bounds: tuple[float, float, float, float],
    face: str,
    edge: str,
    linewidth: float = 1.2,
    radius: float = 0.018,
    zorder: float = 1,
) -> None:
    x, y, width, height = bounds
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle=f"round,pad=0.008,rounding_size={radius}",
            transform=ax.transAxes,
            facecolor=face,
            edgecolor=edge,
            linewidth=linewidth,
            zorder=zorder,
        )
    )


def round_cell(
    ax: plt.Axes,
    center: tuple[float, float],
    radius: float,
    face: str,
    edge: str = "white",
    linewidth: float = 0.7,
    alpha: float = 1.0,
    zorder: float = 4,
) -> None:
    """Draw a physically circular cell on a non-square full-figure axis."""
    ax.add_patch(
        Ellipse(
            center,
            width=2.0 * radius,
            height=2.0 * radius * AXES_ASPECT_CORRECTION,
            transform=ax.transAxes,
            facecolor=face,
            edgecolor=edge,
            linewidth=linewidth,
            alpha=alpha,
            zorder=zorder,
        )
    )


def draw_niche_schematic(ax: plt.Axes) -> None:
    """Draw a focal PT-CD8 cell and exactly 15 non-self nearest neighbors."""
    rounded_box(ax, (0.035, 0.035, 0.345, 0.815), "white", LINE, 1.35, 0.020)
    ax.text(
        0.2075,
        0.805,
        "SPATIAL NICHE SCORE",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=12.2,
        weight="bold",
        color=INK,
    )

    # Abstract primary-tumor ROI.  The dashed oval denotes a nearest-neighbor
    # set, not a fixed micrometre radius.
    rounded_box(ax, (0.072, 0.405, 0.271, 0.355), PALE, "#D6DEE4", 1.0, 0.016)
    cx, cy = 0.2075, 0.592
    ax.add_patch(
        Ellipse(
            (cx, cy),
            0.212,
            0.300,
            transform=ax.transAxes,
            facecolor=PALE_GREEN,
            edgecolor=GREEN,
            linewidth=1.55,
            linestyle=(0, (3.0, 2.0)),
            zorder=2,
        )
    )

    # Fifteen nearest non-self neighbors: 4 CAF, 3 macrophages, 8 other cells.
    neighbor_points = [
        (-0.070, 0.058, GREEN),
        (-0.032, 0.088, OTHER),
        (0.014, 0.092, PINK),
        (0.058, 0.065, OTHER),
        (0.078, 0.019, GREEN),
        (0.072, -0.039, OTHER),
        (0.041, -0.081, PINK),
        (-0.004, -0.094, OTHER),
        (-0.050, -0.077, GREEN),
        (-0.078, -0.032, OTHER),
        (-0.088, 0.014, PINK),
        (-0.043, 0.025, OTHER),
        (0.002, 0.048, GREEN),
        (0.045, 0.024, OTHER),
        (0.026, -0.035, OTHER),
    ]
    for dx, dy, color in neighbor_points:
        round_cell(
            ax, (cx + dx, cy + dy), 0.0082, color,
            edge="white", linewidth=0.75, zorder=4,
        )

    # A few unselected cells outside the nearest-neighbor set establish the ROI
    # background without cluttering the icon.
    outside = [
        (0.087, 0.689, OTHER),
        (0.107, 0.490, GREEN),
        (0.319, 0.683, PINK),
        (0.322, 0.500, OTHER),
        (0.089, 0.612, OTHER),
        (0.305, 0.598, GREEN),
    ]
    for x, y, color in outside:
        round_cell(
            ax, (x, y), 0.0068, color,
            edge="white", linewidth=0.6, alpha=0.90, zorder=3,
        )

    round_cell(
        ax, (cx, cy), 0.0125, CYAN,
        edge=CYAN_EDGE, linewidth=1.25, zorder=6,
    )
    # Compact color key.
    key_y = 0.373
    key = [
        (0.087, CYAN, CYAN_EDGE, "CD8 T CELL"),
        (0.165, GREEN, "white", "CAF"),
        (0.225, PINK, "white", "MACROPHAGE"),
        (0.322, OTHER, "white", "OTHER"),
    ]
    for x, face, edge, label in key:
        round_cell(
            ax, (x, key_y), 0.0053, face,
            edge=edge, linewidth=0.7, zorder=5,
        )
        ax.text(
            x + 0.010,
            key_y,
            label,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=7.2,
            weight="bold",
            color=INK,
        )

    ax.text(
        0.2075,
        0.316,
        "LOCAL CAF/MACROPHAGE ENRICHMENT",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9.3,
        weight="bold",
        color=INK,
    )
    ax.add_patch(
        FancyArrowPatch(
            (0.2075, 0.290),
            (0.2075, 0.175),
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.7,
            color=GREEN,
        )
    )
    ax.text(
        0.2075,
        0.140,
        "MEAN ACROSS CD8 T CELLS",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9.3,
        weight="bold",
        color=INK,
    )
    rounded_box(ax, (0.093, 0.050, 0.229, 0.055), "#E3F4ED", GREEN, 1.0, 0.012)
    ax.text(
        0.2075,
        0.0775,
        "PATIENT NICHE SCORE",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10.1,
        weight="bold",
        color=GREEN,
    )


def draw_survival(fig: plt.Figure, host: plt.Axes) -> None:
    km_path = SURVIVAL / "median_split_km_curve.csv"
    risk_path = SURVIVAL / "median_split_number_at_risk.csv"
    if not km_path.exists() or not risk_path.exists():
        raise FileNotFoundError("Run survival_motivation before building Panel B")
    km = pd.read_csv(km_path)
    risk = pd.read_csv(risk_path)

    host.text(
        0.705,
        0.825,
        "HIGHER NICHE SCORE, POORER SURVIVAL",
        transform=host.transAxes,
        ha="center",
        va="center",
        fontsize=12.2,
        weight="bold",
        color=INK,
    )
    host.text(
        0.705,
        0.770,
        "Continuous Cox: HR = 1.34 per SD  (95% CI, 1.03–1.74),  P = 0.029",
        transform=host.transAxes,
        ha="center",
        va="center",
        fontsize=9.0,
        weight="bold",
        color=INK,
    )

    ax = fig.add_axes([0.460, 0.265, 0.490, 0.420])
    colors = {"Lower": BLUE, "Higher": RED}
    labels = {
        "Lower": "Lower niche score (n=37)",
        "Higher": "Higher niche score (n=37)",
    }
    for group in ("Lower", "Higher"):
        data = km.loc[km["group"].eq(group)].sort_values("time")
        ax.step(
            data["time"],
            data["survival"],
            where="post",
            color=colors[group],
            linewidth=2.5,
            label=labels[group],
        )

    ax.set_xlim(0, 120)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([0, 24, 48, 72, 96, 120])
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("Overall survival (months)", fontsize=10.0, color=INK, labelpad=5)
    ax.set_ylabel("Overall survival probability", fontsize=10.0, color=INK, labelpad=5)
    ax.tick_params(axis="both", labelsize=8.6, colors=INK, length=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.grid(False)

    handles = [
        Line2D([0], [0], color=BLUE, linewidth=2.5, label=labels["Lower"]),
        Line2D([0], [0], color=RED, linewidth=2.5, label=labels["Higher"]),
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.015, 0.035),
        frameon=False,
        fontsize=8.5,
        handlelength=2.2,
    )
    # Compact number-at-risk table aligned to the x-axis.
    host.text(
        0.415,
        0.145,
        "NUMBER\nAT RISK",
        transform=host.transAxes,
        ha="center",
        va="center",
        fontsize=7.6,
        weight="bold",
        color=INK,
        linespacing=0.92,
    )
    for group, y in (("Lower", 0.105), ("Higher", 0.060)):
        host.text(
            0.435,
            y,
            group,
            transform=host.transAxes,
            ha="right",
            va="center",
            fontsize=8.1,
            weight="bold",
            color=colors[group],
        )
        values = risk.loc[risk["group"].eq(group)].set_index("time")["n_risk"]
        for x_axis, month in zip(np.linspace(0.460, 0.950, 6), [0, 24, 48, 72, 96, 120]):
            host.text(
                x_axis,
                y,
                str(int(values.loc[month])),
                transform=host.transAxes,
                ha="center",
                va="center",
                fontsize=8.0,
                weight="bold",
                color=colors[group],
            )


def main() -> None:
    fig, host = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), facecolor="white")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    host.set_axis_off()
    host.set_xlim(0, 1)
    host.set_ylim(0, 1)

    host.text(0.018, 0.970, "B", transform=host.transAxes,
              ha="left", va="top", fontsize=24, weight="bold", color=INK)
    host.text(
        0.072,
        0.943,
        "CAF/MACROPHAGE-RICH CD8 NICHES ASSOCIATE WITH POORER SURVIVAL",
        transform=host.transAxes,
        ha="left",
        va="top",
        fontsize=14.3,
        weight="bold",
        color=INK,
    )

    draw_niche_schematic(host)
    draw_survival(fig, host)

    png = HERE / "panel_B.png"
    pdf = HERE / "panel_B.pdf"
    fig.savefig(png, dpi=320, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
