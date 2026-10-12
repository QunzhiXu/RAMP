#!/usr/bin/env python3
"""Build a polished semi-schematic HNSCC workflow with no real data panels."""

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
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
from PIL import Image


INK = "#182027"
MUTED = "#65717C"
LINE = "#C9D2D9"
GREEN = "#159A72"
GREEN_BG = "#F4FAF7"
GREEN_PALE = "#DFF3EA"
BLUE = "#2775B6"
BLUE_BG = "#F4F8FC"
BLUE_PALE = "#E1ECF8"
ORANGE = "#E47700"
ORANGE_PALE = "#FFF1DF"
RED = "#D84A3F"

LOGO_SOURCE = (
    HERE.parent.parent
    / "figure/introduction_concept/final_publication/RAMP_final_publication_6000x3000.png"
)


def rounded_box(
    ax: plt.Axes,
    bounds: tuple[float, float, float, float],
    face: str,
    edge: str,
    linewidth: float = 1.5,
    radius: float = 0.020,
) -> None:
    x, y, width, height = bounds
    ax.add_patch(FancyBboxPatch(
        (x, y), width, height,
        boxstyle=f"round,pad=0.010,rounding_size={radius}",
        transform=ax.transAxes, facecolor=face, edgecolor=edge,
        linewidth=linewidth,
    ))


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = BLUE,
) -> None:
    ax.add_patch(FancyArrowPatch(
        start, end, transform=ax.transAxes,
        arrowstyle="-|>", mutation_scale=18,
        linewidth=2.25, color=color,
        connectionstyle="arc3,rad=0",
    ))


def tissue_niche(
    ax: plt.Axes,
    bounds: tuple[float, float, float, float],
    returned: bool,
) -> None:
    """Draw one consistent abstract tissue tile for discovery and return."""
    x, y, width, height = bounds
    rounded_box(ax, bounds, "#F7F9FA", "#D6DEE4", linewidth=1.0, radius=0.014)
    niche_center = (x + 0.57 * width, y + 0.49 * height)
    ax.add_patch(Ellipse(
        niche_center, width * 0.42, height * 0.67,
        transform=ax.transAxes, facecolor="#EAF6F1", edgecolor="none",
        linewidth=0, alpha=0.72,
    ))

    # A reproducible multicolor point field evokes multiplexed spatial cells
    # without copying any patient image or encoding a real cell identity.
    rng = np.random.default_rng(20260903)
    positions = rng.uniform((0.045, 0.10), (0.955, 0.90), size=(76, 2))
    palette = [
        "#7BB7D8", "#83C69B", "#D19AC6", "#E9B86B",
        "#9DAAB5", "#91A7D0", "#D98C7B", "#72B8AF",
    ]
    for idx, (px, py) in enumerate(positions):
        radius = height * float(rng.uniform(0.021, 0.038))
        ax.add_patch(Circle(
            (x + px * width, y + py * height), radius,
            transform=ax.transAxes,
            facecolor=palette[idx % len(palette)],
            edgecolor="white", linewidth=0.42, alpha=0.96,
        ))

    ax.add_patch(Ellipse(
        niche_center, width * 0.42, height * 0.67,
        transform=ax.transAxes, facecolor="none", edgecolor=GREEN,
        linewidth=1.55, linestyle=(0, (3, 2)),
    ))

    if returned:
        for px, py in (
            (0.45, 0.55), (0.51, 0.35), (0.56, 0.61),
            (0.62, 0.43), (0.67, 0.58), (0.70, 0.31),
        ):
            ax.add_patch(Circle(
                (x + px * width, y + py * height), height * 0.048,
                transform=ax.transAxes, facecolor="#F4B942",
                edgecolor="#714B00", linewidth=0.68,
            ))
    else:
        for px, py in (
            (0.44, 0.54), (0.50, 0.35), (0.55, 0.62),
            (0.61, 0.43), (0.66, 0.58), (0.70, 0.32),
        ):
            ax.add_patch(Circle(
                (x + px * width, y + py * height), height * 0.048,
                transform=ax.transAxes, facecolor="#3BA9D4",
                edgecolor="#164A61", linewidth=0.68,
            ))


def matrix_icon(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
) -> None:
    columns = 12
    rows = 8
    col_gap = width * 0.011
    row_gap = height * 0.018
    cell_w = (width - col_gap * (columns - 1)) / columns

    # Projected score uses the same cell columns as the protein matrix.
    score_y = y + height * 0.83
    score_h = height * 0.075
    for col in range(columns):
        value = col / (columns - 1)
        ax.add_patch(Rectangle(
            (x + col * (cell_w + col_gap), score_y), cell_w, score_h,
            transform=ax.transAxes,
            facecolor=plt.cm.Blues(0.16 + 0.80 * value),
            edgecolor="white", linewidth=0.35,
        ))

    matrix_height = height * 0.66
    cell_h = (matrix_height - row_gap * (rows - 1)) / rows
    cool = ["#EDF2F5", "#DDE8EF", "#CBDCE7", "#B9D0DE"]
    for row in range(rows):
        for col in range(columns):
            value = col / (columns - 1)
            if row in (2, 5):
                color = plt.cm.Oranges(0.15 + 0.76 * value)
            else:
                color = cool[(row * 3 + col * 5 + col // 2) % len(cool)]
            ax.add_patch(Rectangle(
                (x + col * (cell_w + col_gap),
                 y + (rows - 1 - row) * (cell_h + row_gap)),
                cell_w, cell_h, transform=ax.transAxes,
                facecolor=color, edgecolor="white", linewidth=0.30,
            ))

    for row, label in ((2, "TIGIT"), (5, "CD39")):
        label_y = y + (rows - 1 - row) * (cell_h + row_gap) + cell_h / 2
        ax.text(x + width + 0.012, label_y, label,
                transform=ax.transAxes, ha="left", va="center",
                fontsize=7.7, weight="bold", color=ORANGE)


def ramp_logo(
    fig: plt.Figure,
    host_axis: plt.Axes,
    bounds: tuple[float, float, float, float],
    left_border: float,
    right_border: float,
) -> None:
    """Embed the logo and extend both pairs of gantry rails to the frames."""
    if not LOGO_SOURCE.exists():
        raise FileNotFoundError(f"Missing publication logo source: {LOGO_SOURCE}")
    with Image.open(LOGO_SOURCE) as source:
        # Green RAMP sign plus yellow RANKS PRESERVED plate.  Bounds are in the
        # original 6000 x 3000 publication artwork, not the temporary screenshot.
        crop = source.crop((2420, 380, 3930, 1190)).convert("RGB")

    # In this fixed crop, the two side rails are centered at approximately
    # 28.6% and 45.0% of image height measured from the top.  Extend those
    # exact levels to the two platform borders before placing the logo on top.
    x, y, width, height = bounds
    logo_left = x
    logo_right = x + width
    logo_top = y + height
    pipe_y = [logo_top - height * fraction for fraction in (0.286, 0.450)]
    for level in pipe_y:
        for x0, x1 in ((left_border, logo_left), (logo_right, right_border)):
            host_axis.plot(
                [x0, x1], [level, level], transform=host_axis.transAxes,
                color="#263138", linewidth=2.2, solid_capstyle="butt",
                zorder=18,
            )
            host_axis.plot(
                [x0, x1], [level, level], transform=host_axis.transAxes,
                color="#778087", linewidth=0.65, solid_capstyle="butt",
                zorder=19,
            )
    logo_axis = fig.add_axes(bounds, zorder=20)
    logo_axis.imshow(crop)
    logo_axis.set_axis_off()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 6.5), facecolor="white")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.018, 0.985, "A", transform=ax.transAxes,
            ha="left", va="top", fontsize=23, weight="bold", color=INK)
    ax.text(0.095, 0.965, "SURVIVAL-ASSOCIATED NICHE TRANSFER",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=13.0, weight="bold", color=INK)

    # Leave a wider central gutter so the RAMP logo reads as an independent
    # bridge and never touches either platform border.
    codex = (0.045, 0.125, 0.375, 0.730)
    cite = (0.580, 0.125, 0.375, 0.730)
    codex_center = codex[0] + codex[2] / 2
    cite_center = cite[0] + cite[2] / 2
    rounded_box(ax, codex, GREEN_BG, GREEN, linewidth=2.6, radius=0.026)
    rounded_box(ax, cite, BLUE_BG, BLUE, linewidth=2.6, radius=0.026)

    rounded_box(ax, (codex_center - 0.167, 0.766, 0.334, 0.063), GREEN_PALE, GREEN,
                linewidth=0, radius=0.014)
    rounded_box(ax, (cite_center - 0.167, 0.766, 0.334, 0.063), BLUE_PALE, BLUE,
                linewidth=0, radius=0.014)
    ax.text(codex_center, 0.798, "CODEX", transform=ax.transAxes,
            ha="center", va="center", fontsize=14.0, weight="bold", color=GREEN)
    ax.text(cite_center, 0.798, "CITE-seq", transform=ax.transAxes,
            ha="center", va="center", fontsize=14.0, weight="bold", color=BLUE)

    # CODEX discovery card.
    rounded_box(ax, (codex_center - 0.162, 0.505, 0.324, 0.205), "white", LINE,
                linewidth=1.1, radius=0.014)
    tissue_niche(ax, (codex_center - 0.127, 0.562, 0.254, 0.093), returned=False)
    ax.text(codex_center, 0.688, "SURVIVAL-ASSOCIATED", transform=ax.transAxes,
            ha="center", va="center", fontsize=8.2, weight="bold", color=RED)
    ax.text(codex_center, 0.527, "NICHE", transform=ax.transAxes,
            ha="center", va="center", fontsize=8.2, weight="bold", color=INK)

    # CODEX return-and-validation card reuses the exact visual grammar above.
    rounded_box(ax, (codex_center - 0.162, 0.235, 0.324, 0.205), "white", LINE,
                linewidth=1.1, radius=0.014)
    tissue_niche(ax, (codex_center - 0.127, 0.292, 0.254, 0.093), returned=True)
    ax.text(codex_center, 0.418, "RETURNED STATE", transform=ax.transAxes,
            ha="center", va="center", fontsize=8.2, weight="bold", color=ORANGE)
    ax.text(codex_center, 0.257, "NICHE + SURVIVAL VALIDATION", transform=ax.transAxes,
            ha="center", va="center", fontsize=8.2, weight="bold", color=INK)

    # Single CITE-seq card: score row, protein matrix, and two highlighted rows.
    rounded_box(ax, (cite_center - 0.163, 0.275, 0.326, 0.390), "white", LINE,
                linewidth=1.1, radius=0.014)
    ax.text(cite_center, 0.628, "PROJECTED NICHE SCORE", transform=ax.transAxes,
            ha="center", va="center", fontsize=8.1, weight="bold", color=INK)
    matrix_icon(ax, 0.6375, 0.365, 0.236, 0.245)
    ax.text(cite_center, 0.335, "107 ADDITIONAL PROTEINS", transform=ax.transAxes,
            ha="center", va="center", fontsize=7.4, weight="bold", color=MUTED)
    ax.text(cite_center, 0.302, "2 ADDITIONAL BIOMARKERS", transform=ax.transAxes,
            ha="center", va="center", fontsize=7.7, weight="bold", color=INK)

    # One central method identity, now using the publication logo rather than
    # a generic text badge.
    ramp_logo(
        fig, ax, (0.447, 0.457, 0.106, 0.067),
        left_border=codex[0] + codex[2],
        right_border=cite[0],
    )

    arrow(ax, (codex_center + 0.162, 0.605), (cite_center - 0.163, 0.605))
    ax.text(0.500, 0.648, "SCORE\nPROJECTION", transform=ax.transAxes,
            ha="center", va="center", fontsize=8.8, weight="bold", color=INK,
            linespacing=0.95,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2})

    arrow(ax, (cite_center - 0.163, 0.335), (codex_center + 0.162, 0.335))
    ax.text(0.500, 0.292, "BIOMARKER\nPROJECTION", transform=ax.transAxes,
            ha="center", va="center", fontsize=8.8, weight="bold", color=INK,
            linespacing=0.95,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2})

    png = OUT / "panel_A_polished_semischematic_v30.png"
    pdf = OUT / "panel_A_polished_semischematic_v30.pdf"
    fig.savefig(png, dpi=320, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
