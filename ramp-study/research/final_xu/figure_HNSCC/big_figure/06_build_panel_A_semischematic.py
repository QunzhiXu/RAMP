#!/usr/bin/env python3
"""Build a semi-schematic two-platform workflow for HNSCC Panel A."""

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


INK = "#171717"
GRAY = "#59636E"
LIGHT_GRAY = "#E9EEF2"
GREEN = "#159A72"
BLUE = "#2775B6"
ORANGE = "#E47700"
RED = "#D84A3F"


def rounded_box(
    ax: plt.Axes,
    bounds: tuple[float, float, float, float],
    face: str,
    edge: str,
    linewidth: float = 2.4,
    radius: float = 0.025,
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
    color: str = GRAY,
    linewidth: float = 2.0,
) -> None:
    ax.add_patch(FancyArrowPatch(
        start, end, transform=ax.transAxes,
        arrowstyle="-|>", mutation_scale=17,
        linewidth=linewidth, color=color,
        connectionstyle="arc3,rad=0",
    ))


def km_icon(ax: plt.Axes, x: float, y: float, width: float, height: float) -> None:
    ax.plot([x, x, x + width], [y + height, y, y], transform=ax.transAxes,
            color="#7A848E", linewidth=1.2, solid_capstyle="round")
    x_blue = np.array([0.00, 0.16, 0.16, 0.43, 0.43, 0.72, 0.72, 1.00])
    y_blue = np.array([0.96, 0.96, 0.84, 0.84, 0.72, 0.72, 0.54, 0.54])
    x_red = np.array([0.00, 0.10, 0.10, 0.25, 0.25, 0.48, 0.48, 0.77, 0.77, 1.00])
    y_red = np.array([0.94, 0.94, 0.72, 0.72, 0.50, 0.50, 0.36, 0.36, 0.23, 0.23])
    ax.plot(x + width * x_blue, y + height * y_blue, transform=ax.transAxes,
            color="#3678B8", linewidth=2.3, drawstyle="steps-post")
    ax.plot(x + width * x_red, y + height * y_red, transform=ax.transAxes,
            color=RED, linewidth=2.3, drawstyle="steps-post")


def niche_icon(
    ax: plt.Axes,
    cx: float,
    cy: float,
    radius: float,
    returned: bool = False,
) -> None:
    ax.add_patch(Circle((cx, cy), radius, transform=ax.transAxes,
                        facecolor="#F3F5F7", edgecolor="#B7C0C8", linewidth=1.3))
    positions = [
        (-0.66, 0.22), (-0.50, -0.45), (-0.34, 0.60), (-0.22, 0.08),
        (-0.04, -0.60), (0.08, 0.55), (0.22, -0.05), (0.38, 0.43),
        (0.52, -0.48), (0.66, 0.10), (-0.63, -0.15), (0.02, 0.02),
        (0.36, -0.18), (-0.28, -0.20), (0.62, 0.53),
    ]
    palette = ["#A5B1BB", "#8FC9A7", "#C59AC4", "#9ABCD5"]
    for idx, (dx, dy) in enumerate(positions):
        color = palette[(idx * 3 + 1) % len(palette)]
        size = radius * (0.13 if idx % 3 else 0.15)
        ax.add_patch(Circle((cx + dx * radius, cy + dy * radius), size,
                            transform=ax.transAxes, facecolor=color,
                            edgecolor="white", linewidth=0.7))
    ax.add_patch(Ellipse((cx + 0.10 * radius, cy), 1.12 * radius, 0.92 * radius,
                         transform=ax.transAxes, fill=False, edgecolor="#7A5C8C",
                         linewidth=1.6, linestyle=(0, (3, 2))))
    if returned:
        for dx, dy in ((-0.15, 0.13), (0.10, -0.07), (0.32, 0.20), (0.23, -0.28)):
            ax.add_patch(Circle((cx + dx * radius, cy + dy * radius), radius * 0.13,
                                transform=ax.transAxes, facecolor="#F4B942",
                                edgecolor="#6F4B00", linewidth=0.8))
    else:
        ax.add_patch(Circle((cx + 0.04 * radius, cy - 0.02 * radius), radius * 0.15,
                            transform=ax.transAxes, facecolor="#32A7D8",
                            edgecolor="#07445E", linewidth=0.8))


def score_cloud(ax: plt.Axes, cx: float, cy: float, width: float, height: float) -> None:
    # Deterministic, deliberately schematic cell cloud rather than a data UMAP.
    rng = np.random.default_rng(17)
    centers = np.array([[-0.32, 0.15], [0.25, 0.24], [0.12, -0.30]])
    points = []
    for mx, my in centers:
        block = rng.normal((mx, my), (0.13, 0.10), size=(30, 2))
        points.append(block)
    points = np.vstack(points)
    px = cx + width * points[:, 0]
    py = cy + height * points[:, 1]
    score = (points[:, 0] - points[:, 0].min()) / np.ptp(points[:, 0])
    ax.scatter(px, py, c=score, cmap="Blues", vmin=0, vmax=1,
               s=20, edgecolors="white", linewidths=0.35,
               transform=ax.transAxes, zorder=5)
    # A small gradient key makes the intended meaning explicit without values.
    steps = 18
    for idx in range(steps):
        ax.add_patch(Rectangle(
            (cx - 0.30 * width + idx * 0.60 * width / steps, cy - 0.53 * height),
            0.60 * width / steps, 0.045 * height,
            transform=ax.transAxes,
            facecolor=plt.cm.Blues(0.20 + 0.72 * idx / (steps - 1)),
            edgecolor="none",
        ))


def protein_icon(ax: plt.Axes, x: float, y: float, width: float, height: float) -> None:
    rows = 7
    row_h = height / rows
    for row in range(rows):
        color = "#CFDBE5"
        if row in (2, 5):
            color = ORANGE
        ax.add_patch(FancyBboxPatch(
            (x, y + (rows - 1 - row) * row_h), width, row_h * 0.60,
            boxstyle="round,pad=0.001,rounding_size=0.005",
            transform=ax.transAxes, facecolor=color, edgecolor="none",
        ))
    ax.text(x + width + 0.012, y + (rows - 1 - 2 + 0.30) * row_h, "TIGIT",
            transform=ax.transAxes, ha="left", va="center",
            fontsize=7.8, weight="bold", color=ORANGE)
    ax.text(x + width + 0.012, y + (rows - 1 - 5 + 0.30) * row_h, "CD39",
            transform=ax.transAxes, ha="left", va="center",
            fontsize=7.8, weight="bold", color=ORANGE)


def projected_matrix_icon(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
) -> None:
    """Draw a schematic cell-by-protein matrix with a projected score row."""
    columns = 12
    protein_rows = 8
    gap = width * 0.010
    cell_width = (width - gap * (columns - 1)) / columns

    # The projected niche score is an aligned annotation across the same cell
    # columns as the protein matrix below it.
    score_y = y + height * 0.82
    score_h = height * 0.075
    for col in range(columns):
        value = col / (columns - 1)
        ax.add_patch(Rectangle(
            (x + col * (cell_width + gap), score_y), cell_width, score_h,
            transform=ax.transAxes, facecolor=plt.cm.Blues(0.18 + 0.78 * value),
            edgecolor="white", linewidth=0.35,
        ))

    matrix_y = y
    matrix_h = height * 0.68
    row_gap = height * 0.012
    row_h = (matrix_h - row_gap * (protein_rows - 1)) / protein_rows
    cool = ["#EEF2F5", "#DCE7EF", "#C7D9E5", "#AFC9DB"]
    for row in range(protein_rows):
        for col in range(columns):
            value = col / (columns - 1)
            if row in (2, 5):
                # Two rows become stronger across the projected-score axis.
                color = plt.cm.Oranges(0.16 + 0.72 * value)
            else:
                color = cool[(row * 3 + col * 5 + col // 2) % len(cool)]
            ax.add_patch(Rectangle(
                (x + col * (cell_width + gap),
                 matrix_y + (protein_rows - 1 - row) * (row_h + row_gap)),
                cell_width, row_h,
                transform=ax.transAxes, facecolor=color,
                edgecolor="white", linewidth=0.30,
            ))

    for row, label in ((2, "TIGIT"), (5, "CD39")):
        label_y = matrix_y + (protein_rows - 1 - row) * (row_h + row_gap) + row_h / 2
        ax.text(x + width + 0.012, label_y, label,
                transform=ax.transAxes, ha="left", va="center",
                fontsize=7.8, weight="bold", color=ORANGE)


def ramp_label(ax: plt.Axes, x: float, y: float, text: str) -> None:
    ax.text(x, y, text, transform=ax.transAxes,
            ha="center", va="center", fontsize=7.8, weight="bold", color=BLUE,
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white",
                  "edgecolor": BLUE, "linewidth": 1.3})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 6.5), facecolor="white")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.018, 0.985, "A", transform=ax.transAxes,
            ha="left", va="top", fontsize=23, weight="bold", color=INK)
    ax.text(0.095, 0.965, "SURVIVAL-FIRST NICHE–STATE TRANSFER",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=13.0, weight="bold", color=INK)

    codex = (0.045, 0.110, 0.390, 0.750)
    cite = (0.565, 0.110, 0.390, 0.750)
    rounded_box(ax, codex, "#F5FAF8", GREEN, linewidth=2.7)
    rounded_box(ax, cite, "#F5F8FC", BLUE, linewidth=2.7)
    rounded_box(ax, (0.072, 0.765, 0.336, 0.068), "#DFF3EA", GREEN, linewidth=0, radius=0.015)
    rounded_box(ax, (0.592, 0.765, 0.336, 0.068), "#E1ECF8", BLUE, linewidth=0, radius=0.015)
    ax.text(0.240, 0.799, "CODEX", transform=ax.transAxes,
            ha="center", va="center", fontsize=14.5, weight="bold", color=GREEN)
    ax.text(0.760, 0.799, "CITE-seq", transform=ax.transAxes,
            ha="center", va="center", fontsize=14.5, weight="bold", color=BLUE)

    # CODEX discovery: survival first, then its spatial niche.
    km_icon(ax, 0.095, 0.620, 0.115, 0.095)
    arrow(ax, (0.225, 0.665), (0.284, 0.665), color=RED, linewidth=1.8)
    niche_icon(ax, 0.337, 0.665, 0.070, returned=False)
    ax.text(0.240, 0.570, "SURVIVAL  →  NICHE", transform=ax.transAxes,
            ha="center", va="center", fontsize=8.7, weight="bold", color=INK)

    # CITE-seq characterization: the projected score is aligned with cell
    # columns in a protein matrix, exposing two additional biomarker rows.
    projected_matrix_icon(ax, 0.625, 0.285, 0.245, 0.390)
    ax.text(0.747, 0.705, "PROJECTED NICHE SCORE", transform=ax.transAxes,
            ha="center", va="center", fontsize=8.5, weight="bold", color=INK)
    ax.text(0.747, 0.245, "107 ADDITIONAL PROTEINS", transform=ax.transAxes,
            ha="center", va="center", fontsize=8.0, weight="bold", color=GRAY)
    ax.text(0.747, 0.205, "2 ADDITIONAL BIOMARKERS", transform=ax.transAxes,
            ha="center", va="center", fontsize=8.3, weight="bold", color=INK)
    ax.text(0.747, 0.168, "TIGIT  ·  CD39", transform=ax.transAxes,
            ha="center", va="center", fontsize=9.7, weight="bold", color=ORANGE)

    # CODEX return: the icon combines niche localization and survival, but uses
    # no patient, coordinate, or quantitative result from the real analysis.
    niche_icon(ax, 0.175, 0.300, 0.083, returned=True)
    ax.text(0.272, 0.302, "+", transform=ax.transAxes,
            ha="center", va="center", fontsize=16, weight="bold", color=GRAY)
    km_icon(ax, 0.302, 0.255, 0.095, 0.085)
    ax.text(0.240, 0.178, "NICHE + SURVIVAL VALIDATION", transform=ax.transAxes,
            ha="center", va="center", fontsize=8.3, weight="bold", color=INK)

    # Cross-platform loop.
    arrow(ax, (0.405, 0.625), (0.595, 0.625), color=BLUE, linewidth=2.5)
    ramp_label(ax, 0.500, 0.665, "RAMP TRANSFER")
    arrow(ax, (0.595, 0.285), (0.405, 0.285), color=BLUE, linewidth=2.5)
    ramp_label(ax, 0.500, 0.245, "RAMP RETURN")

    ax.text(0.500, 0.060,
            "SPATIAL PHENOTYPE  →  MOLECULAR STATE  →  SPATIAL + CLINICAL VALIDATION",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=7.7, weight="bold", color=GRAY)

    png = OUT / "panel_A_semischematic_v19.png"
    pdf = OUT / "panel_A_semischematic_v19.pdf"
    fig.savefig(png, dpi=320, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
