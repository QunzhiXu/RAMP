#!/usr/bin/env python3
"""Build a four-scene, image-led square workflow for HNSCC Panel A."""

from __future__ import annotations

import importlib.util
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
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd


SOURCE = HERE / "03_build_panel_A_visual_workflow.py"
SPEC = importlib.util.spec_from_file_location("panel_a_visual_v13", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Cannot import helpers from {SOURCE}")
V13 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V13)

# Square display crop; it preserves nearly all cells supporting the formal
# representative field while avoiding letterboxing inside a square workflow.
V13.CROP = (1230, 205, 1695, 670)

INK = V13.INK
MUTED = V13.MUTED
BLUE = V13.BLUE
GREEN = V13.GREEN
ORANGE = V13.ORANGE
RED = V13.RED
CYAN = V13.CYAN
YELLOW = V13.YELLOW
CAF = V13.CAF
MAC = V13.MAC
LOWER = V13.LOWER
HIGHER = V13.HIGHER


def frame(fig: plt.Figure, bounds: tuple[float, float, float, float], color: str) -> None:
    x, y, width, height = bounds
    fig.add_artist(
        FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.006,rounding_size=0.014",
            transform=fig.transFigure, facecolor="white", edgecolor=color,
            linewidth=2.0, zorder=0,
        )
    )


def draw_spatial(
    fig: plt.Figure,
    bounds: tuple[float, float, float, float],
    image: np.ndarray,
    cd8: pd.DataFrame,
    architecture: pd.DataFrame,
    mode: str,
) -> None:
    frame(fig, bounds, GREEN)
    x, y, width, height = bounds
    axis = fig.add_axes((x + 0.008, y + 0.008, width - 0.016, height - 0.016))
    axis.imshow(np.clip(image * 0.60, 0, 1), origin="upper")
    types = architecture.cell_types_mediumres.astype(str)
    caf = architecture[types.eq("CAF")]
    mac = architecture[types.eq("macrophages")]
    axis.scatter(caf.x, caf.y, s=8, color=CAF, edgecolors="none", alpha=0.76, zorder=3)
    axis.scatter(mac.x, mac.y, s=9, color=MAC, edgecolors="none", alpha=0.78, zorder=4)
    axis.scatter(cd8.x, cd8.y, s=6, color="#B8BEC7", edgecolors="none", alpha=0.40, zorder=5)
    if mode == "niche":
        marked = cd8[cd8.in_niche]
        face_color, edge_color = CYAN, "#061B24"
    else:
        marked = cd8[cd8.high_state]
        face_color, edge_color = YELLOW, "#241A00"
    axis.scatter(
        marked.x, marked.y, s=41, facecolors=face_color, edgecolors=edge_color,
        linewidths=0.9, alpha=0.99, zorder=8,
    )
    axis.set_xlim(V13.CROP[0], V13.CROP[2])
    axis.set_ylim(V13.CROP[3], V13.CROP[1])
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def draw_survival(fig: plt.Figure, bounds: tuple[float, float, float, float]) -> None:
    frame(fig, bounds, RED)
    x, y, width, height = bounds
    axis = fig.add_axes((x + 0.055, y + 0.050, width - 0.085, height - 0.100))
    curves = pd.read_csv(V13.DIRECT_KM)
    for group, color in (("Lower", LOWER), ("Higher", HIGHER)):
        block = curves[curves.group.eq(group)]
        axis.step(block.time, block.survival, where="post", color=color, linewidth=2.6)
    axis.set_xlim(0, 120)
    axis.set_ylim(0.34, 1.03)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.spines[["top", "right", "left", "bottom"]].set_visible(False)


def draw_citeseq(fig: plt.Figure, bounds: tuple[float, float, float, float]) -> None:
    frame(fig, bounds, BLUE)
    axis = fig.add_axes(bounds)
    axis.set_axis_off()
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)

    colors = ["#74B9E6", "#9CCB77", "#D991C3", "#F1BE63"]
    for idx, (cx, cy) in enumerate(((0.12, 0.58), (0.23, 0.67), (0.16, 0.39), (0.29, 0.43))):
        axis.add_patch(Circle((cx, cy), 0.072, facecolor=colors[idx], edgecolor="white", linewidth=1.5))
        axis.add_patch(Circle((cx - 0.012, cy + 0.008), 0.025, facecolor="white", edgecolor="none", alpha=0.78))
        for angle in np.linspace(0, 2 * np.pi, 7, endpoint=False):
            axis.add_patch(Circle((cx + 0.087 * np.cos(angle), cy + 0.087 * np.sin(angle)),
                                  0.009, facecolor=ORANGE, edgecolor="none"))

    gx, gy, gw, gh = 0.45, 0.23, 0.48, 0.55
    rows, cols = 8, 10
    cw, ch = gw / cols, gh / rows
    cool = ["#EEF2F5", "#DCE7EF", "#C7D9E5", "#AFC9DB"]
    for row in range(rows):
        for col in range(cols):
            color = cool[(row * 3 + col * 5 + col // 2) % len(cool)]
            if row in (2, 5):
                color = ["#FFE6C5", "#FFC98A", "#F39A3F", "#E47700"][(col + row) % 4]
            axis.add_patch(Rectangle((gx + col * cw, gy + (rows - 1 - row) * ch),
                                     0.90 * cw, 0.82 * ch, facecolor=color,
                                     edgecolor="white", linewidth=0.25))
    for row, label in ((2, "TIGIT"), (5, "CD39")):
        yy = gy + (rows - 1 - row + 0.42) * ch
        axis.text(gx - 0.018, yy, label, ha="right", va="center",
                  fontsize=7.1, weight="bold", color=ORANGE)
    axis.text(0.18, 0.17, "107 ADDITIONAL\nPROTEINS", ha="center", va="center",
              fontsize=7.7, weight="bold", color=INK, linespacing=1.0)
    axis.text(0.69, 0.11, "TIGIT/CD39 STATE", ha="center", va="center",
              fontsize=8.5, weight="bold", color=ORANGE)


def flow_arrow(
    fig: plt.Figure,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str,
    label_xy: tuple[float, float],
    rotation: float = 0,
) -> None:
    fig.add_artist(FancyArrowPatch(
        start, end, transform=fig.transFigure, arrowstyle="-|>", mutation_scale=17,
        linewidth=2.2, color="#59636E", zorder=30,
    ))
    fig.text(label_xy[0], label_xy[1], label, ha="center", va="center",
             rotation=rotation, fontsize=7.5, weight="bold", color=BLUE,
             bbox={"boxstyle": "round,pad=0.18", "facecolor": "white",
                   "edgecolor": "none", "alpha": 0.96}, zorder=31)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    image = V13.composite()
    cd8, architecture = V13.load_spatial_cells()

    fig = plt.figure(figsize=(6.5, 6.5), facecolor="white")
    fig.text(0.018, 0.982, "A", ha="left", va="top", fontsize=23, weight="bold", color=INK)
    fig.text(0.095, 0.965, "SURVIVAL-LINKED CD8 NICHE DISCOVERY",
             ha="left", va="top", fontsize=12.9, weight="bold", color=INK)

    top_left = (0.055, 0.555, 0.385, 0.330)
    top_right = (0.560, 0.555, 0.385, 0.330)
    bottom_right = (0.560, 0.145, 0.385, 0.305)
    bottom_left = (0.055, 0.145, 0.385, 0.305)

    draw_spatial(fig, top_left, image, cd8, architecture, "niche")
    draw_survival(fig, top_right)
    draw_citeseq(fig, bottom_right)
    draw_spatial(fig, bottom_left, image, cd8, architecture, "return")

    fig.text(0.2475, 0.904, "1  CODEX: CAF/MACROPHAGE-RICH CD8 NICHE", ha="center", va="bottom",
             fontsize=7.9, weight="bold", color=GREEN)
    fig.text(0.7525, 0.904, "2  NICHE ASSOCIATED WITH POORER SURVIVAL", ha="center", va="bottom",
             fontsize=7.9, weight="bold", color=RED)
    fig.text(0.7525, 0.458, "3  CITE-seq MOLECULAR PROFILING", ha="center", va="bottom",
             fontsize=8.5, weight="bold", color=BLUE)
    fig.text(0.2475, 0.468, "4  RETURNED TIGIT/CD39-HIGH CD8", ha="center", va="bottom",
             fontsize=8.5, weight="bold", color=GREEN)

    fig.text(0.072, 0.575, "●", color=CYAN, fontsize=9.0, weight="bold")
    fig.text(0.090, 0.575, "NICHE CD8", color="white", fontsize=7.1, weight="bold",
             bbox={"boxstyle": "round,pad=0.20", "facecolor": "black", "edgecolor": "none", "alpha": 0.72})
    fig.text(0.072, 0.164, "●", color=YELLOW, fontsize=9.0, weight="bold")
    fig.text(0.090, 0.164, "TIGIT/CD39-HIGH CD8", color="white", fontsize=7.1, weight="bold",
             bbox={"boxstyle": "round,pad=0.20", "facecolor": "black", "edgecolor": "none", "alpha": 0.72})

    flow_arrow(fig, (0.450, 0.720), (0.550, 0.720), "SURVIVAL", (0.500, 0.753))
    flow_arrow(fig, (0.822, 0.540), (0.822, 0.464), "RAMP TRANSFER\n22 SHARED", (0.670, 0.512))
    flow_arrow(fig, (0.550, 0.298), (0.450, 0.298), "RAMP RETURN", (0.500, 0.329))

    fig.text(0.500, 0.090, "STATE-HIGH CD8 ENRICHED IN THE ORIGINAL NICHE",
             ha="center", va="center", fontsize=8.4, weight="bold", color=GREEN)
    fig.text(0.500, 0.050, "GREATER NICHE EXPOSURE  →  POORER OVERALL SURVIVAL",
             ha="center", va="center", fontsize=8.4, weight="bold", color=RED)

    png = OUT / "panel_A_visual_cycle_v15.png"
    pdf = OUT / "panel_A_visual_cycle_v15.pdf"
    fig.savefig(png, dpi=320, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
