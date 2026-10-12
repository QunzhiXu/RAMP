#!/usr/bin/env python3
"""Build Panel E from the frozen formal HNSCC spatial-return analysis.

The two maps use the identical real CODEX field.  The left map highlights
CD8 T cells in the independently defined CAF/macrophage-rich niche; the right
map highlights the patient-specific top 40% of the returned TIGIT/CD39 state.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
HNSCC_ROOT = HERE.parent.parent
SOURCE = HNSCC_ROOT / "spatial_return_survival" / "04_plot_spatial_validation.py"
CACHE = HERE / ".cache"
for child in ("matplotlib", "xdg"):
    (CACHE / child).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE / "xdg"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


INK = "#182027"
CYAN = "#20C9FF"
CYAN_EDGE = "#061B24"
GOLD = "#FFD400"
GOLD_EDGE = "#241A00"
OTHER = "#B8BEC7"
CAF = "#39EF72"
MACROPHAGE = "#FF43B5"


def add_flow_box(
    figure: plt.Figure,
    x: float,
    width: float,
    text: str,
    face: str,
    edge: str,
) -> None:
    y, height = 0.815, 0.060
    figure.add_artist(
        FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.005,rounding_size=0.010",
            transform=figure.transFigure,
            facecolor=face,
            edgecolor=edge,
            linewidth=1.05,
            clip_on=False,
        )
    )
    figure.text(
        x + width / 2, y + height / 2, text,
        ha="center", va="center", fontsize=7.5, weight="bold",
        color=INK, linespacing=0.95,
    )


def load_formal_source():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing formal spatial-return script: {SOURCE}")
    spec = importlib.util.spec_from_file_location("hnscc_spatial_return_formal", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=CYAN,
               markeredgecolor=CYAN_EDGE, markersize=6.8,
               label="Observed in-niche CD8"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=GOLD,
               markeredgecolor=GOLD_EDGE, markersize=6.8,
               label="Returned state-high CD8"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=OTHER,
               markeredgecolor="none", markersize=5.7, label="Other CD8"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=CAF,
               markeredgecolor="none", markersize=5.7, label="CAF"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=MACROPHAGE,
               markeredgecolor="none", markersize=5.7, label="Macrophage"),
    ]


def main() -> None:
    source = load_formal_source()
    image = source.composite()
    cd8, architecture = source.load_cells()

    in_niche = cd8.loc[cd8.in_niche]
    state_high = cd8.loc[cd8.high_state]
    overlap = cd8.loc[cd8.in_niche & cd8.high_state]
    if (len(cd8), len(in_niche), len(state_high), len(overlap)) != (61, 32, 28, 22):
        raise AssertionError("Frozen Panel E crop counts have changed")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.4})
    fig = plt.figure(figsize=(10.8, 5.6), facecolor="white")
    fig.text(0.010, 0.975, "E", ha="left", va="top", fontsize=24,
             weight="bold", color=INK)
    fig.text(
        0.054,
        0.949,
        "RETURNED TIGIT/CD39-HIGH CD8 CELLS LOCALIZE TO THE ORIGINAL NICHE",
        ha="left",
        va="top",
        fontsize=13.2,
        weight="bold",
        color=INK,
    )

    add_flow_box(fig, 0.115, 0.190, "TIGIT + CD39\nSCORE", "#FFF0DF", "#E3A064")
    add_flow_box(fig, 0.395, 0.210, "PROJECT TO\nCODEX", "#EAF5F2", "#88BCAF")
    add_flow_box(
        fig, 0.685, 0.230, "CD8 T CELLS WITH\nTOP-40% SCORES", "#FFF8D8", "#D9B72B"
    )
    for start, stop in ((0.305, 0.395), (0.605, 0.685)):
        fig.add_artist(
            FancyArrowPatch(
                (start + 0.005, 0.845), (stop - 0.005, 0.845),
                transform=fig.transFigure, arrowstyle="-|>", mutation_scale=10,
                linewidth=1.15, color="#7C858C", clip_on=False,
            )
        )

    left = fig.add_axes([0.020, 0.080, 0.465, 0.680])
    right = fig.add_axes([0.515, 0.080, 0.465, 0.680])
    for axis in (left, right):
        source.spatial_context(axis, image, cd8, architecture)

    source.highlight(left, in_niche, CYAN, CYAN_EDGE)
    source.highlight(right, state_high, GOLD, GOLD_EDGE)
    left.set_title("OBSERVED CAF/MACROPHAGE-RICH NICHE", loc="left",
                   fontsize=10.3, weight="bold", color=INK, pad=5)
    right.set_title("RETURNED TIGIT/CD39 TOP-40% STATE", loc="left",
                    fontsize=10.3, weight="bold", color=INK, pad=5)

    left.text(
        0.018, 0.023, "CD8 IN NICHE",
        transform=left.transAxes, ha="left", va="bottom", color="white",
        fontsize=8.2, weight="bold",
        bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.76, "pad": 3.2},
        zorder=20,
    )
    right.text(
        0.018, 0.023, "22/28 state-high CD8 are in-niche (78.6%)",
        transform=right.transAxes, ha="left", va="bottom", color="white",
        fontsize=8.2, weight="bold",
        bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.76, "pad": 3.2},
        zorder=20,
    )

    fig.legend(
        handles=legend_handles(),
        ncol=5,
        loc="lower center",
        bbox_to_anchor=(0.54, 0.010),
        frameon=False,
        fontsize=7.5,
        handletextpad=0.35,
        columnspacing=1.0,
    )

    png = HERE / "panel_E.png"
    pdf = HERE / "panel_E.pdf"
    fig.savefig(png, dpi=320, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
