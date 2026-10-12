#!/usr/bin/env python3
"""Build a first layout draft for the survival-first HNSCC story figure.

This script only recomposes existing analysis figures and the already-rendered
representative spatial pair. It does not rerun alignment, biomarker discovery,
or survival models.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "figures"
CACHE = HERE / ".cache"
CACHE.mkdir(parents=True, exist_ok=True)
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
from PIL import Image


PAIR_FIGURE = (
    ROOT.parent
    / "test"
    / "hnscc_survival_first_figure_replacement_20260831"
    / "focused_survival_biomarker"
    / "survival_contrast_pair"
    / "figures"
    / "hnscc_survival_pair_to_biomarkers_figure.png"
)
DIRECT_SURVIVAL = ROOT / "survival_motivation" / "figures" / "hnscc_pt_cd8_niche_survival.png"
ALIGNMENT_UMAP = ROOT / "alignment" / "umap" / "hnscc_pt74_ramp_ver2_full_94488_umap.png"
MARKER_SCREEN = (
    ROOT / "biomarker_discovery" / "screen" / "marker_screen_all_107_all_variants.csv"
)
SPATIAL_RETURN = (
    ROOT / "spatial_return_survival" / "figures" / "hnscc_spatial_return_validation.png"
)
RETURNED_SURVIVAL = (
    ROOT / "spatial_return_survival" / "figures" / "tigit_cd39_complete74_top40_survival.png"
)


BLUE = "#2878B5"
RED = "#D64A3C"
ORANGE = "#E57A00"
GREEN = "#2AA876"
GREY = "#6B7280"


def read_image(path: Path, crop: tuple[int, int, int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if crop is not None:
        image = image.crop(crop)
    return np.asarray(image)


def panel_label(axis: plt.Axes, label: str, x: float = -0.025, y: float = 1.02) -> None:
    axis.text(
        x,
        y,
        label,
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=22,
        weight="bold",
        color="black",
        clip_on=False,
        zorder=20,
    )


def show_figure(axis: plt.Axes, path: Path) -> None:
    axis.imshow(read_image(path))
    axis.set_axis_off()


def draw_workflow(axis: plt.Axes) -> None:
    axis.set_axis_off()
    panel_label(axis, "A", x=-0.012, y=0.86)
    boxes = [
        ("Haist CODEX\n74 patients", "#EAF7F2", GREEN),
        ("CAF/macrophage-rich\nCD8 niche", "#EAF7F2", GREEN),
        ("Poorer overall\nsurvival", "#FCEBEC", RED),
        ("RAMP → Rahim CITE-seq\n22 shared proteins", "#EAF2FB", BLUE),
        ("107 additional ADTs\nTIGIT + CD39", "#FFF3E6", ORANGE),
        ("RAMP → Haist CODEX\nspatial return", "#EAF2FB", BLUE),
        ("Niche enrichment\n+ poorer survival", "#FCEBEC", RED),
    ]
    left = 0.012
    gap = 0.012
    width = (0.976 - gap * (len(boxes) - 1)) / len(boxes)
    y, height = 0.17, 0.66
    for index, (text, face, edge) in enumerate(boxes):
        x = left + index * (width + gap)
        axis.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                transform=axis.transAxes,
                boxstyle="round,pad=0.008,rounding_size=0.014",
                facecolor=face,
                edgecolor=edge,
                linewidth=1.8,
            )
        )
        axis.text(
            x + width / 2,
            y + height / 2,
            text,
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=11.1,
            weight="bold",
            color="#161616",
            linespacing=1.08,
        )
        if index < len(boxes) - 1:
            start = x + width + 0.001
            end = x + width + gap - 0.001
            axis.add_patch(
                FancyArrowPatch(
                    (start, 0.50),
                    (end, 0.50),
                    transform=axis.transAxes,
                    arrowstyle="-|>",
                    mutation_scale=15,
                    linewidth=1.5,
                    color=GREY,
                )
            )


def draw_marker_screen(axis: plt.Axes) -> None:
    data = pd.read_csv(MARKER_SCREEN)
    x = data["robust_median_rank_primary_variants"]
    y = data["paper_clr__mean_donor_association"]
    robust = data["selected_after_redundancy_pruning"].astype(bool)
    compact = data["selected_compact_top5pct"].astype(bool)

    axis.scatter(x, y, s=28, c="#C9CDD2", alpha=0.76, linewidths=0, rasterized=True)
    axis.scatter(
        x[robust & ~compact],
        y[robust & ~compact],
        s=60,
        c=BLUE,
        alpha=0.98,
        linewidths=0,
        rasterized=True,
    )
    axis.scatter(
        x[compact],
        y[compact],
        s=105,
        c=ORANGE,
        edgecolors="#9A3E00",
        linewidths=0.8,
        zorder=5,
    )
    axis.axhline(0, color="#7A7A7A", linestyle="--", linewidth=1.0)
    axis.axvline(15, color="#9B9B9B", linestyle=":", linewidth=1.3)
    axis.axvline(5.35, color=ORANGE, linestyle="--", linewidth=1.3)
    for marker in ["TIGIT", "CD39"]:
        row = data.loc[data["marker"].eq(marker)].iloc[0]
        dy = 0.018 if marker == "CD39" else -0.018
        axis.annotate(
            marker,
            (row["robust_median_rank_primary_variants"], row["paper_clr__mean_donor_association"]),
            xytext=(9, 8 if dy > 0 else -16),
            textcoords="offset points",
            fontsize=13,
            weight="bold",
            color="#7F2D00",
        )
    axis.set_xlim(0, 110)
    axis.set_ylim(-0.53, 0.32)
    axis.set_xlabel("Robust median rank across four views (lower is stronger)", fontsize=11)
    axis.set_ylabel("Equal-donor paper-CLR association", fontsize=11)
    axis.tick_params(labelsize=9)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(color="#E3E6E8", linewidth=0.65, alpha=0.72)
    axis.set_title(
        "Outcome-blind screen: 107 candidates → 8 robust → TIGIT/CD39",
        loc="left",
        fontsize=14,
        weight="bold",
        pad=8,
    )
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#C9CDD2", markeredgewidth=0, label="All candidates"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=BLUE, markeredgewidth=0, label="Robust markers"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=ORANGE, markeredgecolor="#9A3E00", label="Final compact program"),
    ]
    axis.legend(handles=handles, frameon=False, fontsize=9, loc="lower left")
    panel_label(axis, "E", x=-0.04, y=1.02)


def main() -> None:
    required = [PAIR_FIGURE, DIRECT_SURVIVAL, ALIGNMENT_UMAP, MARKER_SCREEN, SPATIAL_RETURN, RETURNED_SURVIVAL]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(missing))
    OUT.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    figure = plt.figure(figsize=(24, 17), facecolor="white")
    outer = figure.add_gridspec(
        4,
        1,
        height_ratios=[1.25, 4.55, 4.60, 4.35],
        hspace=0.24,
        left=0.027,
        right=0.987,
        bottom=0.025,
        top=0.945,
    )
    figure.suptitle(
        "Survival-first bidirectional protein-state discovery in HNSCC",
        x=0.027,
        y=0.987,
        ha="left",
        va="top",
        fontsize=24,
        weight="bold",
    )

    workflow_axis = figure.add_subplot(outer[0])
    draw_workflow(workflow_axis)

    row_one = outer[1].subgridspec(1, 2, width_ratios=[1.10, 1.00], wspace=0.055)
    spatial_pair = row_one[0].subgridspec(1, 2, wspace=0.025)
    pair_source = Image.open(PAIR_FIGURE).convert("RGB")
    # Crop only the two real spatial maps; old workflow and biomarker panels are excluded.
    pair_crops = [
        np.asarray(pair_source.crop((470, 515, 1550, 1250))),
        np.asarray(pair_source.crop((2290, 515, 3370, 1250))),
    ]
    pair_titles = [
        ("Low niche exposure | alive at 72 months", BLUE, "Patient 2427"),
        ("High niche exposure | death at 25 months", RED, "Patient 9566"),
    ]
    for index in range(2):
        axis = figure.add_subplot(spatial_pair[0, index])
        axis.imshow(pair_crops[index])
        axis.set_axis_off()
        title, color, patient = pair_titles[index]
        axis.set_title(f"{title}\n{patient}", loc="left", fontsize=13.2, weight="bold", color=color, pad=6)
        if index == 0:
            panel_label(axis, "B", x=-0.035, y=1.09)

    direct_axis = figure.add_subplot(row_one[1])
    show_figure(direct_axis, DIRECT_SURVIVAL)
    panel_label(direct_axis, "C", x=-0.018, y=1.02)

    row_two = outer[2].subgridspec(1, 2, width_ratios=[1.24, 0.76], wspace=0.060)
    umap_axis = figure.add_subplot(row_two[0])
    show_figure(umap_axis, ALIGNMENT_UMAP)
    panel_label(umap_axis, "D", x=-0.018, y=1.02)
    marker_axis = figure.add_subplot(row_two[1])
    draw_marker_screen(marker_axis)

    row_three = outer[3].subgridspec(1, 2, width_ratios=[1.48, 0.92], wspace=0.055)
    return_axis = figure.add_subplot(row_three[0])
    show_figure(return_axis, SPATIAL_RETURN)
    panel_label(return_axis, "F", x=-0.015, y=1.02)
    survival_axis = figure.add_subplot(row_three[1])
    show_figure(survival_axis, RETURNED_SURVIVAL)
    panel_label(survival_axis, "G", x=-0.018, y=1.02)

    png = OUT / "hnscc_survival_first_big_figure_draft_v1.png"
    pdf = OUT / "hnscc_survival_first_big_figure_draft_v1.pdf"
    figure.savefig(png, dpi=240, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    manifest = {
        "status": "layout draft v1",
        "story": "spatial survival motivation -> RAMP transfer -> outcome-blind TIGIT/CD39 discovery -> spatial return -> survival",
        "excluded_from_main_figure": [
            "Rahim 4,788-cell score-colored UMAP (overall Spearman 0.262)",
            "Haist leave-one-patient-predictor diagnostic (median Spearman 0.100)",
            "full donor/preprocessing robustness panel (reserved for supplement)",
        ],
        "panels": {
            "A": "survival-first bidirectional workflow",
            "B": "illustrative matched-patient real CODEX spatial contrast",
            "C": "74-patient direct spatial-niche survival association",
            "D": "94,488-cell RAMP alignment UMAP",
            "E": "outcome-blind 107-ADT screen identifying TIGIT/CD39",
            "F": "real-field and cohort-wide spatial return validation",
            "G": "returned TIGIT/CD39-high CD8 spatial-exposure survival association",
        },
        "inputs": [str(path) for path in required],
        "outputs": {"png": str(png), "pdf": str(pdf)},
        "note": "Panel B is cropped from a prior rendered figure for layout review; a final build should render its spatial data directly inside the formal folder.",
    }
    manifest_path = OUT / "hnscc_survival_first_big_figure_draft_v1_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
