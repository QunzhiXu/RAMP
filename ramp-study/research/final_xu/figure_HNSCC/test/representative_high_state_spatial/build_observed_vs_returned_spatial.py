#!/usr/bin/env python3
"""Plot observed spatial-niche CD8 versus returned TIGIT/CD39-high CD8.

Both panels use the identical real CODEX field so that the two independently
defined CD8 subsets can be compared directly.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
CACHE = HERE / ".cache" / "matplotlib"
CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import build_representative_spatial as base


def common_context(axis: plt.Axes, image: np.ndarray, cd8, architecture, box) -> None:
    axis.imshow(np.clip(image * 0.60, 0, 1), origin="upper")
    cell_type = architecture["cell_types_mediumres"].astype(str)
    caf = architecture[cell_type.eq("CAF")]
    macrophage = architecture[cell_type.eq("macrophages")]

    # Identical anatomical context in both panels.
    axis.scatter(
        caf["x"], caf["y"], s=16, facecolors="#39EF72", edgecolors="none",
        alpha=0.72, rasterized=True, zorder=3,
    )
    axis.scatter(
        macrophage["x"], macrophage["y"], s=17, facecolors="#FF43B5", edgecolors="none",
        alpha=0.76, rasterized=True, zorder=4,
    )
    axis.scatter(
        cd8["x"], cd8["y"], s=13, facecolors="#B8BEC7", edgecolors="none",
        alpha=0.52, rasterized=True, zorder=5,
    )
    axis.set_xlim(box[0], box[2])
    axis.set_ylim(box[3], box[1])
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_facecolor("black")


def highlight(axis: plt.Axes, cells, color: str, edge: str) -> None:
    axis.scatter(
        cells["x"], cells["y"], s=78,
        facecolors=color, edgecolors=edge, linewidths=1.55,
        alpha=0.99, rasterized=True, zorder=8,
    )


def main(top_fraction: float, box: tuple[int, int, int, int], suffix: str) -> None:
    if not 0 < top_fraction < 1:
        raise ValueError("--top-fraction must lie between 0 and 1")
    top_percent = int(round(100 * top_fraction))
    suffix_text = f"_{suffix}" if suffix else ""
    out_stem = (
        f"patient15219_observed_niche_vs_returned_top{top_percent}_high_state"
        f"{suffix_text}"
    )
    out_png = base.FIGURE_DIR / f"{out_stem}.png"
    out_pdf = base.FIGURE_DIR / f"{out_stem}.pdf"

    predictions = base.load_predictions()
    selected = predictions[
        predictions["patient"].eq(base.SELECTED_PATIENT)
        & predictions["unique_region"].eq(base.SELECTED_ROI)
    ].copy()
    # Recompute only the displayed high-state threshold; the score itself is unchanged.
    selected["high_state"] = selected["predicted_pair_rank_within_patient"].ge(
        1.0 - top_fraction
    )
    architecture = base.load_architecture()
    image = base.make_composite()

    cd8 = base.subset_box(selected, box)
    architecture = base.subset_box(architecture, box)
    in_niche = cd8[cd8["in_niche"]]
    high_state = cd8[cd8["high_state"]]
    overlap = cd8[cd8["in_niche"] & cd8["high_state"]]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5})
    figure, axes = plt.subplots(1, 2, figsize=(15.2, 7.1), facecolor="white")

    for axis in axes:
        common_context(axis, image, cd8, architecture, box)

    highlight(axes[0], in_niche, "#20C9FF", "#061B24")
    highlight(axes[1], high_state, "#FFD400", "#241A00")

    axes[0].set_title(
        "A   Observed CAF/macrophage-rich niche",
        loc="left", fontsize=14.0, weight="bold", pad=8,
    )
    axes[1].set_title(
        f"B   RAMP-returned TIGIT/CD39 top {top_percent}% state",
        loc="left", fontsize=14.0, weight="bold", pad=8,
    )

    axes[0].text(
        0.018, 0.024,
        f"{len(in_niche)}/{len(cd8)} CD8 are in-niche",
        transform=axes[0].transAxes, ha="left", va="bottom",
        color="white", fontsize=11.0, weight="bold",
        bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.76, "pad": 4.0},
        zorder=20,
    )
    axes[1].text(
        0.018, 0.024,
        f"{len(overlap)}/{len(high_state)} high-state CD8 are in-niche ({100*len(overlap)/len(high_state):.1f}%)",
        transform=axes[1].transAxes, ha="left", va="bottom",
        color="white", fontsize=11.0, weight="bold",
        bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.76, "pad": 4.0},
        zorder=20,
    )

    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#20C9FF",
               markeredgecolor="#061B24", markersize=9, label="In-niche CD8 (panel A)"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#FFD400",
               markeredgecolor="#241A00", markersize=9, label="High-state CD8 (panel B)"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#B8BEC7",
               markeredgecolor="none", markersize=7, label="Other CD8"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#39EF72",
               markeredgecolor="none", markersize=7, label="CAF"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#FF43B5",
               markeredgecolor="none", markersize=7, label="Macrophage"),
    ]
    figure.legend(
        handles=handles, ncol=5, loc="upper center", bbox_to_anchor=(0.54, 0.926),
        frameon=False, fontsize=10.2, columnspacing=1.25, handletextpad=0.4,
    )
    figure.suptitle(
        f"Spatial return validation — patient {base.SELECTED_PATIENT}, ROI {base.SELECTED_ROI}",
        x=0.014, y=0.992, ha="left", va="top", fontsize=18.0, weight="bold",
    )
    figure.text(
        0.014, 0.012,
        "The panels show the identical CODEX field. In-niche CD8 is defined only from cell coordinates and "
        f"CAF/macrophage identities; high-state CD8 is the patient-specific top {top_percent}% of the returned TIGIT/CD39 score.",
        ha="left", va="bottom", fontsize=9.4, color="#3F444A",
    )
    figure.subplots_adjust(left=0.012, right=0.992, top=0.855, bottom=0.072, wspace=0.028)
    base.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_png, dpi=240, bbox_inches="tight", facecolor="white")
    figure.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")
    print(
        f"Field CD8={len(cd8)}, in_niche={len(in_niche)}, high_state={len(high_state)}, "
        f"overlap={len(overlap)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-fraction", type=float, default=0.25)
    parser.add_argument(
        "--crop", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
        default=base.ZOOM,
    )
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()
    main(args.top_fraction, tuple(args.crop), args.suffix)
