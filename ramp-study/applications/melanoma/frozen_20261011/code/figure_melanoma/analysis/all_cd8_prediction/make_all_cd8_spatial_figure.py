#!/usr/bin/env python3
"""Render the recommended all-CD8 spatial recovery figure for 22RD."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
OUT = HERE / "spatial_figure"
sys.path.insert(0, str(HERE))

import spatial_plotting as style
from matplotlib.lines import Line2D


PREDICTIONS = HERE / "outer_patient_predictions.csv.gz"
SAMPLE_ID = "22RD"
ROI = (600, 700, 900, 995)
CALL_COLUMN = "called_top20"
CALL_LABEL = "whole-section top 20%"


def load_data() -> tuple[pd.DataFrame, object]:
    cells = pd.read_csv(PREDICTIONS)
    cells = cells.loc[cells["patient"].astype(str).eq(SAMPLE_ID)].copy()

    raw = pd.read_csv(
        style.RAW_TABLE,
        sep="\t",
        usecols=["Sample_ID", "Cell_ID", "Location_Center_X", "Location_Center_Y"],
        low_memory=False,
    )
    raw = raw.loc[raw["Sample_ID"].astype(str).eq(SAMPLE_ID)].copy()
    raw["cell_uid_ordered"] = raw["Sample_ID"].astype(str) + "::" + raw["Cell_ID"].astype(str)
    cells = cells.merge(
        raw[["cell_uid_ordered", "Location_Center_X", "Location_Center_Y"]],
        on="cell_uid_ordered",
        validate="one_to_one",
    )
    if cells[["Location_Center_X", "Location_Center_Y"]].isna().any().any():
        raise AssertionError("Missing coordinates")

    sample_info = pd.read_csv(style.SAMPLE_INFO, sep="\t")
    filename = sample_info.loc[
        sample_info["Sample_ID"].astype(str).eq(SAMPLE_ID), "filename"
    ].iloc[0]
    image = style.make_imc_composite(style.TIFF_ROOT / str(filename))
    return cells, image


def masks(cells: pd.DataFrame) -> dict[str, object]:
    xy = cells[["Location_Center_X", "Location_Center_Y"]].to_numpy(float)
    inside = style.inside_bounds(xy, ROI)
    target = cells["target_30um"].to_numpy(bool)
    called = cells[CALL_COLUMN].to_numpy(bool)
    return {
        "xy": xy,
        "inside": inside,
        "target": target,
        "called": called,
        "overlap": inside & target & called,
        "unmatched": inside & ~target & called,
        "observed": inside & target,
    }


def plot_pool(ax, xy, inside) -> None:
    ax.scatter(
        xy[inside, 0], xy[inside, 1],
        s=25, facecolors="none", edgecolors="#54D8E8",
        linewidths=0.85, alpha=0.72, zorder=5,
    )


def render_main(cells: pd.DataFrame, image: object) -> dict[str, object]:
    m = masks(cells)
    xy = m["xy"]
    inside = m["inside"]
    target = m["target"]
    called = m["called"]
    overlap = m["overlap"]
    unmatched = m["unmatched"]
    observed = m["observed"]

    n_pool_whole = int(len(cells))
    n_target_whole = int(target.sum())
    n_called_whole = int(called.sum())
    n_pool = int(inside.sum())
    n_target = int(observed.sum())
    n_called = int((inside & called).sum())
    tp = int(overlap.sum())
    fp = int(unmatched.sum())
    fn = n_target - tp

    style.plt.rcParams.update({"font.family": "Arial", "font.size": 9})
    fig, axes = style.plt.subplots(1, 2, figsize=(9.15, 4.85), dpi=180)
    style.style_spatial_axis(axes[0], image, ROI, 0.93)
    style.style_spatial_axis(axes[1], image, ROI, 0.79)
    plot_pool(axes[0], xy, inside)
    plot_pool(axes[1], xy, inside)

    axes[0].scatter(
        xy[observed, 0], xy[observed, 1], s=79,
        facecolors="#FFD23F", edgecolors="#D7263D",
        linewidths=1.2, alpha=1.0, zorder=9,
    )
    axes[0].set_title(
        f"A  Fixed ground truth  ({n_target}/{n_pool} CD8 cells)",
        loc="left", fontsize=11.2, fontweight="bold", pad=7,
    )
    style.add_scale_bar(axes[0], ROI, 50)

    axes[1].scatter(
        xy[inside & called, 0], xy[inside & called, 1], s=111,
        facecolors="none", edgecolors="white", linewidths=1.9,
        alpha=1.0, zorder=8,
    )
    axes[1].scatter(
        xy[unmatched, 0], xy[unmatched, 1], s=72,
        facecolors="#0072B2", edgecolors="#063B63",
        linewidths=0.7, alpha=0.98, zorder=9,
    )
    axes[1].scatter(
        xy[overlap, 0], xy[overlap, 1], s=75,
        facecolors="#F0E442", edgecolors="#5A5200",
        linewidths=0.7, alpha=1.0, zorder=10,
    )
    axes[1].set_title(
        f"B  Predicted state-high CD8 cells  ({n_called} shown)",
        loc="left", fontsize=11.2, fontweight="bold", pad=7,
    )
    axes[1].text(
        0.96, 0.96,
        f"{tp}/{n_called} shown predictions matched\n"
        f"{tp}/{n_target} ground-truth cells recovered",
        transform=axes[1].transAxes, ha="right", va="top",
        color="white", fontsize=9.0, fontweight="bold", linespacing=1.24,
        bbox={"facecolor": "black", "alpha": 0.64, "edgecolor": "white", "linewidth": 0.45, "pad": 3.35},
        zorder=15,
    )

    handles = [
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="none",
               markeredgecolor="#54D8E8", markeredgewidth=1.2,
               label="All candidate CD8 cells", markersize=6.4),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="#FFD23F",
               markeredgecolor="#D7263D", markeredgewidth=1.2,
               label="Fixed ground truth", markersize=7.2),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="#F0E442",
               markeredgecolor="white", markeredgewidth=1.4,
               label="Matched prediction", markersize=7.2),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="#0072B2",
               markeredgecolor="white", markeredgewidth=1.4,
               label="Unmatched prediction", markersize=7.2),
    ]
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.012),
        ncol=4, frameon=False, fontsize=8.05, handletextpad=0.4,
        columnspacing=0.9,
    )
    fig.suptitle(
        "Spatial recovery of tumor-proximal Ki67+ CD8 T cells",
        x=0.035, y=0.993, ha="left", fontsize=15.0, fontweight="bold",
    )
    fig.text(
        0.035, 0.944,
        f"22RD · all {n_pool_whole} CD8 cells scored across the whole section · "
        f"global top 20% = {n_called_whole} cells · {n_pool} candidates shown",
        ha="left", va="top", fontsize=8.65, color="#4C4C4C",
    )
    fig.subplots_adjust(left=0.035, right=0.992, top=0.86, bottom=0.13, wspace=0.065)

    stem = "22RD_allCD8_fixed30um_global_top20_recommended"
    png = OUT / f"{stem}.png"
    pdf = OUT / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    style.plt.close(fig)

    return {
        "sample_id": SAMPLE_ID,
        "target_definition": "Tc.ae AND prob.Ki67 >= 0.5 AND nearest melanoma centroid <= 30 um",
        "prediction_pool": "all provider CD8: Tc.ae + Tc.naive",
        "call_denominator": "all candidate CD8 cells in the whole donor image",
        "call_rule": CALL_LABEL,
        "n_all_cd8_whole": n_pool_whole,
        "n_ground_truth_whole": n_target_whole,
        "n_called_whole": n_called_whole,
        "roi_bounds_xyxy": list(ROI),
        "n_all_cd8_shown": n_pool,
        "n_tcae_shown": int((inside & cells["is_tcae"].to_numpy(bool)).sum()),
        "n_tcnaive_shown": int((inside & cells["is_tcnaive"].to_numpy(bool)).sum()),
        "n_ground_truth_shown": n_target,
        "n_called_shown": n_called,
        "tp_shown": tp,
        "fp_shown": fp,
        "fn_shown": fn,
        "precision_shown": tp / n_called,
        "recall_shown": tp / n_target,
        "png": str(png),
        "pdf": str(pdf),
    }


def render_cutoff_comparison(cells: pd.DataFrame, image: object) -> dict[str, object]:
    xy = cells[["Location_Center_X", "Location_Center_Y"]].to_numpy(float)
    inside = style.inside_bounds(xy, ROI)
    target = cells["target_30um"].to_numpy(bool)
    observed = inside & target
    n_target = int(observed.sum())
    n_pool = int(inside.sum())

    style.plt.rcParams.update({"font.family": "Arial", "font.size": 9})
    fig, axes = style.plt.subplots(1, 3, figsize=(13.25, 4.55), dpi=180)
    for ax in axes:
        style.style_spatial_axis(ax, image, ROI, 0.92 if ax is axes[0] else 0.79)
        plot_pool(ax, xy, inside)
    axes[0].scatter(
        xy[observed, 0], xy[observed, 1], s=74,
        facecolors="#FFD23F", edgecolors="#D7263D", linewidths=1.15, zorder=9,
    )
    axes[0].set_title(
        f"A  Fixed ground truth  ({n_target}/{n_pool})", loc="left",
        fontsize=11.1, fontweight="bold", pad=7,
    )
    style.add_scale_bar(axes[0], ROI, 50)

    result: dict[str, object] = {"n_ground_truth_shown": n_target, "n_all_cd8_shown": n_pool}
    for ax, column, percent, letter in (
        (axes[1], "called_top15", 15, "B"),
        (axes[2], "called_top20", 20, "C"),
    ):
        called = cells[column].to_numpy(bool)
        overlap = inside & target & called
        unmatched = inside & ~target & called
        n_called = int((inside & called).sum())
        tp = int(overlap.sum())
        fp = int(unmatched.sum())
        ax.scatter(
            xy[inside & called, 0], xy[inside & called, 1], s=105,
            facecolors="none", edgecolors="white", linewidths=1.8, zorder=8,
        )
        ax.scatter(
            xy[unmatched, 0], xy[unmatched, 1], s=68,
            facecolors="#0072B2", edgecolors="#063B63", linewidths=0.65, zorder=9,
        )
        ax.scatter(
            xy[overlap, 0], xy[overlap, 1], s=71,
            facecolors="#F0E442", edgecolors="#5A5200", linewidths=0.65, zorder=10,
        )
        ax.set_title(
            f"{letter}  Global top {percent}%  ({n_called} shown)", loc="left",
            fontsize=11.1, fontweight="bold", pad=7,
        )
        ax.text(
            0.96, 0.96, f"{tp}/{n_called} matched\n{tp}/{n_target} recovered",
            transform=ax.transAxes, ha="right", va="top", color="white",
            fontsize=8.7, fontweight="bold", linespacing=1.2,
            bbox={"facecolor": "black", "alpha": 0.64, "edgecolor": "white", "linewidth": 0.4, "pad": 3.1},
            zorder=15,
        )
        result[f"top{percent}"] = {
            "n_called_whole": int(called.sum()),
            "n_called_shown": n_called,
            "tp_shown": tp,
            "fp_shown": fp,
            "fn_shown": n_target - tp,
        }

    handles = [
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="none",
               markeredgecolor="#54D8E8", markeredgewidth=1.2,
               label="All candidate CD8 cells", markersize=6.3),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="#FFD23F",
               markeredgecolor="#D7263D", markeredgewidth=1.2,
               label="Fixed ground truth", markersize=7.0),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="#F0E442",
               markeredgecolor="white", markeredgewidth=1.4,
               label="Matched prediction", markersize=7.0),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="#0072B2",
               markeredgecolor="white", markeredgewidth=1.4,
               label="Unmatched prediction", markersize=7.0),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.012),
               ncol=4, frameon=False, fontsize=8.05)
    fig.suptitle(
        "Fixed all-CD8 prediction pool with alternative global cutoffs",
        x=0.025, y=0.993, ha="left", fontsize=15.0, fontweight="bold",
    )
    fig.text(
        0.025, 0.944,
        f"22RD · same real IMC field · cutoffs calculated across all {len(cells)} CD8 cells in the whole section",
        ha="left", va="top", fontsize=8.65, color="#4C4C4C",
    )
    fig.subplots_adjust(left=0.025, right=0.995, top=0.85, bottom=0.13, wspace=0.055)
    stem = "22RD_allCD8_fixed_pool_global_top15_vs_top20"
    png = OUT / f"{stem}.png"
    pdf = OUT / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    style.plt.close(fig)
    result["png"] = str(png)
    result["pdf"] = str(pdf)
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cells, image = load_data()
    manifest = {
        "recommended_top20": render_main(cells, image),
        "fixed_pool_cutoff_comparison": render_cutoff_comparison(cells, image),
        "inputs": {"predictions": str(PREDICTIONS)},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
