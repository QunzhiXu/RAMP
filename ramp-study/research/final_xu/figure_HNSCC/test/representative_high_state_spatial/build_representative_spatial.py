#!/usr/bin/env python3
"""Select and render a representative HNSCC high-state CD8 spatial ROI.

This is an illustrative, test-only figure. The high state is the frozen top
quartile of the returned TIGIT/CD39 score within each patient. The spatial
niche is also frozen: positive residual CAF/macrophage exposure among the 15
nearest non-self cells. The background is the real CODEX DAPI/CD8/FAP/CD68
image; cell centroids are overlaid only to make the relevant cell types clear.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
CACHE = HERE / ".cache" / "matplotlib"
CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


FINAL_XU = HERE.parents[2]
PREDICTIONS = (
    FINAL_XU
    / "figure_HNSCC/test/tigit_cd39_return_complete74/output/haist74_pt_cd8_tigit_cd39_return.csv.gz"
)
RAW_CELLS = FINAL_XU / "raw_data/hnscc/haist_2026_codex/HNSCCdiscovery_CODEX.csv"
PLANE_ROOT = (
    FINAL_XU
    / "test/hnscc_survival_first_figure_replacement_20260831/focused_survival_biomarker/"
    "survival_contrast_pair/spatial_images"
)

WIDTH, HEIGHT = 1920, 1440
HIGH_QUANTILE = 0.75
SELECTED_PATIENT = 15219
SELECTED_ROI = "522_S2reg054"
ZOOM = (750, 150, 1650, 850)  # x0, y0, x1, y1

FIGURE_DIR = HERE / "figures"
OUTPUT_DIR = HERE / "output"
OUT_PNG = FIGURE_DIR / "patient15219_roi522_S2reg054_high_state_cd8_spatial.png"
OUT_PDF = FIGURE_DIR / "patient15219_roi522_S2reg054_high_state_cd8_spatial.pdf"
OUT_CANDIDATES = OUTPUT_DIR / "roi_candidate_metrics.csv"
OUT_MANIFEST = OUTPUT_DIR / "figure_manifest.json"


def load_plane(marker: str) -> np.ndarray:
    path = PLANE_ROOT / SELECTED_ROI / "planes" / f"{marker}.raw"
    values = np.fromfile(path, dtype=">u2")
    if values.size != WIDTH * HEIGHT:
        raise ValueError(f"Unexpected plane size for {path}: {values.size}")
    return values.reshape(HEIGHT, WIDTH).astype(np.float32)


def scale_channel(values: np.ndarray, upper_percentile: float, gamma: float) -> np.ndarray:
    positive = values[values > 0]
    upper = float(np.percentile(positive, upper_percentile)) if positive.size else 1.0
    scaled = np.clip(values / max(upper, 1.0), 0.0, 1.0)
    return np.power(scaled, gamma)


def make_composite() -> np.ndarray:
    dapi = scale_channel(load_plane("DAPI"), 99.80, 0.72)
    cd8 = scale_channel(load_plane("CD8"), 99.55, 0.70)
    fap = scale_channel(load_plane("FAP"), 99.70, 0.74)
    cd68 = scale_channel(load_plane("CD68"), 99.55, 0.70)

    rgb = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
    # DAPI: blue-gray; CD8: cyan; FAP: green; CD68: magenta.
    rgb[..., 0] += 0.20 * dapi
    rgb[..., 1] += 0.22 * dapi
    rgb[..., 2] += 0.34 * dapi
    rgb[..., 1] += 0.78 * cd8
    rgb[..., 2] += 0.92 * cd8
    rgb[..., 1] += 0.76 * fap
    rgb[..., 0] += 0.88 * cd68
    rgb[..., 2] += 0.70 * cd68
    return np.clip(np.power(np.clip(rgb, 0, 1), 0.93), 0, 1)


def load_predictions() -> pd.DataFrame:
    data = pd.read_csv(PREDICTIONS)
    data["high_state"] = data["predicted_pair_rank_within_patient"].ge(HIGH_QUANTILE)
    data["in_niche"] = data["spatial_y"].gt(0)
    return data


def roi_metrics(group: pd.DataFrame) -> pd.Series:
    high = group["high_state"]
    n_high = int(high.sum())
    high_niche = float(group.loc[high, "in_niche"].mean()) if n_high else np.nan
    other_niche = float(group.loc[~high, "in_niche"].mean()) if (~high).any() else np.nan
    n_high_niche = int((high & group["in_niche"]).sum())
    enrichment = high_niche - other_niche
    return pd.Series(
        {
            "n_cd8": len(group),
            "n_high_state": n_high,
            "n_high_state_in_niche": n_high_niche,
            "high_state_niche_fraction": high_niche,
            "other_cd8_niche_fraction": other_niche,
            "niche_fraction_difference": enrichment,
            "selection_score": n_high_niche * max(enrichment, 0.0),
        }
    )


def load_architecture() -> pd.DataFrame:
    keep: list[pd.DataFrame] = []
    columns = ["unique_region", "x", "y", "cell_types_mediumres"]
    for chunk in pd.read_csv(RAW_CELLS, usecols=columns, chunksize=150_000):
        hit = chunk[
            chunk["unique_region"].eq(SELECTED_ROI)
            & chunk["cell_types_mediumres"].astype(str).isin(["CAF", "macrophages"])
        ]
        if not hit.empty:
            keep.append(hit.copy())
    if not keep:
        raise RuntimeError(f"No CAF/macrophage cells found in {SELECTED_ROI}")
    return pd.concat(keep, ignore_index=True)


def subset_box(data: pd.DataFrame, box: tuple[int, int, int, int]) -> pd.DataFrame:
    x0, y0, x1, y1 = box
    return data[
        data["x"].between(x0, x1, inclusive="left")
        & data["y"].between(y0, y1, inclusive="left")
    ]


def overlay_cells(
    axis: plt.Axes,
    cd8: pd.DataFrame,
    architecture: pd.DataFrame,
    *,
    zoomed: bool,
) -> None:
    cell_type = architecture["cell_types_mediumres"].astype(str)
    caf = architecture[cell_type.eq("CAF")]
    macrophage = architecture[cell_type.eq("macrophages")]
    other_cd8 = cd8[~cd8["high_state"]]
    high_cd8 = cd8[cd8["high_state"]]

    base_size = 13 if zoomed else 7
    axis.scatter(
        caf["x"], caf["y"], s=base_size,
        facecolors="#3CF270", edgecolors="none", alpha=0.68,
        rasterized=True, zorder=3,
    )
    axis.scatter(
        macrophage["x"], macrophage["y"], s=base_size + 1,
        facecolors="#FF43B5", edgecolors="none", alpha=0.72,
        rasterized=True, zorder=4,
    )
    axis.scatter(
        other_cd8["x"], other_cd8["y"], s=base_size - 2,
        facecolors="#20D9FF", edgecolors="none", alpha=0.50,
        rasterized=True, zorder=5,
    )
    axis.scatter(
        high_cd8["x"], high_cd8["y"],
        s=82 if zoomed else 48,
        facecolors="#FFD400", edgecolors="#241A00", linewidths=1.8 if zoomed else 1.25,
        alpha=0.98, rasterized=True, zorder=7,
    )


def style_spatial_axis(axis: plt.Axes) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_facecolor("black")


def main() -> None:
    required = [PREDICTIONS, RAW_CELLS]
    required.extend(
        PLANE_ROOT / SELECTED_ROI / "planes" / f"{marker}.raw"
        for marker in ["DAPI", "CD8", "FAP", "CD68"]
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(missing))

    predictions = load_predictions()
    candidates = (
        predictions.groupby(["patient", "unique_region"], sort=False)
        .apply(roi_metrics, include_groups=False)
        .reset_index()
        .sort_values(["selection_score", "n_high_state_in_niche"], ascending=False)
    )
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(OUT_CANDIDATES, index=False)

    selected = predictions[
        predictions["patient"].eq(SELECTED_PATIENT)
        & predictions["unique_region"].eq(SELECTED_ROI)
    ].copy()
    if selected.empty:
        raise RuntimeError("Frozen patient/ROI was not found in the returned-state table")
    architecture = load_architecture()
    composite = make_composite()

    selected_metrics = roi_metrics(selected)
    zoom_cd8 = subset_box(selected, ZOOM)
    zoom_architecture = subset_box(architecture, ZOOM)
    zoom_metrics = roi_metrics(zoom_cd8)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5})
    figure, axes = plt.subplots(
        1, 2, figsize=(16.2, 7.0),
        gridspec_kw={"width_ratios": [1.33, 1.0], "wspace": 0.035},
        facecolor="white",
    )
    overview, zoom = axes

    overview.imshow(np.clip(composite * 0.72, 0, 1), origin="upper")
    overlay_cells(overview, selected, architecture, zoomed=False)
    overview.add_patch(
        Rectangle(
            (ZOOM[0], ZOOM[1]), ZOOM[2] - ZOOM[0], ZOOM[3] - ZOOM[1],
            fill=False, edgecolor="white", linewidth=2.0, zorder=10,
        )
    )
    overview.set_xlim(0, WIDTH)
    overview.set_ylim(HEIGHT, 0)
    overview.set_title(
        f"Representative CODEX ROI — patient {SELECTED_PATIENT}, {SELECTED_ROI}",
        loc="left", fontsize=13.0, weight="bold", pad=7,
    )
    style_spatial_axis(overview)

    zoom.imshow(np.clip(composite * 0.72, 0, 1), origin="upper")
    overlay_cells(zoom, zoom_cd8, zoom_architecture, zoomed=True)
    zoom.set_xlim(ZOOM[0], ZOOM[2])
    zoom.set_ylim(ZOOM[3], ZOOM[1])
    zoom.set_title("Zoomed high-state CD8 hotspot", loc="left", fontsize=13.0, weight="bold", pad=7)
    style_spatial_axis(zoom)

    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#FFE34D",
               markeredgecolor="#241A00", markersize=9, label="High-state CD8"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#20D9FF",
               markeredgecolor="none", markersize=7, label="Other CD8"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#3CF270",
               markeredgecolor="none", markersize=7, label="CAF"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#FF43B5",
               markeredgecolor="none", markersize=7, label="Macrophage"),
    ]
    figure.legend(
        handles=handles, ncol=4, loc="upper center", bbox_to_anchor=(0.63, 0.932),
        frameon=False, fontsize=10.5, columnspacing=1.5, handletextpad=0.45,
    )

    high_n = int(selected_metrics["n_high_state"])
    high_niche_n = int(selected_metrics["n_high_state_in_niche"])
    high_frac = 100 * float(selected_metrics["high_state_niche_fraction"])
    other_frac = 100 * float(selected_metrics["other_cd8_niche_fraction"])
    zoom_high_n = int(zoom_metrics["n_high_state"])
    zoom_high_niche_n = int(zoom_metrics["n_high_state_in_niche"])
    zoom_frac = 100 * float(zoom_metrics["high_state_niche_fraction"])

    overview.text(
        0.018, 0.025,
        f"Whole ROI: {high_niche_n}/{high_n} high-state CD8 in niche ({high_frac:.1f}%)\n"
        f"Other CD8 in niche: {other_frac:.1f}%  |  difference: {high_frac-other_frac:+.1f} pp",
        transform=overview.transAxes, ha="left", va="bottom",
        color="white", fontsize=10.5, weight="bold",
        bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.76, "pad": 4.0},
        zorder=20,
    )
    zoom.text(
        0.018, 0.025,
        f"Zoom: {zoom_high_niche_n}/{zoom_high_n} high-state CD8 in niche ({zoom_frac:.1f}%)",
        transform=zoom.transAxes, ha="left", va="bottom",
        color="white", fontsize=10.5, weight="bold",
        bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.76, "pad": 4.0},
        zorder=20,
    )

    figure.suptitle(
        "High-state CD8 T cells accumulate near CAFs and macrophages",
        x=0.014, y=0.993, ha="left", va="top", fontsize=18.0, weight="bold",
    )
    figure.text(
        0.014, 0.012,
        "High state = patient-specific top 25% of the RAMP-returned TIGIT/CD39 score. "
        "Niche = positive residual CAF/macrophage exposure among 15 nearest non-self cells. "
        "Real CODEX channels: DAPI, CD8, FAP and CD68.",
        ha="left", va="bottom", fontsize=9.3, color="#3F444A",
    )
    figure.subplots_adjust(left=0.012, right=0.992, top=0.855, bottom=0.070)
    figure.savefig(OUT_PNG, dpi=240, bbox_inches="tight", facecolor="white")
    figure.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    manifest = {
        "status": "test-only illustrative spatial ROI",
        "selected_patient": SELECTED_PATIENT,
        "selected_roi": SELECTED_ROI,
        "selection_reason": (
            "Strongest locally available true-image ROI by count of high-state CD8 in niche "
            "times high-state-versus-other niche-fraction difference"
        ),
        "high_state_definition": "predicted_pair_rank_within_patient >= 0.75",
        "niche_definition": "spatial_y > 0; residual CAF/macrophage exposure among 15 nearest non-self cells",
        "whole_roi": {key: float(value) for key, value in selected_metrics.items()},
        "zoom_box_xyxy": list(ZOOM),
        "zoom": {key: float(value) for key, value in zoom_metrics.items()},
        "real_codex_channels": ["DAPI", "CD8", "FAP", "CD68"],
        "output_png": str(OUT_PNG),
        "output_pdf": str(OUT_PDF),
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")
    print(f"Wrote {OUT_CANDIDATES}")
    print(f"Wrote {OUT_MANIFEST}")


if __name__ == "__main__":
    main()
