#!/usr/bin/env python3
"""Plot formal HNSCC spatial-return validation on real CODEX imagery."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
CACHE = HERE / ".cache" / "matplotlib"
CACHE.mkdir(parents=True, exist_ok=True)
XDG_CACHE = HERE / ".cache" / "xdg"
XDG_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


FIGURE_HNSCC = HERE.parent
RETURNED = HERE / "output/haist74_pt_cd8_tigit_cd39_return.csv.gz"
QUINTILE_PATIENT = HERE / "output/state_quintile_patient_summary.csv"
RAW_CELLS = (
    FIGURE_HNSCC
    / "sampled_data/haist_codex/complete/HNSCCdiscovery_CODEX.csv"
)
PLANE_ROOT = HERE / "input/planes"
FIGURES = HERE / "figures"

WIDTH, HEIGHT = 1920, 1440
PATIENT = "15219"
ROI = "522_S2reg054"
TOP40_THRESHOLD = 0.60
# Frozen after outcome-free spatial screening for an interpretable field with
# similar numbers of observed in-niche and returned high-state CD8 cells.
CROP = (1205, 205, 1805, 670)

OUT_MAP_PNG = FIGURES / "hnscc_observed_niche_vs_returned_top40_spatial.png"
OUT_MAP_PDF = FIGURES / "hnscc_observed_niche_vs_returned_top40_spatial.pdf"
OUT_COMBINED_PNG = FIGURES / "hnscc_spatial_return_validation.png"
OUT_COMBINED_PDF = FIGURES / "hnscc_spatial_return_validation.pdf"
OUT_MANIFEST = HERE / "output/spatial_validation_manifest.json"
OUT_ROI_CANDIDATES = HERE / "output/roi_candidate_metrics_top40.csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_plane(marker: str) -> np.ndarray:
    path = PLANE_ROOT / f"{marker}.raw"
    values = np.fromfile(path, dtype=">u2")
    if values.size != WIDTH * HEIGHT:
        raise ValueError(f"Unexpected plane size for {path}: {values.size}")
    return values.reshape(HEIGHT, WIDTH).astype(np.float32)


def scale_channel(values: np.ndarray, percentile: float, gamma: float) -> np.ndarray:
    positive = values[values > 0]
    upper = float(np.percentile(positive, percentile)) if positive.size else 1.0
    return np.power(np.clip(values / max(upper, 1.0), 0, 1), gamma)


def composite() -> np.ndarray:
    dapi = scale_channel(load_plane("DAPI"), 99.80, 0.72)
    cd8 = scale_channel(load_plane("CD8"), 99.55, 0.70)
    fap = scale_channel(load_plane("FAP"), 99.70, 0.74)
    cd68 = scale_channel(load_plane("CD68"), 99.55, 0.70)
    rgb = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
    rgb[..., 0] += 0.20 * dapi
    rgb[..., 1] += 0.22 * dapi
    rgb[..., 2] += 0.34 * dapi
    rgb[..., 1] += 0.78 * cd8
    rgb[..., 2] += 0.92 * cd8
    rgb[..., 1] += 0.76 * fap
    rgb[..., 0] += 0.88 * cd68
    rgb[..., 2] += 0.70 * cd68
    return np.clip(np.power(np.clip(rgb, 0, 1), 0.93), 0, 1)


def in_box(frame: pd.DataFrame) -> pd.DataFrame:
    x0, y0, x1, y1 = CROP
    return frame[
        frame.x.between(x0, x1, inclusive="left")
        & frame.y.between(y0, y1, inclusive="left")
    ].copy()


def load_cells() -> tuple[pd.DataFrame, pd.DataFrame]:
    cd8 = pd.read_csv(RETURNED, dtype={"patient": str})
    cd8 = cd8[
        cd8.patient.eq(PATIENT) & cd8.unique_region.astype(str).eq(ROI)
    ].copy()
    cd8["in_niche"] = cd8.spatial_y.gt(0)
    cd8["high_state"] = cd8.within_patient_rank.ge(TOP40_THRESHOLD)
    cd8 = in_box(cd8)

    blocks: list[pd.DataFrame] = []
    columns = ["unique_region", "x", "y", "cell_types_mediumres"]
    for chunk in pd.read_csv(RAW_CELLS, usecols=columns, chunksize=150_000):
        hit = chunk[
            chunk.unique_region.astype(str).eq(ROI)
            & chunk.cell_types_mediumres.astype(str).isin(["CAF", "macrophages"])
        ]
        if not hit.empty:
            blocks.append(hit.copy())
    if not blocks:
        raise RuntimeError(f"No CAF/macrophage cells found for {ROI}")
    architecture = in_box(pd.concat(blocks, ignore_index=True))
    return cd8, architecture


def rank_roi_candidates() -> pd.DataFrame:
    """Rank all ROIs without outcomes using a frozen spatial-display criterion."""
    cells = pd.read_csv(
        RETURNED,
        usecols=[
            "patient", "unique_region", "spatial_y", "within_patient_rank"
        ],
        dtype={"patient": str, "unique_region": str},
    )
    cells["in_niche"] = cells.spatial_y.gt(0)
    cells["high_state"] = cells.within_patient_rank.ge(TOP40_THRESHOLD)
    rows: list[dict[str, object]] = []
    for (patient, roi), frame in cells.groupby(
        ["patient", "unique_region"], sort=False
    ):
        high = frame.high_state.to_numpy(bool)
        niche = frame.in_niche.to_numpy(bool)
        if not high.any() or high.all():
            continue
        overlap = int(np.sum(high & niche))
        high_fraction = float(np.mean(niche[high]))
        other_fraction = float(np.mean(niche[~high]))
        difference = high_fraction - other_fraction
        rows.append(
            {
                "patient": patient,
                "unique_region": roi,
                "pt_cd8_cells": int(len(frame)),
                "top40_cells": int(high.sum()),
                "top40_in_niche_cells": overlap,
                "top40_niche_fraction": high_fraction,
                "other_cd8_niche_fraction": other_fraction,
                "niche_fraction_difference": difference,
                "selection_score": float(overlap * max(difference, 0.0)),
            }
        )
    candidates = (
        pd.DataFrame(rows)
        .sort_values(
            ["selection_score", "top40_in_niche_cells"],
            ascending=False,
            kind="stable",
        )
        .reset_index(drop=True)
    )
    candidates.insert(0, "outcome_free_rank", np.arange(1, len(candidates) + 1))
    if candidates.empty:
        raise RuntimeError("No eligible ROI remained for spatial display ranking")
    selected = candidates.iloc[0]
    if str(selected.patient) != PATIENT or str(selected.unique_region) != ROI:
        raise AssertionError(
            "Frozen representative ROI is no longer first under the declared "
            "outcome-free display criterion"
        )
    temporary = OUT_ROI_CANDIDATES.with_name(f".{OUT_ROI_CANDIDATES.name}.tmp")
    candidates.to_csv(temporary, index=False)
    temporary.replace(OUT_ROI_CANDIDATES)
    return candidates


def spatial_context(
    axis: plt.Axes, image: np.ndarray, cd8: pd.DataFrame, architecture: pd.DataFrame
) -> None:
    axis.imshow(np.clip(image * 0.60, 0, 1), origin="upper")
    types = architecture.cell_types_mediumres.astype(str)
    caf = architecture[types.eq("CAF")]
    macrophage = architecture[types.eq("macrophages")]
    axis.scatter(
        caf.x, caf.y, s=16, color="#39EF72", edgecolors="none", alpha=0.72,
        rasterized=True, zorder=3,
    )
    axis.scatter(
        macrophage.x, macrophage.y, s=17, color="#FF43B5", edgecolors="none",
        alpha=0.76, rasterized=True, zorder=4,
    )
    axis.scatter(
        cd8.x, cd8.y, s=13, color="#B8BEC7", edgecolors="none", alpha=0.52,
        rasterized=True, zorder=5,
    )
    axis.set_xlim(CROP[0], CROP[2])
    axis.set_ylim(CROP[3], CROP[1])
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_facecolor("black")


def highlight(axis: plt.Axes, cells: pd.DataFrame, face: str, edge: str) -> None:
    axis.scatter(
        cells.x, cells.y, s=78, facecolors=face, edgecolors=edge,
        linewidths=1.55, alpha=0.99, rasterized=True, zorder=8,
    )


def map_legend() -> list[Line2D]:
    return [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#20C9FF",
               markeredgecolor="#061B24", markersize=8, label="In-niche CD8 (A)"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#FFD400",
               markeredgecolor="#241A00", markersize=8, label="Top-40% state CD8 (B)"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#B8BEC7",
               markeredgecolor="none", markersize=6, label="Other CD8"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#39EF72",
               markeredgecolor="none", markersize=6, label="CAF"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#FF43B5",
               markeredgecolor="none", markersize=6, label="Macrophage"),
    ]


def draw_maps(
    axes: tuple[plt.Axes, plt.Axes],
    image: np.ndarray,
    cd8: pd.DataFrame,
    architecture: pd.DataFrame,
) -> tuple[int, int, int]:
    left, right = axes
    for axis in axes:
        spatial_context(axis, image, cd8, architecture)
    niche = cd8[cd8.in_niche]
    high = cd8[cd8.high_state]
    overlap = cd8[cd8.in_niche & cd8.high_state]
    highlight(left, niche, "#20C9FF", "#061B24")
    highlight(right, high, "#FFD400", "#241A00")
    left.set_title("A   Observed CAF/macrophage-rich niche", loc="left", weight="bold")
    right.set_title("B   Returned TIGIT/CD39 top-40% state", loc="left", weight="bold")
    left.text(
        0.018, 0.024, f"{len(niche)}/{len(cd8)} CD8 are in-niche",
        transform=left.transAxes, color="white", weight="bold", ha="left", va="bottom",
        bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.76, "pad": 3.5},
    )
    right.text(
        0.018, 0.024,
        f"{len(overlap)}/{len(high)} state-high CD8 are in-niche ({100*len(overlap)/len(high):.1f}%)",
        transform=right.transAxes, color="white", weight="bold", ha="left", va="bottom",
        bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.76, "pad": 3.5},
    )
    return len(niche), len(high), len(overlap)


def draw_quintiles(axis: plt.Axes) -> dict[str, float]:
    frame = pd.read_csv(QUINTILE_PATIENT, dtype={"patient": str})
    order = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    groups = [
        frame.loc[frame.state_quintile.eq(label), "niche_fraction"].dropna().to_numpy(float)
        for label in order
    ]
    box = axis.boxplot(
        [100 * values for values in groups], positions=np.arange(1, 6), widths=0.58,
        patch_artist=True, showfliers=False,
        medianprops={"color": "#17324D", "linewidth": 1.7},
        whiskerprops={"color": "#667788"}, capprops={"color": "#667788"},
    )
    for patch in box["boxes"]:
        patch.set_facecolor("#D8EBF7")
        patch.set_edgecolor("#437BA4")
    pooled = frame.groupby("state_quintile", observed=True).apply(
        lambda x: np.average(x.niche_fraction, weights=x.cells), include_groups=False
    ).reindex(order)
    axis.plot(np.arange(1, 6), 100 * pooled.to_numpy(), color="#D97706", marker="o",
              linewidth=2.0, markersize=5.5, label="Cell-weighted fraction")
    pivot = frame.pivot(index="patient", columns="state_quintile", values="niche_fraction")
    paired = pivot.dropna(subset=["Q1", "Q5"])
    delta = paired.Q5 - paired.Q1
    p_value = float(wilcoxon(delta, alternative="greater", zero_method="wilcox").pvalue)
    positive = int((delta > 0).sum())
    axis.set_xticks(np.arange(1, 6), order)
    axis.set_ylabel("CD8 cells in niche (%)")
    axis.set_title("C   Cohort-wide state gradient", loc="left", weight="bold")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#D8DEE5", linewidth=0.7, alpha=0.8)
    axis.legend(frameon=False, loc="upper left", fontsize=8.5)
    axis.text(
        0.98, 0.04,
        f"Q1: {100*pooled.iloc[0]:.1f}%  →  Q5: {100*pooled.iloc[-1]:.1f}%\n"
        f"Q5 > Q1 niche fraction: {positive}/{len(delta)} patients\n"
        f"paired Wilcoxon P={p_value:.2g}",
        transform=axis.transAxes, ha="right", va="bottom", fontsize=9.0,
        bbox={"facecolor": "white", "edgecolor": "#C9D1D9", "alpha": 0.94, "pad": 4},
    )
    return {
        "q1_pooled_niche_fraction": float(pooled.iloc[0]),
        "q5_pooled_niche_fraction": float(pooled.iloc[-1]),
        "q5_minus_q1_niche_fraction_positive_patients": positive,
        "q5_minus_q1_niche_fraction_patients": int(len(delta)),
        "q5_minus_q1_niche_fraction_wilcoxon_p": p_value,
    }


def main() -> None:
    required = [RETURNED, QUINTILE_PATIENT, RAW_CELLS]
    required.extend(PLANE_ROOT / f"{marker}.raw" for marker in ["DAPI", "CD8", "FAP", "CD68"])
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing formal inputs:\n" + "\n".join(missing))
    FIGURES.mkdir(parents=True, exist_ok=True)
    roi_candidates = rank_roi_candidates()
    image = composite()
    cd8, architecture = load_cells()

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5})
    maps, axes = plt.subplots(1, 2, figsize=(15.2, 7.1), facecolor="white")
    niche_cells, high_cells, overlap_cells = draw_maps(
        (axes[0], axes[1]), image, cd8, architecture
    )
    maps.legend(handles=map_legend(), ncol=5, loc="upper center", bbox_to_anchor=(0.54, 0.925),
                frameon=False, fontsize=10.0, columnspacing=1.2, handletextpad=0.4)
    maps.suptitle(
        f"Spatial return validation — patient {PATIENT}, ROI {ROI}",
        x=0.014, y=0.992, ha="left", va="top", fontsize=18, weight="bold",
    )
    maps.text(
        0.014, 0.012,
        "Identical real CODEX field in both panels. Niche uses only coordinates and CAF/macrophage identities; "
        "state-high uses only the returned molecular score.",
        ha="left", va="bottom", fontsize=9.3, color="#3F444A",
    )
    maps.subplots_adjust(left=0.012, right=0.992, top=0.855, bottom=0.072, wspace=0.028)
    maps.savefig(OUT_MAP_PNG, dpi=240, bbox_inches="tight", facecolor="white")
    maps.savefig(OUT_MAP_PDF, bbox_inches="tight", facecolor="white")
    plt.close(maps)

    figure = plt.figure(figsize=(19.0, 6.45), facecolor="white")
    grid = figure.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.72], wspace=0.055)
    left = figure.add_subplot(grid[0, 0])
    right = figure.add_subplot(grid[0, 1])
    trend = figure.add_subplot(grid[0, 2])
    draw_maps((left, right), image, cd8, architecture)
    trend_statistics = draw_quintiles(trend)
    figure.legend(handles=map_legend(), ncol=5, loc="upper left", bbox_to_anchor=(0.012, 0.928),
                  frameon=False, fontsize=9.2, columnspacing=1.0, handletextpad=0.35)
    figure.suptitle(
        "Returned TIGIT/CD39-high CD8 cells are enriched in CAF/macrophage-rich niches",
        x=0.012, y=0.995, ha="left", va="top", fontsize=17.5, weight="bold",
    )
    figure.text(
        0.012, 0.012,
        "High state = patient-specific top 40% of the returned TIGIT/CD39 program. "
        "This analysis tests preferential spatial enrichment, not exact niche classification.",
        ha="left", va="bottom", fontsize=9.2, color="#3F444A",
    )
    figure.subplots_adjust(left=0.012, right=0.992, top=0.82, bottom=0.085)
    figure.savefig(OUT_COMBINED_PNG, dpi=240, bbox_inches="tight", facecolor="white")
    figure.savefig(OUT_COMBINED_PDF, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    type_counts = architecture.cell_types_mediumres.astype(str).value_counts()
    manifest = {
        "status": "formal true-CODEX spatial-return validation",
        "patient": PATIENT,
        "roi": ROI,
        "crop_xyxy": list(CROP),
        "roi_selection": {
            "outcome_free": True,
            "candidate_universe": "all Haist primary-tumour ROIs containing eligible PT-CD8 cells",
            "criterion": (
                "top40_in_niche_cells multiplied by the positive difference "
                "between top-40% and other-CD8 niche fractions"
            ),
            "ranking_table": str(OUT_ROI_CANDIDATES.relative_to(HERE)),
            "selected_rank": int(roi_candidates.iloc[0].outcome_free_rank),
            "selected_score": float(roi_candidates.iloc[0].selection_score),
        },
        "high_state_definition": "within_patient_rank >= 0.60",
        "interpretation": (
            "Preferential enrichment of returned TIGIT/CD39-high PT-CD8 cells "
            "in the independently defined CAF/macrophage-rich niche; not exact "
            "cell-by-cell niche classification."
        ),
        "crop_counts": {
            "pt_cd8_cells": int(len(cd8)),
            "observed_in_niche_cd8": int(niche_cells),
            "returned_top40_cd8": int(high_cells),
            "overlap": int(overlap_cells),
            "top40_niche_precision": float(overlap_cells / high_cells),
            "top40_niche_recall": float(overlap_cells / niche_cells),
            "caf_cells": int(type_counts.get("CAF", 0)),
            "macrophage_cells": int(type_counts.get("macrophages", 0)),
        },
        "cohort_state_gradient": trend_statistics,
        "inputs": {
            "returned_cells": str(RETURNED.relative_to(HERE)),
            "patient_quintiles": str(QUINTILE_PATIENT.relative_to(HERE)),
            "complete_codex_cells": os.path.relpath(RAW_CELLS, HERE),
            "real_codex_planes": {
                marker: {
                    "path": str((PLANE_ROOT / f"{marker}.raw").relative_to(HERE)),
                    "bytes": int((PLANE_ROOT / f"{marker}.raw").stat().st_size),
                    "hard_link_count": int((PLANE_ROOT / f"{marker}.raw").stat().st_nlink),
                    "sha256": sha256_file(PLANE_ROOT / f"{marker}.raw"),
                }
                for marker in ["DAPI", "CD8", "FAP", "CD68"]
            },
        },
        "outputs": [
            str(path.relative_to(HERE))
            for path in [OUT_MAP_PNG, OUT_MAP_PDF, OUT_COMBINED_PNG, OUT_COMBINED_PDF]
        ],
        "outcome_used_for_roi_or_crop_selection": False,
        "test_or_test0827_input_dependency": False,
    }
    temporary = OUT_MANIFEST.with_name(f".{OUT_MANIFEST.name}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(OUT_MANIFEST)
    print(f"Wrote {OUT_MAP_PNG}")
    print(f"Wrote {OUT_MAP_PDF}")
    print(f"Wrote {OUT_COMBINED_PNG}")
    print(f"Wrote {OUT_COMBINED_PDF}")
    print(f"Wrote {OUT_MANIFEST}")


if __name__ == "__main__":
    main()
