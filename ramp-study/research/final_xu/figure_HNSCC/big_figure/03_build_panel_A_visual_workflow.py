#!/usr/bin/env python3
"""Build an image-led square overview for the HNSCC niche-state story.

The spatial thumbnails and survival curves are real data.  The compact
CITE-seq protein grid is a schematic of the measured panel; it is deliberately
not presented as a quantitative heatmap because the detailed marker evidence
is shown in a later panel of the main figure.
"""

from __future__ import annotations

import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "panels"
CACHE = HERE / ".cache"
for child in ("matplotlib", "xdg"):
    (CACHE / child).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE / "xdg"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd


INK = "#171717"
MUTED = "#53606B"
BLUE = "#2775B6"
GREEN = "#159A72"
ORANGE = "#E47700"
RED = "#D84A3F"
CYAN = "#20C9FF"
YELLOW = "#FFD400"
CAF = "#39EF72"
MAC = "#FF43B5"
LOWER = "#2F6FB0"
HIGHER = "#C9403A"

RETURN_ROOT = ROOT / "spatial_return_survival"
PLANES = RETURN_ROOT / "input/planes"
RETURNED = RETURN_ROOT / "output/haist74_pt_cd8_tigit_cd39_return.csv.gz"
RAW_CELLS = ROOT / "sampled_data/haist_codex/complete/HNSCCdiscovery_CODEX.csv"
DIRECT_KM = ROOT / "survival_motivation/output/median_split_km_curve.csv"
RETURN_KM = RETURN_ROOT / "output/top40_median_split_km_curve.csv"

WIDTH, HEIGHT = 1920, 1440
PATIENT = "15219"
ROI = "522_S2reg054"
CROP = (1205, 205, 1805, 670)


def load_plane(marker: str) -> np.ndarray:
    values = np.fromfile(PLANES / f"{marker}.raw", dtype=">u2")
    if values.size != WIDTH * HEIGHT:
        raise ValueError(f"Unexpected image size for {marker}: {values.size}")
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


def in_crop(frame: pd.DataFrame) -> pd.DataFrame:
    x0, y0, x1, y1 = CROP
    return frame[
        frame.x.between(x0, x1, inclusive="left")
        & frame.y.between(y0, y1, inclusive="left")
    ].copy()


def load_spatial_cells() -> tuple[pd.DataFrame, pd.DataFrame]:
    cd8 = pd.read_csv(RETURNED, dtype={"patient": str, "unique_region": str})
    cd8 = cd8[cd8.patient.eq(PATIENT) & cd8.unique_region.eq(ROI)].copy()
    cd8["in_niche"] = cd8.spatial_y.gt(0)
    cd8["high_state"] = cd8.within_patient_rank.ge(0.60)
    cd8 = in_crop(cd8)

    pieces: list[pd.DataFrame] = []
    columns = ["unique_region", "x", "y", "cell_types_mediumres"]
    for chunk in pd.read_csv(RAW_CELLS, usecols=columns, chunksize=150_000):
        hit = chunk[
            chunk.unique_region.astype(str).eq(ROI)
            & chunk.cell_types_mediumres.astype(str).isin(["CAF", "macrophages"])
        ]
        if not hit.empty:
            pieces.append(hit.copy())
    architecture = in_crop(pd.concat(pieces, ignore_index=True))
    return cd8, architecture


def rounded_frame(fig: plt.Figure, bounds: tuple[float, float, float, float], color: str) -> None:
    x, y, width, height = bounds
    fig.add_artist(
        FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.007,rounding_size=0.014",
            transform=fig.transFigure,
            facecolor="white", edgecolor=color, linewidth=2.0, zorder=0,
        )
    )


def spatial_axis(
    fig: plt.Figure,
    bounds: tuple[float, float, float, float],
    image: np.ndarray,
    cd8: pd.DataFrame,
    architecture: pd.DataFrame,
    mode: str,
) -> plt.Axes:
    rounded_frame(fig, bounds, GREEN)
    pad = 0.008
    x, y, width, height = bounds
    axis = fig.add_axes((x + pad, y + pad, width - 2 * pad, height - 2 * pad))
    axis.imshow(np.clip(image * 0.60, 0, 1), origin="upper")
    types = architecture.cell_types_mediumres.astype(str)
    caf = architecture[types.eq("CAF")]
    mac = architecture[types.eq("macrophages")]
    axis.scatter(caf.x, caf.y, s=8, color=CAF, edgecolors="none", alpha=0.76, zorder=3)
    axis.scatter(mac.x, mac.y, s=9, color=MAC, edgecolors="none", alpha=0.78, zorder=4)
    axis.scatter(cd8.x, cd8.y, s=6, color="#B8BEC7", edgecolors="none", alpha=0.43, zorder=5)
    if mode == "niche":
        marked = cd8[cd8.in_niche]
        face, edge = CYAN, "#061B24"
    elif mode == "return":
        marked = cd8[cd8.high_state]
        face, edge = YELLOW, "#241A00"
    else:
        raise ValueError(mode)
    axis.scatter(
        marked.x, marked.y, s=35, facecolors=face, edgecolors=edge,
        linewidths=0.8, alpha=0.98, zorder=8,
    )
    axis.set_xlim(CROP[0], CROP[2])
    axis.set_ylim(CROP[3], CROP[1])
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    return axis


def survival_axis(
    fig: plt.Figure,
    bounds: tuple[float, float, float, float],
    path: Path,
    line1: str,
    line2: str,
) -> plt.Axes:
    rounded_frame(fig, bounds, RED)
    x, y, width, height = bounds
    axis = fig.add_axes((x + 0.025, y + 0.050, width - 0.045, height - 0.090))
    frame = pd.read_csv(path)
    for group, color in (("Lower", LOWER), ("Higher", HIGHER)):
        block = frame[frame.group.eq(group)]
        axis.step(block.time, block.survival, where="post", color=color, linewidth=2.3)
    axis.set_xlim(0, 120)
    axis.set_ylim(0.34, 1.03)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.spines[["top", "right", "left", "bottom"]].set_visible(False)
    axis.text(
        0.5, 1.18, f"{line1}\n{line2}", transform=axis.transAxes,
        ha="center", va="bottom", fontsize=7.9, weight="bold", color=INK,
        linespacing=1.0,
    )
    return axis


def cite_seq_panel(fig: plt.Figure, bounds: tuple[float, float, float, float]) -> None:
    rounded_frame(fig, bounds, BLUE)
    x, y, width, height = bounds
    axis = fig.add_axes(bounds)
    axis.set_axis_off()
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)

    axis.text(0.035, 0.88, "CITE-seq", fontsize=10.5, weight="bold", color=BLUE, va="center")
    axis.text(
        0.035, 0.69, "107 ADDITIONAL\nPROTEINS", fontsize=8.4,
        weight="bold", color=INK, va="center", linespacing=1.0,
    )

    # A compact assay icon: cells feeding into a marker-by-cell protein grid.
    cell_colors = ["#74B9E6", "#9CCB77", "#D991C3", "#F1BE63", "#8CB3A4"]
    cell_xy = [(0.085, 0.38), (0.155, 0.47), (0.225, 0.36), (0.145, 0.25), (0.235, 0.22)]
    for idx, ((cx, cy), color) in enumerate(zip(cell_xy, cell_colors)):
        axis.add_patch(Circle((cx, cy), 0.050, facecolor=color, edgecolor="white", linewidth=1.4))
        axis.add_patch(Circle((cx - 0.010, cy + 0.005), 0.018, facecolor="#F7F8FA", edgecolor="none", alpha=0.8))
        for angle in np.linspace(0, 2 * np.pi, 7, endpoint=False):
            px = cx + 0.061 * np.cos(angle + idx * 0.22)
            py = cy + 0.061 * np.sin(angle + idx * 0.22)
            axis.add_patch(Circle((px, py), 0.007, facecolor=ORANGE, edgecolor="none"))

    grid_x0, grid_y0, grid_w, grid_h = 0.35, 0.17, 0.59, 0.62
    rows, cols = 9, 15
    cell_w, cell_h = grid_w / cols, grid_h / rows
    palette = ["#EEF2F5", "#DCE7EF", "#C7D9E5", "#AFC9DB"]
    for row in range(rows):
        for col in range(cols):
            shade = palette[(row * 3 + col * 5 + (col // 3)) % len(palette)]
            if row in (2, 6):
                shade = ["#FFE6C5", "#FFC98A", "#F39A3F", "#E47700"][(col + row) % 4]
            axis.add_patch(
                Rectangle(
                    (grid_x0 + col * cell_w, grid_y0 + (rows - 1 - row) * cell_h),
                    cell_w * 0.91, cell_h * 0.83,
                    facecolor=shade, edgecolor="white", linewidth=0.25,
                )
            )
    for row, label in ((2, "TIGIT"), (6, "CD39")):
        cy = grid_y0 + (rows - 1 - row + 0.42) * cell_h
        axis.text(grid_x0 - 0.012, cy, label, ha="right", va="center", fontsize=7.2,
                  color=ORANGE, weight="bold")
    axis.text(0.645, 0.91, "PROTEIN PANEL", fontsize=8.1, weight="bold", color=MUTED,
              ha="center", va="center")
    axis.text(0.645, 0.075, "TIGIT/CD39 STATE", fontsize=9.0, weight="bold", color=ORANGE,
              ha="center", va="center")


def arrow(
    fig: plt.Figure,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str,
) -> None:
    fig.add_artist(
        FancyArrowPatch(
            start, end, transform=fig.transFigure, arrowstyle="-|>", mutation_scale=17,
            linewidth=2.2, color=BLUE, connectionstyle="arc3,rad=0", zorder=20,
        )
    )
    fig.text(
        (start[0] + end[0]) / 2 + 0.020,
        (start[1] + end[1]) / 2,
        label,
        ha="left", va="center", fontsize=8.2, weight="bold", color=BLUE,
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "none", "alpha": 0.96},
        zorder=21,
    )


def marker_legend(fig: plt.Figure) -> None:
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=CYAN,
               markeredgecolor="#061B24", markersize=6.8, label="NICHE CD8"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=CAF,
               markeredgecolor="none", markersize=6.0, label="CAF"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=MAC,
               markeredgecolor="none", markersize=6.0, label="MACROPHAGE"),
    ]
    fig.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(0.058, 0.877), ncol=3,
        frameon=False, fontsize=6.8, columnspacing=0.72, handletextpad=0.30,
        prop={"weight": "bold", "size": 6.8},
    )


def main() -> None:
    required = [RETURNED, RAW_CELLS, DIRECT_KM, RETURN_KM]
    required += [PLANES / f"{marker}.raw" for marker in ("DAPI", "CD8", "FAP", "CD68")]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(missing))

    OUT.mkdir(parents=True, exist_ok=True)
    image = composite()
    cd8, architecture = load_spatial_cells()

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.0})
    fig = plt.figure(figsize=(6.5, 6.5), facecolor="white")
    fig.text(0.018, 0.982, "A", ha="left", va="top", fontsize=23, weight="bold", color=INK)
    fig.text(
        0.095, 0.965, "SPATIAL NICHE → MOLECULAR STATE",
        ha="left", va="top", fontsize=13.2, weight="bold", color=INK,
    )

    top_map = (0.055, 0.665, 0.585, 0.225)
    top_km = (0.685, 0.665, 0.260, 0.225)
    mid = (0.055, 0.355, 0.890, 0.215)
    bottom_map = (0.055, 0.060, 0.585, 0.225)
    bottom_km = (0.685, 0.060, 0.260, 0.225)

    spatial_axis(fig, top_map, image, cd8, architecture, "niche")
    fig.text(0.066, 0.906, "CODEX: CAF/MACROPHAGE-RICH CD8 NICHE",
             ha="left", va="bottom", fontsize=9.1, weight="bold", color=INK)
    marker_legend(fig)
    survival_axis(fig, top_km, DIRECT_KM, "NICHE SCORE", "POORER SURVIVAL")

    cite_seq_panel(fig, mid)
    arrow(fig, (0.500, 0.650), (0.500, 0.585), "RAMP TRANSFER · 22 SHARED PROTEINS")

    spatial_axis(fig, bottom_map, image, cd8, architecture, "return")
    fig.text(0.066, 0.301, "CODEX: RETURNED TIGIT/CD39-HIGH CD8",
             ha="left", va="bottom", fontsize=9.1, weight="bold", color=INK)
    survival_axis(fig, bottom_km, RETURN_KM, "GREATER NICHE EXPOSURE", "POORER SURVIVAL")
    arrow(fig, (0.500, 0.340), (0.500, 0.296), "RAMP RETURN")

    fig.text(
        0.350, 0.040, "STATE-HIGH CD8 ENRICHED IN THE ORIGINAL NICHE",
        ha="center", va="center", fontsize=8.3, weight="bold", color=GREEN,
    )

    png = OUT / "panel_A_visual_workflow_v13.png"
    pdf = OUT / "panel_A_visual_workflow_v13.pdf"
    fig.savefig(png, dpi=320, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
