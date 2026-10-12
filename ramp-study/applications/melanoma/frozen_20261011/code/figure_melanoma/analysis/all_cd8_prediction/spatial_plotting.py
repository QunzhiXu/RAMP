#!/usr/bin/env python3
"""Shared IMC loading and plotting helpers for the formal melanoma analysis."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
FIGURE_ROOT = HERE.parents[1]
FINAL_XU = FIGURE_ROOT.parent
RAW = FINAL_XU / "raw_data/melanoma/moldoveanu_imc"
DATA = RAW / "Moldoveanu_2022_CyTOF_melanoma/Data"
TIFF_ROOT = RAW / "CP_output_tiff"
RAW_TABLE = DATA / "ICI_meanIntensity.tsv.gz"
SAMPLE_INFO = DATA / "ST1_sample_info.txt"

os.environ.setdefault("MPLCONFIGDIR", str(FIGURE_ROOT / ".mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(FIGURE_ROOT / ".xdg-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_channel(path: Path, upper: float, gamma: float) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(image, dtype=np.float32)
    finite = values[np.isfinite(values)]
    positive = finite[finite > 0]
    reference = positive if len(positive) else finite
    lo, hi = np.percentile(reference, [0.5, upper])
    scaled = np.clip((values - lo) / max(hi - lo, 1e-8), 0, 1)
    return scaled**gamma


def make_imc_composite(roi: Path) -> np.ndarray:
    """Render nuclei gray, melanoma magenta, and CD8 cyan."""
    nuclei = read_channel(roi / "Histone-H3_Yb176.tiff", 99.8, 0.68)
    sox10 = read_channel(roi / "SOX10_Dy164.tiff", 99.5, 0.68)
    s100 = read_channel(roi / "S100_Lu175.tiff", 99.5, 0.68)
    cd8 = read_channel(roi / "CD8a_Dy162.tiff", 99.5, 0.58)
    melanoma = np.maximum(sox10, s100)
    base = 0.27 * nuclei
    rgb = np.stack(
        [
            base + 0.94 * melanoma,
            base + 0.88 * cd8,
            base + 0.77 * melanoma + 0.98 * cd8,
        ],
        axis=-1,
    )
    return np.clip(rgb, 0, 1)


def style_spatial_axis(
    ax: plt.Axes,
    image: np.ndarray,
    bounds: tuple[int, int, int, int],
    dim: float,
    grayscale: bool = False,
) -> None:
    x0, y0, x1, y1 = bounds
    shown_image = image
    if grayscale:
        luminance = (
            0.299 * image[..., 0]
            + 0.587 * image[..., 1]
            + 0.114 * image[..., 2]
        )
        shown_image = np.repeat(luminance[..., None], 3, axis=2)
    ax.imshow(shown_image * dim, origin="upper", interpolation="nearest")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set(xticks=[], yticks=[])
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_scale_bar(
    ax: plt.Axes,
    bounds: tuple[int, int, int, int],
    length_um: int,
) -> None:
    x0, y0, x1, y1 = bounds
    margin_x = 0.045 * (x1 - x0)
    margin_y = 0.055 * (y1 - y0)
    bar_x0 = x0 + margin_x
    bar_y = y1 - margin_y
    ax.plot(
        [bar_x0, bar_x0 + length_um],
        [bar_y, bar_y],
        color="white",
        lw=3.2,
        zorder=20,
    )
    ax.text(
        bar_x0 + length_um / 2,
        bar_y - 0.018 * (y1 - y0),
        f"{length_um} µm",
        color="white",
        fontsize=8.2,
        fontweight="bold",
        ha="center",
        va="bottom",
        zorder=20,
    )


def inside_bounds(
    xy: np.ndarray,
    bounds: tuple[int, int, int, int],
) -> np.ndarray:
    x0, y0, x1, y1 = bounds
    return (
        (xy[:, 0] >= x0)
        & (xy[:, 0] <= x1)
        & (xy[:, 1] >= y0)
        & (xy[:, 1] <= y1)
    )
