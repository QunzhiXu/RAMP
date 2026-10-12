#!/usr/bin/env python3
"""Read the saved BM RAMP alignment and generate post-hoc UMAP plots."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import platform
import tempfile
from importlib.metadata import version
from pathlib import Path

HERE = Path(__file__).resolve().parent
ALIGNMENT_DIR = HERE.parent / "alignment"
ALIGNED_PATH = ALIGNMENT_DIR / "data" / "ramp_aligned.csv.gz"
MARKERS_PATH = ALIGNMENT_DIR / "input" / "markers.txt"
DATA_DIR = HERE / "data"
FIGURE_DIR = HERE / "figures"

cache_root = Path(tempfile.gettempdir()) / "final_xu_bm_ramp_umap_cache"
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_root / "numba"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import umap


LABEL_ORDER = [
    "Mature / CODEX B",
    "Pre / immature B",
    "Plasma cell",
    "CD4 T",
    "CD8 T",
    "Other T / NKT",
    "NK",
    "Monocyte",
    "Macrophage",
    "Dendritic",
    "Other myeloid",
    "Promyelocytic progenitors",
    "HSPC / progenitor",
    "CODEX erythroid",
    "Triana erythroid progenitor",
    "Megakaryocytic",
    "Endothelial",
    "Stromal / VSMC",
    "Unclassified / technical",
]
LABEL_COLORS = {
    "Mature / CODEX B": "#D81B60",
    "Pre / immature B": "#F06292",
    "Plasma cell": "#8E24AA",
    "CD4 T": "#1769AA",
    "CD8 T": "#00A6D6",
    "Other T / NKT": "#6A5ACD",
    "NK": "#00A087",
    "Monocyte": "#F39C12",
    "Macrophage": "#B45F06",
    "Dendritic": "#7CB342",
    "Other myeloid": "#A66E3F",
    "Promyelocytic progenitors": "#FF7043",
    "HSPC / progenitor": "#7E57C2",
    "CODEX erythroid": "#C2185B",
    "Triana erythroid progenitor": "#EC407A",
    "Megakaryocytic": "#8D6E63",
    "Endothelial": "#43A047",
    "Stromal / VSMC": "#00897B",
    "Unclassified / technical": "#697386",
}
DATASET_ORDER = ["BM CODEX", "BM CyTOF", "Triana AbSeq"]
DATASET_COLORS = {
    "BM CODEX": "#3E70B2",
    "BM CyTOF": "#55A868",
    "Triana AbSeq": "#E28B35",
}
UMAP_PARAMETERS = {
    "n_neighbors": 30,
    "min_dist": 0.15,
    "metric": "euclidean",
    "random_state": 42,
    "transform_seed": 42,
    "n_jobs": 1,
    "low_memory": True,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv_gz(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(filename=str(path), mode="wb", mtime=0) as stream:
        frame.to_csv(stream, index=False)


def draw(
    axis: plt.Axes,
    xy: np.ndarray,
    values: np.ndarray,
    order: list[str],
    colors: dict[str, str],
    technical_label: str | None = None,
) -> None:
    for value in order:
        selected = values == value
        if not np.any(selected):
            continue
        technical = value == technical_label
        axis.scatter(
            xy[selected, 0],
            xy[selected, 1],
            s=0.9 if technical else 1.15,
            c=colors[value],
            alpha=0.18 if technical else 0.60,
            linewidths=0,
            rasterized=True,
        )
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xlabel("UMAP 1", fontsize=9)
    axis.set_ylabel("UMAP 2", fontsize=9)
    for spine in axis.spines.values():
        spine.set_visible(False)


def legend_handles(order: list[str], colors: dict[str, str]) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=colors[value],
            markeredgecolor="none",
            markersize=5,
            label=value,
        )
        for value in order
    ]


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(ALIGNED_PATH, low_memory=False)
    markers = [line.strip() for line in MARKERS_PATH.read_text().splitlines() if line.strip()]
    required = ["cell_id", "dataset", "donor", "unit_key", "celltype", *markers]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Missing aligned columns: {missing}")
    if len(frame) != 88_031 or frame["cell_id"].astype(str).duplicated().any():
        raise AssertionError("Aligned cell identities changed")
    if list(pd.unique(frame["dataset"].astype(str))) != DATASET_ORDER:
        raise AssertionError("Dataset order changed")

    aligned = frame[markers].to_numpy(np.float64)
    keep = np.isfinite(aligned).all(axis=0) & (np.nanstd(aligned, axis=0) > 1e-10)
    if keep.sum() < 2 or not np.isfinite(aligned[:, keep]).all():
        raise AssertionError("Invalid aligned matrix for UMAP")
    standardized = StandardScaler().fit_transform(aligned[:, keep]).astype(np.float32)
    initialization = np.random.default_rng(42).normal(
        0.0, 1e-4, size=(len(frame), 2)
    ).astype(np.float32)
    print(f"Fitting UMAP on {len(frame):,} cells x {int(keep.sum())} markers", flush=True)
    coordinates = umap.UMAP(
        **UMAP_PARAMETERS,
        init=initialization,
    ).fit_transform(standardized)
    if coordinates.shape != (len(frame), 2) or not np.isfinite(coordinates).all():
        raise AssertionError("Invalid UMAP coordinates")

    coordinate_frame = frame[
        ["cell_id", "dataset", "donor", "unit_key", "celltype"]
    ].copy()
    coordinate_frame["UMAP1"] = coordinates[:, 0]
    coordinate_frame["UMAP2"] = coordinates[:, 1]
    coordinate_path = DATA_DIR / "umap_coordinates.csv.gz"
    write_csv_gz(coordinate_frame, coordinate_path)
    np.save(DATA_DIR / "umap_coordinates.npy", coordinates.astype(np.float32))

    labels = frame["celltype"].fillna("Unclassified / technical").astype(str).to_numpy()
    datasets = frame["dataset"].astype(str).to_numpy()
    active_labels = [value for value in LABEL_ORDER if np.any(labels == value)]
    draw_labels = ["Unclassified / technical"] + [
        value for value in active_labels if value != "Unclassified / technical"
    ]

    figure, axes = plt.subplots(1, 2, figsize=(17.0, 7.7), dpi=180)
    draw(
        axes[0],
        coordinates,
        labels,
        draw_labels,
        LABEL_COLORS,
        technical_label="Unclassified / technical",
    )
    axes[0].set_title("Post-hoc merged cell type", fontsize=14, fontweight="bold")
    draw(axes[1], coordinates, datasets, DATASET_ORDER, DATASET_COLORS)
    axes[1].set_title("Dataset", fontsize=14, fontweight="bold")
    figure.suptitle(
        "Bone marrow — RAMP 2.3.0 alignment (14 shared markers)",
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )
    figure.legend(
        handles=legend_handles(active_labels, LABEL_COLORS),
        loc="lower center",
        bbox_to_anchor=(0.39, 0.012),
        ncol=5,
        frameon=False,
        fontsize=7.2,
    )
    figure.legend(
        handles=legend_handles(DATASET_ORDER, DATASET_COLORS),
        loc="lower center",
        bbox_to_anchor=(0.80, 0.032),
        ncol=3,
        frameon=False,
        fontsize=8.5,
    )
    figure.subplots_adjust(left=0.035, right=0.995, top=0.92, bottom=0.19, wspace=0.04)
    png_path = FIGURE_DIR / "bone_marrow_ramp_umap.png"
    pdf_path = FIGURE_DIR / "bone_marrow_ramp_umap.pdf"
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)

    manifest = {
        "analysis": "bone_marrow_ramp_umap",
        "cells": int(len(frame)),
        "aligned_markers_available": markers,
        "aligned_markers_used": [marker for marker, selected in zip(markers, keep) if selected],
        "pre_umap_processing": "pooled StandardScaler; no PCA",
        "labels_used_for_umap_fitting": False,
        "umap_parameters": UMAP_PARAMETERS,
        "initialization": "seed-42 Gaussian N(0, 1e-4), matching the BM story",
        "source": {"path": str(ALIGNED_PATH), "sha256": sha256(ALIGNED_PATH)},
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
        "outputs": {
            "umap_coordinates.csv.gz": sha256(coordinate_path),
            "umap_coordinates.npy": sha256(DATA_DIR / "umap_coordinates.npy"),
            "bone_marrow_ramp_umap.png": sha256(png_path),
            "bone_marrow_ramp_umap.pdf": sha256(pdf_path),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "scikit-learn": version("scikit-learn"),
            "umap-learn": version("umap-learn"),
            "matplotlib": version("matplotlib"),
        },
    }
    (HERE / "umap_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"Saved UMAP figure to {png_path}", flush=True)


if __name__ == "__main__":
    main()
