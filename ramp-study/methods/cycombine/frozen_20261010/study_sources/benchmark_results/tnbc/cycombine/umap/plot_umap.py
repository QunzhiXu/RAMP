#!/usr/bin/env python3
"""Plot TNBC cyCombine output with an independent spectral UMAP fit."""

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
METHOD_DIR = HERE.parent
FINAL_XU = METHOD_DIR.parents[2]
ALIGNMENT_PATH = METHOD_DIR / "alignment" / "data" / "corrected_expression.csv.gz"
INPUT_PATH = METHOD_DIR / "preprocessing" / "data" / "combined_analysis_ready.csv.gz"
MARKERS_PATH = METHOD_DIR / "preprocessing" / "markers.txt"
RELABEL_PATH = FINAL_XU / "benchmark_data" / "tnbc" / "relabel" / "relabeled_metadata.csv.gz"
SELECTED_PATH = FINAL_XU / "benchmark_data" / "tnbc" / "selected_cells.csv.gz"
DATA_DIR = HERE / "data"
FIGURE_DIR = HERE / "figures"

cache_root = Path(tempfile.gettempdir()) / "final_xu_tnbc_cycombine_umap_cache"
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


EXPECTED_CELLS = 62_701
MARKERS = [
    "CD11C", "CD20", "CD3", "CD31", "CD4", "CD68",
    "CD8", "HLA_DR", "KI67", "PANCK", "SMA",
]
DATASET_ORDER = ["Tietscher IMC", "Meyer IMC", "Shiao CODEX"]
EXPECTED_COUNTS = {
    "Tietscher IMC": 20_000,
    "Meyer IMC": 22_701,
    "Shiao CODEX": 20_000,
}
LABEL_ORDER = [
    "Non-joint", "B lineage", "CD4 T", "CD8 T", "Endothelial",
    "Myeloid", "NK", "Stroma", "Tumor / epithelial",
]
LABEL_DRAW_ORDER = [
    "Tumor / epithelial", "Non-joint", "B lineage", "CD4 T", "CD8 T",
    "Endothelial", "Myeloid", "NK", "Stroma",
]
LABEL_COLORS = {
    "Non-joint": "#1F77B4",
    "B lineage": "#F2C14E",
    "CD4 T": "#17BECF",
    "CD8 T": "#1F77B4",
    "Endothelial": "#59A14F",
    "Myeloid": "#E17C05",
    "NK": "#00A087",
    "Stroma": "#9467BD",
    "Tumor / epithelial": "#D62728",
}
DATASET_COLORS = {
    "Tietscher IMC": "#2F6FB0",
    "Meyer IMC": "#8C564B",
    "Shiao CODEX": "#9EDAE5",
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


def buffer_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values, dtype=np.float32)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def write_csv_gz(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
            frame.to_csv(stream, index=False, float_format="%.9g")


def draw(axis, xy, values, order, colors, muted_label=None):
    for value in order:
        selected = values == value
        if not np.any(selected):
            continue
        axis.scatter(
            xy[selected, 0], xy[selected, 1], s=0.70,
            c=colors[value], alpha=0.20 if value == muted_label else 0.72,
            linewidths=0, rasterized=True,
        )
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xlabel("UMAP 1", fontsize=9)
    axis.set_ylabel("UMAP 2", fontsize=9)
    axis.set_aspect("equal", adjustable="box")
    for spine in axis.spines.values():
        spine.set_visible(False)


def legend_handles(order, colors):
    return [
        Line2D(
            [0], [0], marker="o", linestyle="", markerfacecolor=colors[value],
            markeredgecolor="none", markersize=5, label=value,
        )
        for value in order
    ]


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for path in (
        ALIGNMENT_PATH, INPUT_PATH, MARKERS_PATH, RELABEL_PATH,
        SELECTED_PATH,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    markers = [line.strip() for line in MARKERS_PATH.read_text().splitlines() if line.strip()]
    if markers != MARKERS:
        raise AssertionError("The frozen eleven-marker TNBC contract changed")
    corrected = pd.read_csv(ALIGNMENT_PATH, dtype={"cell_id": str}, low_memory=False)
    inputs = pd.read_csv(INPUT_PATH, dtype={"cell_id": str}, low_memory=False)
    selected = pd.read_csv(
        SELECTED_PATH,
        usecols=["cell_id", "dataset", "donor", "unit_key"],
        dtype={"cell_id": str, "dataset": str, "donor": str, "unit_key": str},
        low_memory=False,
    )
    if len(corrected) != EXPECTED_CELLS or corrected["cell_id"].duplicated().any():
        raise AssertionError("Corrected cell identities changed")
    if not corrected["cell_id"].equals(inputs["cell_id"]):
        raise AssertionError("Alignment output order differs from method input")
    if not inputs[["cell_id", "dataset", "donor", "unit_key"]].equals(selected):
        raise AssertionError("Method input order differs from frozen selected cells")
    if list(pd.unique(inputs["dataset"].astype(str))) != DATASET_ORDER:
        raise AssertionError("Dataset order changed")
    if inputs["dataset"].value_counts(sort=False).to_dict() != EXPECTED_COUNTS:
        raise AssertionError("Dataset cell counts changed")

    # Fit UMAP before opening the post-hoc relabel table.
    matrix = corrected[markers].to_numpy(np.float64)
    keep = np.isfinite(matrix).all(axis=0) & (np.nanstd(matrix, axis=0) > 1e-10)
    if keep.sum() < 2 or not np.isfinite(matrix[:, keep]).all():
        raise AssertionError("Invalid corrected matrix for UMAP")
    standardized = StandardScaler().fit_transform(matrix[:, keep]).astype(np.float32)
    print(f"Fitting UMAP on {len(matrix):,} cells x {int(keep.sum())} markers", flush=True)
    coordinates = umap.UMAP(
        **UMAP_PARAMETERS,
        init="spectral",
    ).fit_transform(standardized)
    coordinates = np.asarray(coordinates, dtype=np.float32)
    if coordinates.shape != (EXPECTED_CELLS, 2) or not np.isfinite(coordinates).all():
        raise AssertionError("Invalid UMAP coordinates")

    # Labels are post-hoc display metadata and are loaded only after fitting.
    labels = pd.read_csv(RELABEL_PATH, dtype={"cell_id": str}, low_memory=False)
    if not labels["cell_id"].equals(inputs["cell_id"]):
        raise AssertionError("Post-hoc relabel order differs from alignment output")
    if not labels["dataset"].astype(str).equals(inputs["dataset"].astype(str)):
        raise AssertionError("Post-hoc relabel dataset identity differs")
    label_values = labels["celltype"].fillna("Non-joint").astype(str).to_numpy()
    unknown = sorted(set(label_values) - set(LABEL_ORDER))
    if unknown:
        raise AssertionError(f"Unknown relabeled cell types: {unknown}")

    coordinate_frame = inputs[["cell_id", "dataset", "donor", "unit_key"]].copy()
    coordinate_frame["celltype"] = label_values
    coordinate_frame["UMAP1"] = coordinates[:, 0]
    coordinate_frame["UMAP2"] = coordinates[:, 1]
    coordinate_path = DATA_DIR / "umap_coordinates.csv.gz"
    write_csv_gz(coordinate_frame, coordinate_path)
    coordinate_npy_path = DATA_DIR / "umap_coordinates.npy"
    np.save(coordinate_npy_path, coordinates)

    datasets = inputs["dataset"].astype(str).to_numpy()
    figure, axes = plt.subplots(1, 2, figsize=(16.0, 7.4), dpi=180)
    draw(axes[0], coordinates, label_values, LABEL_DRAW_ORDER, LABEL_COLORS, "Non-joint")
    axes[0].set_title("Post-hoc broad cell type", fontsize=14, fontweight="bold")
    draw(axes[1], coordinates, datasets, DATASET_ORDER, DATASET_COLORS)
    axes[1].set_title("Dataset", fontsize=14, fontweight="bold")
    figure.suptitle(
        "TNBC — cyCombine v0.2.5 alignment (11 shared markers)",
        fontsize=16, fontweight="bold", y=0.985,
    )
    figure.legend(
        handles=legend_handles(LABEL_ORDER, LABEL_COLORS),
        loc="lower center", bbox_to_anchor=(0.36, 0.010), ncol=5,
        frameon=False, fontsize=7.5,
    )
    figure.legend(
        handles=legend_handles(DATASET_ORDER, DATASET_COLORS),
        loc="lower center", bbox_to_anchor=(0.80, 0.025), ncol=3,
        frameon=False, fontsize=8.5,
    )
    figure.subplots_adjust(left=0.035, right=0.995, top=0.915, bottom=0.18, wspace=0.035)
    png_path = FIGURE_DIR / "tnbc_cycombine_umap.png"
    pdf_path = FIGURE_DIR / "tnbc_cycombine_umap.pdf"
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)

    manifest = {
        "analysis": "tnbc_cycombine_umap",
        "cells": EXPECTED_CELLS,
        "aligned_markers_available": markers,
        "aligned_markers_used": [marker for marker, chosen in zip(markers, keep) if chosen],
        "pre_umap_processing": "pooled StandardScaler; no PCA",
        "labels_used_for_alignment": False,
        "labels_used_for_umap_fitting": False,
        "labels_loaded_after_umap_fitting": True,
        "umap_parameters": UMAP_PARAMETERS,
        "initialization": {
            "kind": "spectral",
            "method_neutral": True,
            "random_state": UMAP_PARAMETERS["random_state"],
        },
        "coordinate_buffer_sha256_float32": buffer_sha256(coordinates),
        "drawing": {
            "point_size": 0.70,
            "standard_alpha": 0.72,
            "non_joint_alpha": 0.20,
            "celltype_draw_order_bottom_to_top": LABEL_DRAW_ORDER,
            "legend_order": LABEL_ORDER,
        },
        "sources": {
            "corrected_expression": {"path": str(ALIGNMENT_PATH), "sha256": sha256(ALIGNMENT_PATH)},
            "method_input": {"path": str(INPUT_PATH), "sha256": sha256(INPUT_PATH)},
            "posthoc_relabel": {"path": str(RELABEL_PATH), "sha256": sha256(RELABEL_PATH)},
            "frozen_cell_selection": {"path": str(SELECTED_PATH), "sha256": sha256(SELECTED_PATH)},
        },
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
        "outputs": {
            coordinate_path.name: sha256(coordinate_path),
            coordinate_npy_path.name: sha256(coordinate_npy_path),
            png_path.name: sha256(png_path),
            pdf_path.name: sha256(pdf_path),
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
    (HERE / "umap_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Saved UMAP figure to {png_path}", flush=True)


if __name__ == "__main__":
    main()
