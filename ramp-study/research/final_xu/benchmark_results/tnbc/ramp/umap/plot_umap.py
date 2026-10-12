#!/usr/bin/env python3
"""Read the frozen TNBC RAMP alignment and generate post-hoc UMAP plots."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import platform
import tempfile
from importlib.metadata import version
from pathlib import Path


HERE = Path(__file__).resolve().parent
ALIGNMENT_DIR = HERE.parent / "alignment"
ALIGNED_PATH = ALIGNMENT_DIR / "data" / "ramp_aligned_float64.npy"
BASELINE_PATH = ALIGNMENT_DIR / "data" / "kmeanslow_global_float64.npy"
METADATA_PATH = ALIGNMENT_DIR / "input" / "cell_metadata.csv.gz"
MARKERS_PATH = ALIGNMENT_DIR / "input" / "markers.txt"
ALIGNMENT_MANIFEST_PATH = ALIGNMENT_DIR / "alignment_manifest.json"
DATA_DIR = HERE / "data"
FIGURE_DIR = HERE / "figures"

cache_root = Path(tempfile.gettempdir()) / "final_xu_tnbc_ramp_umap_cache"
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
    "CD11C", "CD20", "CD3", "CD31", "CD4", "CD68", "CD8",
    "HLA_DR", "KI67", "PANCK", "SMA",
]
DATASET_ORDER = ["Tietscher IMC", "Meyer IMC", "Shiao CODEX"]
EXPECTED_DATASET_CELLS = {
    "Tietscher IMC": 20_000,
    "Meyer IMC": 22_701,
    "Shiao CODEX": 20_000,
}
LABEL_ORDER = [
    "Non-joint",
    "B lineage",
    "CD4 T",
    "CD8 T",
    "Endothelial",
    "Myeloid",
    "NK",
    "Stroma",
    "Tumor / epithelial",
]
LABEL_DRAW_ORDER = [
    "Tumor / epithelial",
    *[value for value in LABEL_ORDER if value != "Tumor / epithelial"],
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
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="", fileobj=raw_handle, mode="wb", compresslevel=6, mtime=0
        ) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                frame.to_csv(text, index=False, float_format="%.9g")


def canonical(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    keep = np.isfinite(values).all(axis=0) & (np.nanstd(values, axis=0) > 1e-10)
    if int(keep.sum()) < 2:
        raise ValueError("Fewer than two finite, non-constant marker dimensions")
    standardized = StandardScaler().fit_transform(values[:, keep]).astype(np.float32)
    return standardized, keep


def draw(
    axis: plt.Axes,
    coordinates: np.ndarray,
    values: np.ndarray,
    order: list[str],
    colors: dict[str, str],
    muted_label: str | None = None,
) -> None:
    for value in order:
        selected = values == value
        if not np.any(selected):
            continue
        axis.scatter(
            coordinates[selected, 0],
            coordinates[selected, 1],
            s=0.70,
            c=colors[value],
            alpha=0.20 if value == muted_label else 0.72,
            linewidths=0,
            rasterized=True,
        )
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xlabel("UMAP 1", fontsize=9)
    axis.set_ylabel("UMAP 2", fontsize=9)
    axis.set_aspect("equal", adjustable="box")
    for spine in axis.spines.values():
        spine.set_visible(False)


def legend_handles(order: list[str], colors: dict[str, str]) -> list[Line2D]:
    return [
        Line2D(
            [0], [0], marker="o", linestyle="", markerfacecolor=colors[value],
            markeredgecolor="none", markersize=5, label=value,
        )
        for value in order
    ]


def main() -> None:
    for path in (
        ALIGNED_PATH,
        BASELINE_PATH,
        METADATA_PATH,
        MARKERS_PATH,
        ALIGNMENT_MANIFEST_PATH,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    alignment_manifest = json.loads(ALIGNMENT_MANIFEST_PATH.read_text())
    if alignment_manifest.get("labels_used_for_fitting") is not False:
        raise AssertionError("Alignment manifest does not establish label-free fitting")
    if alignment_manifest.get("cells") != EXPECTED_CELLS:
        raise AssertionError("Alignment manifest does not match TNBC benchmark v2")
    if "coverage_guided_seed20260960" not in alignment_manifest.get("family", ""):
        raise AssertionError("Alignment manifest does not identify the Meyer 0960 set")
    markers = [line.strip() for line in MARKERS_PATH.read_text().splitlines() if line.strip()]
    if markers != MARKERS:
        raise AssertionError("TNBC marker contract changed")

    # Fit coordinates before opening the post-hoc label/display metadata.
    aligned = np.load(ALIGNED_PATH, allow_pickle=False)
    baseline = np.load(BASELINE_PATH, allow_pickle=False)
    if aligned.shape != (EXPECTED_CELLS, len(MARKERS)) or baseline.shape != aligned.shape:
        raise AssertionError("TNBC alignment shape changed")
    aligned_z, aligned_keep = canonical(aligned)
    baseline_z, baseline_keep = canonical(baseline)
    if not np.array_equal(aligned_keep, baseline_keep):
        raise AssertionError("Aligned/baseline finite marker sets differ")
    print(
        f"Fitting matched UMAPs on {EXPECTED_CELLS:,} cells x "
        f"{int(aligned_keep.sum())} markers",
        flush=True,
    )
    baseline_coordinates = umap.UMAP(**UMAP_PARAMETERS).fit_transform(baseline_z)
    coordinates = umap.UMAP(
        **UMAP_PARAMETERS,
        init=baseline_coordinates.copy(),
    ).fit_transform(aligned_z)
    baseline_coordinates = np.asarray(baseline_coordinates, dtype=np.float32)
    coordinates = np.asarray(coordinates, dtype=np.float32)
    if coordinates.shape != (EXPECTED_CELLS, 2) or not np.isfinite(coordinates).all():
        raise AssertionError("Invalid aligned UMAP coordinates")
    if not np.isfinite(baseline_coordinates).all():
        raise AssertionError("Invalid baseline UMAP coordinates")
    # Labels and datasets are loaded strictly after both UMAP fits have ended.
    metadata = pd.read_csv(METADATA_PATH, low_memory=False)
    required = ["cell_id", "dataset", "donor", "unit_key", "celltype"]
    missing = [column for column in required if column not in metadata]
    if missing:
        raise ValueError(f"Missing display columns: {missing}")
    if len(metadata) != EXPECTED_CELLS or metadata["cell_id"].astype(str).duplicated().any():
        raise AssertionError("TNBC display cell identities changed")
    if list(pd.unique(metadata["dataset"].astype(str))) != DATASET_ORDER:
        raise AssertionError("TNBC dataset order changed")
    observed_counts = metadata.groupby("dataset", sort=False).size().to_dict()
    if observed_counts != EXPECTED_DATASET_CELLS:
        raise AssertionError(f"TNBC dataset cell counts changed: {observed_counts}")
    labels = metadata["celltype"].fillna("Non-joint").astype(str).to_numpy()
    datasets = metadata["dataset"].astype(str).to_numpy()
    if set(np.unique(labels)) != set(LABEL_ORDER):
        raise AssertionError(f"TNBC display labels changed: {sorted(np.unique(labels))}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    coordinate_path = DATA_DIR / "umap_coordinates.csv.gz"
    coordinate_frame = metadata[required].copy()
    coordinate_frame["UMAP1"] = coordinates[:, 0]
    coordinate_frame["UMAP2"] = coordinates[:, 1]
    write_csv_gz(coordinate_frame, coordinate_path)
    coordinate_npy_path = DATA_DIR / "umap_coordinates.npy"
    baseline_npy_path = DATA_DIR / "baseline_umap_coordinates.npy"
    np.save(coordinate_npy_path, coordinates)
    np.save(baseline_npy_path, baseline_coordinates)

    figure, axes = plt.subplots(1, 2, figsize=(16.0, 7.4), dpi=180)
    draw(
        axes[0], coordinates, labels, LABEL_DRAW_ORDER, LABEL_COLORS,
        muted_label="Non-joint",
    )
    axes[0].set_title("Post-hoc broad cell type", fontsize=14, fontweight="bold")
    draw(axes[1], coordinates, datasets, DATASET_ORDER, DATASET_COLORS)
    axes[1].set_title("Dataset", fontsize=14, fontweight="bold")
    figure.suptitle(
        "TNBC — RAMP 2.3.0 alignment (11 shared markers)",
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )
    figure.legend(
        handles=legend_handles(LABEL_ORDER, LABEL_COLORS),
        loc="lower center",
        bbox_to_anchor=(0.36, 0.010),
        ncol=5,
        frameon=False,
        fontsize=7.5,
    )
    figure.legend(
        handles=legend_handles(DATASET_ORDER, DATASET_COLORS),
        loc="lower center",
        bbox_to_anchor=(0.80, 0.025),
        ncol=3,
        frameon=False,
        fontsize=8.5,
    )
    figure.subplots_adjust(
        left=0.035, right=0.995, top=0.915, bottom=0.18, wspace=0.035
    )
    png_path = FIGURE_DIR / "tnbc_ramp_umap.png"
    pdf_path = FIGURE_DIR / "tnbc_ramp_umap.pdf"
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)

    manifest = {
        "analysis": "tnbc_ramp_umap",
        "cells": EXPECTED_CELLS,
        "markers_available": MARKERS,
        "markers_used": [
            marker for marker, selected in zip(MARKERS, aligned_keep) if selected
        ],
        "input_precision": "float64 canonical API outputs",
        "pre_umap_processing": "separate pooled StandardScaler for baseline/aligned; no PCA",
        "labels_used_for_umap_fitting": False,
        "labels_loaded_after_umap_fitting": True,
        "umap_parameters": UMAP_PARAMETERS,
        "initialization": {
            "baseline": "UMAP default spectral initialization",
            "aligned": "frozen baseline coordinates",
        },
        "drawing": {
            "point_size": 0.70,
            "standard_alpha": 0.72,
            "non_joint_alpha": 0.20,
            "celltype_draw_order_bottom_to_top": LABEL_DRAW_ORDER,
            "legend_order": LABEL_ORDER,
        },
        "coordinate_buffer_sha256": {
            "aligned": buffer_sha256(coordinates),
            "baseline": buffer_sha256(baseline_coordinates),
        },
        "sources": {
            "aligned": {"path": str(ALIGNED_PATH), "sha256": sha256(ALIGNED_PATH)},
            "baseline": {"path": str(BASELINE_PATH), "sha256": sha256(BASELINE_PATH)},
            "metadata": {"path": str(METADATA_PATH), "sha256": sha256(METADATA_PATH)},
            "markers": {"path": str(MARKERS_PATH), "sha256": sha256(MARKERS_PATH)},
            "alignment_manifest": {
                "path": str(ALIGNMENT_MANIFEST_PATH),
                "sha256": sha256(ALIGNMENT_MANIFEST_PATH),
            },
        },
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
        "outputs": {
            "umap_coordinates.csv.gz": sha256(coordinate_path),
            "umap_coordinates.npy": sha256(coordinate_npy_path),
            "baseline_umap_coordinates.npy": sha256(baseline_npy_path),
            "tnbc_ramp_umap.png": sha256(png_path),
            "tnbc_ramp_umap.pdf": sha256(pdf_path),
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
