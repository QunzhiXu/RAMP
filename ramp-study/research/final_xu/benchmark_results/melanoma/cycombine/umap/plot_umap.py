#!/usr/bin/env python3
"""Plot melanoma cyCombine output with an independent spectral UMAP init."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import platform
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path


HERE = Path(__file__).resolve().parent
METHOD_DIR = HERE.parent
FINAL_XU = METHOD_DIR.parents[2]
ALIGNMENT_PATH = METHOD_DIR / "alignment" / "data" / "corrected_expression.csv.gz"
INPUT_PATH = METHOD_DIR / "preprocessing" / "data" / "combined_analysis_ready.csv.gz"
MARKERS_PATH = METHOD_DIR / "preprocessing" / "markers.txt"
ALIGNMENT_AUDIT_PATH = METHOD_DIR / "alignment" / "audit" / "alignment_audit.json"
RELABEL_PATH = FINAL_XU / "benchmark_data" / "melanoma" / "relabel" / "relabeled_metadata.csv.gz"
PALETTE_PATH = FINAL_XU / "benchmark_data" / "melanoma" / "relabel" / "display_palette.csv"
DATA_DIR = HERE / "data"
FIGURE_DIR = HERE / "figures"

cache_root = Path(tempfile.gettempdir()) / "final_xu_melanoma_cycombine_umap_cache"
cache_root.mkdir(parents=True, exist_ok=True)
for variable, target in (
    ("MPLCONFIGDIR", cache_root / "matplotlib"),
    ("NUMBA_CACHE_DIR", cache_root / "numba"),
    ("XDG_CACHE_HOME", cache_root / "xdg"),
    ("JOBLIB_TEMP_FOLDER", cache_root / "joblib"),
):
    target.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(variable, str(target))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import umap


DATASETS = [
    "MEL_Hickey2023_CODEX",
    "MEL_Hoch2021_IMC",
    "MEL_Ptacek2021_MIBI",
]
DATASET_DISPLAY = {
    DATASETS[0]: "Hickey CODEX",
    DATASETS[1]: "Hoch IMC",
    DATASETS[2]: "Ptacek MIBI",
}
DATASET_COLORS = {
    DATASETS[0]: "#2F6FB0",
    DATASETS[1]: "#E58A2B",
    DATASETS[2]: "#38A169",
}
MARKERS = [
    "CD163", "CD3", "CD31", "CD68", "CD8", "HLA_DR",
    "LAG3", "PD1", "PDL1", "SMA", "SOX10", "VIMENTIN",
]
EXPECTED_TOTAL = 60_000
EXPECTED_DATASET_CELLS = {dataset: 20_000 for dataset in DATASETS}
UMAP_PARAMETERS = {
    "n_neighbors": 30,
    "min_dist": 0.15,
    "metric": "euclidean",
    "random_state": 42,
    "transform_seed": 42,
    "n_jobs": 1,
    "low_memory": True,
}
SEED = 42


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def buffer_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def write_csv_gz(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=6, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
                frame.to_csv(text, index=False, float_format="%.9g")


def style_axis(axis: plt.Axes) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_aspect("equal", adjustable="datalim")
    for spine in axis.spines.values():
        spine.set_visible(False)


def draw_celltypes(
    axis: plt.Axes,
    coordinates: np.ndarray,
    labels: np.ndarray,
    palette: pd.DataFrame,
) -> None:
    ordered = palette.sort_values("draw_order_bottom_to_top")
    for row in ordered.itertuples(index=False):
        selected = labels == row.celltype
        if not selected.any():
            continue
        axis.scatter(
            coordinates[selected, 0],
            coordinates[selected, 1],
            s=float(row.point_size),
            color=row.color,
            alpha=float(row.alpha),
            linewidths=0,
            rasterized=True,
            zorder=int(row.zorder),
        )
    style_axis(axis)


def draw_datasets(axis: plt.Axes, coordinates: np.ndarray, datasets: np.ndarray) -> None:
    permutation = np.random.default_rng(SEED).permutation(len(datasets))
    colors = np.asarray([DATASET_COLORS[value] for value in datasets], dtype=object)
    axis.scatter(
        coordinates[permutation, 0],
        coordinates[permutation, 1],
        s=0.62,
        c=colors[permutation],
        alpha=0.68,
        linewidths=0,
        rasterized=True,
    )
    style_axis(axis)


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
        ALIGNMENT_PATH, INPUT_PATH, MARKERS_PATH, ALIGNMENT_AUDIT_PATH,
        RELABEL_PATH, PALETTE_PATH,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    markers = [line.strip() for line in MARKERS_PATH.read_text().splitlines() if line.strip()]
    if markers != MARKERS:
        raise AssertionError("Frozen melanoma marker contract changed")
    alignment_audit = json.loads(ALIGNMENT_AUDIT_PATH.read_text())
    if alignment_audit.get("status") != "pass":
        raise AssertionError("Canonical alignment audit has not passed")

    # Fit the UMAP before opening any cell-type label or palette file.
    corrected = pd.read_csv(ALIGNMENT_PATH, dtype={"cell_id": str}, low_memory=False)
    inputs = pd.read_csv(
        INPUT_PATH,
        dtype={"cell_id": str, "dataset": str, "donor": str, "unit_key": str},
        low_memory=False,
    )
    if len(corrected) != EXPECTED_TOTAL or corrected["cell_id"].duplicated().any():
        raise AssertionError("Corrected cell identities changed")
    if not corrected["cell_id"].equals(inputs["cell_id"]):
        raise AssertionError("Alignment output order differs from method input")
    if list(pd.unique(inputs["dataset"])) != DATASETS:
        raise AssertionError("Dataset block order changed")
    if inputs.groupby("dataset", sort=False).size().astype(int).to_dict() != EXPECTED_DATASET_CELLS:
        raise AssertionError("Dataset cell counts changed")

    matrix = corrected[markers].to_numpy(np.float64)
    keep = np.isfinite(matrix).all(axis=0) & (np.nanstd(matrix, axis=0) > 1e-10)
    if int(keep.sum()) < 2 or not np.isfinite(matrix[:, keep]).all():
        raise AssertionError("Invalid corrected matrix for UMAP")
    standardized = StandardScaler().fit_transform(matrix[:, keep]).astype(np.float32)
    print(
        f"Fitting matched UMAP on {len(matrix):,} cells x {int(keep.sum())} markers "
        "with method-neutral spectral initialization",
        flush=True,
    )
    coordinates = umap.UMAP(
        **UMAP_PARAMETERS,
        init="spectral",
    ).fit_transform(standardized)
    coordinates = np.asarray(coordinates, dtype=np.float32)
    if coordinates.shape != (EXPECTED_TOTAL, 2) or not np.isfinite(coordinates).all():
        raise AssertionError("UMAP produced invalid coordinates")

    # Labels and display palette are deliberately read only after fitting.
    labels = pd.read_csv(RELABEL_PATH, dtype={"cell_id": str, "dataset": str}, low_memory=False)
    palette = pd.read_csv(PALETTE_PATH, low_memory=False)
    if not labels["cell_id"].equals(inputs["cell_id"]):
        raise AssertionError("Post-hoc relabel order differs from method input")
    if not labels["dataset"].equals(inputs["dataset"]):
        raise AssertionError("Post-hoc relabel dataset identity differs")
    if not labels["labels_used_for_alignment"].astype(str).str.lower().eq("false").all():
        raise AssertionError("Relabel package does not certify post-hoc use")
    label_values = labels["celltype"].astype(str).to_numpy()
    if set(label_values) != set(palette["celltype"].astype(str)):
        raise AssertionError("Display palette and active labels differ")
    draw_order = palette.sort_values("draw_order_bottom_to_top")["celltype"].astype(str).tolist()
    if draw_order[0] != "Tumor" or draw_order[-1] != "Epithelial (non-joint)":
        raise AssertionError("Tumor/epithelial draw order changed")

    coordinate_frame = inputs[["cell_id", "dataset", "donor", "unit_key"]].copy()
    coordinate_frame["celltype"] = label_values
    coordinate_frame["UMAP1"] = coordinates[:, 0]
    coordinate_frame["UMAP2"] = coordinates[:, 1]
    coordinate_path = DATA_DIR / "umap_coordinates.csv.gz"
    write_csv_gz(coordinate_frame, coordinate_path)
    coordinate_npy_path = DATA_DIR / "umap_coordinates.npy"
    np.save(coordinate_npy_path, coordinates, allow_pickle=False)

    legend_palette = palette.sort_values("legend_order")
    label_order = legend_palette["celltype"].astype(str).tolist()
    label_colors = dict(zip(legend_palette["celltype"], legend_palette["color"]))
    datasets = inputs["dataset"].astype(str).to_numpy()
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 5.7), dpi=180)
    figure.patch.set_facecolor("white")
    draw_celltypes(axes[0], coordinates, label_values, palette)
    draw_datasets(axes[1], coordinates, datasets)
    axes[0].set_title("Cell type (post hoc)", fontsize=11, fontweight="bold")
    axes[1].set_title("Dataset", fontsize=11, fontweight="bold")
    figure.legend(
        handles=legend_handles(label_order, label_colors),
        loc="lower center", bbox_to_anchor=(0.5, 0.065), ncol=5,
        frameon=False, fontsize=7.6, columnspacing=1.0, handletextpad=0.25,
    )
    figure.legend(
        handles=legend_handles(DATASETS, DATASET_COLORS),
        labels=[DATASET_DISPLAY[value] for value in DATASETS],
        loc="lower center", bbox_to_anchor=(0.5, 0.016), ncol=3,
        frameon=False, fontsize=8,
    )
    figure.suptitle(
        "Melanoma - cyCombine v0.2.5 alignment (12 shared markers)",
        fontsize=13, fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0.17, 1, 0.94), w_pad=0.04)
    png_path = FIGURE_DIR / "melanoma_cycombine_umap.png"
    pdf_path = FIGURE_DIR / "melanoma_cycombine_umap.pdf"
    figure.savefig(png_path, dpi=320, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)

    manifest = {
        "schema_version": "final_xu.melanoma.cycombine_umap.v2",
        "analysis": "melanoma_cycombine_independent_umap",
        "cells": int(len(matrix)),
        "aligned_markers_available": markers,
        "aligned_markers_used": [marker for marker, selected in zip(markers, keep) if selected],
        "pre_umap_processing": "pooled StandardScaler; no PCA",
        "labels_used_for_alignment": False,
        "labels_used_for_umap_fitting": False,
        "labels_loaded_after_fitting_for_plotting_only": True,
        "umap_parameters": UMAP_PARAMETERS,
        "initialization": {
            "kind": "spectral",
            "description": "method-neutral umap-learn spectral initialization computed from this method's standardized corrected coordinates",
        },
        "sources": {
            "corrected_expression": {"path": str(ALIGNMENT_PATH), "sha256": sha256(ALIGNMENT_PATH)},
            "method_input": {"path": str(INPUT_PATH), "sha256": sha256(INPUT_PATH)},
            "alignment_audit": {"path": str(ALIGNMENT_AUDIT_PATH), "sha256": sha256(ALIGNMENT_AUDIT_PATH)},
            "posthoc_relabel": {"path": str(RELABEL_PATH), "sha256": sha256(RELABEL_PATH)},
            "palette": {"path": str(PALETTE_PATH), "sha256": sha256(PALETTE_PATH)},
        },
        "coordinate_buffer_sha256": buffer_sha256(coordinates),
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
    manifest_path = HERE / "umap_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Saved matched UMAP figure to {png_path}", flush=True)


if __name__ == "__main__":
    main()
