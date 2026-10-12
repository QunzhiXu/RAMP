#!/usr/bin/env python3
"""Plot lung cyCombine with an independent, method-neutral spectral UMAP."""

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
RELABEL_PATH = FINAL_XU / "benchmark_data" / "lung_cancer" / "relabel" / "relabeled_metadata.csv.gz"
PALETTE_PATH = FINAL_XU / "benchmark_data" / "lung_cancer" / "relabel" / "display_palette.csv"
DATA_DIR = HERE / "data"
FIGURE_DIR = HERE / "figures"

cache_root = Path(tempfile.gettempdir()) / "final_xu_lung_cycombine_umap_cache"
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
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import umap


DATASETS = ["Senosain CyTOF", "Cords IMC", "Monkman CODEX"]
DATASET_COLORS = {
    "Senosain CyTOF": "#1F77B4",
    "Cords IMC": "#8C564B",
    "Monkman CODEX": "#9EDAE5",
}
MARKERS = ["CD3", "CD4", "CD8", "CD31", "HLA_DR", "KI67", "VIMENTIN", "EPI"]
DRAW_ORDER_BOTTOM_TO_TOP = [
    "Tumor/Epithelial", "Mix", "CD8 T", "Endothelial", "Myeloid", "B-like", "CD4 T",
]
EXPECTED_TOTAL = 60_000
UMAP_PARAMETERS = {
    "n_neighbors": 30,
    "min_dist": 0.12,
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
        with gzip.GzipFile(
            filename="", fileobj=raw, mode="wb", compresslevel=6, mtime=0
        ) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
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
    palette_by_label = palette.set_index("celltype", drop=False)
    for zorder, label in enumerate(DRAW_ORDER_BOTTOM_TO_TOP, start=1):
        selected = labels == label
        if not selected.any():
            continue
        row = palette_by_label.loc[label]
        axis.scatter(
            coordinates[selected, 0], coordinates[selected, 1],
            s=float(row["point_size"]), color=row["color"], alpha=float(row["alpha"]),
            linewidths=0, rasterized=True, zorder=zorder,
        )
    style_axis(axis)


def draw_datasets(axis: plt.Axes, coordinates: np.ndarray, datasets: np.ndarray) -> None:
    permutation = np.random.default_rng(SEED).permutation(len(datasets))
    colors = np.asarray([DATASET_COLORS[value] for value in datasets], dtype=object)
    axis.scatter(
        coordinates[permutation, 0], coordinates[permutation, 1],
        s=0.62, c=colors[permutation], alpha=0.68, linewidths=0, rasterized=True,
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


def boolean_values(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype=bool)
    return series.astype(str).str.lower().isin({"true", "1", "1.0"}).to_numpy()


def diagnostics(values: np.ndarray, metadata: pd.DataFrame) -> dict[str, object]:
    rng = np.random.default_rng(20260827)
    dataset_array = metadata["dataset"].astype(str).to_numpy()
    sampled: list[int] = []
    for dataset in DATASETS:
        candidates = np.flatnonzero(dataset_array == dataset)
        sampled.extend(rng.choice(candidates, 10_000, replace=False).tolist())
    sampled_array = np.sort(np.asarray(sampled, dtype=np.int64))
    z = StandardScaler().fit_transform(values[sampled_array])
    datasets = dataset_array[sampled_array]
    neighbors = NearestNeighbors(n_neighbors=31, n_jobs=1).fit(z).kneighbors(
        z, return_distance=False
    )[:, 1:]
    joint = boolean_values(metadata.iloc[sampled_array]["joint_evaluation"])
    z_joint = z[joint]
    labels_joint = (
        metadata.iloc[sampled_array].loc[joint, "celltype_joint_evaluation"]
        .astype(str).to_numpy()
    )
    datasets_joint = datasets[joint]
    joint_neighbors = NearestNeighbors(n_neighbors=31, n_jobs=1).fit(z_joint).kneighbors(
        z_joint, return_distance=False
    )[:, 1:]
    return {
        "method": "cyCombine v0.2.5",
        "metric_sample_cells": int(len(sampled_array)),
        "joint_metric_cells": int(joint.sum()),
        "cross_dataset_30nn_fraction": float(np.mean(datasets[neighbors] != datasets[:, None])),
        "joint_same_celltype_30nn_purity": float(
            np.mean(labels_joint[joint_neighbors] == labels_joint[:, None])
        ),
        "joint_cross_dataset_30nn_fraction": float(
            np.mean(datasets_joint[joint_neighbors] != datasets_joint[:, None])
        ),
    }


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for path in (ALIGNMENT_PATH, INPUT_PATH, MARKERS_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)
    markers = [line for line in MARKERS_PATH.read_text().splitlines() if line]
    if markers != MARKERS:
        raise AssertionError("Frozen eight-marker contract changed")
    corrected = pd.read_csv(ALIGNMENT_PATH, dtype={"cell_id": str}, low_memory=False)
    inputs = pd.read_csv(
        INPUT_PATH,
        dtype={"cell_id": str, "dataset": str, "donor": str, "unit_key": str},
        low_memory=False,
    )
    if len(corrected) != EXPECTED_TOTAL or not corrected["cell_id"].is_unique:
        raise AssertionError("Corrected cell identity contract changed")
    if not corrected["cell_id"].equals(inputs["cell_id"]):
        raise AssertionError("Alignment output order differs from method input")
    if list(pd.unique(inputs["dataset"])) != DATASETS:
        raise AssertionError("Dataset block order changed")
    matrix = corrected[MARKERS].to_numpy(np.float64)
    keep = np.isfinite(matrix).all(axis=0) & (np.nanstd(matrix, axis=0) > 1e-10)
    if not np.all(keep):
        raise AssertionError(f"Corrected UMAP marker set changed: keep={keep.tolist()}")
    standardized = StandardScaler().fit_transform(matrix).astype(np.float32)
    print(
        f"Fitting independent spectral UMAP on {len(matrix):,} cells x "
        f"{len(MARKERS)} markers",
        flush=True,
    )
    coordinates = umap.UMAP(**UMAP_PARAMETERS, init="spectral").fit_transform(standardized)
    coordinates = np.asarray(coordinates, dtype=np.float32)
    if coordinates.shape != (EXPECTED_TOTAL, 2) or not np.isfinite(coordinates).all():
        raise AssertionError("UMAP returned invalid coordinates")

    # Post-hoc labels are deliberately opened only after the embedding is fixed.
    labels = pd.read_csv(RELABEL_PATH, dtype={"cell_id": str}, low_memory=False)
    palette = pd.read_csv(PALETTE_PATH, low_memory=False)
    if not labels["cell_id"].equals(inputs["cell_id"]):
        raise AssertionError("Post-hoc relabel order differs from alignment output")
    if not labels["dataset"].astype(str).equals(inputs["dataset"].astype(str)):
        raise AssertionError("Post-hoc relabel dataset identity differs")
    if set(palette["celltype"].astype(str)) != set(DRAW_ORDER_BOTTOM_TO_TOP):
        raise AssertionError("Display palette vocabulary changed")

    coordinate_frame = inputs[["cell_id", "dataset", "donor", "unit_key"]].copy()
    coordinate_frame["celltype"] = labels["celltype"].astype(str).to_numpy()
    coordinate_frame["UMAP1"] = coordinates[:, 0]
    coordinate_frame["UMAP2"] = coordinates[:, 1]
    coordinate_path = DATA_DIR / "umap_coordinates.csv.gz"
    write_csv_gz(coordinate_frame, coordinate_path)
    coordinate_npy = DATA_DIR / "umap_coordinates.npy"
    np.save(coordinate_npy, coordinates, allow_pickle=False)

    diagnostic = diagnostics(matrix, labels)
    diagnostic_path = DATA_DIR / "diagnostic_metrics.csv"
    pd.DataFrame([diagnostic]).to_csv(diagnostic_path, index=False)

    label_values = coordinate_frame["celltype"].to_numpy()
    datasets = coordinate_frame["dataset"].astype(str).to_numpy()
    legend_palette = palette.sort_values("legend_order")
    label_order = legend_palette["celltype"].astype(str).tolist()
    label_colors = dict(zip(legend_palette["celltype"], legend_palette["color"]))
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 5.7), dpi=180)
    figure.patch.set_facecolor("white")
    draw_celltypes(axes[0], coordinates, label_values, palette)
    draw_datasets(axes[1], coordinates, datasets)
    axes[0].set_title("Cell type (post hoc)", fontsize=11, fontweight="bold")
    axes[1].set_title("Dataset", fontsize=11, fontweight="bold")
    figure.legend(
        handles=legend_handles(label_order, label_colors),
        loc="lower center", bbox_to_anchor=(0.5, 0.063), ncol=7,
        frameon=False, fontsize=7.5, columnspacing=0.9, handletextpad=0.25,
    )
    figure.legend(
        handles=legend_handles(DATASETS, DATASET_COLORS),
        loc="lower center", bbox_to_anchor=(0.5, 0.014), ncol=3,
        frameon=False, fontsize=8,
    )
    figure.suptitle(
        "Lung cancer — cyCombine v0.2.5 alignment (8 shared markers)",
        fontsize=13, fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0.16, 1, 0.94), w_pad=0.04)
    png_path = FIGURE_DIR / "lung_cancer_cycombine_umap.png"
    pdf_path = FIGURE_DIR / "lung_cancer_cycombine_umap.pdf"
    figure.savefig(png_path, dpi=320, bbox_inches="tight")
    figure.savefig(
        pdf_path,
        bbox_inches="tight",
        metadata={
            "Creator": "final_xu lung-cancer cyCombine independent spectral UMAP",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)

    manifest = {
        "schema_version": "final_xu.lung_cancer.cycombine_umap.v2",
        "status": "complete",
        "analysis": "lung_cancer_cycombine_independent_spectral_umap",
        "cells": EXPECTED_TOTAL,
        "markers_available": MARKERS,
        "markers_used": MARKERS,
        "pre_umap_processing": "pooled StandardScaler; no PCA",
        "labels_used_for_alignment": False,
        "labels_used_for_umap_fitting": False,
        "labels_loaded_after_umap_fitting": True,
        "umap_parameters": UMAP_PARAMETERS,
        "initialization": {
            "kind": "spectral",
            "implementation": "umap-learn",
            "computed_from": "this method's pooled-standardized corrected marker coordinates",
            "external_coordinates_used": False,
            "random_state": UMAP_PARAMETERS["random_state"],
        },
        "coordinate_buffer_sha256": buffer_sha256(coordinates),
        "diagnostics": diagnostic,
        "sources": {
            "corrected_expression": {"path": str(ALIGNMENT_PATH), "sha256": sha256(ALIGNMENT_PATH)},
            "method_input": {"path": str(INPUT_PATH), "sha256": sha256(INPUT_PATH)},
            "posthoc_relabel": {"path": str(RELABEL_PATH), "sha256": sha256(RELABEL_PATH)},
            "palette": {"path": str(PALETTE_PATH), "sha256": sha256(PALETTE_PATH)},
        },
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
        "outputs": {
            coordinate_path.name: sha256(coordinate_path),
            coordinate_npy.name: sha256(coordinate_npy),
            diagnostic_path.name: sha256(diagnostic_path),
            png_path.name: sha256(png_path),
            pdf_path.name: sha256(pdf_path),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "scikit_learn": version("scikit-learn"),
            "umap_learn": version("umap-learn"),
            "matplotlib": version("matplotlib"),
        },
    }
    manifest_path = HERE / "umap_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Saved independent cyCombine UMAP to {png_path}", flush=True)


if __name__ == "__main__":
    main()
