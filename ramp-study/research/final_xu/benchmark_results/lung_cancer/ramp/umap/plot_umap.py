#!/usr/bin/env python3
"""Generate the formal lung-cancer UMAPs from the frozen RAMP alignment.

This script never reruns RAMP.  It reads the row-locked matrices produced by
``../alignment/run_alignment.py``.  Cell-type labels and the display palette
are deliberately opened only after both UMAP fits have completed.
"""

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
ALIGNMENT_DIR = HERE.parent / "alignment"
ALIGNED_PATH = ALIGNMENT_DIR / "data" / "ramp_aligned_float64.npy"
BASELINE_PATH = ALIGNMENT_DIR / "data" / "kmeanslow_global_float64.npy"
METADATA_PATH = ALIGNMENT_DIR / "input" / "cell_metadata.csv.gz"
MARKERS_PATH = ALIGNMENT_DIR / "input" / "markers.txt"
ALIGNMENT_MANIFEST_PATH = ALIGNMENT_DIR / "alignment_manifest.json"
FINAL_XU = HERE.parents[3]
PALETTE_PATH = (
    FINAL_XU / "benchmark_data" / "lung_cancer" / "relabel" / "display_palette.csv"
)
ACCEPTED_COORDINATES_PATH = (
    FINAL_XU / "test0827" / "lung" / "alignment_10_donors" / "output"
    / "umap_coordinates_with_ver2.npz"
)
DATA_DIR = HERE / "data"
FIGURE_DIR = HERE / "figures"

cache_root = Path(tempfile.gettempdir()) / "final_xu_lung_cancer_ramp_umap_cache"
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
    "Tumor/Epithelial",
    "Mix",
    "CD8 T",
    "Endothelial",
    "Myeloid",
    "B-like",
    "CD4 T",
]
EXPECTED_TOTAL = 60_000
EXPECTED_DATASET_CELLS = {dataset: 20_000 for dataset in DATASETS}
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
EXPECTED_ALIGNED_SHA256 = "45b947761a6582a0e790b4eb0ce497aaa5bb21ecbdc24d9110614798f94ebed4"
EXPECTED_BASELINE_SHA256 = "1b5f6b9a571ae07b17259c863535dc6f5625d0959d5d5c47525475ae4d79ded6"
EXPECTED_ALIGNED_COORDINATE_BUFFER_SHA256 = (
    "035e8fc16c5931bc5c4d346e5cd43847fc0bd68788b3163514fadb9c1021fa23"
)
EXPECTED_BASELINE_COORDINATE_BUFFER_SHA256 = (
    "bf06622f9867b2dd0ae41f3eb9b07dcf2b961c431aed5ea9e5d2ccaba176ed6e"
)


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
        raise AssertionError("Fewer than two finite non-constant markers")
    standardized = StandardScaler().fit_transform(values[:, keep]).astype(np.float32)
    return standardized, keep


def fit_umaps(
    baseline: np.ndarray, aligned: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    baseline_z, baseline_keep = canonical(baseline)
    aligned_z, aligned_keep = canonical(aligned)
    if not np.array_equal(baseline_keep, aligned_keep):
        raise AssertionError("Baseline/aligned finite-marker sets differ")
    print(
        f"UMAP 1/2: KMEANSLOW baseline on {EXPECTED_TOTAL:,} cells x "
        f"{int(baseline_keep.sum())} markers",
        flush=True,
    )
    baseline_xy = umap.UMAP(**UMAP_PARAMETERS).fit_transform(baseline_z)
    print("UMAP 2/2: RAMP initialized from the baseline coordinates", flush=True)
    aligned_xy = umap.UMAP(
        **UMAP_PARAMETERS, init=np.asarray(baseline_xy).copy()
    ).fit_transform(aligned_z)
    return (
        np.asarray(baseline_xy, dtype=np.float32),
        np.asarray(aligned_xy, dtype=np.float32),
        aligned_keep,
    )


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
        row = palette_by_label.loc[label]
        selected = labels == label
        if not selected.any():
            continue
        axis.scatter(
            coordinates[selected, 0],
            coordinates[selected, 1],
            s=float(row["point_size"]),
            color=row["color"],
            alpha=float(row["alpha"]),
            linewidths=0,
            rasterized=True,
            zorder=zorder,
        )
    style_axis(axis)


def draw_datasets(
    axis: plt.Axes, coordinates: np.ndarray, datasets: np.ndarray
) -> None:
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


def save_figures(
    baseline_xy: np.ndarray,
    aligned_xy: np.ndarray,
    metadata: pd.DataFrame,
    palette: pd.DataFrame,
) -> dict[str, Path]:
    labels = metadata["celltype"].astype(str).to_numpy()
    datasets = metadata["dataset"].astype(str).to_numpy()
    legend_palette = palette.sort_values("legend_order")
    label_order = legend_palette["celltype"].astype(str).tolist()
    label_colors = dict(zip(legend_palette["celltype"], legend_palette["color"]))

    figure, axes = plt.subplots(1, 2, figsize=(12.2, 5.7), dpi=180)
    figure.patch.set_facecolor("white")
    draw_celltypes(axes[0], aligned_xy, labels, palette)
    draw_datasets(axes[1], aligned_xy, datasets)
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
        "Lung cancer — RAMP 2.3.0 alignment (8 shared markers)",
        fontsize=13, fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0.16, 1, 0.94), w_pad=0.04)
    aligned_png = FIGURE_DIR / "lung_cancer_ramp_umap.png"
    aligned_pdf = FIGURE_DIR / "lung_cancer_ramp_umap.pdf"
    figure.savefig(aligned_png, dpi=320, bbox_inches="tight")
    figure.savefig(aligned_pdf, bbox_inches="tight")
    plt.close(figure)

    methods = (
        ("KMEANSLOW (before RAMP)", baseline_xy),
        ("RAMP 2.3.0", aligned_xy),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12.8, 8.7), dpi=180)
    figure.patch.set_facecolor("white")
    for column, (title, coordinates) in enumerate(methods):
        draw_celltypes(axes[0, column], coordinates, labels, palette)
        draw_datasets(axes[1, column], coordinates, datasets)
        axes[0, column].set_title(title, fontsize=11.5, fontweight="bold")
    axes[0, 0].set_ylabel("Cell type (post hoc)", fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel("Dataset", fontsize=10, fontweight="bold")
    figure.legend(
        handles=legend_handles(label_order, label_colors),
        loc="lower center", bbox_to_anchor=(0.5, 0.052), ncol=7,
        frameon=False, fontsize=7.5, columnspacing=0.9, handletextpad=0.25,
    )
    figure.legend(
        handles=legend_handles(DATASETS, DATASET_COLORS),
        loc="lower center", bbox_to_anchor=(0.5, 0.012), ncol=3,
        frameon=False, fontsize=8.2,
    )
    figure.suptitle(
        "Lung cancer: Senosain + Cords + Monkman (20,000 cells each)",
        fontsize=13.2, fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0.145, 1, 0.955), w_pad=0.08, h_pad=0.10)
    comparison_png = FIGURE_DIR / "baseline_vs_ramp_umap.png"
    comparison_pdf = FIGURE_DIR / "baseline_vs_ramp_umap.pdf"
    figure.savefig(comparison_png, dpi=320, bbox_inches="tight")
    figure.savefig(comparison_pdf, bbox_inches="tight")
    plt.close(figure)
    return {
        "lung_cancer_ramp_umap.png": aligned_png,
        "lung_cancer_ramp_umap.pdf": aligned_pdf,
        "baseline_vs_ramp_umap.png": comparison_png,
        "baseline_vs_ramp_umap.pdf": comparison_pdf,
    }


def boolean_values(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype=bool)
    return series.astype(str).str.lower().isin({"true", "1", "1.0"}).to_numpy()


def diagnostics(
    values: np.ndarray, metadata: pd.DataFrame, method: str
) -> dict[str, object]:
    rng = np.random.default_rng(20260827)
    sampled: list[int] = []
    dataset_array = metadata["dataset"].astype(str).to_numpy()
    for dataset in DATASETS:
        candidates = np.flatnonzero(dataset_array == dataset)
        sampled.extend(rng.choice(candidates, 10_000, replace=False).tolist())
    sampled_array = np.sort(np.asarray(sampled, dtype=np.int64))
    z, _ = canonical(values[sampled_array])
    datasets = dataset_array[sampled_array]
    neighbors = NearestNeighbors(n_neighbors=31, n_jobs=1).fit(z).kneighbors(
        z, return_distance=False
    )[:, 1:]
    cross_dataset = float(np.mean(datasets[neighbors] != datasets[:, None]))

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
        "method": method,
        "metric_sample_cells": int(len(sampled_array)),
        "joint_metric_cells": int(joint.sum()),
        "cross_dataset_30nn_fraction": cross_dataset,
        "joint_same_celltype_30nn_purity": float(
            np.mean(labels_joint[joint_neighbors] == labels_joint[:, None])
        ),
        "joint_cross_dataset_30nn_fraction": float(
            np.mean(datasets_joint[joint_neighbors] != datasets_joint[:, None])
        ),
    }


def main() -> None:
    for path in (
        ALIGNED_PATH,
        BASELINE_PATH,
        METADATA_PATH,
        MARKERS_PATH,
        ALIGNMENT_MANIFEST_PATH,
        PALETTE_PATH,
        ACCEPTED_COORDINATES_PATH,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256(ALIGNED_PATH) != EXPECTED_ALIGNED_SHA256:
        raise AssertionError("Formal full-precision aligned input hash changed")
    if sha256(BASELINE_PATH) != EXPECTED_BASELINE_SHA256:
        raise AssertionError("Formal full-precision baseline input hash changed")
    alignment_manifest = json.loads(ALIGNMENT_MANIFEST_PATH.read_text())
    if alignment_manifest.get("labels_used_for_fitting") is not False:
        raise AssertionError("Alignment manifest does not establish label-free fitting")
    markers = [value.strip() for value in MARKERS_PATH.read_text().splitlines() if value.strip()]
    if markers != MARKERS:
        raise AssertionError("Lung marker contract changed")

    # Fit before opening any cell-type or palette metadata.
    aligned = np.load(ALIGNED_PATH, allow_pickle=False)
    baseline = np.load(BASELINE_PATH, allow_pickle=False)
    if aligned.shape != (EXPECTED_TOTAL, len(MARKERS)) or baseline.shape != aligned.shape:
        raise AssertionError("Lung alignment shape changed")
    if aligned.dtype != np.float64 or baseline.dtype != np.float64:
        raise AssertionError("Formal UMAP must use the frozen full-precision matrices")
    baseline_xy, aligned_xy, marker_keep = fit_umaps(baseline, aligned)
    if not np.isfinite(baseline_xy).all() or not np.isfinite(aligned_xy).all():
        raise AssertionError("UMAP produced non-finite coordinates")
    if buffer_sha256(baseline_xy) != EXPECTED_BASELINE_COORDINATE_BUFFER_SHA256:
        raise AssertionError("Baseline UMAP coordinates changed")
    if buffer_sha256(aligned_xy) != EXPECTED_ALIGNED_COORDINATE_BUFFER_SHA256:
        raise AssertionError("Aligned UMAP coordinates changed")

    accepted = np.load(ACCEPTED_COORDINATES_PATH, allow_pickle=False)
    expected_rows = np.arange(EXPECTED_TOTAL, dtype=np.int64)
    if not np.array_equal(accepted["display_row_ids"], expected_rows):
        raise AssertionError("Accepted UMAP row IDs changed")
    baseline_error = float(np.max(np.abs(accepted["baseline"] - baseline_xy)))
    aligned_error = float(np.max(np.abs(accepted["ver2"] - aligned_xy)))
    if baseline_error != 0.0 or aligned_error != 0.0:
        raise AssertionError(
            f"Formal/accepted UMAP mismatch: baseline={baseline_error}, aligned={aligned_error}"
        )

    # Labels are post-hoc and are loaded only after both UMAPs are frozen.
    metadata = pd.read_csv(METADATA_PATH, low_memory=False)
    palette = pd.read_csv(PALETTE_PATH, low_memory=False)
    required = [
        "row_id", "cell_id", "dataset", "donor", "unit_key", "celltype",
        "celltype_joint_evaluation", "joint_evaluation",
    ]
    missing = [column for column in required if column not in metadata]
    if missing:
        raise ValueError(f"Missing display columns: {missing}")
    if len(metadata) != EXPECTED_TOTAL or metadata["cell_id"].astype(str).duplicated().any():
        raise AssertionError("Lung display cell identities changed")
    if not np.array_equal(metadata["row_id"].to_numpy(np.int64), np.arange(EXPECTED_TOTAL)):
        raise AssertionError("Lung display row order changed")
    if metadata.groupby("dataset", sort=False).size().to_dict() != EXPECTED_DATASET_CELLS:
        raise AssertionError("Lung display dataset counts changed")
    expected_dataset_rows = np.concatenate(
        [np.repeat(dataset, 20_000) for dataset in DATASETS]
    )
    if not np.array_equal(metadata["dataset"].astype(str).to_numpy(), expected_dataset_rows):
        raise AssertionError("Lung display dataset blocks changed")
    labels = set(metadata["celltype"].astype(str))
    if labels != set(palette["celltype"].astype(str)):
        raise AssertionError("Lung display palette/label contract changed")
    draw_order = DRAW_ORDER_BOTTOM_TO_TOP
    if set(draw_order) != labels or len(draw_order) != len(labels):
        raise AssertionError("Lung result draw order does not cover every display label")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    coordinate_frame = metadata[required].copy()
    coordinate_frame["UMAP1"] = aligned_xy[:, 0]
    coordinate_frame["UMAP2"] = aligned_xy[:, 1]
    coordinate_frame["baseline_UMAP1"] = baseline_xy[:, 0]
    coordinate_frame["baseline_UMAP2"] = baseline_xy[:, 1]
    coordinate_csv = DATA_DIR / "umap_coordinates.csv.gz"
    aligned_npy = DATA_DIR / "umap_coordinates.npy"
    baseline_npy = DATA_DIR / "baseline_umap_coordinates.npy"
    archive_path = DATA_DIR / "matched_umap_coordinates.npz"
    write_csv_gz(coordinate_frame, coordinate_csv)
    np.save(aligned_npy, aligned_xy)
    np.save(baseline_npy, baseline_xy)
    np.savez_compressed(
        archive_path,
        row_id=metadata["row_id"].to_numpy(np.int64),
        kmeanslow=baseline_xy,
        ramp=aligned_xy,
    )

    figures = save_figures(baseline_xy, aligned_xy, metadata, palette)
    metrics = pd.DataFrame(
        [
            diagnostics(baseline, metadata, "KMEANSLOW"),
            diagnostics(aligned, metadata, "RAMP_2.3.0"),
        ]
    )
    metrics_path = DATA_DIR / "diagnostic_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    manifest = {
        "analysis": "lung_cancer_ramp_umap",
        "cells": EXPECTED_TOTAL,
        "markers_available": MARKERS,
        "markers_used": [
            marker for marker, selected in zip(MARKERS, marker_keep) if selected
        ],
        "input_precision": "frozen full-precision float64 RAMP outputs",
        "pre_umap_processing": (
            "separate pooled StandardScaler for baseline/aligned; no PCA"
        ),
        "labels_used_for_umap_fitting": False,
        "labels_loaded_after_umap_fitting": True,
        "umap_parameters": UMAP_PARAMETERS,
        "initialization": {
            "baseline": "UMAP default spectral initialization",
            "aligned": "frozen baseline coordinates",
        },
        "coordinate_buffer_sha256": {
            "baseline": buffer_sha256(baseline_xy),
            "aligned": buffer_sha256(aligned_xy),
        },
        "accepted_test_comparison": {
            "path": str(ACCEPTED_COORDINATES_PATH),
            "sha256": sha256(ACCEPTED_COORDINATES_PATH),
            "baseline_maximum_absolute_error": baseline_error,
            "aligned_maximum_absolute_error": aligned_error,
        },
        "drawing": {
            "celltype_palette_source": str(PALETTE_PATH),
            "celltype_draw_order_bottom_to_top": draw_order,
            "tumor_epithelial_drawn_first": True,
            "b_like_drawn_second_to_last": True,
            "cd4_t_drawn_last": True,
            "draw_order_overrides_palette_order": True,
            "dataset_points_randomly_interleaved_seed": SEED,
            "dataset_colors": DATASET_COLORS,
        },
        "diagnostics": {
            "labels_used_posthoc_only": True,
            "balanced_sample_cells_per_dataset": 10_000,
            "neighbors": 30,
        },
        "sources": {
            "aligned": {"path": str(ALIGNED_PATH), "sha256": sha256(ALIGNED_PATH)},
            "baseline": {"path": str(BASELINE_PATH), "sha256": sha256(BASELINE_PATH)},
            "metadata": {"path": str(METADATA_PATH), "sha256": sha256(METADATA_PATH)},
            "markers": {"path": str(MARKERS_PATH), "sha256": sha256(MARKERS_PATH)},
            "palette": {"path": str(PALETTE_PATH), "sha256": sha256(PALETTE_PATH)},
            "alignment_manifest": {
                "path": str(ALIGNMENT_MANIFEST_PATH),
                "sha256": sha256(ALIGNMENT_MANIFEST_PATH),
            },
        },
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
        "outputs": {
            "umap_coordinates.csv.gz": sha256(coordinate_csv),
            "umap_coordinates.npy": sha256(aligned_npy),
            "baseline_umap_coordinates.npy": sha256(baseline_npy),
            "matched_umap_coordinates.npz": sha256(archive_path),
            "diagnostic_metrics.csv": sha256(metrics_path),
            **{name: sha256(path) for name, path in figures.items()},
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
    (HERE / "umap_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(metrics.to_string(index=False), flush=True)
    print(f"Saved formal lung-cancer UMAP package under {HERE}", flush=True)


if __name__ == "__main__":
    main()
