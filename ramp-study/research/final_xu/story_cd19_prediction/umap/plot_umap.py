#!/usr/bin/env python3
"""Draw matched RAMP 2.3.0 UMAPs without and with predicted CD19."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
STORY = HERE.parent
ALIGNMENT = STORY / "alignment" / "output"
DATA = HERE / "data"
FIGURES = HERE / "figures"
CACHE = HERE / ".cache"
for folder in [DATA, FIGURES, CACHE / "matplotlib", CACHE / "numba", CACHE / "xdg", CACHE / "joblib", CACHE / "tmp"]:
    folder.mkdir(parents=True, exist_ok=True)
for name, value in {
    "MPLCONFIGDIR": str(CACHE / "matplotlib"),
    "NUMBA_CACHE_DIR": str(CACHE / "numba"),
    "XDG_CACHE_HOME": str(CACHE / "xdg"),
    "JOBLIB_TEMP_FOLDER": str(CACHE / "joblib"),
    "TMPDIR": str(CACHE / "tmp"),
    "LOKY_MAX_CPU_COUNT": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}.items():
    os.environ.setdefault(name, value)
sys.dont_write_bytecode = True

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import umap


EXPECTED_CELLS = 88_031
SAMPLING_SEED = 20260820
UMAP_SEED = 42
DATASETS = ["BM CODEX", "BM CyTOF", "Triana AbSeq"]
ALL_CONDITIONS = [
    "without_cd19",
    "with_wrong_observed_cd19",
    "with_predicted_cd19",
]
MAIN_CONDITIONS = ["without_cd19", "with_predicted_cd19"]
CONDITION_TITLES = {
    "without_cd19": "14 shared markers\n(without CD19)",
    "with_wrong_observed_cd19": "14 shared markers + observed CD19\n(failed CODEX channel)",
    "with_predicted_cd19": "14 shared markers + predicted CD19",
}
DATASET_COLORS = {
    "BM CODEX": "#3E70B2",
    "BM CyTOF": "#55A868",
    "Triana AbSeq": "#E28B35",
}
LABEL_ORDER = [
    "Mature / CODEX B", "Pre / immature B", "Plasma cell",
    "CD4 T", "CD8 T", "Other T / NKT", "NK", "Monocyte",
    "Macrophage", "Dendritic", "Other myeloid",
    "Promyelocytic progenitors", "HSPC / progenitor",
    "CODEX erythroid", "Triana erythroid progenitor", "Megakaryocytic",
    "Endothelial", "Stromal / VSMC", "Unclassified / technical",
]
LABEL_COLORS = {
    "Mature / CODEX B": "#D81B60", "Pre / immature B": "#F06292",
    "Plasma cell": "#8E24AA", "CD4 T": "#1769AA", "CD8 T": "#00A6D6",
    "Other T / NKT": "#6A5ACD", "NK": "#00A087", "Monocyte": "#F39C12",
    "Macrophage": "#B45F06", "Dendritic": "#7CB342",
    "Other myeloid": "#A66E3F", "Promyelocytic progenitors": "#FF7043",
    "HSPC / progenitor": "#7E57C2", "CODEX erythroid": "#C2185B",
    "Triana erythroid progenitor": "#EC407A", "Megakaryocytic": "#8D6E63",
    "Endothelial": "#43A047", "Stromal / VSMC": "#00897B",
    "Unclassified / technical": "#697386",
}
B_LINEAGE = ["Mature / CODEX B", "Pre / immature B", "Plasma cell"]
UMAP_PROTOCOL = {
    "n_neighbors": 30,
    "min_dist": 0.15,
    "metric": "euclidean",
    "random_state": UMAP_SEED,
    "transform_seed": UMAP_SEED,
    "n_jobs": 1,
    "low_memory": True,
    "initialization": "one shared N(0,1e-4) float32 array generated with numpy default_rng(42); copied per fit",
    "pre_umap_scaling": "condition-wise pooled StandardScaler after removal of zero-variance columns; float32",
    "pca": False,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def stable_local_seed(*parts: object) -> int:
    payload = "|".join(map(str, (SAMPLING_SEED, *parts))).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") % (2**32 - 1)


def load_alignment(condition: str, metadata: pd.DataFrame) -> tuple[np.ndarray, list[str], Path]:
    manifest_path = ALIGNMENT / condition / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["algorithm_version"] != "2.3.0":
        raise AssertionError(f"{condition} is not RAMP 2.3.0")
    path = ALIGNMENT / condition / "ramp_aligned.csv.gz"
    frame = pd.read_csv(path, low_memory=False)
    identity = ["cell_id", "dataset", "donor", "unit_key"]
    if len(frame) != EXPECTED_CELLS:
        raise AssertionError(f"Unexpected rows in {path}: {len(frame)}")
    if not frame[identity].astype(str).equals(metadata[identity].astype(str)):
        raise AssertionError(f"Metadata/order mismatch for {condition}")
    markers = list(manifest["markers"])
    values = frame[markers].to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise AssertionError(f"Non-finite aligned values for {condition}")
    return values, markers, path


def canonical(values: np.ndarray) -> np.ndarray:
    keep = np.nanstd(values, axis=0) > 1e-10
    if not np.any(keep):
        raise AssertionError("No nonconstant UMAP features")
    return StandardScaler().fit_transform(values[:, keep]).astype(np.float32)


def fit_umap(values: np.ndarray, shared_init: np.ndarray) -> np.ndarray:
    return umap.UMAP(
        n_neighbors=UMAP_PROTOCOL["n_neighbors"],
        min_dist=UMAP_PROTOCOL["min_dist"],
        metric=UMAP_PROTOCOL["metric"],
        random_state=UMAP_PROTOCOL["random_state"],
        transform_seed=UMAP_PROTOCOL["transform_seed"],
        n_jobs=UMAP_PROTOCOL["n_jobs"],
        low_memory=UMAP_PROTOCOL["low_memory"],
        init=shared_init.copy(),
    ).fit_transform(canonical(values)).astype(np.float32)


def draw_points(axis: plt.Axes, xy: np.ndarray, metadata: pd.DataFrame, mode: str) -> None:
    if mode == "celltype":
        values = metadata["celltype"].astype(str).to_numpy()
        active = [label for label in LABEL_ORDER if np.any(values == label)]
        order = ["Unclassified / technical"] + [
            label for label in active if label != "Unclassified / technical"
        ]
        colors = LABEL_COLORS
    else:
        values = metadata["dataset"].astype(str).to_numpy()
        order = DATASETS
        colors = DATASET_COLORS
    for label in order:
        selected = values == label
        axis.scatter(
            xy[selected, 0], xy[selected, 1],
            s=0.48,
            color=colors[label],
            alpha=0.18 if label == "Unclassified / technical" else 0.68,
            linewidths=0,
            rasterized=True,
        )
    axis.set(xticks=[], yticks=[])
    axis.set_aspect("equal", adjustable="box")
    for spine in axis.spines.values():
        spine.set_visible(False)


def add_legends(fig: plt.Figure, metadata: pd.DataFrame) -> None:
    labels = metadata["celltype"].astype(str).to_numpy()
    active = [label for label in LABEL_ORDER if np.any(labels == label)]
    fig.legend(
        [Line2D([0], [0], marker="o", linestyle="", color=LABEL_COLORS[x], markersize=4.0) for x in active],
        active,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.038),
        ncol=7,
        frameon=False,
        fontsize=6.1,
        columnspacing=0.60,
        handletextpad=0.18,
    )
    fig.legend(
        [Line2D([0], [0], marker="o", linestyle="", color=DATASET_COLORS[x], markersize=4.4) for x in DATASETS],
        DATASETS,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.008),
        ncol=3,
        frameon=False,
        fontsize=7.2,
    )


def render_main(coordinates: dict[str, np.ndarray], metadata: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.3, 9.4), dpi=180)
    for column, condition in enumerate(MAIN_CONDITIONS):
        draw_points(axes[0, column], coordinates[condition], metadata, "celltype")
        draw_points(axes[1, column], coordinates[condition], metadata, "dataset")
        axes[0, column].set_title(CONDITION_TITLES[condition], fontsize=10.6, fontweight="bold", pad=7)
    axes[0, 0].set_ylabel("Provider-informed cell type", fontsize=9, fontweight="bold")
    axes[1, 0].set_ylabel("Dataset", fontsize=9, fontweight="bold")
    add_legends(fig, metadata)
    fig.suptitle("RAMP restoration of failed CODEX CD19", fontsize=14.5, fontweight="bold", x=0.025, y=0.992, ha="left")
    fig.text(
        0.985, 0.952,
        "RAMP 2.3.0 · same 88,031 cells · identical UMAP parameters and initialization",
        ha="right", va="top", fontsize=7.2, color="#555555",
    )
    fig.tight_layout(rect=(0.01, 0.105, 0.995, 0.915), w_pad=0.15, h_pad=0.20)
    fig.savefig(FIGURES / "cd19_prediction_umap.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / "cd19_prediction_umap.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_three_way(coordinates: dict[str, np.ndarray], metadata: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.45, 9.4), dpi=180)
    for column, condition in enumerate(ALL_CONDITIONS):
        draw_points(axes[0, column], coordinates[condition], metadata, "celltype")
        draw_points(axes[1, column], coordinates[condition], metadata, "dataset")
        axes[0, column].set_title(CONDITION_TITLES[condition], fontsize=10.4, fontweight="bold", pad=7)
    axes[0, 0].set_ylabel("Provider-informed cell type", fontsize=9, fontweight="bold")
    axes[1, 0].set_ylabel("Dataset", fontsize=9, fontweight="bold")
    add_legends(fig, metadata)
    fig.suptitle("Observed versus predicted CD19 in bone marrow", fontsize=14.5, fontweight="bold", x=0.02, y=0.992, ha="left")
    fig.text(
        0.985, 0.952,
        "RAMP 2.3.0 · same 88,031 cells · identical UMAP parameters and initialization",
        ha="right", va="top", fontsize=7.2, color="#555555",
    )
    fig.tight_layout(rect=(0.01, 0.105, 0.995, 0.915), w_pad=0.12, h_pad=0.20)
    fig.savefig(FIGURES / "cd19_three_way_umap.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / "cd19_three_way_umap.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_b_focus(coordinates: dict[str, np.ndarray], metadata: pd.DataFrame) -> None:
    labels = metadata["celltype"].astype(str).to_numpy()
    focus_order = ["Mature / CODEX B", "Pre / immature B", "Plasma cell", "NK"]
    fig, axes = plt.subplots(1, 2, figsize=(10.3, 4.8), dpi=180)
    for axis, condition in zip(axes, MAIN_CONDITIONS):
        xy = coordinates[condition]
        axis.scatter(xy[:, 0], xy[:, 1], s=0.35, color="#D9DDE3", alpha=0.16, linewidths=0, rasterized=True)
        for label in focus_order:
            selected = labels == label
            axis.scatter(xy[selected, 0], xy[selected, 1], s=0.7, color=LABEL_COLORS[label], alpha=0.76, linewidths=0, rasterized=True)
        axis.set_title(CONDITION_TITLES[condition], fontsize=10.2, fontweight="bold")
        axis.set(xticks=[], yticks=[])
        axis.set_aspect("equal", adjustable="box")
        for spine in axis.spines.values():
            spine.set_visible(False)
    fig.legend(
        [Line2D([0], [0], marker="o", linestyle="", color=LABEL_COLORS[x], markersize=5) for x in focus_order],
        focus_order,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    fig.suptitle("B-lineage and NK focus", x=0.025, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0.01, 0.09, 0.995, 0.93), w_pad=0.2)
    fig.savefig(FIGURES / "cd19_prediction_b_lineage_focus.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / "cd19_prediction_b_lineage_focus.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_three_way_b_focus(coordinates: dict[str, np.ndarray], metadata: pd.DataFrame) -> None:
    labels = metadata["celltype"].astype(str).to_numpy()
    focus_order = ["Mature / CODEX B", "Pre / immature B", "Plasma cell", "NK"]
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.8), dpi=180)
    for axis, condition in zip(axes, ALL_CONDITIONS):
        xy = coordinates[condition]
        axis.scatter(xy[:, 0], xy[:, 1], s=0.35, color="#D9DDE3", alpha=0.16, linewidths=0, rasterized=True)
        for label in focus_order:
            selected = labels == label
            axis.scatter(xy[selected, 0], xy[selected, 1], s=0.7, color=LABEL_COLORS[label], alpha=0.76, linewidths=0, rasterized=True)
        axis.set_title(CONDITION_TITLES[condition], fontsize=10.0, fontweight="bold")
        axis.set(xticks=[], yticks=[])
        axis.set_aspect("equal", adjustable="box")
        for spine in axis.spines.values():
            spine.set_visible(False)
    fig.legend(
        [Line2D([0], [0], marker="o", linestyle="", color=LABEL_COLORS[x], markersize=5) for x in focus_order],
        focus_order,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    fig.suptitle("B-lineage and NK focus", x=0.02, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0.01, 0.09, 0.995, 0.93), w_pad=0.15)
    fig.savefig(FIGURES / "cd19_three_way_b_lineage_focus.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / "cd19_three_way_b_lineage_focus.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def choose_rows(mask: np.ndarray, maximum: int, namespace: str) -> np.ndarray:
    rows = np.flatnonzero(mask)
    if len(rows) > maximum:
        rng = np.random.default_rng(stable_local_seed("metric_rows", namespace))
        rows = np.sort(rng.choice(rows, maximum, replace=False))
    return rows


def choose_balanced_rows(labels: np.ndarray, groups: list[list[str]], maximum_per_group: int, namespace: str) -> np.ndarray:
    pieces = [
        choose_rows(np.isin(labels, values), maximum_per_group, f"{namespace}_{index}")
        for index, values in enumerate(groups)
    ]
    return np.sort(np.concatenate(pieces))


def safe_silhouette(values: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=str)
    return float(silhouette_score(values, labels, metric="euclidean"))


def nonself_neighbors(values: np.ndarray, k: int = 30) -> np.ndarray:
    _, candidates = NearestNeighbors(n_neighbors=min(k + 1, len(values)), n_jobs=1).fit(values).kneighbors(values)
    answer = np.empty((len(values), min(k, len(values) - 1)), dtype=int)
    for row in range(len(values)):
        local = candidates[row][candidates[row] != row]
        answer[row] = local[: answer.shape[1]]
    return answer


def neighbor_fraction(values: np.ndarray, labels: np.ndarray, relation: str) -> float:
    neighbors = nonself_neighbors(values, 30)
    labels = np.asarray(labels, dtype=str)
    if relation == "different":
        return float(np.mean(labels[neighbors] != labels[:, None]))
    return float(np.mean(labels[neighbors] == labels[:, None]))


def metric_rows(metadata: pd.DataFrame) -> dict[str, np.ndarray]:
    labels = metadata["celltype"].astype(str).to_numpy()
    return {
        "global": choose_rows(np.ones(len(metadata), dtype=bool), 5_000, "global"),
        "b_vs_nonb": choose_balanced_rows(labels, [B_LINEAGE, [x for x in LABEL_ORDER if x not in B_LINEAGE]], 2_500, "b_vs_nonb"),
        "b_subtypes": choose_balanced_rows(labels, [["Mature / CODEX B"], ["Pre / immature B"], ["Plasma cell"]], 1_700, "b_subtypes"),
        "mature_b_vs_nk": choose_balanced_rows(labels, [["Mature / CODEX B"], ["NK"]], 2_500, "mature_nk"),
        "pre_b_vs_plasma": choose_balanced_rows(labels, [["Pre / immature B"], ["Plasma cell"]], 2_500, "pre_plasma"),
    }


def evaluate(condition: str, aligned: np.ndarray, xy: np.ndarray, metadata: pd.DataFrame, rows: dict[str, np.ndarray]) -> dict[str, Any]:
    labels = metadata["celltype"].astype(str).to_numpy()
    datasets = metadata["dataset"].astype(str).to_numpy()
    highdim = canonical(aligned)
    global_rows = rows["global"]
    answer: dict[str, Any] = {
        "condition": condition,
        "n_cells": int(len(aligned)),
        "n_features": int(aligned.shape[1]),
        "celltype_silhouette_umap": safe_silhouette(xy[global_rows], labels[global_rows]),
        "dataset_silhouette_abs_umap": abs(safe_silhouette(xy[global_rows], datasets[global_rows])),
        "cross_dataset_30nn_umap": neighbor_fraction(xy[global_rows], datasets[global_rows], "different"),
        "same_celltype_30nn_umap": neighbor_fraction(xy[global_rows], labels[global_rows], "same"),
    }
    pairs = {
        "b_vs_nonb": (rows["b_vs_nonb"], np.isin(labels, B_LINEAGE).astype(str)),
        "b_lineage_subtype": (rows["b_subtypes"], labels),
        "matureB_vs_NK": (rows["mature_b_vs_nk"], labels),
        "preB_vs_plasma": (rows["pre_b_vs_plasma"], labels),
    }
    for name, (selected, metric_labels) in pairs.items():
        answer[f"{name}_silhouette_umap"] = safe_silhouette(xy[selected], metric_labels[selected])
        answer[f"{name}_silhouette_highdim"] = safe_silhouette(highdim[selected], metric_labels[selected])
    selected = rows["mature_b_vs_nk"]
    answer["matureB_NK_opposite_30nn_umap"] = neighbor_fraction(xy[selected], labels[selected], "different")
    answer["matureB_NK_opposite_30nn_highdim"] = neighbor_fraction(highdim[selected], labels[selected], "different")
    return answer


def main() -> None:
    reuse_coordinates = sys.argv[1:] == ["--reuse-coordinates"]
    if sys.argv[1:] and not reuse_coordinates:
        raise SystemExit("Usage: plot_umap.py [--reuse-coordinates]")
    alignment_manifest_path = ALIGNMENT / "manifest.json"
    alignment_manifest = json.loads(alignment_manifest_path.read_text())
    if alignment_manifest["algorithm"]["version"] != "2.3.0":
        raise AssertionError("Alignment is not canonical RAMP 2.3.0")
    metadata_path = ALIGNMENT / "cell_metadata.csv.gz"
    metadata = pd.read_csv(metadata_path, low_memory=False)
    if len(metadata) != EXPECTED_CELLS:
        raise AssertionError("Unexpected metadata row count")

    aligned: dict[str, np.ndarray] = {}
    markers: dict[str, list[str]] = {}
    input_paths: dict[str, Path] = {}
    for condition in ALL_CONDITIONS:
        aligned[condition], markers[condition], input_paths[condition] = load_alignment(condition, metadata)

    shared_init = np.random.default_rng(UMAP_SEED).normal(
        0.0, 1e-4, size=(EXPECTED_CELLS, 2)
    ).astype(np.float32)
    init_path = DATA / "shared_umap_initialization.npy"
    np.save(init_path, shared_init)
    coordinates_path = DATA / "umap_coordinates.npz"
    coordinates: dict[str, np.ndarray] = {}
    if reuse_coordinates:
        saved = np.load(coordinates_path, allow_pickle=False)
        if set(saved.files) != set(ALL_CONDITIONS):
            raise AssertionError(f"Saved coordinate keys differ: {saved.files}")
        coordinates = {
            condition: saved[condition].astype(np.float32)
            for condition in ALL_CONDITIONS
        }
        print("Reuse validated saved UMAP coordinates", flush=True)
    else:
        for condition in ALL_CONDITIONS:
            print(f"Fit UMAP: {condition}", flush=True)
            coordinates[condition] = fit_umap(aligned[condition], shared_init)
        np.savez_compressed(coordinates_path, **coordinates)
    for condition in ALL_CONDITIONS:
        if coordinates[condition].shape != (EXPECTED_CELLS, 2) or not np.isfinite(coordinates[condition]).all():
            raise AssertionError(f"Invalid UMAP coordinates for {condition}")

    rows = metric_rows(metadata)
    row_path = DATA / "metric_row_sets.npz"
    np.savez_compressed(row_path, **rows)
    metrics = pd.DataFrame([
        evaluate(condition, aligned[condition], coordinates[condition], metadata, rows)
        for condition in ALL_CONDITIONS
    ])
    metrics_path = DATA / "metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    render_main(coordinates, metadata)
    render_three_way(coordinates, metadata)
    render_b_focus(coordinates, metadata)
    render_three_way_b_focus(coordinates, metadata)

    generated = [init_path, coordinates_path, row_path, metrics_path, *sorted(FIGURES.glob("*"))]
    manifest = {
        "analysis": "matched three-condition UMAP for BM CD19 completion",
        "status": "complete",
        "cells": EXPECTED_CELLS,
        "conditions": ALL_CONDITIONS,
        "markers": markers,
        "alignment_manifest": str(alignment_manifest_path.resolve()),
        "alignment_manifest_sha256": sha256_file(alignment_manifest_path),
        "alignment_inputs": {
            condition: {
                "path": str(input_paths[condition].resolve()),
                "sha256": sha256_file(input_paths[condition]),
                "numeric_dtype_after_read": str(aligned[condition].dtype),
            }
            for condition in ALL_CONDITIONS
        },
        "metadata": {"path": str(metadata_path.resolve()), "sha256": sha256_file(metadata_path)},
        "umap_protocol": UMAP_PROTOCOL,
        "labels_used_only_after_umap_for_metrics_and_display": True,
        "runtime": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "scikit_learn": version("scikit-learn"),
            "umap_learn": version("umap-learn"),
            "matplotlib": version("matplotlib"),
        },
        "outputs": {str(path.relative_to(HERE)): sha256_file(path) for path in generated},
        "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))},
    }
    write_json(HERE / "manifest.json", manifest)
    print(metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
