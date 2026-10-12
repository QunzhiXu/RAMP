#!/usr/bin/env python3
"""Recluster Senosain parent cluster C06 and summarize marker expression.

The frozen 140,000-cell preprocessing is never changed.  This script selects
the cells assigned to C06 and, by default, applies the exact global marker means
and scales inherited from the parent all-cell workflow before fitting a local
all-34-uniform-marker PCA -> 30-NN UMAP graph -> Leiden -> UMAP workflow.  A
C06-local scaling mode is retained only as an explicit sensitivity option.
CD90 remains display-only because it was not measured uniformly across the
provider acquisitions.

No rough label, provider cell type, patient group, clinical variable, or
outcome is used during reclustering.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler


SEED = 42
PARENT_CLUSTER = "C06"
PCA_COMPONENTS = 25
N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.15
DEFAULT_RESOLUTION = 0.50
LEIDEN_ITERATIONS = 2
RESOLUTION_SCAN = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00, 1.20]
STABILITY_SEEDS = [42, 43, 44, 45, 46]
EMBEDDING_EXCLUSIONS = {"CD90"}

HERE = Path(__file__).resolve().parent
ANNOTATION_ROOT = HERE.parent
BASELINE_PATH = (
    ANNOTATION_ROOT
    / "kmeanslow_pca_umap"
    / "senosain_140k_kmeanslow35.float32.npy"
)
EMBEDDING_MANIFEST_PATH = ANNOTATION_ROOT / "kmeanslow_pca_umap" / "run_manifest.json"
SCALING_PATH = ANNOTATION_ROOT / "kmeanslow_pca_umap" / "feature_scaling.csv"
ASSIGNMENTS_PATH = ANNOTATION_ROOT / "leiden_all34" / "cluster_assignments.csv.gz"
MARKERS_PATH = ANNOTATION_ROOT / "senosain_140k_markers35.csv"
DEFAULT_OUTPUT = HERE / "c06_recluster"

MARKER_GROUPS = {
    "Lineage / identity": [
        "CD45",
        "CD3",
        "CD4",
        "CD8",
        "CD56",
        "CD11B",
        "CD31",
        "VIMENTIN",
        "CD90",
        "EPCAM",
        "CYTOKERATIN",
        "CK7",
        "TTF1",
        "TP63",
    ],
    "Tumor phenotype / immune state": [
        "HER2",
        "EGFR",
        "MET",
        "B_CATENIN",
        "CD24",
        "CD44",
        "HLA_ABC",
        "HLA_DR",
        "PD_L1",
    ],
    "Functional / signaling": [
        "KI67",
        "CCASP3",
        "TP53",
        "MDM2",
        "THIOREDOXIN",
        "P_AKT",
        "P_ERK",
        "P_S6",
        "P_STAT3",
        "P_STAT5",
        "P_STAT6",
        "HISTONE_H3",
    ],
}

FEATURE_MARKERS = [
    "CD45",
    "CD3",
    "CD4",
    "CD8",
    "CD56",
    "CD11B",
    "CD31",
    "VIMENTIN",
    "EPCAM",
    "CYTOKERATIN",
    "CK7",
    "TTF1",
    "KI67",
    "HLA_DR",
    "PD_L1",
    "MDM2",
    "P_STAT3",
    "HISTONE_H3",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=DEFAULT_RESOLUTION,
        help=f"Frozen Leiden resolution (default: {DEFAULT_RESOLUTION})",
    )
    parser.add_argument(
        "--scaling-mode",
        choices=["inherited", "local"],
        default="inherited",
        help=(
            "Use frozen all-cell feature scaling (default) or refit scaling "
            "inside C06 as a sensitivity analysis"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output directory after inputs validate.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def import_umap_without_disk_cache():
    """Import umap-learn while disabling only its numba disk caches."""

    import numba

    for name in ("jit", "njit", "vectorize", "guvectorize"):
        original = getattr(numba, name)
        if getattr(original, "_senosain_no_disk_cache", False):
            continue

        def no_disk_cache(*args, _original=original, **kwargs):
            kwargs["cache"] = False
            return _original(*args, **kwargs)

        no_disk_cache._senosain_no_disk_cache = True
        setattr(numba, name, no_disk_cache)

    return importlib.import_module("umap")


def validate_inputs(
    baseline: np.ndarray,
    assignments: pd.DataFrame,
    markers: list[str],
    embedding_manifest: dict,
    scaling_table: pd.DataFrame,
) -> tuple[np.ndarray, list[str], list[int]]:
    required_columns = {
        "sampled_row_zero_based",
        "source_cell_id",
        "patient_id",
        "unit_key",
        "cluster",
        "UMAP1",
        "UMAP2",
    }
    missing = required_columns - set(assignments.columns)
    if missing:
        raise ValueError(f"Frozen assignments lack columns: {sorted(missing)}")
    if baseline.shape != (len(assignments), len(markers)):
        raise ValueError(
            f"Baseline/assignment/marker mismatch: {baseline.shape}, "
            f"{len(assignments)}, {len(markers)}"
        )
    if len(markers) != 35 or len(set(markers)) != 35:
        raise ValueError("Expected 35 unique archived KMEANSLOW markers")
    if not np.isfinite(baseline).all():
        raise ValueError("Archived KMEANSLOW matrix contains non-finite values")
    if not np.array_equal(
        assignments["sampled_row_zero_based"].to_numpy(),
        np.arange(len(assignments)),
    ):
        raise ValueError("Frozen assignments are no longer matrix-row aligned")
    if not assignments["source_cell_id"].astype(str).is_unique:
        raise ValueError("Frozen source_cell_id values are not unique")

    manifest_markers = embedding_manifest["archived_kmeanslow_markers"]
    if markers != manifest_markers:
        raise ValueError("Marker order differs from the frozen embedding manifest")
    embedding_markers = embedding_manifest["embedding_markers"]
    if set(embedding_markers) != set(markers) - EMBEDDING_EXCLUSIONS:
        raise ValueError("Frozen manifest does not define the expected uniform 34 markers")
    if len(embedding_markers) != 34 or "CD90" in embedding_markers:
        raise ValueError("Expected exactly 34 uniform embedding markers without CD90")
    required_scaling_columns = {
        "marker",
        "global_mean_before_zscore",
        "global_scale",
    }
    if not required_scaling_columns.issubset(scaling_table.columns):
        missing_scaling = required_scaling_columns - set(scaling_table.columns)
        raise ValueError(f"Frozen scaling table lacks columns: {sorted(missing_scaling)}")
    if scaling_table["marker"].astype(str).tolist() != embedding_markers:
        raise ValueError("Frozen scaling marker order differs from the manifest")
    scaling_values = scaling_table[
        ["global_mean_before_zscore", "global_scale"]
    ].to_numpy(float)
    if not np.isfinite(scaling_values).all() or np.any(scaling_values[:, 1] <= 0):
        raise ValueError("Frozen global feature scaling is invalid")
    ordered_display = [m for group in MARKER_GROUPS.values() for m in group]
    if len(ordered_display) != 35 or set(ordered_display) != set(markers):
        raise ValueError("Heatmap marker groups do not cover the 35-marker panel once")
    if not set(FEATURE_MARKERS).issubset(markers):
        raise ValueError("A requested feature-plot marker is absent")

    parent_mask = assignments["cluster"].astype(str).eq(PARENT_CLUSTER).to_numpy()
    parent_cells = int(parent_mask.sum())
    if parent_cells != 9_778:
        raise ValueError(f"Expected 9,778 frozen C06 cells; found {parent_cells:,}")
    embedding_indices = [markers.index(marker) for marker in embedding_markers]
    return parent_mask, embedding_markers, embedding_indices


def stable_subcluster_names(
    raw_membership: np.ndarray,
) -> tuple[np.ndarray, dict[int, str]]:
    raw_labels, counts = np.unique(raw_membership, return_counts=True)
    order = raw_labels[np.argsort(-counts, kind="stable")]
    mapping = {int(raw): f"S{rank:02d}" for rank, raw in enumerate(order)}
    names = np.asarray([mapping[int(raw)] for raw in raw_membership], dtype=object)
    return names, mapping


def leiden_partition(graph, igraph_module, resolution: float, seed: int):
    igraph_module.set_random_number_generator(random.Random(seed))
    return graph.community_leiden(
        objective_function="modularity",
        weights="weight",
        resolution=resolution,
        n_iterations=LEIDEN_ITERATIONS,
    )


def resolution_and_stability_scan(graph, igraph_module, resolution: float):
    resolutions = sorted(set(RESOLUTION_SCAN + [float(resolution)]))
    resolution_records: list[dict] = []
    chosen_membership: np.ndarray | None = None
    for current in resolutions:
        partition = leiden_partition(graph, igraph_module, current, SEED)
        sizes = np.asarray(partition.sizes(), dtype=int)
        resolution_records.append(
            {
                "resolution": current,
                "n_subclusters": int(len(sizes)),
                "minimum_subcluster_cells": int(sizes.min()),
                "median_subcluster_cells": float(np.median(sizes)),
                "maximum_subcluster_cells": int(sizes.max()),
                "subclusters_below_100_cells": int(np.sum(sizes < 100)),
                "modularity": float(partition.modularity),
                "quality": float(partition.quality),
            }
        )
        if np.isclose(current, resolution):
            chosen_membership = np.asarray(partition.membership, dtype=np.int32)
    if chosen_membership is None:
        raise AssertionError("Requested resolution was absent from the scan")

    stability_records: list[dict] = []
    for seed in STABILITY_SEEDS:
        if seed == SEED:
            membership = chosen_membership
        else:
            membership = np.asarray(
                leiden_partition(graph, igraph_module, resolution, seed).membership,
                dtype=np.int32,
            )
        stability_records.append(
            {
                "seed": seed,
                "n_subclusters": int(np.unique(membership).size),
                "adjusted_rand_vs_seed42": float(
                    adjusted_rand_score(chosen_membership, membership)
                ),
                "normalized_mutual_information_vs_seed42": float(
                    normalized_mutual_info_score(chosen_membership, membership)
                ),
            }
        )
    return (
        chosen_membership,
        pd.DataFrame(resolution_records),
        pd.DataFrame(stability_records),
    )


def summarize_profiles(
    baseline: np.ndarray,
    markers: list[str],
    subclusters: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records: list[dict] = []
    subcluster_order = sorted(np.unique(subclusters))
    for subcluster in subcluster_order:
        values = baseline[subclusters == subcluster]
        quantiles = np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90], axis=0)
        means = values.mean(axis=0)
        standard_deviations = values.std(axis=0, ddof=0)
        for marker_index, marker in enumerate(markers):
            records.append(
                {
                    "parent_cluster": PARENT_CLUSTER,
                    "subcluster": subcluster,
                    "marker": marker,
                    "mean_kmeanslow": float(means[marker_index]),
                    "sd_kmeanslow": float(standard_deviations[marker_index]),
                    "q10_kmeanslow": float(quantiles[0, marker_index]),
                    "q25_kmeanslow": float(quantiles[1, marker_index]),
                    "median_kmeanslow": float(quantiles[2, marker_index]),
                    "q75_kmeanslow": float(quantiles[3, marker_index]),
                    "q90_kmeanslow": float(quantiles[4, marker_index]),
                }
            )
    long = pd.DataFrame(records)
    median = long.pivot(
        index="subcluster", columns="marker", values="median_kmeanslow"
    ).loc[subcluster_order, markers]
    mean = long.pivot(
        index="subcluster", columns="marker", values="mean_kmeanslow"
    ).loc[subcluster_order, markers]
    return long, median, mean


def marker_wise_z(median: pd.DataFrame) -> pd.DataFrame:
    values = median.to_numpy(float)
    center = values.mean(axis=0, keepdims=True)
    scale = values.std(axis=0, ddof=0, keepdims=True)
    scale = np.maximum(scale, 1e-12)
    return pd.DataFrame(
        (values - center) / scale,
        index=median.index,
        columns=median.columns,
    )


def subcluster_size_table(
    subset_assignments: pd.DataFrame, subclusters: np.ndarray
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "subcluster": subclusters,
            "patient_id": subset_assignments["patient_id"].astype(str).to_numpy(),
        }
    )
    patient_counts = pd.crosstab(frame["subcluster"], frame["patient_id"])
    totals = patient_counts.sum(axis=1)
    return pd.DataFrame(
        {
            "parent_cluster": PARENT_CLUSTER,
            "subcluster": patient_counts.index,
            "cells": totals.to_numpy(int),
            "fraction_of_C06": (totals / len(frame)).to_numpy(float),
            "patients_with_cells": (patient_counts > 0).sum(axis=1).to_numpy(int),
            "median_cells_per_patient": patient_counts.median(axis=1).to_numpy(float),
            "maximum_single_patient_fraction_of_subcluster": (
                patient_counts.max(axis=1) / totals
            ).to_numpy(float),
        }
    ).sort_values("subcluster").reset_index(drop=True)


def top_relative_marker_table(z_values: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    ranking_values = z_values.drop(columns=["CD90"], errors="ignore")
    rows: list[dict] = []
    for subcluster in ranking_values.index:
        values = (
            ranking_values.loc[subcluster].sort_values(ascending=False).head(top_n)
        )
        for rank, (marker, z_value) in enumerate(values.items(), start=1):
            rows.append(
                {
                    "subcluster": subcluster,
                    "rank_within_subcluster": rank,
                    "marker": marker,
                    "marker_wise_z_of_median": float(z_value),
                }
            )
    return pd.DataFrame(rows)


def patient_quartile_coexpression_audit(
    baseline_all: np.ndarray,
    assignments_all: pd.DataFrame,
    parent_mask: np.ndarray,
    markers: list[str],
    subclusters: np.ndarray,
) -> pd.DataFrame:
    """Summarize rank-robust coexpression relative to each patient's q75.

    Thresholds are estimated from all 2,000 sampled cells for each patient, not
    from C06 alone.  KMEANSLOW preserves within-patient ranks, so this provides
    a simple cell-level check of whether cluster medians reflect homogeneous
    cross-lineage expression or mixtures of distinct cells.
    """

    audit_markers = [
        "CD45",
        "CD3",
        "TTF1",
        "CYTOKERATIN",
        "EPCAM",
        "CK7",
        "KI67",
        "HLA_DR",
        "PD_L1",
    ]
    marker_indices = [markers.index(marker) for marker in audit_markers]
    patient_all = assignments_all["patient_id"].astype(str).to_numpy()
    patient_parent = patient_all[parent_mask]
    parent_values = baseline_all[parent_mask][:, marker_indices]
    high = np.zeros(parent_values.shape, dtype=bool)
    for patient in pd.unique(patient_all):
        all_rows = patient_all == patient
        parent_rows = patient_parent == patient
        thresholds = np.quantile(
            baseline_all[all_rows][:, marker_indices], 0.75, axis=0
        )
        high[parent_rows] = parent_values[parent_rows] > thresholds

    high_by_marker = {
        marker: high[:, index] for index, marker in enumerate(audit_markers)
    }
    t_like = high_by_marker["CD45"] & high_by_marker["CD3"]
    epithelial_like = (
        high_by_marker["TTF1"] & high_by_marker["CYTOKERATIN"]
    )
    epcam_or_ck7 = high_by_marker["EPCAM"] | high_by_marker["CK7"]
    flags = {
        "KI67_above_patient_q75": high_by_marker["KI67"],
        "CD45_and_CD3_above_patient_q75": t_like,
        "TTF1_and_CYTOKERATIN_above_patient_q75": epithelial_like,
        "T_like_and_epithelial_like": t_like & epithelial_like,
        "EPCAM_or_CK7_above_patient_q75": epcam_or_ck7,
        "HLA_DR_above_patient_q75": high_by_marker["HLA_DR"],
        "PD_L1_above_patient_q75": high_by_marker["PD_L1"],
    }
    records: list[dict] = []
    for subcluster in sorted(np.unique(subclusters)):
        rows = subclusters == subcluster
        record: dict[str, object] = {
            "parent_cluster": PARENT_CLUSTER,
            "subcluster": subcluster,
            "cells": int(rows.sum()),
        }
        for name, values in flags.items():
            record[f"{name}_cells"] = int(values[rows].sum())
            record[f"{name}_fraction"] = float(values[rows].mean())
        records.append(record)
    return pd.DataFrame(records)


def setup_plotting(output: Path):
    mpl_cache = output / ".matplotlib_cache"
    xdg_cache = output / ".xdg_cache"
    mpl_cache.mkdir(exist_ok=True)
    xdg_cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titleweight": "bold",
            "savefig.dpi": 300,
        }
    )
    return plt


def style_umap_axis(axis, title: str) -> None:
    axis.set_title(title, fontsize=13, fontweight="bold")
    axis.set_xlabel("local UMAP 1")
    axis.set_ylabel("local UMAP 2")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.spines[:].set_visible(False)


def plot_subcluster_umap(
    output: Path,
    coordinates: np.ndarray,
    subclusters: np.ndarray,
    sizes: pd.DataFrame,
    resolution: float,
    plt,
) -> list[Path]:
    from matplotlib import patheffects

    subcluster_order = sizes["subcluster"].tolist()
    palette = plt.get_cmap("tab10")
    colors = {name: palette(i) for i, name in enumerate(subcluster_order)}
    fig, axis = plt.subplots(figsize=(10.6, 7.4), constrained_layout=True)
    for subcluster in subcluster_order:
        rows = subclusters == subcluster
        n = int(rows.sum())
        fraction = n / len(subclusters)
        axis.scatter(
            coordinates[rows, 0],
            coordinates[rows, 1],
            s=1.5,
            alpha=0.66,
            linewidths=0,
            rasterized=True,
            color=colors[subcluster],
            label=f"{subcluster}  n={n:,} ({fraction:.1%})",
        )
        x, y = np.median(coordinates[rows], axis=0)
        label = axis.text(
            x,
            y,
            subcluster,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="black",
        )
        label.set_path_effects(
            [patheffects.withStroke(linewidth=3.0, foreground="white")]
        )
    style_umap_axis(
        axis,
        f"C06 local all-34-marker Leiden subclusters · resolution {resolution:g}",
    )
    axis.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        markerscale=4,
        fontsize=9,
    )
    created: list[Path] = []
    for suffix in ("png", "pdf"):
        path = output / f"c06_local_umap_subclusters.{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        created.append(path)
    plt.close(fig)
    return created


def plot_feature_umaps(
    output: Path,
    coordinates: np.ndarray,
    baseline: np.ndarray,
    markers: list[str],
    plt,
) -> list[Path]:
    marker_index = {marker: index for index, marker in enumerate(markers)}
    fig, axes = plt.subplots(3, 6, figsize=(21.0, 10.8), constrained_layout=True)
    for axis, marker in zip(axes.flat, FEATURE_MARKERS, strict=True):
        values = baseline[:, marker_index[marker]]
        low, high = np.quantile(values, [0.01, 0.99])
        if high <= low:
            low, high = float(values.min()), float(values.max())
        order = np.argsort(values, kind="stable")
        scatter = axis.scatter(
            coordinates[order, 0],
            coordinates[order, 1],
            c=np.clip(values[order], low, high),
            s=0.72,
            cmap="viridis",
            vmin=low,
            vmax=high,
            linewidths=0,
            rasterized=True,
        )
        style_umap_axis(axis, marker)
        colorbar = fig.colorbar(scatter, ax=axis, fraction=0.045, pad=0.015)
        colorbar.set_label("KMEANSLOW\n1st–99th pct.", fontsize=7.5)
        colorbar.ax.tick_params(labelsize=7)
    fig.suptitle(
        "C06 local UMAP · diagnostic marker expression",
        fontsize=16,
        fontweight="bold",
    )
    created: list[Path] = []
    for suffix in ("png", "pdf"):
        path = output / f"c06_local_umap_key_marker_expression.{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        created.append(path)
    plt.close(fig)
    return created


def plot_profile_heatmap(
    output: Path,
    median: pd.DataFrame,
    z_values: pd.DataFrame,
    sizes: pd.DataFrame,
    plt,
) -> list[Path]:
    from matplotlib.colors import TwoSlopeNorm

    display_markers = [m for group in MARKER_GROUPS.values() for m in group]
    subcluster_order = sizes["subcluster"].tolist()
    z = z_values.loc[subcluster_order, display_markers].to_numpy(float)
    exact = median.loc[subcluster_order, display_markers].to_numpy(float)
    size_index = sizes.set_index("subcluster")
    row_labels = [
        f"{name}  |  n={int(size_index.loc[name, 'cells']):,}  "
        f"({size_index.loc[name, 'fraction_of_C06']:.1%})"
        for name in subcluster_order
    ]

    fig = plt.figure(figsize=(30.0, 8.3), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[35, 4.0])
    axis = fig.add_subplot(grid[0, 0])
    bar_axis = fig.add_subplot(grid[0, 1])
    norm = TwoSlopeNorm(vmin=-2.2, vcenter=0.0, vmax=2.2)
    image = axis.imshow(
        np.clip(z, -2.2, 2.2),
        aspect="auto",
        cmap="RdBu_r",
        norm=norm,
        interpolation="nearest",
        rasterized=True,
    )
    xlabels = ["CD90*" if marker == "CD90" else marker for marker in display_markers]
    axis.set_xticks(np.arange(len(display_markers)), labels=xlabels)
    axis.tick_params(axis="x", labelrotation=58, labelsize=9, pad=3)
    for label in axis.get_xticklabels():
        label.set_ha("right")
    axis.set_yticks(np.arange(len(subcluster_order)), labels=row_labels)
    axis.tick_params(axis="y", labelsize=10)
    axis.set_xlabel("Biomarker")
    axis.set_ylabel("C06 Leiden subcluster")

    group_start = 0
    group_colors = ["#3B82F6", "#10B981", "#F59E0B"]
    for (group_name, group_markers), color in zip(
        MARKER_GROUPS.items(), group_colors, strict=True
    ):
        group_end = group_start + len(group_markers)
        axis.plot(
            [group_start - 0.45, group_end - 0.55],
            [-0.82, -0.82],
            color=color,
            linewidth=4,
            clip_on=False,
        )
        axis.text(
            (group_start + group_end - 1) / 2,
            -1.05,
            group_name,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=color,
            clip_on=False,
        )
        if group_end < len(display_markers):
            axis.axvline(group_end - 0.5, color="white", linewidth=2.2)
        group_start = group_end
    for row in range(exact.shape[0]):
        for column in range(exact.shape[1]):
            axis.text(
                column,
                row,
                f"{exact[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=7.2,
                color="white" if abs(z[row, column]) >= 1.2 else "black",
            )
    colorbar = fig.colorbar(image, ax=axis, orientation="horizontal", shrink=0.35, pad=0.16)
    colorbar.set_label("Marker-wise z-score of C06 subcluster median")

    fractions = size_index.loc[subcluster_order, "fraction_of_C06"].to_numpy(float)
    y = np.arange(len(subcluster_order))
    bar_axis.barh(y, fractions, color="#5B6472", height=0.70)
    bar_axis.set_ylim(len(subcluster_order) - 0.5, -0.5)
    bar_axis.set_yticks([])
    bar_axis.set_xlabel("C06 cells (%)")
    bar_axis.set_title("Subcluster size")
    bar_axis.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    bar_axis.spines[["top", "right", "left"]].set_visible(False)
    for row, fraction in enumerate(fractions):
        bar_axis.text(
            fraction,
            row,
            f" {fraction:.1%}",
            va="center",
            ha="left",
            fontsize=8.5,
        )
    fig.suptitle(
        "C06 subcluster biomarker profiles · exact KMEANSLOW medians printed\n"
        "Color is relative within C06 for each marker; CD90* is display-only",
        fontsize=14,
        fontweight="bold",
    )
    created: list[Path] = []
    for suffix in ("png", "pdf"):
        path = output / f"c06_subcluster_marker_heatmap.{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        created.append(path)
    plt.close(fig)
    return created


def main() -> None:
    args = parse_args()
    if args.resolution <= 0:
        raise ValueError("Leiden resolution must be positive")
    output = args.output.resolve()

    print("Loading frozen all-cell C06 inputs", flush=True)
    baseline_all = np.load(BASELINE_PATH)
    assignments_all = pd.read_csv(ASSIGNMENTS_PATH, low_memory=False)
    markers = pd.read_csv(MARKERS_PATH)["marker"].astype(str).tolist()
    embedding_manifest = json.loads(EMBEDDING_MANIFEST_PATH.read_text())
    scaling_table = pd.read_csv(SCALING_PATH)
    parent_mask, embedding_markers, embedding_indices = validate_inputs(
        baseline_all,
        assignments_all,
        markers,
        embedding_manifest,
        scaling_table,
    )
    input_hashes = {
        path.name: sha256(path)
        for path in [
            BASELINE_PATH,
            EMBEDDING_MANIFEST_PATH,
            SCALING_PATH,
            ASSIGNMENTS_PATH,
            MARKERS_PATH,
            Path(__file__).resolve(),
        ]
    }

    if output.exists():
        if not args.force:
            raise FileExistsError(
                f"Output directory already exists: {output}. Use --force to replace it."
            )
        shutil.rmtree(output)
    output.mkdir(parents=True)
    plt = setup_plotting(output)
    started = time.perf_counter()

    baseline = baseline_all[parent_mask]
    subset_assignments = assignments_all.loc[parent_mask].copy().reset_index(drop=True)
    embedding_values = baseline[:, embedding_indices]
    variance = np.var(embedding_values, axis=0, ddof=0)
    retained = variance > 1e-10
    if not retained.all():
        dropped = [m for m, keep in zip(embedding_markers, retained, strict=True) if not keep]
        print(f"Dropping constant markers inside C06: {dropped}", flush=True)
        embedding_values = embedding_values[:, retained]
        embedding_markers = [
            m for m, keep in zip(embedding_markers, retained, strict=True) if keep
        ]

    if args.scaling_mode == "inherited":
        inherited = scaling_table.set_index("marker").loc[embedding_markers]
        inherited_mean = inherited["global_mean_before_zscore"].to_numpy(float)
        inherited_scale = inherited["global_scale"].to_numpy(float)
        normalized = (embedding_values - inherited_mean) / inherited_scale
        scaling_output = pd.DataFrame(
            {
                "marker": embedding_markers,
                "scaling_mode": "inherited_all_cell",
                "mean_subtracted": inherited_mean,
                "scale_divisor": inherited_scale,
                "C06_mean_after_scaling": normalized.mean(axis=0),
                "C06_sd_after_scaling": normalized.std(axis=0, ddof=0),
            }
        )
        print(
            f"C06 cells={len(baseline):,}; applying inherited all-cell scaling "
            f"to {len(embedding_markers)} markers",
            flush=True,
        )
    else:
        scaler = StandardScaler(copy=True)
        normalized = scaler.fit_transform(embedding_values)
        scaling_output = pd.DataFrame(
            {
                "marker": embedding_markers,
                "scaling_mode": "C06_local_sensitivity",
                "mean_subtracted": scaler.mean_,
                "scale_divisor": scaler.scale_,
                "C06_mean_after_scaling": normalized.mean(axis=0),
                "C06_sd_after_scaling": normalized.std(axis=0, ddof=0),
            }
        )
        print(
            f"C06 cells={len(baseline):,}; fitting sensitivity-only local scaling "
            f"on {len(embedding_markers)} markers",
            flush=True,
        )
    pca = PCA(
        n_components=PCA_COMPONENTS,
        svd_solver="randomized",
        random_state=SEED,
    )
    pca_coordinates = pca.fit_transform(normalized)
    np.save(output / "c06_local_pca25.float32.npy", pca_coordinates.astype(np.float32))
    scaling_output.to_csv(output / "c06_feature_scaling_used.csv", index=False)
    pd.DataFrame(
        {
            "PC": np.arange(1, PCA_COMPONENTS + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance_ratio": np.cumsum(
                pca.explained_variance_ratio_
            ),
        }
    ).to_csv(output / "c06_local_pca_explained_variance.csv", index=False)

    print("Fitting the C06 local 30-NN UMAP graph and coordinates", flush=True)
    umap_module = import_umap_without_disk_cache()
    reducer = umap_module.UMAP(
        n_neighbors=N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        n_components=2,
        metric="euclidean",
        init="spectral",
        random_state=SEED,
        n_jobs=1,
        low_memory=True,
        verbose=False,
    )
    local_umap = reducer.fit_transform(pca_coordinates)
    if not np.isfinite(local_umap).all():
        raise AssertionError("Local UMAP returned non-finite coordinates")
    np.save(output / "c06_local_umap2.float32.npy", local_umap.astype(np.float32))

    connectivities = reducer.graph_.maximum(reducer.graph_.T).tocsr()
    connectivities.setdiag(0)
    connectivities.eliminate_zeros()
    import igraph as ig

    graph = ig.Graph.Weighted_Adjacency(
        connectivities,
        mode="undirected",
        attr="weight",
        loops=False,
    )
    if graph.vcount() != len(subset_assignments):
        raise AssertionError("C06 graph vertex count does not match C06 cells")

    print(
        f"Scanning Leiden resolutions and freezing resolution={args.resolution:g}",
        flush=True,
    )
    raw_membership, resolution_table, stability_table = resolution_and_stability_scan(
        graph, ig, args.resolution
    )
    subclusters, raw_to_named = stable_subcluster_names(raw_membership)
    resolution_table.to_csv(output / "c06_leiden_resolution_scan.csv", index=False)
    stability_table.to_csv(output / "c06_leiden_seed_stability.csv", index=False)

    sizes = subcluster_size_table(subset_assignments, subclusters)
    sizes.to_csv(output / "c06_subcluster_sizes.csv", index=False)
    subset_assignments.insert(
        subset_assignments.columns.get_loc("cluster") + 1,
        "parent_cluster",
        PARENT_CLUSTER,
    )
    subset_assignments.insert(
        subset_assignments.columns.get_loc("parent_cluster") + 1,
        "subcluster",
        subclusters,
    )
    subset_assignments["C06_local_UMAP1"] = local_umap[:, 0]
    subset_assignments["C06_local_UMAP2"] = local_umap[:, 1]
    subset_assignments.to_csv(
        output / "c06_subcluster_assignments.csv.gz",
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )

    profile_long, profile_median, profile_mean = summarize_profiles(
        baseline, markers, subclusters
    )
    profile_z = marker_wise_z(profile_median)
    profile_long.to_csv(
        output / "c06_subcluster_profile_long.csv.gz",
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )
    profile_median.to_csv(output / "c06_subcluster_median_wide.csv")
    profile_mean.to_csv(output / "c06_subcluster_mean_wide.csv")
    profile_z.to_csv(output / "c06_subcluster_median_marker_z.csv")
    top_relative_marker_table(profile_z).to_csv(
        output / "c06_subcluster_top_relative_markers.csv", index=False
    )
    patient_quartile_coexpression_audit(
        baseline_all,
        assignments_all,
        parent_mask,
        markers,
        subclusters,
    ).to_csv(output / "c06_subcluster_coexpression_audit.csv", index=False)

    created_plots = [
        *plot_subcluster_umap(
            output,
            local_umap,
            subclusters,
            sizes,
            args.resolution,
            plt,
        ),
        *plot_feature_umaps(output, local_umap, baseline, markers, plt),
        *plot_profile_heatmap(output, profile_median, profile_z, sizes, plt),
    ]

    runtime_seconds = time.perf_counter() - started
    manifest = {
        "schema_version": "senosain-c06-local-recluster-v1",
        "inputs": input_hashes,
        "parent_cluster": {
            "name": PARENT_CLUSTER,
            "cells": int(parent_mask.sum()),
            "source": str(ASSIGNMENTS_PATH),
            "selection_column": "cluster",
        },
        "preprocessing": {
            "input": "frozen patient-wise KMEANSLOW values from the 140k workflow",
            "additional_kmeanslow_or_arcsinh": False,
            "scaling_mode": args.scaling_mode,
            "primary_transform": (
                "inherited marker means and scales from the frozen 140k workflow"
                if args.scaling_mode == "inherited"
                else "marker-wise StandardScaler fitted only on C06; sensitivity mode"
            ),
            "scaling_parameters_file": "c06_feature_scaling_used.csv",
            "clustering_markers": embedding_markers,
            "clustering_markers_count": len(embedding_markers),
            "display_only": {
                "CD90": "not measured uniformly across provider acquisitions"
            },
            "cell_labels_or_clinical_outcomes_used": False,
        },
        "pca": {
            "components": PCA_COMPONENTS,
            "solver": "randomized",
            "seed": SEED,
            "cumulative_explained_variance_ratio": float(
                pca.explained_variance_ratio_.sum()
            ),
        },
        "umap_graph_and_coordinates": {
            "neighbors": N_NEIGHBORS,
            "min_dist": UMAP_MIN_DIST,
            "metric": "euclidean",
            "init": "spectral",
            "seed": SEED,
            "n_jobs": 1,
            "graph_edges": int(graph.ecount()),
        },
        "leiden": {
            "implementation": "python-igraph Graph.community_leiden",
            "objective": "modularity",
            "resolution": float(args.resolution),
            "iterations": LEIDEN_ITERATIONS,
            "seed": SEED,
            "subclusters": int(np.unique(subclusters).size),
            "subcluster_names": "S00.. ordered by decreasing cell count",
            "raw_to_named_subcluster": raw_to_named,
            "resolution_scan": "c06_leiden_resolution_scan.csv",
            "seed_stability": "c06_leiden_seed_stability.csv",
        },
        "profile_summary": {
            "scale": "frozen KMEANSLOW values before C06-local StandardScaler",
            "all_archived_markers_displayed": markers,
            "heatmap_color": "marker-wise z-score of subcluster medians within C06",
            "heatmap_cell_text": "exact subcluster median KMEANSLOW value",
        },
        "coexpression_audit": {
            "file": "c06_subcluster_coexpression_audit.csv",
            "threshold": (
                "strictly above each marker's 75th percentile among all 2,000 "
                "sampled cells from the same patient"
            ),
            "purpose": "cell-level check of mixed-lineage cluster medians",
        },
        "plots": [path.name for path in created_plots],
        "runtime_seconds": runtime_seconds,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "umap_learn": importlib.metadata.version("umap-learn"),
            "igraph": importlib.metadata.version("igraph"),
        },
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"Complete: {len(subset_assignments):,} C06 cells -> "
        f"{len(sizes)} subclusters in {runtime_seconds:.1f} s",
        flush=True,
    )
    print(output, flush=True)


if __name__ == "__main__":
    main()
