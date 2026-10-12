#!/usr/bin/env python3
"""Cluster the all-34-marker Senosain PCA graph and print cluster profiles.

The graph uses the existing PCA25 representation that was fitted from all 34
uniformly measured biological markers.  A weighted 30-nearest-neighbor fuzzy
graph is constructed exactly as for a standard UMAP/Scanpy-style neighbor
workflow, and Leiden is run on that graph.  The frozen UMAP coordinates are used
only for display.

All 35 KMEANSLOW markers are summarized after clustering.  CD90 is shown in the
profile tables and heatmap but was not used in PCA, graph construction, Leiden,
or selection of the frozen resolution because it was not measured uniformly
across the full published cohort.
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
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.cluster.hierarchy import leaves_list, linkage
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


SEED = 42
N_NEIGHBORS = 30
FROZEN_RESOLUTION = 1.0
LEIDEN_ITERATIONS = 2
RESOLUTION_SCAN = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
STABILITY_SEEDS = [42, 43, 44, 45, 46]

HERE = Path(__file__).resolve().parent
EMBEDDING_DIR = HERE / "kmeanslow_pca_umap"
BASELINE_PATH = EMBEDDING_DIR / "senosain_140k_kmeanslow35.float32.npy"
PCA_PATH = EMBEDDING_DIR / "senosain_140k_pca25.float32.npy"
UMAP_PATH = EMBEDDING_DIR / "senosain_140k_umap2.float32.npy"
EMBEDDING_MANIFEST_PATH = EMBEDDING_DIR / "run_manifest.json"
METADATA_PATH = HERE / "senosain_140k_cell_metadata.csv.gz"
MARKERS_PATH = HERE / "senosain_140k_markers35.csv"
DEFAULT_OUTPUT = HERE / "leiden_all34"

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT})",
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

    return importlib.import_module("umap.umap_")


def validate_inputs(
    baseline: np.ndarray,
    pca: np.ndarray,
    umap_coordinates: np.ndarray,
    metadata: pd.DataFrame,
    markers: list[str],
    embedding_manifest: dict,
) -> None:
    n = len(metadata)
    if baseline.shape != (n, 35):
        raise ValueError(f"Expected a {n} x 35 KMEANSLOW matrix; got {baseline.shape}")
    if pca.shape != (n, 25):
        raise ValueError(f"Expected a {n} x 25 PCA matrix; got {pca.shape}")
    if umap_coordinates.shape != (n, 2):
        raise ValueError(f"Expected a {n} x 2 UMAP matrix; got {umap_coordinates.shape}")
    for name, values in (
        ("KMEANSLOW", baseline),
        ("PCA", pca),
        ("UMAP", umap_coordinates),
    ):
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite values")
    if len(markers) != 35 or len(set(markers)) != 35:
        raise ValueError("Expected 35 unique markers")
    expected_marker_order = embedding_manifest["archived_kmeanslow_markers"]
    if markers != expected_marker_order:
        raise ValueError("Marker order differs from the embedding manifest")
    embedding_markers = embedding_manifest["embedding_markers"]
    if len(embedding_markers) != 34 or "CD90" in embedding_markers:
        raise ValueError("Expected PCA25 to use all uniform markers except CD90")
    if set(embedding_markers) != set(markers) - {"CD90"}:
        raise ValueError("The PCA embedding marker set is not exactly the uniform 34")
    ordered_heatmap_markers = [m for group in MARKER_GROUPS.values() for m in group]
    if set(ordered_heatmap_markers) != set(markers) or len(ordered_heatmap_markers) != 35:
        raise ValueError("Heatmap marker groups do not cover the 35-marker panel once")
    if not np.array_equal(
        metadata["sampled_row_zero_based"].to_numpy(), np.arange(n)
    ):
        raise ValueError("Metadata and matrices are no longer row aligned")
    unit_sizes = metadata.groupby("unit_key", sort=False).size()
    if len(unit_sizes) != 70 or not unit_sizes.eq(2000).all():
        raise ValueError("Expected 70 patients with exactly 2,000 cells each")


def build_graph(pca: np.ndarray):
    umap_module = import_umap_without_disk_cache()
    connectivities, _, _ = umap_module.fuzzy_simplicial_set(
        pca,
        n_neighbors=N_NEIGHBORS,
        random_state=np.random.RandomState(SEED),
        metric="euclidean",
        metric_kwds={},
        angular=False,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
        apply_set_operations=True,
        verbose=True,
    )
    connectivities = connectivities.maximum(connectivities.T).tocsr()
    connectivities.setdiag(0)
    connectivities.eliminate_zeros()
    return connectivities, umap_module


def leiden_partition(graph, igraph_module, resolution: float, seed: int):
    igraph_module.set_random_number_generator(random.Random(seed))
    return graph.community_leiden(
        objective_function="modularity",
        weights="weight",
        resolution=resolution,
        n_iterations=LEIDEN_ITERATIONS,
    )


def run_resolution_and_stability_scan(graph, igraph_module):
    resolution_records: list[dict] = []
    chosen_membership: np.ndarray | None = None
    for resolution in RESOLUTION_SCAN:
        partition = leiden_partition(graph, igraph_module, resolution, SEED)
        sizes = np.asarray(partition.sizes(), dtype=int)
        resolution_records.append(
            {
                "resolution": resolution,
                "n_clusters": int(len(sizes)),
                "minimum_cluster_cells": int(sizes.min()),
                "median_cluster_cells": float(np.median(sizes)),
                "maximum_cluster_cells": int(sizes.max()),
                "clusters_below_100_cells": int(np.sum(sizes < 100)),
                "modularity": float(partition.modularity),
                "quality": float(partition.quality),
            }
        )
        if resolution == FROZEN_RESOLUTION:
            chosen_membership = np.asarray(partition.membership, dtype=np.int32)

    if chosen_membership is None:
        raise AssertionError("Frozen resolution was absent from the scan")

    stability_records: list[dict] = []
    for seed in STABILITY_SEEDS:
        if seed == SEED:
            membership = chosen_membership
        else:
            membership = np.asarray(
                leiden_partition(
                    graph, igraph_module, FROZEN_RESOLUTION, seed
                ).membership,
                dtype=np.int32,
            )
        stability_records.append(
            {
                "seed": seed,
                "n_clusters": int(np.unique(membership).size),
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


def stable_cluster_names(raw_membership: np.ndarray) -> tuple[np.ndarray, dict[int, str]]:
    raw_labels, counts = np.unique(raw_membership, return_counts=True)
    order = raw_labels[np.argsort(-counts, kind="stable")]
    mapping = {int(raw): f"C{rank:02d}" for rank, raw in enumerate(order)}
    named = np.asarray([mapping[int(raw)] for raw in raw_membership], dtype=object)
    return named, mapping


def summarize_profiles(
    baseline: np.ndarray,
    markers: list[str],
    clusters: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records: list[dict] = []
    cluster_order = sorted(np.unique(clusters))
    for cluster in cluster_order:
        values = baseline[clusters == cluster]
        quantiles = np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90], axis=0)
        means = values.mean(axis=0)
        standard_deviations = values.std(axis=0, ddof=0)
        for j, marker in enumerate(markers):
            records.append(
                {
                    "cluster": cluster,
                    "marker": marker,
                    "mean_kmeanslow": float(means[j]),
                    "sd_kmeanslow": float(standard_deviations[j]),
                    "q10_kmeanslow": float(quantiles[0, j]),
                    "q25_kmeanslow": float(quantiles[1, j]),
                    "median_kmeanslow": float(quantiles[2, j]),
                    "q75_kmeanslow": float(quantiles[3, j]),
                    "q90_kmeanslow": float(quantiles[4, j]),
                }
            )
    long = pd.DataFrame(records)
    median = long.pivot(index="cluster", columns="marker", values="median_kmeanslow")
    mean = long.pivot(index="cluster", columns="marker", values="mean_kmeanslow")
    median = median.loc[cluster_order, markers]
    mean = mean.loc[cluster_order, markers]
    return long, median, mean


def cluster_size_table(metadata: pd.DataFrame, clusters: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "cluster": clusters,
            "patient_id": metadata["patient_id"].astype(str).to_numpy(),
        }
    )
    patient_counts = pd.crosstab(frame["cluster"], frame["patient_id"])
    totals = patient_counts.sum(axis=1)
    result = pd.DataFrame(
        {
            "cluster": patient_counts.index,
            "cells": totals.to_numpy(int),
            "fraction_of_all_cells": (totals / len(frame)).to_numpy(float),
            "patients_with_cells": (patient_counts > 0).sum(axis=1).to_numpy(int),
            "median_cells_per_patient": patient_counts.median(axis=1).to_numpy(float),
            "maximum_single_patient_fraction_of_cluster": (
                patient_counts.max(axis=1) / totals
            ).to_numpy(float),
        }
    )
    return result.sort_values("cluster").reset_index(drop=True)


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
            "font.size": 10,
            "axes.titleweight": "bold",
            "savefig.dpi": 300,
        }
    )
    return plt


def plot_cluster_umap(
    output: Path,
    coordinates: np.ndarray,
    clusters: np.ndarray,
    sizes: pd.DataFrame,
    plt,
) -> list[Path]:
    cluster_order = sizes["cluster"].tolist()
    palette = plt.get_cmap("tab20")
    colors = {cluster: palette(i) for i, cluster in enumerate(cluster_order)}

    fig, axis = plt.subplots(figsize=(12.8, 8.8), constrained_layout=True)
    for cluster in cluster_order:
        rows = clusters == cluster
        size = int(rows.sum())
        fraction = size / len(clusters)
        axis.scatter(
            coordinates[rows, 0],
            coordinates[rows, 1],
            s=0.35,
            alpha=0.62,
            linewidths=0,
            rasterized=True,
            color=colors[cluster],
            label=f"{cluster}  n={size:,} ({fraction:.1%})",
        )
        x, y = np.median(coordinates[rows], axis=0)
        label = axis.text(
            x,
            y,
            cluster,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="black",
        )
        from matplotlib import patheffects

        label.set_path_effects(
            [patheffects.withStroke(linewidth=2.5, foreground="white")]
        )

    axis.set_title(
        "Senosain CyTOF · all-34-marker PCA graph · Leiden resolution 1.0",
        fontsize=14,
    )
    axis.set_xlabel("UMAP 1")
    axis.set_ylabel("UMAP 2")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.spines[:].set_visible(False)
    axis.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        markerscale=8,
        fontsize=9,
        ncol=1,
    )

    created: list[Path] = []
    for suffix in ("png", "pdf"):
        path = output / f"leiden_all34_cluster_umap.{suffix}"
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
    clustering_values = z_values.drop(columns="CD90").to_numpy(float)
    row_order = leaves_list(linkage(clustering_values, method="average"))
    ordered_clusters = z_values.index.to_numpy()[row_order].tolist()

    z = z_values.loc[ordered_clusters, display_markers].to_numpy(float)
    exact = median.loc[ordered_clusters, display_markers].to_numpy(float)
    size_index = sizes.set_index("cluster")
    row_labels = [
        f"{cluster}  |  n={int(size_index.loc[cluster, 'cells']):,}  "
        f"({size_index.loc[cluster, 'fraction_of_all_cells']:.1%})"
        for cluster in ordered_clusters
    ]

    fig = plt.figure(figsize=(31.0, 15.5), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[35, 4.5])
    axis = fig.add_subplot(grid[0, 0])
    bar_axis = fig.add_subplot(grid[0, 1])
    norm = TwoSlopeNorm(vmin=-2.5, vcenter=0.0, vmax=2.5)
    image = axis.imshow(
        np.clip(z, -2.5, 2.5),
        aspect="auto",
        cmap="RdBu_r",
        norm=norm,
        interpolation="nearest",
        rasterized=True,
    )

    xlabels = ["CD90*" if marker == "CD90" else marker for marker in display_markers]
    axis.set_xticks(np.arange(len(display_markers)), labels=xlabels)
    axis.tick_params(axis="x", labelrotation=58, labelsize=10, pad=3)
    for label in axis.get_xticklabels():
        label.set_ha("right")
    axis.set_yticks(np.arange(len(ordered_clusters)), labels=row_labels)
    axis.tick_params(axis="y", labelsize=10)
    axis.set_xlabel("Biomarker")
    axis.set_ylabel("Leiden cluster")

    group_start = 0
    group_colors = ["#3B82F6", "#10B981", "#F59E0B"]
    for (group_name, group_markers), color in zip(
        MARKER_GROUPS.items(), group_colors, strict=True
    ):
        group_end = group_start + len(group_markers)
        axis.plot(
            [group_start - 0.45, group_end - 0.55],
            [-1.18, -1.18],
            color=color,
            linewidth=5,
            clip_on=False,
        )
        axis.text(
            (group_start + group_end - 1) / 2,
            -1.52,
            group_name,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color=color,
            clip_on=False,
        )
        if group_end < len(display_markers):
            axis.axvline(group_end - 0.5, color="white", linewidth=2.5)
        group_start = group_end

    for row in range(exact.shape[0]):
        for column in range(exact.shape[1]):
            axis.text(
                column,
                row,
                f"{exact[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=7.0,
                color="white" if abs(z[row, column]) >= 1.20 else "black",
            )

    colorbar = fig.colorbar(image, ax=axis, orientation="horizontal", shrink=0.34, pad=0.10)
    colorbar.set_label("Marker-wise z-score of cluster median KMEANSLOW value")

    fractions = size_index.loc[ordered_clusters, "fraction_of_all_cells"].to_numpy(float)
    y = np.arange(len(ordered_clusters))
    bar_axis.barh(y, fractions, color="#5B6472", height=0.72)
    bar_axis.set_ylim(len(ordered_clusters) - 0.5, -0.5)
    bar_axis.set_yticks([])
    bar_axis.set_xlabel("Cells (%)")
    bar_axis.set_title("Cluster size", fontsize=11)
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
        "Leiden cluster biomarker profiles · exact KMEANSLOW medians printed in cells\n"
        "Color compares each marker across clusters; CD90* is display-only",
        fontsize=16,
        fontweight="bold",
    )

    created: list[Path] = []
    for suffix in ("png", "pdf"):
        path = output / f"leiden_all34_cluster_marker_heatmap.{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        created.append(path)
    plt.close(fig)
    return created


def main() -> None:
    args = parse_args()
    output = args.output.resolve()

    print("Loading frozen KMEANSLOW, PCA25, UMAP, and metadata", flush=True)
    baseline = np.load(BASELINE_PATH)
    pca = np.load(PCA_PATH)
    umap_coordinates = np.load(UMAP_PATH)
    metadata = pd.read_csv(METADATA_PATH, low_memory=False)
    markers = pd.read_csv(MARKERS_PATH)["marker"].astype(str).tolist()
    embedding_manifest = json.loads(EMBEDDING_MANIFEST_PATH.read_text())
    validate_inputs(
        baseline,
        pca,
        umap_coordinates,
        metadata,
        markers,
        embedding_manifest,
    )

    input_paths = [
        BASELINE_PATH,
        PCA_PATH,
        UMAP_PATH,
        EMBEDDING_MANIFEST_PATH,
        METADATA_PATH,
        MARKERS_PATH,
        Path(__file__).resolve(),
    ]
    input_hashes = {path.name: sha256(path) for path in input_paths}

    if output.exists():
        if not args.force:
            raise FileExistsError(
                f"Output directory already exists: {output}. Use --force to replace it."
            )
        shutil.rmtree(output)
    output.mkdir(parents=True)
    setup_plotting(output)

    # Import igraph only after writable cache directories exist; python-igraph
    # imports its plotting module even though this script uses matplotlib.
    import igraph as ig

    started = time.perf_counter()
    print("Building the all-34-marker PCA25 30-NN fuzzy graph", flush=True)
    connectivities, umap_module = build_graph(pca)
    graph = ig.Graph.Weighted_Adjacency(
        connectivities,
        mode="undirected",
        attr="weight",
        loops=False,
    )
    if graph.vcount() != len(metadata):
        raise AssertionError("Graph vertex count does not match the cells")

    print("Scanning Leiden resolution and seed stability", flush=True)
    raw_membership, resolution_scan, stability_scan = run_resolution_and_stability_scan(
        graph, ig
    )
    clusters, raw_to_named = stable_cluster_names(raw_membership)
    if np.unique(clusters).size != 18:
        raise AssertionError(
            f"Frozen resolution expected 18 clusters; found {np.unique(clusters).size}"
        )

    sizes = cluster_size_table(metadata, clusters)
    long, median, mean = summarize_profiles(baseline, markers, clusters)
    z_values = marker_wise_z(median)

    assignments = metadata[
        [
            "sampled_row_zero_based",
            "source_cell_id",
            "patient_id",
            "unit_key",
            "sample_rank_within_patient",
        ]
    ].copy()
    assignments["leiden_raw_label"] = raw_membership
    assignments["cluster"] = clusters
    assignments["UMAP1"] = umap_coordinates[:, 0]
    assignments["UMAP2"] = umap_coordinates[:, 1]

    assignments_path = output / "cluster_assignments.csv.gz"
    sizes_path = output / "cluster_sizes.csv"
    long_path = output / "cluster_kmeanslow_summary_long.csv.gz"
    median_path = output / "cluster_kmeanslow_median_wide.csv"
    mean_path = output / "cluster_kmeanslow_mean_wide.csv"
    z_path = output / "cluster_kmeanslow_median_marker_z.csv"
    resolution_path = output / "leiden_resolution_scan.csv"
    stability_path = output / "leiden_seed_stability.csv"

    assignments.to_csv(assignments_path, index=False, compression="gzip")
    sizes.to_csv(sizes_path, index=False)
    long.to_csv(long_path, index=False, compression="gzip")
    median.to_csv(median_path)
    mean.to_csv(mean_path)
    z_values.to_csv(z_path)
    resolution_scan.to_csv(resolution_path, index=False)
    stability_scan.to_csv(stability_path, index=False)

    print("Drawing cluster UMAP and exact-median heatmap", flush=True)
    plt = setup_plotting(output)
    plot_paths = [
        *plot_cluster_umap(output, umap_coordinates, clusters, sizes, plt),
        *plot_profile_heatmap(output, median, z_values, sizes, plt),
    ]
    elapsed = time.perf_counter() - started

    output_paths = [
        assignments_path,
        sizes_path,
        long_path,
        median_path,
        mean_path,
        z_path,
        resolution_path,
        stability_path,
        *plot_paths,
    ]
    manifest = {
        "schema_version": "senosain-all34-leiden-cluster-profiles-v1",
        "inputs": input_hashes,
        "cells": int(len(metadata)),
        "patients": int(metadata["patient_id"].nunique()),
        "clustering_input": {
            "representation": "PCA25 fitted from all 34 uniform KMEANSLOW markers",
            "markers": embedding_manifest["embedding_markers"],
            "markers_count": 34,
            "excluded_from_clustering": {
                "CD90": "not measured uniformly across all acquisitions"
            },
            "cell_labels_or_clinical_data_used": False,
            "umap_coordinates_used_for_clustering": False,
        },
        "graph": {
            "neighbors": N_NEIGHBORS,
            "metric": "euclidean",
            "method": "UMAP fuzzy simplicial set",
            "vertices": int(graph.vcount()),
            "undirected_edges": int(graph.ecount()),
            "seed": SEED,
        },
        "leiden": {
            "implementation": "python-igraph Graph.community_leiden",
            "objective": "modularity",
            "resolution": FROZEN_RESOLUTION,
            "iterations": LEIDEN_ITERATIONS,
            "seed": SEED,
            "clusters": int(np.unique(clusters).size),
            "cluster_names": "C00.. ordered by decreasing cell count",
            "raw_to_named_cluster": raw_to_named,
            "resolution_scan": resolution_path.name,
            "seed_stability": stability_path.name,
        },
        "profile_summary": {
            "markers": markers,
            "markers_count": 35,
            "scale": "KMEANSLOW values, before global PCA StandardScaler",
            "exact_statistics": ["mean", "sd", "q10", "q25", "median", "q75", "q90"],
            "heatmap_color": "marker-wise z-score of cluster medians",
            "heatmap_cell_text": "exact cluster median KMEANSLOW value",
            "CD90_display_only": True,
        },
        "runtime_seconds": elapsed,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "igraph": ig.__version__,
            "umap_learn": importlib.metadata.version("umap-learn"),
        },
        "outputs": {
            path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in output_paths
        },
    }
    manifest_path = output / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(
        f"DONE: {len(metadata):,} cells, {len(sizes)} Leiden clusters; "
        f"min/median/max cluster size={sizes.cells.min():,}/"
        f"{sizes.cells.median():,.0f}/{sizes.cells.max():,}; "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )
    print(output, flush=True)


if __name__ == "__main__":
    main()
