#!/usr/bin/env python3
"""Compute the frozen 31-metric battery for one three-platform benchmark.

This is a post-hoc evaluation runner.  Cell-type labels are never passed to
any alignment method; they enter here only after all method outputs have been
frozen.  Every method is evaluated on the same cells, the same 20 matched
dataset-balanced resamples, and UMAPs fitted with one common protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import tempfile
from itertools import combinations
from pathlib import Path

CACHE = Path(
    os.environ.get(
        "RAMP_EVALUATION_CACHE",
        str(Path(tempfile.gettempdir()) / "ramp-comparison-evaluation-cache"),
    )
)
for variable, subdirectory in (
    ("MPLCONFIGDIR", "matplotlib"),
    ("NUMBA_CACHE_DIR", "numba"),
    ("XDG_CACHE_HOME", "xdg"),
):
    destination = CACHE / subdirectory
    destination.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(variable, str(destination))
os.environ.setdefault("MPLBACKEND", "Agg")

import igraph as ig
import leidenalg
import numpy as np
import pandas as pd
import umap
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.stats import chi2
from sklearn.metrics import (
    adjusted_rand_score,
    balanced_accuracy_score,
    f1_score,
    normalized_mutual_info_score,
    silhouette_samples,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


CASES = ("bone_marrow", "tnbc", "melanoma", "lung_cancer")
SEEDS = (
    20220524,
    20230817,
    20200502,
    20230801,
    20190826,
    20220525,
    20230818,
    20200503,
    20230829,
    20190827,
    20137910,
    20349283,
    24918503,
    38298493,
    150,
    2940,
    10,
    302934245,
    3428,
    9123784,
)
INVALID_LABEL_PATTERN = re.compile(
    r"unannotated|ambiguous|non-joint|unknown|unclassified|technical",
    flags=re.IGNORECASE,
)

# The registry is intentionally identical to the previously frozen 31-metric
# battery, so historical HNSCC results and the four new benchmarks have the
# same metric vocabulary.
REGISTRY = (
    ("leiden_celltype_ari", "biology", "higher", "Leiden clusters versus joint cell type"),
    ("leiden_celltype_nmi", "biology", "higher", "Leiden clusters versus joint cell type"),
    ("leiden_dataset_ari", "mixing", "lower", "Leiden clusters versus dataset; negative ARI clipped to zero"),
    ("leiden_dataset_nmi", "mixing", "lower", "Leiden clusters versus dataset"),
    ("leiden_unit_ari", "mixing", "lower", "Leiden clusters versus donor/ROI; negative ARI clipped to zero"),
    ("leiden_unit_nmi", "mixing", "lower", "Leiden clusters versus donor/ROI"),
    ("louvain_celltype_ari", "biology", "higher", "Louvain clusters versus joint cell type"),
    ("louvain_celltype_nmi", "biology", "higher", "Louvain clusters versus joint cell type"),
    ("louvain_dataset_ari", "mixing", "lower", "Louvain clusters versus dataset; negative ARI clipped to zero"),
    ("louvain_dataset_nmi", "mixing", "lower", "Louvain clusters versus dataset"),
    ("louvain_unit_ari", "mixing", "lower", "Louvain clusters versus donor/ROI; negative ARI clipped to zero"),
    ("louvain_unit_nmi", "mixing", "lower", "Louvain clusters versus donor/ROI"),
    ("highdim_celltype_asw_macro", "biology", "higher", "Macro cell-type silhouette in corrected space"),
    ("highdim_dataset_asw_conditional", "mixing", "higher", "Mean 1-|dataset ASW| within cell type"),
    ("highdim_unit_asw_conditional", "mixing", "higher", "Mean 1-|donor/ROI ASW| within cell type"),
    ("celltype_knn_purity", "biology", "higher", "Same-cell-type fraction among 30-NN"),
    ("celltype_lisi_purity", "biology", "higher", "Normalized inverse cell-type LISI"),
    ("dataset_ilisi_norm", "mixing", "higher", "Normalized dataset iLISI among 30-NN"),
    ("unit_ilisi_norm", "mixing", "higher", "Normalized donor/ROI iLISI among 30-NN"),
    ("dataset_entropy_norm", "mixing", "higher", "Normalized dataset entropy among 30-NN"),
    ("conditional_kbet_acceptance", "mixing", "higher", "kBET-like acceptance conditional on cell type"),
    ("celltype_graph_connectivity", "biology", "higher", "Largest within-type graph component fraction"),
    ("crossplatform_knn_macro_f1", "transfer", "higher", "Leave-one-dataset-out 15-NN macro F1"),
    ("crossplatform_knn_balanced_accuracy", "transfer", "higher", "Leave-one-dataset-out 15-NN balanced accuracy"),
    ("native_knn_jaccard", "geometry", "higher", "Within-dataset 15-NN Jaccard versus source input"),
    ("crossplatform_mnn_coverage", "matching", "higher", "Coverage by cross-dataset mutual 5-NN"),
    ("cross_dataset_30nn_fraction", "mixing", "higher", "Cross-dataset fraction among all 30-NN"),
    ("conditional_cross_dataset_30nn_fraction", "mixing", "higher", "Cross-dataset 30-NN fraction among same-type neighbors"),
    ("umap_celltype_asw_macro", "visual", "higher", "Macro cell-type silhouette in common-protocol UMAP"),
    ("umap_dataset_abs_asw", "visual", "lower", "Absolute dataset silhouette in common-protocol UMAP"),
    ("umap_unit_abs_asw", "visual", "lower", "Absolute donor/ROI silhouette in common-protocol UMAP"),
)

UMAP_PROTOCOL = {
    "n_neighbors": 30,
    "min_dist": 0.15,
    "metric": "euclidean",
    "init": "spectral",
    "random_state": 42,
    "transform_seed": 42,
    "n_jobs": 1,
    "low_memory": True,
    "preprocessing": "pooled dimension-wise StandardScaler; no additional PCA",
}
EVALUATION_SCHEMA = "final_xu.comparison_evaluation.v2"


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode())
    digest.update(str(contiguous.dtype).encode())
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def sha256_json(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_run_provenance(
    case: str,
    output: Path,
    sources: dict[str, Path],
    inventory: pd.DataFrame,
    methods: list[str],
    seeds: tuple[int, ...],
    cells_per_dataset: int,
    label_column: str,
) -> tuple[str, dict[str, object], list[dict[str, object]]]:
    source_records = []
    for label, path in sources.items():
        source_records.append(
            {
                "label": label,
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    umap_records = []
    for method in methods:
        path = output / "umap" / "manifests" / f"{safe_name(method)}.json"
        if not path.exists():
            raise FileNotFoundError(f"missing validated UMAP manifest: {path}")
        record = json.loads(path.read_text())
        umap_records.append(
            {
                "method": method,
                "representation_sha256": record.get("representation_sha256"),
                "coordinate_sha256": record.get("coordinate_sha256"),
                "protocol": record.get("protocol"),
            }
        )
    inventory_records = inventory[
        [
            "method",
            "method_family",
            "variant",
            "role",
            "include_in_method_rank",
            "output_type",
            "n_dimensions_raw",
            "n_dimensions_evaluated",
        ]
    ].to_dict("records")
    config: dict[str, object] = {
        "schema_version": EVALUATION_SCHEMA,
        "benchmark": case,
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        "metric_registry": [list(row) for row in REGISTRY],
        "methods": methods,
        "inventory": inventory_records,
        "sampling_seeds": list(seeds),
        "cells_per_dataset_per_resample": int(cells_per_dataset),
        "label_column": label_column,
        "source_files": source_records,
        "umap": umap_records,
    }
    return sha256_json(config), config, source_records


def read_markers(path: Path) -> list[str]:
    markers = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(markers) < 2 or len(markers) != len(set(markers)):
        raise ValueError(f"invalid marker file: {path}")
    return markers


def read_ordered_csv(
    path: Path,
    expected_ids: np.ndarray,
    value_columns: list[str] | None = None,
    id_column: str = "cell_id",
) -> tuple[np.ndarray, list[str]]:
    frame = pd.read_csv(path, low_memory=False)
    if id_column not in frame:
        raise ValueError(f"{path}: missing {id_column}")
    ids = frame[id_column].astype(str).to_numpy()
    if not np.array_equal(ids, expected_ids):
        raise ValueError(f"{path}: cell order differs from frozen RAMP input")
    if value_columns is None:
        value_columns = [column for column in frame.columns if column != id_column]
    missing = [column for column in value_columns if column not in frame]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    values = frame[value_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: output contains non-finite values")
    return values, value_columns


def standardized(values: np.ndarray, method: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError(f"{method}: invalid representation shape {values.shape}")
    keep = np.isfinite(values).all(axis=0) & (np.std(values, axis=0) > 1e-10)
    if int(keep.sum()) < 2:
        raise ValueError(f"{method}: fewer than two usable dimensions")
    return StandardScaler().fit_transform(values[:, keep]).astype(np.float32)


def load_case(root: Path, case: str) -> tuple[pd.DataFrame, list[str], dict[str, np.ndarray], pd.DataFrame, dict[str, Path]]:
    case_root = root / case
    ramp_input = case_root / "ramp" / "alignment" / "input"
    metadata_path = ramp_input / "cell_metadata.csv.gz"
    marker_path = ramp_input / "markers.txt"
    raw_path = ramp_input / "raw.npy"
    metadata = pd.read_csv(metadata_path, low_memory=False)
    required = {"cell_id", "dataset", "unit_key", "celltype"}
    if missing := sorted(required - set(metadata.columns)):
        raise ValueError(f"{case}: frozen metadata missing {missing}")
    if metadata["cell_id"].astype(str).duplicated().any():
        raise ValueError(f"{case}: duplicate cell_id")
    ids = metadata["cell_id"].astype(str).to_numpy()
    markers = read_markers(marker_path)
    raw = np.asarray(np.load(raw_path, mmap_mode="r"), dtype=np.float64)
    if raw.shape != (len(metadata), len(markers)) or not np.isfinite(raw).all():
        raise ValueError(f"{case}: raw matrix is not aligned or finite")

    sources: dict[str, Path] = {
        "metadata": metadata_path,
        "markers": marker_path,
        "Unaligned": raw_path,
    }
    representations: dict[str, np.ndarray] = {"Unaligned": raw}
    rows: list[dict[str, object]] = [
        {
            "method": "Unaligned",
            "method_family": "Unaligned",
            "variant": "source_input",
            "role": "baseline",
            "include_in_method_rank": False,
            "output_type": "shared_marker_matrix",
        }
    ]

    ramp_path = case_root / "ramp" / "alignment" / "data" / "ramp_aligned.npy"
    ramp = np.asarray(np.load(ramp_path, mmap_mode="r"), dtype=np.float64)
    if ramp.shape != raw.shape or not np.isfinite(ramp).all():
        raise ValueError(f"{case}: invalid RAMP matrix")
    representations["RAMP"] = ramp
    sources["RAMP"] = ramp_path
    rows.append(
        {
            "method": "RAMP",
            "method_family": "RAMP",
            "variant": "ver2",
            "role": "proposed",
            "include_in_method_rank": True,
            "output_type": "shared_marker_matrix",
        }
    )

    cy_path = case_root / "cycombine" / "alignment" / "data" / "corrected_expression.csv.gz"
    cy, cy_columns = read_ordered_csv(cy_path, ids, markers)
    if cy_columns != markers:
        raise ValueError(f"{case}: cyCombine marker order mismatch")
    representations["cyCombine"] = cy
    sources["cyCombine"] = cy_path
    rows.append(
        {
            "method": "cyCombine",
            "method_family": "cyCombine",
            "variant": "formal",
            "role": "comparator",
            "include_in_method_rank": True,
            "output_type": "corrected_marker_matrix",
        }
    )

    fast_path = case_root / "fastmnn" / "alignment" / "data" / "fastmnn_corrected_embedding.csv.gz"
    fast, fast_columns = read_ordered_csv(fast_path, ids)
    if not all(column.startswith("fastMNN_") for column in fast_columns):
        raise ValueError(f"{case}: unexpected fastMNN output columns")
    representations["fastMNN"] = fast
    sources["fastMNN"] = fast_path
    rows.append(
        {
            "method": "fastMNN",
            "method_family": "fastMNN",
            "variant": "official_PCA_embedding",
            "role": "comparator",
            "include_in_method_rank": True,
            "output_type": "corrected_latent_embedding",
        }
    )

    for variant, label in (
        ("source_scale", "Harmony-direct[source_scale]"),
        ("adtnorm_fixed_arcsinh", "Harmony-direct[fixed_arcsinh]"),
    ):
        harmony_path = (
            case_root
            / "harmony"
            / variant
            / "alignment"
            / "data"
            / "harmony_adjusted_marker_coordinates.csv.gz"
        )
        harmony, columns = read_ordered_csv(harmony_path, ids, markers)
        if columns != markers:
            raise ValueError(f"{case}/{variant}: Harmony marker order mismatch")
        representations[label] = harmony
        sources[label] = harmony_path
        rows.append(
            {
                "method": label,
                "method_family": "Harmony-direct",
                "variant": variant,
                "role": "comparator_sensitivity" if variant != "source_scale" else "comparator",
                "include_in_method_rank": variant == "source_scale",
                "output_type": "adjusted_marker_coordinates",
            }
        )

    maxfuse_root = case_root / "maxfuse_archive"
    hub_paths = sorted(maxfuse_root.glob("hub_*/alignment/data/maxfuse_embedding_float64.npy"))
    if len(hub_paths) != 3:
        raise ValueError(f"{case}: expected three formal MaxFuse hub outputs, found {len(hub_paths)}")
    for embedding_path in hub_paths:
        hub = embedding_path.parents[2].name
        ids_path = embedding_path.with_name("cell_ids.npy")
        maxfuse_ids = np.load(ids_path, allow_pickle=True).astype(str)
        if not np.array_equal(maxfuse_ids, ids):
            raise ValueError(f"{case}/{hub}: MaxFuse cell order mismatch")
        values = np.asarray(np.load(embedding_path, mmap_mode="r"), dtype=np.float64)
        if len(values) != len(metadata) or values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError(f"{case}/{hub}: invalid MaxFuse embedding")
        label = f"MaxFuse[{hub.removeprefix('hub_')}]"
        representations[label] = values
        sources[label] = embedding_path
        rows.append(
            {
                "method": label,
                "method_family": "MaxFuse",
                "variant": hub,
                "role": "comparator",
                "include_in_method_rank": True,
                "output_type": "archive_iterative_GCCA_latent_embedding",
            }
        )

    inventory = pd.DataFrame(rows)
    inventory.insert(0, "benchmark", case)
    inventory["n_cells"] = len(metadata)
    inventory["n_dimensions_raw"] = inventory["method"].map(
        {method: values.shape[1] for method, values in representations.items()}
    )
    inventory["source_path"] = inventory["method"].map(
        {method: str(path.resolve()) for method, path in sources.items() if method in representations}
    )
    return metadata, markers, representations, inventory, sources


def sample_rows(metadata: pd.DataFrame, seed: int, cap: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    datasets = metadata["dataset"].astype(str).to_numpy()
    blocks = []
    for dataset in pd.unique(datasets):
        rows = np.flatnonzero(datasets == dataset)
        if len(rows) < cap:
            raise ValueError(f"dataset {dataset!r} has only {len(rows)} cells; requested {cap}")
        blocks.append(np.sort(rng.choice(rows, cap, replace=False)))
    return np.concatenate(blocks)


def knn(values: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    count = min(k + 1, len(values))
    distance, index = NearestNeighbors(n_neighbors=count, n_jobs=1).fit(values).kneighbors(values)
    target = min(k, len(values) - 1)
    clean_index = np.empty((len(values), target), dtype=index.dtype)
    clean_distance = np.empty((len(values), target), dtype=distance.dtype)
    for row in range(len(values)):
        keep = index[row] != row
        local_index = index[row][keep]
        local_distance = distance[row][keep]
        # In an extreme all-ties case sklearn may omit self from the returned
        # k+1 neighbours.  Then the first k entries are already all non-self.
        if len(local_index) < target:
            local_index = index[row][:target]
            local_distance = distance[row][:target]
        clean_index[row] = local_index[:target]
        clean_distance[row] = local_distance[:target]
    return clean_index, clean_distance


def graph_from_knn(index: np.ndarray, distance: np.ndarray, k: int = 15) -> sparse.csr_matrix:
    """Symmetric unweighted kNN graph used only for clustering/connectivity.

    A binary graph avoids numerical fragmentation caused by a single global
    exponential bandwidth when a corrected space contains both very dense and
    very diffuse regions.  Distance-based metrics still use the original
    corrected coordinates and neighbour indices.
    """
    local_index = index[:, :k]
    rows = np.repeat(np.arange(len(local_index)), local_index.shape[1])
    graph = sparse.csr_matrix(
        (np.ones(rows.size, dtype=np.float32), (rows, local_index.ravel())),
        shape=(len(local_index), len(local_index)),
    )
    return graph.maximum(graph.T)


def igraph_from_sparse(graph: sparse.csr_matrix) -> ig.Graph:
    upper = sparse.triu(graph, k=1).tocoo()
    answer = ig.Graph(
        n=graph.shape[0],
        edges=list(zip(upper.row.tolist(), upper.col.tolist())),
        directed=False,
    )
    answer.es["weight"] = upper.data.astype(float).tolist()
    return answer


def target_k_partition(
    graph: ig.Graph,
    target: int,
    seed: int,
    algorithm: str,
    iterations: int = 14,
) -> tuple[np.ndarray, float, int, bool]:
    """Deterministically tune graph resolution to the post-hoc target K.

    Labels only determine the desired number of clusters, exactly as in the
    historical battery.  If graph topology makes target K unattainable, the
    returned validity flag is false and that algorithm's ARI/NMI metrics are
    written as NA rather than silently scoring thousands of clusters against
    a seven-class target.
    """

    def fit(resolution: float) -> np.ndarray:
        if algorithm == "leiden":
            partition = leidenalg.find_partition(
                graph,
                leidenalg.RBConfigurationVertexPartition,
                weights="weight",
                resolution_parameter=float(resolution),
                seed=int(seed % (2**31 - 1)),
            )
        elif algorithm == "louvain":
            ig.set_random_number_generator(random.Random(int(seed)))
            partition = graph.community_multilevel(
                weights="weight",
                return_levels=False,
                resolution=float(resolution),
            )
        else:
            raise ValueError(f"unknown clustering algorithm: {algorithm}")
        return np.asarray(partition.membership, dtype=np.int32)

    low = 1e-5
    high = 8.0
    candidates: list[tuple[int, float, np.ndarray]] = []
    for resolution in (low, high):
        labels = fit(resolution)
        candidates.append((len(np.unique(labels)), resolution, labels))
    for _ in range(iterations):
        resolution = float(np.sqrt(low * high))
        labels = fit(resolution)
        count = len(np.unique(labels))
        candidates.append((count, resolution, labels))
        if count < target:
            low = resolution
        elif count > target:
            high = resolution
        else:
            # Continue locally to make tie-breaking stable when a plateau of
            # exact-K resolutions exists.
            high = resolution
    count, resolution, labels = min(
        candidates,
        key=lambda item: (abs(item[0] - target), abs(np.log(item[1]))),
    )
    tolerance = max(1, int(np.ceil(0.20 * target)))
    valid = abs(count - target) <= tolerance
    return labels, float(resolution), int(count), bool(valid)


def normalized_entropy(codes: np.ndarray, categories: int) -> float:
    if categories <= 1:
        return 1.0
    counts = np.bincount(codes, minlength=categories).astype(float)
    probabilities = counts[counts > 0] / counts.sum()
    return float(-(probabilities * np.log(probabilities)).sum() / np.log(categories))


def normalized_lisi(neighbor_codes: np.ndarray, categories: int) -> np.ndarray:
    proportions = np.stack(
        [(neighbor_codes == category).mean(axis=1) for category in range(categories)],
        axis=1,
    )
    lisi = 1.0 / np.sum(proportions**2, axis=1)
    if categories <= 1:
        return np.ones(len(lisi))
    return (lisi - 1.0) / (categories - 1.0)


def graph_connectivity(graph: sparse.csr_matrix, labels: np.ndarray) -> float:
    values = []
    for label in np.unique(labels):
        rows = np.flatnonzero(labels == label)
        if len(rows) < 2:
            continue
        _, components = connected_components(graph[rows][:, rows], directed=False)
        values.append(np.bincount(components).max() / len(rows))
    return float(np.mean(values)) if values else float("nan")


def safe_macro_silhouette(values: np.ndarray, labels: np.ndarray, seed: int, cap: int = 3000) -> float:
    labels = np.asarray(labels).astype(str)
    counts = pd.Series(labels).value_counts()
    levels = counts[counts >= 20].index.astype(str).to_numpy()
    if len(levels) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    chunks = []
    quota = max(20, cap // len(levels))
    for level in levels:
        rows = np.flatnonzero(labels == level)
        chunks.append(rng.choice(rows, min(quota, len(rows)), replace=False))
    selected = np.concatenate(chunks)
    if len(np.unique(labels[selected])) < 2 or len(selected) < 40:
        return float("nan")
    scores = silhouette_samples(values[selected], labels[selected], metric="euclidean")
    return float(np.mean([scores[labels[selected] == level].mean() for level in np.unique(labels[selected])]))


def safe_abs_silhouette(values: np.ndarray, labels: np.ndarray, seed: int, cap: int = 3000) -> float:
    score = safe_macro_silhouette(values, labels, seed, cap)
    return float(abs(score)) if np.isfinite(score) else score


def conditional_asw(values: np.ndarray, biological: np.ndarray, batch: np.ndarray, seed: int) -> float:
    scores = []
    for index, label in enumerate(np.unique(biological)):
        rows = np.flatnonzero(biological == label)
        if len(rows) < 40 or len(np.unique(batch[rows])) < 2:
            continue
        if len(rows) > 1000:
            rows = np.random.default_rng(seed + index).choice(rows, 1000, replace=False)
        local = silhouette_samples(values[rows], batch[rows])
        scores.append(float(np.mean(1.0 - np.abs(local))))
    return float(np.mean(scores)) if scores else float("nan")


def conditional_kbet(
    index: np.ndarray,
    celltype: np.ndarray,
    dataset: np.ndarray,
    eligible: np.ndarray,
    k: int = 30,
) -> float:
    dataset_codes, dataset_levels = pd.factorize(dataset, sort=True)
    accepted = []
    for label in np.unique(celltype[eligible]):
        rows = np.flatnonzero(eligible & (celltype == label))
        if len(rows) <= k:
            continue
        expected = np.bincount(dataset_codes[rows], minlength=len(dataset_levels)).astype(float)
        expected /= expected.sum()
        active = expected > 0
        degrees = int(active.sum() - 1)
        if degrees < 1:
            continue
        for row in rows:
            same_type = index[row][eligible[index[row]] & (celltype[index[row]] == label)][:k]
            if len(same_type) < k:
                continue
            observed = np.bincount(dataset_codes[same_type], minlength=len(dataset_levels)).astype(float)
            target = k * expected
            statistic = np.sum((observed[active] - target[active]) ** 2 / target[active])
            accepted.append(chi2.sf(statistic, degrees) >= 0.05)
    return float(np.mean(accepted)) if accepted else float("nan")


def crossplatform_transfer(
    values: np.ndarray,
    dataset: np.ndarray,
    labels: np.ndarray,
    k: int = 15,
) -> tuple[float, float]:
    predicted = np.empty(len(labels), dtype=object)
    for query_dataset in np.unique(dataset):
        query = np.flatnonzero(dataset == query_dataset)
        reference = np.flatnonzero(dataset != query_dataset)
        neighbors = NearestNeighbors(n_neighbors=min(k, len(reference)), n_jobs=1).fit(
            values[reference]
        ).kneighbors(values[query], return_distance=False)
        reference_labels = labels[reference]
        for local_index, row in enumerate(query):
            levels, counts = np.unique(reference_labels[neighbors[local_index]], return_counts=True)
            predicted[row] = levels[np.argmax(counts)]
    return (
        float(f1_score(labels, predicted, average="macro")),
        float(balanced_accuracy_score(labels, predicted)),
    )


def neighbor_jaccard(native: np.ndarray, corrected: np.ndarray, dataset: np.ndarray, k: int = 15) -> float:
    values = []
    for level in np.unique(dataset):
        rows = np.flatnonzero(dataset == level)
        before, _ = knn(native[rows], k)
        after, _ = knn(corrected[rows], k)
        values.extend(
            len(set(left) & set(right)) / len(set(left) | set(right))
            for left, right in zip(before, after)
        )
    return float(np.mean(values))


def mnn_coverage(values: np.ndarray, dataset: np.ndarray, k: int = 5) -> float:
    scores = []
    for left, right in combinations(np.unique(dataset), 2):
        left_rows = np.flatnonzero(dataset == left)
        right_rows = np.flatnonzero(dataset == right)
        left_to_right = NearestNeighbors(n_neighbors=min(k, len(right_rows)), n_jobs=1).fit(
            values[right_rows]
        ).kneighbors(values[left_rows], return_distance=False)
        right_to_left = NearestNeighbors(n_neighbors=min(k, len(left_rows)), n_jobs=1).fit(
            values[left_rows]
        ).kneighbors(values[right_rows], return_distance=False)
        reverse = [set(item) for item in right_to_left]
        covered_left = np.zeros(len(left_rows), dtype=bool)
        covered_right = np.zeros(len(right_rows), dtype=bool)
        for left_index, neighbors in enumerate(left_to_right):
            for right_index in neighbors:
                if left_index in reverse[right_index]:
                    covered_left[left_index] = True
                    covered_right[right_index] = True
        scores.append((covered_left.sum() + covered_right.sum()) / (len(left_rows) + len(right_rows)))
    return float(np.mean(scores)) if scores else float("nan")


def fit_or_load_umap(
    values: np.ndarray,
    method: str,
    output: Path,
    allow_fit: bool = True,
) -> np.ndarray:
    coordinates_dir = output / "umap" / "coordinates"
    manifest_dir = output / "umap" / "manifests"
    coordinates_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_name(method)
    coordinate_path = coordinates_dir / f"{stem}.npy"
    csv_path = coordinates_dir / f"{stem}.csv.gz"
    manifest_path = manifest_dir / f"{stem}.json"
    representation_sha256 = sha256_array(values)
    if coordinate_path.exists() and manifest_path.exists():
        prior = json.loads(manifest_path.read_text())
        if (
            prior.get("method") == method
            and prior.get("representation_sha256") == representation_sha256
            and prior.get("protocol") == UMAP_PROTOCOL
        ):
            coordinates = np.load(coordinate_path)
            if (
                coordinates.shape == (len(values), 2)
                and np.isfinite(coordinates).all()
                and prior.get("coordinate_sha256") == sha256_array(coordinates)
            ):
                return coordinates.astype(np.float32, copy=False)
    if not allow_fit:
        raise RuntimeError(
            f"--skip-umap requested, but the cached UMAP for {method!r} "
            "does not match the current representation and protocol"
        )
    print(f"[{output.name}] common-protocol UMAP: {method}", flush=True)
    coordinates = umap.UMAP(
        **{key: value for key, value in UMAP_PROTOCOL.items() if key != "preprocessing"}
    ).fit_transform(values)
    coordinates = np.asarray(coordinates, dtype=np.float32)
    np.save(coordinate_path, coordinates)
    pd.DataFrame(
        {
            "row_id": np.arange(len(coordinates), dtype=np.int64),
            "UMAP1": coordinates[:, 0],
            "UMAP2": coordinates[:, 1],
        }
    ).to_csv(csv_path, index=False, compression="gzip")
    manifest_path.write_text(
        json.dumps(
            {
                "method": method,
                "representation_sha256": representation_sha256,
                "coordinate_sha256": sha256_array(coordinates),
                "n_cells": len(values),
                "n_dimensions": values.shape[1],
                "protocol": UMAP_PROTOCOL,
                "labels_used_for_fit": False,
            },
            indent=2,
        )
        + "\n"
    )
    return coordinates


def evaluate_one(
    values: np.ndarray,
    native: np.ndarray,
    celltype: np.ndarray,
    dataset: np.ndarray,
    unit: np.ndarray,
    valid_joint: np.ndarray,
    seed: int,
    coordinates: np.ndarray,
) -> dict[str, float]:
    index, distance = knn(values, 90)
    graph_sparse = graph_from_knn(index, distance, 15)
    graph = igraph_from_sparse(graph_sparse)
    target = max(2, len(np.unique(celltype[valid_joint])))
    leiden, leiden_resolution, n_leiden, leiden_valid = target_k_partition(
        graph, target, seed, "leiden"
    )
    louvain, louvain_resolution, n_louvain, louvain_valid = target_k_partition(
        graph, target, seed, "louvain"
    )

    def cluster_metric(valid: bool, function, left: np.ndarray, right: np.ndarray) -> float:
        return float(function(left, right)) if valid else float("nan")

    def clipped_cluster_ari(valid: bool, left: np.ndarray, right: np.ndarray) -> float:
        return max(0.0, float(adjusted_rand_score(left, right))) if valid else float("nan")

    dataset_codes, dataset_levels = pd.factorize(dataset, sort=True)
    unit_codes, unit_levels = pd.factorize(unit, sort=True)
    celltype_codes, celltype_levels = pd.factorize(celltype, sort=True)
    top = index[:, :30]
    same_type = celltype[top] == celltype[:, None]
    conditional_cross = (dataset[top] != dataset[:, None]) & same_type
    same_type_count = same_type.sum(axis=1)
    conditional_fraction = np.divide(
        conditional_cross.sum(axis=1),
        same_type_count,
        out=np.full(len(values), np.nan),
        where=same_type_count > 0,
    )
    macro_f1, balanced_accuracy = crossplatform_transfer(
        values[valid_joint], dataset[valid_joint], celltype[valid_joint]
    )
    result = {
        "target_cluster_count": float(target),
        "n_leiden_clusters": float(n_leiden),
        "n_louvain_clusters": float(n_louvain),
        "leiden_resolution": float(leiden_resolution),
        "louvain_resolution": float(louvain_resolution),
        "leiden_cluster_count_error": float(abs(n_leiden - target)),
        "louvain_cluster_count_error": float(abs(n_louvain - target)),
        "leiden_cluster_count_valid": float(leiden_valid),
        "louvain_cluster_count_valid": float(louvain_valid),
        "leiden_celltype_ari": cluster_metric(
            leiden_valid, adjusted_rand_score, celltype[valid_joint], leiden[valid_joint]
        ),
        "leiden_celltype_nmi": cluster_metric(
            leiden_valid, normalized_mutual_info_score, celltype[valid_joint], leiden[valid_joint]
        ),
        "leiden_dataset_ari": clipped_cluster_ari(leiden_valid, dataset, leiden),
        "leiden_dataset_nmi": cluster_metric(
            leiden_valid, normalized_mutual_info_score, dataset, leiden
        ),
        "leiden_unit_ari": clipped_cluster_ari(leiden_valid, unit, leiden),
        "leiden_unit_nmi": cluster_metric(
            leiden_valid, normalized_mutual_info_score, unit, leiden
        ),
        "louvain_celltype_ari": cluster_metric(
            louvain_valid, adjusted_rand_score, celltype[valid_joint], louvain[valid_joint]
        ),
        "louvain_celltype_nmi": cluster_metric(
            louvain_valid, normalized_mutual_info_score, celltype[valid_joint], louvain[valid_joint]
        ),
        "louvain_dataset_ari": clipped_cluster_ari(louvain_valid, dataset, louvain),
        "louvain_dataset_nmi": cluster_metric(
            louvain_valid, normalized_mutual_info_score, dataset, louvain
        ),
        "louvain_unit_ari": clipped_cluster_ari(louvain_valid, unit, louvain),
        "louvain_unit_nmi": cluster_metric(
            louvain_valid, normalized_mutual_info_score, unit, louvain
        ),
        "highdim_celltype_asw_macro": safe_macro_silhouette(values[valid_joint], celltype[valid_joint], seed),
        "highdim_dataset_asw_conditional": conditional_asw(values[valid_joint], celltype[valid_joint], dataset[valid_joint], seed),
        "highdim_unit_asw_conditional": conditional_asw(values[valid_joint], celltype[valid_joint], unit[valid_joint], seed + 1),
        "celltype_knn_purity": float(np.nanmean(same_type[valid_joint])),
        "celltype_lisi_purity": float(
            np.mean(1.0 - normalized_lisi(celltype_codes[top][valid_joint], len(celltype_levels)))
        ),
        "dataset_ilisi_norm": float(np.mean(normalized_lisi(dataset_codes[top], len(dataset_levels)))),
        "unit_ilisi_norm": float(np.mean(normalized_lisi(unit_codes[top], len(unit_levels)))),
        "dataset_entropy_norm": float(
            np.mean([normalized_entropy(row, len(dataset_levels)) for row in dataset_codes[top]])
        ),
        "conditional_kbet_acceptance": conditional_kbet(index, celltype, dataset, valid_joint),
        "celltype_graph_connectivity": graph_connectivity(
            graph_sparse[valid_joint][:, valid_joint], celltype[valid_joint]
        ),
        "crossplatform_knn_macro_f1": macro_f1,
        "crossplatform_knn_balanced_accuracy": balanced_accuracy,
        "native_knn_jaccard": neighbor_jaccard(native, values, dataset),
        "crossplatform_mnn_coverage": mnn_coverage(values, dataset),
        "cross_dataset_30nn_fraction": float(np.mean(dataset[top] != dataset[:, None])),
        "conditional_cross_dataset_30nn_fraction": float(
            np.nanmean(conditional_fraction[valid_joint])
        ),
        "umap_celltype_asw_macro": safe_macro_silhouette(
            coordinates[valid_joint], celltype[valid_joint], seed + 2
        ),
        "umap_dataset_abs_asw": safe_abs_silhouette(coordinates, dataset, seed + 3),
        "umap_unit_abs_asw": safe_abs_silhouette(coordinates, unit, seed + 4),
    }
    return {key: float(value) for key, value in result.items()}


def write_atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def summarize(replicates: pd.DataFrame, inventory: pd.DataFrame, output: Path) -> None:
    registry = pd.DataFrame(REGISTRY, columns=["metric", "family", "direction", "definition"])
    registry.to_csv(output / "metric_registry.csv", index=False)
    meta = inventory.set_index("method")
    summary_rows = []
    rank_rows = []
    for metric, family, direction, definition in REGISTRY:
        for method, block in replicates.groupby("method", sort=False):
            summary_rows.append(
                {
                    "metric": metric,
                    "family": family,
                    "direction": direction,
                    "definition": definition,
                    "method": method,
                    "method_family": meta.loc[method, "method_family"],
                    "variant": meta.loc[method, "variant"],
                    "role": meta.loc[method, "role"],
                    "mean": block[metric].mean(),
                    "sd": block[metric].std(ddof=1),
                    "n_valid_resamples": int(block[metric].notna().sum()),
                }
            )
        eligible = [
            method
            for method in replicates["method"].unique()
            if bool(meta.loc[method, "include_in_method_rank"])
        ]
        means = replicates[replicates["method"].isin(eligible)].groupby("method")[metric].mean()
        ranks = means.rank(ascending=(direction == "lower"), method="average")
        for method in means.index:
            rank_rows.append(
                {
                    "metric": metric,
                    "family": family,
                    "direction": direction,
                    "method": method,
                    "method_family": meta.loc[method, "method_family"],
                    "variant": meta.loc[method, "variant"],
                    "mean": means[method],
                    "rank": ranks[method],
                }
            )
    pd.DataFrame(summary_rows).to_csv(output / "metric_battery_summary.csv", index=False)
    ranks = pd.DataFrame(rank_rows)
    ranks.to_csv(output / "metric_method_ranks.csv", index=False)
    (
        ranks.groupby(["method", "method_family", "variant"], as_index=False)
        .agg(mean_rank=("rank", "mean"), median_rank=("rank", "median"), n_metrics=("metric", "nunique"))
        .sort_values(["mean_rank", "method"])
        .to_csv(output / "method_mean_rank_across_31_metrics.csv", index=False)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=CASES)
    parser.add_argument(
        "--benchmark-results",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--cells-per-dataset", type=int, default=5000)
    parser.add_argument("--n-resamples", type=int, default=20)
    parser.add_argument("--skip-umap", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.n_resamples <= len(SEEDS):
        raise ValueError(f"n-resamples must be 1..{len(SEEDS)}")
    if args.cells_per_dataset < 100:
        raise ValueError("cells-per-dataset must be at least 100")

    benchmark_results = args.benchmark_results.resolve()
    output = (args.output_root / args.case).resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata, markers, raw_representations, inventory, sources = load_case(
        benchmark_results, args.case
    )
    inventory.to_csv(output / "method_inventory.csv", index=False)
    standardized_representations = {
        method: standardized(values, method)
        for method, values in raw_representations.items()
    }
    inventory["n_dimensions_evaluated"] = inventory["method"].map(
        {method: values.shape[1] for method, values in standardized_representations.items()}
    )
    inventory.to_csv(output / "method_inventory.csv", index=False)

    seeds = SEEDS[: args.n_resamples]
    label_column = (
        "celltype_joint_evaluation"
        if "celltype_joint_evaluation" in metadata.columns
        else "celltype"
    )

    coordinates: dict[str, np.ndarray] = {}
    for method, values in standardized_representations.items():
        coordinates[method] = fit_or_load_umap(
            values,
            method,
            output,
            allow_fit=not args.skip_umap,
        )

    method_order = list(standardized_representations)
    run_fingerprint, run_config, source_records = build_run_provenance(
        args.case,
        output,
        sources,
        inventory,
        method_order,
        seeds,
        args.cells_per_dataset,
        label_column,
    )

    partial_path = output / "metric_battery_replicates.partial.csv"
    checkpoint_path = output / "checkpoint_manifest.json"
    if partial_path.exists():
        if not checkpoint_path.exists():
            raise RuntimeError(
                f"refusing to resume {partial_path}: checkpoint manifest is missing"
            )
        prior_checkpoint = json.loads(checkpoint_path.read_text())
        if prior_checkpoint.get("run_fingerprint") != run_fingerprint:
            raise RuntimeError(
                "refusing to mix an existing partial result with changed inputs, "
                "methods, UMAP coordinates, seeds, or evaluation code"
            )
    checkpoint_path.write_text(
        json.dumps(
            {"run_fingerprint": run_fingerprint, "configuration": run_config},
            indent=2,
        )
        + "\n"
    )

    records: list[dict[str, object]] = []
    if partial_path.exists():
        partial = pd.read_csv(partial_path)
        if "run_fingerprint" not in partial:
            raise RuntimeError(f"refusing legacy partial without run fingerprint: {partial_path}")
        if set(partial["run_fingerprint"].astype(str)) != {run_fingerprint}:
            raise RuntimeError(f"partial rows have a different run fingerprint: {partial_path}")
        records = partial.to_dict("records")
    completed = {(str(row["method"]), int(row["sampling_seed"])) for row in records}
    native = standardized_representations["Unaligned"]
    for seed_index, seed in enumerate(seeds, start=1):
        rows = sample_rows(metadata, seed, args.cells_per_dataset)
        local = metadata.iloc[rows].reset_index(drop=True)
        celltype = local[label_column].fillna("Unannotated").astype(str).to_numpy()
        dataset = local["dataset"].astype(str).to_numpy()
        unit = local["unit_key"].astype(str).to_numpy()
        if "joint_evaluation" in local.columns:
            joint_flag = local["joint_evaluation"].astype(bool).to_numpy()
        else:
            joint_flag = np.asarray(
                [not bool(INVALID_LABEL_PATTERN.search(label)) for label in celltype],
                dtype=bool,
            )
        counts = pd.Series(celltype[joint_flag]).value_counts()
        present = (
            pd.DataFrame({"celltype": celltype[joint_flag], "dataset": dataset[joint_flag]})
            .groupby("celltype")
            .dataset.nunique()
        )
        joint_levels = present[(present >= 2) & (counts >= 30)].index.astype(str).to_numpy()
        valid_joint = joint_flag & np.isin(celltype, joint_levels)
        if len(joint_levels) < 2:
            raise ValueError(f"{args.case}/seed={seed}: fewer than two evaluable joint labels")
        for method, values in standardized_representations.items():
            if (method, seed) in completed:
                continue
            print(
                f"[{args.case}] resample {seed_index}/{len(seeds)}: {method}",
                flush=True,
            )
            result = evaluate_one(
                values[rows],
                native[rows],
                celltype,
                dataset,
                unit,
                valid_joint,
                seed,
                coordinates[method][rows],
            )
            records.append(
                {
                    "benchmark": args.case,
                    "method": method,
                    "sampling_seed": seed,
                    "run_fingerprint": run_fingerprint,
                    "n_cells": len(rows),
                    "n_joint_celltypes": len(joint_levels),
                    **result,
                }
            )
            completed.add((method, seed))
            write_atomic_csv(pd.DataFrame(records), partial_path)

    replicates = pd.DataFrame(records)
    expected = {(method, seed) for method in standardized_representations for seed in seeds}
    observed = {(str(row.method), int(row.sampling_seed)) for row in replicates.itertuples()}
    if expected != observed:
        raise RuntimeError(
            f"incomplete metric grid: missing={sorted(expected-observed)[:5]}, "
            f"extra={sorted(observed-expected)[:5]}"
        )
    replicates["_method_order"] = replicates["method"].map(
        {method: index for index, method in enumerate(method_order)}
    )
    replicates = replicates.sort_values(["sampling_seed", "_method_order"]).drop(
        columns="_method_order"
    )
    replicates.to_csv(output / "metric_battery_replicates.csv", index=False)
    summarize(replicates, inventory, output)

    pd.DataFrame(source_records).to_csv(output / "input_output_file_hashes.csv", index=False)
    manifest = {
        "schema_version": EVALUATION_SCHEMA,
        "run_fingerprint": run_fingerprint,
        "benchmark": args.case,
        "n_cells_full": len(metadata),
        "n_markers": len(markers),
        "markers": markers,
        "methods": method_order,
        "metric_count": len(REGISTRY),
        "metric_registry": [row[0] for row in REGISTRY],
        "sampling_seeds": list(seeds),
        "cells_per_dataset_per_resample": args.cells_per_dataset,
        "label_column": label_column,
        "celltype_metrics_use_joint_labels_present_in_at_least_two_datasets": True,
        "labels_used_for_alignment": False,
        "labels_used_posthoc_for_evaluation": True,
        "representation_preprocessing": "pooled dimension-wise StandardScaler for every method; no evaluation PCA",
        "common_umap_protocol": UMAP_PROTOCOL,
        "maxfuse_policy": "all three frozen maxfuse_archive hub orientations reported separately",
        "harmony_policy": "source_scale is primary; fixed_arcsinh is a preprocessing sensitivity",
        "cytofruv": "NOT_APPLICABLE: no cross-batch replicated reference specimen in this benchmark",
        "paired_metrics_omitted": ["FOSCTTM", "FOSKNN", "true-pair distance"],
        "omission_reason": "No same-cell cross-platform ground truth is available.",
        "python": sys.version,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    partial_path.unlink(missing_ok=True)
    print(f"[{args.case}] complete: {len(replicates)} method-resample rows", flush=True)


if __name__ == "__main__":
    main()
