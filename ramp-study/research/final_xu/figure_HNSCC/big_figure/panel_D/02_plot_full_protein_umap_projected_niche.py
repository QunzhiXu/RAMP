#!/usr/bin/env python3
"""Compare the transferred niche score on shared- and full-protein CD8 UMAPs.

The right-hand embedding uses every biological Rahim CITE-seq ADT: the seven
isotype-control channels are removed and all 130 remaining proteins are kept.
No marker (including LAG3) is held out.  The transferred spatial-niche score is
used only after fitting the embeddings, as the plotting colour.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
HNSCC_ROOT = HERE.parent.parent
ADT_H5AD = (
    HNSCC_ROOT
    / "sampled_data/rahim_citeseq/complete/GSE212797_adata_ADT.h5ad"
)
PROJECTION = (
    HNSCC_ROOT
    / "biomarker_discovery/projection/rahim_tumor_cd8_projected_scores.csv.gz"
)
SHARED_COORDINATES = (
    HNSCC_ROOT
    / "biomarker_discovery/projection/rahim_tumor_cd8_transfer_program_umap_coordinates.csv.gz"
)
CACHE = HERE / ".cache"
for variable, child in (
    ("MPLCONFIGDIR", "matplotlib"),
    ("NUMBA_CACHE_DIR", "numba"),
    ("XDG_CACHE_HOME", "xdg"),
    ("JOBLIB_TEMP_FOLDER", "joblib"),
    ("TMPDIR", "tmp"),
):
    path = CACHE / child
    path.mkdir(parents=True, exist_ok=True)
    os.environ[variable] = str(path)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from umap import UMAP


EXPECTED_CELL_N = 4_788
EXPECTED_ADT_N = 137
EXPECTED_ISOTYPE_N = 7
EXPECTED_BIOLOGICAL_N = 130
UMAP_SEED = 20260824
N_NEIGHBORS = 30
MIN_DIST = 0.25
PCA_COMPONENTS = 30
INK = "#182027"
SCORE_CMAP = LinearSegmentedColormap.from_list(
    "transferred_niche",
    ["#253494", "#2C7FB8", "#7FCDBB", "#FEC44F", "#D7301F"],
)


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode() if isinstance(value, bytes) else str(value) for value in values],
        dtype=object,
    )


def read_csr(group: h5py.Group, n_columns: int) -> sparse.csr_matrix:
    return sparse.csr_matrix(
        (group["data"][:], group["indices"][:], group["indptr"][:]),
        shape=(len(group["indptr"]) - 1, n_columns),
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv_gz(frame: pd.DataFrame, path: Path) -> None:
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", fileobj=raw_handle, mode="wb", mtime=0) as handle:
            frame.to_csv(handle, index=False)


def clean_axis(axis: plt.Axes) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_aspect("equal", adjustable="datalim")
    for spine in axis.spines.values():
        spine.set_visible(False)


def neighborhood_summary(coordinates: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    neighbors = NearestNeighbors(n_neighbors=N_NEIGHBORS + 1).fit(coordinates)
    indices = neighbors.kneighbors(return_distance=False)[:, 1:]
    local_mean = scores[indices].mean(axis=1)
    return {
        "score_vs_30nn_mean_spearman": float(spearmanr(scores, local_mean).statistic),
        "mean_absolute_30nn_score_difference": float(
            np.mean(np.abs(scores[:, None] - scores[indices]))
        ),
    }


def main() -> None:
    projection = pd.read_csv(PROJECTION, dtype={"cell_uid": str, "donor": str})
    required = {"cell_uid", "donor", "projected_rank_within_rahim_donor"}
    if not required.issubset(projection.columns):
        raise KeyError(f"Projection is missing {sorted(required - set(projection.columns))}")
    if len(projection) != EXPECTED_CELL_N or projection.cell_uid.duplicated().any():
        raise AssertionError("The formal 4,788-cell projection has changed")
    projection = projection.sort_values(["donor", "cell_uid"], kind="stable").reset_index(drop=True)

    with h5py.File(ADT_H5AD, "r") as handle:
        cell_ids = decode(handle["obs/_index"][:])
        markers = decode(handle["var/_index"][:])
        raw = read_csr(handle["X"], len(markers))
    if len(markers) != EXPECTED_ADT_N:
        raise AssertionError(f"Expected 137 ADTs; observed {len(markers)}")
    if len(set(cell_ids)) != len(cell_ids):
        raise AssertionError("Rahim ADT cell IDs are not unique")

    row_by_cell = pd.Series(np.arange(len(cell_ids), dtype=int), index=cell_ids)
    rows = row_by_cell.reindex(projection.cell_uid).to_numpy()
    if pd.isna(rows).any():
        raise AssertionError("Some projected cells are absent from the complete ADT object")
    rows = rows.astype(int)

    isotype = np.asarray(["isotype" in marker.lower() for marker in markers])
    if int(isotype.sum()) != EXPECTED_ISOTYPE_N:
        raise AssertionError(f"Expected seven isotype controls; observed {int(isotype.sum())}")
    biological_markers = markers[~isotype]
    if len(biological_markers) != EXPECTED_BIOLOGICAL_N:
        raise AssertionError("Expected 130 biological ADTs after removing isotypes")
    if "CD223--LAG-3" not in set(biological_markers):
        raise AssertionError("LAG3 must remain in the full-protein embedding")

    counts = raw[rows, :][:, ~isotype].toarray().astype(np.float64)
    if not np.all(counts >= 0):
        raise AssertionError("ADT counts must be nonnegative")

    # Protein-wise CLR used for CITE-seq ADT data: log1p count minus the
    # within-cell mean log1p count across all 130 biological proteins.
    log_counts = np.log1p(counts)
    clr = log_counts - log_counts.mean(axis=1, keepdims=True)
    scaled = StandardScaler().fit_transform(clr)
    pca = PCA(n_components=PCA_COMPONENTS, random_state=UMAP_SEED)
    pcs = pca.fit_transform(scaled)
    full_coordinates = UMAP(
        n_neighbors=N_NEIGHBORS,
        min_dist=MIN_DIST,
        metric="euclidean",
        random_state=UMAP_SEED,
        n_jobs=1,
    ).fit_transform(pcs)

    old = pd.read_csv(SHARED_COORDINATES, dtype={"cell_uid": str, "donor": str})
    old = old[["cell_uid", "donor", "UMAP1", "UMAP2"]]
    old = projection[["cell_uid", "donor"]].merge(
        old,
        on=["cell_uid", "donor"],
        how="left",
        validate="one_to_one",
    )
    if old[["UMAP1", "UMAP2"]].isna().any().any():
        raise AssertionError("Shared-protein coordinates do not cover all projected cells")

    scores = projection["projected_rank_within_rahim_donor"].to_numpy(float)
    order = np.argsort(scores, kind="stable")
    shared_coordinates = old[["UMAP1", "UMAP2"]].to_numpy(float)

    output = projection[["cell_uid", "donor", "projected_rank_within_rahim_donor"]].copy()
    output["UMAP1_full130"] = full_coordinates[:, 0]
    output["UMAP2_full130"] = full_coordinates[:, 1]
    coordinates_path = HERE / "full130_protein_umap_coordinates.csv.gz"
    write_csv_gz(output, coordinates_path)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.25))
    panels = (
        (axes[0], shared_coordinates, "22 shared proteins", "RAMP-aligned shared panel"),
        (axes[1], full_coordinates, "130 CITE-seq proteins", "All biological ADTs"),
    )
    scatter = None
    for axis, coordinates, title, subtitle in panels:
        scatter = axis.scatter(
            coordinates[order, 0],
            coordinates[order, 1],
            c=scores[order],
            cmap=SCORE_CMAP,
            vmin=0.0,
            vmax=1.0,
            s=8.0,
            alpha=0.90,
            linewidths=0,
            rasterized=True,
        )
        axis.set_title(title, loc="left", fontsize=13.0, weight="bold", color=INK, pad=15)
        axis.text(
            0.0,
            1.005,
            subtitle,
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.7,
            color="#59636C",
        )
        clean_axis(axis)

    colorbar = fig.colorbar(scatter, ax=axes, fraction=0.026, pad=0.018)
    colorbar.set_ticks([0.0, 0.25, 0.50, 0.75, 1.0])
    colorbar.set_label("Transferred niche-score rank", fontsize=9.5, labelpad=8)
    colorbar.ax.tick_params(labelsize=8.3, length=2.5)
    colorbar.outline.set_linewidth(0.6)
    fig.suptitle(
        "Transferred spatial-niche score on Rahim tumour CD8 T cells",
        x=0.045,
        y=0.995,
        ha="left",
        fontsize=15,
        weight="bold",
        color=INK,
    )
    fig.text(
        0.045,
        0.018,
        "Same 4,788 cells and scores · score not used to fit either UMAP · full-protein: CLR → scale → PCA30 → UMAP",
        ha="left",
        va="bottom",
        fontsize=8.8,
        color="#4F5962",
    )
    fig.subplots_adjust(left=0.045, right=0.910, bottom=0.070, top=0.875, wspace=0.12)
    comparison_png = HERE / "comparison_shared22_vs_full130_protein_umap.png"
    comparison_pdf = HERE / "comparison_shared22_vs_full130_protein_umap.pdf"
    fig.savefig(comparison_png, dpi=320, facecolor="white")
    fig.savefig(comparison_pdf, facecolor="white")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.3, 5.3))
    scatter = axis.scatter(
        full_coordinates[order, 0],
        full_coordinates[order, 1],
        c=scores[order],
        cmap=SCORE_CMAP,
        vmin=0.0,
        vmax=1.0,
        s=8.8,
        alpha=0.91,
        linewidths=0,
        rasterized=True,
    )
    clean_axis(axis)
    colorbar = fig.colorbar(scatter, ax=axis, fraction=0.045, pad=0.020)
    colorbar.set_ticks([0.0, 0.25, 0.50, 0.75, 1.0])
    colorbar.set_label("Transferred niche-score rank", fontsize=9.5, labelpad=8)
    colorbar.ax.tick_params(labelsize=8.3, length=2.5)
    colorbar.outline.set_linewidth(0.6)
    fig.suptitle(
        "SPATIAL-NICHE SCORE ON THE CITE-seq PROTEIN UMAP",
        x=0.055,
        y=0.980,
        ha="left",
        fontsize=12.8,
        weight="bold",
        color=INK,
    )
    fig.text(
        0.055,
        0.925,
        "Rahim tumour CD8 T cells (n=4,788) · 130 biological proteins",
        ha="left",
        va="top",
        fontsize=8.8,
        color="#4F5962",
    )
    fig.subplots_adjust(left=0.055, right=0.875, bottom=0.050, top=0.885)
    full_png = HERE / "candidate_full130_protein_umap_projected_niche.png"
    full_pdf = HERE / "candidate_full130_protein_umap_projected_niche.pdf"
    fig.savefig(full_png, dpi=320, facecolor="white")
    fig.savefig(full_pdf, facecolor="white")
    plt.close(fig)

    metrics = {
        "shared22": neighborhood_summary(shared_coordinates, scores),
        "full130": neighborhood_summary(full_coordinates, scores),
    }
    manifest = {
        "cells": EXPECTED_CELL_N,
        "embedding_score_used": False,
        "full_protein_definition": {
            "total_adt_channels": EXPECTED_ADT_N,
            "removed_isotype_controls": EXPECTED_ISOTYPE_N,
            "retained_biological_proteins": EXPECTED_BIOLOGICAL_N,
            "lag3_retained": True,
        },
        "preprocessing": "log1p raw ADT counts; subtract within-cell mean across 130 proteins; StandardScaler; PCA30; UMAP",
        "umap": {
            "n_neighbors": N_NEIGHBORS,
            "min_dist": MIN_DIST,
            "metric": "euclidean",
            "random_state": UMAP_SEED,
        },
        "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "visual_neighborhood_metrics": metrics,
        "inputs": {
            "adt_h5ad": str(ADT_H5AD),
            "projection": str(PROJECTION),
            "shared_coordinates": str(SHARED_COORDINATES),
        },
        "input_sha256": {
            "adt_h5ad": sha256(ADT_H5AD),
            "projection": sha256(PROJECTION),
            "shared_coordinates": sha256(SHARED_COORDINATES),
        },
        "outputs": {
            "coordinates": str(coordinates_path),
            "comparison_png": str(comparison_png),
            "comparison_pdf": str(comparison_pdf),
            "full_png": str(full_png),
            "full_pdf": str(full_pdf),
        },
    }
    manifest_path = HERE / "full130_protein_umap_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(comparison_png)
    print(full_png)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
