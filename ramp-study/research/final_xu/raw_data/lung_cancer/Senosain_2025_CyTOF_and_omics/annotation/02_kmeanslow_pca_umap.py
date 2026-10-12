#!/usr/bin/env python3
"""Build the all-patient Senosain KMEANSLOW -> PCA -> UMAP representation.

The frozen 140k sample is not changed.  RAMP's current KMEANSLOW function is
applied independently to every patient (``unit_key``) and every biological
marker.  The 70 patient-level outputs are then kept in their original row order,
globally feature-z-scored, reduced by PCA, and embedded with UMAP.

All 35 biological markers are retained in the archived KMEANSLOW matrix.  CD90
is excluded from PCA/UMAP because it was not measured uniformly across the
published 70-patient CyTOF cohort.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


SEED = 42
PCA_COMPONENTS = 25
UMAP_NEIGHBORS = 30
UMAP_MIN_DIST = 0.15
EMBEDDING_EXCLUSIONS = {"CD90"}
LINEAGE_MARKERS = [
    "EPCAM",
    "CD31",
    "CD45",
    "VIMENTIN",
    "CYTOKERATIN",
    "CK7",
]

HERE = Path(__file__).resolve().parent
RAW_PATH = HERE / "senosain_140k_raw35.npy"
METADATA_PATH = HERE / "senosain_140k_cell_metadata.csv.gz"
MARKERS_PATH = HERE / "senosain_140k_markers35.csv"
SAMPLE_MANIFEST_PATH = HERE / "senosain_140k_manifest.json"
DEFAULT_OUTPUT = HERE / "kmeanslow_pca_umap"
RAMP_PARENT = HERE.parents[3] / "algorithm"
RAMP_SOURCE = RAMP_PARENT / "ramp" / "ramp.py"


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
        help="Replace an existing output directory after all inputs validate.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_inputs(
    raw: np.ndarray, metadata: pd.DataFrame, marker_table: pd.DataFrame
) -> list[str]:
    required_metadata = {
        "sampled_row_zero_based",
        "source_cell_id",
        "patient_id",
        "dataset",
        "unit_key",
    }
    missing = required_metadata - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")
    if "marker" not in marker_table.columns:
        raise ValueError("Marker table lacks the marker column")

    markers = marker_table["marker"].astype(str).tolist()
    if raw.ndim != 2 or raw.shape != (len(metadata), len(markers)):
        raise ValueError(
            f"Matrix/metadata/marker mismatch: raw={raw.shape}, "
            f"metadata={len(metadata)}, markers={len(markers)}"
        )
    if not np.isfinite(raw).all():
        raise ValueError("Raw biological matrix contains non-finite values")
    if len(set(markers)) != len(markers):
        raise ValueError("Marker names are not unique")
    if not np.array_equal(
        metadata["sampled_row_zero_based"].to_numpy(), np.arange(len(metadata))
    ):
        raise ValueError("Metadata row index no longer matches matrix row order")
    if not metadata["source_cell_id"].astype(str).is_unique:
        raise ValueError("source_cell_id is not unique")
    unit_sizes = metadata.groupby("unit_key", sort=False).size()
    if len(unit_sizes) != 70 or not unit_sizes.eq(2000).all():
        raise ValueError(
            "Expected exactly 70 unit_key values with 2,000 cells each; found "
            f"{len(unit_sizes)} units, range={unit_sizes.min()}-{unit_sizes.max()}"
        )
    if metadata["dataset"].astype(str).nunique() != 1:
        raise ValueError("This workflow expects one Senosain dataset")
    if "CD90" not in markers:
        raise ValueError("Expected CD90 in the archived 35-marker panel")
    for marker in LINEAGE_MARKERS:
        if marker not in markers:
            raise ValueError(f"Required lineage marker is absent: {marker}")
    return markers


def import_ramp_functions():
    if not RAMP_SOURCE.exists():
        raise FileNotFoundError(f"Current RAMP source not found: {RAMP_SOURCE}")
    sys.path.insert(0, str(RAMP_PARENT))
    module = importlib.import_module("ramp.ramp")
    return (
        module.kmeanslow_global,
        module.rank_inversion_count,
        module.__version__,
    )


def import_umap_without_disk_cache():
    """Import umap while keeping numba from writing or requiring disk caches.

    This is process-local.  It leaves JIT compilation enabled but forces the
    decorators used by umap-learn/pynndescent to use ``cache=False``.  The
    workaround is needed by the locally installed Python 3.14 packages, whose
    source locator currently cannot create numba caches.
    """

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


def write_plots(
    output: Path,
    coordinates: np.ndarray,
    baseline: np.ndarray,
    markers: list[str],
) -> list[Path]:
    mpl_cache = output / ".matplotlib_cache"
    mpl_cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )

    created: list[Path] = []

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0), constrained_layout=True)
    axes[0].scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        s=0.18,
        c="#383838",
        alpha=0.32,
        linewidths=0,
        rasterized=True,
    )
    axes[0].set_title("All 140,000 cells")
    density = axes[1].hexbin(
        coordinates[:, 0],
        coordinates[:, 1],
        gridsize=180,
        mincnt=1,
        bins="log",
        cmap="magma",
        linewidths=0,
    )
    axes[1].set_title("Cell density")
    colorbar = fig.colorbar(density, ax=axes[1], fraction=0.045, pad=0.02)
    colorbar.set_label("log10 cells per hexagon")
    for axis in axes:
        axis.set_xlabel("UMAP 1")
        axis.set_ylabel("UMAP 2")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.spines[:].set_visible(False)
    fig.suptitle(
        "Senosain CyTOF · patient-wise KMEANSLOW → z-score → PCA → UMAP",
        fontsize=13,
        fontweight="bold",
    )
    for suffix in ("png", "pdf"):
        path = output / f"senosain_140k_umap_overview.{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        created.append(path)
    plt.close(fig)

    marker_index = {marker: idx for idx, marker in enumerate(markers)}
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 8.0), constrained_layout=True)
    for axis, marker in zip(axes.flat, LINEAGE_MARKERS, strict=True):
        values = baseline[:, marker_index[marker]]
        low, high = np.quantile(values, [0.01, 0.99])
        if high <= low:
            low, high = float(values.min()), float(values.max())
        order = np.argsort(values, kind="stable")
        scatter = axis.scatter(
            coordinates[order, 0],
            coordinates[order, 1],
            c=np.clip(values[order], low, high),
            s=0.22,
            cmap="viridis",
            vmin=low,
            vmax=high,
            linewidths=0,
            rasterized=True,
        )
        axis.set_title(marker)
        axis.set_xlabel("UMAP 1")
        axis.set_ylabel("UMAP 2")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.spines[:].set_visible(False)
        colorbar = fig.colorbar(scatter, ax=axis, fraction=0.042, pad=0.015)
        colorbar.set_label("KMEANSLOW value\n(1st–99th percentile)")
    fig.suptitle(
        "Lineage-marker structure on the all-cell UMAP",
        fontsize=14,
        fontweight="bold",
    )
    for suffix in ("png", "pdf"):
        path = output / f"senosain_140k_umap_lineage_markers.{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        created.append(path)
    plt.close(fig)
    return created


def main() -> None:
    args = parse_args()
    output = args.output.resolve()

    print("Loading and validating the frozen 140k sample", flush=True)
    raw = np.load(RAW_PATH)
    metadata = pd.read_csv(METADATA_PATH, low_memory=False)
    marker_table = pd.read_csv(MARKERS_PATH)
    markers = validate_inputs(raw, metadata, marker_table)

    input_hashes = {
        RAW_PATH.name: sha256(RAW_PATH),
        METADATA_PATH.name: sha256(METADATA_PATH),
        MARKERS_PATH.name: sha256(MARKERS_PATH),
        SAMPLE_MANIFEST_PATH.name: sha256(SAMPLE_MANIFEST_PATH),
        RAMP_SOURCE.name: sha256(RAMP_SOURCE),
        Path(__file__).name: sha256(Path(__file__).resolve()),
    }

    if output.exists():
        if not args.force:
            raise FileExistsError(
                f"Output directory already exists: {output}. Use --force to replace it."
            )
        shutil.rmtree(output)
    output.mkdir(parents=True)

    kmeanslow_global, rank_inversion_count, ramp_version = import_ramp_functions()
    started = time.perf_counter()

    print("Running current RAMP KMEANSLOW for 70 patients × 35 markers", flush=True)
    baseline = kmeanslow_global(
        raw,
        metadata["unit_key"].astype(str).to_numpy(),
        metadata["dataset"].astype(str).to_numpy(),
        markers,
    )
    if baseline.shape != raw.shape or not np.isfinite(baseline).all():
        raise AssertionError("KMEANSLOW returned an invalid matrix")
    inversions = rank_inversion_count(
        raw,
        baseline,
        metadata[["unit_key"]].copy(),
    )
    if inversions != 0:
        raise AssertionError(f"KMEANSLOW introduced {inversions} strict rank inversions")
    baseline_path = output / "senosain_140k_kmeanslow35.float32.npy"
    np.save(baseline_path, baseline.astype(np.float32))
    print(f"Saved KMEANSLOW matrix: {baseline_path.name}", flush=True)

    embedding_markers = [m for m in markers if m not in EMBEDDING_EXCLUSIONS]
    embedding_indices = [markers.index(m) for m in embedding_markers]
    embedding_values = baseline[:, embedding_indices]
    variance = np.var(embedding_values, axis=0, ddof=0)
    retained = variance > 1e-10
    if not retained.all():
        dropped = [m for m, keep in zip(embedding_markers, retained) if not keep]
        print(f"Dropping constant embedding markers: {dropped}", flush=True)
        embedding_values = embedding_values[:, retained]
        embedding_markers = [
            m for m, keep in zip(embedding_markers, retained) if keep
        ]

    print(
        f"Globally z-scoring {len(embedding_markers)} markers and fitting "
        f"{PCA_COMPONENTS}-component PCA",
        flush=True,
    )
    scaler = StandardScaler(copy=True)
    normalized = scaler.fit_transform(embedding_values)
    pca = PCA(
        n_components=PCA_COMPONENTS,
        svd_solver="randomized",
        random_state=SEED,
    )
    pca_coordinates = pca.fit_transform(normalized)
    if not np.isfinite(pca_coordinates).all():
        raise AssertionError("PCA returned non-finite coordinates")
    pca_path = output / "senosain_140k_pca25.float32.npy"
    np.save(pca_path, pca_coordinates.astype(np.float32))

    scaling = pd.DataFrame(
        {
            "marker": embedding_markers,
            "global_mean_before_zscore": scaler.mean_,
            "global_scale": scaler.scale_,
        }
    )
    scaling_path = output / "feature_scaling.csv"
    scaling.to_csv(scaling_path, index=False)

    variance_table = pd.DataFrame(
        {
            "principal_component": np.arange(1, PCA_COMPONENTS + 1),
            "explained_variance": pca.explained_variance_,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance_ratio": np.cumsum(
                pca.explained_variance_ratio_
            ),
        }
    )
    variance_path = output / "pca_explained_variance.csv"
    variance_table.to_csv(variance_path, index=False)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=embedding_markers,
        columns=[f"PC{i}" for i in range(1, PCA_COMPONENTS + 1)],
    )
    loadings.index.name = "marker"
    loadings_path = output / "pca_loadings.csv"
    loadings.to_csv(loadings_path)

    print(
        f"Fitting all-cell UMAP (neighbors={UMAP_NEIGHBORS}, "
        f"min_dist={UMAP_MIN_DIST}, seed={SEED})",
        flush=True,
    )
    umap = import_umap_without_disk_cache()
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=UMAP_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="euclidean",
        init="spectral",
        random_state=SEED,
        transform_seed=SEED,
        n_jobs=1,
        low_memory=True,
        verbose=True,
    )
    umap_coordinates = reducer.fit_transform(pca_coordinates)
    if umap_coordinates.shape != (len(raw), 2) or not np.isfinite(
        umap_coordinates
    ).all():
        raise AssertionError("UMAP returned invalid coordinates")
    umap_path = output / "senosain_140k_umap2.float32.npy"
    np.save(umap_path, umap_coordinates.astype(np.float32))

    embedding_table = metadata[
        ["sampled_row_zero_based", "source_cell_id", "patient_id", "unit_key"]
    ].copy()
    embedding_table["UMAP1"] = umap_coordinates[:, 0]
    embedding_table["UMAP2"] = umap_coordinates[:, 1]
    embedding_table_path = output / "senosain_140k_umap_coordinates.csv.gz"
    embedding_table.to_csv(embedding_table_path, index=False, compression="gzip")

    print("Drawing overview and lineage-marker UMAPs", flush=True)
    plot_paths = write_plots(output, umap_coordinates, baseline, markers)
    elapsed = time.perf_counter() - started

    output_paths = [
        baseline_path,
        pca_path,
        umap_path,
        embedding_table_path,
        scaling_path,
        variance_path,
        loadings_path,
        *plot_paths,
    ]
    manifest = {
        "schema_version": "senosain-kmeanslow-pca-umap-v1",
        "source_sample_manifest": str(SAMPLE_MANIFEST_PATH),
        "input_hashes": input_hashes,
        "cells": int(len(raw)),
        "patients": int(metadata["unit_key"].nunique()),
        "cells_per_patient": 2000,
        "archived_kmeanslow_markers": markers,
        "embedding_markers": embedding_markers,
        "embedding_exclusions": {
            "CD90": (
                "excluded from PCA/UMAP because the provider documented "
                "separate acquisitions with and without the CD90 antibody"
            )
        },
        "preprocessing": {
            "implementation": str(RAMP_SOURCE),
            "ramp_version": ramp_version,
            "unit_key": "patient",
            "input_scale": "unchanged provider-normalized/denoised FCS values",
            "patient_marker_operation": "current RAMP kmeanslow_global",
            "post_kmeanslow_transform": "global marker-wise StandardScaler only",
            "additional_arcsinh_or_log": False,
            "strict_raw_to_kmeanslow_rank_inversions": int(inversions),
        },
        "pca": {
            "components": PCA_COMPONENTS,
            "solver": "randomized",
            "seed": SEED,
            "cumulative_explained_variance_ratio": float(
                np.sum(pca.explained_variance_ratio_)
            ),
        },
        "umap": {
            "components": 2,
            "neighbors": UMAP_NEIGHBORS,
            "min_dist": UMAP_MIN_DIST,
            "metric": "euclidean",
            "init": "spectral",
            "seed": SEED,
            "n_jobs": 1,
            "fitted_cells": int(len(raw)),
        },
        "labels_or_clinical_outcomes_used": False,
        "runtime_seconds": elapsed,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "umap_learn": umap.__version__,
        },
        "outputs": {
            path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in output_paths
        },
    }
    manifest_path = output / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"DONE: {len(raw):,} cells; PCA25 cumulative variance "
        f"{np.sum(pca.explained_variance_ratio_):.4f}; "
        f"rank inversions={inversions}; elapsed={elapsed:.1f}s",
        flush=True,
    )
    print(output, flush=True)


if __name__ == "__main__":
    main()
