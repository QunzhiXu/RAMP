#!/usr/bin/env python3
"""Plot the transferred spatial phenotype and TIGIT/CD39 program on one CD8 UMAP.

The embedding is fitted once using only the 22 RAMP-aligned shared markers from
all Rahim tumour CD8 T cells.  Neither plotted score, cell labels, nor outcomes
are used to fit the UMAP; the two panels therefore use identical coordinates.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
ALIGNMENT = HERE.parent / "alignment"
PROJECTION = HERE / "projection"
SCREEN = HERE / "screen"
FIGURES = HERE / "figures"
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgba
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from umap import UMAP


EXPECTED_CELLS = 4_788
EXPECTED_DONORS = {"126": 812, "indiv_1": 3_345, "indiv_2": 631}
UMAP_SEED = 20260824
N_NEIGHBORS = 30
MIN_DIST = 0.25
CMAP = LinearSegmentedColormap.from_list(
    "spatial_program",
    ["#253494", "#2C7FB8", "#7FCDBB", "#FEC44F", "#D7301F"],
)
_HIGH_CMAP = LinearSegmentedColormap.from_list(
    "high_state_only", ["#FEE08B", "#FDAE61", "#F46D43", "#C62828"]
)
# With the standard 0--1 normalization, the first half of this lookup table is
# exactly one blue.  The color can change only after a score exceeds 0.5.
LOW50_BLUE_CMAP = ListedColormap(
    np.vstack(
        [
            np.repeat(np.asarray(to_rgba("#2166AC"))[None, :], 1_025, axis=0),
            _HIGH_CMAP(np.linspace(0.0, 1.0, 1_023)),
        ]
    ),
    name="low50_fixed_blue",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_axis(axis: plt.Axes) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def render_bridge_umap(
    cd8: pd.DataFrame,
    coordinates: np.ndarray,
    rho: float,
    cmap: LinearSegmentedColormap | ListedColormap,
    stem: str,
    threshold_note: str = "",
) -> tuple[Path, Path]:
    panels = [
        (
            "Transferred spatial-niche score",
            "Within-donor rank",
            "projected_rank_within_rahim_donor",
        ),
        (
            "TIGIT/CD39 program score",
            "Mean within-donor protein rank",
            "compact_top5pct_clr_program",
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.25), sharex=True, sharey=True)
    for index, (title, colorbar_label, column) in enumerate(panels):
        axis = axes[index]
        values = cd8[column].to_numpy(dtype=float)
        order = np.argsort(values, kind="stable")
        scatter = axis.scatter(
            coordinates[order, 0],
            coordinates[order, 1],
            c=values[order],
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            s=8.2,
            alpha=0.90,
            linewidths=0,
            rasterized=True,
        )
        axis.set_title(title, loc="left", fontsize=13, weight="bold", pad=8)
        colorbar = fig.colorbar(scatter, ax=axis, fraction=0.045, pad=0.018)
        colorbar.set_ticks([0.0, 0.5, 0.75, 1.0])
        colorbar.set_label(colorbar_label, fontsize=9)
        colorbar.ax.tick_params(labelsize=8, length=2)
        colorbar.outline.set_linewidth(0.5)
        clean_axis(axis)

    fig.suptitle(
        "Rahim tumour CD8 T cells (n=4,788; 3 donors)",
        x=0.055,
        y=0.995,
        ha="left",
        fontsize=15,
        weight="bold",
    )
    footer = (
        "Identical coordinates in both panels · UMAP fitted from 22 shared RAMP-aligned proteins only"
        f" · Spearman ρ = {rho:.3f}"
    )
    if threshold_note:
        footer += f" · {threshold_note}"
    fig.text(
        0.055,
        0.016,
        footer,
        ha="left",
        va="bottom",
        fontsize=9.2,
        color="#444444",
    )
    fig.tight_layout(rect=(0.02, 0.055, 1.0, 0.94), w_pad=0.75)

    png_path = FIGURES / f"{stem}.png"
    pdf_path = FIGURES / f"{stem}.pdf"
    fig.savefig(png_path, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    metadata_path = ALIGNMENT / "input" / "cell_metadata.csv.gz"
    matrix_path = ALIGNMENT / "output" / "aligned_ramp_ver2.float32.npy"
    marker_path = ALIGNMENT / "input" / "markers.txt"
    projection_path = PROJECTION / "rahim_tumor_cd8_projected_scores.csv.gz"
    program_path = SCREEN / "rahim_tumor_cd8_biomarker_programs.csv.gz"

    metadata = pd.read_csv(metadata_path, low_memory=False)
    aligned = np.load(matrix_path, mmap_mode="r")
    if len(metadata) != aligned.shape[0]:
        raise AssertionError("Alignment metadata and matrix have different row counts")

    mask = (
        metadata["dataset"].eq("GSE212797_ADT")
        & metadata["tissue"].eq("tumor")
        & metadata["provider_cell_type"].eq("CD8 T cell")
    ).to_numpy()
    rows = np.flatnonzero(mask)
    cd8 = metadata.loc[mask, ["cell_uid", "donor", "tissue", "provider_cell_type"]].copy()
    cd8["cell_uid"] = cd8["cell_uid"].str.removeprefix("RAHIM|")
    cd8["donor"] = cd8["donor"].astype(str)

    projection = pd.read_csv(projection_path, dtype={"donor": str})
    program = pd.read_csv(program_path, dtype={"donor": str})
    projection_columns = ["cell_uid", "donor", "projected_rank_within_rahim_donor"]
    program_columns = ["cell_uid", "donor", "compact_top5pct_clr_program"]
    if projection.duplicated(["cell_uid", "donor"]).any():
        raise AssertionError("Projection cell keys are not unique")
    if program.duplicated(["cell_uid", "donor"]).any():
        raise AssertionError("Program cell keys are not unique")
    cd8 = cd8.merge(
        projection[projection_columns], on=["cell_uid", "donor"], how="left", validate="one_to_one"
    ).merge(
        program[program_columns], on=["cell_uid", "donor"], how="left", validate="one_to_one"
    )

    score_columns = ["projected_rank_within_rahim_donor", "compact_top5pct_clr_program"]
    if len(cd8) != EXPECTED_CELLS or cd8[score_columns].isna().any().any():
        raise AssertionError("Expected complete score matches for all 4,788 Rahim tumour CD8 cells")
    donor_counts = cd8["donor"].value_counts().sort_index().astype(int).to_dict()
    if donor_counts != dict(sorted(EXPECTED_DONORS.items())):
        raise AssertionError(f"Unexpected Rahim donor counts: {donor_counts}")
    for column in score_columns:
        if not cd8[column].between(0.0, 1.0, inclusive="both").all():
            raise AssertionError(f"{column} is not bounded within [0, 1]")

    # Fit once on the RAMP-aligned shared-protein matrix; scores are added only afterwards.
    cd8_aligned = np.asarray(aligned[rows, :], dtype=np.float64)
    scaled = StandardScaler().fit_transform(cd8_aligned)
    coordinates = UMAP(
        n_neighbors=N_NEIGHBORS,
        min_dist=MIN_DIST,
        metric="euclidean",
        random_state=UMAP_SEED,
        n_jobs=1,
    ).fit_transform(scaled)
    cd8["UMAP1"] = coordinates[:, 0]
    cd8["UMAP2"] = coordinates[:, 1]

    rho = float(spearmanr(cd8[score_columns[0]], cd8[score_columns[1]]).statistic)
    donor_rho = {
        donor: float(spearmanr(group[score_columns[0]], group[score_columns[1]]).statistic)
        for donor, group in cd8.groupby("donor", sort=True)
    }

    stem = "hnscc_rahim4788_cd8_spatial_niche_vs_tigit_cd39_umap"
    png_path, pdf_path = render_bridge_umap(cd8, coordinates, rho, CMAP, stem)
    threshold_stem = f"{stem}_low50_fixed_blue"
    threshold_png, threshold_pdf = render_bridge_umap(
        cd8,
        coordinates,
        rho,
        LOW50_BLUE_CMAP,
        threshold_stem,
        threshold_note="Scores ≤0.5 shown in one blue",
    )

    coordinate_path = PROJECTION / "rahim_tumor_cd8_transfer_program_umap_coordinates.csv.gz"
    cd8.to_csv(coordinate_path, index=False, compression="gzip")
    marker_count = sum(1 for line in marker_path.read_text().splitlines() if line.strip())
    manifest = {
        "analysis": "Rahim tumour-CD8 transfer-to-program bridge UMAP",
        "cells": int(len(cd8)),
        "donor_counts": donor_counts,
        "embedding_input": "RAMP 2.3.0 aligned shared-protein values",
        "embedding_marker_count": marker_count,
        "scores_or_labels_used_for_embedding": False,
        "umap": {
            "fit_scope": "all 4,788 Rahim tumour CD8 cells; no downsampling",
            "pre_umap_scaling": "StandardScaler fit on these 4,788 cells",
            "n_neighbors": N_NEIGHBORS,
            "min_dist": MIN_DIST,
            "metric": "euclidean",
            "random_state": UMAP_SEED,
        },
        "coloring": {
            "left": "projected_rank_within_rahim_donor",
            "right": "compact_top5pct_clr_program",
            "limits": [0.0, 1.0],
            "high_values_drawn_last": True,
            "threshold_variant": "scores <= 0.5 use exactly one blue; scores > 0.5 use yellow-to-red",
        },
        "program_definition": (
            "Mean of within-donor percentile ranks for strict-no-LAG3 paper-CLR TIGIT and CD39"
        ),
        "score_association": {
            "overall_spearman": rho,
            "within_donor_spearman": donor_rho,
        },
        "inputs": {
            "metadata": str(metadata_path),
            "aligned_matrix": str(matrix_path),
            "markers": str(marker_path),
            "projection_scores": str(projection_path),
            "program_scores": str(program_path),
        },
        "inputs_sha256": {
            "metadata": sha256(metadata_path),
            "aligned_matrix": sha256(matrix_path),
            "markers": sha256(marker_path),
            "projection_scores": sha256(projection_path),
            "program_scores": sha256(program_path),
        },
        "outputs": {
            "coordinates": str(coordinate_path),
            "png": str(png_path),
            "pdf": str(pdf_path),
            "low50_fixed_blue_png": str(threshold_png),
            "low50_fixed_blue_pdf": str(threshold_pdf),
        },
    }
    manifest_path = PROJECTION / "rahim_tumor_cd8_transfer_program_umap_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
