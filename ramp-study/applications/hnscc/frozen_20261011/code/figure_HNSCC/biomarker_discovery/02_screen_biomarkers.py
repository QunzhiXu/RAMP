#!/usr/bin/env python3
"""Run the formal outcome-blind Rahim 107-ADT biomarker screen.

This is a self-contained implementation of rules developed in the PT70 screen
and frozen before this formal PT74 run.  It joins
the formal PT74 projected spatial score to the copied complete Rahim ADT/RNA
objects, calculates the four donor-blocked primary views and two quality
sensitivities, applies the frozen robustness and redundancy rules, and only
then applies the compact top-5% cutoff.  Marker names and marker count are not
forced at any selection stage.

The compact cutoff is 0.05 * 107 = 5.35 on the original 107-marker rank scale.
Candidates are never reranked after robustness or redundancy filtering.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Iterable


HERE = Path(__file__).resolve().parent
FIGURE_HNSCC = HERE.parent
PROJECTION = HERE / "projection/rahim_tumor_cd8_projected_scores.csv.gz"
SCREEN_DIR = HERE / "screen"
FIGURE_DIR = HERE / "figures"
ADT_H5AD = (
    FIGURE_HNSCC
    / "sampled_data/rahim_citeseq/complete/GSE212797_adata_ADT.h5ad"
)
RNA_H5AD = (
    FIGURE_HNSCC
    / "sampled_data/rahim_citeseq/complete/GSE212797_adata.h5ad"
)
CACHE = HERE / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE / "xdg"))
sys.dont_write_bytecode = True

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata


EXPECTED_CELL_N = 4_788
EXPECTED_DONOR_N = 3
EXPECTED_ADT_N = 137
EXPECTED_ISOTYPE_N = 7
EXPECTED_BACKGROUND_N = 129
EXPECTED_CANDIDATE_N = 107
EXPECTED_GLOBAL_LOO_N = 128
ROBUST_RANK_CUTOFF = 15.0
REDUNDANCY_ABS_RHO = 0.85
COMPACT_FRACTION = 0.05
COMPACT_RANK_DENOMINATOR = EXPECTED_CANDIDATE_N
COMPACT_RANK_CUTOFF = round(COMPACT_FRACTION * COMPACT_RANK_DENOMINATOR, 2)

HOLDOUT_MARKER = "CD223--LAG-3"
SHARED_RAHIM_MARKERS = {
    "CD45",
    "CD3",
    "CD7-1",
    "CD4-1",
    "CD8",
    "CD25",
    "CD20",
    "CD38-1",
    "CD56--NCAM",
    "CD57",
    "CD16",
    "CD11c",
    "CD11b",
    "HLA-DR",
    "CD21",
    "CD31",
    "CD69-1",
    "CD278--ICOS",
    "CD279--PD-1",
    "CD45RO",
    "CD45RA",
    "CD274--B7-H1--PD-L1",
}
FRIENDLY_NAMES = {
    "CD137": "CD137 (4-1BB)",
    "HLA-E-1": "HLA-E",
    "CD83-1": "CD83",
    "TIGIT--VSTM3": "TIGIT",
    "CD152": "CD152 (CTLA-4)",
    "CD270--HVEM--TR2": "HVEM",
    "CD272--BTLA": "BTLA",
    "HLA-A-B-C": "HLA-ABC",
    "CD95--Fas": "CD95 (Fas)",
    "CD88--C5aR": "CD88 (C5aR)",
    "CD244--2B4": "CD244 (2B4)",
    "CD223--LAG-3": "LAG3",
}
PRIMARY_VARIANTS = (
    "raw_rank",
    "paper_clr",
    "isotype_adjusted",
    "global_adt_loo_adjusted",
)
ALL_VARIANTS = PRIMARY_VARIANTS + (
    "rna_qc_adjusted",
    "paper_clr_explicit_singlets",
)
SCORE_COLUMN_PRIORITY = (
    "projected_roi_adjusted_exposure",
    "projected_pt_cd8_niche_rank",
    "projected_rank_within_rahim_donor",
    "projected_score",
    "score",
)
RULE_COLUMNS = (
    "rule_clr_positive_3of3",
    "rule_isotype_positive_3of3",
    "rule_global_adt_positive_at_least_2of3",
    "rule_median_primary_rank_at_most_15",
    "rule_all_primary_lodo_positive",
)


def decode(values: np.ndarray) -> np.ndarray:
    """Decode an H5AD string vector without relying on anndata."""
    return np.asarray(
        [value.decode() if isinstance(value, bytes) else str(value) for value in values],
        dtype=object,
    )


def categorical(handle: h5py.File, key: str) -> np.ndarray:
    categories = decode(handle[f"obs/__categories/{key}"][:])
    codes = handle[f"obs/{key}"][:]
    answer = np.full(len(codes), "__MISSING__", dtype=object)
    valid = codes >= 0
    answer[valid] = categories[codes[valid]]
    textual_missing = np.isin(
        np.char.lower(answer.astype(str)), ["nan", "na", "none", "null", ""]
    )
    answer[textual_missing] = "__MISSING__"
    return answer


def read_csr(group: h5py.Group, n_columns: int) -> sparse.csr_matrix:
    return sparse.csr_matrix(
        (group["data"][:], group["indices"][:], group["indptr"][:]),
        shape=(len(group["indptr"]) - 1, n_columns),
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value: object) -> object:
    """Convert numpy/pandas scalars and non-finite floats for strict JSON."""
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(payload: dict[str, object], path: Path) -> None:
    path.write_text(
        json.dumps(json_ready(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv_gz(frame: pd.DataFrame, path: Path) -> None:
    """Write a byte-reproducible gzip CSV."""
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", fileobj=raw_handle, mode="wb", mtime=0) as handle:
            frame.to_csv(handle, index=False)


def read_score(path: Path) -> tuple[pd.DataFrame, str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Formal projection is missing: {path}. Run 01_project_spatial_state.py first."
        )
    score = pd.read_csv(path, dtype={"cell_uid": str, "donor": str})
    if "cell_uid" not in score:
        aliases = [name for name in ("UID", "barcode", "cell_id") if name in score]
        if len(aliases) != 1:
            raise KeyError("Projection must contain cell_uid or one unambiguous ID alias")
        score = score.rename(columns={aliases[0]: "cell_uid"})
    score["cell_uid"] = score.cell_uid.astype(str)
    if score.cell_uid.duplicated().any():
        duplicate = score.loc[score.cell_uid.duplicated(), "cell_uid"].iloc[0]
        raise ValueError(f"Duplicate projected cell_uid: {duplicate}")
    known = [column for column in SCORE_COLUMN_PRIORITY if column in score]
    if not known:
        numeric = [
            column
            for column in score.select_dtypes(include=[np.number]).columns
            if any(token in column.lower() for token in ("score", "project", "predict"))
        ]
        if len(numeric) != 1:
            raise KeyError(
                "Could not infer exactly one projected-score column; "
                f"candidate numeric columns were {numeric}"
            )
        score_column = numeric[0]
    else:
        score_column = known[0]
    score[score_column] = pd.to_numeric(score[score_column], errors="coerce")
    if not np.isfinite(score[score_column]).all():
        raise ValueError(f"Non-finite values in projected score column {score_column}")
    if len(score) != EXPECTED_CELL_N:
        raise AssertionError(
            f"Expected {EXPECTED_CELL_N:,} projected Rahim tumor CD8 cells; got {len(score):,}"
        )
    return score, score_column


def load_adt() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if not ADT_H5AD.exists():
        raise FileNotFoundError(f"Copied complete Rahim ADT object is missing: {ADT_H5AD}")
    with h5py.File(ADT_H5AD, "r") as handle:
        ids = decode(handle["obs/_index"][:])
        markers = decode(handle["var/_index"][:])
        matrix = read_csr(handle["X"], len(markers)).toarray().astype(np.float64)
        obs = pd.DataFrame(
            {
                "cell_uid": ids,
                "adt_donor": categorical(handle, "patient"),
                "adt_tissue": categorical(handle, "tissue"),
                "droplet_type": categorical(handle, "droplet_type"),
                "adt_n_counts_obs": handle["obs/n_counts"][:].astype(float),
            }
        )
    if len(markers) != EXPECTED_ADT_N:
        raise AssertionError(f"Expected {EXPECTED_ADT_N} ADTs; observed {len(markers)}")
    if pd.Series(ids).duplicated().any():
        raise AssertionError("Rahim ADT cell IDs are not unique")
    if not np.all(matrix >= 0) or not np.allclose(matrix, np.round(matrix)):
        raise AssertionError("Rahim ADT X is not a nonnegative raw-count matrix")
    obs["adt_library_size_137"] = matrix.sum(axis=1)
    return obs, markers, matrix


def load_rna_obs() -> pd.DataFrame:
    if not RNA_H5AD.exists():
        raise FileNotFoundError(f"Copied complete Rahim RNA object is missing: {RNA_H5AD}")
    with h5py.File(RNA_H5AD, "r") as handle:
        frame = pd.DataFrame(
            {
                "cell_uid": decode(handle["obs/_index"][:]),
                "rna_donor": categorical(handle, "patient"),
                "rna_tissue": categorical(handle, "tissue"),
                "rna_cell_type": categorical(handle, "cell_type"),
                "rna_n_counts": handle["obs/n_counts"][:].astype(float),
                "rna_n_genes": handle["obs/n_genes"][:].astype(float),
                "rna_percent_mito": handle["obs/percent_mito"][:].astype(float),
            }
        )
    if frame.cell_uid.duplicated().any():
        raise AssertionError("Rahim RNA cell IDs are not unique")
    return frame


def join_inputs(
    score: pd.DataFrame,
    score_column: str,
    adt_obs: pd.DataFrame,
    markers: np.ndarray,
    adt_all: np.ndarray,
    rna_obs: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    adt_row = adt_obs[["cell_uid"]].copy()
    adt_row["adt_row"] = np.arange(len(adt_obs), dtype=int)
    joined = (
        score.merge(adt_row, on="cell_uid", how="inner", validate="one_to_one")
        .merge(adt_obs, on="cell_uid", validate="one_to_one")
        .merge(rna_obs, on="cell_uid", validate="one_to_one")
    )
    if len(joined) != len(score):
        raise AssertionError(
            f"{len(score) - len(joined)} projected cells were absent from ADT/RNA objects"
        )
    joined = joined.loc[
        joined.rna_tissue.eq("tumor") & joined.rna_cell_type.eq("CD8 T cell")
    ].copy()
    if len(joined) != EXPECTED_CELL_N:
        raise AssertionError(
            "The formal projection did not resolve to exactly "
            f"{EXPECTED_CELL_N:,} original tumor CD8 cells"
        )
    if not joined.adt_tissue.eq("tumor").all():
        raise AssertionError("RNA and ADT tissue annotations disagree")
    if not joined.adt_donor.eq(joined.rna_donor).all():
        raise AssertionError("RNA and ADT donor annotations disagree")
    if "donor" in joined and not joined.donor.astype(str).eq(joined.rna_donor).all():
        raise AssertionError("Projection donor disagrees with original Rahim donor")
    joined["donor"] = joined.rna_donor.astype(str)
    joined = joined.sort_values(["donor", "cell_uid"], kind="stable").reset_index(drop=True)
    if joined.donor.nunique() != EXPECTED_DONOR_N:
        raise AssertionError(
            f"Expected {EXPECTED_DONOR_N} Rahim donors; observed {joined.donor.nunique()}"
        )
    selected = adt_all[joined.adt_row.to_numpy(int)]
    if selected.shape != (len(joined), len(markers)):
        raise AssertionError("Joined ADT matrix shape mismatch")
    joined["projected_score"] = joined[score_column].to_numpy(float)
    return joined, selected


def rank_z(values: np.ndarray) -> np.ndarray:
    """Average-tie ranks centered and scaled to root-mean-square one."""
    values = np.asarray(values, float)
    ranks = rankdata(values, method="average")
    ranks -= ranks.mean()
    scale = np.sqrt(np.mean(ranks * ranks))
    if scale <= 0:
        return np.full(len(values), np.nan)
    return ranks / scale


def rank_percentile(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    return (rankdata(values, method="average") - 0.5) / len(values)


def rank_z_matrix(values: np.ndarray, donors: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    if values.ndim == 1:
        values = values[:, None]
    answer = np.full(values.shape, np.nan)
    for donor in pd.unique(donors):
        rows = np.flatnonzero(donors == donor)
        for column in range(values.shape[1]):
            answer[rows, column] = rank_z(values[rows, column])
    return answer


def percentile_matrix(values: np.ndarray, donors: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    if values.ndim == 1:
        values = values[:, None]
    answer = np.full(values.shape, np.nan)
    for donor in pd.unique(donors):
        rows = np.flatnonzero(donors == donor)
        for column in range(values.shape[1]):
            answer[rows, column] = rank_percentile(values[rows, column])
    return answer


def residualize_standardized(y: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    """Residualize standardized/ranked y on ranked covariates, then rescale."""
    y = np.asarray(y, float)
    covariates = np.asarray(covariates, float)
    if covariates.ndim == 1:
        covariates = covariates[:, None]
    finite_columns = np.isfinite(covariates).all(axis=0)
    covariates = covariates[:, finite_columns]
    if covariates.shape[1]:
        covariates = covariates[:, np.nanstd(covariates, axis=0) > 1e-12]
    if covariates.shape[1]:
        design = np.column_stack([np.ones(len(y)), covariates])
        residual = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    else:
        residual = y - np.nanmean(y)
    scale = np.sqrt(np.nanmean(residual * residual))
    if not np.isfinite(scale) or scale <= 0:
        return np.full(len(y), np.nan)
    return residual / scale


def common_partial_rank(
    marker_values: np.ndarray,
    score: np.ndarray,
    covariates: np.ndarray,
    donors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    marker_values = np.asarray(marker_values, float)
    score = np.asarray(score, float)
    covariates = np.asarray(covariates, float)
    marker_residual = np.full(marker_values.shape, np.nan)
    score_residual = np.full(len(score), np.nan)
    for donor in pd.unique(donors):
        rows = np.flatnonzero(donors == donor)
        covariate_rank = np.column_stack(
            [rank_z(covariates[rows, col]) for col in range(covariates.shape[1])]
        )
        score_residual[rows] = residualize_standardized(
            rank_z(score[rows]), covariate_rank
        )
        for column in range(marker_values.shape[1]):
            marker_residual[rows, column] = residualize_standardized(
                rank_z(marker_values[rows, column]), covariate_rank
            )
    return marker_residual, score_residual


def marker_specific_partial_rank(
    marker_values: np.ndarray,
    score: np.ndarray,
    covariate_by_marker: np.ndarray,
    donors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    marker_values = np.asarray(marker_values, float)
    covariate_by_marker = np.asarray(covariate_by_marker, float)
    marker_residual = np.full(marker_values.shape, np.nan)
    score_residual = np.full(marker_values.shape, np.nan)
    for donor in pd.unique(donors):
        rows = np.flatnonzero(donors == donor)
        score_rank = rank_z(score[rows])
        for column in range(marker_values.shape[1]):
            covariate_rank = rank_z(covariate_by_marker[rows, column])[:, None]
            marker_residual[rows, column] = residualize_standardized(
                rank_z(marker_values[rows, column]), covariate_rank
            )
            score_residual[rows, column] = residualize_standardized(
                score_rank, covariate_rank
            )
    return marker_residual, score_residual


def summarize_variant(
    variant: str,
    marker_names: np.ndarray,
    donors: np.ndarray,
    projected_score: np.ndarray,
    marker_signal_z: np.ndarray,
    projected_signal_z: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    marker_signal_z = np.asarray(marker_signal_z, float)
    projected_signal_z = np.asarray(projected_signal_z, float)
    if projected_signal_z.ndim == 1:
        projected_signal_z = np.repeat(
            projected_signal_z[:, None], len(marker_names), axis=1
        )
    donor_names = sorted(pd.unique(donors), key=str)
    score_percentile = percentile_matrix(projected_score, donors)[:, 0]
    marker_percentile = percentile_matrix(marker_signal_z, donors)
    donor_rows: list[dict[str, object]] = []
    donor_association = np.full((len(donor_names), len(marker_names)), np.nan)

    for donor_index, donor in enumerate(donor_names):
        rows = np.flatnonzero(donors == donor)
        high = score_percentile[rows] >= 0.75
        low = score_percentile[rows] <= 0.25
        for marker_index, marker in enumerate(marker_names):
            x = marker_signal_z[rows, marker_index]
            y = projected_signal_z[rows, marker_index]
            valid = np.isfinite(x) & np.isfinite(y)
            association = float(np.mean(x[valid] * y[valid])) if valid.any() else np.nan
            donor_association[donor_index, marker_index] = association
            quartile_effect = float(
                np.nanmean(marker_percentile[rows, marker_index][high])
                - np.nanmean(marker_percentile[rows, marker_index][low])
            )
            donor_rows.append(
                {
                    "variant": variant,
                    "marker_raw": marker,
                    "donor": donor,
                    "cells": int(len(rows)),
                    "association": association,
                    "top_minus_bottom_projected_quartile_marker_rank": quartile_effect,
                }
            )

    equal_donor_mean = np.nanmean(donor_association, axis=0)
    rank_107 = (
        pd.Series(equal_donor_mean)
        .rank(ascending=False, method="min")
        .astype(int)
        .to_numpy()
    )
    summary = pd.DataFrame(
        {
            "marker_raw": marker_names,
            f"{variant}__mean_donor_association": equal_donor_mean,
            f"{variant}__median_donor_association": np.nanmedian(
                donor_association, axis=0
            ),
            f"{variant}__min_donor_association": np.nanmin(
                donor_association, axis=0
            ),
            f"{variant}__positive_donors": np.sum(donor_association > 0, axis=0),
            f"{variant}__rank_107": rank_107,
        }
    )
    lodo_rows: list[dict[str, object]] = []
    for donor_index, donor in enumerate(donor_names):
        estimate = np.nanmean(np.delete(donor_association, donor_index, axis=0), axis=0)
        summary[f"{variant}__lodo_holdout_{donor}"] = estimate
        for marker_index, marker in enumerate(marker_names):
            lodo_rows.append(
                {
                    "variant": variant,
                    "marker_raw": marker,
                    "held_out_donor": donor,
                    "remaining_donors": len(donor_names) - 1,
                    "equal_donor_association": estimate[marker_index],
                }
            )
    lodo_columns = [
        column for column in summary if column.startswith(f"{variant}__lodo_holdout_")
    ]
    summary[f"{variant}__min_lodo_association"] = summary[lodo_columns].min(axis=1)
    return summary, pd.DataFrame(donor_rows), pd.DataFrame(lodo_rows)


def merge_summaries(summaries: Iterable[pd.DataFrame]) -> pd.DataFrame:
    iterator = iter(summaries)
    merged = next(iterator)
    for frame in iterator:
        merged = merged.merge(frame, on="marker_raw", validate="one_to_one")
    return merged


def equal_donor_marker_correlations(
    clr: np.ndarray, marker_names: np.ndarray, donors: np.ndarray
) -> pd.DataFrame:
    donor_names = sorted(pd.unique(donors), key=str)
    donor_correlations = []
    for donor in donor_names:
        donor_rows = donors == donor
        ranked = np.column_stack(
            [rank_z(clr[donor_rows, col]) for col in range(clr.shape[1])]
        )
        donor_correlations.append((ranked.T @ ranked) / donor_rows.sum())
    equal_donor = np.mean(donor_correlations, axis=0)
    rows: list[dict[str, object]] = []
    for left in range(len(marker_names)):
        for right in range(left + 1, len(marker_names)):
            row: dict[str, object] = {
                "marker_a": marker_names[left],
                "marker_b": marker_names[right],
                "equal_donor_mean_spearman": equal_donor[left, right],
            }
            for donor_index, donor in enumerate(donor_names):
                row[f"spearman_donor_{donor}"] = donor_correlations[donor_index][
                    left, right
                ]
            rows.append(row)
    return pd.DataFrame(rows)


def redundancy_prune(
    candidates: pd.DataFrame,
    pairwise: pd.DataFrame,
    threshold: float = REDUNDANCY_ABS_RHO,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = candidates.sort_values(
        ["robust_median_rank_primary_variants", "paper_clr__rank_107", "marker_raw"],
        kind="stable",
    ).copy()
    correlations: dict[tuple[str, str], float] = {}
    for row in pairwise.itertuples(index=False):
        correlations[(row.marker_a, row.marker_b)] = row.equal_donor_mean_spearman
        correlations[(row.marker_b, row.marker_a)] = row.equal_donor_mean_spearman
    retained: list[str] = []
    mapping: list[dict[str, object]] = []
    for marker in ordered.marker_raw:
        matches = [
            (kept, correlations.get((marker, kept), np.nan))
            for kept in retained
            if np.isfinite(correlations.get((marker, kept), np.nan))
            and abs(correlations[(marker, kept)]) >= threshold
        ]
        if not matches:
            retained.append(marker)
            mapping.append(
                {
                    "marker_raw": marker,
                    "retained": True,
                    "redundant_to": "",
                    "equal_donor_mean_spearman": np.nan,
                }
            )
        else:
            redundant_to, rho = max(matches, key=lambda item: abs(item[1]))
            mapping.append(
                {
                    "marker_raw": marker,
                    "retained": False,
                    "redundant_to": redundant_to,
                    "equal_donor_mean_spearman": rho,
                }
            )
    map_frame = pd.DataFrame(mapping)
    pruned = ordered.loc[ordered.marker_raw.isin(retained)].copy()
    return pruned, map_frame


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    png = FIGURE_DIR / f"{stem}.png"
    pdf = FIGURE_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def plot_rank_overview(screen: pd.DataFrame) -> list[Path]:
    fig, axis = plt.subplots(figsize=(8.0, 5.8))
    axis.scatter(
        screen.robust_median_rank_primary_variants,
        screen.paper_clr__mean_donor_association,
        s=20,
        color="#B8B8B8",
        alpha=0.65,
        linewidths=0,
        label="All 107 candidates",
        zorder=1,
    )
    stage_styles = (
        ("passes_frozen_rule_pre_redundancy", "#E69F00", "Robust before pruning", 35),
        ("selected_after_redundancy_pruning", "#0072B2", "After redundancy pruning", 46),
        ("selected_compact_top5pct", "#D55E00", "Final compact markers", 66),
    )
    for column, color, label, size in stage_styles:
        block = screen.loc[screen[column]]
        if block.empty:
            continue
        axis.scatter(
            block.robust_median_rank_primary_variants,
            block.paper_clr__mean_donor_association,
            s=size,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            label=label,
            zorder=size,
        )
    for row in screen.loc[screen.selected_compact_top5pct].itertuples(index=False):
        axis.annotate(
            row.marker,
            (row.robust_median_rank_primary_variants, row.paper_clr__mean_donor_association),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            weight="bold",
        )
    axis.axhline(0, color="#777777", linewidth=0.8, linestyle="--")
    axis.axvline(
        ROBUST_RANK_CUTOFF,
        color="#999999",
        linewidth=0.9,
        linestyle=":",
        label="Robust rank cutoff (15)",
    )
    axis.axvline(
        COMPACT_RANK_CUTOFF,
        color="#D55E00",
        linewidth=0.9,
        linestyle="--",
        label=f"Compact cutoff ({COMPACT_RANK_CUTOFF:.2f}/107)",
    )
    axis.set_xlim(0, EXPECTED_CANDIDATE_N + 2)
    axis.set_xlabel("Median rank across four primary views (original 107-marker scale)")
    axis.set_ylabel("Equal-donor paper-CLR association")
    axis.set_title("Outcome-blind Rahim ADT biomarker screen", loc="left", weight="bold")
    axis.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    return save_figure(fig, "hnscc_biomarker_rank_overview")


def plot_donor_robustness(
    screen: pd.DataFrame,
    pruned: pd.DataFrame,
    donor_effects: pd.DataFrame,
) -> list[Path]:
    focus = pruned.marker_raw.tolist()
    focus_label = "robust markers retained after redundancy pruning"
    if not focus:
        focus = screen.head(10).marker_raw.tolist()
        focus_label = "top ten candidates (no marker passed all gates)"
    display = dict(zip(screen.marker_raw, screen.marker))
    donor_names = sorted(pd.unique(donor_effects.donor.astype(str)), key=str)
    donor_colors = dict(
        zip(donor_names, ("#0072B2", "#D55E00", "#009E73", "#CC79A7"))
    )
    variant_labels = {
        "raw_rank": "Raw rank",
        "paper_clr": "Strict no-LAG3 CLR",
        "isotype_adjusted": "Isotype-adjusted",
        "global_adt_loo_adjusted": "Global-ADT LOO",
    }
    fig_height = max(4.4, 0.38 * len(focus) + 1.5)
    fig, axes = plt.subplots(
        1, len(PRIMARY_VARIANTS), figsize=(13.0, fig_height), sharex=False, sharey=True
    )
    y = np.arange(len(focus))
    for axis, variant in zip(axes, PRIMARY_VARIANTS):
        block = donor_effects.loc[
            donor_effects.variant.eq(variant) & donor_effects.marker_raw.isin(focus)
        ].copy()
        for donor in donor_names:
            donor_block = block.loc[block.donor.astype(str).eq(donor)].set_index(
                "marker_raw"
            )
            values = donor_block.reindex(focus).association.to_numpy(float)
            axis.scatter(
                values,
                y,
                s=24,
                alpha=0.80,
                color=donor_colors[donor],
                linewidths=0,
                label=f"Donor {donor}",
            )
        means = block.groupby("marker_raw", sort=False).association.mean().reindex(focus)
        axis.scatter(
            means.to_numpy(float),
            y,
            marker="D",
            s=27,
            facecolor="black",
            edgecolor="white",
            linewidth=0.5,
            label="Equal-donor mean",
            zorder=10,
        )
        axis.axvline(0, color="#777777", linewidth=0.8, linestyle="--")
        axis.set_title(variant_labels[variant], fontsize=10, weight="bold")
        axis.set_xlabel("Within-donor association")
        axis.grid(axis="y", color="#EEEEEE", linewidth=0.6)
    axes[0].set_yticks(y, [display.get(marker, marker) for marker in focus])
    axes[0].set_ylim(len(focus) - 0.5, -0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=len(labels),
    )
    fig.suptitle(
        f"Donor consistency across frozen primary views: {focus_label}",
        x=0.01,
        y=0.985,
        ha="left",
        fontsize=14,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    return save_figure(fig, "hnscc_biomarker_donor_robustness")


def main() -> None:
    SCREEN_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    score, score_column = read_score(PROJECTION)
    adt_obs, all_markers, adt_all = load_adt()
    rna_obs = load_rna_obs()
    cells, adt = join_inputs(
        score, score_column, adt_obs, all_markers, adt_all, rna_obs
    )
    donors = cells.donor.to_numpy(str)
    projected = cells.projected_score.to_numpy(float)

    isotype_mask = np.asarray(["isotype" in marker.lower() for marker in all_markers])
    lag3_index = np.flatnonzero(all_markers == HOLDOUT_MARKER)
    if len(lag3_index) != 1:
        raise AssertionError(f"Expected exactly one {HOLDOUT_MARKER} channel")
    background_mask = ~isotype_mask & (all_markers != HOLDOUT_MARKER)
    candidate_mask = np.asarray(
        [
            marker not in SHARED_RAHIM_MARKERS
            and marker != HOLDOUT_MARKER
            and "isotype" not in marker.lower()
            for marker in all_markers
        ]
    )
    marker_names = all_markers[candidate_mask]
    if isotype_mask.sum() != EXPECTED_ISOTYPE_N:
        raise AssertionError(
            f"Expected {EXPECTED_ISOTYPE_N} isotypes; got {isotype_mask.sum()}"
        )
    if background_mask.sum() != EXPECTED_BACKGROUND_N:
        raise AssertionError(
            f"Expected {EXPECTED_BACKGROUND_N} strict CLR-background ADTs; "
            f"got {background_mask.sum()}"
        )
    if len(marker_names) != EXPECTED_CANDIDATE_N:
        raise AssertionError(
            f"Expected {EXPECTED_CANDIDATE_N} candidate ADTs; got {len(marker_names)}"
        )
    if not np.all(background_mask[candidate_mask]):
        raise AssertionError("Every screened marker must occur in the CLR background")
    if background_mask[lag3_index[0]]:
        raise AssertionError("LAG3 unexpectedly remained in the strict background")

    raw = adt[:, candidate_mask]
    log_all = np.log1p(adt)
    log_candidate = log_all[:, candidate_mask]
    background_log_sum = log_all[:, background_mask].sum(axis=1)
    paper_clr = log_candidate - (
        background_log_sum / EXPECTED_BACKGROUND_N
    )[:, None]
    isotype_burden = log_all[:, isotype_mask].mean(axis=1)
    global_loo = (
        background_log_sum[:, None] - log_candidate
    ) / EXPECTED_GLOBAL_LOO_N

    raw_z = rank_z_matrix(raw, donors)
    clr_z = rank_z_matrix(paper_clr, donors)
    projected_z = rank_z_matrix(projected, donors)[:, 0]
    isotype_marker_z, isotype_projected_z = common_partial_rank(
        paper_clr, projected, isotype_burden[:, None], donors
    )
    global_marker_z, global_projected_z = marker_specific_partial_rank(
        log_candidate, projected, global_loo, donors
    )
    rna_covariates = np.column_stack(
        [
            np.log1p(cells.rna_n_counts.to_numpy(float)),
            np.log1p(cells.rna_n_genes.to_numpy(float)),
            cells.rna_percent_mito.to_numpy(float),
        ]
    )
    rna_marker_z, rna_projected_z = common_partial_rank(
        paper_clr, projected, rna_covariates, donors
    )
    variant_inputs: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "raw_rank": (raw_z, projected_z),
        "paper_clr": (clr_z, projected_z),
        "isotype_adjusted": (isotype_marker_z, isotype_projected_z),
        "global_adt_loo_adjusted": (global_marker_z, global_projected_z),
        "rna_qc_adjusted": (rna_marker_z, rna_projected_z),
    }

    summaries: list[pd.DataFrame] = []
    donor_effect_frames: list[pd.DataFrame] = []
    lodo_effect_frames: list[pd.DataFrame] = []
    for variant, (marker_signal, score_signal) in variant_inputs.items():
        summary, donor_effect, lodo_effect = summarize_variant(
            variant,
            marker_names,
            donors,
            projected,
            marker_signal,
            score_signal,
        )
        summaries.append(summary)
        donor_effect_frames.append(donor_effect)
        lodo_effect_frames.append(lodo_effect)

    explicit_singlet = cells.droplet_type.str.upper().isin({"SNG", "SINGLET"}).to_numpy()
    singlet_donors = donors[explicit_singlet]
    if explicit_singlet.sum() and pd.unique(singlet_donors).size >= 2:
        singlet_clr_z = rank_z_matrix(paper_clr[explicit_singlet], singlet_donors)
        singlet_projected_z = rank_z_matrix(
            projected[explicit_singlet], singlet_donors
        )[:, 0]
        summary, donor_effect, lodo_effect = summarize_variant(
            "paper_clr_explicit_singlets",
            marker_names,
            singlet_donors,
            projected[explicit_singlet],
            singlet_clr_z,
            singlet_projected_z,
        )
        summaries.append(summary)
        donor_effect_frames.append(donor_effect)
        lodo_effect_frames.append(lodo_effect)

    screen = merge_summaries(summaries)
    screen.insert(
        1,
        "marker",
        [FRIENDLY_NAMES.get(marker, marker) for marker in screen.marker_raw],
    )
    for marker_index, marker in enumerate(marker_names):
        selected_row = screen.marker_raw.eq(marker)
        screen.loc[selected_row, "raw_detection_fraction"] = np.mean(
            raw[:, marker_index] > 0
        )
        screen.loc[selected_row, "raw_median_count"] = np.median(raw[:, marker_index])

    primary_rank_columns = [f"{variant}__rank_107" for variant in PRIMARY_VARIANTS]
    screen["robust_median_rank_primary_variants"] = screen[
        primary_rank_columns
    ].median(axis=1)
    primary_lodo_columns = [
        column
        for column in screen
        if any(
            column.startswith(f"{variant}__lodo_holdout_")
            for variant in PRIMARY_VARIANTS
        )
    ]
    screen["robust_min_lodo_all_primary_variants"] = screen[
        primary_lodo_columns
    ].min(axis=1)
    screen["rule_clr_positive_3of3"] = screen["paper_clr__positive_donors"] == 3
    screen["rule_isotype_positive_3of3"] = (
        screen["isotype_adjusted__positive_donors"] == 3
    )
    screen["rule_global_adt_positive_at_least_2of3"] = (
        screen["global_adt_loo_adjusted__positive_donors"] >= 2
    )
    screen["rule_median_primary_rank_at_most_15"] = (
        screen["robust_median_rank_primary_variants"] <= ROBUST_RANK_CUTOFF
    )
    screen["rule_all_primary_lodo_positive"] = (
        screen["robust_min_lodo_all_primary_variants"] > 0
    )
    screen["passes_frozen_rule_pre_redundancy"] = screen[list(RULE_COLUMNS)].all(
        axis=1
    )
    screen["frozen_rule_failure_reasons"] = screen.apply(
        lambda row: ";".join(column for column in RULE_COLUMNS if not row[column]),
        axis=1,
    )
    screen = screen.sort_values(
        ["robust_median_rank_primary_variants", "paper_clr__rank_107", "marker_raw"],
        kind="stable",
    ).reset_index(drop=True)

    pairwise = equal_donor_marker_correlations(paper_clr, marker_names, donors)
    robust = screen.loc[screen.passes_frozen_rule_pre_redundancy].copy()
    pruned, redundancy_map = redundancy_prune(robust, pairwise)
    screen["selected_after_redundancy_pruning"] = screen.marker_raw.isin(
        pruned.marker_raw
    )

    # The compact rule is deliberately applied only after both preceding gates.
    # The rank is not recomputed: it remains the original 107-candidate rank.
    compact = pruned.loc[
        pruned.robust_median_rank_primary_variants <= COMPACT_RANK_CUTOFF
    ].copy()
    compact.insert(0, "compact_selection_order", np.arange(1, len(compact) + 1))
    compact.insert(1, "compact_rank_denominator", COMPACT_RANK_DENOMINATOR)
    compact.insert(2, "compact_rank_fraction", COMPACT_FRACTION)
    compact.insert(3, "compact_rank_cutoff", COMPACT_RANK_CUTOFF)
    compact.insert(4, "reranked_after_gates", False)
    screen["selected_compact_top5pct"] = screen.marker_raw.isin(compact.marker_raw)
    screen["selection_stage"] = np.select(
        [
            screen.selected_compact_top5pct,
            screen.selected_after_redundancy_pruning,
            screen.passes_frozen_rule_pre_redundancy,
        ],
        [
            "final_compact_top5pct",
            "robust_redundancy_pruned_above_compact_cutoff",
            "robust_but_redundant",
        ],
        default="failed_robustness_gate",
    )
    compact_from_screen = screen.loc[screen.selected_compact_top5pct, "marker_raw"].tolist()
    if compact.marker_raw.tolist() != compact_from_screen:
        raise AssertionError("Compact-table order or membership disagrees with screen audit")

    donor_effects = pd.concat(donor_effect_frames, ignore_index=True)
    lodo_effects = pd.concat(lodo_effect_frames, ignore_index=True)
    output_paths: list[Path] = []
    tables = {
        "marker_screen_all_107_all_variants.csv": screen,
        "marker_donor_effects_long.csv": donor_effects,
        "marker_leave_one_donor_out_long.csv": lodo_effects,
        "marker_pairwise_paper_clr_correlations.csv": pairwise,
        "robust_candidates_pre_redundancy.csv": robust,
        "robust_candidates_redundancy_pruned.csv": pruned,
        "redundancy_map.csv": redundancy_map,
        "final_compact_markers_top5pct.csv": compact,
    }
    for filename, frame in tables.items():
        path = SCREEN_DIR / filename
        frame.to_csv(path, index=False)
        output_paths.append(path)

    clr_percentiles = percentile_matrix(paper_clr, donors)
    pruned_indices = np.asarray(
        [np.flatnonzero(marker_names == marker)[0] for marker in pruned.marker_raw], int
    )
    compact_indices = np.asarray(
        [np.flatnonzero(marker_names == marker)[0] for marker in compact.marker_raw], int
    )
    robust_program = (
        np.mean(clr_percentiles[:, pruned_indices], axis=1)
        if len(pruned_indices)
        else np.full(len(cells), np.nan)
    )
    compact_program = (
        np.mean(clr_percentiles[:, compact_indices], axis=1)
        if len(compact_indices)
        else np.full(len(cells), np.nan)
    )
    source_state = cells[["cell_uid", "donor"]].copy()
    source_state["projected_roi_adjusted_exposure"] = projected
    source_state["robust_redundancy_pruned_clr_program"] = robust_program
    source_state["compact_top5pct_clr_program"] = compact_program
    source_state_path = SCREEN_DIR / "rahim_tumor_cd8_biomarker_programs.csv.gz"
    write_csv_gz(source_state, source_state_path)
    output_paths.append(source_state_path)

    paper_qc_pass = (
        cells.rna_n_genes.between(100, 3000, inclusive="both")
        & (cells.rna_percent_mito <= 0.10)
        & ~cells.droplet_type.str.upper().isin({"DBL", "DOUBLET"})
    )
    qc_by_donor = (
        cells.assign(
            explicit_singlet=explicit_singlet,
            doublet_call_available=cells.droplet_type.ne("__MISSING__"),
            paper_qc_pass=paper_qc_pass,
        )
        .groupby("donor", sort=True)
        .agg(
            cells=("cell_uid", "size"),
            explicit_singlets=("explicit_singlet", "sum"),
            doublet_call_available=("doublet_call_available", "sum"),
            paper_qc_pass=("paper_qc_pass", "sum"),
            median_rna_counts=("rna_n_counts", "median"),
            median_rna_genes=("rna_n_genes", "median"),
            median_rna_percent_mito=("rna_percent_mito", "median"),
            median_adt_library_size_137=("adt_library_size_137", "median"),
        )
        .reset_index()
    )
    qc_path = SCREEN_DIR / "cell_quality_audit_by_donor.csv"
    qc_by_donor.to_csv(qc_path, index=False)
    output_paths.append(qc_path)

    output_paths.extend(plot_rank_overview(screen))
    output_paths.extend(plot_donor_robustness(screen, pruned, donor_effects))

    selected_metric_columns = [
        "marker_raw",
        "marker",
        "robust_median_rank_primary_variants",
        "raw_rank__rank_107",
        "paper_clr__rank_107",
        "isotype_adjusted__rank_107",
        "global_adt_loo_adjusted__rank_107",
        "paper_clr__mean_donor_association",
        "robust_min_lodo_all_primary_variants",
    ]
    donor_counts = {
        str(donor): int(count)
        for donor, count in cells.donor.value_counts().sort_index().items()
    }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "analysis": "formal PT74 outcome-blind Rahim 107-ADT biomarker screen",
        "analysis_role": (
            "Formal same-dataset reproduction of a previously developed rule; "
            "not an independent validation cohort"
        ),
        "outcome_blind": True,
        "survival_outcomes_read": False,
        "forced_marker_names": False,
        "forced_marker_count": False,
        "projection": {
            "path": str(PROJECTION.resolve()),
            "sha256": sha256(PROJECTION),
            "score_column": score_column,
        },
        "copied_complete_rahim_inputs": {
            "ADT": {"path": str(ADT_H5AD.resolve()), "sha256": sha256(ADT_H5AD)},
            "RNA": {"path": str(RNA_H5AD.resolve()), "sha256": sha256(RNA_H5AD)},
        },
        "cells": int(len(cells)),
        "donors": sorted(pd.unique(donors), key=str),
        "donor_cell_counts": donor_counts,
        "adt_panel_markers": int(len(all_markers)),
        "isotype_controls": int(isotype_mask.sum()),
        "candidate_universe_n": int(len(marker_names)),
        "candidate_definition": (
            "All Rahim ADTs absent from the 22-marker spatial panel, excluding "
            "seven isotype controls and the prespecified LAG3 holdout"
        ),
        "holdout_not_screened": HOLDOUT_MARKER,
        "background_mode": "strict_no_lag3",
        "clr_background_marker_n": int(background_mask.sum()),
        "global_adt_loo_nuisance_marker_n": EXPECTED_GLOBAL_LOO_N,
        "lag3_in_clr_or_global_background": False,
        "formulas": {
            "paper_clr": (
                "log(count+1) minus the within-cell mean log(count+1) across "
                "129 non-isotype ADTs after excluding LAG3"
            ),
            "association": (
                "Equal-weight mean across donors of within-donor Spearman/partial-"
                "Spearman associations; tied ranks use average ranks"
            ),
            "isotype_adjustment": (
                "Within-donor partial Spearman controlling mean log1p count across "
                "the seven isotype controls"
            ),
            "global_adt_loo_adjustment": (
                "Marker-specific within-donor partial Spearman controlling mean "
                "log1p count across the other 128 strict-background ADTs"
            ),
            "rna_qc_sensitivity": (
                "Within-donor partial Spearman controlling log1p RNA n_counts, "
                "log1p RNA n_genes, and RNA mitochondrial fraction"
            ),
        },
        "frozen_robustness_rule": {
            "primary_variants": list(PRIMARY_VARIANTS),
            "paper_clr_positive_donors": "3/3",
            "isotype_adjusted_positive_donors": "3/3",
            "global_adt_loo_positive_donors": ">=2/3",
            "robust_median_rank_cutoff": ROBUST_RANK_CUTOFF,
            "all_primary_variant_by_lodo_associations_positive": True,
        },
        "redundancy_rule": {
            "threshold": f"absolute equal-donor paper-CLR Spearman >= {REDUNDANCY_ABS_RHO}",
            "order": [
                "robust_median_rank_primary_variants ascending",
                "paper_clr__rank_107 ascending",
                "marker_raw ascending",
            ],
        },
        "compact_rule": {
            "application_order": [
                "frozen robustness gates",
                "redundancy pruning",
                "top-5% rank cutoff",
            ],
            "applied_to": "robust candidates retained after redundancy pruning",
            "rank_denominator": COMPACT_RANK_DENOMINATOR,
            "fraction": COMPACT_FRACTION,
            "rank_cutoff": COMPACT_RANK_CUTOFF,
            "reranked_after_robustness_or_redundancy": False,
            "forced_marker_names": False,
            "forced_marker_count": False,
        },
        "stage_counts": {
            "screened": int(len(screen)),
            "passed_robustness_pre_redundancy": int(len(robust)),
            "retained_after_redundancy": int(len(pruned)),
            "final_compact_top5pct": int(len(compact)),
        },
        "robust_markers_pre_redundancy": robust.marker_raw.tolist(),
        "markers_after_redundancy": pruned.marker_raw.tolist(),
        "final_compact_markers": compact.marker_raw.tolist(),
        "final_compact_marker_metrics": compact[selected_metric_columns].to_dict(
            orient="records"
        ),
        "population_pvalues_calculated": False,
        "quality_sensitivities_used_for_selection": False,
        "explicit_singlet_sensitivity": {
            "cells": int(explicit_singlet.sum()),
            "donors": sorted(pd.unique(singlet_donors), key=str),
        },
        "paper_qc_pass_cells": int(paper_qc_pass.sum()),
        "outputs": {
            str(path.relative_to(HERE)): {"sha256": sha256(path)}
            for path in output_paths
        },
    }
    manifest_path = SCREEN_DIR / "screen_manifest.json"
    write_json(manifest, manifest_path)

    summary = {
        "outcome_blind": True,
        "forced_marker_names": False,
        "forced_marker_count": False,
        "cells": int(len(cells)),
        "donors": int(pd.unique(donors).size),
        "screened_markers": int(len(screen)),
        "robust_pre_redundancy_n": int(len(robust)),
        "redundancy_pruned_n": int(len(pruned)),
        "compact_rank_denominator": COMPACT_RANK_DENOMINATOR,
        "compact_rank_cutoff": COMPACT_RANK_CUTOFF,
        "reranked_after_gates": False,
        "final_compact_marker_n": int(len(compact)),
        "final_compact_markers": compact.marker_raw.tolist(),
        "manifest": str(manifest_path.resolve()),
    }
    summary_path = SCREEN_DIR / "RUN_SUMMARY.json"
    write_json(summary, summary_path)
    print(json.dumps(json_ready(summary), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
