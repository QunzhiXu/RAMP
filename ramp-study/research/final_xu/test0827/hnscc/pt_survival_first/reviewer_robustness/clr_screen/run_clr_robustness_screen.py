#!/usr/bin/env python3
"""Reviewer-robust Rahim ADT marker screen.

The script is deliberately downstream of RAMP: it consumes a projected score
for Rahim tumor CD8 cells, joins that score to the original 137-ADT and RNA
H5AD objects by cell barcode, and compares four predeclared marker transforms:

1. raw ADT within-donor ranks (legacy analysis),
2. paper-matched cell-wise CLR after excluding seven isotype controls,
3. CLR partial-rank adjusted for the seven-isotype burden, and
4. raw log1p ADT partial-rank adjusted for a leave-one-marker-out global ADT
   burden.

RNA-depth/complexity/mitochondrial adjustment and an explicit genetic-singlet
subset are reported as sensitivities and are not used for marker selection.

Only donor-blocked descriptive estimates are reported. With three donors, no
cell-level permutation P value or BH q value is interpreted as population
evidence, so this script does not calculate either.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata


HERE = Path(__file__).resolve().parent
FINAL_XU = HERE.parents[4]
RAHIM_DIR = FINAL_XU / "raw_data/hnscc/rahim_2023_gse212797_citeseq"
ADT_H5AD = RAHIM_DIR / "GSE212797_adata_ADT.h5ad"
RNA_H5AD = RAHIM_DIR / "GSE212797_adata.h5ad"
LEGACY_SCORE = (
    FINAL_XU
    / "test0827/hnscc/pt_survival_first/output/reverse_primary_rahim_tumor_cd8.csv.gz"
)
RULE_PATH = HERE / "frozen_selection_rule.json"

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
SCORE_COLUMN_PRIORITY = (
    "projected_roi_adjusted_exposure",
    "projected_pt_cd8_niche_rank",
    "projected_rank_within_rahim_donor",
    "projected_score",
    "score",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--score-file",
        type=Path,
        default=None,
        help="CSV/CSV.GZ with cell_uid and projected Rahim score. If omitted, "
        "look for a canonical reviewer-robustness output, then use the legacy "
        "development score.",
    )
    parser.add_argument(
        "--score-column",
        default=None,
        help="Projected-score column. Known names are detected automatically.",
    )
    parser.add_argument(
        "--run-label",
        default="development",
        help="Output subdirectory below output/ (default: development).",
    )
    parser.add_argument(
        "--background-mode",
        choices=("paper_all_nonisotype", "no_lag3"),
        default="paper_all_nonisotype",
        help=(
            "ADT background used for CLR and global-ADT nuisance. The default "
            "matches the original paper-matched implementation (all 130 "
            "non-isotypes). 'no_lag3' is a post-hoc strict sensitivity that "
            "excludes CD223--LAG-3 from every background calculation."
        ),
    )
    return parser.parse_args()


def decode(values: np.ndarray) -> np.ndarray:
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
    text_missing = np.isin(
        np.char.lower(answer.astype(str)), ["nan", "na", "none", "null", ""]
    )
    answer[text_missing] = "__MISSING__"
    return answer


def read_csr(group: h5py.Group, n_cols: int) -> sparse.csr_matrix:
    return sparse.csr_matrix(
        (group["data"][:], group["indices"][:], group["indptr"][:]),
        shape=(len(group["indptr"]) - 1, n_cols),
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_score_file(explicit: Path | None) -> tuple[Path, str]:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path, "explicit_cli"

    canonical_root = HERE.parent / "canonical_ramp"
    exact_candidates = [
        canonical_root / "output/rahim_tumor_cd8_projected_scores.csv.gz",
        canonical_root / "output/reverse_primary_rahim_tumor_cd8.csv.gz",
        canonical_root / "output/rahim_tumor_cd8_projection.csv.gz",
        canonical_root / "rahim_tumor_cd8_projected_scores.csv.gz",
    ]
    existing = [path.resolve() for path in exact_candidates if path.exists()]
    if len(existing) == 1:
        return existing[0], "canonical_exact_name"
    if len(existing) > 1:
        raise RuntimeError(
            "Multiple canonical score files found; select one with --score-file: "
            + ", ".join(map(str, existing))
        )
    if not LEGACY_SCORE.exists():
        raise FileNotFoundError(
            "No canonical exact-name score or legacy score found; use --score-file."
        )
    return LEGACY_SCORE.resolve(), "legacy_development_fallback"


def read_score(path: Path, requested_column: str | None) -> tuple[pd.DataFrame, str]:
    score = pd.read_csv(path, dtype={"cell_uid": str, "donor": str})
    if "cell_uid" not in score:
        aliases = [name for name in ("UID", "barcode", "cell_id") if name in score]
        if len(aliases) != 1:
            raise KeyError("Score file must contain cell_uid or one unambiguous ID alias")
        score = score.rename(columns={aliases[0]: "cell_uid"})
    if score.cell_uid.duplicated().any():
        duplicate = score.loc[score.cell_uid.duplicated(), "cell_uid"].iloc[0]
        raise ValueError(f"Duplicate score cell_uid: {duplicate}")

    if requested_column is not None:
        if requested_column not in score:
            raise KeyError(f"Requested score column not found: {requested_column}")
        score_column = requested_column
    else:
        known = [column for column in SCORE_COLUMN_PRIORITY if column in score]
        if known:
            score_column = known[0]
        else:
            numeric = [
                column
                for column in score.select_dtypes(include=[np.number]).columns
                if any(token in column.lower() for token in ("score", "project", "predict"))
            ]
            if len(numeric) != 1:
                raise KeyError(
                    "Could not infer one projected score column; use --score-column. "
                    f"Candidates: {numeric}"
                )
            score_column = numeric[0]
    score[score_column] = pd.to_numeric(score[score_column], errors="coerce")
    if not np.isfinite(score[score_column]).all():
        raise ValueError(f"Non-finite values in score column {score_column}")
    return score, score_column


def load_adt() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
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
    if not np.all(matrix >= 0) or not np.allclose(matrix, np.round(matrix)):
        raise AssertionError("ADT H5AD X is not a nonnegative raw-count matrix")
    obs["adt_library_size_137"] = matrix.sum(axis=1)
    return obs, markers, matrix


def load_rna_obs() -> pd.DataFrame:
    with h5py.File(RNA_H5AD, "r") as handle:
        return pd.DataFrame(
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
    missing_adt = len(score) - len(joined)
    if missing_adt:
        raise AssertionError(f"{missing_adt} score cells were absent from ADT/RNA H5ADs")
    joined = joined.loc[
        joined.rna_tissue.eq("tumor") & joined.rna_cell_type.eq("CD8 T cell")
    ].copy()
    if len(joined) == 0:
        raise AssertionError("No tumor CD8 cells remained after RNA metadata filter")
    if not joined.adt_tissue.eq("tumor").all():
        raise AssertionError("RNA/ADT tissue mismatch")
    if not joined.adt_donor.eq(joined.rna_donor).all():
        raise AssertionError("RNA/ADT donor mismatch")
    if "donor" in joined and not joined.donor.eq(joined.rna_donor).all():
        raise AssertionError("Score-file donor disagrees with original H5AD donor")
    joined["donor"] = joined.rna_donor.astype(str)
    joined = joined.sort_values(["donor", "cell_uid"], kind="stable").reset_index(drop=True)
    selected = adt_all[joined.adt_row.to_numpy(int)]
    if selected.shape != (len(joined), len(markers)):
        raise AssertionError("Joined ADT matrix shape mismatch")
    joined["projected_score"] = joined[score_column].to_numpy(float)
    return joined, selected


def rank_z(values: np.ndarray) -> np.ndarray:
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
    """Residualize standardized/ranked y on ranked covariates and re-standardize."""
    y = np.asarray(y, float)
    covariates = np.asarray(covariates, float)
    if covariates.ndim == 1:
        covariates = covariates[:, None]
    finite_columns = np.isfinite(covariates).all(axis=0)
    covariates = covariates[:, finite_columns]
    if covariates.shape[1]:
        variable = np.nanstd(covariates, axis=0) > 1e-12
        covariates = covariates[:, variable]
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
        cov_rank = np.column_stack(
            [rank_z(covariates[rows, column]) for column in range(covariates.shape[1])]
        )
        score_residual[rows] = residualize_standardized(rank_z(score[rows]), cov_rank)
        for column in range(marker_values.shape[1]):
            marker_residual[rows, column] = residualize_standardized(
                rank_z(marker_values[rows, column]), cov_rank
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
            cov_rank = rank_z(covariate_by_marker[rows, column])[:, None]
            marker_residual[rows, column] = residualize_standardized(
                rank_z(marker_values[rows, column]), cov_rank
            )
            score_residual[rows, column] = residualize_standardized(score_rank, cov_rank)
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
    donor_rho = np.full((len(donor_names), len(marker_names)), np.nan)

    for donor_index, donor in enumerate(donor_names):
        rows = np.flatnonzero(donors == donor)
        high = score_percentile[rows] >= 0.75
        low = score_percentile[rows] <= 0.25
        for marker_index, marker in enumerate(marker_names):
            x = marker_signal_z[rows, marker_index]
            y = projected_signal_z[rows, marker_index]
            valid = np.isfinite(x) & np.isfinite(y)
            rho = float(np.mean(x[valid] * y[valid])) if valid.any() else np.nan
            donor_rho[donor_index, marker_index] = rho
            effect = float(
                np.nanmean(marker_percentile[rows, marker_index][high])
                - np.nanmean(marker_percentile[rows, marker_index][low])
            )
            donor_rows.append(
                {
                    "variant": variant,
                    "marker_raw": marker,
                    "donor": donor,
                    "cells": int(len(rows)),
                    "association": rho,
                    "top_minus_bottom_projected_quartile_marker_rank": effect,
                }
            )

    mean_rho = np.nanmean(donor_rho, axis=0)
    rank = pd.Series(mean_rho).rank(ascending=False, method="min").astype(int).to_numpy()
    summary = pd.DataFrame(
        {
            "marker_raw": marker_names,
            f"{variant}__mean_donor_association": mean_rho,
            f"{variant}__median_donor_association": np.nanmedian(donor_rho, axis=0),
            f"{variant}__min_donor_association": np.nanmin(donor_rho, axis=0),
            f"{variant}__positive_donors": np.sum(donor_rho > 0, axis=0),
            f"{variant}__rank_107": rank,
        }
    )
    lodo_rows: list[dict[str, object]] = []
    for donor_index, donor in enumerate(donor_names):
        estimate = np.nanmean(np.delete(donor_rho, donor_index, axis=0), axis=0)
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
    donor_correlations = []
    donor_names = sorted(pd.unique(donors), key=str)
    for donor in donor_names:
        rows = donors == donor
        ranked = np.column_stack([rank_z(clr[rows, col]) for col in range(clr.shape[1])])
        donor_correlations.append((ranked.T @ ranked) / rows.sum())
    equal = np.mean(donor_correlations, axis=0)
    rows = []
    for left in range(len(marker_names)):
        for right in range(left + 1, len(marker_names)):
            row = {
                "marker_a": marker_names[left],
                "marker_b": marker_names[right],
                "equal_donor_mean_spearman": equal[left, right],
            }
            for donor_index, donor in enumerate(donor_names):
                row[f"spearman_donor_{donor}"] = donor_correlations[donor_index][left, right]
            rows.append(row)
    return pd.DataFrame(rows)


def redundancy_prune(
    candidates: pd.DataFrame, pairwise: pd.DataFrame, threshold: float = 0.85
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = candidates.sort_values(
        ["robust_median_rank_primary_variants", "paper_clr__rank_107", "marker_raw"],
        kind="stable",
    ).copy()
    correlation: dict[tuple[str, str], float] = {}
    for row in pairwise.itertuples(index=False):
        correlation[(row.marker_a, row.marker_b)] = row.equal_donor_mean_spearman
        correlation[(row.marker_b, row.marker_a)] = row.equal_donor_mean_spearman
    retained: list[str] = []
    mapping: list[dict[str, object]] = []
    for marker in ordered.marker_raw:
        matches = [
            (kept, correlation.get((marker, kept), np.nan))
            for kept in retained
            if np.isfinite(correlation.get((marker, kept), np.nan))
            and abs(correlation[(marker, kept)]) >= threshold
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


def main() -> None:
    args = parse_args()
    if not RULE_PATH.exists():
        raise FileNotFoundError("Frozen selection rule is required before running")
    if args.background_mode == "no_lag3" and args.run_label == "canonical":
        raise ValueError(
            "The post-hoc no-LAG3 sensitivity may not overwrite output/canonical; "
            "use --run-label canonical_no_lag3_background."
        )
    rule = json.loads(RULE_PATH.read_text())
    output = HERE / "output" / args.run_label
    output.mkdir(parents=True, exist_ok=True)

    score_path, score_source = discover_score_file(args.score_file)
    score, score_column = read_score(score_path, args.score_column)
    adt_obs, all_markers, adt_all = load_adt()
    rna_obs = load_rna_obs()
    cells, adt = join_inputs(score, score_column, adt_obs, all_markers, adt_all, rna_obs)
    donors = cells.donor.to_numpy(str)
    projected = cells.projected_score.to_numpy(float)

    isotype_mask = np.asarray(["isotype" in marker.lower() for marker in all_markers])
    nonisotype_mask = ~isotype_mask
    candidate_mask = np.asarray(
        [
            marker not in SHARED_RAHIM_MARKERS
            and marker != HOLDOUT_MARKER
            and "isotype" not in marker.lower()
            for marker in all_markers
        ]
    )
    marker_names = all_markers[candidate_mask]
    if isotype_mask.sum() != 7 or len(marker_names) != 107:
        raise AssertionError(
            f"Expected 7 isotypes and 107 screened markers; got "
            f"{isotype_mask.sum()} and {len(marker_names)}"
        )
    raw = adt[:, candidate_mask]
    log_all = np.log1p(adt)
    log_candidate = log_all[:, candidate_mask]
    background_mask = nonisotype_mask.copy()
    if args.background_mode == "no_lag3":
        background_mask &= all_markers != HOLDOUT_MARKER
    background_log_sum = log_all[:, background_mask].sum(axis=1)
    background_count = int(background_mask.sum())
    expected_background_count = 129 if args.background_mode == "no_lag3" else 130
    if background_count != expected_background_count:
        raise AssertionError(
            f"Expected {expected_background_count} CLR-background markers for "
            f"{args.background_mode}, got {background_count}"
        )
    if not np.all(background_mask[candidate_mask]):
        raise AssertionError("Every screened marker must be present in the ADT background")
    if args.background_mode == "no_lag3" and background_mask[
        np.flatnonzero(all_markers == HOLDOUT_MARKER)[0]
    ]:
        raise AssertionError("LAG3 unexpectedly remained in the strict background")
    clr_center = background_log_sum / background_count
    paper_clr = log_candidate - clr_center[:, None]
    isotype_burden = log_all[:, isotype_mask].mean(axis=1)
    global_loo = (
        background_log_sum[:, None] - log_candidate
    ) / (background_count - 1)

    if args.background_mode == "no_lag3":
        clr_formula = (
            "log(count + 1) - mean_j[log(count_j + 1)] within each cell, "
            "with j spanning 129 non-isotype ADTs after excluding "
            "CD223--LAG-3 from the background"
        )
        global_formula = (
            "marker-specific within-donor partial Spearman of log1p marker and "
            "projected score controlling mean log1p count across 128 ADTs: "
            "all non-isotype ADTs except CD223--LAG-3 and the current marker"
        )
        state_name = "query_robust_clr_program_no_lag3_background_v1"
        marker_set_id = "rahim_query_robust_clr_no_lag3_background_v1"
        analysis_status = (
            "Post-hoc strict sensitivity added after independent QA identified "
            "that the original 130-marker CLR/global-ADT background included "
            "the LAG3 holdout. It reuses the previously frozen selection rule "
            "but is not prospective evidence."
        )
    else:
        clr_formula = (
            "log(count + 1) - mean_j[log(count_j + 1)] within each cell, "
            "with j spanning all 130 non-isotype ADTs"
        )
        global_formula = (
            "marker-specific within-donor partial Spearman of log1p marker and "
            "projected score controlling mean log1p count across the other 129 "
            "non-isotype ADTs"
        )
        state_name = "query_robust_clr_program_v1"
        marker_set_id = "rahim_query_robust_clr_v1"
        analysis_status = "Canonical paper-matched background analysis"

    raw_z = rank_z_matrix(raw, donors)
    clr_z = rank_z_matrix(paper_clr, donors)
    projected_z = rank_z_matrix(projected, donors)[:, 0]
    iso_marker_z, iso_projected_z = common_partial_rank(
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

    variant_inputs = {
        "raw_rank": (raw_z, projected_z),
        "paper_clr": (clr_z, projected_z),
        "isotype_adjusted": (iso_marker_z, iso_projected_z),
        "global_adt_loo_adjusted": (global_marker_z, global_projected_z),
        "rna_qc_adjusted": (rna_marker_z, rna_projected_z),
    }
    summaries = []
    donor_effects = []
    lodo_effects = []
    for variant, (marker_signal, score_signal) in variant_inputs.items():
        summary, donor, lodo = summarize_variant(
            variant,
            marker_names,
            donors,
            projected,
            marker_signal,
            score_signal,
        )
        summaries.append(summary)
        donor_effects.append(donor)
        lodo_effects.append(lodo)

    explicit_singlet = cells.droplet_type.str.upper().isin({"SNG", "SINGLET"}).to_numpy()
    singlet_donors = donors[explicit_singlet]
    if explicit_singlet.sum() and pd.unique(singlet_donors).size >= 2:
        singlet_clr_z = rank_z_matrix(paper_clr[explicit_singlet], singlet_donors)
        singlet_projected_z = rank_z_matrix(
            projected[explicit_singlet], singlet_donors
        )[:, 0]
        summary, donor, lodo = summarize_variant(
            "paper_clr_explicit_singlets",
            marker_names,
            singlet_donors,
            projected[explicit_singlet],
            singlet_clr_z,
            singlet_projected_z,
        )
        summaries.append(summary)
        donor_effects.append(donor)
        lodo_effects.append(lodo)

    screen = merge_summaries(summaries)
    screen.insert(
        1, "marker", [FRIENDLY_NAMES.get(marker, marker) for marker in screen.marker_raw]
    )
    for marker_index, marker in enumerate(marker_names):
        screen.loc[screen.marker_raw.eq(marker), "raw_detection_fraction"] = np.mean(
            raw[:, marker_index] > 0
        )
        screen.loc[screen.marker_raw.eq(marker), "raw_median_count"] = np.median(
            raw[:, marker_index]
        )
    primary_rank_columns = [f"{variant}__rank_107" for variant in PRIMARY_VARIANTS]
    screen["robust_median_rank_primary_variants"] = screen[primary_rank_columns].median(
        axis=1
    )
    primary_lodo_columns = [
        column
        for column in screen
        if any(
            column.startswith(f"{variant}__lodo_holdout_") for variant in PRIMARY_VARIANTS
        )
    ]
    screen["robust_min_lodo_all_primary_variants"] = screen[primary_lodo_columns].min(
        axis=1
    )
    screen["rule_clr_positive_3of3"] = screen["paper_clr__positive_donors"] == 3
    screen["rule_isotype_positive_3of3"] = (
        screen["isotype_adjusted__positive_donors"] == 3
    )
    screen["rule_global_adt_positive_at_least_2of3"] = (
        screen["global_adt_loo_adjusted__positive_donors"] >= 2
    )
    screen["rule_median_primary_rank_at_most_15"] = (
        screen["robust_median_rank_primary_variants"] <= 15
    )
    screen["rule_all_primary_lodo_positive"] = (
        screen["robust_min_lodo_all_primary_variants"] > 0
    )
    rule_columns = [
        "rule_clr_positive_3of3",
        "rule_isotype_positive_3of3",
        "rule_global_adt_positive_at_least_2of3",
        "rule_median_primary_rank_at_most_15",
        "rule_all_primary_lodo_positive",
    ]
    screen["passes_frozen_rule_pre_redundancy"] = screen[rule_columns].all(axis=1)
    screen["frozen_rule_failure_reasons"] = screen.apply(
        lambda row: ";".join(column for column in rule_columns if not row[column]),
        axis=1,
    )
    screen = screen.sort_values(
        ["robust_median_rank_primary_variants", "paper_clr__rank_107", "marker_raw"],
        kind="stable",
    ).reset_index(drop=True)

    pairwise = equal_donor_marker_correlations(paper_clr, marker_names, donors)
    candidates = screen.loc[screen.passes_frozen_rule_pre_redundancy].copy()
    pruned, redundancy_map = redundancy_prune(candidates, pairwise, threshold=0.85)
    screen["selected_after_redundancy_pruning"] = screen.marker_raw.isin(
        pruned.marker_raw
    )

    donor_effects_frame = pd.concat(donor_effects, ignore_index=True)
    lodo_effects_frame = pd.concat(lodo_effects, ignore_index=True)
    screen.to_csv(output / "marker_screen_all_107_all_variants.csv", index=False)
    donor_effects_frame.to_csv(output / "marker_donor_effects_long.csv", index=False)
    lodo_effects_frame.to_csv(output / "marker_leave_one_donor_out_long.csv", index=False)
    pairwise.to_csv(output / "marker_pairwise_paper_clr_correlations.csv", index=False)
    candidates.to_csv(output / "robust_candidates_pre_redundancy.csv", index=False)
    pruned.to_csv(output / "robust_candidates_redundancy_pruned.csv", index=False)
    redundancy_map.to_csv(output / "redundancy_map.csv", index=False)

    selected_marker_indices = np.asarray(
        [np.flatnonzero(marker_names == marker)[0] for marker in pruned.marker_raw],
        dtype=int,
    )
    if len(selected_marker_indices):
        clr_percentiles = percentile_matrix(paper_clr, donors)
        robust_state_score = np.mean(
            clr_percentiles[:, selected_marker_indices], axis=1
        )
    else:
        robust_state_score = np.full(len(cells), np.nan)
    source_state = cells[["cell_uid", "donor"]].copy()
    source_state[state_name] = robust_state_score
    source_state.to_csv(
        output / "rahim_tumor_cd8_robust_source_state.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )

    canonical_state_comparison = None
    if args.background_mode == "no_lag3":
        canonical_marker_manifest_path = (
            HERE / "output/canonical/robust_selected_marker_manifest.json"
        )
        if not canonical_marker_manifest_path.exists():
            raise FileNotFoundError(
                "Canonical selected-marker manifest is required for the strict "
                "sensitivity comparison"
            )
        canonical_marker_manifest = json.loads(
            canonical_marker_manifest_path.read_text()
        )
        if canonical_marker_manifest.get("selected_marker_n") != 6:
            raise AssertionError(
                "Expected the canonical comparator to be the frozen six-marker state"
            )
        canonical_state_path = Path(canonical_marker_manifest["source_state_table"])
        canonical_state_column = canonical_marker_manifest[
            "source_state_score_column"
        ]
        canonical_state = pd.read_csv(
            canonical_state_path, dtype={"cell_uid": str, "donor": str}
        )
        state_comparison = source_state.merge(
            canonical_state[["cell_uid", "donor", canonical_state_column]],
            on=["cell_uid", "donor"],
            how="inner",
            validate="one_to_one",
        )
        if len(state_comparison) != len(source_state):
            raise AssertionError("Canonical and strict source-state cells differ")
        comparison_rows = []
        for donor in sorted(pd.unique(donors), key=str):
            donor_rows = state_comparison.donor.eq(donor).to_numpy()
            strict_z = rank_z(state_comparison.loc[donor_rows, state_name].to_numpy())
            canonical_z = rank_z(
                state_comparison.loc[
                    donor_rows, canonical_state_column
                ].to_numpy()
            )
            comparison_rows.append(
                {
                    "donor": donor,
                    "cells": int(donor_rows.sum()),
                    "spearman_strict_no_lag3_vs_canonical_six_marker": float(
                        np.mean(strict_z * canonical_z)
                    ),
                }
            )
        comparison_frame = pd.DataFrame(comparison_rows)
        comparison_path = output / "source_state_vs_canonical_six_marker_within_donor_rho.csv"
        comparison_frame.to_csv(comparison_path, index=False)
        canonical_state_comparison = {
            "canonical_marker_manifest": str(canonical_marker_manifest_path.resolve()),
            "canonical_marker_manifest_sha256": sha256(
                canonical_marker_manifest_path
            ),
            "canonical_selected_markers_raw": canonical_marker_manifest[
                "selected_markers_raw"
            ],
            "canonical_source_state_table": str(canonical_state_path.resolve()),
            "canonical_source_state_table_sha256": sha256(canonical_state_path),
            "canonical_source_state_score_column": canonical_state_column,
            "comparison_table": str(comparison_path.resolve()),
            "within_donor_spearman": {
                row["donor"]: row[
                    "spearman_strict_no_lag3_vs_canonical_six_marker"
                ]
                for row in comparison_rows
            },
            "equal_donor_mean_spearman": float(
                comparison_frame[
                    "spearman_strict_no_lag3_vs_canonical_six_marker"
                ].mean()
            ),
        }

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
    qc_by_donor.to_csv(output / "cell_quality_audit_by_donor.csv", index=False)

    selected_records = pruned[
        [
            "marker_raw",
            "marker",
            "robust_median_rank_primary_variants",
            "paper_clr__mean_donor_association",
            "isotype_adjusted__mean_donor_association",
            "global_adt_loo_adjusted__mean_donor_association",
            "rna_qc_adjusted__mean_donor_association",
        ]
    ].to_dict(orient="records")
    manifest = {
        "run_label": args.run_label,
        "score_file": str(score_path),
        "score_file_sha256": sha256(score_path),
        "score_source": score_source,
        "score_column": score_column,
        "frozen_rule": rule,
        "input_h5ad_sha256": {
            "ADT": sha256(ADT_H5AD),
            "RNA": sha256(RNA_H5AD),
        },
        "cells": int(len(cells)),
        "donors": sorted(pd.unique(donors), key=str),
        "donor_cell_counts": cells.donor.value_counts().sort_index().to_dict(),
        "adt_panel_markers": int(len(all_markers)),
        "isotype_controls": int(isotype_mask.sum()),
        "background_mode": args.background_mode,
        "analysis_status": analysis_status,
        "nonisotype_clr_denominator_markers": background_count,
        "global_adt_loo_nuisance_markers_per_candidate": background_count - 1,
        "lag3_in_clr_background": bool(
            background_mask[np.flatnonzero(all_markers == HOLDOUT_MARKER)[0]]
        ),
        "screen_family_markers": int(len(marker_names)),
        "holdout_not_screened": HOLDOUT_MARKER,
        "paper_matched_clr_formula": clr_formula,
        "isotype_adjustment": "within-donor partial Spearman of paper-matched CLR and projected score controlling mean log1p count across 7 isotypes",
        "global_adt_loo_adjustment": global_formula,
        "rna_quality_adjustment": "within-donor partial Spearman of paper-matched CLR and projected score controlling log1p RNA n_counts, log1p RNA n_genes, and RNA mitochondrial fraction",
        "explicit_singlet_sensitivity": {
            "cells": int(explicit_singlet.sum()),
            "donors": sorted(pd.unique(singlet_donors), key=str),
            "interpretation": "Patient 126 was not genetically demultiplexed and has missing droplet_type, not a doublet call; the explicit-SNG subset therefore contains only the two pooled donors and is sensitivity-only.",
        },
        "paper_qc_pass_cells": int(paper_qc_pass.sum()),
        "population_pvalues_calculated": False,
        "robust_candidates_before_redundancy": candidates.marker_raw.tolist(),
        "selected_after_redundancy": pruned.marker_raw.tolist(),
        "selected_marker_metrics": selected_records,
        "comparison_to_canonical_six_marker_state": canonical_state_comparison,
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    selected_marker_manifest = {
        "schema_version": 1,
        "marker_set_id": marker_set_id,
        "state_name": state_name,
        "purpose": (
            "Post-hoc strict no-LAG3-background sensitivity for the repaired "
            "Rahim source state"
            if args.background_mode == "no_lag3"
            else "Frozen-marker input for the repaired Rahim-to-Haist return transfer"
        ),
        "analysis_status": analysis_status,
        "selection_rule_path": str(RULE_PATH.resolve()),
        "selection_rule_frozen_utc": rule["frozen_utc"],
        "selection_rule_last_clarified_utc": rule.get("last_clarified_utc"),
        "canonical_projection_path": str(score_path),
        "canonical_projection_sha256": sha256(score_path),
        "canonical_projection_column": score_column,
        "screen_family_n": int(len(marker_names)),
        "forced_marker_count": False,
        "selected_marker_n": int(len(pruned)),
        "selected_markers_raw": pruned.marker_raw.tolist(),
        "selected_markers_display": pruned.marker.tolist(),
        "source_state_table": str(
            (output / "rahim_tumor_cd8_robust_source_state.csv.gz").resolve()
        ),
        "source_state_table_sha256": sha256(
            output / "rahim_tumor_cd8_robust_source_state.csv.gz"
        ),
        "source_state_cell_id_column": "cell_uid",
        "source_state_donor_column": "donor",
        "source_state_score_column": state_name,
        "source_state_formula": (
            "Unweighted mean across selected markers of each marker's "
            "within-Rahim-donor percentile midrank after the declared cell-wise CLR"
        ),
        "background_mode": args.background_mode,
        "clr_denominator_marker_n": background_count,
        "global_adt_loo_nuisance_marker_n": background_count - 1,
        "lag3_in_clr_or_global_background": bool(
            background_mask[np.flatnonzero(all_markers == HOLDOUT_MARKER)[0]]
        ),
        "paper_matched_clr_formula": clr_formula,
        "global_adt_loo_adjustment": global_formula,
        "selection_metrics": selected_records,
        "biological_label_status": "Neutral technical-robust program name; biological naming should be decided after repaired return validation.",
        "comparison_to_canonical_six_marker_state": canonical_state_comparison,
    }
    (output / "robust_selected_marker_manifest.json").write_text(
        json.dumps(selected_marker_manifest, indent=2) + "\n"
    )
    if args.run_label == "canonical":
        # Stable handoff path consumed by the repaired return-transfer script.
        # Only the explicitly labelled canonical run may update this alias.
        (HERE / "output/robust_marker_manifest.json").write_text(
            json.dumps(selected_marker_manifest, indent=2) + "\n"
        )

    summary = {
        "run_label": args.run_label,
        "background_mode": args.background_mode,
        "analysis_status": analysis_status,
        "score_source": score_source,
        "cells": int(len(cells)),
        "donors": int(pd.unique(donors).size),
        "screened_markers": int(len(marker_names)),
        "robust_pre_redundancy_n": int(len(candidates)),
        "robust_pre_redundancy": candidates.marker_raw.tolist(),
        "selected_after_redundancy_n": int(len(pruned)),
        "selected_after_redundancy": pruned.marker_raw.tolist(),
        "legacy_four": screen.loc[
            screen.marker_raw.isin(["CD137", "HLA-E-1", "CD83-1", "TIGIT--VSTM3"]),
            [
                "marker_raw",
                "robust_median_rank_primary_variants",
                "paper_clr__rank_107",
                "isotype_adjusted__rank_107",
                "global_adt_loo_adjusted__rank_107",
                "rna_qc_adjusted__rank_107",
                "passes_frozen_rule_pre_redundancy",
                "selected_after_redundancy_pruning",
            ],
        ].to_dict(orient="records"),
    }
    (output / "RUN_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
