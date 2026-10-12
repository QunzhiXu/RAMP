#!/usr/bin/env python3
"""Compress the fixed PT70 robust-marker set with donor-aware prediction.

This is a post-screen panel-compression analysis.  It neither reruns the
107-marker screen nor uses the historical screen.  All nonempty subsets of the
10 fixed PT70 robust candidates are assessed by leave-one-donor-out (LODO)
positive least-squares prediction of the within-donor ranked projected niche
score.  Training cells receive inverse donor-size weights.

The selected panel is the smallest subset whose equal-donor mean LODO Spearman
correlation is within one donor-fold standard error of the best subset.  Among
qualifying subsets of the smallest size, the subset with the largest mean LODO
Spearman correlation is selected; mean LODO MSE and marker names break exact
ties.  The fixed, unweighted mean-percentile score is reported as a transparent
secondary check.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import os
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
SCREEN = HERE.parent
PT70 = HERE.parents[2]
FIGURE_HNSCC = HERE.parents[4]
FINAL_XU = HERE.parents[5]
PROJECTION = PT70 / "marker_rerun/projection/rahim_tumor_cd8_projected_scores.csv.gz"
CANDIDATES = SCREEN / "robust_candidates_redundancy_pruned.csv"
HISTORICAL_HELPERS = (
    FINAL_XU
    / "test0827/hnscc/pt_survival_first/reviewer_robustness/clr_screen/"
    / "run_clr_robustness_screen.py"
)

for variable, child in (
    ("MPLCONFIGDIR", "matplotlib"),
    ("NUMBA_CACHE_DIR", "numba"),
    ("XDG_CACHE_HOME", "xdg"),
    ("JOBLIB_TEMP_FOLDER", "joblib"),
    ("TMPDIR", "tmp"),
):
    os.environ[variable] = str(PT70 / ".cache" / child)
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression


def load_helpers():
    """Load the frozen numerical/data-join helpers used by the PT70 screen."""
    spec = importlib.util.spec_from_file_location("frozen_hnscc_screen", HISTORICAL_HELPERS)
    if spec is None or spec.loader is None:
        raise ImportError(HISTORICAL_HELPERS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ADT_H5AD = (
        FIGURE_HNSCC / "sampled_data/rahim_citeseq/complete/GSE212797_adata_ADT.h5ad"
    )
    module.RNA_H5AD = (
        FIGURE_HNSCC / "sampled_data/rahim_citeseq/complete/GSE212797_adata.h5ad"
    )
    return module


def equal_donor_training_weights(donors: np.ndarray) -> np.ndarray:
    """Return weights summing equally within every donor."""
    weights = np.zeros(len(donors), dtype=float)
    units = sorted(set(donors))
    for donor in units:
        selected = donors == donor
        weights[selected] = len(donors) / (len(units) * selected.sum())
    return weights


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    result = float(spearmanr(x, y).statistic)
    if not np.isfinite(result):
        raise ValueError("Undefined Spearman correlation")
    return result


def lodo_predictions(
    features: np.ndarray,
    response: np.ndarray,
    donors: np.ndarray,
    selected_columns: tuple[int, ...],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Fit positive OLS on two donors and predict the held-out donor."""
    predictions = np.full(len(response), np.nan, dtype=float)
    coefficients: dict[str, np.ndarray] = {}
    for held_out in sorted(set(donors)):
        test = donors == held_out
        train = ~test
        model = LinearRegression(positive=True)
        model.fit(
            features[train][:, selected_columns],
            response[train],
            sample_weight=equal_donor_training_weights(donors[train]),
        )
        predictions[test] = model.predict(features[test][:, selected_columns])
        coefficients[held_out] = model.coef_.copy()
    if not np.isfinite(predictions).all():
        raise AssertionError("Incomplete LODO predictions")
    return predictions, coefficients


def donor_metrics(
    predictions: np.ndarray, response: np.ndarray, donors: np.ndarray
) -> tuple[dict[str, float], dict[str, float]]:
    correlations = {}
    mean_squared_errors = {}
    for donor in sorted(set(donors)):
        selected = donors == donor
        correlations[donor] = safe_spearman(predictions[selected], response[selected])
        mean_squared_errors[donor] = float(
            np.mean(np.square(predictions[selected] - response[selected]))
        )
    return correlations, mean_squared_errors


def summarize_subset(
    selected_columns: tuple[int, ...],
    marker_names: list[str],
    features: np.ndarray,
    response: np.ndarray,
    donors: np.ndarray,
) -> dict[str, object]:
    lodo, _ = lodo_predictions(features, response, donors, selected_columns)
    lodo_correlations, lodo_mse = donor_metrics(lodo, response, donors)
    unweighted = features[:, selected_columns].mean(axis=1)
    score_correlations, score_mse = donor_metrics(unweighted, response, donors)

    correlation_values = np.asarray(list(lodo_correlations.values()), dtype=float)
    mse_values = np.asarray(list(lodo_mse.values()), dtype=float)
    score_correlation_values = np.asarray(list(score_correlations.values()), dtype=float)
    score_mse_values = np.asarray(list(score_mse.values()), dtype=float)
    result: dict[str, object] = {
        "panel_size": len(selected_columns),
        "marker_indices_pipe": "|".join(str(column) for column in selected_columns),
        "marker_raw_pipe": "|".join(marker_names[column] for column in selected_columns),
        "lodo_nnls_mean_spearman": correlation_values.mean(),
        "lodo_nnls_se_spearman_across_donors": correlation_values.std(ddof=1)
        / np.sqrt(len(correlation_values)),
        "lodo_nnls_min_spearman": correlation_values.min(),
        "lodo_nnls_mean_mse": mse_values.mean(),
        "lodo_nnls_se_mse_across_donors": mse_values.std(ddof=1) / np.sqrt(len(mse_values)),
        "unweighted_mean_spearman": score_correlation_values.mean(),
        "unweighted_se_spearman_across_donors": score_correlation_values.std(ddof=1)
        / np.sqrt(len(score_correlation_values)),
        "unweighted_min_spearman": score_correlation_values.min(),
        "unweighted_mean_mse": score_mse_values.mean(),
    }
    for donor in sorted(set(donors)):
        result[f"lodo_nnls_spearman__{donor}"] = lodo_correlations[donor]
        result[f"lodo_nnls_mse__{donor}"] = lodo_mse[donor]
        result[f"unweighted_spearman__{donor}"] = score_correlations[donor]
        result[f"unweighted_mse__{donor}"] = score_mse[donor]
    return result


def panel_row_sort(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(
        ["lodo_nnls_mean_spearman", "lodo_nnls_mean_mse", "marker_raw_pipe"],
        ascending=[False, True, True],
        kind="stable",
    )


def main() -> None:
    helpers = load_helpers()
    score, score_column = helpers.read_score(PROJECTION, "projected_roi_adjusted_exposure")
    adt_obs, all_markers, adt_all = helpers.load_adt()
    rna_obs = helpers.load_rna_obs()
    cells, adt = helpers.join_inputs(
        score, score_column, adt_obs, all_markers, adt_all, rna_obs
    )

    candidate_table = pd.read_csv(CANDIDATES)
    marker_names = candidate_table.marker_raw.astype(str).tolist()
    if len(marker_names) != 10 or candidate_table.marker_raw.duplicated().any():
        raise AssertionError("Expected the fixed set of 10 unique PT70 candidates")

    donors = cells.donor.to_numpy(str)
    response = helpers.percentile_matrix(cells.projected_score.to_numpy(float), donors)[:, 0]
    isotype_mask = np.asarray(["isotype" in marker.lower() for marker in all_markers])
    background_mask = ~isotype_mask & (all_markers != helpers.HOLDOUT_MARKER)
    if isotype_mask.sum() != 7 or background_mask.sum() != 129:
        raise AssertionError("Expected the frozen 7-isotype/129-background panel")

    log_adt = np.log1p(adt)
    paper_clr_all = log_adt - (
        log_adt[:, background_mask].sum(axis=1) / background_mask.sum()
    )[:, None]
    marker_indices = np.asarray(
        [np.flatnonzero(all_markers == marker)[0] for marker in marker_names], dtype=int
    )
    features = helpers.percentile_matrix(paper_clr_all[:, marker_indices], donors)

    subset_rows = []
    for panel_size in range(1, len(marker_names) + 1):
        for selected_columns in itertools.combinations(range(len(marker_names)), panel_size):
            subset_rows.append(
                summarize_subset(
                    selected_columns, marker_names, features, response, donors
                )
            )
    subsets = pd.DataFrame(subset_rows)
    subsets = subsets.sort_values(
        ["panel_size", "marker_raw_pipe"], kind="stable"
    ).reset_index(drop=True)

    best_by_size = pd.concat(
        [
            panel_row_sort(block).head(1)
            for _, block in subsets.groupby("panel_size", sort=True)
        ],
        ignore_index=True,
    )
    best_overall = panel_row_sort(subsets).iloc[0]
    one_se_threshold = float(
        best_overall.lodo_nnls_mean_spearman
        - best_overall.lodo_nnls_se_spearman_across_donors
    )
    eligible = subsets.loc[
        subsets.lodo_nnls_mean_spearman >= one_se_threshold
    ].copy()
    minimum_size = int(eligible.panel_size.min())
    selected = panel_row_sort(eligible.loc[eligible.panel_size.eq(minimum_size)]).iloc[0]
    full = subsets.loc[subsets.panel_size.eq(len(marker_names))].iloc[0]

    selected_indices = tuple(int(value) for value in selected.marker_indices_pipe.split("|"))
    full_indices = tuple(range(len(marker_names)))
    selected_lodo, selected_coefficients = lodo_predictions(
        features, response, donors, selected_indices
    )
    full_lodo, full_coefficients = lodo_predictions(features, response, donors, full_indices)
    selected_unweighted = features[:, selected_indices].mean(axis=1)
    # Use the same explicit column-selection path as the exhaustive table.  The
    # operation order can otherwise change floating-point ties at ~1e-16 and
    # cause tiny differences in Spearman midranks.
    full_unweighted = features[:, full_indices].mean(axis=1)

    comparison_rows = []
    for panel_label, panel_indices, lodo, unweighted in (
        ("selected_compact", selected_indices, selected_lodo, selected_unweighted),
        ("full_10", full_indices, full_lodo, full_unweighted),
    ):
        for method, predictions in (
            ("LODO_positive_least_squares", lodo),
            ("unweighted_mean_percentile", unweighted),
        ):
            correlations, errors = donor_metrics(predictions, response, donors)
            values = np.asarray(list(correlations.values()))
            comparison_rows.append(
                {
                    "panel": panel_label,
                    "panel_size": len(panel_indices),
                    "marker_raw_pipe": "|".join(marker_names[index] for index in panel_indices),
                    "method": method,
                    "donor": "equal_donor_summary",
                    "spearman": values.mean(),
                    "spearman_se_across_donors": values.std(ddof=1) / np.sqrt(len(values)),
                    "mse": np.mean(list(errors.values())),
                }
            )
            for donor in sorted(correlations):
                comparison_rows.append(
                    {
                        "panel": panel_label,
                        "panel_size": len(panel_indices),
                        "marker_raw_pipe": "|".join(
                            marker_names[index] for index in panel_indices
                        ),
                        "method": method,
                        "donor": donor,
                        "spearman": correlations[donor],
                        "spearman_se_across_donors": np.nan,
                        "mse": errors[donor],
                    }
                )
    comparison = pd.DataFrame(comparison_rows)

    coefficient_rows = []
    for panel_label, panel_indices, coefficients in (
        ("selected_compact", selected_indices, selected_coefficients),
        ("full_10", full_indices, full_coefficients),
    ):
        for held_out, values in coefficients.items():
            for index, coefficient in zip(panel_indices, values, strict=True):
                coefficient_rows.append(
                    {
                        "panel": panel_label,
                        "held_out_donor": held_out,
                        "marker_raw": marker_names[index],
                        "coefficient": coefficient,
                    }
                )
    coefficients = pd.DataFrame(coefficient_rows)

    subsets.to_csv(HERE / "all_1023_subset_performance.csv", index=False)
    best_by_size.to_csv(HERE / "best_panel_by_size.csv", index=False)
    comparison.to_csv(HERE / "selected_vs_full_donor_performance.csv", index=False)
    coefficients.to_csv(HERE / "lodo_coefficients_selected_vs_full.csv", index=False)

    manifest = {
        "purpose": "single-screen post-selection compression of the fixed PT70 10-marker set",
        "candidate_source": str(CANDIDATES),
        "projection_source": str(PROJECTION),
        "cells": len(cells),
        "donor_cell_counts": {
            donor: int(np.sum(donors == donor)) for donor in sorted(set(donors))
        },
        "candidate_markers": marker_names,
        "feature": "paper CLR followed by within-donor percentile midrank",
        "response": "projected niche score followed by within-donor percentile midrank",
        "prediction": "positive least squares trained with equal total weight per training donor",
        "validation": "leave one Rahim donor out; Spearman correlations averaged equally across donors",
        "search": "exhaustive 2^10-1 nonempty subsets",
        "selection_rule": (
            "smallest panel with mean LODO Spearman >= best mean LODO Spearman "
            "minus the best panel's donor-fold standard error; best performance breaks size ties"
        ),
        "best_overall_panel": best_overall.marker_raw_pipe.split("|"),
        "best_overall_panel_size": int(best_overall.panel_size),
        "best_overall_lodo_mean_spearman": float(best_overall.lodo_nnls_mean_spearman),
        "best_overall_lodo_se_spearman": float(
            best_overall.lodo_nnls_se_spearman_across_donors
        ),
        "one_se_threshold": one_se_threshold,
        "selected_panel": selected.marker_raw_pipe.split("|"),
        "selected_panel_size": int(selected.panel_size),
        "selected_lodo_mean_spearman": float(selected.lodo_nnls_mean_spearman),
        "selected_lodo_mean_mse": float(selected.lodo_nnls_mean_mse),
        "selected_unweighted_mean_spearman": float(selected.unweighted_mean_spearman),
        "full_10_lodo_mean_spearman": float(full.lodo_nnls_mean_spearman),
        "full_10_lodo_mean_mse": float(full.lodo_nnls_mean_mse),
        "full_10_unweighted_mean_spearman": float(full.unweighted_mean_spearman),
        "limitations": [
            "Only three donors are available, so donor-fold standard errors are imprecise.",
            "The fixed 10-marker candidate set was screened using these same donors; this is internal panel compression, not independent validation.",
            "The selected panel should be frozen before validation in another cohort.",
        ],
    }
    with (HERE / "selection_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
