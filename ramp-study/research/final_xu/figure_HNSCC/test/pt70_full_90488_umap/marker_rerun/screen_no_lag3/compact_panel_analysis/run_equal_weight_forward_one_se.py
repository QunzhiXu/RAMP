#!/usr/bin/env python3
"""Donor-balanced forward compression of the fixed PT70 10-marker panel.

This is the recommended presentation-facing compression analysis.  It uses one
continuous endpoint and one nested sequence of equal-weight marker scores:

1. Lock the 10 candidates already returned by the PT70 robust screen.
2. Convert paper-CLR marker values and the projected niche score to donor-wise
   percentile midranks.
3. Starting from no markers, add the marker maximizing equal-donor mean
   Spearman correlation for the unweighted subset score.
4. Choose the first step within one donor-level standard error of the full
   10-marker panel.

An outer-donor sensitivity analysis repeats the complete compression rule using
only the other two donors and evaluates the chosen score in the held-out donor.
It is conditional on the candidate pool, because the screen used all donors.
"""

from __future__ import annotations

import importlib.util
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
from sklearn.metrics import average_precision_score, roc_auc_score


def load_helpers():
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


def score_metrics(
    score: np.ndarray,
    response_percentile: np.ndarray,
    donors: np.ndarray,
    evaluation_donors: list[str],
) -> dict[str, float]:
    correlations = []
    aurocs = []
    average_precisions = []
    result: dict[str, float] = {}
    for donor in evaluation_donors:
        selected = donors == donor
        high_niche = response_percentile[selected] >= 0.75
        rho = float(spearmanr(score[selected], response_percentile[selected]).statistic)
        auroc = float(roc_auc_score(high_niche, score[selected]))
        average_precision = float(average_precision_score(high_niche, score[selected]))
        correlations.append(rho)
        aurocs.append(auroc)
        average_precisions.append(average_precision)
        result[f"spearman__{donor}"] = rho
        result[f"auroc_top_quartile__{donor}"] = auroc
        result[f"average_precision_top_quartile__{donor}"] = average_precision
    correlation_array = np.asarray(correlations)
    result["equal_donor_mean_spearman"] = float(correlation_array.mean())
    result["donor_se_spearman"] = (
        float(correlation_array.std(ddof=1) / np.sqrt(len(correlation_array)))
        if len(correlation_array) > 1
        else np.nan
    )
    result["equal_donor_mean_auroc_top_quartile"] = float(np.mean(aurocs))
    result["equal_donor_mean_average_precision_top_quartile"] = float(
        np.mean(average_precisions)
    )
    return result


def forward_path(
    features: np.ndarray,
    response_percentile: np.ndarray,
    donors: np.ndarray,
    marker_names: list[str],
    optimization_donors: list[str],
) -> pd.DataFrame:
    selected: list[int] = []
    remaining = list(range(len(marker_names)))
    rows = []
    for step in range(1, len(marker_names) + 1):
        choices = []
        for addition in remaining:
            proposed = selected + [addition]
            # Sum columns in fixed candidate order so floating-point ties do not
            # depend on the order in which markers entered the path.
            score = features[:, sorted(proposed)].mean(axis=1)
            metrics = score_metrics(
                score, response_percentile, donors, optimization_donors
            )
            choices.append((metrics["equal_donor_mean_spearman"], addition, metrics))
        # Marker name is an explicit deterministic tie-breaker.
        _, addition, metrics = sorted(
            choices, key=lambda item: (-item[0], marker_names[item[1]])
        )[0]
        selected.append(addition)
        remaining.remove(addition)
        rows.append(
            {
                "step": step,
                "added_marker": marker_names[addition],
                "marker_raw_pipe": "|".join(marker_names[index] for index in selected),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def choose_one_se(path: pd.DataFrame) -> tuple[pd.Series, float]:
    full = path.loc[path.step.eq(path.step.max())].iloc[0]
    threshold = float(full.equal_donor_mean_spearman - full.donor_se_spearman)
    eligible = path.loc[path.equal_donor_mean_spearman >= threshold]
    chosen = eligible.sort_values("step", kind="stable").iloc[0]
    return chosen, threshold


def main() -> None:
    helpers = load_helpers()
    score, score_column = helpers.read_score(PROJECTION, "projected_roi_adjusted_exposure")
    adt_obs, all_markers, adt_all = helpers.load_adt()
    rna_obs = helpers.load_rna_obs()
    cells, adt = helpers.join_inputs(
        score, score_column, adt_obs, all_markers, adt_all, rna_obs
    )
    marker_names = pd.read_csv(CANDIDATES).marker_raw.astype(str).tolist()
    if len(marker_names) != 10 or len(set(marker_names)) != 10:
        raise AssertionError("Expected the fixed set of 10 unique PT70 candidates")

    donors = cells.donor.to_numpy(str)
    donor_names = sorted(set(donors))
    response = helpers.percentile_matrix(cells.projected_score.to_numpy(float), donors)[:, 0]
    isotype_mask = np.asarray(["isotype" in marker.lower() for marker in all_markers])
    background_mask = ~isotype_mask & (all_markers != helpers.HOLDOUT_MARKER)
    if isotype_mask.sum() != 7 or background_mask.sum() != 129:
        raise AssertionError("Expected the frozen 7-isotype/129-background panel")
    log_adt = np.log1p(adt)
    paper_clr = log_adt - (
        log_adt[:, background_mask].sum(axis=1) / background_mask.sum()
    )[:, None]
    marker_indices = [np.flatnonzero(all_markers == marker)[0] for marker in marker_names]
    features = helpers.percentile_matrix(paper_clr[:, marker_indices], donors)

    path = forward_path(features, response, donors, marker_names, donor_names)
    chosen, threshold = choose_one_se(path)
    selected_markers = chosen.marker_raw_pipe.split("|")
    selected_indices = [marker_names.index(marker) for marker in selected_markers]
    compact_score = features[:, sorted(selected_indices)].mean(axis=1)
    full_score = features[:, list(range(len(marker_names)))].mean(axis=1)

    fixed_comparison_rows = []
    for label, names, values in (
        ("selected_compact", selected_markers, compact_score),
        ("full_10", marker_names, full_score),
    ):
        metrics = score_metrics(values, response, donors, donor_names)
        fixed_comparison_rows.append(
            {
                "panel": label,
                "panel_size": len(names),
                "marker_raw_pipe": "|".join(names),
                **metrics,
            }
        )
    fixed_comparison = pd.DataFrame(fixed_comparison_rows)

    outer_rows = []
    for held_out in donor_names:
        training_donors = [donor for donor in donor_names if donor != held_out]
        training_path = forward_path(
            features, response, donors, marker_names, training_donors
        )
        outer_choice, training_threshold = choose_one_se(training_path)
        outer_markers = outer_choice.marker_raw_pipe.split("|")
        outer_indices = [marker_names.index(marker) for marker in outer_markers]
        outer_score = features[:, sorted(outer_indices)].mean(axis=1)
        full_metrics = score_metrics(full_score, response, donors, [held_out])
        compact_metrics = score_metrics(outer_score, response, donors, [held_out])
        outer_rows.append(
            {
                "held_out_donor": held_out,
                "training_donors_pipe": "|".join(training_donors),
                "training_one_se_threshold": training_threshold,
                "selected_panel_size": len(outer_markers),
                "selected_marker_raw_pipe": "|".join(outer_markers),
                "held_out_spearman": compact_metrics[f"spearman__{held_out}"],
                "held_out_auroc_top_quartile": compact_metrics[
                    f"auroc_top_quartile__{held_out}"
                ],
                "held_out_average_precision_top_quartile": compact_metrics[
                    f"average_precision_top_quartile__{held_out}"
                ],
                "full_10_held_out_spearman": full_metrics[f"spearman__{held_out}"],
                "full_10_held_out_auroc_top_quartile": full_metrics[
                    f"auroc_top_quartile__{held_out}"
                ],
                "full_10_held_out_average_precision_top_quartile": full_metrics[
                    f"average_precision_top_quartile__{held_out}"
                ],
            }
        )
    outer = pd.DataFrame(outer_rows)
    outer_summary = {
        "held_out_donor": "equal_donor_summary",
        "training_donors_pipe": "",
        "training_one_se_threshold": np.nan,
        "selected_panel_size": float(outer.selected_panel_size.mean()),
        "selected_marker_raw_pipe": "fold-specific; see donor rows",
    }
    for column in (
        "held_out_spearman",
        "held_out_auroc_top_quartile",
        "held_out_average_precision_top_quartile",
        "full_10_held_out_spearman",
        "full_10_held_out_auroc_top_quartile",
        "full_10_held_out_average_precision_top_quartile",
    ):
        outer_summary[column] = float(outer[column].mean())
    outer = pd.concat([outer, pd.DataFrame([outer_summary])], ignore_index=True)

    path.to_csv(HERE / "equal_weight_forward_path.csv", index=False)
    fixed_comparison.to_csv(HERE / "equal_weight_selected_vs_full.csv", index=False)
    outer.to_csv(HERE / "nested_outer_donor_sensitivity.csv", index=False)

    selected_metrics = fixed_comparison.loc[
        fixed_comparison.panel.eq("selected_compact")
    ].iloc[0]
    full_metrics = fixed_comparison.loc[fixed_comparison.panel.eq("full_10")].iloc[0]
    manifest = {
        "purpose": "single-screen parsimonious representation of the fixed PT70 marker set",
        "candidate_source": str(CANDIDATES),
        "projection_source": str(PROJECTION),
        "cells": len(cells),
        "donor_cell_counts": {
            donor: int(np.sum(donors == donor)) for donor in donor_names
        },
        "candidate_markers": marker_names,
        "feature": "paper CLR followed by within-donor percentile midrank",
        "panel_score": "unweighted mean of marker percentiles",
        "selection_metric": "equal-donor mean Spearman with continuous projected niche score",
        "search": "greedy forward addition over the fixed 10 candidates",
        "selection_rule": (
            "first forward-path panel within one donor-level SE of the full 10-marker panel"
        ),
        "full_10_mean_spearman": float(full_metrics.equal_donor_mean_spearman),
        "full_10_donor_se_spearman": float(full_metrics.donor_se_spearman),
        "one_se_threshold": threshold,
        "selected_panel": selected_markers,
        "selected_panel_size": len(selected_markers),
        "selected_mean_spearman": float(selected_metrics.equal_donor_mean_spearman),
        "selected_mean_auroc_top_quartile": float(
            selected_metrics.equal_donor_mean_auroc_top_quartile
        ),
        "selected_mean_average_precision_top_quartile": float(
            selected_metrics.equal_donor_mean_average_precision_top_quartile
        ),
        "outer_donor_sensitivity_mean_spearman": float(
            outer_rows[0]["held_out_spearman"]
            + outer_rows[1]["held_out_spearman"]
            + outer_rows[2]["held_out_spearman"]
        )
        / 3,
        "outer_donor_selected_panel_sizes": [
            int(row["selected_panel_size"]) for row in outer_rows
        ],
        "limitations": [
            "Only three donors are available; donor-level standard errors and marker identity are unstable.",
            "The fixed 10-marker candidate set was screened using these same donors, so outer-donor sensitivity is conditional on that pool.",
            "This is an exploratory parsimonious representation, not an independently validated signature.",
        ],
    }
    with (HERE / "equal_weight_forward_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
