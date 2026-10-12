#!/usr/bin/env python3
"""Apply the frozen robust 107-ADT rule to the new PT70 projection.

The numerical helpers and formulas are imported from the frozen historical
screen, but every output is written inside this test directory. The strict
no-LAG3 background and equal-donor projection are used because that combination
defined the previous final five-marker program.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
FINAL_XU = HERE.parents[2]
FIGURE_HNSCC = HERE.parents[1]
PROJECTION = HERE / "marker_rerun/projection/rahim_tumor_cd8_projected_scores.csv.gz"
OUT = HERE / "marker_rerun/screen_no_lag3"
CACHE = HERE / ".cache"
for variable, child in (
    ("MPLCONFIGDIR", "matplotlib"),
    ("NUMBA_CACHE_DIR", "numba"),
    ("XDG_CACHE_HOME", "xdg"),
    ("JOBLIB_TEMP_FOLDER", "joblib"),
    ("TMPDIR", "tmp"),
):
    os.environ[variable] = str(CACHE / child)
sys.dont_write_bytecode = True

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HISTORICAL_SCRIPT = (
    FINAL_XU
    / "test0827/hnscc/pt_survival_first/reviewer_robustness/clr_screen/"
      "run_clr_robustness_screen.py"
)
OLD_SCREEN = (
    FINAL_XU
    / "test0827/hnscc/pt_survival_first/reviewer_robustness/clr_screen/output/"
      "equal_donor_scaling_no_lag3_background/marker_screen_all_107_all_variants.csv"
)
OLD_FINAL = ["TIGIT--VSTM3", "CD137", "CD244--2B4", "CD39", "TCR-A--TCR-B"]
PRIMARY_VARIANTS = (
    "raw_rank",
    "paper_clr",
    "isotype_adjusted",
    "global_adt_loo_adjusted",
)


def load_historical_module():
    spec = importlib.util.spec_from_file_location("frozen_hnscc_clr_screen", HISTORICAL_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(HISTORICAL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ADT_H5AD = (
        FIGURE_HNSCC
        / "sampled_data/rahim_citeseq/complete/GSE212797_adata_ADT.h5ad"
    )
    module.RNA_H5AD = (
        FIGURE_HNSCC
        / "sampled_data/rahim_citeseq/complete/GSE212797_adata.h5ad"
    )
    return module


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    old = load_historical_module()
    score, score_column = old.read_score(PROJECTION, "projected_roi_adjusted_exposure")
    adt_obs, all_markers, adt_all = old.load_adt()
    rna_obs = old.load_rna_obs()
    cells, adt = old.join_inputs(score, score_column, adt_obs, all_markers, adt_all, rna_obs)
    donors = cells.donor.to_numpy(str)
    projected = cells.projected_score.to_numpy(float)

    isotype_mask = np.asarray(["isotype" in marker.lower() for marker in all_markers])
    background_mask = ~isotype_mask & (all_markers != old.HOLDOUT_MARKER)
    candidate_mask = np.asarray(
        [
            marker not in old.SHARED_RAHIM_MARKERS
            and marker != old.HOLDOUT_MARKER
            and "isotype" not in marker.lower()
            for marker in all_markers
        ]
    )
    marker_names = all_markers[candidate_mask]
    if isotype_mask.sum() != 7 or background_mask.sum() != 129 or len(marker_names) != 107:
        raise AssertionError("The frozen 7-isotype/129-background/107-candidate panel changed")

    raw = adt[:, candidate_mask]
    log_all = np.log1p(adt)
    log_candidate = log_all[:, candidate_mask]
    background_log_sum = log_all[:, background_mask].sum(axis=1)
    paper_clr = log_candidate - (background_log_sum / 129)[:, None]
    isotype_burden = log_all[:, isotype_mask].mean(axis=1)
    global_loo = (background_log_sum[:, None] - log_candidate) / 128

    raw_z = old.rank_z_matrix(raw, donors)
    clr_z = old.rank_z_matrix(paper_clr, donors)
    projected_z = old.rank_z_matrix(projected, donors)[:, 0]
    iso_marker_z, iso_projected_z = old.common_partial_rank(
        paper_clr, projected, isotype_burden[:, None], donors
    )
    global_marker_z, global_projected_z = old.marker_specific_partial_rank(
        log_candidate, projected, global_loo, donors
    )
    rna_covariates = np.column_stack(
        [
            np.log1p(cells.rna_n_counts.to_numpy(float)),
            np.log1p(cells.rna_n_genes.to_numpy(float)),
            cells.rna_percent_mito.to_numpy(float),
        ]
    )
    rna_marker_z, rna_projected_z = old.common_partial_rank(
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
        summary, donor, lodo = old.summarize_variant(
            variant, marker_names, donors, projected, marker_signal, score_signal
        )
        summaries.append(summary)
        donor_effects.append(donor)
        lodo_effects.append(lodo)

    screen = old.merge_summaries(summaries)
    screen.insert(
        1,
        "marker",
        [old.FRIENDLY_NAMES.get(marker, marker) for marker in screen.marker_raw],
    )
    for marker_index, marker in enumerate(marker_names):
        selected = screen.marker_raw.eq(marker)
        screen.loc[selected, "raw_detection_fraction"] = np.mean(raw[:, marker_index] > 0)
        screen.loc[selected, "raw_median_count"] = np.median(raw[:, marker_index])

    primary_rank_columns = [f"{variant}__rank_107" for variant in PRIMARY_VARIANTS]
    screen["robust_median_rank_primary_variants"] = screen[primary_rank_columns].median(axis=1)
    primary_lodo_columns = [
        column
        for column in screen
        if any(column.startswith(f"{variant}__lodo_holdout_") for variant in PRIMARY_VARIANTS)
    ]
    screen["robust_min_lodo_all_primary_variants"] = screen[primary_lodo_columns].min(axis=1)
    screen["rule_clr_positive_3of3"] = screen["paper_clr__positive_donors"] == 3
    screen["rule_isotype_positive_3of3"] = screen["isotype_adjusted__positive_donors"] == 3
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
        lambda row: ";".join(column for column in rule_columns if not row[column]), axis=1
    )
    screen = screen.sort_values(
        ["robust_median_rank_primary_variants", "paper_clr__rank_107", "marker_raw"],
        kind="stable",
    ).reset_index(drop=True)

    pairwise = old.equal_donor_marker_correlations(paper_clr, marker_names, donors)
    candidates = screen.loc[screen.passes_frozen_rule_pre_redundancy].copy()
    pruned, redundancy_map = old.redundancy_prune(candidates, pairwise, threshold=0.85)
    screen["selected_after_redundancy_pruning"] = screen.marker_raw.isin(pruned.marker_raw)

    screen.to_csv(OUT / "marker_screen_all_107_all_variants.csv", index=False)
    pd.concat(donor_effects, ignore_index=True).to_csv(
        OUT / "marker_donor_effects_long.csv", index=False
    )
    pd.concat(lodo_effects, ignore_index=True).to_csv(
        OUT / "marker_leave_one_donor_out_long.csv", index=False
    )
    pairwise.to_csv(OUT / "marker_pairwise_paper_clr_correlations.csv", index=False)
    candidates.to_csv(OUT / "robust_candidates_pre_redundancy.csv", index=False)
    pruned.to_csv(OUT / "robust_candidates_redundancy_pruned.csv", index=False)
    redundancy_map.to_csv(OUT / "redundancy_map.csv", index=False)

    selected_indices = np.asarray(
        [np.flatnonzero(marker_names == marker)[0] for marker in pruned.marker_raw], int
    )
    source_state = cells[["cell_uid", "donor"]].copy()
    source_state["pt70_robust_clr_program_no_lag3"] = np.mean(
        old.percentile_matrix(paper_clr, donors)[:, selected_indices], axis=1
    )
    source_state.to_csv(
        OUT / "rahim_tumor_cd8_pt70_source_state.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )

    old_screen = pd.read_csv(OLD_SCREEN)
    comparison = screen.merge(
        old_screen[
            [
                "marker_raw",
                "marker",
                "robust_median_rank_primary_variants",
                "paper_clr__rank_107",
                "selected_after_redundancy_pruning",
            ]
        ],
        on="marker_raw",
        validate="one_to_one",
        suffixes=("_pt70", "_old40"),
    )
    comparison["old_final_marker"] = comparison.marker_raw.isin(OLD_FINAL)
    comparison["retention_status"] = np.select(
        [
            comparison.selected_after_redundancy_pruning_pt70
            & comparison.selected_after_redundancy_pruning_old40,
            comparison.selected_after_redundancy_pruning_pt70,
            comparison.selected_after_redundancy_pruning_old40,
        ],
        ["retained_in_both", "PT70_only", "old40_only"],
        default="selected_in_neither",
    )
    comparison.to_csv(OUT / "pt70_vs_old40_marker_comparison.csv", index=False)

    selected_new = pruned.marker_raw.tolist()
    selected_old = OLD_FINAL
    selected_union = sorted(set(selected_new) | set(selected_old))
    focused = comparison.loc[comparison.marker_raw.isin(selected_union)].copy()
    focused.to_csv(OUT / "selected_marker_comparison.csv", index=False)

    fig, axis = plt.subplots(figsize=(7.2, 6.4))
    axis.scatter(
        comparison.robust_median_rank_primary_variants_old40,
        comparison.robust_median_rank_primary_variants_pt70,
        s=16,
        color="#BDBDBD",
        alpha=0.50,
        linewidths=0,
    )
    colors = {
        "retained_in_both": "#009E73",
        "PT70_only": "#0072B2",
        "old40_only": "#D55E00",
    }
    for status, color in colors.items():
        block = focused.loc[focused.retention_status.eq(status)]
        if block.empty:
            continue
        axis.scatter(
            block.robust_median_rank_primary_variants_old40,
            block.robust_median_rank_primary_variants_pt70,
            s=52,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            label=status.replace("_", " "),
            zorder=5,
        )
        for row in block.itertuples(index=False):
            axis.annotate(
                row.marker_pt70,
                (
                    row.robust_median_rank_primary_variants_old40,
                    row.robust_median_rank_primary_variants_pt70,
                ),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
    axis.plot([0, 108], [0, 108], color="#666666", linewidth=0.8, linestyle="--")
    axis.axhline(15, color="#999999", linewidth=0.7, linestyle=":")
    axis.axvline(15, color="#999999", linewidth=0.7, linestyle=":")
    axis.set_xlim(0, 108)
    axis.set_ylim(0, 108)
    axis.set_xlabel("Previous old40 robust median rank")
    axis.set_ylabel("New PT70 robust median rank")
    axis.set_title("HNSCC biomarker-screen stability", loc="left", weight="bold")
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "pt70_vs_old40_marker_rank_stability.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "pt70_vs_old40_marker_rank_stability.pdf", bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "screen": "frozen 107-marker robustness rule",
        "projection": str(PROJECTION),
        "cells": int(len(cells)),
        "donors": sorted(pd.unique(donors), key=str),
        "background": "129 non-isotype ADTs with LAG3 excluded",
        "forced_marker_count": False,
        "selected_old40": selected_old,
        "selected_PT70": selected_new,
        "intersection": sorted(set(selected_old) & set(selected_new)),
        "old40_only": sorted(set(selected_old) - set(selected_new)),
        "PT70_only": sorted(set(selected_new) - set(selected_old)),
        "exact_marker_set_match": set(selected_old) == set(selected_new),
    }
    (OUT / "screen_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
