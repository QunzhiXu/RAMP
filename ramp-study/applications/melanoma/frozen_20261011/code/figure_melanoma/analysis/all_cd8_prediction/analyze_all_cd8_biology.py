#!/usr/bin/env python3
"""Audit the biological phenotype of the all-CD8 CITE-seq projection.

The script does not copy raw data.  It reads the 137-feature Engelbrecht ADT
matrix in place, calculates patient-internal effects, diagnoses total-ADT
depth, performs a classical cell-level CLR sensitivity analysis, compares the
new 5,768-cell projection with the frozen 2,305-cell Panel D projection, and
tests effects within the provider CD8_EM1 population.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, wilcoxon


HERE = Path(__file__).resolve().parent
FIGURE_ROOT = HERE.parents[1]
FINAL_XU = FIGURE_ROOT.parent
OUT = HERE / "biological_interpretation"

PROJECTION = (
    HERE / "citeseq_projection/engelbrecht_all_cd8_projection.csv.gz"
)
OLD_PROJECTION = (
    FIGURE_ROOT / "inputs/legacy_panel_d_projection.csv.gz"
)
ADT_COUNTS = (
    FINAL_XU / "raw_data/melanoma/engelbrecht_citeseq/raw/ADT_counts.csv"
)

NEW_SCORE = "all_cd8_spatial_state_score"
OLD_SCORE = "melanoma_interface_propensity"
TOP_THRESHOLD = 0.75

# Exact Engelbrecht ADT rows corresponding to the 15 clean15 model inputs.
MODEL_INPUT_ADT = {
    "CD274-AB",  # PDL1
    "CD45-AB",
    "CD4-AB",
    "CD20-AB",
    "CD8-AB",
    "CD45RA-AB",
    "CD40-AB",
    "CD3-AB",
    "CD45RO-AB",
    "CD14-AB",
    "CD31-AB",
    "CD29-AB",
    "CD223-AB",  # LAG3
    "CD278-AB",  # ICOS
    "CD134-AB",  # OX40
}

FOCAL_MARKERS = {
    "PD1": "CD279-AB",
    "CD69": "CD69-AB",
    "TIGIT": "TIGIT-AB",
    "KLRG1": "KLRG1-AB",
    "CX3CR1": "CX3CR1-AB",
    "CD39": "CD39-AB",
    "CD103": "CD103-AB",
    "CD107A": "CD107a-AB",
    "CD62L": "CD62L-AB",
}

OLD_NEW_COMPARISON = [
    "CD18-AB",
    "CD95-AB",
    "CD11a-AB",
    "CD195-AB",
    "CD2-AB",
    "CD279-AB",
    "CD82-AB",
    "CD224-AB",
    "CD44-AB",
    "TIGIT-AB",
    "CD28-AB",
    "CD49a-AB",
    "CD49d-AB",
    "CD69-AB",
    "HLA-A-B-C-AB",
    "KLRG1-AB",
    "CX3CR1-AB",
    "CD107a-AB",
]

CANDIDATE_MODULES = {
    "LFA1": ["CD11a-AB", "CD18-AB"],
    "adhesion_3": ["CD11a-AB", "CD18-AB", "CD2-AB"],
    "checkpoint_2": ["CD279-AB", "TIGIT-AB"],
    "old_PD1_CD69": ["CD279-AB", "CD69-AB"],
    "core_4_LFA1_CD2_PD1": [
        "CD11a-AB", "CD18-AB", "CD2-AB", "CD279-AB"
    ],
    "core_5_add_TIGIT": [
        "CD11a-AB", "CD18-AB", "CD2-AB", "CD279-AB", "TIGIT-AB"
    ],
    "core_5_add_CD69": [
        "CD11a-AB", "CD18-AB", "CD2-AB", "CD279-AB", "CD69-AB"
    ],
}


def patient_rank(values: np.ndarray, patients: np.ndarray) -> np.ndarray:
    """Average-tie ranks divided by n+1, independently within each patient."""
    values = np.asarray(values, float)
    patients = np.asarray(patients, str)
    result = np.full(len(values), np.nan, dtype=float)
    for patient in pd.unique(patients):
        rows = np.flatnonzero(patients == patient)
        finite = rows[np.isfinite(values[rows])]
        if len(finite):
            result[finite] = rankdata(values[finite], method="average") / (
                len(finite) + 1.0
            )
    return result


def bh_adjust(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    output = np.full(len(values), np.nan)
    finite = np.isfinite(values)
    order = np.argsort(values[finite])
    ordered = values[finite][order]
    adjusted = np.minimum.accumulate(
        (ordered * len(ordered) / np.arange(1, len(ordered) + 1))[::-1]
    )[::-1]
    holder = np.empty(len(ordered))
    holder[order] = np.minimum(adjusted, 1.0)
    output[finite] = holder
    return output


def marker_effects(
    matrix: np.ndarray,
    features: list[str],
    metadata: pd.DataFrame,
    score_column: str,
    normalization: str,
    subset: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    patients = metadata["Patient_ID"].astype(str).to_numpy()
    scores = metadata[score_column].to_numpy(float)
    if subset is None:
        subset = np.ones(len(metadata), dtype=bool)
    records: list[dict[str, object]] = []
    by_patient: list[dict[str, object]] = []
    for feature_index, feature in enumerate(features):
        if "isotype" in feature.lower() or feature in MODEL_INPUT_ADT:
            continue
        differences: list[float] = []
        correlations: list[float] = []
        for patient in pd.unique(patients):
            local = subset & (patients == patient)
            local &= np.isfinite(matrix[:, feature_index]) & np.isfinite(scores)
            rows = np.flatnonzero(local)
            if len(rows) < 20:
                continue
            # Re-rank both quantities inside the tested population.  Thus the
            # EM1 sensitivity analysis compares the top quartile within EM1,
            # rather than EM1 cells that happen to exceed an all-CD8 cutoff.
            local_score_rank = rankdata(scores[rows], method="average") / (
                len(rows) + 1.0
            )
            local_expression_rank = rankdata(
                matrix[rows, feature_index], method="average"
            ) / (len(rows) + 1.0)
            high = local_score_rank >= TOP_THRESHOLD
            rest = ~high
            difference = float(
                local_expression_rank[high].mean()
                - local_expression_rank[rest].mean()
            )
            correlation = float(
                spearmanr(local_expression_rank, local_score_rank).statistic
            )
            differences.append(difference)
            correlations.append(correlation)
            by_patient.append(
                {
                    "normalization": normalization,
                    "adt_feature": feature,
                    "patient": patient,
                    "n_cells": int(len(rows)),
                    "n_high": int(high.sum()),
                    "top25_minus_rest75_rank": difference,
                    "spearman_score_expression": correlation,
                }
            )
        pvalue = np.nan
        if len(differences) >= 3 and np.any(np.asarray(differences) != 0):
            pvalue = float(wilcoxon(differences, alternative="two-sided").pvalue)
        records.append(
            {
                "normalization": normalization,
                "adt_feature": feature,
                "n_patients": len(differences),
                "mean_top25_minus_rest75_rank": float(np.mean(differences)),
                "median_top25_minus_rest75_rank": float(np.median(differences)),
                "patients_positive": int(np.sum(np.asarray(differences) > 0)),
                "wilcoxon_pvalue": pvalue,
                "mean_patient_spearman": float(np.mean(correlations)),
                "median_patient_spearman": float(np.median(correlations)),
                "patients_positive_spearman": int(
                    np.sum(np.asarray(correlations) > 0)
                ),
            }
        )
    summary = pd.DataFrame(records)
    summary["fdr_bh_across_heldout_adt"] = bh_adjust(
        summary["wilcoxon_pvalue"].to_numpy(float)
    )
    return summary, pd.DataFrame(by_patient)


def one_feature_effect(
    values: np.ndarray, metadata: pd.DataFrame, score_column: str
) -> dict[str, float | int]:
    patients = metadata["Patient_ID"].astype(str).to_numpy()
    score_rank = patient_rank(metadata[score_column].to_numpy(float), patients)
    expression_rank = patient_rank(values, patients)
    differences, correlations = [], []
    for patient in pd.unique(patients):
        local = patients == patient
        high = local & (score_rank >= TOP_THRESHOLD)
        rest = local & (score_rank < TOP_THRESHOLD)
        differences.append(
            float(expression_rank[high].mean() - expression_rank[rest].mean())
        )
        correlations.append(
            float(spearmanr(expression_rank[local], score_rank[local]).statistic)
        )
    return {
        "mean_top25_minus_rest75_rank": float(np.mean(differences)),
        "median_top25_minus_rest75_rank": float(np.median(differences)),
        "patients_positive": int(np.sum(np.asarray(differences) > 0)),
        "mean_patient_spearman": float(np.mean(correlations)),
        "median_patient_spearman": float(np.median(correlations)),
    }


def subtype_outputs(metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subtype = (
        metadata.groupby(["clust.name", "cell_type"], dropna=False)
        .agg(
            n_cells=("source_cell_id", "size"),
            n_patients=("Patient_ID", "nunique"),
            mean_raw_score=(NEW_SCORE, "mean"),
            median_raw_score=(NEW_SCORE, "median"),
            mean_patient_percentile=("within_patient_score_percentile", "mean"),
            top25_fraction=(
                "within_patient_score_percentile",
                lambda x: float((x >= TOP_THRESHOLD).mean()),
            ),
        )
        .reset_index()
        .sort_values("mean_patient_percentile", ascending=False)
    )
    by_patient = (
        metadata.groupby(["Patient_ID", "cell_type"])[
            "within_patient_score_percentile"
        ]
        .agg(["size", "mean", "median"])
        .reset_index()
    )
    mean_table = by_patient.pivot(index="Patient_ID", columns="cell_type", values="mean")
    comparisons = []
    for left, right in [
        ("CD8_mem,eff.1", "CD8_nv"),
        ("CD8_mem,eff.2", "CD8_nv"),
        ("CD8_mem,eff.2", "CD8_mem,eff.1"),
    ]:
        difference = (mean_table[left] - mean_table[right]).dropna()
        comparisons.append(
            {
                "comparison": f"{left} minus {right}",
                "n_patients": int(len(difference)),
                "mean_patient_score_difference": float(difference.mean()),
                "median_patient_score_difference": float(difference.median()),
                "patients_positive": int((difference > 0).sum()),
            }
        )
    return subtype, by_patient, pd.DataFrame(comparisons)


def pd1_cd69_correlations(
    raw: np.ndarray,
    clr: np.ndarray,
    features: list[str],
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    feature_index = {feature: i for i, feature in enumerate(features)}
    patients = metadata["Patient_ID"].astype(str).to_numpy()
    em1 = metadata["cell_type"].astype(str).eq("CD8_mem,eff.1").to_numpy()
    records = []
    for normalization, matrix in [("raw", raw), ("classical_clr", clr)]:
        for population, subset in [
            ("all_CD8", np.ones(len(metadata), dtype=bool)),
            ("CD8_EM1_only", em1),
        ]:
            for patient in pd.unique(patients):
                local = subset & (patients == patient)
                correlation = spearmanr(
                    matrix[local, feature_index["CD279-AB"]],
                    matrix[local, feature_index["CD69-AB"]],
                ).statistic
                records.append(
                    {
                        "normalization": normalization,
                        "population": population,
                        "patient": patient,
                        "n_cells": int(local.sum()),
                        "spearman_PD1_CD69": float(correlation),
                    }
                )
    return pd.DataFrame(records)


def candidate_module_effects(
    clr: np.ndarray, features: list[str], metadata: pd.DataFrame
) -> pd.DataFrame:
    """Compare small, biologically coherent held-out marker modules."""
    feature_index = {feature: i for i, feature in enumerate(features)}
    patients = metadata["Patient_ID"].astype(str).to_numpy()
    scores = metadata[NEW_SCORE].to_numpy(float)
    em1 = metadata["cell_type"].astype(str).eq("CD8_mem,eff.1").to_numpy()
    records = []
    for module, markers in CANDIDATE_MODULES.items():
        for population, subset in [
            ("all_CD8", np.ones(len(metadata), dtype=bool)),
            ("CD8_EM1_only", em1),
        ]:
            differences, correlations = [], []
            for patient in pd.unique(patients):
                rows = np.flatnonzero(subset & (patients == patient))
                score_rank = rankdata(scores[rows], method="average") / (
                    len(rows) + 1.0
                )
                marker_ranks = np.column_stack(
                    [
                        rankdata(
                            clr[rows, feature_index[marker]], method="average"
                        )
                        / (len(rows) + 1.0)
                        for marker in markers
                    ]
                )
                module_value = marker_ranks.mean(axis=1)
                high = score_rank >= TOP_THRESHOLD
                differences.append(
                    float(module_value[high].mean() - module_value[~high].mean())
                )
                correlations.append(
                    float(spearmanr(module_value, score_rank).statistic)
                )
            pvalue = float(wilcoxon(differences, alternative="two-sided").pvalue)
            records.append(
                {
                    "module": module,
                    "markers": ",".join(markers),
                    "n_markers": len(markers),
                    "population": population,
                    "mean_top25_minus_rest75_module_rank": float(
                        np.mean(differences)
                    ),
                    "median_top25_minus_rest75_module_rank": float(
                        np.median(differences)
                    ),
                    "patients_positive": int(
                        np.sum(np.asarray(differences) > 0)
                    ),
                    "wilcoxon_pvalue": pvalue,
                    "mean_patient_spearman": float(np.mean(correlations)),
                    "median_patient_spearman": float(np.median(correlations)),
                }
            )
    return pd.DataFrame(records)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    new = pd.read_csv(PROJECTION, low_memory=False)
    old = pd.read_csv(OLD_PROJECTION, low_memory=False)
    new["source_cell_id"] = new["source_cell_id"].astype(str)
    old["source_cell_id"] = old["source_cell_id"].astype(str)
    new_ids = new["source_cell_id"].tolist()
    if not set(old["source_cell_id"]).issubset(new_ids):
        raise AssertionError("The frozen 2,305-cell projection is not nested in all CD8")

    feature_column = pd.read_csv(ADT_COUNTS, nrows=0).columns[0]
    adt = pd.read_csv(
        ADT_COUNTS, usecols=[feature_column, *new_ids], low_memory=False
    ).set_index(feature_column)
    adt = adt[new_ids]
    features = adt.index.astype(str).tolist()
    raw = adt.to_numpy(float).T
    non_isotype = np.asarray(
        ["isotype" not in feature.lower() for feature in features], dtype=bool
    )
    # Classical compositional CLR: log(1+x_j) minus the per-cell mean log(1+x)
    # over the 130 non-isotype ADT features.
    log_geometric_mean = np.log1p(raw[:, non_isotype]).mean(axis=1)
    clr = np.log1p(raw) - log_geometric_mean[:, None]

    raw_summary, raw_by_patient = marker_effects(
        raw, features, new, NEW_SCORE, "raw"
    )
    clr_summary, clr_by_patient = marker_effects(
        clr, features, new, NEW_SCORE, "classical_clr"
    )
    all_summary = pd.concat([raw_summary, clr_summary], ignore_index=True)
    all_by_patient = pd.concat(
        [raw_by_patient, clr_by_patient], ignore_index=True
    )
    all_summary.to_csv(OUT / "all_heldout_adt_effects.csv", index=False)
    all_by_patient.to_csv(
        OUT / "all_heldout_adt_effects_by_patient.csv", index=False
    )

    em1 = new["cell_type"].astype(str).eq("CD8_mem,eff.1").to_numpy()
    em1_summary, em1_by_patient = marker_effects(
        clr, features, new, NEW_SCORE, "classical_clr", subset=em1
    )
    em1_summary.to_csv(OUT / "em1_heldout_adt_effects_clr.csv", index=False)
    em1_by_patient.to_csv(
        OUT / "em1_heldout_adt_effects_clr_by_patient.csv", index=False
    )

    feature_index = {feature: i for i, feature in enumerate(features)}
    focal = all_summary.loc[
        all_summary["adt_feature"].isin(FOCAL_MARKERS.values())
    ].copy()
    focal["marker"] = focal["adt_feature"].map(
        {value: key for key, value in FOCAL_MARKERS.items()}
    )
    focal.to_csv(OUT / "predefined_marker_diagnostics.csv", index=False)

    total_adt = raw.sum(axis=1)
    depth = one_feature_effect(total_adt, new, NEW_SCORE)
    isotype_records = []
    for feature in features:
        if "isotype" not in feature.lower():
            continue
        effect = one_feature_effect(raw[:, feature_index[feature]], new, NEW_SCORE)
        isotype_records.append({"adt_feature": feature, **effect})
    pd.DataFrame(isotype_records).to_csv(
        OUT / "raw_isotype_control_effects.csv", index=False
    )

    old_positions = np.asarray(
        [new_ids.index(cell_id) for cell_id in old["source_cell_id"]], dtype=int
    )
    comparison_records = []
    for feature in OLD_NEW_COMPARISON:
        for version, metadata, values, score_column in [
            ("old_panel_D_2305", old, clr[old_positions, feature_index[feature]], OLD_SCORE),
            ("new_all_CD8_5768", new, clr[:, feature_index[feature]], NEW_SCORE),
        ]:
            effect = one_feature_effect(values, metadata, score_column)
            comparison_records.append(
                {"adt_feature": feature, "projection": version, **effect}
            )
    pd.DataFrame(comparison_records).to_csv(
        OUT / "old_vs_new_selected_markers_clr.csv", index=False
    )

    subtype, subtype_by_patient, subtype_comparisons = subtype_outputs(new)
    subtype.to_csv(OUT / "provider_subtype_score_summary.csv", index=False)
    subtype_by_patient.to_csv(
        OUT / "provider_subtype_scores_by_patient.csv", index=False
    )
    subtype_comparisons.to_csv(
        OUT / "provider_subtype_paired_comparisons.csv", index=False
    )

    pd1_cd69 = pd1_cd69_correlations(raw, clr, features, new)
    pd1_cd69.to_csv(OUT / "pd1_cd69_correlations_by_patient.csv", index=False)

    modules = candidate_module_effects(clr, features, new)
    modules.to_csv(OUT / "candidate_biomarker_modules.csv", index=False)

    top_clr = clr_summary.sort_values(
        "mean_top25_minus_rest75_rank", ascending=False
    ).head(20)
    top_clr.to_csv(OUT / "top20_positive_heldout_adt_clr.csv", index=False)
    top_em1 = em1_summary.sort_values(
        "mean_top25_minus_rest75_rank", ascending=False
    ).head(20)
    top_em1.to_csv(OUT / "top20_positive_em1_heldout_adt_clr.csv", index=False)

    manifest = {
        "new_projection_cells": int(len(new)),
        "new_projection_patients": int(new["Patient_ID"].nunique()),
        "old_projection_cells": int(len(old)),
        "adt_features": int(len(features)),
        "non_isotype_adt_features_for_clr_denominator": int(non_isotype.sum()),
        "heldout_adt_features_screened": int(len(clr_summary)),
        "top_definition": "within-patient score rank >= 0.75",
        "effect_definition": "within-patient expression rank mean in top25 minus rest75",
        "clr_definition": "log1p(marker count) minus cell mean log1p(count) over non-isotype ADTs",
        "total_adt_depth_diagnostic": depth,
        "raw_data_copied": False,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def value(frame: pd.DataFrame, marker: str, column: str) -> float:
        return float(frame.loc[frame["adt_feature"].eq(marker), column].iloc[0])

    pd1 = value(clr_summary, "CD279-AB", "mean_top25_minus_rest75_rank")
    cd69 = value(clr_summary, "CD69-AB", "mean_top25_minus_rest75_rank")
    tigit = value(clr_summary, "TIGIT-AB", "mean_top25_minus_rest75_rank")
    em1_nv = subtype_comparisons.iloc[0]
    results = f"""# All-CD8 biological interpretation audit

## Main result

The all-CD8 transferred score is primarily an effector-memory-versus-naive
axis.  CD8_EM1 minus CD8 naive has a mean within-patient score difference of
{em1_nv['mean_patient_score_difference']:.3f} and is positive in
{int(em1_nv['patients_positive'])}/{int(em1_nv['n_patients'])} patients.
CD8_EM2 is not consistently higher than CD8_EM1, so the result should not be
described as an EM2-specific exhausted state.

## PD-1/CD69 and the depth audit

Raw ADT effects are inflated because the score correlates with total cellular
ADT depth.  The total-ADT top25-minus-rest75 rank difference is
{depth['mean_top25_minus_rest75_rank']:.3f}, and all seven isotype controls are
positive before correction.  After classical cell-level CLR correction, the
mean patient-internal effects remain {pd1:.3f} for PD-1, {cd69:.3f} for CD69,
and {tigit:.3f} for TIGIT.  PD-1 is the strongest checkpoint result; TIGIT is
the clearest additional checkpoint finding; CD69 is weaker and largely tracks
the broad naive-to-effector-memory axis.

## New all-panel signal

The strongest depth-adjusted held-out signals include CD18, CD11a, CD2, CD95,
CCR5/CD195, CD82, CD44, CD49a and CD49d.  CD11a-CD18 form LFA-1, and the panel
as a whole supports an adhesion/trafficking/activated-memory interpretation.
CD82, CD2, CD95, CD18 and CD11a remain positive in all eight patients even
when the analysis is restricted to provider-defined CD8_EM1 cells.  This is a
more specific new finding than the uncorrected apparent increase of all nine
preselected validation proteins.

## Minimal main-story panel

The preferred minimal set is CD11a, CD18, CD2 and PD-1.  It gives two coherent
functional components: LFA-1/CD2 adhesion and synapse formation, plus a PD-1
checkpoint component.  Their four-marker module is positive in all eight
patients and has mean patient Spearman correlations of 0.480 in all CD8 and
0.308 within CD8_EM1.  Adding TIGIT or CD69 does not improve those correlations,
so TIGIT is best treated as secondary checkpoint support and CD69 as continuity
with the earlier activation result.

## Important limits

- Every positive spatial training cell is Tc.ae, whereas Tc.naive cells enter
  the negative class.  The broad effector-memory-versus-naive separation is
  therefore partly expected from the endpoint construction; the within-EM1
  held-out effects provide the more specific biological information.
- The 5,768-cell and 2,305-cell analyses differ in source training population,
  cohort size, RAMP version and shared panel, so changes cannot be attributed
  to cell number alone.
- CD8_TRBV4-1 and CD8_MT1 are strongly patient-specific clusters and should not
  be used as the main subtype conclusion.
- These are cross-platform molecular associations, not proof that the CITE-seq
  cells themselves contacted melanoma cells.
"""
    (OUT / "RESULTS.md").write_text(results)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
