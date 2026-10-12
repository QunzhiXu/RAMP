#!/usr/bin/env python3
"""Validate dimensions, locks, geometry, and score reconstruction for PT74."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
PAIR70 = HERE.parent / "tigit_cd39_return/output"
FIGURE_HNSCC = HERE.parents[1]


def main() -> None:
    source = pd.read_csv(OUT / "rahim_tumor_cd8_locked_tigit_cd39.csv.gz")
    target = pd.read_csv(
        OUT / "haist74_pt_cd8_tigit_cd39_return.csv.gz",
        dtype={"cell_uid": str, "patient": str},
    )
    features = pd.read_csv(
        OUT / "haist74_tigit_cd39_patient_features.csv", dtype={"patient": str}
    )
    models = pd.read_csv(OUT / "joint_spatial_molecular_survival_models.csv")
    manifest = json.loads((OUT / "PAIR_LOCK_AND_RETURN_MANIFEST.json").read_text())
    alignment_manifest = json.loads(
        (HERE / "alignment/alignment_manifest.json").read_text()
    )

    if len(source) != 4_788 or source.donor.nunique() != 3:
        raise AssertionError("Source dimensions changed")
    if len(target) != 38_299 or target.patient.nunique() != 74:
        raise AssertionError("Complete-cohort target dimensions changed")
    if len(features) != 74 or len(models) != 6:
        raise AssertionError("Patient/model dimensions changed")
    if int(features.pt_cd8_cells.min()) != 3 or int(features.pt_cd8_cells.sum()) != 38_299:
        raise AssertionError("Unfiltered PT-CD8 counts changed")
    if not np.isfinite(target.select_dtypes(include=[np.number])).all().all():
        raise AssertionError("Non-finite numeric target output")
    if manifest["program"]["markers_raw"] != ["TIGIT--VSTM3", "CD39"]:
        raise AssertionError("Locked program changed")
    if manifest["survival_outcome_read"] is not False:
        raise AssertionError("Outcome-blind phase claims to have opened survival")
    if alignment_manifest["canonical_version"] != "2.3.0":
        raise AssertionError("PT74 alignment did not use canonical RAMP 2.3.0")
    if alignment_manifest["strict_rank_inversions"] != 0:
        raise AssertionError("PT74 alignment failed strict rank preservation")

    old = pd.read_csv(
        PAIR70 / "haist70_pt_cd8_tigit_cd39_return.csv.gz",
        dtype={"cell_uid": str, "patient": str},
        usecols=["cell_uid", "patient", "predicted_tigit_cd39"],
    ).rename(columns={"predicted_tigit_cd39": "prediction_pt70"})
    paired = old.merge(
        target[["cell_uid", "patient", "predicted_tigit_cd39"]],
        on=["cell_uid", "patient"], validate="one_to_one",
    ).rename(columns={"predicted_tigit_cd39": "prediction_pt74"})
    if len(paired) != 38_194 or paired.patient.nunique() != 70:
        raise AssertionError("PT70/PT74 overlap is not the expected 38,194 cells")
    patient_rhos = []
    for patient, frame in paired.groupby("patient", sort=True):
        patient_rhos.append(
            {
                "patient": patient,
                "cells": len(frame),
                "spearman_pt70_vs_pt74": float(
                    spearmanr(frame.prediction_pt70, frame.prediction_pt74).statistic
                ),
            }
        )
    patient_rhos = pd.DataFrame(patient_rhos)
    patient_rhos.to_csv(OUT / "pt70_vs_pt74_prediction_by_patient.csv", index=False)

    direct = pd.read_csv(
        FIGURE_HNSCC / "survival_motivation/output/survival_analysis_input.csv",
        usecols=["CaseID", "patient_spatial_niche_score"],
        dtype={"CaseID": str},
    )
    checked = features.merge(
        direct, left_on="patient", right_on="CaseID", validate="one_to_one",
        suffixes=("_reconstructed", "_original"),
    )
    max_direct_error = float(
        np.max(
            np.abs(
                checked.patient_spatial_niche_score_reconstructed
                - checked.patient_spatial_niche_score_original
            )
        )
    )
    if max_direct_error > 1e-12:
        raise AssertionError(f"Reconstructed direct spatial score differs by {max_direct_error}")

    primary = models.loc[
        models.model.eq("Primary: high-state CD8 spatial exposure")
    ].iloc[0]
    leave = models.loc[models.model.eq("Sensitivity: exclude Case 41403")].iloc[0]
    report = {
        "status": "passed",
        "source_cells": len(source),
        "target_cells": len(target),
        "target_patients": target.patient.nunique(),
        "deaths": int(primary.deaths),
        "locked_markers": manifest["program"]["markers_raw"],
        "ramp_version": alignment_manifest["canonical_version"],
        "strict_rank_inversions": alignment_manifest["strict_rank_inversions"],
        "pt70_pt74_overlap_cells": len(paired),
        "pt70_vs_pt74_prediction_overall_spearman": float(
            spearmanr(paired.prediction_pt70, paired.prediction_pt74).statistic
        ),
        "pt70_vs_pt74_prediction_median_patient_spearman": float(
            patient_rhos.spearman_pt70_vs_pt74.median()
        ),
        "pt70_vs_pt74_prediction_min_patient_spearman": float(
            patient_rhos.spearman_pt70_vs_pt74.min()
        ),
        "max_direct_spatial_score_reconstruction_error": max_direct_error,
        "primary_complete74": {
            "hazard_ratio_per_sd": float(primary.hazard_ratio_per_sd),
            "ci95_low": float(primary.ci95_low),
            "ci95_high": float(primary.ci95_high),
            "wald_p": float(primary.wald_p),
        },
        "exclude_case41403": {
            "patients": int(leave.patients),
            "deaths": int(leave.deaths),
            "hazard_ratio_per_sd": float(leave.hazard_ratio_per_sd),
            "wald_p": float(leave.wald_p),
        },
    }
    (OUT / "VALIDATION_REPORT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
