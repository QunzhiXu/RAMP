#!/usr/bin/env python3
"""Audit the locked pair return against inputs and the historical Haist30 cube."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
FIGURE_HNSCC = HERE.parents[1]
OLD = (
    FIGURE_HNSCC.parent
    / "test0827/hnscc/pt_survival_first/biomarker_combination_search/"
      "output_haist40_selected"
)


def main() -> None:
    source = pd.read_csv(OUT / "rahim_tumor_cd8_locked_tigit_cd39.csv.gz")
    target = pd.read_csv(
        OUT / "haist70_pt_cd8_tigit_cd39_return.csv.gz",
        dtype={"cell_uid": str, "patient": str},
    )
    features = pd.read_csv(
        OUT / "haist70_tigit_cd39_patient_features.csv", dtype={"patient": str}
    )
    models = pd.read_csv(OUT / "survival_models.csv")
    joint_models = pd.read_csv(OUT / "joint_spatial_molecular_survival_models.csv")
    manifest = json.loads((OUT / "PAIR_LOCK_AND_RETURN_MANIFEST.json").read_text())

    if len(source) != 4_788 or source.donor.nunique() != 3:
        raise AssertionError("Source dimensions changed")
    if len(target) != 38_194 or target.patient.nunique() != 70:
        raise AssertionError("Target dimensions changed")
    if len(features) != 70 or len(models) != 7 or len(joint_models) != 5:
        raise AssertionError("Patient/model dimensions changed")
    if not np.isfinite(target.select_dtypes(include=[np.number])).all().all():
        raise AssertionError("Non-finite numeric target output")
    if manifest["program"]["markers_raw"] != ["TIGIT--VSTM3", "CD39"]:
        raise AssertionError("Program lock changed")
    if manifest["survival_outcome_read"] is not False:
        raise AssertionError("Outcome-blind phase claims to have opened survival")

    cube = np.load(OLD / "haist30_donor_marker_prediction_cube.float32.npy", mmap_mode="r")
    columns = pd.read_csv(OLD / "marker_prediction_cube_columns.csv")
    indices = [
        int(columns.loc[columns.marker_raw.eq(marker), "marker_index"].iloc[0])
        for marker in ["TIGIT--VSTM3", "CD39"]
    ]
    old_prediction = np.median(np.asarray(cube[:, :, indices]).mean(axis=2), axis=0)
    old_meta = pd.read_csv(
        OLD / "haist30_cells_endpoint.csv.gz",
        dtype={"raw_cell_uid": str, "patient": str},
    )[["raw_cell_uid", "patient"]]
    old_meta["historical_pair_prediction"] = old_prediction
    paired = old_meta.merge(
        target[["cell_uid", "patient", "predicted_tigit_cd39"]],
        left_on=["raw_cell_uid", "patient"],
        right_on=["cell_uid", "patient"],
        validate="one_to_one",
    )
    patient_correlations = []
    for patient, frame in paired.groupby("patient", sort=True):
        patient_correlations.append(
            spearmanr(
                frame.historical_pair_prediction, frame.predicted_tigit_cd39
            ).statistic
        )
    report = {
        "status": "passed",
        "source_cells": len(source),
        "target_cells": len(target),
        "target_patients": target.patient.nunique(),
        "survival_models": len(models),
        "joint_survival_models": len(joint_models),
        "historical_haist30_overlap_cells": len(paired),
        "historical_vs_pt70_pair_prediction_overall_spearman": float(
            spearmanr(
                paired.historical_pair_prediction, paired.predicted_tigit_cd39
            ).statistic
        ),
        "historical_vs_pt70_pair_prediction_median_patient_spearman": float(
            np.median(patient_correlations)
        ),
        "historical_vs_pt70_pair_prediction_min_patient_spearman": float(
            np.min(patient_correlations)
        ),
    }
    if report["historical_vs_pt70_pair_prediction_min_patient_spearman"] < 0.8:
        raise AssertionError("PT70 return disagrees unexpectedly with historical geometry")
    (OUT / "VALIDATION_REPORT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
