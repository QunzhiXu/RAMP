#!/usr/bin/env python3
"""Validate the newly generated HNSCC survival-motivation artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
FIG = HERE / "figures"


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    cell_path = OUT / "pt_cd8_cell_spatial_niche_scores.csv.gz"
    roi_path = OUT / "pt_roi_spatial_niche_scores.csv"
    patient_path = OUT / "patient_spatial_niche_scores.csv"
    survival_path = OUT / "survival_analysis_input.csv"
    model_path = OUT / "survival_models.csv"

    cells = pd.read_csv(cell_path, dtype={"UID": str})
    rois = pd.read_csv(roi_path)
    patients = pd.read_csv(patient_path)
    survival = pd.read_csv(survival_path)
    models = pd.read_csv(model_path)

    check(not cells["UID"].duplicated().any(), "Duplicate cell UID")
    check(not rois["unique_region"].duplicated().any(), "Duplicate PT ROI")
    check(not patients["CaseID"].duplicated().any(), "Duplicate spatial patient")
    check(not survival["CaseID"].duplicated().any(), "Duplicate survival patient")

    check(cells["local_caf_macrophage_count_k15"].between(0, 15).all(), "Invalid k15 count")
    check(
        np.allclose(
            cells["local_caf_macrophage_fraction_k15"],
            cells["local_caf_macrophage_count_k15"] / 15.0,
            atol=1e-12,
        ),
        "Local fraction does not equal count/15",
    )
    check(
        np.allclose(
            cells["roi_adjusted_caf_macrophage_score"],
            cells["local_caf_macrophage_fraction_k15"]
            - cells["roi_nonself_caf_macrophage_fraction"],
            atol=1e-12,
        ),
        "Cell residual-score identity failed",
    )

    recomputed = (
        cells.groupby("Pat_ID")["roi_adjusted_caf_macrophage_score"].mean().sort_index()
    )
    stored = patients.set_index("CaseID")["patient_spatial_niche_score"].sort_index()
    recomputed.index = recomputed.index.astype(int)
    check(recomputed.index.equals(stored.index), "Patient IDs differ between cell and patient tables")
    check(np.allclose(recomputed, stored, atol=1e-12), "Patient score is not the cell-weighted mean")

    check(len(cells) == int(patients["pt_cd8_cells"].sum()), "Patient CD8 counts do not sum to cells")
    check(len(rois) == int(patients["evaluable_pt_rois"].sum()), "Patient ROI counts do not sum")
    check(len(survival) == 74, "Survival cohort is not all 74 complete-OS patients")
    check(int(survival["os_event"].sum()) == 24, "Expected 24 deaths")
    check((survival["pt_cd8_cells"] >= 1).all(), "A survival patient has no PT CD8 cell")

    primary = models.loc[models["model"].eq("Primary: continuous spatial score")]
    check(len(primary) == 1, "Primary continuous Cox model is missing or duplicated")
    check(int(primary.iloc[0]["patients"]) == 74, "Primary model did not use all 74 patients")
    check(int(primary.iloc[0]["deaths"]) == 24, "Primary model death count is wrong")
    check(float(primary.iloc[0]["hazard_ratio"]) > 0, "Primary HR is invalid")

    png_path = FIG / "hnscc_pt_cd8_niche_survival.png"
    pdf_path = FIG / "hnscc_pt_cd8_niche_survival.pdf"
    check(png_path.is_file() and png_path.stat().st_size > 10_000, "Missing or empty PNG figure")
    check(pdf_path.is_file() and pdf_path.stat().st_size > 1_000, "Missing or empty PDF figure")
    check(pdf_path.read_bytes()[:5] == b"%PDF-", "Figure PDF has an invalid file header")

    report = {
        "status": "PASS",
        "checks": {
            "cell_score_formula": True,
            "patient_score_is_cell_weighted_mean": True,
            "unique_cell_roi_patient_ids": True,
            "survival_is_patient_level": True,
            "complete_os_cohort": {"patients": 74, "deaths": 24},
            "minimum_pt_cd8_threshold_used": False,
            "figures_exist": True,
        },
        "observed_counts": {
            "pt_cd8_cells": int(len(cells)),
            "evaluable_pt_rois": int(len(rois)),
            "patients_with_spatial_score": int(len(patients)),
            "survival_patients": int(len(survival)),
            "survival_deaths": int(survival["os_event"].sum()),
        },
        "primary_cox": {
            "hazard_ratio_per_sd": float(primary.iloc[0]["hazard_ratio"]),
            "ci95_low": float(primary.iloc[0]["ci95_low"]),
            "ci95_high": float(primary.iloc[0]["ci95_high"]),
            "wald_p": float(primary.iloc[0]["wald_p"]),
        },
    }
    (OUT / "validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
