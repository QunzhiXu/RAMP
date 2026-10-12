#!/usr/bin/env python3
"""Return the already-locked TIGIT/CD39 state to all 74 complete-OS patients.

This phase is outcome blind.  The molecular state and its selection rule are
imported unchanged from the PT70 analysis; only the RAMP alignment and return
target are rebuilt after removing the PT-CD8>=40 cohort filter.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
FIGURE_HNSCC = HERE.parents[1]
PT70 = HERE.parent / "pt70_full_90488_umap"
PAIR70 = HERE.parent / "tigit_cd39_return"
INPUT = HERE / "input"
ALIGNMENT = HERE / "alignment"
OUT = HERE / "output"
FORWARD_SCRIPT = PT70 / "04_project_niche_score_to_rahim.py"
PAIR_SCRIPT = PAIR70 / "01_project_fixed_pair_to_codex.py"
HAIST_CELLS = FIGURE_HNSCC / "sampled_data/haist_codex/complete/HNSCCdiscovery_CODEX.csv"
EXPOSURE = FIGURE_HNSCC / "survival_motivation/output/pt_cd8_cell_spatial_niche_scores.csv.gz"

EXPECTED_PATIENTS = 74
EXPECTED_TARGET_CELLS = 38_299


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_haist_curves(
    raw: np.ndarray,
    aligned: np.ndarray,
    metadata: pd.DataFrame,
    markers: list[str],
    monotone_curve_fn,
) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    target_rows = np.flatnonzero(metadata.dataset.eq("Haist_CODEX").to_numpy())
    local = metadata.iloc[target_rows].reset_index(drop=True)
    target_raw = np.asarray(raw[target_rows], np.float64)
    target_aligned = np.asarray(aligned[target_rows], np.float64)
    curves: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for patient, group_rows in local.groupby("donor", sort=True).groups.items():
        rows = np.asarray(list(group_rows), int)
        for marker_index, marker in enumerate(markers):
            curves[(str(patient), marker)] = monotone_curve_fn(
                target_raw[rows, marker_index], target_aligned[rows, marker_index]
            )
    expected = EXPECTED_PATIENTS * len(markers)
    if len(curves) != expected:
        raise AssertionError(f"Expected {expected} patient-marker curves; observed {len(curves)}")
    return curves


def load_full_haist_pt_cd8(
    selected: set[str],
    curves: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    markers: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    exposure = pd.read_csv(
        EXPOSURE,
        usecols=["UID", "Pat_ID", "roi_adjusted_caf_macrophage_score"],
        dtype={"UID": str, "Pat_ID": str},
    )
    usecols = ["UID", "Pat_ID", "Tissue_type_broad", "cell_types_mediumres"] + markers
    blocks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(HAIST_CELLS, usecols=usecols, chunksize=150_000):
        patients = chunk.Pat_ID.astype(str)
        keep = (
            patients.isin(selected)
            & chunk.Tissue_type_broad.astype(str).eq("PT")
            & chunk.cell_types_mediumres.astype(str).eq("CD8T")
        )
        if keep.any():
            block = chunk.loc[keep].copy()
            block["Pat_ID"] = patients.loc[keep].to_numpy(str)
            blocks.append(block)
    cells = pd.concat(blocks, ignore_index=True)
    cells["UID"] = cells.UID.astype(str)
    cells = cells.merge(exposure, on=["UID", "Pat_ID"], validate="one_to_one")
    if cells.roi_adjusted_caf_macrophage_score.isna().any():
        raise AssertionError("A complete-cohort PT-CD8 cell lacks its spatial score")
    counts = cells.groupby("Pat_ID").size()
    if len(counts) != EXPECTED_PATIENTS or len(cells) != EXPECTED_TARGET_CELLS:
        raise AssertionError(
            f"Expected {EXPECTED_TARGET_CELLS} PT-CD8 cells from {EXPECTED_PATIENTS} patients; "
            f"observed {len(cells)} cells from {len(counts)} patients"
        )
    if int(counts.min()) != 3:
        raise AssertionError("The unfiltered complete cohort should include the 3-cell patient")

    transformed = np.empty((len(cells), len(markers)), np.float64)
    for patient, group_rows in cells.groupby("Pat_ID", sort=True).groups.items():
        rows = np.asarray(list(group_rows), int)
        patient_raw = cells.loc[rows, markers].to_numpy(float)
        for marker_index, marker in enumerate(markers):
            x_curve, y_curve = curves[(str(patient), marker)]
            transformed[rows, marker_index] = np.interp(
                patient_raw[:, marker_index], x_curve, y_curve
            )
    return cells, transformed


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    forward = import_file("pt74_forward_helpers", FORWARD_SCRIPT)
    pair = import_file("pt74_locked_pair_helpers", PAIR_SCRIPT)
    lock = pair.lock_high_confidence_pair()

    raw = np.load(INPUT / "raw_shared_markers.float64.npy", mmap_mode="r")
    aligned = np.load(ALIGNMENT / "aligned_ramp_ver2.float32.npy", mmap_mode="r")
    metadata = pd.read_csv(
        INPUT / "cell_metadata.csv.gz",
        dtype={"cell_uid": str, "dataset": str, "donor": str, "unit_key": str},
        low_memory=False,
    )
    selected = set(
        pd.read_csv(INPUT / "selected_74_patients.csv", usecols=["CaseID"]).CaseID.astype(str)
    )
    if len(selected) != EXPECTED_PATIENTS:
        raise AssertionError("Complete-cohort patient list changed")
    curves = build_haist_curves(raw, aligned, metadata, forward.MARKERS, forward.monotone_curve)
    haist, haist_aligned = load_full_haist_pt_cd8(selected, curves, forward.MARKERS)

    rahim_rows = np.flatnonzero(
        metadata.dataset.eq("GSE212797_ADT")
        & metadata.tissue.astype(str).eq("tumor")
        & metadata.provider_cell_type.astype(str).eq("CD8 T cell")
    )
    rahim_meta = metadata.iloc[rahim_rows].copy().reset_index(drop=True)
    rahim_ids = rahim_meta.cell_uid.str.removeprefix("RAHIM|").to_numpy(str)
    rahim_donors = rahim_meta.donor.to_numpy(str)
    rahim_aligned = np.asarray(aligned[rahim_rows], np.float64)
    if len(rahim_meta) != 4_788:
        raise AssertionError("Expected 4,788 Rahim tumour-CD8 cells")

    source, source_score = pair.read_rahim_pair(rahim_ids, rahim_donors)
    projected = pd.read_csv(
        pair.PROJECTED_NICHE, dtype={"cell_uid": str, "donor": str}
    )[["cell_uid", "donor", "projected_roi_adjusted_exposure"]]
    source = source.merge(projected, on=["cell_uid", "donor"], validate="one_to_one")
    source_donor_results = []
    for donor, frame in source.groupby("donor", sort=True):
        source_donor_results.append(
            {
                "donor": str(donor),
                "cells": int(len(frame)),
                "spearman_pair_vs_projected_niche": float(
                    spearmanr(frame.tigit_cd39_score, frame.projected_roi_adjusted_exposure).statistic
                ),
            }
        )
    source_donor_results = pd.DataFrame(source_donor_results)

    haist_patients = haist.Pat_ID.to_numpy(str)
    mean, scale = forward.equal_donor_patient_scaler(
        rahim_aligned, rahim_donors, haist_aligned, haist_patients
    )
    source_z = (rahim_aligned - mean) / scale
    target_z = (haist_aligned - mean) / scale
    predicted, donor_predictions = pair.predict_by_source_donor(
        source_z, target_z, rahim_donors, source_score, forward.weighted_predict
    )

    exposure = pd.read_csv(
        EXPOSURE,
        usecols=[
            "UID", "Pat_ID", "unique_region", "x", "y",
            "roi_adjusted_caf_macrophage_score", "in_caf_macrophage_rich_niche",
        ],
        dtype={"UID": str, "Pat_ID": str},
    ).rename(
        columns={
            "UID": "cell_uid", "Pat_ID": "patient",
            "roi_adjusted_caf_macrophage_score": "spatial_y",
        }
    )
    target = haist[["UID", "Pat_ID"]].rename(columns={"UID": "cell_uid", "Pat_ID": "patient"})
    target["patient"] = target.patient.astype(str)
    target = target.merge(exposure, on=["cell_uid", "patient"], how="left", validate="one_to_one")
    if target.spatial_y.isna().any() or len(target) != EXPECTED_TARGET_CELLS:
        raise AssertionError("The complete 38,299-cell target did not join exactly")
    target["predicted_tigit_cd39"] = predicted
    donor_order = sorted(pd.unique(rahim_donors), key=str)
    for column, donor in enumerate(donor_order):
        target[f"predicted_from_rahim_{donor}"] = donor_predictions[:, column]
    target["predicted_pair_rank_within_patient"] = target.groupby(
        "patient", sort=False
    ).predicted_tigit_cd39.transform(pair.percentile_midrank)
    target["spatial_y_rank_within_patient"] = target.groupby(
        "patient", sort=False
    ).spatial_y.transform(pair.percentile_midrank)

    features = pair.build_patient_features(target)
    enrichment = pair.summarize_enrichment(features)
    source_path = OUT / "rahim_tumor_cd8_locked_tigit_cd39.csv.gz"
    target_path = OUT / "haist74_pt_cd8_tigit_cd39_return.csv.gz"
    feature_path = OUT / "haist74_tigit_cd39_patient_features.csv"
    source_effect_path = OUT / "rahim_pair_vs_projected_niche_by_donor.csv"
    pair.write_csv_gz(source, source_path)
    pair.write_csv_gz(target, target_path)
    features.to_csv(feature_path, index=False)
    source_donor_results.to_csv(source_effect_path, index=False)

    manifest = {
        "status": "outcome-blind locked TIGIT/CD39 return to all 74 complete-OS patients",
        "survival_outcome_read": False,
        "program": lock,
        "program_lock_source": "PT70 screen retained unchanged; no marker reselection on PT74",
        "source_state": (
            "unweighted mean of within-Rahim-donor percentile midranks of "
            "strict-no-LAG3 paper-CLR TIGIT and CD39"
        ),
        "return_alignment": "new complete-OS PT74 canonical RAMP 2.3.0 alignment",
        "return_predictor": (
            "distance-weighted k=30 Rahim-to-Haist predictor per Rahim donor; "
            "median across the three donor-specific predictions"
        ),
        "source_cells": int(len(source)),
        "source_donors": donor_order,
        "target_cells": int(len(target)),
        "target_patients": int(features.patient.nunique()),
        "minimum_target_pt_cd8_per_patient": int(features.pt_cd8_cells.min()),
        "niche_definition": "ROI-adjusted local k15 CAF/macrophage exposure > 0",
        "joint_survival_feature": (
            "mean continuous ROI-adjusted CAF/macrophage exposure among each patient's "
            "top within-patient quartile of returned TIGIT/CD39 scores"
        ),
        "enrichment": enrichment,
        "inputs_sha256": {
            "alignment": pair.sha256(ALIGNMENT / "aligned_ramp_ver2.float32.npy"),
            "metadata": pair.sha256(INPUT / "cell_metadata.csv.gz"),
            "locked_screen": pair.sha256(pair.SCREEN),
            "locked_projected_niche": pair.sha256(pair.PROJECTED_NICHE),
            "spatial_exposure": pair.sha256(EXPOSURE),
        },
        "outputs_sha256": {
            "source": pair.sha256(source_path),
            "target": pair.sha256(target_path),
            "patient_features": pair.sha256(feature_path),
            "source_associations": pair.sha256(source_effect_path),
        },
        "boundary": (
            "This is a complete-cohort stability analysis after removing the PT-CD8>=40 "
            "filter. The pair and cell-level feature are outcome blind, but the endpoint was "
            "already examined in PT70, and the four added patients are not an independent cohort."
        ),
    }
    (OUT / "PAIR_LOCK_AND_RETURN_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT / "NICHE_ENRICHMENT_SUMMARY.json").write_text(json.dumps(enrichment, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
