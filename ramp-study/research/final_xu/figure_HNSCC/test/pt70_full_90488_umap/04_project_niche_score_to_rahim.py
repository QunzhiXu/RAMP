#!/usr/bin/env python3
"""Transfer the PT70 spatial CD8 niche score into Rahim tumor CD8 cells.

This reproduces the final historical transfer design with the new PT-only,
70-patient RAMP 2.3.0 alignment: patient-specific Haist predictors, equal
Rahim-donor/equal-Haist-patient feature scaling, and the median prediction
across Haist patients. No survival outcome enters this calculation.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score


HERE = Path(__file__).resolve().parent
FIGURE_HNSCC = HERE.parents[1]
INPUT = HERE / "input"
ALIGNMENT = HERE / "alignment"
OUT = HERE / "marker_rerun/projection"
HAIST_CELLS = (
    FIGURE_HNSCC
    / "sampled_data/haist_codex/complete/HNSCCdiscovery_CODEX.csv"
)
EXPOSURE = (
    FIGURE_HNSCC
    / "survival_motivation/output/pt_cd8_cell_spatial_niche_scores.csv.gz"
)
OLD_PROJECTION = (
    FIGURE_HNSCC.parent
    / "test0827/hnscc/pt_survival_first/reviewer_robustness/canonical_ramp/"
      "equal_rahim_donor_scaling/output/rahim_tumor_cd8_projected_scores.csv.gz"
)

K_TRANSFER = 30
MARKERS = [
    "CD45", "CD3", "CD7", "CD4", "CD8", "CD25", "CD20", "CD38",
    "CD56", "CD57", "CD16", "CD11c", "CD11b", "HLA-DR", "CD21",
    "CD31", "CD69", "ICOS", "PD1", "CD45RO", "CD45RA", "PD-L1",
]


def write_csv_gz(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", fileobj=raw_handle, mode="wb", mtime=0) as handle:
            frame.to_csv(handle, index=False)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def percentile_midrank(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    answer = np.full(len(values), np.nan, float)
    for group in pd.unique(groups):
        rows = np.flatnonzero(groups == group)
        answer[rows] = (rankdata(values[rows], method="average") - 0.5) / len(rows)
    return answer


def monotone_curve(raw: np.ndarray, aligned: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(raw, kind="stable")
    frame = pd.DataFrame({"x": raw[order], "y": aligned[order]})
    frame = frame.groupby("x", as_index=False).y.mean()
    return frame.x.to_numpy(float), np.maximum.accumulate(frame.y.to_numpy(float))


def build_haist_curves(
    raw: np.ndarray, aligned: np.ndarray, metadata: pd.DataFrame
) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    target_rows = np.flatnonzero(metadata.dataset.eq("Haist_CODEX").to_numpy())
    local = metadata.iloc[target_rows].reset_index(drop=True)
    target_raw = np.asarray(raw[target_rows], float)
    target_aligned = np.asarray(aligned[target_rows], float)
    curves: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for patient, group_rows in local.groupby("donor", sort=True).groups.items():
        rows = np.asarray(list(group_rows), int)
        for marker_index, marker in enumerate(MARKERS):
            curves[(str(patient), marker)] = monotone_curve(
                target_raw[rows, marker_index], target_aligned[rows, marker_index]
            )
    if len(curves) != 70 * len(MARKERS):
        raise AssertionError(f"Expected {70 * len(MARKERS)} patient-marker curves")
    return curves


def load_full_haist_pt_cd8(
    selected: set[str],
    curves: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
) -> tuple[pd.DataFrame, np.ndarray]:
    exposure = pd.read_csv(
        EXPOSURE,
        usecols=["UID", "Pat_ID", "roi_adjusted_caf_macrophage_score"],
        dtype={"UID": str, "Pat_ID": str},
    )
    usecols = ["UID", "Pat_ID", "Tissue_type_broad", "cell_types_mediumres"] + MARKERS
    blocks = []
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
        raise AssertionError("A selected Haist PT CD8 cell lacks its spatial score")
    counts = cells.groupby("Pat_ID").size()
    if len(counts) != 70 or (counts < 40).any() or len(cells) != 38_194:
        raise AssertionError(
            f"Expected 38,194 PT CD8 cells from 70 patients; observed {len(cells)}"
        )

    transformed = np.empty((len(cells), len(MARKERS)), np.float64)
    for patient, group_rows in cells.groupby("Pat_ID", sort=True).groups.items():
        rows = np.asarray(list(group_rows), int)
        patient_raw = cells.loc[rows, MARKERS].to_numpy(float)
        for marker_index, marker in enumerate(MARKERS):
            x_curve, y_curve = curves[(str(patient), marker)]
            transformed[rows, marker_index] = np.interp(
                patient_raw[:, marker_index], x_curve, y_curve
            )
    return cells, transformed


def equal_unit_moments(values: np.ndarray, units: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = []
    seconds = []
    for unit in sorted(pd.unique(units), key=str):
        block = values[units == unit]
        means.append(block.mean(axis=0))
        seconds.append(np.mean(block * block, axis=0))
    return np.mean(means, axis=0), np.mean(seconds, axis=0)


def equal_donor_patient_scaler(
    rahim_x: np.ndarray,
    rahim_donors: np.ndarray,
    haist_x: np.ndarray,
    haist_patients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rahim_mean, rahim_second = equal_unit_moments(rahim_x, rahim_donors)
    haist_mean, haist_second = equal_unit_moments(haist_x, haist_patients)
    mean = 0.5 * rahim_mean + 0.5 * haist_mean
    second = 0.5 * rahim_second + 0.5 * haist_second
    return mean, np.sqrt(np.maximum(second - mean * mean, 1e-12))


def weighted_predict(tree: cKDTree, response: np.ndarray, query: np.ndarray) -> np.ndarray:
    distance, index = tree.query(query, k=min(K_TRANSFER, len(response)), workers=1)
    if distance.ndim == 1:
        distance, index = distance[:, None], index[:, None]
    local_scale = np.maximum(np.median(distance, axis=1, keepdims=True), 1e-6)
    weight = np.exp(-distance / local_scale)
    return np.sum(weight * response[index], axis=1) / np.maximum(weight.sum(axis=1), 1e-12)


def build_patient_predictors(
    cells: pd.DataFrame, values: np.ndarray
) -> dict[str, tuple[cKDTree, np.ndarray]]:
    predictors = {}
    patients = cells.Pat_ID.to_numpy(str)
    response = cells.roi_adjusted_caf_macrophage_score.to_numpy(float)
    for patient in sorted(pd.unique(patients), key=int):
        rows = np.flatnonzero(patients == patient)
        predictors[patient] = (cKDTree(values[rows]), response[rows])
    return predictors


def aggregate_predictions(
    predictors: dict[str, tuple[cKDTree, np.ndarray]],
    query: np.ndarray,
    exclude_patient: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    columns = [
        weighted_predict(tree, response, query)
        for patient, (tree, response) in predictors.items()
        if patient != exclude_patient
    ]
    matrix = np.column_stack(columns)
    return (
        np.median(matrix, axis=1),
        np.quantile(matrix, 0.25, axis=1),
        np.quantile(matrix, 0.75, axis=1),
    )


def cross_validate(
    cells: pd.DataFrame,
    values: np.ndarray,
    predictors: dict[str, tuple[cKDTree, np.ndarray]],
) -> pd.DataFrame:
    rows = []
    patients = cells.Pat_ID.to_numpy(str)
    truth_all = cells.roi_adjusted_caf_macrophage_score.to_numpy(float)
    for patient in sorted(predictors, key=int):
        test = np.flatnonzero(patients == patient)
        prediction, _, _ = aggregate_predictions(predictors, values[test], patient)
        truth = truth_all[test]
        truth_rank = (rankdata(truth, method="average") - 0.5) / len(truth)
        high = truth_rank >= 0.75
        rows.append(
            {
                "held_out_patient": patient,
                "cells": int(len(test)),
                "spearman": float(spearmanr(truth, prediction).statistic),
                "top_quartile_auroc": float(roc_auc_score(high, prediction)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = np.load(INPUT / "raw_shared_markers.float64.npy", mmap_mode="r")
    aligned = np.load(ALIGNMENT / "aligned_ramp_ver2.float32.npy", mmap_mode="r")
    metadata = pd.read_csv(
        INPUT / "cell_metadata.csv.gz",
        dtype={"cell_uid": str, "dataset": str, "donor": str, "unit_key": str},
        low_memory=False,
    )
    selected = set(pd.read_csv(INPUT / "selected_70_patients.csv").CaseID.astype(str))
    curves = build_haist_curves(raw, aligned, metadata)

    rahim_rows = np.flatnonzero(
        metadata.dataset.eq("GSE212797_ADT")
        & metadata.tissue.astype(str).eq("tumor")
        & metadata.provider_cell_type.astype(str).eq("CD8 T cell")
    )
    query = metadata.iloc[rahim_rows].copy().reset_index(drop=True)
    query["cell_uid"] = query.cell_uid.str.removeprefix("RAHIM|")
    query_aligned = np.asarray(aligned[rahim_rows], np.float64)
    if len(query) != 4_788:
        raise AssertionError(f"Expected 4,788 Rahim tumor CD8 cells; observed {len(query)}")

    haist, haist_aligned = load_full_haist_pt_cd8(selected, curves)
    query_donors = query.donor.to_numpy(str)
    haist_patients = haist.Pat_ID.to_numpy(str)
    mean, scale = equal_donor_patient_scaler(
        query_aligned, query_donors, haist_aligned, haist_patients
    )
    query_z = (query_aligned - mean) / scale
    haist_z = (haist_aligned - mean) / scale
    predictors = build_patient_predictors(haist, haist_z)
    prediction, q25, q75 = aggregate_predictions(predictors, query_z)
    cv = cross_validate(haist, haist_z, predictors)

    projected = query[["cell_uid", "donor", "tissue", "provider_cell_type"]].copy()
    projected["projected_roi_adjusted_exposure"] = prediction
    projected["patient_predictor_q25"] = q25
    projected["patient_predictor_q75"] = q75
    projected["projected_rank_within_rahim_donor"] = percentile_midrank(
        prediction, query_donors
    )
    projection_path = OUT / "rahim_tumor_cd8_projected_scores.csv.gz"
    write_csv_gz(projected, projection_path)
    cv.to_csv(OUT / "haist70_leave_one_patient_out.csv", index=False)

    comparison = None
    if OLD_PROJECTION.exists():
        old = pd.read_csv(OLD_PROJECTION, dtype={"cell_uid": str, "donor": str})
        paired = projected.merge(
            old[["cell_uid", "donor", "projected_roi_adjusted_exposure"]],
            on=["cell_uid", "donor"],
            validate="one_to_one",
            suffixes=("_pt70", "_old40"),
        )
        comparison_rows = []
        for scope, block in [("all", paired)] + list(paired.groupby("donor", sort=True)):
            comparison_rows.append(
                {
                    "scope": str(scope),
                    "cells": int(len(block)),
                    "spearman_pt70_vs_old40": float(
                        spearmanr(
                            block.projected_roi_adjusted_exposure_pt70,
                            block.projected_roi_adjusted_exposure_old40,
                        ).statistic
                    ),
                }
            )
        comparison = pd.DataFrame(comparison_rows)
        comparison.to_csv(OUT / "projection_pt70_vs_old40.csv", index=False)

    manifest = {
        "purpose": "PT70 Haist spatial-score transfer to Rahim tumor CD8",
        "survival_outcome_used": False,
        "alignment": "canonical RAMP 2.3.0 PT70 test alignment",
        "Haist_full_PT_CD8_cells": int(len(haist)),
        "Haist_patients": int(haist.Pat_ID.nunique()),
        "Rahim_tumor_CD8_cells": int(len(query)),
        "Rahim_donors": int(query.donor.nunique()),
        "response": "cell-level ROI-adjusted k15 CAF/macrophage exposure",
        "scaler": "50% equal-Rahim-donor moments + 50% equal-Haist-patient moments",
        "predictor": "one distance-weighted k30 predictor per Haist patient; median across 70",
        "projection_sha256": sha256(projection_path),
        "LOPO_median_spearman": float(cv.spearman.median()),
        "LOPO_positive_patients": int((cv.spearman > 0).sum()),
        "LOPO_median_AUROC": float(cv.top_quartile_auroc.median()),
        "old40_projection_comparison": (
            comparison.set_index("scope").spearman_pt70_vs_old40.to_dict()
            if comparison is not None else None
        ),
    }
    (OUT / "projection_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
