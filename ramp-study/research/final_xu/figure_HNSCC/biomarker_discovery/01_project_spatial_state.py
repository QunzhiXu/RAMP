#!/usr/bin/env python3
"""Project the formal PT74 spatial CD8 state into Rahim tumour CD8 cells.

This is the outcome-blind spatial-to-single-cell transfer used by the formal
biomarker-discovery workflow.  It is a self-contained implementation of the
validated transfer design:

1. learn one monotone raw-to-RAMP curve for every Haist patient and marker
   from the formal 1,000-cell-per-patient alignment input;
2. use those curves to transform all 38,299 provider-labelled Haist PT-CD8
   cells carrying the formal ROI-adjusted CAF/macrophage spatial response;
3. scale the 22 features with 50% equal-Rahim-donor moments and 50% equal-
   Haist-patient moments;
4. fit one distance-weighted k=30 predictor per Haist patient and take the
   median of all 74 predictions for each of 4,788 Rahim tumour CD8 cells.

No clinical or survival outcome is read.  The leave-one-patient-out analysis
is a diagnostic of the patient-specific kNN transfer: as in the validated
design, its RAMP curves and common feature scaler remain fixed rather than
being refit inside each held-out fold.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score


HERE = Path(__file__).resolve().parent
FIGURE_HNSCC = HERE.parent
ALIGNMENT_INPUT = FIGURE_HNSCC / "alignment" / "input"
ALIGNMENT_OUTPUT = FIGURE_HNSCC / "alignment" / "output"
SPATIAL_RESPONSE = (
    FIGURE_HNSCC
    / "survival_motivation"
    / "output"
    / "pt_cd8_cell_spatial_niche_scores.csv.gz"
)
HAIST_CELL_TABLE = (
    FIGURE_HNSCC
    / "sampled_data"
    / "haist_codex"
    / "complete"
    / "HNSCCdiscovery_CODEX.csv"
)
OUT = HERE / "projection"

EXPECTED_ALIGNMENT_CELLS = 94_488
EXPECTED_HAIST_PATIENTS = 74
EXPECTED_HAIST_ALIGNMENT_CELLS = 74_000
EXPECTED_FULL_HAIST_PT_CD8 = 38_299
EXPECTED_RAHIM_CELLS = 20_488
EXPECTED_RAHIM_TUMOUR_CD8 = 4_788
EXPECTED_RAHIM_DONORS = 3
K_TRANSFER = 30
CHUNK_SIZE = 150_000

EXPECTED_MARKERS = [
    "CD45",
    "CD3",
    "CD7",
    "CD4",
    "CD8",
    "CD25",
    "CD20",
    "CD38",
    "CD56",
    "CD57",
    "CD16",
    "CD11c",
    "CD11b",
    "HLA-DR",
    "CD21",
    "CD31",
    "CD69",
    "ICOS",
    "PD1",
    "CD45RO",
    "CD45RA",
    "PD-L1",
]


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv_gz(frame: pd.DataFrame, path: Path) -> None:
    """Write a deterministic gzip-compressed CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", fileobj=raw_handle, mode="wb", mtime=0) as handle:
            frame.to_csv(handle, index=False)


def percentile_midrank(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Calculate percentile midranks independently inside each group."""
    answer = np.full(len(values), np.nan, dtype=float)
    for group in pd.unique(groups):
        rows = np.flatnonzero(groups == group)
        answer[rows] = (rankdata(values[rows], method="average") - 0.5) / len(rows)
    return answer


def monotone_curve(raw: np.ndarray, aligned: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build the empirical monotone raw-to-aligned interpolation curve."""
    raw = np.asarray(raw, dtype=float)
    aligned = np.asarray(aligned, dtype=float)
    if raw.ndim != 1 or aligned.ndim != 1 or len(raw) != len(aligned):
        raise ValueError("Curve inputs must be same-length one-dimensional arrays")
    if len(raw) == 0 or not np.isfinite(raw).all() or not np.isfinite(aligned).all():
        raise ValueError("Curve inputs are empty or non-finite")
    order = np.argsort(raw, kind="stable")
    frame = pd.DataFrame({"x": raw[order], "y": aligned[order]})
    frame = frame.groupby("x", as_index=False, sort=True).y.mean()
    x = frame.x.to_numpy(dtype=float)
    y = np.maximum.accumulate(frame.y.to_numpy(dtype=float))
    if np.any(np.diff(x) <= 0) or np.any(np.diff(y) < 0):
        raise AssertionError("A raw-to-aligned curve is not monotone")
    return x, y


def read_formal_alignment() -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    """Read and validate the formal PT74 alignment artifacts."""
    markers = [
        line.strip()
        for line in (ALIGNMENT_INPUT / "markers.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if markers != EXPECTED_MARKERS:
        raise AssertionError(f"Formal shared-marker order changed: {markers}")

    raw = np.load(
        ALIGNMENT_INPUT / "raw_shared_markers.float64.npy", mmap_mode="r"
    )
    aligned = np.load(
        ALIGNMENT_OUTPUT / "aligned_ramp_ver2.float32.npy", mmap_mode="r"
    )
    metadata = pd.read_csv(
        ALIGNMENT_INPUT / "cell_metadata.csv.gz",
        dtype={
            "cell_uid": str,
            "dataset": str,
            "donor": str,
            "unit_key": str,
            "tissue": str,
            "provider_cell_type": str,
        },
        low_memory=False,
    )
    expected_shape = (EXPECTED_ALIGNMENT_CELLS, len(markers))
    if raw.shape != expected_shape or aligned.shape != expected_shape:
        raise AssertionError(
            f"Expected raw/aligned shape {expected_shape}; observed {raw.shape}/{aligned.shape}"
        )
    if len(metadata) != EXPECTED_ALIGNMENT_CELLS or not metadata.cell_uid.is_unique:
        raise AssertionError("Formal alignment metadata has changed size or duplicated cell IDs")
    dataset_counts = metadata.groupby("dataset", sort=True).size().to_dict()
    if dataset_counts != {
        "GSE212797_ADT": EXPECTED_RAHIM_CELLS,
        "Haist_CODEX": EXPECTED_HAIST_ALIGNMENT_CELLS,
    }:
        raise AssertionError(f"Unexpected formal alignment dataset counts: {dataset_counts}")
    if not np.isfinite(raw).all() or not np.isfinite(aligned).all():
        raise AssertionError("Formal alignment matrices contain non-finite values")
    return raw, aligned, metadata, markers


def read_selected_patients() -> pd.DataFrame:
    """Read the outcome-free patient identifier/count table from alignment input."""
    selected = pd.read_csv(
        ALIGNMENT_INPUT / "selected_74_patients.csv",
        usecols=["CaseID", "pt_cd8_cells"],
        dtype={"CaseID": str},
    )
    selected["pt_cd8_cells"] = pd.to_numeric(
        selected.pt_cd8_cells, errors="raise"
    ).astype(int)
    if len(selected) != EXPECTED_HAIST_PATIENTS or selected.CaseID.nunique() != len(selected):
        raise AssertionError("Formal selected-patient list is not 74 unique patients")
    if int(selected.pt_cd8_cells.sum()) != EXPECTED_FULL_HAIST_PT_CD8:
        raise AssertionError("Formal selected-patient PT-CD8 total changed")
    if int(selected.pt_cd8_cells.min()) != 3:
        raise AssertionError("The unfiltered PT74 cohort should include the 3-cell patient")
    return selected.sort_values("CaseID", key=lambda s: s.astype(int)).reset_index(drop=True)


def build_haist_curves(
    raw: np.ndarray,
    aligned: np.ndarray,
    metadata: pd.DataFrame,
    markers: list[str],
) -> tuple[
    dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    pd.DataFrame,
]:
    """Build one empirical RAMP mapping per formal Haist patient and marker."""
    target_rows = np.flatnonzero(metadata.dataset.eq("Haist_CODEX").to_numpy())
    local = metadata.iloc[target_rows].reset_index(drop=True)
    target_raw = np.asarray(raw[target_rows], dtype=np.float64)
    target_aligned = np.asarray(aligned[target_rows], dtype=np.float64)
    patient_counts = local.groupby("donor", sort=True).size()
    if len(patient_counts) != EXPECTED_HAIST_PATIENTS or not patient_counts.eq(1_000).all():
        raise AssertionError("Expected 1,000 formal alignment cells for each of 74 Haist patients")

    curves: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    audit_rows: list[dict[str, object]] = []
    for patient, group_rows in local.groupby("donor", sort=True).groups.items():
        rows = np.asarray(list(group_rows), dtype=int)
        for marker_index, marker in enumerate(markers):
            raw_values = target_raw[rows, marker_index]
            aligned_values = target_aligned[rows, marker_index]
            x_curve, y_curve = monotone_curve(raw_values, aligned_values)
            patient_key = str(patient)
            curves[(patient_key, marker)] = (x_curve, y_curve)
            audit_rows.append(
                {
                    "patient": patient_key,
                    "marker": marker,
                    "alignment_sample_cells": int(len(rows)),
                    "raw_unique_knots": int(len(x_curve)),
                    "raw_min": float(x_curve[0]),
                    "raw_max": float(x_curve[-1]),
                    "mapped_min": float(y_curve[0]),
                    "mapped_max": float(y_curve[-1]),
                    "mapped_monotone_non_decreasing": bool(np.all(np.diff(y_curve) >= 0)),
                }
            )
    expected_curves = EXPECTED_HAIST_PATIENTS * len(markers)
    if len(curves) != expected_curves:
        raise AssertionError(f"Expected {expected_curves} patient-marker curves")
    return curves, pd.DataFrame(audit_rows)


def load_full_haist_pt_cd8(
    selected: pd.DataFrame,
    curves: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    markers: list[str],
    curve_audit: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Read all selected PT-CD8 cells, join response, and apply RAMP curves."""
    selected_ids = set(selected.CaseID.astype(str))
    spatial = pd.read_csv(
        SPATIAL_RESPONSE,
        usecols=["UID", "Pat_ID", "roi_adjusted_caf_macrophage_score"],
        dtype={"UID": str, "Pat_ID": str},
    )
    if spatial[["UID", "Pat_ID"]].duplicated().any():
        raise AssertionError("Formal spatial-response file contains duplicate cell keys")
    spatial = spatial.loc[spatial.Pat_ID.isin(selected_ids)].copy()
    if len(spatial) != EXPECTED_FULL_HAIST_PT_CD8:
        raise AssertionError(
            "The formal spatial-response file does not contain exactly 38,299 selected cells"
        )
    if not np.isfinite(spatial.roi_adjusted_caf_macrophage_score.to_numpy(float)).all():
        raise AssertionError("Formal spatial response contains non-finite values")

    usecols = ["UID", "Pat_ID", "Tissue_type_broad", "cell_types_mediumres"] + markers
    blocks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(HAIST_CELL_TABLE, usecols=usecols, chunksize=CHUNK_SIZE):
        patients = chunk.Pat_ID.astype(str)
        keep = (
            patients.isin(selected_ids)
            & chunk.Tissue_type_broad.astype(str).eq("PT")
            & chunk.cell_types_mediumres.astype(str).eq("CD8T")
        )
        if keep.any():
            block = chunk.loc[keep].copy()
            block["Pat_ID"] = patients.loc[keep].to_numpy(str)
            blocks.append(block)
    if not blocks:
        raise RuntimeError("No selected Haist PT-CD8 cells were found")
    cells = pd.concat(blocks, ignore_index=True)
    cells["UID"] = cells.UID.astype(str)
    if cells[["UID", "Pat_ID"]].duplicated().any():
        raise AssertionError("The full Haist PT-CD8 source contains duplicated cell keys")
    cells = cells.merge(spatial, on=["UID", "Pat_ID"], validate="one_to_one")

    observed_counts = cells.groupby("Pat_ID", sort=True).size().rename("observed")
    expected_counts = selected.set_index("CaseID").pt_cd8_cells.rename("expected")
    count_check = expected_counts.to_frame().join(observed_counts, how="outer")
    if count_check.isna().any().any() or not count_check.expected.equals(count_check.observed):
        raise AssertionError(f"Full PT-CD8 counts disagree with formal selection:\n{count_check}")
    if len(cells) != EXPECTED_FULL_HAIST_PT_CD8:
        raise AssertionError(f"Expected 38,299 full PT-CD8 cells; observed {len(cells)}")

    transformed = np.empty((len(cells), len(markers)), dtype=np.float64)
    full_audit_rows: list[dict[str, object]] = []
    for patient, group_rows in cells.groupby("Pat_ID", sort=True).groups.items():
        rows = np.asarray(list(group_rows), dtype=int)
        patient_raw = cells.loc[rows, markers].to_numpy(dtype=float)
        if not np.isfinite(patient_raw).all():
            raise AssertionError(f"Patient {patient} has non-finite shared-marker values")
        for marker_index, marker in enumerate(markers):
            x_curve, y_curve = curves[(str(patient), marker)]
            values = patient_raw[:, marker_index]
            transformed[rows, marker_index] = np.interp(values, x_curve, y_curve)
            full_audit_rows.append(
                {
                    "patient": str(patient),
                    "marker": marker,
                    "full_pt_cd8_cells": int(len(rows)),
                    "full_raw_below_curve": int(np.sum(values < x_curve[0])),
                    "full_raw_above_curve": int(np.sum(values > x_curve[-1])),
                    "full_raw_below_curve_fraction": float(np.mean(values < x_curve[0])),
                    "full_raw_above_curve_fraction": float(np.mean(values > x_curve[-1])),
                }
            )
    if not np.isfinite(transformed).all():
        raise AssertionError("Transformed full Haist PT-CD8 matrix is non-finite")
    curve_audit = curve_audit.merge(
        pd.DataFrame(full_audit_rows),
        on=["patient", "marker"],
        how="left",
        validate="one_to_one",
    )
    return cells, transformed, curve_audit


def equal_unit_moments(
    values: np.ndarray, units: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Mean first and second moments with equal rather than cell unit weight."""
    means = []
    seconds = []
    for unit in sorted(pd.unique(units), key=str):
        block = values[units == unit]
        if len(block) == 0:
            raise AssertionError(f"Empty scaling unit: {unit}")
        means.append(block.mean(axis=0))
        seconds.append(np.mean(block * block, axis=0))
    return np.mean(means, axis=0), np.mean(seconds, axis=0)


def equal_donor_patient_scaler(
    rahim_x: np.ndarray,
    rahim_donors: np.ndarray,
    haist_x: np.ndarray,
    haist_patients: np.ndarray,
    markers: list[str],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Fit the frozen 50:50 equal-donor/equal-patient feature scaler."""
    rahim_mean, rahim_second = equal_unit_moments(rahim_x, rahim_donors)
    haist_mean, haist_second = equal_unit_moments(haist_x, haist_patients)
    mean = 0.5 * rahim_mean + 0.5 * haist_mean
    second = 0.5 * rahim_second + 0.5 * haist_second
    scale = np.sqrt(np.maximum(second - mean * mean, 1e-12))
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise AssertionError("The equal-donor/equal-patient scaler is invalid")
    audit = pd.DataFrame(
        {
            "marker": markers,
            "rahim_equal_donor_mean": rahim_mean,
            "rahim_equal_donor_second_moment": rahim_second,
            "haist_equal_patient_mean": haist_mean,
            "haist_equal_patient_second_moment": haist_second,
            "combined_equal_domain_mean": mean,
            "combined_equal_domain_scale": scale,
        }
    )
    return mean, scale, audit


def weighted_predict(
    tree: cKDTree, response: np.ndarray, query: np.ndarray
) -> np.ndarray:
    """Distance-weighted kNN prediction with the validated local bandwidth."""
    distance, index = tree.query(
        query, k=min(K_TRANSFER, len(response)), workers=1
    )
    if distance.ndim == 1:
        distance = distance[:, None]
        index = index[:, None]
    local_scale = np.maximum(np.median(distance, axis=1, keepdims=True), 1e-6)
    weight = np.exp(-distance / local_scale)
    prediction = np.sum(weight * response[index], axis=1) / np.maximum(
        weight.sum(axis=1), 1e-12
    )
    if not np.isfinite(prediction).all():
        raise AssertionError("A patient-specific weighted prediction is non-finite")
    return prediction


def build_patient_predictors(
    cells: pd.DataFrame, values: np.ndarray
) -> dict[str, tuple[cKDTree, np.ndarray]]:
    """Fit one source predictor for every Haist patient."""
    predictors: dict[str, tuple[cKDTree, np.ndarray]] = {}
    patients = cells.Pat_ID.to_numpy(str)
    response = cells.roi_adjusted_caf_macrophage_score.to_numpy(float)
    for patient in sorted(pd.unique(patients), key=int):
        rows = np.flatnonzero(patients == patient)
        predictors[str(patient)] = (cKDTree(values[rows]), response[rows])
    if len(predictors) != EXPECTED_HAIST_PATIENTS:
        raise AssertionError("Expected 74 patient-specific Haist predictors")
    return predictors


def prediction_matrix(
    predictors: dict[str, tuple[cKDTree, np.ndarray]], query: np.ndarray
) -> tuple[list[str], np.ndarray]:
    """Evaluate every patient-specific predictor on the same query matrix."""
    patients = list(predictors)
    columns = [
        weighted_predict(tree, response, query)
        for tree, response in predictors.values()
    ]
    matrix = np.column_stack(columns)
    if matrix.shape != (len(query), len(patients)):
        raise AssertionError("Unexpected patient-specific prediction matrix shape")
    return patients, matrix


def cross_validate_from_matrix(
    cells: pd.DataFrame,
    predictor_patients: list[str],
    patient_prediction: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Calculate the fixed-map/fixed-scaler LOPO transfer diagnostic."""
    cell_patients = cells.Pat_ID.to_numpy(str)
    truth_all = cells.roi_adjusted_caf_macrophage_score.to_numpy(float)
    patient_column = {patient: index for index, patient in enumerate(predictor_patients)}
    lopo_prediction = np.full(len(cells), np.nan, dtype=float)
    rows: list[dict[str, object]] = []
    for patient in predictor_patients:
        test = np.flatnonzero(cell_patients == patient)
        keep_columns = np.arange(len(predictor_patients)) != patient_column[patient]
        prediction = np.median(patient_prediction[test][:, keep_columns], axis=1)
        lopo_prediction[test] = prediction
        truth = truth_all[test]
        truth_rank = (rankdata(truth, method="average") - 0.5) / len(truth)
        high = truth_rank >= 0.75
        if len(np.unique(high)) != 2:
            raise AssertionError(f"Patient {patient} has no valid top-quartile AUROC split")
        correlation = float(spearmanr(truth, prediction).statistic)
        auroc = float(roc_auc_score(high, prediction))
        rows.append(
            {
                "held_out_patient": patient,
                "cells": int(len(test)),
                "source_predictors_used": int(keep_columns.sum()),
                "spearman": correlation,
                "top_quartile_auroc": auroc,
                "truth_mean": float(np.mean(truth)),
                "truth_sd": float(np.std(truth, ddof=1)) if len(truth) > 1 else 0.0,
                "prediction_mean": float(np.mean(prediction)),
                "prediction_sd": (
                    float(np.std(prediction, ddof=1)) if len(prediction) > 1 else 0.0
                ),
            }
        )
    if not np.isfinite(lopo_prediction).all():
        raise AssertionError("LOPO prediction vector is incomplete or non-finite")
    return pd.DataFrame(rows), lopo_prediction


def build_predictor_audit(
    predictors: dict[str, tuple[cKDTree, np.ndarray]],
    rahim_predictions: np.ndarray,
    predictor_patients: list[str],
) -> pd.DataFrame:
    """Summarize each Haist source predictor and its Rahim predictions."""
    rows: list[dict[str, object]] = []
    for column, patient in enumerate(predictor_patients):
        _, response = predictors[patient]
        prediction = rahim_predictions[:, column]
        rows.append(
            {
                "haist_patient": patient,
                "source_pt_cd8_cells": int(len(response)),
                "effective_k": int(min(K_TRANSFER, len(response))),
                "source_response_mean": float(np.mean(response)),
                "source_response_sd": (
                    float(np.std(response, ddof=1)) if len(response) > 1 else 0.0
                ),
                "rahim_prediction_mean": float(np.mean(prediction)),
                "rahim_prediction_sd": float(np.std(prediction, ddof=1)),
                "rahim_prediction_min": float(np.min(prediction)),
                "rahim_prediction_median": float(np.median(prediction)),
                "rahim_prediction_max": float(np.max(prediction)),
            }
        )
    return pd.DataFrame(rows)


def build_donor_audit(projected: pd.DataFrame) -> pd.DataFrame:
    """Summarize the final projected state within each Rahim donor."""
    rows: list[dict[str, object]] = []
    for donor, frame in projected.groupby("donor", sort=True):
        values = frame.projected_roi_adjusted_exposure.to_numpy(float)
        rows.append(
            {
                "donor": str(donor),
                "cells": int(len(frame)),
                "projection_mean": float(np.mean(values)),
                "projection_sd": float(np.std(values, ddof=1)),
                "projection_min": float(np.min(values)),
                "projection_q25": float(np.quantile(values, 0.25)),
                "projection_median": float(np.median(values)),
                "projection_q75": float(np.quantile(values, 0.75)),
                "projection_max": float(np.max(values)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    raw, aligned, metadata, markers = read_formal_alignment()
    selected = read_selected_patients()
    curves, curve_audit = build_haist_curves(raw, aligned, metadata, markers)

    rahim_rows = np.flatnonzero(
        metadata.dataset.eq("GSE212797_ADT").to_numpy()
        & metadata.tissue.eq("tumor").to_numpy()
        & metadata.provider_cell_type.eq("CD8 T cell").to_numpy()
    )
    query = metadata.iloc[rahim_rows].copy().reset_index(drop=True)
    query["cell_uid"] = query.cell_uid.str.removeprefix("RAHIM|")
    query_aligned = np.asarray(aligned[rahim_rows], dtype=np.float64)
    if (
        len(query) != EXPECTED_RAHIM_TUMOUR_CD8
        or query.donor.nunique() != EXPECTED_RAHIM_DONORS
        or not query.cell_uid.is_unique
    ):
        raise AssertionError("Expected 4,788 unique Rahim tumour-CD8 cells from 3 donors")

    haist, haist_aligned, curve_audit = load_full_haist_pt_cd8(
        selected, curves, markers, curve_audit
    )
    query_donors = query.donor.to_numpy(str)
    haist_patients = haist.Pat_ID.to_numpy(str)
    mean, scale, scaling_audit = equal_donor_patient_scaler(
        query_aligned,
        query_donors,
        haist_aligned,
        haist_patients,
        markers,
    )
    query_z = (query_aligned - mean) / scale
    haist_z = (haist_aligned - mean) / scale

    predictors = build_patient_predictors(haist, haist_z)
    predictor_patients, rahim_prediction_matrix = prediction_matrix(predictors, query_z)
    prediction = np.median(rahim_prediction_matrix, axis=1)
    q25 = np.quantile(rahim_prediction_matrix, 0.25, axis=1)
    q75 = np.quantile(rahim_prediction_matrix, 0.75, axis=1)

    projected = query[["cell_uid", "donor", "tissue", "provider_cell_type"]].copy()
    projected["projected_roi_adjusted_exposure"] = prediction
    projected["patient_predictor_q25"] = q25
    projected["patient_predictor_q75"] = q75
    projected["patient_predictor_iqr"] = q75 - q25
    projected["projected_rank_within_rahim_donor"] = percentile_midrank(
        prediction, query_donors
    )

    # Evaluate each Haist cell against all 74 patient-specific predictors once;
    # each fold then removes its own patient's column before taking the median.
    lopo_predictor_patients, haist_prediction_matrix = prediction_matrix(
        predictors, haist_z
    )
    if lopo_predictor_patients != predictor_patients:
        raise AssertionError("Patient predictor order changed between projection and LOPO")
    lopo, lopo_prediction = cross_validate_from_matrix(
        haist, predictor_patients, haist_prediction_matrix
    )

    predictor_audit = build_predictor_audit(
        predictors, rahim_prediction_matrix, predictor_patients
    )
    donor_audit = build_donor_audit(projected)
    patient_audit = (
        haist.assign(lopo_prediction=lopo_prediction)
        .groupby("Pat_ID", as_index=False, sort=True)
        .agg(
            full_pt_cd8_cells=("UID", "size"),
            response_mean=("roi_adjusted_caf_macrophage_score", "mean"),
            response_sd=("roi_adjusted_caf_macrophage_score", "std"),
            lopo_prediction_mean=("lopo_prediction", "mean"),
            lopo_prediction_sd=("lopo_prediction", "std"),
        )
        .rename(columns={"Pat_ID": "patient"})
    )
    patient_audit[["response_sd", "lopo_prediction_sd"]] = patient_audit[
        ["response_sd", "lopo_prediction_sd"]
    ].fillna(0.0)
    patient_audit = patient_audit.merge(
        lopo.rename(columns={"held_out_patient": "patient"}),
        on="patient",
        validate="one_to_one",
        suffixes=("", "_lopo"),
    )

    projection_path = OUT / "rahim_tumor_cd8_projected_scores.csv.gz"
    lopo_path = OUT / "haist74_leave_one_patient_out.csv"
    scaling_path = OUT / "feature_scaling_audit.csv"
    curve_path = OUT / "patient_marker_curve_audit.csv.gz"
    predictor_path = OUT / "haist74_predictor_audit.csv"
    donor_path = OUT / "rahim_donor_projection_audit.csv"
    patient_path = OUT / "haist74_patient_projection_audit.csv"
    write_csv_gz(projected, projection_path)
    lopo.to_csv(lopo_path, index=False)
    scaling_audit.to_csv(scaling_path, index=False)
    write_csv_gz(curve_audit, curve_path)
    predictor_audit.to_csv(predictor_path, index=False)
    donor_audit.to_csv(donor_path, index=False)
    patient_audit.to_csv(patient_path, index=False)

    input_paths = {
        "raw_shared_markers": ALIGNMENT_INPUT / "raw_shared_markers.float64.npy",
        "alignment_metadata": ALIGNMENT_INPUT / "cell_metadata.csv.gz",
        "selected_patients": ALIGNMENT_INPUT / "selected_74_patients.csv",
        "markers": ALIGNMENT_INPUT / "markers.txt",
        "aligned_ramp": ALIGNMENT_OUTPUT / "aligned_ramp_ver2.float32.npy",
        "spatial_response": SPATIAL_RESPONSE,
        "full_haist_cell_table": HAIST_CELL_TABLE,
    }
    output_paths = {
        "projection": projection_path,
        "lopo": lopo_path,
        "feature_scaling_audit": scaling_path,
        "patient_marker_curve_audit": curve_path,
        "patient_predictor_audit": predictor_path,
        "rahim_donor_audit": donor_path,
        "haist_patient_audit": patient_path,
    }
    manifest = {
        "stage": "outcome-blind formal PT74 spatial-state transfer",
        "purpose": "Haist PT-CD8 spatial-score transfer to Rahim tumour CD8",
        "algorithm": {
            "alignment": "formal canonical RAMP 2.3.0 PT74 alignment",
            "patient_marker_mapping": (
                "empirical monotone interpolation per Haist patient and shared marker; "
                "constant boundary extension outside sampled raw range"
            ),
            "shared_marker_dimensions": len(markers),
            "shared_markers": markers,
            "feature_scaling": (
                "50% equal-Rahim-donor first/second moments plus 50% "
                "equal-Haist-patient first/second moments"
            ),
            "predictor": (
                "one distance-weighted k30 predictor per Haist patient; "
                "exp(-distance/within-query median-neighbour-distance) weights"
            ),
            "aggregation": "cell-wise median across all 74 Haist-patient predictors",
            "lopo_diagnostic": (
                "leave one Haist predictor out; formal RAMP mappings and common "
                "feature scaler held fixed"
            ),
        },
        "response": "cell-level ROI-adjusted local-k15 CAF/macrophage exposure",
        "counts": {
            "alignment_cells": int(len(metadata)),
            "haist_alignment_cells": int(metadata.dataset.eq("Haist_CODEX").sum()),
            "haist_alignment_patients": int(
                metadata.loc[metadata.dataset.eq("Haist_CODEX"), "donor"].nunique()
            ),
            "haist_full_pt_cd8_cells": int(len(haist)),
            "haist_full_pt_cd8_patients": int(haist.Pat_ID.nunique()),
            "minimum_haist_pt_cd8_cells_per_patient": int(
                haist.groupby("Pat_ID").size().min()
            ),
            "rahim_tumour_cd8_cells": int(len(query)),
            "rahim_donors": int(query.donor.nunique()),
            "patient_marker_curves": int(len(curve_audit)),
            "patient_specific_predictors": int(len(predictors)),
        },
        "projection_summary": {
            "mean": float(np.mean(prediction)),
            "sd": float(np.std(prediction, ddof=1)),
            "min": float(np.min(prediction)),
            "median": float(np.median(prediction)),
            "max": float(np.max(prediction)),
            "median_patient_predictor_iqr": float(np.median(q75 - q25)),
        },
        "lopo_summary": {
            "patients": int(len(lopo)),
            "median_spearman": float(lopo.spearman.median()),
            "positive_spearman_patients": int((lopo.spearman > 0).sum()),
            "median_top_quartile_auroc": float(lopo.top_quartile_auroc.median()),
            "auroc_above_0_5_patients": int((lopo.top_quartile_auroc > 0.5).sum()),
        },
        "outcome_blinding": {
            "clinical_or_survival_outcome_columns_read": False,
            "survival_used_for_projection": False,
            "survival_used_for_lopo": False,
            "cell_type_labels_used_for_alignment_fit": False,
            "test_directory_dependency": False,
        },
        "inputs": {name: str(path) for name, path in input_paths.items()},
        "inputs_sha256": {name: sha256(path) for name, path in input_paths.items()},
        "outputs": {name: str(path) for name, path in output_paths.items()},
        "outputs_sha256": {name: sha256(path) for name, path in output_paths.items()},
    }
    manifest_path = OUT / "projection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
