#!/usr/bin/env python3
"""Build outcome-blind top-40% spatial-return features for the HNSCC cohort.

The only input is the formal cell-level return table in this directory's
``output`` folder.  High state is fixed as a returned TIGIT/CD39 within-patient
percentile midrank >= 0.60.  No survival outcome or test-directory artifact is
read here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
from scipy.stats import binomtest, linregress, wilcoxon


HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
CELL_PATH = OUT / "haist74_pt_cd8_tigit_cd39_return.csv.gz"
PATIENT_FEATURE_PATH = OUT / "haist74_tigit_cd39_top40_patient_features.csv"
POOLED_QUINTILE_PATH = OUT / "state_quintile_pooled_summary.csv"
PATIENT_QUINTILE_PATH = OUT / "state_quintile_patient_summary.csv"
MANIFEST_PATH = OUT / "TOP40_FEATURE_MANIFEST.json"

EXPECTED_PATIENTS = 74
EXPECTED_PT_CD8_CELLS = 38_299
HIGH_STATE_THRESHOLD = 0.60
QUINTILE_COUNT = 5
QUINTILE_LABELS = tuple(f"Q{i}" for i in range(1, QUINTILE_COUNT + 1))
REQUIRED_COLUMNS = (
    "cell_uid",
    "patient",
    "spatial_y",
    "in_niche",
    "predicted_score",
    "within_patient_rank",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a CSV atomically so an interrupted run cannot leave a partial file."""
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def write_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def coerce_bool(series: pd.Series, column: str) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.to_numpy(bool)
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    unknown = sorted(set(normalized) - set(mapping))
    if unknown:
        raise AssertionError(f"{column} contains non-Boolean values: {unknown[:5]}")
    return normalized.map(mapping).to_numpy(bool)


def load_and_validate_cells() -> pd.DataFrame:
    if not CELL_PATH.is_file():
        raise FileNotFoundError(
            f"Required formal cell-level return table is absent: {CELL_PATH}"
        )
    if any(part in {"test", "test0827"} for part in CELL_PATH.resolve().parts):
        raise AssertionError("Formal feature construction cannot read a test-directory input")
    cells = pd.read_csv(
        CELL_PATH,
        usecols=list(REQUIRED_COLUMNS),
        dtype={"cell_uid": str, "patient": str},
        low_memory=False,
    )
    if len(cells) != EXPECTED_PT_CD8_CELLS:
        raise AssertionError(
            f"Expected {EXPECTED_PT_CD8_CELLS:,} PT-CD8 cells; observed {len(cells):,}"
        )
    if cells["patient"].nunique() != EXPECTED_PATIENTS:
        raise AssertionError(
            f"Expected {EXPECTED_PATIENTS} patients; observed {cells['patient'].nunique()}"
        )
    if cells["cell_uid"].isna().any() or cells["cell_uid"].duplicated().any():
        raise AssertionError("cell_uid must be complete and unique")

    numeric_columns = (
        "spatial_y",
        "predicted_score",
        "within_patient_rank",
    )
    for column in numeric_columns:
        cells[column] = pd.to_numeric(cells[column], errors="raise")
        if not np.isfinite(cells[column].to_numpy(float)).all():
            raise AssertionError(f"{column} contains a missing or non-finite value")

    state_rank = cells["within_patient_rank"].to_numpy(float)
    if np.any(state_rank < 0.0) or np.any(state_rank >= 1.0):
        raise AssertionError(
            "within_patient_rank must be a percentile midrank in [0, 1)"
        )

    # Verify both pieces of upstream semantics used in this stage rather than
    # silently accepting a stale or differently constructed return table.
    groups = cells.groupby("patient", sort=False)["predicted_score"]
    recomputed_rank = (groups.rank(method="average") - 0.5) / groups.transform("size")
    maximum_rank_error = float(
        np.max(np.abs(recomputed_rank.to_numpy(float) - state_rank))
    )
    if maximum_rank_error > 1e-12:
        raise AssertionError(
            "The supplied state rank is not the within-patient percentile midrank "
            f"of predicted_score (maximum absolute error {maximum_rank_error:.3g})"
        )

    supplied_niche = coerce_bool(
        cells["in_niche"],
        "in_niche",
    )
    threshold_niche = cells["spatial_y"].to_numpy(float) > 0.0
    if not np.array_equal(supplied_niche, threshold_niche):
        mismatches = int(np.sum(supplied_niche != threshold_niche))
        raise AssertionError(
            "The niche flag must equal spatial_y > 0; "
            f"observed {mismatches} inconsistent cells"
        )

    cells["in_niche"] = threshold_niche
    cells["high_state_top40"] = state_rank >= HIGH_STATE_THRESHOLD
    quintile_index = np.floor(state_rank * QUINTILE_COUNT).astype(np.int8) + 1
    if np.any(quintile_index < 1) or np.any(quintile_index > QUINTILE_COUNT):
        raise AssertionError("Failed to assign every state rank to Q1-Q5")
    cells["state_quintile_index"] = quintile_index
    cells["state_quintile"] = pd.Categorical(
        [f"Q{value}" for value in quintile_index],
        categories=list(QUINTILE_LABELS),
        ordered=True,
    )
    cells.attrs["maximum_recomputed_rank_error"] = maximum_rank_error
    return cells


def sort_patients(frame: pd.DataFrame) -> pd.DataFrame:
    patient_number = pd.to_numeric(frame["patient"], errors="coerce")
    if patient_number.isna().any():
        return frame.sort_values("patient", kind="stable").reset_index(drop=True)
    return (
        frame.assign(_patient_number=patient_number)
        .sort_values("_patient_number", kind="stable")
        .drop(columns="_patient_number")
        .reset_index(drop=True)
    )


def build_patient_features(cells: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for patient, frame in cells.groupby("patient", sort=False):
        high = frame["high_state_top40"].to_numpy(bool)
        niche = frame["in_niche"].to_numpy(bool)
        spatial = frame["spatial_y"].to_numpy(float)
        if not high.any():
            raise AssertionError(f"Patient {patient} has no rank >= 0.60 PT-CD8 cell")
        if not niche.any():
            raise AssertionError(f"Patient {patient} has no in-niche PT-CD8 cell")
        overlap = high & niche
        rows.append(
            {
                "patient": str(patient),
                "pt_cd8_cells": int(len(frame)),
                "high_state_cells": int(high.sum()),
                "non_high_state_cells": int((~high).sum()),
                "niche_cells": int(niche.sum()),
                "outside_niche_cells": int((~niche).sum()),
                "high_state_niche_cells": int(overlap.sum()),
                "high_state_outside_niche_cells": int((high & ~niche).sum()),
                "high_state_fraction": float(high.mean()),
                "niche_fraction": float(niche.mean()),
                "high_state_niche_precision": float(overlap.sum() / high.sum()),
                "high_state_niche_recall": float(overlap.sum() / niche.sum()),
                # This is the locked primary survival score used downstream.
                "high_pair_mean_spatial_y": float(spatial[high].mean()),
                # This direct score uses every PT-CD8 cell, irrespective of state.
                "patient_spatial_niche_score": float(spatial.mean()),
            }
        )
    features = sort_patients(pd.DataFrame(rows))
    if len(features) != EXPECTED_PATIENTS:
        raise AssertionError("Patient feature table did not retain all 74 patients")
    if int(features["pt_cd8_cells"].sum()) != EXPECTED_PT_CD8_CELLS:
        raise AssertionError("Patient feature cell counts do not sum to 38,299")
    if not np.array_equal(
        features["pt_cd8_cells"].to_numpy(int),
        (
            features["high_state_cells"].to_numpy(int)
            + features["non_high_state_cells"].to_numpy(int)
        ),
    ):
        raise AssertionError("High/non-high patient counts are inconsistent")
    return features


def build_quintile_summaries(
    cells: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    patient_rows: list[dict[str, Any]] = []
    for (patient, quintile), frame in cells.groupby(
        ["patient", "state_quintile"], observed=True, sort=False
    ):
        niche = frame["in_niche"].to_numpy(bool)
        spatial = frame["spatial_y"].to_numpy(float)
        patient_rows.append(
            {
                "patient": str(patient),
                "state_quintile": str(quintile),
                "state_quintile_index": int(str(quintile).removeprefix("Q")),
                "cells": int(len(frame)),
                "niche_cells": int(niche.sum()),
                "outside_niche_cells": int((~niche).sum()),
                "niche_fraction": float(niche.mean()),
                "mean_spatial_y": float(spatial.mean()),
                "median_spatial_y": float(np.median(spatial)),
                "mean_state_rank": float(
                    frame["within_patient_rank"].mean()
                ),
                "mean_predicted_score": float(
                    frame["predicted_score"].mean()
                ),
            }
        )
    patient_summary = pd.DataFrame(patient_rows)
    patient_summary = sort_patients(patient_summary)
    patient_summary = patient_summary.sort_values(
        ["patient", "state_quintile_index"],
        key=lambda column: (
            pd.to_numeric(column, errors="coerce")
            if column.name == "patient"
            else column
        ),
        kind="stable",
    ).reset_index(drop=True)

    pooled_rows: list[dict[str, Any]] = []
    for quintile_index, quintile in enumerate(QUINTILE_LABELS, start=1):
        frame = cells.loc[cells["state_quintile"].eq(quintile)]
        if frame.empty:
            raise AssertionError(f"No cells were assigned to {quintile}")
        patient_frame = patient_summary.loc[
            patient_summary["state_quintile"].eq(quintile)
        ]
        niche = frame["in_niche"].to_numpy(bool)
        spatial = frame["spatial_y"].to_numpy(float)
        pooled_rows.append(
            {
                "state_quintile": quintile,
                "state_quintile_index": quintile_index,
                "rank_lower_inclusive": float((quintile_index - 1) / QUINTILE_COUNT),
                "rank_upper_exclusive": float(quintile_index / QUINTILE_COUNT),
                "cells": int(len(frame)),
                "patients": int(frame["patient"].nunique()),
                "niche_cells": int(niche.sum()),
                "niche_fraction": float(niche.mean()),
                "mean_spatial_y": float(spatial.mean()),
                "median_spatial_y": float(np.median(spatial)),
                "mean_state_rank": float(
                    frame["within_patient_rank"].mean()
                ),
                "mean_predicted_score": float(
                    frame["predicted_score"].mean()
                ),
                "patient_balanced_mean_niche_fraction": float(
                    patient_frame["niche_fraction"].mean()
                ),
                "patient_balanced_median_niche_fraction": float(
                    patient_frame["niche_fraction"].median()
                ),
                "patient_balanced_mean_spatial_y": float(
                    patient_frame["mean_spatial_y"].mean()
                ),
                "patient_balanced_median_spatial_y": float(
                    patient_frame["mean_spatial_y"].median()
                ),
            }
        )
    pooled_summary = pd.DataFrame(pooled_rows)

    # Q1 and Q5 must each exist for every patient, even for the three-cell case.
    edge_counts = patient_summary.loc[
        patient_summary["state_quintile"].isin(["Q1", "Q5"])
    ].groupby("state_quintile", observed=True)["patient"].nunique()
    if any(int(edge_counts.get(label, 0)) != EXPECTED_PATIENTS for label in ("Q1", "Q5")):
        raise AssertionError("Q1 and Q5 must each be represented in all 74 patients")
    if int(patient_summary["cells"].sum()) != EXPECTED_PT_CD8_CELLS:
        raise AssertionError("Quintile summary cell counts do not sum to 38,299")
    return pooled_summary, patient_summary


def directional_stats(values: np.ndarray, *, label: str) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise AssertionError(f"Cannot test empty/non-finite values for {label}")
    positive = int(np.sum(values > 0.0))
    negative = int(np.sum(values < 0.0))
    ties = int(np.sum(values == 0.0))
    nonzero = positive + negative
    if nonzero == 0:
        raise AssertionError(f"All directional contrasts are zero for {label}")
    signed_rank = wilcoxon(
        values,
        alternative="greater",
        zero_method="wilcox",
        method="auto",
    )
    sign_test = binomtest(positive, nonzero, p=0.5, alternative="greater")
    return {
        "contrast": label,
        "patients": int(values.size),
        "positive_patients": positive,
        "negative_patients": negative,
        "tied_patients": ties,
        "mean_contrast": float(np.mean(values)),
        "median_contrast": float(np.median(values)),
        "one_sided_paired_wilcoxon_p": float(signed_rank.pvalue),
        "one_sided_exact_sign_p_excluding_ties": float(sign_test.pvalue),
        "alternative": "greater than zero",
    }


def quintile_trend_statistics(
    pooled: pd.DataFrame,
    patient_summary: pd.DataFrame,
) -> dict[str, Any]:
    def paired_edge_contrast(column: str) -> np.ndarray:
        wide = patient_summary.pivot(
            index="patient", columns="state_quintile", values=column
        )
        if "Q1" not in wide or "Q5" not in wide or wide[["Q1", "Q5"]].isna().any().any():
            raise AssertionError(f"Incomplete Q1/Q5 patient pairing for {column}")
        return (wide["Q5"] - wide["Q1"]).to_numpy(float)

    spatial_edge = directional_stats(
        paired_edge_contrast("mean_spatial_y"),
        label="patient Q5 mean_spatial_y minus Q1 mean_spatial_y",
    )
    niche_edge = directional_stats(
        paired_edge_contrast("niche_fraction"),
        label="patient Q5 niche_fraction minus Q1 niche_fraction",
    )

    spatial_slopes: list[float] = []
    niche_slopes: list[float] = []
    for _, frame in patient_summary.groupby("patient", sort=False):
        x = frame["state_quintile_index"].to_numpy(float)
        if len(np.unique(x)) < 2:
            raise AssertionError("Each patient must occupy at least two state quintiles")
        spatial_slopes.append(float(linregress(x, frame["mean_spatial_y"]).slope))
        niche_slopes.append(float(linregress(x, frame["niche_fraction"]).slope))

    pooled_x = pooled["state_quintile_index"].to_numpy(float)
    pooled_niche_fit = linregress(pooled_x, pooled["niche_fraction"].to_numpy(float))
    pooled_spatial_fit = linregress(pooled_x, pooled["mean_spatial_y"].to_numpy(float))
    q1 = pooled.set_index("state_quintile").loc["Q1"]
    q5 = pooled.set_index("state_quintile").loc["Q5"]
    return {
        "primary_patient_paired_q5_vs_q1_spatial_y": spatial_edge,
        "patient_paired_q5_vs_q1_niche_fraction": niche_edge,
        "within_patient_quintile_slope_spatial_y": directional_stats(
            np.asarray(spatial_slopes),
            label="within-patient slope of quintile mean_spatial_y per Q1-Q5 step",
        ),
        "within_patient_quintile_slope_niche_fraction": directional_stats(
            np.asarray(niche_slopes),
            label="within-patient slope of niche_fraction per Q1-Q5 step",
        ),
        "pooled_descriptive": {
            "q1_niche_fraction": float(q1["niche_fraction"]),
            "q5_niche_fraction": float(q5["niche_fraction"]),
            "q5_minus_q1_niche_fraction": float(
                q5["niche_fraction"] - q1["niche_fraction"]
            ),
            "q1_mean_spatial_y": float(q1["mean_spatial_y"]),
            "q5_mean_spatial_y": float(q5["mean_spatial_y"]),
            "q5_minus_q1_mean_spatial_y": float(
                q5["mean_spatial_y"] - q1["mean_spatial_y"]
            ),
            "linear_slope_niche_fraction_per_quintile": float(
                pooled_niche_fit.slope
            ),
            "linear_fit_niche_fraction_r_squared": float(
                pooled_niche_fit.rvalue**2
            ),
            "linear_slope_mean_spatial_y_per_quintile": float(
                pooled_spatial_fit.slope
            ),
            "linear_fit_mean_spatial_y_r_squared": float(
                pooled_spatial_fit.rvalue**2
            ),
            "note": (
                "Pooled Q1-Q5 linear fits are descriptive five-point summaries; "
                "patient-paired statistics above provide the inferential checks."
            ),
        },
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cells = load_and_validate_cells()
    maximum_rank_error = float(cells.attrs["maximum_recomputed_rank_error"])
    features = build_patient_features(cells)
    pooled_quintiles, patient_quintiles = build_quintile_summaries(cells)
    trends = quintile_trend_statistics(pooled_quintiles, patient_quintiles)

    write_csv(features, PATIENT_FEATURE_PATH)
    write_csv(pooled_quintiles, POOLED_QUINTILE_PATH)
    write_csv(patient_quintiles, PATIENT_QUINTILE_PATH)

    high = cells["high_state_top40"].to_numpy(bool)
    niche = cells["in_niche"].to_numpy(bool)
    overlap = high & niche
    manifest: dict[str, Any] = {
        "status": "formal outcome-blind top-40% spatial-return feature construction",
        "survival_outcome_read": False,
        "input": {
            "path": str(CELL_PATH.relative_to(HERE)),
            "sha256": sha256(CELL_PATH),
            "required_columns": list(REQUIRED_COLUMNS),
            "patients": int(cells["patient"].nunique()),
            "pt_cd8_cells": int(len(cells)),
        },
        "validation": {
            "expected_patients": EXPECTED_PATIENTS,
            "expected_pt_cd8_cells": EXPECTED_PT_CD8_CELLS,
            "cell_uid_complete_and_unique": True,
            "niche_flag_verified_as_spatial_y_gt_zero": True,
            "state_rank_verified_as_within_patient_percentile_midrank": True,
            "maximum_recomputed_rank_absolute_error": maximum_rank_error,
        },
        "high_state": {
            "definition": "within_patient_rank >= 0.60",
            "threshold": HIGH_STATE_THRESHOLD,
            "threshold_inclusive": True,
            "pooled_high_state_cells": int(high.sum()),
            "pooled_high_state_fraction": float(high.mean()),
            "pooled_niche_cells": int(niche.sum()),
            "pooled_high_state_niche_cells": int(overlap.sum()),
            "pooled_high_state_niche_precision": float(overlap.sum() / high.sum()),
            "pooled_high_state_niche_recall": float(overlap.sum() / niche.sum()),
        },
        "patient_features": {
            "path": str(PATIENT_FEATURE_PATH.relative_to(HERE)),
            "rows": int(len(features)),
            "columns": list(features.columns),
            "primary_survival_feature": {
                "column": "high_pair_mean_spatial_y",
                "definition": (
                    "For each patient, mean continuous ROI-adjusted CAF/macrophage "
                    "spatial_y among PT-CD8 cells with returned TIGIT/CD39 "
                    "within-patient rank >= 0.60."
                ),
            },
            "direct_spatial_feature": {
                "column": "patient_spatial_niche_score",
                "definition": (
                    "For each patient, mean continuous ROI-adjusted CAF/macrophage "
                    "spatial_y among all PT-CD8 cells."
                ),
            },
            "niche_precision_definition": (
                "high-state in-niche cells divided by all high-state cells"
            ),
            "niche_recall_definition": (
                "high-state in-niche cells divided by all in-niche cells"
            ),
        },
        "state_quintiles": {
            "definition": (
                "Fixed bins of within_patient_rank: Q1=[0.0,0.2), "
                "Q2=[0.2,0.4), Q3=[0.4,0.6), Q4=[0.6,0.8), Q5=[0.8,1.0)."
            ),
            "pooled_summary": {
                "path": str(POOLED_QUINTILE_PATH.relative_to(HERE)),
                "rows": int(len(pooled_quintiles)),
                "unit": "state quintile; cells pooled across patients",
            },
            "patient_summary": {
                "path": str(PATIENT_QUINTILE_PATH.relative_to(HERE)),
                "rows": int(len(patient_quintiles)),
                "unit": "patient-by-state-quintile",
                "note": (
                    "A patient with only three PT-CD8 cells has no Q2 or Q4 row; "
                    "Q1 and Q5 are present for all 74 patients."
                ),
            },
        },
        "trend_statistics": trends,
        "outputs_sha256": {
            "patient_features": sha256(PATIENT_FEATURE_PATH),
            "state_quintile_pooled_summary": sha256(POOLED_QUINTILE_PATH),
            "state_quintile_patient_summary": sha256(PATIENT_QUINTILE_PATH),
        },
        "software": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "analysis_boundary": (
            "This stage reads only the formal cell-level return table listed above. "
            "It does not read survival outcomes and has no dependency on figure_HNSCC/test."
        ),
    }
    write_json(manifest, MANIFEST_PATH)
    print(json.dumps(manifest, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
