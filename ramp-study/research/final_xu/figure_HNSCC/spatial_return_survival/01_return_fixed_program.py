#!/usr/bin/env python3
"""Return the formally selected TIGIT/CD39 program to all Haist PT-CD8 cells.

This is the outcome-blind molecular-state return stage of the formal HNSCC
workflow.  It uses only formal artifacts outside ``test``/``test0827``:

* the canonical 22-marker PT74 RAMP alignment;
* the formal biomarker screen's fixed compact TIGIT/CD39 program in 4,788
  Rahim tumour CD8 cells;
* the formal cell-level spatial scores; and
* the copied complete Haist CODEX cell table.

The full 38,299 Haist PT-CD8 cells are placed in the RAMP-aligned space using
the validated patient/marker monotone mappings from biomarker discovery.  One
distance-weighted k=30 predictor is fitted per Rahim donor, and the returned
score is the cell-wise median of the three donor-specific predictions.  No
clinical or survival outcome is read or written.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
FIGURE_HNSCC = HERE.parent
ALIGNMENT_INPUT = FIGURE_HNSCC / "alignment" / "input"
ALIGNMENT_OUTPUT = FIGURE_HNSCC / "alignment" / "output"
BIOMARKER = FIGURE_HNSCC / "biomarker_discovery"
SCREEN = BIOMARKER / "screen"
SPATIAL = (
    FIGURE_HNSCC
    / "survival_motivation"
    / "output"
    / "pt_cd8_cell_spatial_niche_scores.csv.gz"
)
HAIST_COMPLETE = (
    FIGURE_HNSCC
    / "sampled_data"
    / "haist_codex"
    / "complete"
    / "HNSCCdiscovery_CODEX.csv"
)
FORMAL_HELPER = BIOMARKER / "01_project_spatial_state.py"
COMPACT_MARKERS = SCREEN / "final_compact_markers_top5pct.csv"
SOURCE_PROGRAM = SCREEN / "rahim_tumor_cd8_biomarker_programs.csv.gz"
SCREEN_MANIFEST = SCREEN / "screen_manifest.json"
ALIGNMENT_MANIFEST = ALIGNMENT_OUTPUT / "alignment_manifest.json"
OUT = HERE / "output"

EXPECTED_FRIENDLY_MARKERS = ["TIGIT", "CD39"]
EXPECTED_RAW_MARKERS = ["TIGIT--VSTM3", "CD39"]
EXPECTED_ALIGNMENT_MARKERS = 22
EXPECTED_RAHIM_CELLS = 4_788
EXPECTED_RAHIM_DONORS = 3
EXPECTED_HAIST_CELLS = 38_299
EXPECTED_HAIST_PATIENTS = 74
K_TRANSFER = 30
SOURCE_RESPONSE = "compact_top5pct_clr_program"
FORBIDDEN_DIRECTORY_NAMES = frozenset({"test", "test0827"})


def assert_formal_path(path: Path) -> None:
    """Reject missing, out-of-tree, or test-derived input paths before reading."""
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(FIGURE_HNSCC.resolve()):
        raise AssertionError(f"Input is outside the formal HNSCC tree: {resolved}")
    forbidden = FORBIDDEN_DIRECTORY_NAMES.intersection(
        part.lower() for part in resolved.parts
    )
    if forbidden:
        raise AssertionError(
            f"Forbidden test/test0827 dependency ({sorted(forbidden)}): {resolved}"
        )


def read_json(path: Path) -> dict[str, object]:
    assert_formal_path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def load_formal_helper() -> ModuleType:
    """Import the validated formal biomarker-projection implementation."""
    assert_formal_path(FORMAL_HELPER)
    spec = importlib.util.spec_from_file_location(
        "hnscc_formal_biomarker_projection", FORMAL_HELPER
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import formal helper: {FORMAL_HELPER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = (
        "sha256",
        "write_csv_gz",
        "percentile_midrank",
        "read_formal_alignment",
        "read_selected_patients",
        "build_haist_curves",
        "load_full_haist_pt_cd8",
        "equal_donor_patient_scaler",
        "weighted_predict",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise AttributeError(f"Formal helper is missing validated functions: {missing}")
    return module


def validate_input_manifests(helper: ModuleType) -> tuple[dict[str, object], dict[str, object]]:
    """Confirm that the selected program and RAMP matrix match formal manifests."""
    alignment_manifest = read_json(ALIGNMENT_MANIFEST)
    screen_manifest = read_json(SCREEN_MANIFEST)

    if alignment_manifest.get("algorithm") != "RAMP":
        raise AssertionError("The formal alignment manifest is not a RAMP alignment")
    if alignment_manifest.get("canonical_version") != "2.3.0":
        raise AssertionError("Expected canonical RAMP version 2.3.0")
    if alignment_manifest.get("cells") != 94_488:
        raise AssertionError("Formal RAMP cell count changed")
    if len(alignment_manifest.get("shared_markers", [])) != EXPECTED_ALIGNMENT_MARKERS:
        raise AssertionError("Formal RAMP shared-marker dimension changed")
    recorded_aligned_hash = alignment_manifest.get("sha256", {}).get("aligned")
    if recorded_aligned_hash != helper.sha256(
        ALIGNMENT_OUTPUT / "aligned_ramp_ver2.float32.npy"
    ):
        raise AssertionError("The formal aligned matrix disagrees with its manifest")

    if screen_manifest.get("outcome_blind") is not True:
        raise AssertionError("Formal biomarker screen is not marked outcome-blind")
    if screen_manifest.get("survival_outcomes_read") is not False:
        raise AssertionError("Formal biomarker screen unexpectedly read survival outcomes")
    if screen_manifest.get("final_compact_markers") != EXPECTED_RAW_MARKERS:
        raise AssertionError(
            "Formal compact raw markers are not exactly TIGIT--VSTM3 and CD39"
        )
    stage_counts = screen_manifest.get("stage_counts", {})
    if stage_counts.get("final_compact_top5pct") != 2:
        raise AssertionError("Formal compact marker count is not exactly two")

    recorded_outputs = screen_manifest.get("outputs", {})
    for relative, path in (
        ("screen/final_compact_markers_top5pct.csv", COMPACT_MARKERS),
        ("screen/rahim_tumor_cd8_biomarker_programs.csv.gz", SOURCE_PROGRAM),
    ):
        recorded_hash = recorded_outputs.get(relative, {}).get("sha256")
        if recorded_hash != helper.sha256(path):
            raise AssertionError(f"Formal screen artifact disagrees with manifest: {path}")
    return alignment_manifest, screen_manifest


def read_locked_source(
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    """Read and align the fixed Rahim TIGIT/CD39 source response."""
    assert_formal_path(COMPACT_MARKERS)
    compact = pd.read_csv(COMPACT_MARKERS)
    if compact.marker.astype(str).tolist() != EXPECTED_FRIENDLY_MARKERS:
        raise AssertionError("Final compact friendly markers are not exactly TIGIT/CD39")
    if compact.marker_raw.astype(str).tolist() != EXPECTED_RAW_MARKERS:
        raise AssertionError(
            "Final compact raw-marker order is not TIGIT--VSTM3 followed by CD39"
        )
    if len(compact) != 2 or compact.compact_selection_order.tolist() != [1, 2]:
        raise AssertionError("The formal compact selection is not a locked two-marker panel")

    assert_formal_path(SOURCE_PROGRAM)
    source = pd.read_csv(
        SOURCE_PROGRAM,
        dtype={"cell_uid": str, "donor": str},
    )
    required = {"cell_uid", "donor", SOURCE_RESPONSE}
    if not required.issubset(source.columns):
        raise KeyError(f"Source program lacks required columns: {sorted(required - set(source))}")
    if len(source) != EXPECTED_RAHIM_CELLS or source.cell_uid.nunique() != len(source):
        raise AssertionError("Expected 4,788 unique cells in the formal source program")
    response = pd.to_numeric(source[SOURCE_RESPONSE], errors="raise").to_numpy(float)
    if not np.isfinite(response).all() or np.any((response < 0) | (response > 1)):
        raise AssertionError("The compact CLR percentile program must be finite in [0, 1]")

    source_rows = np.flatnonzero(
        metadata.dataset.eq("GSE212797_ADT").to_numpy()
        & metadata.tissue.eq("tumor").to_numpy()
        & metadata.provider_cell_type.eq("CD8 T cell").to_numpy()
    )
    source_metadata = metadata.iloc[source_rows].copy().reset_index(drop=True)
    source_metadata["cell_uid"] = source_metadata.cell_uid.str.removeprefix("RAHIM|")
    if (
        len(source_metadata) != EXPECTED_RAHIM_CELLS
        or source_metadata.cell_uid.nunique() != EXPECTED_RAHIM_CELLS
        or source_metadata.donor.nunique() != EXPECTED_RAHIM_DONORS
    ):
        raise AssertionError("Formal alignment does not contain the expected Rahim tumor-CD8 set")

    source_indexed = source.set_index("cell_uid", verify_integrity=True)
    missing = source_metadata.loc[
        ~source_metadata.cell_uid.isin(source_indexed.index), "cell_uid"
    ]
    extra = source.loc[
        ~source.cell_uid.isin(source_metadata.cell_uid), "cell_uid"
    ]
    if len(missing) or len(extra):
        raise AssertionError(
            f"Source/alignment cell mismatch: {len(missing)} missing, {len(extra)} extra"
        )
    ordered = source_indexed.loc[source_metadata.cell_uid].reset_index()
    if not ordered.donor.astype(str).eq(source_metadata.donor.astype(str)).all():
        raise AssertionError("Rahim donor labels disagree between source and alignment")
    ordered_response = ordered[SOURCE_RESPONSE].to_numpy(float)
    return source_metadata, source_rows, ordered_response, compact


def read_spatial_details(selected_ids: set[str]) -> pd.DataFrame:
    """Read coordinates and the formal observed spatial phenotype for output."""
    assert_formal_path(SPATIAL)
    details = pd.read_csv(
        SPATIAL,
        usecols=[
            "UID",
            "Pat_ID",
            "unique_region",
            "x",
            "y",
            "roi_adjusted_caf_macrophage_score",
            "in_caf_macrophage_rich_niche",
        ],
        dtype={"UID": str, "Pat_ID": str, "unique_region": str},
    )
    details = details.loc[details.Pat_ID.isin(selected_ids)].copy()
    if len(details) != EXPECTED_HAIST_CELLS:
        raise AssertionError("Formal spatial file does not contain 38,299 selected PT-CD8 cells")
    if details[["UID", "Pat_ID"]].duplicated().any():
        raise AssertionError("Formal spatial details have duplicated cell keys")
    numeric = details[["x", "y", "roi_adjusted_caf_macrophage_score"]].to_numpy(float)
    if not np.isfinite(numeric).all() or details.unique_region.isna().any():
        raise AssertionError("Formal spatial details contain missing/non-finite values")

    niche = details.in_caf_macrophage_rich_niche
    if niche.dtype != bool:
        parsed = niche.astype(str).str.lower().map({"true": True, "false": False})
        if parsed.isna().any():
            raise AssertionError("Formal in-niche indicator is not Boolean")
        details["in_caf_macrophage_rich_niche"] = parsed.astype(bool)
    return details


def donor_predict(
    source_z: np.ndarray,
    source_donors: np.ndarray,
    response: np.ndarray,
    target_z: np.ndarray,
    helper: ModuleType,
) -> tuple[list[str], np.ndarray]:
    """Fit one k=30 source predictor per Rahim donor and query all Haist cells."""
    donors = sorted(pd.unique(source_donors), key=str)
    if len(donors) != EXPECTED_RAHIM_DONORS:
        raise AssertionError("Expected exactly three Rahim source donors")
    predictions: list[np.ndarray] = []
    for donor in donors:
        rows = np.flatnonzero(source_donors == donor)
        if len(rows) < K_TRANSFER:
            raise AssertionError(f"Rahim donor {donor} has fewer than k={K_TRANSFER} cells")
        tree = cKDTree(source_z[rows])
        predictions.append(helper.weighted_predict(tree, response[rows], target_z))
    matrix = np.column_stack(predictions)
    if matrix.shape != (EXPECTED_HAIST_CELLS, EXPECTED_RAHIM_DONORS):
        raise AssertionError(f"Unexpected donor-prediction shape: {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise AssertionError("Returned donor-specific predictions are non-finite")
    return donors, matrix


def build_source_audit(
    donors: list[str],
    source_donors: np.ndarray,
    response: np.ndarray,
    donor_predictions: np.ndarray,
) -> pd.DataFrame:
    """Summarize each locked Rahim donor source and its Haist predictions."""
    rows: list[dict[str, object]] = []
    for column, donor in enumerate(donors):
        source_values = response[source_donors == donor]
        target_values = donor_predictions[:, column]
        rows.append(
            {
                "rahim_donor": donor,
                "source_cells": int(len(source_values)),
                "source_response_column": SOURCE_RESPONSE,
                "effective_k": K_TRANSFER,
                "source_response_mean": float(np.mean(source_values)),
                "source_response_sd": float(np.std(source_values, ddof=1)),
                "source_response_min": float(np.min(source_values)),
                "source_response_q25": float(np.quantile(source_values, 0.25)),
                "source_response_median": float(np.median(source_values)),
                "source_response_q75": float(np.quantile(source_values, 0.75)),
                "source_response_max": float(np.max(source_values)),
                "haist_query_cells": int(len(target_values)),
                "haist_prediction_mean": float(np.mean(target_values)),
                "haist_prediction_sd": float(np.std(target_values, ddof=1)),
                "haist_prediction_min": float(np.min(target_values)),
                "haist_prediction_median": float(np.median(target_values)),
                "haist_prediction_max": float(np.max(target_values)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    sys.dont_write_bytecode = True
    formal_inputs = [
        ALIGNMENT_INPUT / "raw_shared_markers.float64.npy",
        ALIGNMENT_INPUT / "cell_metadata.csv.gz",
        ALIGNMENT_INPUT / "selected_74_patients.csv",
        ALIGNMENT_INPUT / "markers.txt",
        ALIGNMENT_OUTPUT / "aligned_ramp_ver2.float32.npy",
        ALIGNMENT_MANIFEST,
        COMPACT_MARKERS,
        SOURCE_PROGRAM,
        SCREEN_MANIFEST,
        SPATIAL,
        HAIST_COMPLETE,
        FORMAL_HELPER,
    ]
    for path in formal_inputs:
        assert_formal_path(path)

    helper = load_formal_helper()
    alignment_manifest, screen_manifest = validate_input_manifests(helper)
    raw, aligned, metadata, markers = helper.read_formal_alignment()
    if len(markers) != EXPECTED_ALIGNMENT_MARKERS:
        raise AssertionError("Expected a 22-marker formal RAMP space")
    source_metadata, source_rows, response, compact = read_locked_source(metadata)
    source_aligned = np.asarray(aligned[source_rows], dtype=np.float64)
    source_donors = source_metadata.donor.to_numpy(str)

    selected = helper.read_selected_patients()
    curves, curve_audit = helper.build_haist_curves(raw, aligned, metadata, markers)
    haist, haist_aligned, curve_audit = helper.load_full_haist_pt_cd8(
        selected, curves, markers, curve_audit
    )
    if len(haist) != EXPECTED_HAIST_CELLS or haist.Pat_ID.nunique() != EXPECTED_HAIST_PATIENTS:
        raise AssertionError("Expected 38,299 PT-CD8 cells from 74 Haist patients")
    haist_patients = haist.Pat_ID.to_numpy(str)

    center, scale, scaling_audit = helper.equal_donor_patient_scaler(
        source_aligned,
        source_donors,
        haist_aligned,
        haist_patients,
        markers,
    )
    source_z = (source_aligned - center) / scale
    haist_z = (haist_aligned - center) / scale
    donors, donor_predictions = donor_predict(
        source_z, source_donors, response, haist_z, helper
    )
    predicted = np.median(donor_predictions, axis=1)
    donor_q25 = np.quantile(donor_predictions, 0.25, axis=1)
    donor_q75 = np.quantile(donor_predictions, 0.75, axis=1)
    within_patient_rank = helper.percentile_midrank(predicted, haist_patients)
    if not np.isfinite(predicted).all() or np.std(predicted) <= 0:
        raise AssertionError("Final returned program is non-finite or constant")
    rank_means = pd.Series(within_patient_rank).groupby(haist_patients).mean()
    if not np.allclose(rank_means.to_numpy(float), 0.5, atol=1e-12):
        raise AssertionError("Within-patient percentile ranks are not centered at 0.5")

    selected_ids = set(selected.CaseID.astype(str))
    details = read_spatial_details(selected_ids)
    returned = pd.DataFrame(
        {
            "cell_uid": haist.UID.astype(str),
            "patient": haist.Pat_ID.astype(str),
            "_target_row": np.arange(len(haist), dtype=int),
        }
    )
    returned = returned.merge(
        details.rename(columns={"UID": "cell_uid", "Pat_ID": "patient"}),
        on=["cell_uid", "patient"],
        how="left",
        validate="one_to_one",
        sort=False,
    ).sort_values("_target_row", kind="stable")
    if returned.unique_region.isna().any() or not np.array_equal(
        returned._target_row.to_numpy(int), np.arange(len(haist))
    ):
        raise AssertionError("Spatial-detail join lost or reordered Haist cells")

    returned = returned.rename(
        columns={
            "roi_adjusted_caf_macrophage_score": "spatial_y",
            "in_caf_macrophage_rich_niche": "in_niche",
        }
    )
    returned["predicted_score"] = predicted
    returned["donor_prediction_q25"] = donor_q25
    returned["donor_prediction_q75"] = donor_q75
    returned["donor_prediction_iqr"] = donor_q75 - donor_q25
    returned["within_patient_rank"] = within_patient_rank
    returned = returned.drop(columns="_target_row")
    returned["_patient_numeric"] = pd.to_numeric(returned.patient, errors="raise")
    returned = (
        returned.sort_values(["_patient_numeric", "cell_uid"], kind="stable")
        .drop(columns="_patient_numeric")
        .reset_index(drop=True)
    )
    required_output = {
        "patient",
        "unique_region",
        "x",
        "y",
        "spatial_y",
        "in_niche",
        "predicted_score",
        "within_patient_rank",
    }
    if not required_output.issubset(returned.columns):
        raise AssertionError("Returned cell table is missing required output fields")
    if len(returned) != EXPECTED_HAIST_CELLS or returned.cell_uid.nunique() != len(returned):
        raise AssertionError("Returned cell table does not contain 38,299 unique cells")

    source_audit = build_source_audit(
        donors, source_donors, response, donor_predictions
    )
    OUT.mkdir(parents=True, exist_ok=True)
    returned_path = OUT / "haist74_pt_cd8_tigit_cd39_return.csv.gz"
    source_audit_path = OUT / "rahim_donor_source_audit.csv"
    manifest_path = OUT / "return_manifest.json"
    helper.write_csv_gz(returned, returned_path)
    source_audit.to_csv(source_audit_path, index=False)

    pairwise_prediction_spearman: dict[str, float] = {}
    for left in range(len(donors)):
        for right in range(left + 1, len(donors)):
            key = f"{donors[left]}__{donors[right]}"
            pairwise_prediction_spearman[key] = float(
                spearmanr(
                    donor_predictions[:, left], donor_predictions[:, right]
                ).statistic
            )

    input_paths = {
        "raw_shared_markers": ALIGNMENT_INPUT / "raw_shared_markers.float64.npy",
        "alignment_metadata": ALIGNMENT_INPUT / "cell_metadata.csv.gz",
        "selected_patients": ALIGNMENT_INPUT / "selected_74_patients.csv",
        "alignment_markers": ALIGNMENT_INPUT / "markers.txt",
        "aligned_ramp": ALIGNMENT_OUTPUT / "aligned_ramp_ver2.float32.npy",
        "alignment_manifest": ALIGNMENT_MANIFEST,
        "compact_marker_table": COMPACT_MARKERS,
        "rahim_source_program": SOURCE_PROGRAM,
        "biomarker_screen_manifest": SCREEN_MANIFEST,
        "formal_spatial_cell_scores": SPATIAL,
        "complete_haist_cell_table": HAIST_COMPLETE,
        "validated_projection_helper": FORMAL_HELPER,
    }
    output_paths = {
        "cell_return": returned_path,
        "rahim_donor_source_audit": source_audit_path,
    }
    manifest = {
        "schema_version": 1,
        "stage": "formal outcome-blind fixed-program return",
        "purpose": "Rahim TIGIT/CD39 program return to all Haist PT-CD8 cells",
        "fixed_biomarker_program": {
            "friendly_markers": compact.marker.astype(str).tolist(),
            "raw_markers": compact.marker_raw.astype(str).tolist(),
            "source_response_column": SOURCE_RESPONSE,
            "source_response_definition": (
                "mean of within-Rahim-donor strict paper-CLR percentiles for "
                "the formally selected TIGIT and CD39 ADTs"
            ),
            "source_artifact_preferred_and_used": True,
        },
        "algorithm": {
            "alignment": (
                f"canonical RAMP {alignment_manifest['canonical_version']} formal PT74 alignment"
            ),
            "shared_marker_dimensions": len(markers),
            "shared_markers": markers,
            "full_haist_mapping": (
                "validated empirical monotone raw-to-RAMP mapping per Haist patient "
                "and shared marker with constant boundary extension"
            ),
            "feature_scaling": (
                "50% equal-Rahim-donor first/second moments plus 50% "
                "equal-Haist-patient first/second moments"
            ),
            "predictor": (
                "one distance-weighted k30 predictor per Rahim donor; "
                "exp(-distance/within-query median-neighbour-distance) weights"
            ),
            "aggregation": "cell-wise median across the three Rahim-donor predictions",
            "within_patient_rank": "average-tie percentile midrank",
        },
        "counts": {
            "rahim_tumour_cd8_source_cells": len(source_metadata),
            "rahim_source_donors": len(donors),
            "rahim_cells_by_donor": {
                str(key): int(value)
                for key, value in pd.Series(source_donors).value_counts(sort=False).items()
            },
            "haist_pt_cd8_query_cells": len(returned),
            "haist_query_patients": int(returned.patient.nunique()),
            "minimum_pt_cd8_cells_per_patient": int(
                returned.groupby("patient").size().min()
            ),
            "patient_marker_mapping_curves": len(curve_audit),
        },
        "returned_score_summary": {
            "mean": float(np.mean(predicted)),
            "sd": float(np.std(predicted, ddof=1)),
            "min": float(np.min(predicted)),
            "q25": float(np.quantile(predicted, 0.25)),
            "median": float(np.median(predicted)),
            "q75": float(np.quantile(predicted, 0.75)),
            "max": float(np.max(predicted)),
            "median_donor_prediction_iqr": float(np.median(donor_q75 - donor_q25)),
            "pairwise_donor_prediction_spearman": pairwise_prediction_spearman,
        },
        "feature_scaler": scaling_audit.to_dict(orient="records"),
        "formal_source_audit": {
            "screen_analysis": screen_manifest.get("analysis"),
            "formal_compact_marker_count": len(compact),
            "source_cell_ids_match_formal_alignment": True,
            "source_donor_labels_match_formal_alignment": True,
            "all_input_paths_rejected_if_under_test_or_test0827": True,
        },
        "outcome_blinding": {
            "clinical_or_survival_outcome_columns_read": False,
            "survival_used_for_return": False,
            "survival_analysis_written": False,
            "test_or_test0827_dependency": False,
        },
        "inputs": {name: str(path.resolve()) for name, path in input_paths.items()},
        "inputs_sha256": {name: helper.sha256(path) for name, path in input_paths.items()},
        "outputs": {name: str(path.resolve()) for name, path in output_paths.items()},
        "outputs_sha256": {name: helper.sha256(path) for name, path in output_paths.items()},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
