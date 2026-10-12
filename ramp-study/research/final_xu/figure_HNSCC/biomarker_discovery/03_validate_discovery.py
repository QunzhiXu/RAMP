#!/usr/bin/env python3
"""Validate the formal PT74 projection and outcome-blind biomarker screen."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECTION = HERE / "projection"
SCREEN = HERE / "screen"
FORMAL_ALIGNMENT_MANIFEST = HERE.parent / "alignment/output/alignment_manifest.json"


def main() -> None:
    projection_manifest = json.loads((PROJECTION / "projection_manifest.json").read_text())
    screen_manifest = json.loads((SCREEN / "screen_manifest.json").read_text())
    alignment_manifest = json.loads(FORMAL_ALIGNMENT_MANIFEST.read_text())
    projected = pd.read_csv(
        PROJECTION / "rahim_tumor_cd8_projected_scores.csv.gz",
        dtype={"cell_uid": str, "donor": str},
    )
    all_markers = pd.read_csv(SCREEN / "marker_screen_all_107_all_variants.csv")
    robust = pd.read_csv(SCREEN / "robust_candidates_pre_redundancy.csv")
    pruned = pd.read_csv(SCREEN / "robust_candidates_redundancy_pruned.csv")
    compact = pd.read_csv(SCREEN / "final_compact_markers_top5pct.csv")
    pairwise = pd.read_csv(SCREEN / "marker_pairwise_paper_clr_correlations.csv")
    programs = pd.read_csv(
        SCREEN / "rahim_tumor_cd8_biomarker_programs.csv.gz",
        dtype={"cell_uid": str, "donor": str},
    )

    if alignment_manifest["canonical_version"] != "2.3.0":
        raise AssertionError("Formal alignment is not canonical RAMP 2.3.0")
    if projection_manifest["inputs_sha256"]["aligned_ramp"] != alignment_manifest["sha256"]["aligned"]:
        raise AssertionError("Projection did not consume the current formal alignment")
    if projection_manifest["counts"]["haist_full_pt_cd8_cells"] != 38_299:
        raise AssertionError("Projection did not use every complete-cohort PT-CD8 cell")
    if len(projected) != 4_788 or projected.donor.nunique() != 3:
        raise AssertionError("Projection dimensions changed")
    if projected.cell_uid.duplicated().any():
        raise AssertionError("Duplicated projected Rahim cell ID")
    if not np.isfinite(projected.select_dtypes(include=[np.number])).all().all():
        raise AssertionError("Non-finite projected score")
    if len(all_markers) != 107 or all_markers.marker_raw.nunique() != 107:
        raise AssertionError("Candidate universe changed")
    if len(pairwise) != 5_671:
        raise AssertionError("Expected all 107 choose 2 marker correlations")
    if len(robust) != 8 or len(pruned) != 8:
        raise AssertionError("Robustness/redundancy stage counts changed")
    if set(compact.marker_raw) != {"TIGIT--VSTM3", "CD39"} or len(compact) != 2:
        raise AssertionError("Formal compact program did not reproduce TIGIT/CD39")
    if not compact.robust_median_rank_primary_variants.le(5.35).all():
        raise AssertionError("A compact marker fails the fixed 5.35/107 cutoff")
    if len(programs) != 4_788 or programs.cell_uid.duplicated().any():
        raise AssertionError("Cell-level program table does not cover projected cells exactly")
    if not set(programs.cell_uid) == set(projected.cell_uid):
        raise AssertionError("Program and projection cell IDs disagree")
    if screen_manifest["outcome_blind"] is not True:
        raise AssertionError("Screen is not declared outcome blind")
    if screen_manifest["forced_marker_names"] or screen_manifest["forced_marker_count"]:
        raise AssertionError("Compact result was forced")
    if screen_manifest["compact_rule"]["reranked_after_robustness_or_redundancy"]:
        raise AssertionError("Compact candidates were improperly reranked")
    for script in (HERE / "01_project_spatial_state.py", HERE / "02_screen_biomarkers.py"):
        source = script.read_text()
        if "test0827" in source or "/test/" in source:
            raise AssertionError(f"Formal script contains a test-directory dependency: {script.name}")

    marker_rows = compact.set_index("marker_raw")
    report = {
        "status": "passed",
        "formal_alignment_version": alignment_manifest["canonical_version"],
        "projection": {
            "rahim_tumour_cd8_cells": len(projected),
            "donors": projected.donor.nunique(),
            "haist_predictors": projection_manifest["counts"]["patient_specific_predictors"],
            "lopo_median_spearman": projection_manifest["lopo_summary"]["median_spearman"],
            "lopo_positive_patients": projection_manifest["lopo_summary"]["positive_spearman_patients"],
        },
        "screen_stage_counts": screen_manifest["stage_counts"],
        "compact_cutoff": screen_manifest["compact_rule"]["rank_cutoff"],
        "forced_marker_names": screen_manifest["forced_marker_names"],
        "forced_marker_count": screen_manifest["forced_marker_count"],
        "final_compact_markers": {
            marker: {
                "robust_median_rank": float(marker_rows.loc[marker, "robust_median_rank_primary_variants"]),
                "paper_clr_equal_donor_association": float(marker_rows.loc[marker, "paper_clr__mean_donor_association"]),
            }
            for marker in ("TIGIT--VSTM3", "CD39")
        },
        "outcome_blind": True,
        "test_directory_dependency": False,
    }
    (HERE / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
