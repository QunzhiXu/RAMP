#!/usr/bin/env python3
"""Validate the formal melanoma RAMP 2.3.0 analysis and provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ALIGNMENT = HERE / "analysis/alignment/output"
ALL_CD8 = HERE / "analysis/all_cd8_prediction"
INPUTS = HERE / "inputs"
OUTPUT = HERE / "output"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_close(observed: float, expected: float, tolerance: float = 1e-12) -> None:
    if not np.isclose(observed, expected, atol=tolerance, rtol=0):
        raise AssertionError(f"Expected {expected}, observed {observed}")


def main() -> None:
    required = [
        ALIGNMENT / "ramp_aligned.npy",
        ALIGNMENT / "cell_metadata.csv.gz",
        ALIGNMENT / "manifest.json",
        ALL_CD8 / "patient_metrics.csv",
        ALL_CD8 / "outer_patient_predictions.csv.gz",
        ALL_CD8 / "summary.json",
        ALL_CD8 / "spatial_figure/manifest.json",
        ALL_CD8 / "citeseq_projection/engelbrecht_all_cd8_projection.csv.gz",
        ALL_CD8 / "biological_interpretation/candidate_biomarker_modules.csv",
        HERE / "output/melanoma_cd8_three_row_figure.png",
        HERE / "output/pdf/melanoma_cd8_three_row_figure.pdf",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing formal outputs: {missing}")

    alignment_manifest = json.loads((ALIGNMENT / "manifest.json").read_text())
    if alignment_manifest.get("algorithm_version") != "2.3.0":
        raise AssertionError("Formal alignment is not RAMP 2.3.0")
    if alignment_manifest.get("strict_rank_inversions") != 0:
        raise AssertionError("Formal alignment violates the rank-preserving contract")
    aligned = np.load(ALIGNMENT / "ramp_aligned.npy", mmap_mode="r")
    if aligned.shape != (165_686, 15):
        raise AssertionError(f"Unexpected aligned shape: {aligned.shape}")

    summary = json.loads((ALL_CD8 / "summary.json").read_text())
    if summary.get("n_all_cd8") != 5_649 or summary.get("n_targets") != 478:
        raise AssertionError("Unexpected spatial all-CD8 cohort")
    if summary.get("n_evaluable_patients") != 14:
        raise AssertionError("Expected 14 evaluable held-out patients")
    require_close(float(summary["median_AUROC"]), 0.7068785512466234)
    require_close(
        float(summary["median_AUPRC_over_prevalence"]),
        2.3164780616740583,
    )

    projection = pd.read_csv(
        ALL_CD8 / "citeseq_projection/engelbrecht_all_cd8_projection.csv.gz",
        usecols=["Patient_ID"],
    )
    if len(projection) != 5_768 or projection["Patient_ID"].astype(str).nunique() != 8:
        raise AssertionError("Unexpected formal CITE-seq CD8 projection cohort")

    spatial = json.loads((ALL_CD8 / "spatial_figure/manifest.json").read_text())
    selected = spatial["recommended_top20"]
    if (selected["tp_shown"], selected["fp_shown"], selected["fn_shown"]) != (17, 1, 6):
        raise AssertionError("The frozen 22RD spatial display changed")

    modules = pd.read_csv(
        ALL_CD8 / "biological_interpretation/candidate_biomarker_modules.csv"
    )
    core = modules.loc[
        modules["module"].eq("core_4_LFA1_CD2_PD1")
        & modules["population"].eq("all_CD8")
    ]
    if len(core) != 1 or int(core.iloc[0]["patients_positive"]) != 8:
        raise AssertionError("The fixed four-marker biological program changed")

    forbidden = (
        "/" + "test" + "/",
        "/" + "test" + "0827" + "/",
        "final_xu/" + "test" + "/",
    )
    inspected: list[str] = []
    for path in sorted(HERE.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".py", ".r", ".sh", ".json"}:
            continue
        text = path.read_text(errors="ignore")
        if any(token in text for token in forbidden):
            raise AssertionError(f"Development-directory dependency remains in {path}")
        inspected.append(str(path))

    input_manifest = {
        "schema_version": "final_xu.figure_melanoma.inputs.v1",
        "raw_data_copied": False,
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(INPUTS.iterdir())
            if path.is_file() and path.name != "manifest.json"
        },
    }
    (INPUTS / "manifest.json").write_text(json.dumps(input_manifest, indent=2) + "\n")

    report = {
        "status": "PASS",
        "algorithm_version": "2.3.0",
        "aligned_shape": list(aligned.shape),
        "strict_rank_inversions": 0,
        "alignment_sha256": sha256(ALIGNMENT / "ramp_aligned.npy"),
        "spatial_all_cd8_cells": 5_649,
        "spatial_targets": 478,
        "evaluable_patients": 14,
        "median_AUROC": float(summary["median_AUROC"]),
        "median_AUPRC_over_prevalence": float(
            summary["median_AUPRC_over_prevalence"]
        ),
        "citeseq_cd8_cells": 5_768,
        "citeseq_patients": 8,
        "spatial_display_tp_fp_fn": [17, 1, 6],
        "runtime_files_checked_for_development_paths": len(inspected),
    }
    (OUTPUT / "formal_validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
