#!/usr/bin/env python3
"""Fit the formal clean 15-marker melanoma alignment with canonical RAMP 2.3.0.

The cell cohort, cell order, dataset order, and RAMP family are identical to
the frozen shared13 fit.  The panel change is exactly:

* remove HLA_DR (nonspecific staining flagged by the source paper);
* add LAG3, ICOS, and OX40 using exact cross-assay protein mappings.

The three added columns are read directly from centralized raw files in memory.
No raw matrix is copied into this directory.  The frozen formal story metadata
is used only to assert the exact 165,686-cell identity and row order.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
FIGURE_ROOT = HERE.parents[1]
FINAL_XU = FIGURE_ROOT.parent
ALGORITHM_ROOT = FINAL_XU / "algorithm"
BASE_INPUT_ROOT = FIGURE_ROOT / "inputs/cohort_package"
RAW_ROOT = FINAL_XU / "raw_data/melanoma"
OUTPUT = HERE / "output"

MOLD_INPUT = BASE_INPUT_ROOT / "moldoveanu_imc/alignment_input.csv.gz"
ENG_INPUT = BASE_INPUT_ROOT / "engelbrecht_citeseq/alignment_input.csv.gz"
MOLD_RAW = RAW_ROOT / "moldoveanu_imc/Moldoveanu_2022_CyTOF_melanoma/Data/ICI_meanIntensity.tsv.gz"
ENG_ADT = RAW_ROOT / "engelbrecht_citeseq/raw/ADT_counts.csv"
FROZEN_STORY_METADATA = FIGURE_ROOT / "inputs/spatial_cell_metadata.csv.gz"

IDENTITY = ["cell_uid", "dataset", "donor"]
DATASET_ORDER = ["Moldoveanu IMC", "Engelbrecht CITE-seq"]
FAMILY = "melanoma_full_cohort_survival_projection"
CLEAN15 = [
    "PDL1", "CD45", "CD4", "CD20", "CD8", "CD45RA", "CD40",
    "CD3", "CD45RO", "CD14", "CD31", "CD29", "LAG3", "ICOS", "OX40",
]
MOLD_NEW = {"LAG3": "LAG3", "ICOS": "ICOS", "OX40": "OX40"}
ENG_NEW = {"LAG3": "CD223-AB", "ICOS": "CD278-AB", "OX40": "CD134-AB"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv_gz(frame: pd.DataFrame, path: Path) -> None:
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(fileobj=raw_handle, mode="wb", mtime=0) as handle:
            frame.to_csv(handle, index=False)


def build_moldoveanu(base: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    raw = pd.read_csv(
        MOLD_RAW,
        sep="\t",
        usecols=["Sample_ID", "Cell_ID", *MOLD_NEW.values()],
        low_memory=False,
    )
    raw["cell_uid"] = raw["Sample_ID"].astype(str) + "::" + raw["Cell_ID"].astype(str)
    if raw["cell_uid"].duplicated().any():
        raise AssertionError("Duplicate Moldoveanu raw cell_uid")
    raw = raw[["cell_uid", *MOLD_NEW.values()]]
    result = base[IDENTITY + [m for m in CLEAN15 if m in base.columns]].merge(
        raw,
        on="cell_uid",
        how="left",
        sort=False,
        validate="one_to_one",
    )
    for marker, source in MOLD_NEW.items():
        result[marker] = pd.to_numeric(result.pop(source), errors="raise")
    result = result[IDENTITY + CLEAN15]
    if result[CLEAN15].isna().any().any():
        raise AssertionError("Missing Moldoveanu clean15 value")
    fractions = {marker: float((result[marker] > 0).mean()) for marker in MOLD_NEW}
    return result, fractions


def build_engelbrecht(base: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    header = pd.read_csv(ENG_ADT, nrows=0).columns.astype(str).tolist()
    feature_column = header[0]
    barcodes = base["cell_uid"].astype(str).tolist()
    missing_barcodes = set(barcodes) - set(header[1:])
    if missing_barcodes:
        raise AssertionError(f"Missing {len(missing_barcodes)} Engelbrecht barcodes")
    matrix = pd.read_csv(ENG_ADT, usecols=[feature_column, *barcodes])
    matrix[feature_column] = matrix[feature_column].astype(str)
    matrix = matrix.loc[matrix[feature_column].isin(ENG_NEW.values())].set_index(feature_column)
    if set(ENG_NEW.values()) != set(matrix.index):
        raise AssertionError("Missing Engelbrecht clean marker feature")

    result = base[IDENTITY + [m for m in CLEAN15 if m in base.columns]].copy()
    for marker, source in ENG_NEW.items():
        result[marker] = matrix.loc[source, barcodes].to_numpy(np.float64)
    result = result[IDENTITY + CLEAN15]
    if result[CLEAN15].isna().any().any():
        raise AssertionError("Missing Engelbrecht clean15 value")
    fractions = {marker: float((result[marker] > 0).mean()) for marker in ENG_NEW}
    return result, fractions


def main() -> None:
    sys.path.insert(0, str(ALGORITHM_ROOT))
    from ramp import __version__, fit_transform

    if __version__ != "2.3.0":
        raise AssertionError(f"Expected canonical RAMP 2.3.0, imported {__version__}")

    mold_base = pd.read_csv(MOLD_INPUT, low_memory=False)
    eng_base = pd.read_csv(ENG_INPUT, low_memory=False)
    if list(mold_base.columns) != list(eng_base.columns):
        raise AssertionError("Frozen shared13 input column order differs")
    if str(mold_base["dataset"].iloc[0]) != DATASET_ORDER[0]:
        raise AssertionError("Unexpected Moldoveanu dataset name")
    if str(eng_base["dataset"].iloc[0]) != DATASET_ORDER[1]:
        raise AssertionError("Unexpected Engelbrecht dataset name")

    mold, mold_fractions = build_moldoveanu(mold_base)
    eng, eng_fractions = build_engelbrecht(eng_base)
    combined = pd.concat([mold, eng], ignore_index=True)
    if combined["cell_uid"].duplicated().any():
        raise AssertionError("Clean15 cell_uid is not globally unique")
    raw = combined[CLEAN15].to_numpy(np.float64)
    if not np.isfinite(raw).all():
        raise AssertionError("Non-finite clean15 input")

    metadata = combined[IDENTITY].copy()
    metadata["unit_key"] = metadata["dataset"].astype(str) + "::" + metadata["donor"].astype(str)
    frozen = pd.read_csv(
        FROZEN_STORY_METADATA,
        usecols=["cell_uid_ordered", "dataset", "donor"],
        low_memory=False,
    )
    if len(frozen) != len(metadata):
        raise AssertionError("Frozen story metadata and clean15 cohort sizes differ")
    for current, expected in (
        (metadata["cell_uid"], frozen["cell_uid_ordered"]),
        (metadata["dataset"], frozen["dataset"]),
        (metadata["donor"], frozen["donor"]),
    ):
        if not current.astype(str).equals(expected.astype(str)):
            raise AssertionError("Clean15 identity or row order differs from the frozen story cohort")

    result = fit_transform(
        raw,
        metadata[["dataset", "unit_key"]],
        CLEAN15,
        datasets=DATASET_ORDER,
        family=FAMILY,
    )
    if result.strict_rank_inversions != 0:
        raise AssertionError(f"Rank-preserving contract failed: {result.strict_rank_inversions}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT / "kmeanslow_baseline.npy", result.baseline.astype(np.float32))
    np.save(OUTPUT / "ramp_aligned.npy", result.aligned.astype(np.float32))
    write_csv_gz(metadata, OUTPUT / "cell_metadata.csv.gz")
    (OUTPUT / "markers.txt").write_text("\n".join(CLEAN15) + "\n")
    result.coupling_audit.to_csv(OUTPUT / "coupling_audit.csv", index=False)
    result.strength_audit.to_csv(OUTPUT / "strength_audit.csv", index=False)
    result.evidence_audit.to_csv(OUTPUT / "evidence_audit.csv", index=False)
    result.reference_weight_audit.to_csv(OUTPUT / "reference_weight_audit.csv", index=False)
    result.profile_audit.to_csv(OUTPUT / "profile_audit.csv", index=False)
    result.algorithm_audit.to_csv(OUTPUT / "algorithm_audit.csv", index=False)

    canonical_source = ALGORITHM_ROOT / "ramp/ramp_ver2.py"
    manifest = {
        "algorithm": "RAMP",
        "algorithm_version": __version__,
        "canonical_source": str(canonical_source),
        "canonical_source_sha256": sha256(canonical_source),
        "family": FAMILY,
        "family_matches_historical_shared13_comparator": True,
        "dataset_order": DATASET_ORDER,
        "cells": int(len(combined)),
        "dataset_cells": combined["dataset"].value_counts(sort=False).to_dict(),
        "dataset_patients": combined.groupby("dataset")["donor"].nunique().to_dict(),
        "shared_markers": CLEAN15,
        "marker_count": len(CLEAN15),
        "panel_change_vs_shared13": {
            "removed": ["HLA_DR"],
            "added": ["LAG3", "ICOS", "OX40"],
        },
        "new_marker_mapping": {
            marker: {"Moldoveanu IMC": MOLD_NEW[marker], "Engelbrecht CITE-seq": ENG_NEW[marker]}
            for marker in MOLD_NEW
        },
        "positive_fraction_new_markers": {
            "Moldoveanu IMC": mold_fractions,
            "Engelbrecht CITE-seq": eng_fractions,
        },
        "strict_rank_inversions": int(result.strict_rank_inversions),
        "labels_coordinates_clinical_used": False,
        "raw_data_copied": False,
        "row_order_matches_frozen_story_cohort": True,
        "sources": {
            "moldoveanu_shared13_input": str(MOLD_INPUT),
            "moldoveanu_shared13_input_sha256": sha256(MOLD_INPUT),
            "engelbrecht_shared13_input": str(ENG_INPUT),
            "engelbrecht_shared13_input_sha256": sha256(ENG_INPUT),
            "moldoveanu_raw": str(MOLD_RAW),
            "moldoveanu_raw_sha256": sha256(MOLD_RAW),
            "engelbrecht_adt": str(ENG_ADT),
            "engelbrecht_adt_sha256": sha256(ENG_ADT),
            "frozen_story_metadata": str(FROZEN_STORY_METADATA),
            "frozen_story_metadata_sha256": sha256(FROZEN_STORY_METADATA),
        },
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
