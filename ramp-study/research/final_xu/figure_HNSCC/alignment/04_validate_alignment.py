#!/usr/bin/env python3
"""Validate the formal PT74 alignment and UMAP without using outcomes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
INPUT = HERE / "input"
OUTPUT = HERE / "output"
UMAP = HERE / "umap"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    raw = np.load(INPUT / "raw_shared_markers.float64.npy", mmap_mode="r")
    aligned = np.load(OUTPUT / "aligned_ramp_ver2.float32.npy", mmap_mode="r")
    metadata = pd.read_csv(INPUT / "cell_metadata.csv.gz", low_memory=False)
    selected = pd.read_csv(INPUT / "selected_74_patients.csv")
    coordinates = pd.read_csv(UMAP / "full_94488_umap_coordinates.csv.gz", low_memory=False)
    alignment_manifest = json.loads((OUTPUT / "alignment_manifest.json").read_text())
    umap_manifest = json.loads((UMAP / "umap_manifest.json").read_text())

    if raw.shape != (94_488, 22) or aligned.shape != raw.shape:
        raise AssertionError("Expected matching 94,488 x 22 raw and aligned matrices")
    if len(metadata) != 94_488 or len(coordinates) != 94_488:
        raise AssertionError("Metadata or UMAP coordinates do not cover all cells")
    if len(selected) != 74 or selected.CaseID.nunique() != 74:
        raise AssertionError("Expected all 74 complete-OS patients")
    forbidden = {"OS", "OS_status", "os_event", "survival"}
    if forbidden.intersection(metadata.columns) or forbidden.intersection(selected.columns):
        raise AssertionError("An outcome column entered the alignment inputs")
    if alignment_manifest["canonical_version"] != "2.3.0":
        raise AssertionError("Formal alignment is not canonical RAMP 2.3.0")
    if alignment_manifest["strict_rank_inversions"] != 0:
        raise AssertionError("Strict rank-preservation audit failed")
    if umap_manifest["labels_used_for_embedding"] is not False:
        raise AssertionError("Cell labels were used to fit UMAP")
    if not np.isfinite(raw).all() or not np.isfinite(aligned).all():
        raise AssertionError("Non-finite matrix values found")

    report = {
        "status": "passed",
        "cells": 94_488,
        "markers": 22,
        "datasets": metadata.dataset.value_counts(sort=False).astype(int).to_dict(),
        "units": metadata.groupby("dataset").unit_key.nunique().astype(int).to_dict(),
        "complete_os_patients": 74,
        "outcome_columns_in_alignment_input": False,
        "ramp_version": alignment_manifest["canonical_version"],
        "strict_rank_inversions": alignment_manifest["strict_rank_inversions"],
        "umap_cells": len(coordinates),
        "labels_used_for_embedding": umap_manifest["labels_used_for_embedding"],
        "sha256": {
            "raw": sha256(INPUT / "raw_shared_markers.float64.npy"),
            "aligned": sha256(OUTPUT / "aligned_ramp_ver2.float32.npy"),
            "umap_coordinates": sha256(UMAP / "full_94488_umap_coordinates.csv.gz"),
        },
    }
    (HERE / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
