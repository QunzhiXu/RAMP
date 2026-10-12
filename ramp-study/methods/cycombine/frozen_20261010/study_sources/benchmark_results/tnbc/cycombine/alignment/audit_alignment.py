#!/usr/bin/env python3
"""Audit the completed canonical TNBC cyCombine run without changing it."""

from __future__ import annotations

import gzip
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
METHOD_DIR = HERE.parent
DATA_DIR = HERE / "data"
INPUT_PATH = METHOD_DIR / "preprocessing" / "data" / "combined_analysis_ready.csv.gz"
MARKERS_PATH = METHOD_DIR / "preprocessing" / "markers.txt"
AUDIT_DIR = HERE / "audit"

EXPECTED_CELLS = 62_701
EXPECTED_COUNTS = {
    "Tietscher IMC": 20_000,
    "Meyer IMC": 22_701,
    "Shiao CODEX": 20_000,
}
EXPECTED_MARKERS = [
    "CD11C", "CD20", "CD3", "CD31", "CD4", "CD68",
    "CD8", "HLA_DR", "KI67", "PANCK", "SMA",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv_gz(frame: pd.DataFrame, path: Path) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
            frame.to_csv(stream, index=False)


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = DATA_DIR / "manifest.json"
    complete_path = DATA_DIR / "COMPLETE.json"
    manifest = json.loads(manifest_path.read_text())
    complete = json.loads(complete_path.read_text())
    if complete.get("status") != "complete" or complete.get("manifest_sha256") != sha256(manifest_path):
        raise AssertionError("COMPLETE.json does not authenticate manifest.json")
    for item in manifest["outputs"].values():
        path = DATA_DIR / item["path"]
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise AssertionError(f"Canonical output hash mismatch: {path}")

    markers = [line.strip() for line in MARKERS_PATH.read_text().splitlines() if line.strip()]
    if markers != EXPECTED_MARKERS:
        raise AssertionError("Frozen TNBC shared-marker contract changed")
    inputs = pd.read_csv(INPUT_PATH, dtype={"cell_id": str}, low_memory=False)
    corrected = pd.read_csv(DATA_DIR / "corrected_expression.csv.gz", dtype={"cell_id": str}, low_memory=False)
    som = pd.read_csv(DATA_DIR / "correction_som_assignments.csv.gz", dtype={"cell_id": str})
    metadata = pd.read_csv(DATA_DIR / "cell_metadata_label_free.csv.gz", dtype={"cell_id": str})
    if len(inputs) != EXPECTED_CELLS or inputs["dataset"].value_counts(sort=False).to_dict() != EXPECTED_COUNTS:
        raise AssertionError("Frozen dataset/cell counts changed")
    if not (
        inputs["cell_id"].equals(corrected["cell_id"])
        and inputs["cell_id"].equals(som["cell_id"])
        and inputs["cell_id"].equals(metadata["cell_id"])
    ):
        raise AssertionError("Cell ID/order changed between preprocessing and output")
    prohibited_tokens = ("celltype", "label", "outcome", "survival", "response", "district", "coordinate")
    if not inputs["cell_id"].is_unique or any(
        any(token in column.lower() for token in prohibited_tokens)
        for column in metadata.columns
    ):
        raise AssertionError("Cell identities are duplicated or post-hoc fields entered alignment metadata")
    if list(corrected.columns) != ["cell_id", *EXPECTED_MARKERS]:
        raise AssertionError("Corrected-expression columns changed")
    if not np.isfinite(inputs[markers].to_numpy(np.float64)).all():
        raise AssertionError("Input marker matrix contains non-finite values")
    if not np.isfinite(corrected[markers].to_numpy(np.float64)).all():
        raise AssertionError("Corrected marker matrix contains non-finite values")

    joined = inputs[["cell_id", "dataset", *markers]].copy()
    joined["som_node"] = som["som_node"].to_numpy()
    uniform_rows: list[dict[str, object]] = []
    for (node, dataset), group in joined.groupby(["som_node", "dataset"], sort=True):
        for marker in markers:
            values = group[marker].to_numpy(np.float64)
            if np.all(values == 0):
                uniform_rows.append({
                    "som_node": int(node), "dataset": dataset, "marker": marker,
                    "cells": int(len(values)), "condition": "all_zero",
                })
            elif np.ptp(values) == 0:
                uniform_rows.append({
                    "som_node": int(node), "dataset": dataset, "marker": marker,
                    "cells": int(len(values)), "condition": "uniform_nonzero",
                })
    uniform = pd.DataFrame(
        uniform_rows,
        columns=["som_node", "dataset", "marker", "cells", "condition"],
    )
    uniform_path = AUDIT_DIR / "uniform_node_batch_markers.csv.gz"
    write_csv_gz(uniform, uniform_path)

    audit = {
        "schema_version": "final_xu.tnbc.cycombine_alignment_audit.v1",
        "status": "pass",
        "cells": int(len(inputs)),
        "dataset_cell_counts": EXPECTED_COUNTS,
        "markers": markers,
        "cell_id_and_order_exact": True,
        "label_free_alignment_metadata": True,
        "all_values_finite": True,
        "canonical_output_hashes_verified": True,
        "uniform_node_batch_marker_combinations": int(len(uniform)),
        "uniform_node_batch_marker_file": {
            "path": str(uniform_path), "sha256": sha256(uniform_path),
        },
        "note": (
            "Official ComBat leaves a listed marker unadjusted only within the listed "
            "SOM node/batch group. In this run the listed Shiao values are uniform nonzero "
            "provider-floor values even though the upstream message uses generic all-zero "
            "wording; this is upstream cyCombine behavior, not a wrapper edit."
        ),
        "sources": {
            "combined_input": {"path": str(INPUT_PATH), "sha256": sha256(INPUT_PATH)},
            "run_manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
            "complete_record": {"path": str(complete_path), "sha256": sha256(complete_path)},
        },
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
        "runtime": {
            "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        },
    }
    audit_path = AUDIT_DIR / "alignment_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(f"Alignment audit passed; saved {audit_path}")
    if len(uniform):
        print(uniform.to_string(index=False))


if __name__ == "__main__":
    main()
