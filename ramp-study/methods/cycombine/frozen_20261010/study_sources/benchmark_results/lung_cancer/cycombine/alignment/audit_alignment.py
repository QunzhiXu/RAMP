#!/usr/bin/env python3
"""Audit the canonical lung-cancer cyCombine run without changing it."""

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
PREPROCESSING_MANIFEST = METHOD_DIR / "preprocessing" / "preprocessing_manifest.json"
AUDIT_DIR = HERE / "audit"
DATASETS = ["Senosain CyTOF", "Cords IMC", "Monkman CODEX"]
MARKERS = ["CD3", "CD4", "CD8", "CD31", "HLA_DR", "KI67", "VIMENTIN", "EPI"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv_gz(frame: pd.DataFrame, path: Path) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as stream:
            with pd.io.common.get_handle(stream, "w", encoding="utf-8", is_text=True) as handle:
                frame.to_csv(handle.handle, index=False)


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

    markers = [line for line in MARKERS_PATH.read_text().splitlines() if line]
    if markers != MARKERS:
        raise AssertionError("Frozen shared-marker contract changed")
    inputs = pd.read_csv(INPUT_PATH, dtype={"cell_id": str}, low_memory=False)
    corrected = pd.read_csv(DATA_DIR / "corrected_expression.csv.gz", dtype={"cell_id": str}, low_memory=False)
    som = pd.read_csv(DATA_DIR / "correction_som_assignments.csv.gz", dtype={"cell_id": str})
    metadata = pd.read_csv(DATA_DIR / "cell_metadata_label_free.csv.gz", dtype={"cell_id": str})
    expected_counts = {dataset: 20_000 for dataset in DATASETS}
    observed_counts = inputs["dataset"].value_counts(sort=False).to_dict()
    if len(inputs) != 60_000 or observed_counts != expected_counts:
        raise AssertionError(f"Frozen dataset/cell counts changed: {observed_counts}")
    if list(pd.unique(inputs["dataset"].astype(str))) != DATASETS:
        raise AssertionError("Dataset block order changed")
    if not (
        inputs["cell_id"].equals(corrected["cell_id"])
        and inputs["cell_id"].equals(som["cell_id"])
        and inputs["cell_id"].equals(metadata["cell_id"])
    ):
        raise AssertionError("Cell ID/order changed between preprocessing and output")
    if not inputs["cell_id"].is_unique:
        raise AssertionError("Cell IDs are duplicated")
    forbidden = ("celltype", "label", "survival", "outcome", "response", "spatial", "x", "y")
    lower_metadata = {column.lower() for column in metadata.columns}
    if any(name in lower_metadata for name in forbidden):
        raise AssertionError(f"Forbidden metadata entered alignment output: {lower_metadata}")
    if list(corrected.columns) != ["cell_id", *MARKERS]:
        raise AssertionError("Corrected-expression columns changed")
    if not np.isfinite(inputs[MARKERS].to_numpy(np.float64)).all():
        raise AssertionError("Input marker matrix contains non-finite values")
    if not np.isfinite(corrected[MARKERS].to_numpy(np.float64)).all():
        raise AssertionError("Corrected marker matrix contains non-finite values")
    preprocessing = json.loads(PREPROCESSING_MANIFEST.read_text())
    if preprocessing.get("schema_version") != "final_xu.cycombine_preprocessing.v2":
        raise AssertionError("Preprocessing manifest is not v2")
    if preprocessing.get("combined_input_sha256") != sha256(INPUT_PATH):
        raise AssertionError("Preprocessing manifest does not authenticate combined input")

    joined = inputs[["cell_id", "dataset", *MARKERS]].copy()
    joined["som_node"] = som["som_node"].to_numpy()
    uniform_rows: list[dict[str, object]] = []
    for (node, dataset), group in joined.groupby(["som_node", "dataset"], sort=True):
        for marker in MARKERS:
            values = group[marker].to_numpy(np.float64)
            if np.all(values == 0):
                condition = "all_zero"
            elif np.ptp(values) == 0:
                condition = "uniform_nonzero"
            else:
                continue
            uniform_rows.append({
                "som_node": int(node), "dataset": dataset, "marker": marker,
                "cells": int(len(values)), "condition": condition,
            })
    uniform = pd.DataFrame(
        uniform_rows,
        columns=["som_node", "dataset", "marker", "cells", "condition"],
    )
    uniform_path = AUDIT_DIR / "uniform_node_batch_markers.csv.gz"
    with uniform_path.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as stream:
            uniform.to_csv(stream, index=False)

    audit = {
        "schema_version": "final_xu.lung_cancer.cycombine_alignment_audit.v1",
        "status": "pass",
        "cells": int(len(inputs)),
        "dataset_cell_counts": expected_counts,
        "dataset_order": DATASETS,
        "markers": MARKERS,
        "cell_id_and_order_exact": True,
        "label_outcome_spatial_free_alignment_metadata": True,
        "all_values_finite": True,
        "canonical_output_hashes_verified": True,
        "preprocessing_manifest_v2_verified": True,
        "uniform_node_batch_marker_combinations": int(len(uniform)),
        "uniform_node_batch_marker_file": {
            "path": str(uniform_path), "sha256": sha256(uniform_path),
        },
        "note": (
            "Official ComBat leaves any listed marker unadjusted only within the listed "
            "small SOM node/batch group; this is upstream cyCombine behavior, not a wrapper edit."
        ),
        "sources": {
            "combined_input": {"path": str(INPUT_PATH), "sha256": sha256(INPUT_PATH)},
            "preprocessing_manifest": {"path": str(PREPROCESSING_MANIFEST), "sha256": sha256(PREPROCESSING_MANIFEST)},
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

