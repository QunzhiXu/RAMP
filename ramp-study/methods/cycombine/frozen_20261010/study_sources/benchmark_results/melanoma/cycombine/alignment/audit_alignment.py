#!/usr/bin/env python3
"""Audit the completed canonical melanoma cyCombine run without changing it."""

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
DATASET_ORDER = [
    "MEL_Hickey2023_CODEX",
    "MEL_Hoch2021_IMC",
    "MEL_Ptacek2021_MIBI",
]
EXPECTED_COUNTS = {dataset: 20_000 for dataset in DATASET_ORDER}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv_gz(frame: pd.DataFrame, path: Path) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped:
            frame.to_csv(zipped, index=False)


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
    expected_markers = [
        "CD163", "CD3", "CD31", "CD68", "CD8", "HLA_DR",
        "LAG3", "PD1", "PDL1", "SMA", "SOX10", "VIMENTIN",
    ]
    if markers != expected_markers:
        raise AssertionError("Frozen melanoma shared-marker contract changed")

    inputs = pd.read_csv(INPUT_PATH, dtype={"cell_id": str, "dataset": str}, low_memory=False)
    corrected = pd.read_csv(DATA_DIR / "corrected_expression.csv.gz", dtype={"cell_id": str})
    som = pd.read_csv(DATA_DIR / "correction_som_assignments.csv.gz", dtype={"cell_id": str})
    metadata = pd.read_csv(DATA_DIR / "cell_metadata_label_free.csv.gz", dtype={"cell_id": str})
    observed_counts = inputs.groupby("dataset", sort=False).size().astype(int).to_dict()
    if len(inputs) != 60_000 or observed_counts != EXPECTED_COUNTS:
        raise AssertionError("Frozen melanoma dataset/cell counts changed")
    if list(pd.unique(inputs["dataset"])) != DATASET_ORDER:
        raise AssertionError("Frozen melanoma dataset block order changed")
    if not (
        inputs["cell_id"].equals(corrected["cell_id"])
        and inputs["cell_id"].equals(som["cell_id"])
        and inputs["cell_id"].equals(metadata["cell_id"])
    ):
        raise AssertionError("Cell ID/order changed between preprocessing and output")
    if not inputs["cell_id"].is_unique:
        raise AssertionError("Cell identities are duplicated")
    forbidden_fragments = (
        "celltype", "response", "survival", "outcome", "stage", "timepoint",
        "coordinate", "centroid", "spatial", "x_pos", "y_pos",
    )
    if any(any(fragment in column.lower() for fragment in forbidden_fragments) for column in metadata.columns):
        raise AssertionError("Biological/clinical/spatial metadata entered canonical alignment metadata")
    if not np.isfinite(inputs[markers].to_numpy(np.float64)).all():
        raise AssertionError("Input marker matrix contains non-finite values")
    if not np.isfinite(corrected[markers].to_numpy(np.float64)).all():
        raise AssertionError("Corrected marker matrix contains non-finite values")
    if corrected.columns.tolist() != ["cell_id", *markers]:
        raise AssertionError("Corrected-expression columns changed")

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

    modes = pd.read_csv(DATA_DIR / "correction_som_node_modes.csv")
    mode_counts = modes["correction_mode"].value_counts(sort=False).astype(int).to_dict()
    audit = {
        "schema_version": "final_xu.melanoma.cycombine_alignment_audit.v1",
        "status": "pass",
        "cells": int(len(inputs)),
        "dataset_order": DATASET_ORDER,
        "dataset_cell_counts": EXPECTED_COUNTS,
        "markers": markers,
        "cell_id_and_order_exact": True,
        "label_outcome_and_spatial_metadata_absent": True,
        "all_values_finite": True,
        "canonical_output_hashes_verified": True,
        "correction_node_mode_counts": mode_counts,
        "uniform_node_batch_marker_combinations": int(len(uniform)),
        "uniform_node_batch_marker_file": {
            "path": str(uniform_path), "sha256": sha256(uniform_path),
        },
        "note": (
            "Official ComBat may leave a listed marker unadjusted within a listed "
            "small or uniform SOM node/batch group; this records upstream cyCombine "
            "behavior and does not edit the canonical output."
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
    print(f"Uniform node/batch/marker combinations: {len(uniform)}")
    print(f"Correction node modes: {mode_counts}")


if __name__ == "__main__":
    main()
