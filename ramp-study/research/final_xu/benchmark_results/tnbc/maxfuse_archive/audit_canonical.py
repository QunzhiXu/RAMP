#!/usr/bin/env python3
"""Fail-closed audit of the current 62,701-cell TNBC MaxFuse result."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
BENCHMARK_RESULTS = HERE.parents[1]
FINAL_XU = HERE.parents[2]
if str(BENCHMARK_RESULTS) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_RESULTS))

import _maxfuse_benchmark as benchmark


EXPECTED_CELLS = 62_701
EXPECTED_DIMENSIONS = 15
EXPECTED_COUNTS = {
    "Tietscher IMC": 20_000,
    "Meyer IMC": 22_701,
    "Shiao CODEX": 20_000,
}
EXPECTED_HUBS = ("tietscher_imc", "meyer_imc", "shiao_codex")
FIT_METADATA_COLUMNS = ("cell_id", "dataset", "donor", "unit_key")


def verify_hashes(root: Path, records: dict[str, str]) -> None:
    if not records:
        raise AssertionError(f"Missing output hash map: {root}")
    for relative, expected in records.items():
        path = root / relative
        if not path.is_file() or benchmark.sha256(path) != expected:
            raise AssertionError(f"Hash failure: {path}")


def main() -> None:
    config_path = HERE / "config.json"
    config, _ = benchmark.load_config(config_path)
    if config["benchmark_slug"] != "tnbc":
        raise AssertionError("Wrong benchmark slug")
    if config["expected_cells"] != EXPECTED_CELLS:
        raise AssertionError("Wrong total cell count")
    if config["fit_sampling"]["cap_per_dataset"] is not None:
        raise AssertionError("Canonical MaxFuse fit must use all frozen cells")
    observed_config_counts = {
        row["name"]: int(row["expected_cells"]) for row in config["datasets"]
    }
    if observed_config_counts != EXPECTED_COUNTS:
        raise AssertionError(f"Dataset count drift: {observed_config_counts}")

    selected_path = FINAL_XU / "benchmark_data" / "tnbc" / "selected_cells.csv.gz"
    selected = pd.read_csv(selected_path, low_memory=False)
    if len(selected) != EXPECTED_CELLS:
        raise AssertionError("Canonical selected-cell lock has wrong length")
    selected_ids = selected["cell_id"].astype(str).to_numpy()
    selected_counts = selected["dataset"].astype(str).value_counts(sort=False).to_dict()
    if selected_counts != EXPECTED_COUNTS:
        raise AssertionError(f"Selected-cell count drift: {selected_counts}")

    input_root = HERE / "input"
    input_manifest = json.loads((input_root / "input_manifest.json").read_text())
    if (
        input_manifest.get("status") != "complete"
        or input_manifest.get("schema_version")
        != "final_xu.tnbc.maxfuse_archive.inputs.v1"
        or input_manifest.get("cells") != EXPECTED_CELLS
        or input_manifest.get("combined_shared_shape") != [EXPECTED_CELLS, 11]
    ):
        raise AssertionError("Canonical method-input manifest is stale")
    verify_hashes(input_root, input_manifest["output_sha256"])
    for record in input_manifest["source_files"].values():
        path = Path(record["path"])
        if not path.is_file() or benchmark.sha256(path) != record["sha256"]:
            raise AssertionError(f"Input-source hash failure: {path}")

    prepared_root = HERE / "prepared"
    prepared = json.loads((prepared_root / "manifest.json").read_text())
    if (
        prepared.get("status") != "complete"
        or prepared.get("cells") != EXPECTED_CELLS
        or prepared.get("config_sha256") != benchmark.sha256(config_path)
        or prepared.get("labels_loaded_during_preparation") is not False
    ):
        raise AssertionError("Prepared manifest is stale or label-contaminated")
    verify_hashes(prepared_root, prepared["outputs"])
    if not benchmark.verify_source_records(prepared["sources"]):
        raise AssertionError("Prepared-source hash failure")

    reference_ids: np.ndarray | None = None
    hub_rows: list[dict[str, object]] = []
    for hub in EXPECTED_HUBS:
        alignment = HERE / f"hub_{hub}" / "alignment"
        alignment_manifest_path = alignment / "alignment_manifest.json"
        manifest = json.loads(alignment_manifest_path.read_text())
        if (
            manifest.get("schema_version") != benchmark.ALIGNMENT_SCHEMA
            or manifest.get("status") != "complete"
            or manifest.get("full_cells") != EXPECTED_CELLS
            or manifest.get("embedding_dimensions") != EXPECTED_DIMENSIONS
            or manifest.get("config_sha256") != benchmark.sha256(config_path)
            or manifest.get("labels_used_for_fitting") is not False
            or manifest.get("labels_loaded_before_alignment_completed") is not False
        ):
            raise AssertionError(f"{hub}: stale or label-contaminated alignment")
        if manifest.get("fit_cells") != {
            "tietscher_imc": 20_000,
            "meyer_imc": 22_701,
            "shiao_codex": 20_000,
        }:
            raise AssertionError(f"{hub}: wrong fit-cell counts")
        verify_hashes(alignment, manifest["outputs"])

        formal_root = alignment / "formal_fit"
        formal_path = formal_root / "manifest.json"
        formal = json.loads(formal_path.read_text())
        if (
            formal.get("status") != "complete"
            or formal.get("labels_supplied_to_maxfuse") is not False
            or manifest.get("formal_fit_manifest_sha256")
            != benchmark.sha256(formal_path)
        ):
            raise AssertionError(f"{hub}: formal-fit provenance failure")
        verify_hashes(formal_root, formal["output_sha256"])
        audit = formal["audit"]
        if (
            audit.get("gcca_components") != EXPECTED_DIMENSIONS
            or audit.get("gcca_all_components_converged") is not True
            or max(audit["projection_roundtrip_max_abs_error"].values()) > 1e-9
        ):
            raise AssertionError(f"{hub}: GCCA/projection audit failure")

        embedding = np.load(
            alignment / "data" / "maxfuse_embedding_float64.npy",
            allow_pickle=False,
        )
        cell_ids = np.load(alignment / "data" / "cell_ids.npy", allow_pickle=False)
        metadata = pd.read_csv(
            alignment / "data" / "cell_metadata_label_free.csv.gz",
            low_memory=False,
        )
        if embedding.shape != (EXPECTED_CELLS, EXPECTED_DIMENSIONS):
            raise AssertionError(f"{hub}: wrong embedding shape {embedding.shape}")
        if not np.isfinite(embedding).all():
            raise AssertionError(f"{hub}: non-finite embedding")
        if tuple(metadata.columns) != FIT_METADATA_COLUMNS:
            raise AssertionError(f"{hub}: fit metadata contains unexpected columns")
        if not np.array_equal(cell_ids.astype(str), selected_ids):
            raise AssertionError(f"{hub}: cell IDs differ from canonical selected order")
        if not np.array_equal(metadata["cell_id"].astype(str).to_numpy(), selected_ids):
            raise AssertionError(f"{hub}: metadata row order drift")
        if reference_ids is None:
            reference_ids = cell_ids.astype(str)
        elif not np.array_equal(reference_ids, cell_ids.astype(str)):
            raise AssertionError(f"{hub}: cell IDs differ across hubs")

        umap_root = HERE / f"hub_{hub}" / "umap"
        umap_manifest = json.loads((umap_root / "umap_manifest.json").read_text())
        if (
            umap_manifest.get("schema_version") != benchmark.UMAP_SCHEMA
            or umap_manifest.get("status") != "complete"
            or umap_manifest.get("cells") != EXPECTED_CELLS
            or umap_manifest.get("labels_used_for_umap_fitting") is not False
            or umap_manifest.get("labels_loaded_after_umap_fitting") is not True
            or umap_manifest.get("alignment_manifest_sha256")
            != benchmark.sha256(alignment_manifest_path)
        ):
            raise AssertionError(f"{hub}: stale or label-contaminated UMAP")
        verify_hashes(umap_root, umap_manifest["outputs"])
        coordinates = np.load(
            umap_root / "data" / "umap_coordinates.npy", allow_pickle=False
        )
        if coordinates.shape != (EXPECTED_CELLS, 2) or not np.isfinite(coordinates).all():
            raise AssertionError(f"{hub}: invalid UMAP coordinates")

        hub_rows.append({
            "hub": hub,
            "cells": EXPECTED_CELLS,
            "embedding_dimensions": EXPECTED_DIMENSIONS,
            "triplets": int(audit["triplets"]),
            "gcca_all_components_converged": True,
            "projection_roundtrip_max_abs_error": float(
                max(audit["projection_roundtrip_max_abs_error"].values())
            ),
            "alignment_hashes": "PASS",
            "umap_hashes": "PASS",
        })

    summary_root = HERE / "summary"
    summary = json.loads((summary_root / "summary_manifest.json").read_text())
    if (
        summary.get("schema_version") != "final_xu.maxfuse_all_hubs_summary.v2"
        or summary.get("status") != "complete"
        or summary.get("config_sha256") != benchmark.sha256(config_path)
        or summary.get("labels_used_for_alignment_or_umap_fitting") is not False
    ):
        raise AssertionError("Summary manifest is stale")
    verify_hashes(summary_root, summary["outputs"])

    result = {
        "schema_version": "final_xu.tnbc.maxfuse_archive.canonical_audit.v1",
        "status": "PASS",
        "cells": EXPECTED_CELLS,
        "dataset_counts": EXPECTED_COUNTS,
        "shared_markers": 11,
        "active_widths": {
            "tietscher_imc": 40,
            "meyer_imc": 36,
            "shiao_codex": 39,
        },
        "config_sha256": benchmark.sha256(config_path),
        "selected_cells_sha256": benchmark.sha256(selected_path),
        "hubs": hub_rows,
        "old_65211_result_reused": False,
        "labels_used_for_alignment_or_umap_fitting": False,
    }
    output = HERE / "canonical_audit.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"TNBC_MAXFUSE_CANONICAL_AUDIT_PASS hubs={len(hub_rows)} cells={EXPECTED_CELLS}")
    print(output)


if __name__ == "__main__":
    main()
