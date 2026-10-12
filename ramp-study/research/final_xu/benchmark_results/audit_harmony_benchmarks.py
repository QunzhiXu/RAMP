#!/usr/bin/env python3
"""Fail-closed audit and aggregate the four completed Harmony benchmarks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


HERE = Path(__file__).resolve().parent
BENCHMARKS = ("bone_marrow", "tnbc", "melanoma", "lung_cancer")
VARIANTS = ("source_scale", "adtnorm_fixed_arcsinh")
METHOD = "Harmony-direct (ADTnorm protocol; no PCA)"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_hashed_outputs(directory: Path, mapping: dict[str, str]) -> None:
    for name, expected in mapping.items():
        path = directory / name
        require(path.is_file(), f"Missing output: {path}")
        require(sha256(path) == expected, f"Output hash mismatch: {path}")


def main() -> None:
    combined: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    common_algorithm: tuple[str, str] | None = None
    for benchmark in BENCHMARKS:
        method_dir = HERE / benchmark / "harmony"
        config = load_json(method_dir / "config.json")
        require(config["benchmark"] == benchmark, f"Config benchmark mismatch: {benchmark}")
        expected_cells = int(config["expected"]["cells"])
        expected_markers = config["expected"]["markers"]
        summary = pd.read_csv(method_dir / "harmony_variant_summary.csv")
        require(summary["variant"].tolist() == list(VARIANTS), f"Summary variants changed: {benchmark}")
        combined.append(summary)

        reference_ids: pd.Series | None = None
        reference_batches: pd.DataFrame | None = None
        reference_parameters: dict[str, Any] | None = None
        reference_initialization: dict[str, Any] | None = None
        for variant in VARIANTS:
            variant_dir = method_dir / variant
            preprocessing_dir = variant_dir / "preprocessing"
            alignment_dir = variant_dir / "alignment"
            alignment_data = alignment_dir / "data"
            umap_dir = variant_dir / "umap"

            pre = load_json(preprocessing_dir / "manifest.json")
            require(pre["status"] == "complete" and pre["cells"] == expected_cells, f"Preprocessing incomplete: {benchmark}/{variant}")
            require(pre["markers"] == expected_markers and pre["labels_used"] is False, f"Preprocessing contract changed: {benchmark}/{variant}")
            require(pre["deterministic_csv_roundtrip_exact"] is True, f"CSV roundtrip failed: {benchmark}/{variant}")
            verify_hashed_outputs(preprocessing_dir / "data", {"combined_analysis_ready.csv.gz": pre["outputs"]["combined_analysis_ready.csv.gz"]})
            verify_hashed_outputs(preprocessing_dir, {
                "markers.txt": pre["outputs"]["markers.txt"],
                "preprocessing_record.txt": pre["outputs"]["preprocessing_record.txt"],
                "source_to_method_input_rank_audit.csv": pre["outputs"]["source_to_method_input_rank_audit.csv"],
            })
            pre_rank = pd.read_csv(preprocessing_dir / "source_to_method_input_rank_audit.csv")
            if "adjacent_order_inversions" in pre_rank:
                inversion_count = int(pre_rank["adjacent_order_inversions"].sum())
                new_tie_count = int(pre_rank["new_adjacent_ties_from_finite_precision"].sum())
            else:
                # Early completed identity/exact-rank outputs predate the two
                # more granular columns. Exact equality proves both are zero.
                require(bool(pre_rank["rank_vectors_exactly_equal"].all()), f"Legacy preprocessing audit is not exact: {benchmark}/{variant}")
                inversion_count = 0
                new_tie_count = 0
            require(inversion_count == 0, f"Preprocessing inversion: {benchmark}/{variant}")

            input_path = preprocessing_dir / "data" / "combined_analysis_ready.csv.gz"
            input_header = pd.read_csv(input_path, nrows=0).columns.tolist()
            require(input_header == ["cell_id", "dataset", "unit_key", *expected_markers], f"Unexpected fitting columns: {benchmark}/{variant}")
            identities = pd.read_csv(input_path, usecols=["cell_id", "dataset", "unit_key"], dtype=str)
            require(len(identities) == expected_cells and identities["cell_id"].is_unique, f"Cell identity failure: {benchmark}/{variant}")
            if reference_ids is None:
                reference_ids = identities["cell_id"]
                reference_batches = identities[["dataset", "unit_key"]]
            else:
                require(reference_ids.equals(identities["cell_id"]), f"Variant cell order differs: {benchmark}")
                require(reference_batches.equals(identities[["dataset", "unit_key"]]), f"Variant batch/unit differs: {benchmark}")

            resolved = load_json(alignment_data / "resolved_config.json")
            require(resolved["method_label"] == METHOD, f"Method label changed: {benchmark}/{variant}")
            require(resolved["batch_levels"] == config["expected"]["dataset_order"], f"Batch order changed: {benchmark}/{variant}")
            require(resolved["input_scale"]["algorithm_applies_transformation"] is False, f"Hidden algorithm transform: {benchmark}/{variant}")
            if reference_parameters is None:
                reference_parameters = resolved["algorithm_parameters"]
            else:
                require(reference_parameters == resolved["algorithm_parameters"], f"Algorithm parameters differ between variants: {benchmark}")

            alignment_manifest = load_json(alignment_data / "manifest.json")
            algorithm_identity = (alignment_manifest["algorithm"]["version"], alignment_manifest["algorithm"]["source_sha256"])
            if common_algorithm is None:
                common_algorithm = algorithm_identity
            else:
                require(common_algorithm == algorithm_identity, "Harmony package/source differs across benchmarks")
            require(alignment_manifest["input"]["n_cells"] == expected_cells, f"Alignment cells changed: {benchmark}/{variant}")
            require(alignment_manifest["input"]["markers"] == expected_markers, f"Alignment markers changed: {benchmark}/{variant}")
            require(alignment_manifest["input"]["sha256"] == sha256(input_path), f"Alignment input hash mismatch: {benchmark}/{variant}")
            for item in alignment_manifest["output_files"]:
                verify_hashed_outputs(alignment_data, {item["file"]: item["sha256"]})

            umap_manifest = load_json(umap_dir / "umap_manifest.json")
            require(umap_manifest["labels_used_for_alignment"] is False, f"Labels entered alignment: {benchmark}/{variant}")
            require(umap_manifest["labels_used_for_umap_fitting"] is False, f"Labels entered UMAP: {benchmark}/{variant}")
            require(umap_manifest["labels_loaded_after_umap_fitting"] is True, f"Posthoc label contract failed: {benchmark}/{variant}")
            require(umap_manifest["pre_umap_processing"] == "pooled StandardScaler; no PCA", f"UMAP processing changed: {benchmark}/{variant}")
            if reference_initialization is None:
                reference_initialization = umap_manifest["initialization"]
            else:
                require(reference_initialization == umap_manifest["initialization"], f"UMAP initialization differs between variants: {benchmark}")
            verify_hashed_outputs(umap_dir / "data", {
                "umap_coordinates.csv.gz": umap_manifest["outputs"]["umap_coordinates.csv.gz"],
                "umap_coordinates.npy": umap_manifest["outputs"]["umap_coordinates.npy"],
            })
            figure_names = [name for name in umap_manifest["outputs"] if name.endswith((".png", ".pdf"))]
            verify_hashed_outputs(umap_dir / "figures", {name: umap_manifest["outputs"][name] for name in figure_names})

            audit_rows.append({
                "benchmark": benchmark,
                "variant": variant,
                "cells": expected_cells,
                "markers": len(expected_markers),
                "preprocessing_order_inversions": inversion_count,
                "preprocessing_new_adjacent_ties": new_tie_count,
                "labels_used_for_alignment_or_umap": False,
                "frozen_cell_marker_batch_contract_passed": True,
                "output_hashes_passed": True,
            })

    combined_frame = pd.concat(combined, ignore_index=True)
    combined_path = HERE / "harmony_benchmark_summary.csv"
    combined_frame.to_csv(combined_path, index=False)
    audit_frame = pd.DataFrame(audit_rows)
    audit_csv = HERE / "harmony_reproducibility_audit.csv"
    audit_frame.to_csv(audit_csv, index=False)
    result = {
        "schema_version": "final_xu.harmony_benchmark_global_audit.v1",
        "status": "pass",
        "benchmarks": list(BENCHMARKS),
        "variants": list(VARIANTS),
        "method": METHOD,
        "harmony_package_version": common_algorithm[0] if common_algorithm else None,
        "harmony_source_sha256": common_algorithm[1] if common_algorithm else None,
        "checks": {
            "same_frozen_cells_markers_batches_across_variants": True,
            "same_locked_algorithm_parameters_across_variants": True,
            "same_matched_umap_initialization_across_variants": True,
            "labels_absent_from_alignment_and_umap_fitting": True,
            "no_source_to_method_input_order_inversions": True,
            "all_recorded_output_hashes_verified": True,
        },
        "finite_precision_tie_note": "The official asinh(1 + x/5) sensitivity can merge extremely close floating-point values into ties; these are audited separately and no order inversion is allowed.",
        "outputs": {
            combined_path.name: sha256(combined_path),
            audit_csv.name: sha256(audit_csv),
        },
    }
    output_json = HERE / "harmony_reproducibility_audit.json"
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"PASS: audited {len(BENCHMARKS)} benchmarks x {len(VARIANTS)} variants")
    print(audit_frame.to_string(index=False))


if __name__ == "__main__":
    main()
