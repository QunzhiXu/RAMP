#!/usr/bin/env python3
"""Build MaxFuse-ready melanoma inputs from provider/source active panels."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
BENCHMARK = None
from melanoma_support import claim, publish_new, fail, check_inputs, verify_reference
SOURCE_RECIPE = HERE / "rebuild_source_active.py"
SOURCE_ACTIVE = OUTPUT = None
SHARED = [
    "CD163", "CD3", "CD31", "CD68", "CD8", "HLA_DR",
    "LAG3", "PD1", "PDL1", "SMA", "SOX10", "VIMENTIN",
]
DATASETS = [
    (
        "MEL_Hickey2023_CODEX", "hickey_codex",
        "provider tissue-wise marker-z-normalized CODEX values",
    ),
    (
        "MEL_Hoch2021_IMC", "hoch_imc",
        "provider exprs assay: asinh(counts / 1)",
    ),
    (
        "MEL_Ptacek2021_MIBI", "ptacek_mibi",
        "provider denoised/normalized/scaled/arcsinh MIBI values",
    ),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def buffer_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(values, dtype=np.float64).tobytes()
    ).hexdigest()


def deterministic_csv(frame: pd.DataFrame, path: Path) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=6, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
                frame.to_csv(text, index=False, float_format="%.17g")


def zscore(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    center = values.mean(axis=0)
    scale = values.std(axis=0, ddof=1)
    if np.any(~np.isfinite(scale)) or np.any(scale < 1e-10):
        raise AssertionError("non-informative active feature before MaxFuse z-score")
    transformed = (values - center) / scale
    if not np.isfinite(transformed).all():
        raise AssertionError("MaxFuse z-score produced non-finite values")
    return transformed, center, scale


def ensure_source_active() -> dict[str, object]:
    """Verify the explicit completed rebuild; never silently use a legacy cache."""
    manifest = verify_reference(SOURCE_ACTIVE, "source")
    complete = json.loads((SOURCE_ACTIVE / "COMPLETE.json").read_text())
    if complete.get("status") != "complete" or complete.get("stage") != "source" or complete.get("manifest_sha256") != sha256(SOURCE_ACTIVE / "manifest.json"):
        raise AssertionError("source-active completion/manifest differs")
    if manifest.get("recipe_sha256") != sha256(SOURCE_RECIPE):
        raise AssertionError("source-active recipe/manifest hash differs")
    return manifest


def canonical_metadata(dataset: str, slug: str) -> pd.DataFrame:
    path = BENCHMARK / slug / "alignment_input_12.csv.gz"
    required = ["cell_id", "dataset", "donor", "unit_key"]
    frame = pd.read_csv(path, usecols=required, dtype={"cell_id": str}, low_memory=False)
    if len(frame) != 20_000 or frame["dataset"].astype(str).unique().tolist() != [dataset]:
        raise AssertionError(f"{slug}: canonical benchmark contract changed")
    if frame["cell_id"].duplicated().any():
        raise AssertionError(f"{slug}: canonical cell IDs are not unique")
    return frame


def build_dataset(
    dataset: str, slug: str, source_transform: str, stage: Path
) -> dict[str, object]:
    metadata = canonical_metadata(dataset, slug)
    ids_path = SOURCE_ACTIVE / slug / "cell_ids.csv.gz"
    ids = pd.read_csv(
        ids_path,
        usecols=["cell_id", "source_cell_id"],
        dtype={"cell_id": str, "source_cell_id": str},
    )
    if len(ids) != 20_000 or ids["source_cell_id"].duplicated().any():
        raise AssertionError(f"{slug}: invalid source-active identity lock")
    if not np.array_equal(ids["cell_id"].to_numpy(), metadata["cell_id"].to_numpy()):
        raise AssertionError(f"{slug}: source-active/canonical row identities differ")

    source_path = SOURCE_ACTIVE / slug / "active_source_float64.npy"
    marker_path = SOURCE_ACTIVE / slug / "active_markers.txt"
    source = np.load(source_path, allow_pickle=False).astype(np.float64, copy=False)
    markers = [line for line in marker_path.read_text().splitlines() if line]
    if source.shape != (20_000, len(markers)) or len(set(markers)) != len(markers):
        raise AssertionError(f"{slug}: invalid source-active panel")
    if any(marker not in markers for marker in SHARED) or not np.isfinite(source).all():
        raise AssertionError(f"{slug}: source-active panel lacks shared features or is non-finite")

    processed, center, scale = zscore(source)
    shared = processed[:, [markers.index(marker) for marker in SHARED]]
    active_frame = pd.DataFrame(processed, columns=markers)
    active_frame.insert(0, "cell_id", metadata["cell_id"].to_numpy())
    shared_frame = metadata.copy()
    for index, marker in enumerate(SHARED):
        shared_frame[marker] = shared[:, index]
    active_path = stage / f"{slug}_active.csv.gz"
    shared_path = stage / f"{slug}_shared.csv.gz"
    deterministic_csv(active_frame, active_path)
    deterministic_csv(shared_frame, shared_path)

    active_check = pd.read_csv(active_path, low_memory=False)
    shared_check = pd.read_csv(shared_path, low_memory=False)
    if not np.array_equal(
        active_check["cell_id"].astype(str).to_numpy(),
        shared_check["cell_id"].astype(str).to_numpy(),
    ):
        raise AssertionError(f"{slug}: serialized active/shared IDs differ")
    serialized_error = float(np.max(np.abs(
        active_check[SHARED].to_numpy(np.float64)
        - shared_check[SHARED].to_numpy(np.float64)
    )))
    if serialized_error != 0.0:
        raise AssertionError(f"{slug}: shared columns are not an exact active slice")
    mean_error = float(np.max(np.abs(processed.mean(axis=0))))
    sample_sd_error = float(np.max(np.abs(processed.std(axis=0, ddof=1) - 1.0)))
    if mean_error > 1e-12 or sample_sd_error > 1e-12:
        raise AssertionError(f"{slug}: method-level sample-SD z-score audit failed")
    return {
        "dataset": dataset,
        "slug": slug,
        "cells": 20_000,
        "active_features": len(markers),
        "active_markers": markers,
        "shared_markers": SHARED,
        "provider_or_paper_transform_before_method_zscore": source_transform,
        "method_level_transform": "within-dataset marker z-score using sample SD (ddof=1)",
        "zscore_center": center.tolist(),
        "zscore_sample_sd": scale.tolist(),
        "source_buffer_sha256_float64": buffer_sha256(source),
        "processed_buffer_sha256_float64": buffer_sha256(processed),
        "processed_max_abs_feature_mean": mean_error,
        "processed_max_abs_feature_sample_sd_minus_one": sample_sd_error,
        "serialized_shared_active_max_abs_error": serialized_error,
        "source_files": {
            "active_source": sha256(source_path),
            "active_markers": sha256(marker_path),
            "cell_ids": sha256(ids_path),
        },
        "outputs": {
            active_path.name: sha256(active_path),
            shared_path.name: sha256(shared_path),
        },
    }


def main() -> None:
    global BENCHMARK, SOURCE_ACTIVE, OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--source-active", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    BENCHMARK, SOURCE_ACTIVE = args.benchmark_root.resolve(strict=True), args.source_active.resolve(strict=True)
    check_inputs(None, BENCHMARK)
    source_manifest = ensure_source_active()
    source_manifest_path = SOURCE_ACTIVE / "manifest.json"
    source_manifest_sha = sha256(source_manifest_path)
    OUTPUT = claim(args.output, "method", BENCHMARK, source_active=SOURCE_ACTIVE)
    stage = OUTPUT / "_work"
    stage.mkdir(exist_ok=False)
    try:
        records = [build_dataset(*spec, stage) for spec in DATASETS]
        outputs = {
            path.name: sha256(path) for path in stage.iterdir() if path.is_file()
        }
        manifest = {
            "schema_version": "final_xu.maxfuse_method_inputs.v2",
            "status": "complete",
            "benchmark": "melanoma",
            "selection": "exact frozen 20,000 cells per dataset; no resampling",
            "labels_or_outcomes_read": False,
            "shared_from_processed_active_columns": True,
            "all_active_features_method_zscored": True,
            "zscore_standard_deviation": "sample SD (ddof=1)",
            "datasets": records,
            "source_active_manifest": str(source_manifest_path.resolve()),
            "source_active_manifest_sha256": source_manifest_sha,
            "source_active_recipe": str(SOURCE_RECIPE.resolve()),
            "source_active_recipe_sha256": sha256(SOURCE_RECIPE),
            "source_active_schema": source_manifest.get("schema_version"),
            "recipe": str(Path(__file__).resolve()),
            "recipe_sha256": sha256(Path(__file__)),
            "outputs": outputs,
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        check_inputs(None, BENCHMARK)
        if sha256(source_manifest_path) != source_manifest_sha:
            raise AssertionError("source-active manifest changed during scaling")
        ensure_source_active()
        publish_new(stage, OUTPUT, "method")
    except BaseException as error:
        fail(OUTPUT, error)
        raise
    print(f"rebuilt MaxFuse inputs: {OUTPUT}")


if __name__ == "__main__":
    main()
