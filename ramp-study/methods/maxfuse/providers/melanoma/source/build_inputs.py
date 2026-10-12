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
FINAL_XU = HERE.parents[2]
BENCHMARK = FINAL_XU / "benchmark_data" / "melanoma"
SOURCE_RECIPE = HERE / "rebuild_source_active.py"
SOURCE_ACTIVE = HERE / "source_active"
OUTPUT = HERE / "method_inputs"
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


def publish(stage: Path, output: Path) -> None:
    if not output.exists():
        stage.rename(output)
        return
    backup = output.with_name(f".{output.name}.old-{next(tempfile._get_candidate_names())}")
    output.rename(backup)
    try:
        stage.rename(output)
    except BaseException:
        backup.rename(output)
        raise
    shutil.rmtree(backup)


def ensure_source_active(force: bool) -> dict[str, object]:
    command = [sys.executable, str(SOURCE_RECIPE)]
    if force:
        command.append("--force")
    subprocess.run(command, check=True, cwd=HERE)
    manifest_path = SOURCE_ACTIVE / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("recipe_sha256") != sha256(SOURCE_RECIPE):
        raise AssertionError("source-active recipe/manifest hash differs")
    for relative, expected in manifest.get("outputs", {}).items():
        path = SOURCE_ACTIVE / relative
        if not path.is_file() or sha256(path) != expected:
            raise AssertionError(f"source-active output changed: {relative}")
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


def verified_existing(source_manifest_sha: str) -> bool:
    manifest_path = OUTPUT / "manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("recipe_sha256") != sha256(Path(__file__)):
        return False
    if manifest.get("source_active_manifest_sha256") != source_manifest_sha:
        return False
    for relative, expected in manifest.get("outputs", {}).items():
        path = OUTPUT / relative
        if not path.is_file() or sha256(path) != expected:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="atomically rebuild method inputs")
    parser.add_argument(
        "--rebuild-source", action="store_true",
        help="also force a fresh provider-source extraction",
    )
    args = parser.parse_args()
    source_manifest = ensure_source_active(args.rebuild_source)
    source_manifest_path = SOURCE_ACTIVE / "manifest.json"
    source_manifest_sha = sha256(source_manifest_path)
    if not args.force and verified_existing(source_manifest_sha):
        print(f"verified existing MaxFuse inputs: {OUTPUT}")
        return

    stage = Path(tempfile.mkdtemp(prefix=".method_inputs.stage-", dir=HERE))
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
        publish(stage, OUTPUT)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(f"rebuilt MaxFuse inputs: {OUTPUT}")


if __name__ == "__main__":
    main()
