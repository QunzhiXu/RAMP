#!/usr/bin/env python3
"""Validate lung source extraction, method scaling, row locks, and hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
FINAL_XU = HERE.parents[2]
BENCHMARK = FINAL_XU / "benchmark_data" / "lung_cancer"
SOURCE = HERE / "source_active"
METHOD = HERE / "method_inputs"
CONFIG = HERE / "config.json"
SOURCE_RECIPE = HERE / "rebuild_source_active.py"
METHOD_RECIPE = HERE / "build_inputs.py"
SHARED = ["CD3", "CD4", "CD8", "CD31", "HLA_DR", "KI67", "VIMENTIN", "EPI"]


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


def verify_files(root: Path, mapping: dict[str, str], title: str) -> None:
    for relative, expected in mapping.items():
        path = root / relative
        if not path.is_file() or sha256(path) != expected:
            raise AssertionError(f"{title} hash mismatch: {relative}")


def verify_identities(value: object, deep: bool) -> None:
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            path = Path(str(value["path"]))
            if not path.is_file():
                raise FileNotFoundError(path)
            if deep and sha256(path) != value["sha256"]:
                raise AssertionError(f"source/canonical file hash mismatch: {path}")
        for child in value.values():
            verify_identities(child, deep)
    elif isinstance(value, list):
        for child in value:
            verify_identities(child, deep)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick", action="store_true",
        help="skip re-hashing multi-GB provider sources; derived hashes are always checked",
    )
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    source_manifest = json.loads((SOURCE / "manifest.json").read_text())
    method_manifest = json.loads((METHOD / "manifest.json").read_text())
    if source_manifest.get("recipe_sha256") != sha256(SOURCE_RECIPE):
        raise AssertionError("source recipe hash mismatch")
    if method_manifest.get("recipe_sha256") != sha256(METHOD_RECIPE):
        raise AssertionError("method recipe hash mismatch")
    if method_manifest.get("source_active_manifest_sha256") != sha256(SOURCE / "manifest.json"):
        raise AssertionError("method inputs do not lock the current source-active manifest")
    if not source_manifest.get("labels_or_outcomes_read") is False:
        raise AssertionError("source extraction label policy is not locked false")
    if not method_manifest.get("labels_or_outcomes_read") is False:
        raise AssertionError("method input label policy is not locked false")
    if not method_manifest.get("all_active_features_method_zscored") is True:
        raise AssertionError("not all active features are declared method-z-scored")
    verify_files(SOURCE, source_manifest["outputs"], "source-active output")
    verify_files(METHOD, method_manifest["outputs"], "method-input output")
    verify_identities(source_manifest.get("canonical_inputs", {}), not args.quick)
    verify_identities(source_manifest.get("datasets", []), not args.quick)

    source_by_slug = {record["slug"]: record for record in source_manifest["datasets"]}
    method_by_slug = {record["slug"]: record for record in method_manifest["datasets"]}
    summaries = []
    for spec in config["datasets"]:
        slug = spec["slug"]
        markers = spec["active_markers"]
        source_record = source_by_slug[slug]
        method_record = method_by_slug[slug]
        if source_record["markers"] != markers or method_record["active_markers"] != markers:
            raise AssertionError(f"{slug}: config/source/method marker order differs")
        if method_record["shared_markers"] != SHARED:
            raise AssertionError(f"{slug}: shared marker order differs")

        source_values = np.load(
            SOURCE / slug / "active_source_float64.npy", allow_pickle=False
        ).astype(np.float64, copy=False)
        if source_values.shape != (20_000, len(markers)):
            raise AssertionError(f"{slug}: source-active shape differs")
        if buffer_sha256(source_values) != source_record["source_buffer_sha256_float64"]:
            raise AssertionError(f"{slug}: source-active buffer hash differs")
        source_ids = pd.read_csv(
            SOURCE / slug / "cell_ids.csv.gz", dtype=str
        )["cell_id"].to_numpy()
        canonical_ids = pd.read_csv(
            BENCHMARK / slug / "alignment_input_8.csv.gz",
            usecols=["cell_id"], dtype=str,
        )["cell_id"].to_numpy()
        if not np.array_equal(source_ids, canonical_ids):
            raise AssertionError(f"{slug}: source/canonical IDs or order differ")

        active = pd.read_csv(
            METHOD / f"{slug}_active.csv.gz", low_memory=False,
            float_precision="round_trip",
        )
        shared = pd.read_csv(
            METHOD / f"{slug}_shared.csv.gz", low_memory=False,
            float_precision="round_trip",
        )
        if active.columns.tolist() != ["cell_id", *markers]:
            raise AssertionError(f"{slug}: serialized active marker order differs")
        if not np.array_equal(active["cell_id"].astype(str).to_numpy(), canonical_ids):
            raise AssertionError(f"{slug}: method/canonical IDs or order differ")
        if not np.array_equal(shared["cell_id"].astype(str).to_numpy(), canonical_ids):
            raise AssertionError(f"{slug}: shared/canonical IDs or order differ")
        values = active[markers].to_numpy(np.float64)
        exact_error = float(np.max(np.abs(
            values[:, [markers.index(marker) for marker in SHARED]]
            - shared[SHARED].to_numpy(np.float64)
        )))
        mean_error = float(np.max(np.abs(values.mean(axis=0))))
        sd_error = float(np.max(np.abs(values.std(axis=0, ddof=1) - 1.0)))
        if exact_error != 0.0 or mean_error > 1e-12 or sd_error > 1e-12:
            raise AssertionError(f"{slug}: exact-slice or sample-SD z-score audit failed")
        if buffer_sha256(values) != method_record["processed_buffer_sha256_float64"]:
            raise AssertionError(f"{slug}: processed buffer hash differs")
        summaries.append((slug, len(values), len(markers), mean_error, sd_error))

    print("VALID: lung MaxFuse source and method inputs")
    for slug, cells, features, mean_error, sd_error in summaries:
        print(
            f"  {slug}: {cells:,} cells x {features}; "
            f"max|mean|={mean_error:.3g}; max|sampleSD-1|={sd_error:.3g}; shared_error=0"
        )
    print(f"  deep_provider_hashes={'no' if args.quick else 'yes'}")


if __name__ == "__main__":
    main()
