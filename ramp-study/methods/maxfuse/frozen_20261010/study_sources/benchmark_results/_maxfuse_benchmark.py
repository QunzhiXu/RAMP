#!/usr/bin/env python3
"""Reproducible three-hub benchmark for the MaxFuse Archive workflow.

Alignment is label-free.  Display labels are opened only by the UMAP command,
after every aligned coordinate matrix has already been written.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import gzip
import hashlib
from importlib.metadata import version
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any


HERE = Path(__file__).resolve().parent
FINAL_XU = HERE.parent
ALGORITHM_DIR = FINAL_XU / "algorithm" / "maxfuse"
FORMAL_RUNNER = ALGORITHM_DIR / "run_maxfuse_three_view_extension.py"
DEFAULT_MAXFUSE_PYTHON = ALGORITHM_DIR / ".venv" / "bin" / "python"
THREE_VIEW_EXTENSION_METHOD = (
    "MaxFuse Archive three-view workflow (pairwise hub-pivot chaining + iterative GCCA)"
)
PAIRWISE_RANK_ADAPTATION = (
    "clip_non_null_official_targets_to_each_view_feature_rank"
)

cache_root = Path(tempfile.gettempdir()) / "final_xu_maxfuse_benchmark_cache"
for variable, leaf in (
    ("MPLCONFIGDIR", "matplotlib"),
    ("NUMBA_CACHE_DIR", "numba"),
    ("XDG_CACHE_HOME", "xdg"),
    ("JOBLIB_TEMP_FOLDER", "joblib"),
):
    target = cache_root / leaf
    target.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(variable, str(target))
for variable in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS", "LOKY_MAX_CPU_COUNT",
):
    os.environ.setdefault(variable, "1")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


SCHEMA = "final_xu.maxfuse_benchmark.v3"
PREPARED_SCHEMA = "final_xu.maxfuse_prepared.v5"
ALIGNMENT_SCHEMA = "final_xu.maxfuse_alignment.v6"
UMAP_SCHEMA = "final_xu.maxfuse_umap.v5"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_hash_map(root: Path, expected: dict[str, str]) -> bool:
    if not isinstance(expected, dict) or not expected:
        return False
    for relative, digest in expected.items():
        path = root / relative
        if not path.is_file() or sha256(path) != digest:
            return False
    return True


def verify_source_records(records: list[dict[str, Any]]) -> bool:
    if not isinstance(records, list) or not records:
        return False
    for record in records:
        path = Path(record.get("path", ""))
        if not path.is_file() or sha256(path) != record.get("sha256"):
            return False
    return True


def replace_path_prefix(value: Any, old: str, new: str) -> Any:
    """Recursively relocate absolute paths embedded in a JSON payload."""
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [replace_path_prefix(item, old, new) for item in value]
    if isinstance(value, dict):
        return {
            key: replace_path_prefix(item, old, new)
            for key, item in value.items()
        }
    return value


def buffer_sha256(values: np.ndarray, dtype: np.dtype[Any] | None = None) -> str:
    array = np.asarray(values, dtype=dtype) if dtype is not None else np.asarray(values)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def stable_seed(base: int, *parts: object) -> int:
    payload = "|".join(map(str, (base, *parts))).encode("utf-8")
    return int.from_bytes(
        hashlib.blake2b(payload, digest_size=8).digest(), "little"
    ) % (2**32 - 1)


def resolve_from_final_xu(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (FINAL_XU / path).resolve()


def load_config(path: Path) -> tuple[dict[str, Any], Path]:
    path = path.resolve()
    config = json.loads(path.read_text())
    if config.get("schema_version") != SCHEMA:
        raise ValueError(f"Unsupported benchmark config schema: {config.get('schema_version')}")
    required = {
        "benchmark_slug", "benchmark_title", "shared_markers", "datasets",
        "expected_cells", "display_metadata", "label_column", "base_seed",
        "feature_protocol", "fit_sampling", "pairwise_config",
        "three_view_extension", "umap", "palette", "dataset_colors",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Config lacks required keys: {sorted(missing)}")
    datasets = config["datasets"]
    if len(datasets) != 3:
        raise ValueError("Exactly three datasets are required")
    names = [row["name"] for row in datasets]
    slugs = [row["slug"] for row in datasets]
    if len(set(names)) != 3 or len(set(slugs)) != 3:
        raise ValueError("Dataset names and slugs must each be unique")
    shared_markers = config["shared_markers"]
    if (
        not isinstance(shared_markers, list)
        or not shared_markers
        or any(not isinstance(marker, str) or not marker for marker in shared_markers)
        or len(set(shared_markers)) != len(shared_markers)
    ):
        raise ValueError("shared_markers must be a nonempty unique string list")
    for spec in datasets:
        dataset_required = {
            "name", "slug", "display", "expected_cells", "shared_input",
            "input_protocol",
        }
        dataset_missing = dataset_required - set(spec)
        if dataset_missing:
            raise ValueError(
                f"Dataset {spec.get('slug', '<unknown>')} lacks keys: "
                f"{sorted(dataset_missing)}"
            )
        protocol = spec["input_protocol"]
        if (
            not isinstance(protocol, dict)
            or not isinstance(protocol.get("name"), str)
            or not protocol["name"]
            or not isinstance(protocol.get("description"), str)
            or not protocol["description"]
        ):
            raise ValueError(f"{spec['slug']}: invalid input_protocol")
    feature_protocol = config["feature_protocol"]
    if not isinstance(feature_protocol, dict):
        raise TypeError("feature_protocol must be an object")
    active_mode = feature_protocol.get("active_mode")
    if active_mode not in {"shared_only", "dataset_active_inputs"}:
        raise ValueError(
            "feature_protocol.active_mode must be shared_only or dataset_active_inputs"
        )
    if active_mode == "dataset_active_inputs":
        for spec in datasets:
            active_markers = spec.get("active_markers")
            if (
                "active_input" not in spec
                or not isinstance(active_markers, list)
                or not active_markers
                or len(set(active_markers)) != len(active_markers)
            ):
                raise ValueError(
                    f"{spec['slug']}: dataset_active_inputs requires active_input and "
                    "unique active_markers"
                )
    sampling = config["fit_sampling"]
    sampling_required = {
        "strategy", "cap_per_dataset", "drop_zero_within_cell_variance",
        "reuse_across_hubs",
    }
    if not isinstance(sampling, dict) or sampling_required - set(sampling):
        raise ValueError("fit_sampling lacks required fields")
    if sampling["strategy"] not in {"unit_balanced", "uniform_random"}:
        raise ValueError("fit_sampling.strategy is unsupported")
    cap = sampling["cap_per_dataset"]
    if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap < 3):
        raise ValueError("fit_sampling.cap_per_dataset must be null or an integer >= 3")
    if not isinstance(sampling["drop_zero_within_cell_variance"], bool):
        raise TypeError("drop_zero_within_cell_variance must be Boolean")
    if sampling["reuse_across_hubs"] is not True:
        raise ValueError("the current three-hub runner requires reuse_across_hubs=true")
    extension = config["three_view_extension"]
    extension_required = {
        "method", "svd_components", "gcca_components",
        "gcca_initialization_max_iter", "gcca_refinement_max_iter",
        "gcca_tolerance", "view_order_policy", "outer_reuse_policy",
    }
    if not isinstance(extension, dict) or extension_required - set(extension):
        raise ValueError("three_view_extension lacks required fields")
    if extension["method"] != THREE_VIEW_EXTENSION_METHOD:
        raise ValueError("three_view_extension.method is not the canonical explicit name")
    if extension["view_order_policy"] != "cyclic_from_dataset_config_order":
        raise ValueError(
            "three_view_extension.view_order_policy must be "
            "cyclic_from_dataset_config_order"
        )
    svd_components = extension["svd_components"]
    if not isinstance(svd_components, dict) or set(svd_components) != set(slugs):
        raise ValueError(
            "three_view_extension.svd_components must explicitly map every dataset slug"
        )
    for slug, value in svd_components.items():
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 1
        ):
            raise ValueError(
                f"three_view_extension.svd_components[{slug!r}] must be null or positive"
            )
    for key in (
        "gcca_components", "gcca_initialization_max_iter",
        "gcca_refinement_max_iter",
    ):
        value = extension[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"three_view_extension.{key} must be a positive integer")
    tolerance = extension["gcca_tolerance"]
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not np.isfinite(tolerance)
        or tolerance <= 0
    ):
        raise ValueError("three_view_extension.gcca_tolerance must be finite and positive")
    if extension["outer_reuse_policy"] not in {"allow", "drop_reused"}:
        raise ValueError("unsupported three_view_extension.outer_reuse_policy")
    pairwise_path = resolve_from_final_xu(config["pairwise_config"])
    if not pairwise_path.is_file():
        raise FileNotFoundError(pairwise_path)
    pairwise_payload = json.loads(pairwise_path.read_text())
    if pairwise_payload.get("schema_version") != "maxfuse_repro.PairwiseConfig.v1":
        raise ValueError("pairwise_config is not a complete PairwiseConfig.v1 object")
    rank_adaptation = config.get("pairwise_rank_adaptation")
    if rank_adaptation is not None and rank_adaptation != PAIRWISE_RANK_ADAPTATION:
        raise ValueError(
            "pairwise_rank_adaptation must be the canonical explicit rank-clipping rule"
        )
    return config, path


def load_pairwise_config(config: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    path = resolve_from_final_xu(config["pairwise_config"])
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != "maxfuse_repro.PairwiseConfig.v1":
        raise ValueError("pairwise_config is not a complete PairwiseConfig.v1 object")
    return payload, path


def resolve_pairwise_dimensions(
    payload: dict[str, Any],
    *,
    policy: str | None,
    shared_width_first: int,
    shared_width_second: int,
    active_width_first: int,
    active_width_second: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve declared tutorial dimensions against one ordered data pair.

    MaxFuse's exact-SVD path calls ``scipy.sparse.linalg.svds``, which requires
    ``k < min(matrix.shape)``.  The benchmark therefore declares the official
    tutorial targets in its pairwise JSON and clips each non-null target to
    ``feature_width - 1`` for the actual ordered views.  This is a mechanical
    feasibility rule fixed before outcomes or UMAPs are read; it is not an
    algorithm-side silent fallback.
    """
    resolved = copy.deepcopy(payload)
    widths = {
        "shared1": int(shared_width_first),
        "shared2": int(shared_width_second),
        "active1": int(active_width_first),
        "active2": int(active_width_second),
    }
    if any(width < 2 for width in widths.values()):
        raise ValueError(f"pairwise views cannot support rank adaptation: {widths}")
    audit: dict[str, Any] = {
        "policy": policy or "none",
        "feature_widths": widths,
        "changes": [],
    }
    if policy is None:
        return resolved, audit
    if policy != PAIRWISE_RANK_ADAPTATION:
        raise ValueError(f"unsupported pairwise rank adaptation: {policy}")

    def clip(section: str, key: str, width_key: str) -> int | None:
        requested = resolved[section][key]
        if requested is None:
            return None
        if isinstance(requested, bool) or not isinstance(requested, int) or requested < 1:
            raise ValueError(f"invalid {section}.{key} target: {requested}")
        value = min(requested, widths[width_key] - 1)
        if value < 1:
            raise ValueError(f"{section}.{key} cannot be resolved for {width_key}")
        resolved[section][key] = value
        audit["changes"].append({
            "parameter": f"{section}.{key}",
            "requested": requested,
            "resolved": value,
            "bounded_by": f"{width_key}-1",
        })
        return value

    # Batching SVD targets are null in the official tutorial but are handled
    # here if a future explicit benchmark profile makes them non-null.
    clip("batching", "shared_svd_components1", "shared1")
    clip("batching", "shared_svd_components2", "shared2")
    clip("batching", "active_svd_components1", "active1")
    clip("batching", "active_svd_components2", "active2")
    clip("graph", "svd_components1", "active1")
    clip("graph", "svd_components2", "active2")
    clip("initial", "svd_components1", "shared1")
    clip("initial", "svd_components2", "shared2")
    refine1 = clip("refine", "svd_components1", "active1")
    refine2 = clip("refine", "svd_components2", "active2")
    clip("propagation", "svd_components1", "active1")
    clip("propagation", "svd_components2", "active2")

    requested_cca = resolved["refine"]["cca_components"]
    if isinstance(requested_cca, bool) or not isinstance(requested_cca, int) or requested_cca < 1:
        raise ValueError(f"invalid refine.cca_components target: {requested_cca}")
    cca_bound1 = refine1 if refine1 is not None else widths["active1"]
    cca_bound2 = refine2 if refine2 is not None else widths["active2"]
    resolved_cca = min(requested_cca, cca_bound1, cca_bound2)
    if resolved_cca < 1:
        raise ValueError("refine CCA cannot be resolved for the active panels")
    resolved["refine"]["cca_components"] = resolved_cca
    audit["changes"].append({
        "parameter": "refine.cca_components",
        "requested": requested_cca,
        "resolved": resolved_cca,
        "bounded_by": "resolved refine active dimensions",
    })
    return resolved, audit


def output_root(config: dict[str, Any]) -> Path:
    # Historical results live in ``maxfuse``.  A benchmark config may select a
    # sibling, versioned result directory so a corrected preprocessing protocol
    # can be run without overwriting those immutable artifacts.
    subdir = config.get("result_subdir", "maxfuse")
    if (
        not isinstance(subdir, str)
        or not subdir
        or Path(subdir).name != subdir
        or subdir in {".", ".."}
    ):
        raise ValueError("result_subdir must be one nonempty directory name")
    return HERE / config["benchmark_slug"] / subdir


def deterministic_gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(
            filename="", fileobj=raw, mode="wb", compresslevel=6, mtime=0
        ) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
                frame.to_csv(text, index=False, float_format="%.9g")


def load_label_free_inputs(
    config: dict[str, Any],
) -> tuple[
    np.ndarray,
    pd.DataFrame,
    dict[str, np.ndarray],
    dict[str, list[str]],
    list[dict[str, Any]],
]:
    markers = list(map(str, config["shared_markers"]))
    active_mode = str(config["feature_protocol"]["active_mode"])
    blocks: list[np.ndarray] = []
    metadata_blocks: list[pd.DataFrame] = []
    active_blocks: dict[str, np.ndarray] = {}
    active_markers: dict[str, list[str]] = {}
    sources: list[dict[str, Any]] = []
    for spec in config["datasets"]:
        path = resolve_from_final_xu(spec["shared_input"])
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, low_memory=False, float_precision="round_trip")
        required = ["cell_id", "dataset", "donor", "unit_key", *markers]
        missing = [column for column in required if column not in frame]
        if missing:
            raise ValueError(f"{path} lacks columns: {missing}")
        if len(frame) != int(spec["expected_cells"]):
            raise AssertionError(f"{spec['name']}: cell count changed")
        observed_names = frame["dataset"].astype(str).unique().tolist()
        if observed_names != [spec["name"]]:
            raise AssertionError(
                f"{path}: dataset field {observed_names} != {[spec['name']]}"
            )
        values = frame[markers].to_numpy(np.float64)
        if values.shape != (len(frame), len(markers)) or not np.isfinite(values).all():
            raise ValueError(f"{path}: invalid marker matrix")
        blocks.append(values)
        metadata_blocks.append(
            frame[["cell_id", "dataset", "donor", "unit_key"]].copy()
        )
        sources.append({
            "path": str(path),
            "sha256": sha256(path),
            "role": f"{spec['slug']}.shared_input",
            "input_protocol": spec["input_protocol"],
        })
        if active_mode == "shared_only":
            active_blocks[spec["slug"]] = values.copy()
            active_markers[spec["slug"]] = markers.copy()
        else:
            active_path = resolve_from_final_xu(spec["active_input"])
            if not active_path.is_file():
                raise FileNotFoundError(active_path)
            active_frame = pd.read_csv(
                active_path, low_memory=False, float_precision="round_trip"
            )
            names = list(map(str, spec["active_markers"]))
            active_required = ["cell_id", *names]
            active_missing = [column for column in active_required if column not in active_frame]
            if active_missing:
                raise ValueError(f"{active_path} lacks columns: {active_missing}")
            if len(active_frame) != len(frame):
                raise AssertionError(f"{spec['name']}: active/shared cell counts differ")
            if not np.array_equal(
                active_frame["cell_id"].astype(str).to_numpy(),
                frame["cell_id"].astype(str).to_numpy(),
            ):
                raise AssertionError(f"{spec['name']}: active/shared cell order differs")
            active_values = active_frame[names].to_numpy(np.float64)
            if (
                active_values.shape != (len(frame), len(names))
                or not np.isfinite(active_values).all()
            ):
                raise ValueError(f"{active_path}: invalid active marker matrix")
            try:
                shared_positions = [names.index(marker) for marker in markers]
            except ValueError as error:
                raise ValueError(
                    f"{spec['name']}: every shared marker must also occur in active_markers"
                ) from error
            if len(set(names)) != len(names):
                raise ValueError(f"{spec['name']}: active marker names are not unique")
            if not np.array_equal(active_values[:, shared_positions], values):
                maximum_error = float(
                    np.max(np.abs(active_values[:, shared_positions] - values))
                )
                raise AssertionError(
                    f"{spec['name']}: shared input is not an exact active-matrix slice "
                    f"(maximum absolute difference {maximum_error:.17g})"
                )
            active_blocks[spec["slug"]] = active_values
            active_markers[spec["slug"]] = names
            sources.append({
                "path": str(active_path),
                "sha256": sha256(active_path),
                "role": f"{spec['slug']}.active_input",
                "input_protocol": spec["input_protocol"],
            })
    matrix = np.vstack(blocks)
    metadata = pd.concat(metadata_blocks, ignore_index=True)
    if matrix.shape != (int(config["expected_cells"]), len(markers)):
        raise AssertionError("Combined label-free input dimensions changed")
    cell_ids = metadata["cell_id"].astype(str)
    if cell_ids.duplicated().any():
        duplicates = cell_ids[cell_ids.duplicated()].head().tolist()
        raise AssertionError(f"Global cell IDs are not unique: {duplicates}")
    expected_order = [row["name"] for row in config["datasets"]]
    if metadata["dataset"].astype(str).drop_duplicates().tolist() != expected_order:
        raise AssertionError("Dataset block order changed")
    reference_npy = config.get("combined_shared_reference_npy")
    if reference_npy:
        reference_path = resolve_from_final_xu(reference_npy)
        reference = np.load(reference_path, allow_pickle=False)
        reference_float64 = np.asarray(reference, dtype=np.float64)
        maximum = (
            float(np.max(np.abs(reference_float64 - matrix)))
            if reference.shape == matrix.shape else None
        )
        if (
            reference.shape != matrix.shape
            or maximum is None
            or not np.isfinite(reference_float64).all()
            or not np.array_equal(reference_float64, matrix)
        ):
            raise AssertionError(
                "CSV blocks do not reproduce the configured shared reference NPY "
                f"bit-for-bit after float64 parsing (max abs={maximum})"
            )
        # The reference is allowed only as an exact serialization duplicate; it
        # must never replace shared values by approximately equal alternatives.
        matrix = reference_float64
        if active_mode == "shared_only":
            start = 0
            for spec in config["datasets"]:
                stop = start + int(spec["expected_cells"])
                active_blocks[spec["slug"]] = matrix[start:stop].copy()
                start = stop
        sources.append({
            "path": str(reference_path),
            "sha256": sha256(reference_path),
            "role": "combined_shared_reference_npy",
            "maximum_absolute_difference_from_shared_csv_blocks": maximum,
        })
    return matrix, metadata, active_blocks, active_markers, sources


def select_fit_positions(
    metadata: pd.DataFrame,
    dataset: str,
    sampling: dict[str, Any],
    base_seed: int,
) -> np.ndarray:
    global_rows = np.flatnonzero(metadata["dataset"].astype(str).to_numpy() == dataset)
    cap = sampling["cap_per_dataset"]
    if cap is None or len(global_rows) <= int(cap):
        return global_rows
    cap = int(cap)
    local = metadata.iloc[global_rows].reset_index(drop=True)
    rng = np.random.default_rng(stable_seed(base_seed, dataset, "maxfuse-fit"))
    if sampling["strategy"] == "uniform_random":
        return np.sort(rng.choice(global_rows, size=cap, replace=False)).astype(np.int64)
    if sampling["strategy"] != "unit_balanced":
        raise ValueError(f"Unsupported fit sampling strategy: {sampling['strategy']}")
    groups = [group.index.to_numpy(np.int64) for _, group in local.groupby("unit_key", sort=True)]
    if len(groups) > cap:
        raise ValueError(f"{dataset}: more units than fit cap")
    quota = max(1, cap // len(groups))
    selected: list[int] = []
    leftovers: list[int] = []
    for rows in groups:
        permuted = rng.permutation(rows)
        take = min(quota, len(rows))
        selected.extend(permuted[:take].tolist())
        leftovers.extend(permuted[take:].tolist())
    remaining = cap - len(selected)
    if remaining > 0:
        selected.extend(
            rng.permutation(np.asarray(leftovers, dtype=np.int64))[:remaining].tolist()
        )
    local_selected = np.sort(np.asarray(selected, dtype=np.int64))
    if len(local_selected) != cap or len(np.unique(local_selected)) != cap:
        raise AssertionError(f"{dataset}: invalid balanced fit sample")
    return global_rows[local_selected]


def prepare(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    benchmark_runner_path = Path(__file__).resolve()
    benchmark_runner_hash = sha256(benchmark_runner_path)
    config_hash = sha256(config_path)
    root = output_root(config)
    prepared = root / "prepared"
    manifest_path = prepared / "manifest.json"
    shared_input_path = prepared / "label_free_shared_float64.npy"
    metadata_path = prepared / "fit_metadata.csv.gz"
    markers_path = prepared / "shared_markers.txt"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        required = [shared_input_path, metadata_path, markers_path]
        required.extend(
            prepared / relative
            for relative in manifest.get("active_input_files", {}).values()
        )
        required.extend(
            prepared / relative
            for relative in manifest.get("active_marker_files", {}).values()
        )
        if (
            manifest.get("schema_version") == PREPARED_SCHEMA
            and manifest.get("status") == "complete"
            and all(path.is_file() for path in required)
            and verify_hash_map(prepared, manifest.get("outputs", {}))
            and verify_source_records(manifest.get("sources", []))
            and manifest["config_sha256"] == config_hash
            and manifest.get("shared_markers") == list(config["shared_markers"])
            and manifest.get("datasets") == [row["name"] for row in config["datasets"]]
            and manifest.get("feature_protocol") == config["feature_protocol"]
            and manifest.get("fit_sampling") == config["fit_sampling"]
            and manifest.get("base_seed") == int(config["base_seed"])
            and manifest.get("benchmark_runner_sha256")
            == benchmark_runner_hash
        ):
            print(f"[{config['benchmark_slug']}] prepared inputs already verified", flush=True)
            return manifest
        raise RuntimeError(f"Incomplete or stale prepared directory: {prepared}")

    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".prepared.tmp-", dir=root))
    try:
        shared, metadata, active_blocks, active_markers, sources = load_label_free_inputs(config)
        datasets = metadata["dataset"].astype(str).to_numpy()
        method_shared = np.asarray(shared, dtype=np.float64).copy()
        if not np.isfinite(method_shared).all():
            raise RuntimeError("Configured shared matrix contains non-finite values")
        informative_global = (
            np.isfinite(method_shared).all(axis=1)
            & (np.std(method_shared, axis=1) > 1e-10)
        )
        input_audit: list[dict[str, Any]] = []
        fit_audit: list[dict[str, Any]] = []
        fit_indices: dict[str, np.ndarray] = {}
        sampling = dict(config["fit_sampling"])
        for spec in config["datasets"]:
            name = spec["name"]
            selected = datasets == name
            shared_block = method_shared[selected]
            active_block = active_blocks[spec["slug"]]
            input_audit.append({
                "dataset": name,
                "slug": spec["slug"],
                "cells": int(selected.sum()),
                "input_protocol": spec["input_protocol"]["name"],
                "shared_features": int(shared_block.shape[1]),
                "active_features": int(active_block.shape[1]),
                "minimum_shared_value": float(shared_block.min()),
                "maximum_shared_value": float(shared_block.max()),
                "shared_zero_fraction": float(np.mean(shared_block == 0)),
                "shared_negative_fraction": float(np.mean(shared_block < 0)),
                "shared_buffer_sha256_float64": buffer_sha256(shared_block, np.float64),
                "active_buffer_sha256_float64": buffer_sha256(active_block, np.float64),
            })
            rows = select_fit_positions(
                metadata, name, sampling, int(config["base_seed"])
            )
            informative = np.isfinite(method_shared[rows]).all(axis=1)
            if sampling["drop_zero_within_cell_variance"]:
                informative &= informative_global[rows]
            retained = rows[informative]
            if len(retained) < 3:
                raise RuntimeError(f"{name}: fewer than 3 informative fit cells")
            fit_indices[spec["slug"]] = retained.astype(np.int64)
            unit_counts = metadata.iloc[retained].groupby("unit_key", sort=True).size()
            fit_audit.append({
                "dataset": name,
                "slug": spec["slug"],
                "sampled_before_informative_filter": int(len(rows)),
                "fit_cells": int(len(retained)),
                "dropped_zero_within_cell_variance": int(len(rows) - len(retained)),
                "fit_units": int(len(unit_counts)),
                "minimum_cells_per_unit": int(unit_counts.min()),
                "maximum_cells_per_unit": int(unit_counts.max()),
            })
        np.save(temporary / shared_input_path.name, method_shared)
        deterministic_gzip_csv(metadata, temporary / metadata_path.name)
        (temporary / markers_path.name).write_text(
            "\n".join(config["shared_markers"]) + "\n"
        )
        pd.DataFrame(input_audit).to_csv(temporary / "input_audit.csv", index=False)
        pd.DataFrame(fit_audit).to_csv(temporary / "fit_sample_audit.csv", index=False)
        fit_files: dict[str, str] = {}
        for slug, rows in fit_indices.items():
            filename = f"fit_indices_{slug}.npy"
            np.save(temporary / filename, rows)
            fit_files[slug] = filename
        active_input_files: dict[str, str] = {}
        active_marker_files: dict[str, str] = {}
        for spec in config["datasets"]:
            slug = spec["slug"]
            input_filename = f"active_input_{slug}_float64.npy"
            marker_filename = f"active_markers_{slug}.txt"
            np.save(temporary / input_filename, active_blocks[slug])
            (temporary / marker_filename).write_text("\n".join(active_markers[slug]) + "\n")
            active_input_files[slug] = input_filename
            active_marker_files[slug] = marker_filename
        output_hashes = {
            path.name: sha256(path)
            for path in temporary.iterdir()
            if path.is_file() and path.name != "manifest.json"
        }
        manifest = {
            "schema_version": PREPARED_SCHEMA,
            "status": "complete",
            "benchmark": config["benchmark_slug"],
            "cells": int(len(metadata)),
            "shared_markers": list(config["shared_markers"]),
            "active_markers": active_markers,
            "datasets": [row["name"] for row in config["datasets"]],
            "input_protocols": {
                spec["slug"]: spec["input_protocol"] for spec in config["datasets"]
            },
            "feature_protocol": config["feature_protocol"],
            "fit_sampling": config["fit_sampling"],
            "metadata_use": (
                "donor/unit_key may be used only by the configured fit-sampling strategy and "
                "is never supplied to MaxFuse; cell-type labels are never opened"
            ),
            "base_seed": int(config["base_seed"]),
            "labels_loaded_during_preparation": False,
            "fit_index_files": fit_files,
            "active_input_files": active_input_files,
            "active_marker_files": active_marker_files,
            "sources": sources,
            "config": str(config_path),
            "config_sha256": config_hash,
            "benchmark_runner": str(benchmark_runner_path),
            "benchmark_runner_sha256": benchmark_runner_hash,
            "outputs": output_hashes,
            "shared_matrix_buffer_sha256_float64": buffer_sha256(
                method_shared, np.float64
            ),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        if sha256(benchmark_runner_path) != benchmark_runner_hash:
            raise RuntimeError("Benchmark runner changed during input preparation")
        if sha256(config_path) != config_hash:
            raise RuntimeError("Benchmark config changed during input preparation")
        if not verify_source_records(sources):
            raise RuntimeError("A configured input changed during input preparation")
        temporary.replace(prepared)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(
        f"[{config['benchmark_slug']}] prepared {len(metadata):,} cells x "
        f"{len(config['shared_markers'])} shared markers",
        flush=True,
    )
    return json.loads(manifest_path.read_text())


def resolve_maxfuse_python() -> Path:
    requested = os.environ.get("MAXFUSE_PYTHON")
    candidate = Path(requested).expanduser() if requested else DEFAULT_MAXFUSE_PYTHON
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    # Do not resolve the venv's Python symlink: invoking its resolved base
    # interpreter would bypass the virtual environment site-packages.
    candidate = candidate.absolute()
    if not candidate.is_file():
        raise FileNotFoundError(
            f"MaxFuse Python not found at {candidate}; set MAXFUSE_PYTHON to an environment "
            "containing maxfuse==0.0.2"
        )
    return candidate


def align_one_hub(
    config: dict[str, Any], config_path: Path, hub_slug: str
) -> Path:
    benchmark_runner_path = Path(__file__).resolve()
    benchmark_runner_hash = sha256(benchmark_runner_path)
    config_hash = sha256(config_path)
    root = output_root(config)
    prepared = root / "prepared"
    pairwise_payload, pairwise_path = load_pairwise_config(config)
    pairwise_hash = sha256(pairwise_path)
    prepared_manifest_path = prepared / "manifest.json"
    if not prepared_manifest_path.is_file():
        raise FileNotFoundError(prepared_manifest_path)
    prepared_manifest_hash = sha256(prepared_manifest_path)
    algorithm_runner_hash = sha256(FORMAL_RUNNER)
    extension = dict(config["three_view_extension"])
    target = root / f"hub_{hub_slug}" / "alignment"
    manifest_path = target / "alignment_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        formal_manifest_path = target / "formal_fit" / "manifest.json"
        formal_ok = False
        if formal_manifest_path.is_file():
            formal_manifest = json.loads(formal_manifest_path.read_text())
            formal_ok = verify_hash_map(
                target / "formal_fit", formal_manifest.get("output_sha256", {})
            )
        if (
            manifest.get("schema_version") == ALIGNMENT_SCHEMA
            and manifest.get("status") == "complete"
            and verify_hash_map(target, manifest.get("outputs", {}))
            and manifest["config_sha256"] == config_hash
            and prepared_manifest_path.is_file()
            and manifest.get("preprocessing_manifest_sha256") == prepared_manifest_hash
            and manifest.get("pairwise_config_sha256") == pairwise_hash
            and manifest.get("algorithm_runner_sha256") == algorithm_runner_hash
            and manifest.get("benchmark_runner_sha256")
            == benchmark_runner_hash
            and formal_ok
            and manifest.get("formal_fit_manifest_sha256") == sha256(formal_manifest_path)
        ):
            print(f"[{config['benchmark_slug']}/{hub_slug}] alignment already verified", flush=True)
            return target
        raise RuntimeError(f"Incomplete or stale alignment directory: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".alignment.tmp-", dir=target.parent))
    started = time.time()
    try:
        shared_input_path = prepared / "label_free_shared_float64.npy"
        metadata_path = prepared / "fit_metadata.csv.gz"
        prepared_manifest = json.loads((prepared / "manifest.json").read_text())
        method_shared = np.load(shared_input_path, allow_pickle=False)
        metadata = pd.read_csv(metadata_path, low_memory=False)
        shared_markers = list(config["shared_markers"])
        by_slug = {row["slug"]: row for row in config["datasets"]}
        if hub_slug not in by_slug:
            raise KeyError(f"Unknown hub slug: {hub_slug}")
        canonical_slugs = [row["slug"] for row in config["datasets"]]
        hub_position = canonical_slugs.index(hub_slug)
        view_order = (
            canonical_slugs[hub_position:] + canonical_slugs[:hub_position]
        )
        arrays: dict[str, np.ndarray] = {}
        fit_counts: dict[str, int] = {}
        full_active: dict[str, np.ndarray] = {}
        active_markers: dict[str, list[str]] = {}
        for position, slug in enumerate(view_order):
            rows = np.load(prepared / prepared_manifest["fit_index_files"][slug], allow_pickle=False)
            dataset_name = by_slug[slug]["name"]
            dataset_rows = np.flatnonzero(
                metadata["dataset"].astype(str).to_numpy() == dataset_name
            )
            local_rows = np.searchsorted(dataset_rows, rows)
            if (
                len(local_rows) != len(rows)
                or np.any(local_rows >= len(dataset_rows))
                or not np.array_equal(dataset_rows[local_rows], rows)
            ):
                raise AssertionError(f"{slug}: fit indices do not map to the dataset block")
            active = np.load(
                prepared / prepared_manifest["active_input_files"][slug],
                allow_pickle=False,
            )
            names = (
                prepared / prepared_manifest["active_marker_files"][slug]
            ).read_text().splitlines()
            if active.shape != (len(dataset_rows), len(names)) or not np.isfinite(active).all():
                raise ValueError(f"{slug}: invalid prepared active matrix")
            full_active[slug] = active
            active_markers[slug] = names
            shared_names = np.asarray(shared_markers, dtype=np.str_)
            active_names = np.asarray(names, dtype=np.str_)
            ids = metadata.iloc[rows]["cell_id"].astype(str).to_numpy(dtype=np.str_)
            arrays[f"view{position}_shared"] = method_shared[rows]
            arrays[f"view{position}_active"] = active[local_rows]
            arrays[f"view{position}_shared_feature_names"] = shared_names
            arrays[f"view{position}_active_feature_names"] = active_names
            arrays[f"view{position}_cell_ids"] = ids
            fit_counts[slug] = int(len(rows))
        input_dir = temporary / "input"
        input_dir.mkdir()
        fit_input = input_dir / "fit_input.npz"
        np.savez_compressed(fit_input, **arrays)
        pair_map: dict[str, dict[str, Any]] = {}
        pair_resolution: dict[str, dict[str, Any]] = {}
        for slug in view_order:
            if slug == hub_slug:
                continue
            resolved_pair, resolution_audit = resolve_pairwise_dimensions(
                pairwise_payload,
                policy=config.get("pairwise_rank_adaptation"),
                shared_width_first=len(shared_markers),
                shared_width_second=len(shared_markers),
                active_width_first=len(active_markers[hub_slug]),
                active_width_second=len(active_markers[slug]),
            )
            pair_map[slug] = resolved_pair
            pair_resolution[slug] = {
                "first_view": hub_slug,
                "second_view": slug,
                **resolution_audit,
            }
        pair_path = input_dir / "pair_config_map.json"
        pair_path.write_text(
            json.dumps(pair_map, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        pair_resolution_path = input_dir / "pair_config_resolution.json"
        pair_resolution_path.write_text(
            json.dumps(
                pair_resolution, indent=2, sort_keys=True, allow_nan=False
            ) + "\n"
        )
        formal_output = temporary / "formal_fit"
        requested_svd = {
            slug: extension["svd_components"][slug] for slug in view_order
        }
        for slug, value in requested_svd.items():
            if value is not None and value >= min(fit_counts[slug], len(active_markers[slug])):
                raise ValueError(
                    f"{slug}: configured Archive SVD width {value} is not smaller than "
                    "the fitted matrix rank bound"
                )
        requested_gcca = int(extension["gcca_components"])
        maxfuse_python = resolve_maxfuse_python()
        command = [
            str(maxfuse_python), str(FORMAL_RUNNER),
            "--input", str(fit_input),
            "--views", *view_order,
            "--hub", hub_slug,
            "--pair-config-map", str(pair_path),
            "--svd-components", *[
                "none" if requested_svd[slug] is None else str(requested_svd[slug])
                for slug in view_order
            ],
            "--gcca-components", str(requested_gcca),
            "--gcca-initialization-max-iter",
            str(int(extension["gcca_initialization_max_iter"])),
            "--gcca-refinement-max-iter",
            str(int(extension["gcca_refinement_max_iter"])),
            "--gcca-tolerance", str(float(extension["gcca_tolerance"])),
            "--outer-reuse-policy", str(extension["outer_reuse_policy"]),
            "--output", str(formal_output),
        ]
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            completed = subprocess.run(
                command, cwd=str(ALGORITHM_DIR), env=environment,
                check=True, text=True, capture_output=True,
            )
        except subprocess.CalledProcessError as error:
            if error.stdout:
                print(error.stdout, file=sys.stderr, flush=True)
            if error.stderr:
                print(error.stderr, file=sys.stderr, flush=True)
            raise
        print(completed.stdout.strip(), flush=True)

        sys.path.insert(0, str(ALGORITHM_DIR))
        from maxfuse_repro import ThreeViewExtensionProjectionModel

        model = ThreeViewExtensionProjectionModel.load(
            formal_output / "projection_model.npz"
        )
        actual_dims = int(model.views[model.view_order[0]].gcca_weight.shape[1])
        formal_manifest = json.loads((formal_output / "manifest.json").read_text())
        if actual_dims != int(formal_manifest["audit"]["gcca_components"]):
            raise AssertionError("Projection-model width and formal GCCA audit disagree")
        if any(state.gcca_weight.shape[1] != actual_dims for state in model.views.values()):
            raise AssertionError("Projection-model GCCA widths differ across views")
        # The result directory is written atomically under a temporary name.
        # Relocate embedded absolute paths before the directory is moved so
        # every completed manifest points at files that will actually exist.
        formal_manifest = replace_path_prefix(
            formal_manifest, str(temporary.resolve()), str(target.resolve())
        )
        (formal_output / "manifest.json").write_text(
            json.dumps(formal_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        final_command = replace_path_prefix(
            command, str(temporary.resolve()), str(target.resolve())
        )
        embedding = np.empty((len(metadata), actual_dims), dtype=np.float64)
        dataset_array = metadata["dataset"].astype(str).to_numpy()
        projection_audit: list[dict[str, Any]] = []
        for spec in config["datasets"]:
            rows = np.flatnonzero(dataset_array == spec["name"])
            projected = model.transform(
                spec["slug"], full_active[spec["slug"]], active_markers[spec["slug"]]
            )
            if projected.shape != (len(rows), actual_dims) or not np.isfinite(projected).all():
                raise RuntimeError(f"{spec['name']}: invalid full projection")
            embedding[rows] = projected
            projection_audit.append({
                "dataset": spec["name"], "slug": spec["slug"],
                "fit_cells": fit_counts[spec["slug"]],
                "projected_full_cells": int(len(rows)),
            })
        data_dir = temporary / "data"
        data_dir.mkdir()
        embedding_path = data_dir / "maxfuse_embedding_float64.npy"
        np.save(embedding_path, embedding)
        ids_path = data_dir / "cell_ids.npy"
        np.save(ids_path, metadata["cell_id"].astype(str).to_numpy(dtype=np.str_))
        deterministic_gzip_csv(metadata, data_dir / "cell_metadata_label_free.csv.gz")
        pd.DataFrame(projection_audit).to_csv(
            temporary / "full_projection_audit.csv", index=False
        )
        outputs = {
            "input/pair_config_map.json": sha256(pair_path),
            "input/pair_config_resolution.json": sha256(pair_resolution_path),
            "data/maxfuse_embedding_float64.npy": sha256(embedding_path),
            "data/cell_ids.npy": sha256(ids_path),
            "data/cell_metadata_label_free.csv.gz": sha256(
                data_dir / "cell_metadata_label_free.csv.gz"
            ),
            "full_projection_audit.csv": sha256(temporary / "full_projection_audit.csv"),
        }
        manifest = {
            "schema_version": ALIGNMENT_SCHEMA,
            "status": "complete",
            "benchmark": config["benchmark_slug"],
            "method": formal_manifest["method"],
            "hub_slug": hub_slug,
            "hub_dataset": by_slug[hub_slug]["name"],
            "view_input_order": view_order,
            "fit_cells": fit_counts,
            "full_cells": int(len(metadata)),
            "shared_markers": shared_markers,
            "active_markers": active_markers,
            "feature_protocol": config["feature_protocol"],
            "fit_sampling": config["fit_sampling"],
            "requested_archive_svd_components": requested_svd,
            "requested_gcca_components": requested_gcca,
            "embedding_dimensions": actual_dims,
            "labels_used_for_fitting": False,
            "labels_loaded_before_alignment_completed": False,
            "preprocessing_manifest": str(prepared / "manifest.json"),
            "preprocessing_manifest_sha256": prepared_manifest_hash,
            "formal_fit_manifest": str(target / "formal_fit" / "manifest.json"),
            "formal_fit_manifest_sha256": sha256(formal_output / "manifest.json"),
            "formal_fit_audit": formal_manifest["audit"],
            "full_projection": (
                "saved label-free ThreeViewExtensionProjectionModel applied to every "
                "full-cohort cell; fit selection affects model estimation only"
            ),
            "pairwise_config": str(pairwise_path),
            "pairwise_config_sha256": pairwise_hash,
            "pairwise_rank_adaptation": config.get("pairwise_rank_adaptation"),
            "resolved_pairwise_config_map": str(target / "input" / pair_path.name),
            "resolved_pairwise_config_map_sha256": sha256(pair_path),
            "pairwise_rank_resolution": pair_resolution,
            "three_view_extension": extension,
            "config": str(config_path),
            "config_sha256": config_hash,
            "algorithm_runner": str(FORMAL_RUNNER),
            "algorithm_runner_sha256": algorithm_runner_hash,
            "benchmark_runner": str(benchmark_runner_path),
            "benchmark_runner_sha256": benchmark_runner_hash,
            "maxfuse_python": str(maxfuse_python),
            "command": final_command,
            "subprocess_stdout": completed.stdout,
            "subprocess_stderr": completed.stderr,
            "runtime_seconds": time.time() - started,
            "outputs": outputs,
            "embedding_buffer_sha256_float64": buffer_sha256(embedding, np.float64),
        }
        (temporary / "alignment_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        if sha256(benchmark_runner_path) != benchmark_runner_hash:
            raise RuntimeError("Benchmark runner changed during alignment")
        if sha256(config_path) != config_hash:
            raise RuntimeError("Benchmark config changed during alignment")
        if sha256(pairwise_path) != pairwise_hash:
            raise RuntimeError("Pairwise config changed during alignment")
        if sha256(prepared_manifest_path) != prepared_manifest_hash:
            raise RuntimeError("Prepared manifest changed during alignment")
        if sha256(FORMAL_RUNNER) != algorithm_runner_hash:
            raise RuntimeError("Formal algorithm runner changed during alignment")
        if not verify_hash_map(prepared, prepared_manifest.get("outputs", {})):
            raise RuntimeError("A prepared input changed during alignment")
        temporary.replace(target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(
        f"[{config['benchmark_slug']}/{hub_slug}] full alignment saved "
        f"({len(metadata):,} cells x {actual_dims} dimensions)",
        flush=True,
    )
    return target


def run_alignments(config: dict[str, Any], config_path: Path, hub: str) -> None:
    prepare(config, config_path)
    slugs = [row["slug"] for row in config["datasets"]]
    selected = slugs if hub == "all" else [hub]
    unknown = set(selected) - set(slugs)
    if unknown:
        raise KeyError(f"Unknown hub(s): {sorted(unknown)}")
    for slug in selected:
        align_one_hub(config, config_path, slug)


def viewwise_standardized_embedding(
    values: np.ndarray,
    datasets: np.ndarray,
    dataset_order: list[str],
) -> np.ndarray:
    """Apply the official Archive final full-view display normalization.

    Archive GCCA already scales the fitted view.  This population-SD pass is
    still required after projecting the fitted model to additional full-cohort
    cells; it is an identity up to roundoff when fitted and displayed rows are
    the same.  No sample-SD substitution is made.
    """
    values = np.asarray(values, dtype=np.float64)
    datasets = np.asarray(datasets, dtype=str)
    if values.ndim != 2 or values.shape[0] != len(datasets) or values.shape[1] < 2:
        raise ValueError("Invalid embedding/dataset dimensions")
    if not np.isfinite(values).all():
        raise ValueError("Embedding contains non-finite values")
    result = np.empty(values.shape, dtype=np.float32)
    for dataset in dataset_order:
        selected = datasets == dataset
        if int(selected.sum()) < 2:
            raise ValueError(f"{dataset}: fewer than two embedding rows")
        block = values[selected]
        center = block.mean(axis=0)
        scale = block.std(axis=0, ddof=0)
        scale[~np.isfinite(scale) | (scale < 1e-10)] = 1.0
        result[selected] = ((block - center) / scale).astype(np.float32)
    if not np.isfinite(result).all():
        raise RuntimeError("View-wise display standardization produced non-finite values")
    return result


def legend_handles(order: list[str], colors: dict[str, str], display: dict[str, str] | None = None) -> list[Line2D]:
    display = display or {}
    return [
        Line2D(
            [0], [0], marker="o", linestyle="", markerfacecolor=colors[value],
            markeredgecolor="none", markersize=5, label=display.get(value, value),
        )
        for value in order
    ]


def style_axis(axis: plt.Axes) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xlabel("UMAP 1", fontsize=8)
    axis.set_ylabel("UMAP 2", fontsize=8)
    axis.set_aspect("equal", adjustable="box")
    for spine in axis.spines.values():
        spine.set_visible(False)


def draw_categories(
    axis: plt.Axes,
    coordinates: np.ndarray,
    values: np.ndarray,
    order: list[str],
    palette: dict[str, dict[str, Any]],
) -> None:
    for value in order:
        selected = values == value
        if not selected.any():
            continue
        row = palette[value]
        axis.scatter(
            coordinates[selected, 0], coordinates[selected, 1],
            s=float(row.get("point_size", 0.7)), c=str(row["color"]),
            alpha=float(row.get("alpha", 0.72)), linewidths=0,
            rasterized=True,
        )
    style_axis(axis)


def draw_datasets(
    axis: plt.Axes,
    coordinates: np.ndarray,
    values: np.ndarray,
    order: list[str],
    colors: dict[str, str],
    seed: int,
) -> None:
    permutation = np.random.default_rng(seed).permutation(len(values))
    color_array = np.asarray([colors[value] for value in values], dtype=object)
    axis.scatter(
        coordinates[permutation, 0], coordinates[permutation, 1],
        s=0.68, c=color_array[permutation], alpha=0.68,
        linewidths=0, rasterized=True,
    )
    style_axis(axis)


def load_display_metadata(config: dict[str, Any], expected_ids: np.ndarray) -> pd.DataFrame:
    path = resolve_from_final_xu(config["display_metadata"])
    frame = pd.read_csv(path, low_memory=False)
    required = ["cell_id", "dataset", "donor", "unit_key", config["label_column"]]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Display metadata lacks columns: {missing}")
    if len(frame) != len(expected_ids):
        raise AssertionError("Display metadata cell count changed")
    observed_ids = frame["cell_id"].astype(str).to_numpy()
    if not np.array_equal(observed_ids, expected_ids.astype(str)):
        mismatch = int(np.flatnonzero(observed_ids != expected_ids.astype(str))[0])
        raise AssertionError(f"Display metadata row identity mismatch at row {mismatch}")
    return frame


def fit_umap_one(
    config: dict[str, Any], config_path: Path, hub_slug: str
) -> Path:
    import umap

    root = output_root(config)
    alignment = root / f"hub_{hub_slug}" / "alignment"
    if not (alignment / "alignment_manifest.json").is_file():
        raise FileNotFoundError(f"Run alignment first: {alignment}")
    target = root / f"hub_{hub_slug}" / "umap"
    manifest_path = target / "umap_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        alignment_manifest_path = (
            output_root(config) / f"hub_{hub_slug}" / "alignment" / "alignment_manifest.json"
        )
        display_path = resolve_from_final_xu(config["display_metadata"])
        if (
            manifest.get("schema_version") == UMAP_SCHEMA
            and manifest.get("status") == "complete"
            and verify_hash_map(target, manifest.get("outputs", {}))
            and manifest["config_sha256"] == sha256(config_path)
            and alignment_manifest_path.is_file()
            and manifest.get("alignment_manifest_sha256") == sha256(alignment_manifest_path)
            and display_path.is_file()
            and manifest.get("display_metadata_sha256") == sha256(display_path)
            and manifest.get("benchmark_runner_sha256") == sha256(Path(__file__).resolve())
        ):
            print(f"[{config['benchmark_slug']}/{hub_slug}] UMAP already verified", flush=True)
            return target
        raise RuntimeError(f"Incomplete or stale UMAP directory: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".umap.tmp-", dir=target.parent))
    started = time.time()
    try:
        alignment_manifest = json.loads((alignment / "alignment_manifest.json").read_text())
        if alignment_manifest.get("labels_used_for_fitting") is not False:
            raise AssertionError("Alignment does not establish label-free fitting")
        embedding_path = alignment / "data" / "maxfuse_embedding_float64.npy"
        ids_path = alignment / "data" / "cell_ids.npy"
        embedding = np.load(embedding_path, allow_pickle=False)
        cell_ids = np.load(ids_path, allow_pickle=False)
        label_free_metadata = pd.read_csv(
            alignment / "data" / "cell_metadata_label_free.csv.gz",
            low_memory=False,
        )
        if len(label_free_metadata) != len(embedding):
            raise AssertionError("Label-free alignment metadata row count changed")
        if not np.array_equal(
            label_free_metadata["cell_id"].astype(str).to_numpy(),
            cell_ids.astype(str),
        ):
            raise AssertionError("Label-free alignment metadata cell order changed")
        dataset_order = [row["name"] for row in config["datasets"]]
        standardized = viewwise_standardized_embedding(
            embedding,
            label_free_metadata["dataset"].astype(str).to_numpy(),
            dataset_order,
        )
        umap_config = dict(config["umap"])
        init_kind = umap_config.pop("init_kind")
        init_source: dict[str, Any]
        if init_kind == "ramp_baseline":
            init_path = resolve_from_final_xu(umap_config.pop("init_path"))
            initialization = np.load(init_path, allow_pickle=False).astype(np.float32)
            if initialization.shape != (len(embedding), 2) or not np.isfinite(initialization).all():
                raise AssertionError("Frozen RAMP baseline UMAP initialization is invalid")
            init_source = {"kind": init_kind, "path": str(init_path), "sha256": sha256(init_path)}
        elif init_kind == "gaussian_seed42":
            umap_config.pop("init_path", None)
            initialization = np.random.default_rng(42).normal(
                0.0, 1e-4, size=(len(embedding), 2)
            ).astype(np.float32)
            init_source = {
                "kind": init_kind,
                "description": "seed-42 Gaussian N(0, 1e-4), matching the BM RAMP figure",
            }
        else:
            raise ValueError(f"Unknown UMAP initialization: {init_kind}")
        parameters = {
            "n_neighbors": int(umap_config["n_neighbors"]),
            "min_dist": float(umap_config["min_dist"]),
            "metric": "euclidean",
            "random_state": 42,
            "transform_seed": 42,
            "n_jobs": 1,
            "low_memory": True,
        }
        print(
            f"[{config['benchmark_slug']}/{hub_slug}] UMAP on {len(embedding):,} cells x "
            f"{standardized.shape[1]} dimensions",
            flush=True,
        )
        coordinates = umap.UMAP(
            **parameters, init=initialization.copy()
        ).fit_transform(standardized)
        coordinates = np.asarray(coordinates, dtype=np.float32)
        if coordinates.shape != (len(embedding), 2) or not np.isfinite(coordinates).all():
            raise RuntimeError("Invalid UMAP coordinates")

        # Labels are intentionally opened only after UMAP coordinates are frozen.
        display = load_display_metadata(config, cell_ids)
        label_column = config["label_column"]
        labels = display[label_column].fillna(config.get("missing_label", "Non-joint")).astype(str).to_numpy()
        datasets = display["dataset"].astype(str).to_numpy()
        palette_rows = {str(row["label"]): row for row in config["palette"]}
        if set(np.unique(labels)) != set(palette_rows):
            raise AssertionError(
                f"Display label/palette mismatch: labels={sorted(np.unique(labels))}, "
                f"palette={sorted(palette_rows)}"
            )
        legend_order = [
            row["label"] for row in sorted(config["palette"], key=lambda row: row["legend_order"])
        ]
        draw_order = [
            row["label"] for row in sorted(config["palette"], key=lambda row: row["draw_order"])
        ]
        dataset_colors = dict(config["dataset_colors"])
        if set(np.unique(datasets)) != set(dataset_order):
            raise AssertionError("Display dataset contract changed")

        data_dir = temporary / "data"
        figures = temporary / "figures"
        data_dir.mkdir()
        figures.mkdir()
        coords_path = data_dir / "umap_coordinates.npy"
        np.save(coords_path, coordinates)
        coordinate_frame = display[["cell_id", "dataset", "donor", "unit_key", label_column]].copy()
        coordinate_frame.insert(0, "row_id", np.arange(len(display), dtype=np.int64))
        coordinate_frame["UMAP1"] = coordinates[:, 0]
        coordinate_frame["UMAP2"] = coordinates[:, 1]
        csv_path = data_dir / "umap_coordinates.csv.gz"
        deterministic_gzip_csv(coordinate_frame, csv_path)

        figure, axes = plt.subplots(1, 2, figsize=(15.2, 7.0), dpi=180)
        draw_categories(axes[0], coordinates, labels, draw_order, palette_rows)
        draw_datasets(
            axes[1], coordinates, datasets, dataset_order, dataset_colors,
            stable_seed(int(config["base_seed"]), hub_slug, "dataset-draw"),
        )
        axes[0].set_title("Cell type (post hoc)", fontsize=13, fontweight="bold")
        axes[1].set_title("Dataset", fontsize=13, fontweight="bold")
        hub_name = next(row["name"] for row in config["datasets"] if row["slug"] == hub_slug)
        figure.suptitle(
            f"{config['benchmark_title']} — MaxFuse Archive three-view workflow "
            f"(hub: {hub_name})",
            fontsize=15, fontweight="bold", y=0.985,
        )
        label_colors = {label: str(palette_rows[label]["color"]) for label in legend_order}
        figure.legend(
            handles=legend_handles(legend_order, label_colors),
            loc="lower center", bbox_to_anchor=(0.36, 0.012),
            ncol=min(6, max(3, len(legend_order))), frameon=False, fontsize=7.2,
        )
        dataset_display = {row["name"]: row.get("display", row["name"]) for row in config["datasets"]}
        figure.legend(
            handles=legend_handles(dataset_order, dataset_colors, dataset_display),
            loc="lower center", bbox_to_anchor=(0.80, 0.028),
            ncol=3, frameon=False, fontsize=8.0,
        )
        figure.subplots_adjust(left=0.035, right=0.995, top=0.91, bottom=0.18, wspace=0.035)
        png_path = figures / f"{config['benchmark_slug']}_maxfuse_hub_{hub_slug}_umap.png"
        pdf_path = figures / f"{config['benchmark_slug']}_maxfuse_hub_{hub_slug}_umap.pdf"
        figure.savefig(png_path, dpi=300, bbox_inches="tight")
        figure.savefig(pdf_path, bbox_inches="tight")
        plt.close(figure)

        display_path = resolve_from_final_xu(config["display_metadata"])
        outputs = {
            "data/umap_coordinates.npy": sha256(coords_path),
            "data/umap_coordinates.csv.gz": sha256(csv_path),
            f"figures/{png_path.name}": sha256(png_path),
            f"figures/{pdf_path.name}": sha256(pdf_path),
        }
        manifest = {
            "schema_version": UMAP_SCHEMA,
            "status": "complete",
            "benchmark": config["benchmark_slug"],
            "hub_slug": hub_slug,
            "hub_dataset": hub_name,
            "cells": int(len(embedding)),
            "embedding_dimensions_available": int(embedding.shape[1]),
            "embedding_dimensions_used": int(standardized.shape[1]),
            "pre_umap_processing": (
                "each dataset/view's final Archive GCCA embedding population-centered "
                "and population-SD-scaled over all displayed cells, then concatenated; "
                "no additional PCA"
            ),
            "labels_used_for_umap_fitting": False,
            "labels_loaded_after_umap_fitting": True,
            "umap_parameters": parameters,
            "initialization": init_source,
            "drawing": {
                "celltype_draw_order_bottom_to_top": draw_order,
                "celltype_legend_order": legend_order,
                "dataset_points_randomly_permuted_for_unbiased_layering": True,
            },
            "alignment_manifest": str(alignment / "alignment_manifest.json"),
            "alignment_manifest_sha256": sha256(alignment / "alignment_manifest.json"),
            "display_metadata": str(display_path),
            "display_metadata_sha256": sha256(display_path),
            "config": str(config_path),
            "config_sha256": sha256(config_path),
            "benchmark_runner": str(Path(__file__).resolve()),
            "benchmark_runner_sha256": sha256(Path(__file__).resolve()),
            "runtime_seconds": time.time() - started,
            "coordinate_buffer_sha256_float32": buffer_sha256(coordinates, np.float32),
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "umap-learn": version("umap-learn"),
                "scikit-learn": version("scikit-learn"),
                "numba": version("numba"),
                "pynndescent": version("pynndescent"),
                "matplotlib": version("matplotlib"),
            },
            "outputs": outputs,
        }
        (temporary / "umap_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        temporary.replace(target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"[{config['benchmark_slug']}/{hub_slug}] UMAP figure saved", flush=True)
    return target


def combined_figure(config: dict[str, Any], config_path: Path) -> None:
    root = output_root(config)
    summary = root / "summary"
    figures = summary / "figures"
    data = summary / "data"
    figures.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    slugs = [row["slug"] for row in config["datasets"]]
    names = {row["slug"]: row["name"] for row in config["datasets"]}
    first_ids = np.load(
        root / f"hub_{slugs[0]}" / "alignment" / "data" / "cell_ids.npy",
        allow_pickle=False,
    )
    display = load_display_metadata(config, first_ids)
    labels = display[config["label_column"]].fillna(
        config.get("missing_label", "Non-joint")
    ).astype(str).to_numpy()
    datasets = display["dataset"].astype(str).to_numpy()
    palette_rows = {str(row["label"]): row for row in config["palette"]}
    legend_order = [
        row["label"] for row in sorted(config["palette"], key=lambda row: row["legend_order"])
    ]
    draw_order = [
        row["label"] for row in sorted(config["palette"], key=lambda row: row["draw_order"])
    ]
    dataset_order = [row["name"] for row in config["datasets"]]
    dataset_colors = dict(config["dataset_colors"])

    figure, axes = plt.subplots(3, 2, figsize=(13.4, 18.2), dpi=160)
    coordinate_hashes: dict[str, str] = {}
    for row_index, slug in enumerate(slugs):
        coordinate_path = root / f"hub_{slug}" / "umap" / "data" / "umap_coordinates.npy"
        coordinates = np.load(coordinate_path, allow_pickle=False)
        coordinate_hashes[slug] = sha256(coordinate_path)
        draw_categories(axes[row_index, 0], coordinates, labels, draw_order, palette_rows)
        draw_datasets(
            axes[row_index, 1], coordinates, datasets, dataset_order, dataset_colors,
            stable_seed(int(config["base_seed"]), slug, "dataset-draw"),
        )
        axes[row_index, 0].set_title(
            f"Hub: {names[slug]} — cell type", fontsize=11.5, fontweight="bold"
        )
        axes[row_index, 1].set_title(
            f"Hub: {names[slug]} — dataset", fontsize=11.5, fontweight="bold"
        )
    label_colors = {label: str(palette_rows[label]["color"]) for label in legend_order}
    dataset_display = {row["name"]: row.get("display", row["name"]) for row in config["datasets"]}
    figure.legend(
        handles=legend_handles(legend_order, label_colors),
        loc="lower center", bbox_to_anchor=(0.38, 0.018),
        ncol=min(6, max(3, len(legend_order))), frameon=False, fontsize=7.4,
    )
    figure.legend(
        handles=legend_handles(dataset_order, dataset_colors, dataset_display),
        loc="lower center", bbox_to_anchor=(0.80, 0.022),
        ncol=3, frameon=False, fontsize=8.0,
    )
    figure.suptitle(
        f"{config['benchmark_title']} — MaxFuse Archive three-view workflow "
        "(three-hub sensitivity)",
        fontsize=15.5, fontweight="bold", y=0.995,
    )
    figure.subplots_adjust(left=0.035, right=0.995, top=0.965, bottom=0.075, wspace=0.035, hspace=0.12)
    png_path = figures / f"{config['benchmark_slug']}_maxfuse_all_hubs_umap.png"
    pdf_path = figures / f"{config['benchmark_slug']}_maxfuse_all_hubs_umap.pdf"
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)

    audits = []
    for slug in slugs:
        manifest = json.loads(
            (root / f"hub_{slug}" / "alignment" / "alignment_manifest.json").read_text()
        )
        audit = manifest["formal_fit_audit"]
        audits.append({
            "hub_slug": slug,
            "hub_dataset": names[slug],
            "triplets": audit["triplets"],
            "training_fit_mean_matched_component_correlation": audit[
                "mean_matched_component_correlation"
            ],
            "outer_left_reused_records": audit["outer_reused_records"][audit["view_order"][1]],
            "outer_right_reused_records": audit["outer_reused_records"][audit["view_order"][2]],
            "outer_left_maximum_multiplicity": audit["outer_maximum_multiplicity"][audit["view_order"][1]],
            "outer_right_maximum_multiplicity": audit["outer_maximum_multiplicity"][audit["view_order"][2]],
        })
    audit_path = data / "all_hubs_alignment_audit.csv"
    pd.DataFrame(audits).to_csv(audit_path, index=False)
    manifest = {
        "schema_version": "final_xu.maxfuse_all_hubs_summary.v2",
        "status": "complete",
        "benchmark": config["benchmark_slug"],
        "hubs": slugs,
        "hub_selection": "none; all three datasets are reported",
        "matched_component_correlation_role": (
            "training-fit diagnostic on the same triplets used by GCCA; "
            "not an independent benchmark metric"
        ),
        "labels_used_for_alignment_or_umap_fitting": False,
        "coordinate_sha256": coordinate_hashes,
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "outputs": {
            f"figures/{png_path.name}": sha256(png_path),
            f"figures/{pdf_path.name}": sha256(pdf_path),
            f"data/{audit_path.name}": sha256(audit_path),
        },
    }
    (summary / "summary_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(f"[{config['benchmark_slug']}] combined three-hub figure saved", flush=True)


def run_umaps(config: dict[str, Any], config_path: Path, hub: str) -> None:
    slugs = [row["slug"] for row in config["datasets"]]
    selected = slugs if hub == "all" else [hub]
    unknown = set(selected) - set(slugs)
    if unknown:
        raise KeyError(f"Unknown hub(s): {sorted(unknown)}")
    for slug in selected:
        fit_umap_one(config, config_path, slug)
    if hub == "all":
        combined_figure(config, config_path)


def validate_config_inputs(config: dict[str, Any], config_path: Path) -> None:
    """Read and validate configured inputs without writing benchmark artifacts."""

    shared, metadata, active_blocks, active_markers, sources = load_label_free_inputs(config)
    pairwise_payload, pairwise_path = load_pairwise_config(config)
    expected_slugs = {row["slug"] for row in config["datasets"]}
    if set(active_blocks) != expected_slugs or set(active_markers) != expected_slugs:
        raise AssertionError("active panel dictionaries do not match configured datasets")
    if pairwise_payload["schema_version"] != "maxfuse_repro.PairwiseConfig.v1":
        raise AssertionError("pairwise configuration schema changed during validation")
    print(
        f"[{config['benchmark_slug']}] CONFIG_VALID: {len(metadata):,} cells x "
        f"{shared.shape[1]} shared markers; active widths "
        f"{ {slug: block.shape[1] for slug, block in active_blocks.items()} }; "
        f"{len(sources)} source records; pairwise={pairwise_path.name}; "
        f"config={config_path.name}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "align", "umap", "all"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--hub", default="all")
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    if args.command == "validate":
        validate_config_inputs(config, config_path)
        return
    if args.command in {"align", "all"}:
        run_alignments(config, config_path, args.hub)
    if args.command in {"umap", "all"}:
        run_umaps(config, config_path, args.hub)


if __name__ == "__main__":
    main()
