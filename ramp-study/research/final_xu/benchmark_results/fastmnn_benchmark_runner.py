#!/usr/bin/env python3
"""Shared, label-free fastMNN benchmark orchestration.

Benchmark folders own cell selection, marker harmonization, and assay-specific
preprocessing.  This runner authenticates those artifacts, calls the locked
algorithm/fastmnn implementation once across all configured datasets, and
draws UMAP directly from the corrected latent embedding.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
FINAL_XU = HERE.parent
ALGORITHM_RUNNER = FINAL_XU / "algorithm" / "fastmnn" / "run_fastmnn.R"
BENCHMARK_SCHEMA = "final_xu.fastmnn_benchmark.v1"
PREPROCESSING_SCHEMA = "final_xu.fastmnn_preprocessing.v1"
ALGORITHM_SCHEMA = "final_xu.fastmnn_repro.v1"
UMAP_SCHEMA = "final_xu.fastmnn_umap.v1"
COMPLETE_SCHEMA = "final_xu.fastmnn_benchmark_complete.v1"
METHOD_LABEL = "fastMNN (batchelor 1.18.1; corrected embedding)"


cache_root = Path(tempfile.gettempdir()) / "final_xu_fastmnn_benchmark_cache"
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_root / "numba"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_buffer(array: np.ndarray, dtype: np.dtype[Any]) -> str:
    contiguous = np.ascontiguousarray(array, dtype=dtype)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def write_json_atomic(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def deterministic_gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(filename=str(path), mode="wb", mtime=0) as stream:
        frame.to_csv(stream, index=False)


def resolve(base: Path, value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve(strict=True)


def validate_config(config_path: Path) -> tuple[dict[str, Any], Path]:
    config_path = config_path.resolve(strict=True)
    config = load_json(config_path)
    required = {
        "schema_version",
        "benchmark_slug",
        "benchmark_title",
        "method_label",
        "expected_cells",
        "expected_embedding_dimensions",
        "cell_id_column",
        "batch_column",
        "sample_column",
        "label_column",
        "missing_label",
        "datasets",
        "markers",
        "paths",
        "umap",
        "palette",
        "dataset_colors",
    }
    if set(config) != required:
        raise ValueError(
            f"Benchmark config fields differ from v1 schema: "
            f"missing={sorted(required - set(config))}, extra={sorted(set(config) - required)}"
        )
    if config["schema_version"] != BENCHMARK_SCHEMA:
        raise ValueError("Unsupported benchmark schema")
    if config["method_label"] != METHOD_LABEL:
        raise ValueError("Method label differs from the audited algorithm name")
    if int(config["expected_cells"]) < 1:
        raise ValueError("expected_cells must be positive")
    if int(config["expected_embedding_dimensions"]) < 2:
        raise ValueError("expected_embedding_dimensions must be at least two")

    datasets = config["datasets"]
    if not isinstance(datasets, list) or len(datasets) < 2:
        raise ValueError("At least two ordered datasets are required")
    names = [str(row["name"]) for row in datasets]
    counts = [int(row["expected_cells"]) for row in datasets]
    if len(names) != len(set(names)) or any(count < 1 for count in counts):
        raise ValueError("Dataset names/counts are invalid")
    if sum(counts) != int(config["expected_cells"]):
        raise ValueError("Dataset counts do not sum to expected_cells")
    markers = [str(marker) for marker in config["markers"]]
    if len(markers) < 3 or len(markers) != len(set(markers)):
        raise ValueError("Marker list must contain at least three unique entries")
    if int(config["expected_embedding_dimensions"]) != min(50, len(markers) - 1):
        raise ValueError("Expected fastMNN dimensions do not match the locked d policy")

    path_fields = {
        "preprocessing_builder",
        "combined_input",
        "marker_file",
        "preprocessing_manifest",
        "algorithm_config",
        "alignment_output",
        "display_metadata",
        "umap_output",
    }
    if set(config["paths"]) != path_fields:
        raise ValueError("paths fields differ from the v1 schema")

    umap_required = {
        "n_neighbors",
        "min_dist",
        "metric",
        "random_state",
        "transform_seed",
        "n_jobs",
        "low_memory",
        "initialization",
        "initialization_sd",
    }
    if set(config["umap"]) != umap_required:
        raise ValueError("UMAP fields differ from the v1 schema")
    if config["umap"]["metric"] != "euclidean":
        raise ValueError("Canonical fastMNN UMAP uses Euclidean distance")
    if config["umap"]["initialization"] not in {"seeded_gaussian", "spectral"}:
        raise ValueError("UMAP initialization must be seeded_gaussian or spectral")
    if config["umap"]["initialization"] == "seeded_gaussian":
        if float(config["umap"]["initialization_sd"]) <= 0:
            raise ValueError("seeded_gaussian requires a positive initialization_sd")
    elif config["umap"]["initialization_sd"] is not None:
        raise ValueError("spectral initialization requires initialization_sd=null")

    palette_labels = [str(row["label"]) for row in config["palette"]]
    if len(palette_labels) != len(set(palette_labels)):
        raise ValueError("Palette labels must be unique")
    if set(config["dataset_colors"]) != set(names):
        raise ValueError("Dataset colors must cover every configured dataset exactly")
    return config, config_path


def method_root(config_path: Path) -> Path:
    return config_path.parent


def configured_paths(config: dict[str, Any], config_path: Path) -> dict[str, Path]:
    base = method_root(config_path)
    result: dict[str, Path] = {}
    for key, value in config["paths"].items():
        candidate = Path(str(value))
        if not candidate.is_absolute():
            candidate = base / candidate
        result[key] = candidate.resolve(strict=False)
    return result


def marker_lines(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"Marker file is empty or duplicated: {path}")
    return values


def validate_preprocessing(
    config: dict[str, Any], config_path: Path, require_complete: bool = True
) -> dict[str, Any] | None:
    paths = configured_paths(config, config_path)
    required = [
        paths["combined_input"],
        paths["marker_file"],
        paths["preprocessing_manifest"],
        paths["algorithm_config"],
    ]
    if not all(path.is_file() for path in required):
        if require_complete:
            missing = [str(path) for path in required if not path.is_file()]
            raise FileNotFoundError(f"Incomplete preprocessing: {missing}")
        return None

    manifest = load_json(paths["preprocessing_manifest"])
    if manifest.get("schema_version") != PREPROCESSING_SCHEMA:
        raise ValueError("Preprocessing manifest does not use the fastMNN v1 schema")
    expected_names = [row["name"] for row in config["datasets"]]
    observed_names = [row.get("dataset") for row in manifest.get("datasets", [])]
    if observed_names != expected_names:
        raise ValueError("Preprocessing dataset order changed")
    if manifest.get("dataset_order_is_progressive_merge_order") is not True:
        raise ValueError("Preprocessing does not bind the progressive merge order")
    for field in (
        "shared_marker_selection_used_celltype_or_outcome_labels",
        "merge_order_selected_using_posthoc_labels_or_metrics",
        "canonical_parameters_selected_using_posthoc_labels_or_metrics",
    ):
        if manifest.get(field) is not False:
            raise ValueError(f"Preprocessing label-separation declaration failed: {field}")
    if sha256(paths["combined_input"]) != manifest.get("combined_input_sha256"):
        raise ValueError("Combined input hash differs from preprocessing manifest")
    if sha256(paths["marker_file"]) != manifest.get("correction_marker_sha256"):
        raise ValueError("Marker file hash differs from preprocessing manifest")
    if marker_lines(paths["marker_file"]) != list(config["markers"]):
        raise ValueError("Marker order differs from benchmark config")

    preprocessing_dir = paths["preprocessing_manifest"].parent
    build_manifest_path = preprocessing_dir / "preprocessing_build_manifest.json"
    preprocessing_complete_path = preprocessing_dir / "COMPLETE.json"
    if not build_manifest_path.is_file() or not preprocessing_complete_path.is_file():
        raise FileNotFoundError("Preprocessing build manifest/COMPLETE record is absent")
    preprocessing_complete = load_json(preprocessing_complete_path)
    if preprocessing_complete.get("status") != "complete":
        raise ValueError("Preprocessing COMPLETE record is not complete")
    if preprocessing_complete.get("preprocessing_manifest_sha256") != sha256(
        paths["preprocessing_manifest"]
    ):
        raise ValueError("Preprocessing COMPLETE record does not bind the manifest")
    if preprocessing_complete.get("preprocessing_build_manifest_sha256") != sha256(
        build_manifest_path
    ):
        raise ValueError("Preprocessing COMPLETE record does not bind the build manifest")
    if preprocessing_complete.get("algorithm_config_sha256") != sha256(
        paths["algorithm_config"]
    ):
        raise ValueError("Preprocessing COMPLETE record does not bind the algorithm config")

    for path_field, hash_field in (
        ("marker_harmonization_manifest_file", "marker_harmonization_manifest_sha256"),
        ("cell_selection_manifest_file", "cell_selection_manifest_sha256"),
    ):
        artifact = resolve(preprocessing_dir, str(manifest[path_field]))
        if sha256(artifact) != manifest[hash_field]:
            raise ValueError(f"Preprocessing provenance hash failed: {artifact}")
    for dataset_record in manifest["datasets"]:
        matrix_path = resolve(
            preprocessing_dir, str(dataset_record["analysis_ready_matrix_file"])
        )
        if sha256(matrix_path) != dataset_record["analysis_ready_matrix_sha256"]:
            raise ValueError(f"Dataset analysis-ready matrix hash failed: {matrix_path}")

    character_types = {
        config["cell_id_column"]: str,
        config["batch_column"]: str,
        "donor": str,
        config["sample_column"]: str,
    }
    combined = pd.read_csv(
        paths["combined_input"], dtype=character_types, low_memory=False
    )
    if len(combined) != int(config["expected_cells"]):
        raise ValueError("Analysis-ready cell count changed")
    if combined[config["cell_id_column"]].duplicated().any():
        raise ValueError("Analysis-ready cell IDs are duplicated")
    observed_order = pd.unique(combined[config["batch_column"]]).tolist()
    if observed_order != expected_names:
        raise ValueError("Analysis-ready dataset order changed")
    observed_counts = combined.groupby(config["batch_column"], sort=False).size().to_dict()
    expected_counts = {
        row["name"]: int(row["expected_cells"]) for row in config["datasets"]
    }
    if observed_counts != expected_counts:
        raise ValueError("Analysis-ready dataset counts changed")
    numeric = combined[config["markers"]].to_numpy(np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("Analysis-ready matrix contains non-finite values")
    l2_norm = np.linalg.norm(numeric, axis=1)
    zero_l2_count = int(np.sum(l2_norm == 0))
    if int(manifest.get("zero_l2_cell_count", -1)) != zero_l2_count:
        raise ValueError("Preprocessing zero-L2 declaration differs from the input")
    zero_l2_by_dataset = {
        dataset: int(np.sum(l2_norm[combined[config["batch_column"]].to_numpy() == dataset] == 0))
        for dataset in expected_names
    }
    if manifest.get("zero_l2_cell_count_by_dataset") != zero_l2_by_dataset:
        raise ValueError("Per-dataset zero-L2 declarations differ from the input")
    if np.any(np.std(numeric, axis=0) < 1e-12):
        raise ValueError("Analysis-ready matrix contains a constant marker")
    selection_path = resolve(
        preprocessing_dir, str(manifest["cell_selection_manifest_file"])
    )
    selection = pd.read_csv(selection_path, dtype=str, low_memory=False)
    selection_id = str(manifest["cell_selection_id_column"])
    if len(selection) != len(combined) or not np.array_equal(
        selection[selection_id].astype(str).to_numpy(),
        combined[config["cell_id_column"]].astype(str).to_numpy(),
    ):
        raise ValueError("Analysis-ready rows differ from the frozen cell selection")
    for column in (config["batch_column"], "donor"):
        if column in selection and not np.array_equal(
            selection[column].astype(str).to_numpy(),
            combined[column].astype(str).to_numpy(),
        ):
            raise ValueError(f"Analysis-ready {column} differs from frozen selection")

    algorithm_config = load_json(paths["algorithm_config"])
    if algorithm_config.get("schema_version") != ALGORITHM_SCHEMA:
        raise ValueError("Algorithm config schema changed")
    if algorithm_config.get("method_label") != METHOD_LABEL:
        raise ValueError("Algorithm config method label changed")
    if algorithm_config.get("batch_levels") != expected_names:
        raise ValueError("Algorithm config batch order changed")
    if algorithm_config.get("cell_id_column") != config["cell_id_column"]:
        raise ValueError("Algorithm config cell ID column changed")
    if algorithm_config.get("batch_column") != config["batch_column"]:
        raise ValueError("Algorithm config batch column changed")
    if algorithm_config.get("sample_column") != config["sample_column"]:
        raise ValueError("Algorithm config sample column changed")
    scale = algorithm_config.get("input_scale", {})
    expected_hashes = {
        "input_sha256": sha256(paths["combined_input"]),
        "marker_file_sha256": sha256(paths["marker_file"]),
        "preprocessing_record_sha256": sha256(paths["preprocessing_manifest"]),
    }
    for field, expected in expected_hashes.items():
        if scale.get(field) != expected:
            raise ValueError(f"Algorithm config {field} is stale")

    header = pd.read_csv(paths["combined_input"], nrows=0).columns.tolist()
    required_columns = [
        config["cell_id_column"],
        config["batch_column"],
        "donor",
        config["sample_column"],
        *config["markers"],
    ]
    if header != required_columns:
        raise ValueError(f"Analysis-ready input columns/order changed: {header}")
    return manifest


def run_preprocessing(config: dict[str, Any], config_path: Path) -> None:
    existing = validate_preprocessing(config, config_path, require_complete=False)
    if existing is not None:
        print(f"[{config['benchmark_slug']}] preprocessing already verified", flush=True)
        return
    paths = configured_paths(config, config_path)
    builder = paths["preprocessing_builder"]
    if not builder.is_file():
        raise FileNotFoundError(f"Preprocessing builder is absent: {builder}")
    print(f"[{config['benchmark_slug']}] building method-owned preprocessing", flush=True)
    if builder.suffix.lower() == ".r":
        command = ["Rscript", "--vanilla", str(builder)]
    elif builder.suffix.lower() == ".py":
        command = [sys.executable, str(builder)]
    else:
        raise ValueError(f"Unsupported preprocessing-builder suffix: {builder.suffix}")
    subprocess.run(command, check=True, cwd=builder.parent)
    validate_preprocessing(config, config_path, require_complete=True)


def verify_hash_entries(root: Path, entries: list[dict[str, Any]]) -> None:
    for row in entries:
        path = root / str(row["file"])
        if not path.is_file() or sha256(path) != row["sha256"]:
            raise ValueError(f"Output hash failed: {path}")


def validate_alignment(
    config: dict[str, Any], config_path: Path, require_complete: bool = True
) -> dict[str, Any] | None:
    paths = configured_paths(config, config_path)
    output = paths["alignment_output"]
    manifest_path = output / "manifest.json"
    complete_path = output / "COMPLETE.json"
    if not manifest_path.is_file() or not complete_path.is_file():
        if require_complete:
            raise FileNotFoundError(f"Incomplete fastMNN alignment: {output}")
        if output.exists():
            raise RuntimeError(f"Partial fastMNN alignment exists: {output}")
        return None
    manifest = load_json(manifest_path)
    complete = load_json(complete_path)
    if complete.get("status") != "complete" or complete.get("manifest_sha256") != sha256(manifest_path):
        raise ValueError("Alignment COMPLETE.json does not authenticate manifest.json")
    if manifest.get("schema_version") != ALGORITHM_SCHEMA or manifest.get("status") != "complete":
        raise ValueError("Alignment manifest schema/status changed")
    if manifest.get("method_label") != METHOD_LABEL:
        raise ValueError("Alignment method label changed")
    paths_now = configured_paths(config, config_path)
    input_record = manifest.get("input", {})
    if input_record.get("sha256") != sha256(paths_now["combined_input"]):
        raise ValueError("Alignment was not fit on the authenticated benchmark input")
    if input_record.get("marker_file_sha256") != sha256(paths_now["marker_file"]):
        raise ValueError("Alignment marker file changed")
    if input_record.get("preprocessing_record_sha256") != sha256(paths_now["preprocessing_manifest"]):
        raise ValueError("Alignment preprocessing record changed")
    if input_record.get("markers") != config["markers"]:
        raise ValueError("Alignment marker order changed")
    expected_batches = [row["name"] for row in config["datasets"]]
    if input_record.get("batch_levels_in_progressive_merge_order") != expected_batches:
        raise ValueError("Alignment progressive merge order changed")
    algorithm = manifest.get("algorithm", {})
    if algorithm.get("package") != "batchelor" or algorithm.get("version") != "1.18.1":
        raise ValueError("Alignment is not the locked batchelor 1.18.1 fit")
    if algorithm.get("reconstructed_assay_exported") is not False:
        raise ValueError("Noncanonical reconstructed assay was exported")
    if int(algorithm.get("actual_d", -1)) != int(config["expected_embedding_dimensions"]):
        raise ValueError("Corrected embedding dimensionality changed")
    verify_hash_entries(output, manifest.get("output_files", []))

    metadata_path = output / "cell_metadata_label_free.csv.gz"
    embedding_path = output / "fastmnn_corrected_embedding.csv.gz"
    if not metadata_path.is_file() or not embedding_path.is_file():
        raise FileNotFoundError("Primary alignment outputs are absent")
    metadata = pd.read_csv(metadata_path, dtype=str)
    if metadata.columns.tolist() != ["cell_id", "batch", "sample"]:
        raise ValueError("Label-free alignment metadata columns changed")
    if len(metadata) != int(config["expected_cells"]) or metadata["cell_id"].duplicated().any():
        raise ValueError("Alignment cell identities/count changed")
    observed_order = pd.unique(metadata["batch"].astype(str)).tolist()
    if observed_order != expected_batches:
        raise ValueError("Alignment cell batch order changed")
    source_identity = pd.read_csv(
        paths_now["combined_input"],
        usecols=[
            config["cell_id_column"], config["batch_column"], config["sample_column"]
        ],
        dtype=str,
    )
    if not np.array_equal(
        metadata["cell_id"].astype(str).to_numpy(),
        source_identity[config["cell_id_column"]].astype(str).to_numpy(),
    ):
        raise ValueError("Alignment cell order differs from analysis-ready input")
    if not np.array_equal(
        metadata["batch"].astype(str).to_numpy(),
        source_identity[config["batch_column"]].astype(str).to_numpy(),
    ) or not np.array_equal(
        metadata["sample"].astype(str).to_numpy(),
        source_identity[config["sample_column"]].astype(str).to_numpy(),
    ):
        raise ValueError("Alignment batch/sample provenance changed")
    embedding = pd.read_csv(embedding_path, low_memory=False)
    expected_dimensions = int(config["expected_embedding_dimensions"])
    expected_embedding_columns = [
        "cell_id", *[f"fastMNN_{index:02d}" for index in range(1, expected_dimensions + 1)]
    ]
    if embedding.columns.tolist() != expected_embedding_columns:
        raise ValueError("Corrected embedding columns changed")
    if len(embedding) != len(metadata) or not np.array_equal(
        embedding["cell_id"].astype(str).to_numpy(),
        metadata["cell_id"].astype(str).to_numpy(),
    ):
        raise ValueError("Corrected embedding cell order changed")
    embedding_values = embedding.iloc[:, 1:].to_numpy(np.float64)
    if not np.isfinite(embedding_values).all() or np.any(
        np.std(embedding_values, axis=0) < 1e-12
    ):
        raise ValueError("Corrected embedding is non-finite or degenerate")
    return manifest


def run_alignment(config: dict[str, Any], config_path: Path) -> None:
    validate_preprocessing(config, config_path, require_complete=True)
    existing = validate_alignment(config, config_path, require_complete=False)
    if existing is not None:
        print(f"[{config['benchmark_slug']}] alignment already verified", flush=True)
        return
    paths = configured_paths(config, config_path)
    paths["alignment_output"].parent.mkdir(parents=True, exist_ok=True)
    command = [
        "Rscript",
        "--vanilla",
        str(ALGORITHM_RUNNER),
        "--input",
        str(paths["combined_input"]),
        "--markers",
        str(paths["marker_file"]),
        "--config",
        str(paths["algorithm_config"]),
        "--output",
        str(paths["alignment_output"]),
    ]
    print(
        f"[{config['benchmark_slug']}] canonical fastMNN on "
        f"{config['expected_cells']:,} cells x {len(config['markers'])} markers",
        flush=True,
    )
    subprocess.run(command, check=True, cwd=method_root(config_path))
    validate_alignment(config, config_path, require_complete=True)


def style_axis(axis: Any) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xlabel("UMAP 1", fontsize=9)
    axis.set_ylabel("UMAP 2", fontsize=9)
    axis.set_aspect("equal", adjustable="box")
    for spine in axis.spines.values():
        spine.set_visible(False)


def legend_handles(order: list[str], colors: dict[str, str]) -> list[Any]:
    from matplotlib.lines import Line2D

    return [
        Line2D(
            [0], [0], marker="o", linestyle="", markerfacecolor=colors[value],
            markeredgecolor="none", markersize=5, label=value,
        )
        for value in order
    ]


def validate_umap(
    config: dict[str, Any], config_path: Path, require_complete: bool = True
) -> dict[str, Any] | None:
    paths = configured_paths(config, config_path)
    output = paths["umap_output"]
    manifest_path = output / "umap_manifest.json"
    complete_path = output / "COMPLETE.json"
    if not manifest_path.is_file() or not complete_path.is_file():
        if require_complete:
            raise FileNotFoundError(f"Incomplete fastMNN UMAP: {output}")
        if (output / "data").exists() or (output / "figures").exists():
            raise RuntimeError(f"Partial fastMNN UMAP exists: {output}")
        return None
    manifest = load_json(manifest_path)
    complete = load_json(complete_path)
    if complete.get("status") != "complete" or complete.get("manifest_sha256") != sha256(manifest_path):
        raise ValueError("UMAP COMPLETE.json does not authenticate umap_manifest.json")
    if manifest.get("schema_version") != UMAP_SCHEMA or manifest.get("status") != "complete":
        raise ValueError("UMAP manifest schema/status changed")
    if manifest.get("pre_umap_processing") != "direct corrected fastMNN embedding; no z-score; no PCA":
        raise ValueError("UMAP does not use the canonical direct corrected embedding")
    if manifest.get("labels_used_for_umap_fitting") is not False or manifest.get("labels_loaded_after_coordinates_frozen") is not True:
        raise ValueError("UMAP label-separation audit failed")
    alignment_manifest = paths["alignment_output"] / "manifest.json"
    if manifest.get("alignment_manifest_sha256") != sha256(alignment_manifest):
        raise ValueError("UMAP alignment provenance changed")
    if manifest.get("config_sha256") != sha256(config_path):
        raise ValueError("UMAP benchmark config changed")
    if manifest.get("display_metadata_sha256") != sha256(paths["display_metadata"]):
        raise ValueError("UMAP post-hoc display metadata changed")
    if manifest.get("runner_sha256") != sha256(Path(__file__).resolve()):
        raise ValueError("UMAP shared runner changed")
    expected_parameters = {
        key: value
        for key, value in config["umap"].items()
        if key not in {"initialization", "initialization_sd"}
    }
    if manifest.get("umap_parameters") != expected_parameters:
        raise ValueError("UMAP parameters differ from benchmark config")
    if manifest.get("initialization", {}).get("kind") != config["umap"]["initialization"]:
        raise ValueError("UMAP initialization differs from benchmark config")
    for relative, expected in manifest.get("outputs", {}).items():
        path = output / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"UMAP output hash failed: {path}")
    coordinates = pd.read_csv(output / "data" / "umap_coordinates.csv.gz", low_memory=False)
    if len(coordinates) != int(config["expected_cells"]) or coordinates["cell_id"].astype(str).duplicated().any():
        raise ValueError("UMAP coordinate identities/count changed")
    if not np.isfinite(coordinates[["UMAP1", "UMAP2"]].to_numpy(float)).all():
        raise ValueError("UMAP coordinates contain non-finite values")
    label_free = pd.read_csv(
        paths["alignment_output"] / "cell_metadata_label_free.csv.gz", dtype=str
    )
    if not np.array_equal(
        coordinates["cell_id"].astype(str).to_numpy(),
        label_free["cell_id"].astype(str).to_numpy(),
    ):
        raise ValueError("UMAP cell order differs from alignment")
    npy_coordinates = np.load(output / "data" / "umap_coordinates.npy", allow_pickle=False)
    csv_coordinates = coordinates[["UMAP1", "UMAP2"]].to_numpy(np.float32)
    if npy_coordinates.dtype != np.float32 or not np.array_equal(
        npy_coordinates, csv_coordinates
    ):
        raise ValueError("UMAP CSV and NPY coordinate buffers differ")
    if manifest.get("coordinate_buffer_sha256_float32") != sha256_buffer(
        npy_coordinates, np.float32
    ):
        raise ValueError("UMAP coordinate buffer hash changed")
    return manifest


def run_umap(config: dict[str, Any], config_path: Path) -> None:
    validate_alignment(config, config_path, require_complete=True)
    existing = validate_umap(config, config_path, require_complete=False)
    if existing is not None:
        print(f"[{config['benchmark_slug']}] UMAP already verified", flush=True)
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import umap

    paths = configured_paths(config, config_path)
    output = paths["umap_output"]
    output.mkdir(parents=True, exist_ok=True)
    alignment = paths["alignment_output"]
    embedding_frame = pd.read_csv(
        alignment / "fastmnn_corrected_embedding.csv.gz", low_memory=False
    )
    metadata = pd.read_csv(alignment / "cell_metadata_label_free.csv.gz", dtype=str)
    if not np.array_equal(
        embedding_frame["cell_id"].astype(str).to_numpy(),
        metadata["cell_id"].astype(str).to_numpy(),
    ):
        raise ValueError("Corrected embedding and label-free metadata row order differ")
    dimension_columns = [column for column in embedding_frame if column != "cell_id"]
    expected_dimensions = int(config["expected_embedding_dimensions"])
    expected_names = [f"fastMNN_{index:02d}" for index in range(1, expected_dimensions + 1)]
    if dimension_columns != expected_names:
        raise ValueError(f"Corrected embedding dimensions changed: {dimension_columns}")
    embedding = embedding_frame[dimension_columns].to_numpy(np.float64)
    if embedding.shape != (int(config["expected_cells"]), expected_dimensions):
        raise ValueError("Corrected embedding shape changed")
    if not np.isfinite(embedding).all() or np.any(np.std(embedding, axis=0) < 1e-12):
        raise ValueError("Corrected embedding is non-finite or degenerate")

    parameters = dict(config["umap"])
    initialization_kind = parameters.pop("initialization")
    initialization_sd_raw = parameters.pop("initialization_sd")
    seed = int(parameters["random_state"])
    if initialization_kind == "seeded_gaussian":
        initialization_sd = float(initialization_sd_raw)
        initialization: str | np.ndarray = np.random.default_rng(seed).normal(
            0.0, initialization_sd, size=(len(embedding), 2)
        ).astype(np.float32)
        initialization_record = {
            "kind": initialization_kind,
            "seed": seed,
            "distribution": f"N(0, {initialization_sd}^2)",
            "independent_of_other_methods": True,
        }
    elif initialization_kind == "spectral":
        if initialization_sd_raw is not None:
            raise ValueError("spectral initialization requires initialization_sd=null")
        initialization = "spectral"
        initialization_record = {
            "kind": initialization_kind,
            "random_state": seed,
            "independent_of_other_methods": True,
        }
    else:
        raise ValueError("Noncanonical UMAP initialization")
    print(
        f"[{config['benchmark_slug']}] direct UMAP on {len(embedding):,} cells x "
        f"{embedding.shape[1]} corrected dimensions (no scaling/PCA)",
        flush=True,
    )
    started = time.time()
    coordinates = umap.UMAP(
        n_neighbors=int(parameters["n_neighbors"]),
        min_dist=float(parameters["min_dist"]),
        metric=str(parameters["metric"]),
        random_state=int(parameters["random_state"]),
        transform_seed=int(parameters["transform_seed"]),
        n_jobs=int(parameters["n_jobs"]),
        low_memory=bool(parameters["low_memory"]),
        init=initialization,
    ).fit_transform(embedding)
    coordinates = np.asarray(coordinates, dtype=np.float32)
    if coordinates.shape != (len(embedding), 2) or not np.isfinite(coordinates).all():
        raise ValueError("UMAP returned invalid coordinates")

    stage = Path(tempfile.mkdtemp(prefix=".fastmnn-umap-stage-", dir=output))
    try:
        data_dir = stage / "data"
        figure_dir = stage / "figures"
        data_dir.mkdir()
        figure_dir.mkdir()
        npy_path = data_dir / "umap_coordinates.npy"
        np.save(npy_path, coordinates)

        # Post-hoc display metadata is intentionally opened only after the
        # coordinates have been fitted and written to the staging area.
        display = pd.read_csv(paths["display_metadata"], low_memory=False)
        display_ids = display[config["cell_id_column"]].astype(str).to_numpy()
        fitted_ids = metadata["cell_id"].astype(str).to_numpy()
        if len(display) != len(metadata) or not np.array_equal(display_ids, fitted_ids):
            raise ValueError("Post-hoc display metadata differs from aligned cell order")
        expected_batches = [row["name"] for row in config["datasets"]]
        if pd.unique(display[config["batch_column"]].astype(str)).tolist() != expected_batches:
            raise ValueError("Display metadata dataset order changed")

        coordinate_frame = display[
            [
                config["cell_id_column"],
                config["batch_column"],
                "donor",
                config["sample_column"],
                config["label_column"],
            ]
        ].copy()
        coordinate_frame.insert(0, "row_id", np.arange(len(display), dtype=np.int64))
        coordinate_frame["UMAP1"] = coordinates[:, 0]
        coordinate_frame["UMAP2"] = coordinates[:, 1]
        csv_path = data_dir / "umap_coordinates.csv.gz"
        deterministic_gzip_csv(coordinate_frame, csv_path)

        labels = display[config["label_column"]].fillna(config["missing_label"]).astype(str).to_numpy()
        datasets = display[config["batch_column"]].astype(str).to_numpy()
        palette_rows = {str(row["label"]): row for row in config["palette"]}
        if set(np.unique(labels)) != set(palette_rows):
            raise ValueError(
                f"Display labels/palette differ: labels={sorted(np.unique(labels))}, "
                f"palette={sorted(palette_rows)}"
            )
        legend_order = [
            row["label"] for row in sorted(config["palette"], key=lambda row: int(row["legend_order"]))
        ]
        draw_order = [
            row["label"] for row in sorted(config["palette"], key=lambda row: int(row["draw_order"]))
        ]
        colors = {label: str(palette_rows[label]["color"]) for label in palette_rows}

        figure, axes = plt.subplots(1, 2, figsize=(17.0, 7.7), dpi=180)
        for label in draw_order:
            selected = labels == label
            row = palette_rows[label]
            axes[0].scatter(
                coordinates[selected, 0], coordinates[selected, 1],
                s=float(row.get("point_size", 1.1)), c=colors[label],
                alpha=float(row.get("alpha", 0.60)), linewidths=0, rasterized=True,
            )
        style_axis(axes[0])
        axes[0].set_title("Post-hoc merged cell type", fontsize=14, fontweight="bold")

        dataset_colors = {str(key): str(value) for key, value in config["dataset_colors"].items()}
        permutation = np.random.default_rng(seed + 1).permutation(len(datasets))
        axes[1].scatter(
            coordinates[permutation, 0], coordinates[permutation, 1],
            s=0.9,
            c=np.asarray([dataset_colors[value] for value in datasets], dtype=object)[permutation],
            alpha=0.65, linewidths=0, rasterized=True,
        )
        style_axis(axes[1])
        axes[1].set_title("Dataset", fontsize=14, fontweight="bold")
        figure.suptitle(
            f"{config['benchmark_title']} — fastMNN corrected embedding",
            fontsize=16, fontweight="bold", y=0.985,
        )
        figure.legend(
            handles=legend_handles(legend_order, colors),
            loc="lower center", bbox_to_anchor=(0.39, 0.012),
            ncol=5, frameon=False, fontsize=7.2,
        )
        figure.legend(
            handles=legend_handles(expected_batches, dataset_colors),
            loc="lower center", bbox_to_anchor=(0.80, 0.032),
            ncol=3, frameon=False, fontsize=8.5,
        )
        figure.subplots_adjust(left=0.035, right=0.995, top=0.92, bottom=0.19, wspace=0.04)
        png_name = f"{config['benchmark_slug']}_fastmnn_umap.png"
        pdf_name = f"{config['benchmark_slug']}_fastmnn_umap.pdf"
        png_path = figure_dir / png_name
        pdf_path = figure_dir / pdf_name
        figure.savefig(png_path, dpi=300, bbox_inches="tight")
        figure.savefig(pdf_path, bbox_inches="tight")
        plt.close(figure)

        output_hashes = {
            "data/umap_coordinates.npy": sha256(npy_path),
            "data/umap_coordinates.csv.gz": sha256(csv_path),
            f"figures/{png_name}": sha256(png_path),
            f"figures/{pdf_name}": sha256(pdf_path),
        }
        manifest = {
            "schema_version": UMAP_SCHEMA,
            "status": "complete",
            "benchmark": config["benchmark_slug"],
            "method_label": METHOD_LABEL,
            "cells": int(len(embedding)),
            "embedding_dimensions_available": int(embedding.shape[1]),
            "embedding_dimensions_used": int(embedding.shape[1]),
            "pre_umap_processing": "direct corrected fastMNN embedding; no z-score; no PCA",
            "labels_used_for_alignment": False,
            "labels_used_for_umap_fitting": False,
            "labels_loaded_after_coordinates_frozen": True,
            "umap_parameters": parameters,
            "initialization": initialization_record,
            "alignment_manifest": str(alignment / "manifest.json"),
            "alignment_manifest_sha256": sha256(alignment / "manifest.json"),
            "display_metadata": str(paths["display_metadata"]),
            "display_metadata_sha256": sha256(paths["display_metadata"]),
            "config": str(config_path),
            "config_sha256": sha256(config_path),
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": sha256(Path(__file__).resolve()),
            "embedding_buffer_sha256_float64": sha256_buffer(embedding, np.float64),
            "coordinate_buffer_sha256_float32": sha256_buffer(coordinates, np.float32),
            "runtime_seconds": time.time() - started,
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "umap-learn": version("umap-learn"),
                "matplotlib": version("matplotlib"),
            },
            "outputs": output_hashes,
        }
        manifest_path = stage / "umap_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        complete_path = stage / "COMPLETE.json"
        complete_path.write_text(
            json.dumps(
                {
                    "schema_version": UMAP_SCHEMA,
                    "status": "complete",
                    "manifest_sha256": sha256(manifest_path),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        for name in ("data", "figures", "umap_manifest.json", "COMPLETE.json"):
            source = stage / name
            target = output / name
            if target.exists():
                raise RuntimeError(f"Refusing to overwrite UMAP artifact: {target}")
            source.replace(target)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    validate_umap(config, config_path, require_complete=True)
    print(f"[{config['benchmark_slug']}] UMAP and figures saved to {output}", flush=True)


def write_benchmark_complete(config: dict[str, Any], config_path: Path) -> None:
    validate_preprocessing(config, config_path, require_complete=True)
    validate_alignment(config, config_path, require_complete=True)
    validate_umap(config, config_path, require_complete=True)
    paths = configured_paths(config, config_path)
    value = {
        "schema_version": COMPLETE_SCHEMA,
        "status": "complete",
        "benchmark": config["benchmark_slug"],
        "method_label": METHOD_LABEL,
        "config_sha256": sha256(config_path),
        "preprocessing_manifest_sha256": sha256(paths["preprocessing_manifest"]),
        "alignment_manifest_sha256": sha256(paths["alignment_output"] / "manifest.json"),
        "umap_manifest_sha256": sha256(paths["umap_output"] / "umap_manifest.json"),
        "label_free_alignment_and_umap": True,
        "corrected_embedding_used_directly_for_umap": True,
        "zscore_before_umap": False,
        "pca_after_fastmnn": False,
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": sha256(Path(__file__).resolve()),
    }
    write_json_atomic(value, method_root(config_path) / "COMPLETE.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["preprocess", "align", "umap", "audit", "all"])
    parser.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args()
    config, config_path = validate_config(arguments.config)

    if arguments.command in {"preprocess", "all"}:
        run_preprocessing(config, config_path)
    if arguments.command in {"align", "all"}:
        run_alignment(config, config_path)
    if arguments.command in {"umap", "all"}:
        run_umap(config, config_path)
    if arguments.command in {"audit", "all"}:
        write_benchmark_complete(config, config_path)
        print(f"[{config['benchmark_slug']}] complete benchmark audit passed", flush=True)


if __name__ == "__main__":
    main()
