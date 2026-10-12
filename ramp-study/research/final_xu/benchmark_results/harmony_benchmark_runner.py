#!/usr/bin/env python3
"""Run frozen three-dataset Harmony-direct benchmarks reproducibly.

Dataset-specific sampling and input-scale decisions live in each benchmark's
``harmony/config.json``.  The algorithm implementation itself remains in
``final_xu/algorithm/harmony`` and performs no hidden preprocessing.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path
from typing import Any


SCRIPT = Path(__file__).resolve()
BENCHMARK_RESULTS = SCRIPT.parent
FINAL_XU = BENCHMARK_RESULTS.parent
ALGORITHM_DIR = FINAL_XU / "algorithm" / "harmony"
HARMONY_RUNNER = ALGORITHM_DIR / "run_harmony_direct.R"
HARMONY_EXAMPLE_CONFIG = ALGORITHM_DIR / "config.example.json"
VARIANTS = ("source_scale", "adtnorm_fixed_arcsinh")
METHOD_LABEL = "Harmony-direct (ADTnorm protocol; no PCA)"

cache_root = Path(tempfile.gettempdir()) / "final_xu_harmony_benchmark_cache"
for variable, directory in (
    ("MPLCONFIGDIR", cache_root / "matplotlib"),
    ("NUMBA_CACHE_DIR", cache_root / "numba"),
    ("XDG_CACHE_HOME", cache_root / "xdg"),
    ("JOBLIB_TEMP_FOLDER", cache_root / "joblib"),
):
    directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(variable, str(directory))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import umap


def fail(message: str) -> None:
    raise RuntimeError(message)


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = FINAL_XU / candidate
    return candidate.resolve(strict=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def buffer_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def write_json(object_: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(object_, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv_gz(frame: pd.DataFrame, path: Path, float_format: str = "%.17g") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=6, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
                frame.to_csv(text, index=False, float_format=float_format)


def read_markers(config: dict[str, Any]) -> list[str]:
    source = config["input"]
    if "marker_file" in source:
        markers = [line.strip() for line in resolve(source["marker_file"]).read_text().splitlines() if line.strip()]
    else:
        markers = [str(value) for value in source["markers"]]
    if len(markers) < 2 or len(markers) != len(set(markers)):
        fail("Marker contract must contain at least two unique names")
    if markers != config["expected"]["markers"]:
        fail("Observed marker order differs from the frozen benchmark contract")
    return markers


def load_frozen_input(config: dict[str, Any]) -> tuple[np.ndarray, pd.DataFrame, list[str], list[Path]]:
    source = config["input"]
    markers = read_markers(config)
    source_paths: list[Path] = []
    if source["kind"] == "combined_npy":
        matrix_path = resolve(source["matrix"])
        fit_path = resolve(source["fit_metadata"])
        identity_path = resolve(source["identity_metadata"])
        marker_path = resolve(source["marker_file"])
        source_paths.extend([matrix_path, fit_path, identity_path, marker_path])
        matrix = np.asarray(np.load(matrix_path, allow_pickle=False), dtype=np.float64)
        fit = pd.read_csv(fit_path, dtype=str, low_memory=False)
        # This file contains identities/provenance only.  Label-bearing locked
        # metadata is intentionally not opened before alignment or UMAP fitting.
        identity = pd.read_csv(identity_path, dtype={"cell_id": str}, low_memory=False)
        required_fit = ["dataset", "unit_key"]
        if list(fit.columns) != required_fit:
            fail(f"Fit metadata must contain exactly {required_fit}")
        if len(identity) != len(fit) or len(matrix) != len(fit):
            fail("Frozen matrix and metadata row counts differ")
        if not fit["dataset"].astype(str).equals(identity["dataset"].astype(str)):
            fail("Fit and identity-metadata dataset order differ")
        if not fit["unit_key"].astype(str).equals(identity["unit_key"].astype(str)):
            fail("Fit and identity-metadata unit order differ")
        metadata = pd.DataFrame({
            "cell_id": identity["cell_id"].astype(str),
            "dataset": fit["dataset"].astype(str),
            "unit_key": fit["unit_key"].astype(str),
        })
    elif source["kind"] == "split_csv":
        frames: list[pd.DataFrame] = []
        for path_text in source["paths"]:
            path = resolve(path_text)
            source_paths.append(path)
            frame = pd.read_csv(path, dtype={"cell_id": str, "dataset": str, "unit_key": str}, low_memory=False)
            required = ["cell_id", "dataset", "unit_key", *markers]
            missing = [column for column in required if column not in frame]
            if missing:
                fail(f"Missing columns in {path}: {missing}")
            frames.append(frame[required])
        combined = pd.concat(frames, ignore_index=True)
        matrix = combined[markers].to_numpy(np.float64)
        metadata = combined[["cell_id", "dataset", "unit_key"]].copy()
    else:
        fail(f"Unsupported input kind: {source['kind']}")

    expected = config["expected"]
    if matrix.shape != (int(expected["cells"]), len(markers)):
        fail(f"Frozen matrix shape changed: {matrix.shape}")
    if not np.isfinite(matrix).all():
        fail("Frozen matrix contains non-finite values")
    if not metadata["cell_id"].is_unique or metadata.isna().any().any():
        fail("Cell IDs and fitting metadata must be complete and unique")
    observed_order = list(pd.unique(metadata["dataset"]))
    if observed_order != expected["dataset_order"]:
        fail(f"Dataset block order changed: {observed_order}")
    observed_counts = metadata.groupby("dataset", sort=False).size().astype(int).to_dict()
    if observed_counts != {key: int(value) for key, value in expected["dataset_counts"].items()}:
        fail(f"Dataset counts changed: {observed_counts}")
    return matrix, metadata, markers, source_paths


def transform_matrix(source: np.ndarray, variant: str) -> tuple[np.ndarray, str, str]:
    if variant == "source_scale":
        return source.copy(), "source_scale", "Identity: x. Frozen provider/source analysis-ready values are supplied unchanged."
    if variant == "adtnorm_fixed_arcsinh":
        transformed = np.arcsinh(1.0 + source / 5.0)
        return transformed, "ADTnorm_fixed_arcsinh_a1_b1over5_c0", "ADTnorm official fixed transform: asinh(1 + x/5), applied element-wise."
    fail(f"Unknown variant: {variant}")


def preprocessing_rank_audit(source: np.ndarray, transformed: np.ndarray, metadata: pd.DataFrame, markers: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    datasets = metadata["dataset"].to_numpy(str)
    for dataset in pd.unique(datasets):
        selected = datasets == dataset
        for marker_index, marker in enumerate(markers):
            before = source[selected, marker_index]
            after = transformed[selected, marker_index]
            before_rank = pd.Series(before).rank(method="average").to_numpy(np.float64)
            after_rank = pd.Series(after).rank(method="average").to_numpy(np.float64)
            rank_equal = bool(np.array_equal(before_rank, after_rank))
            order = np.argsort(before, kind="mergesort")
            source_difference = np.diff(before[order])
            transformed_difference = np.diff(after[order])
            adjacent_order_inversions = int(np.sum((source_difference > 0) & (transformed_difference < 0)))
            new_adjacent_ties = int(np.sum((source_difference > 0) & (transformed_difference == 0)))
            if np.std(before_rank) == 0 or np.std(after_rank) == 0:
                spearman: float | None = None
            else:
                spearman = float(np.corrcoef(before_rank, after_rank)[0, 1])
            rows.append({
                "dataset": dataset,
                "marker": marker,
                "n_cells": int(selected.sum()),
                "rank_vectors_exactly_equal": rank_equal,
                "adjacent_order_inversions": adjacent_order_inversions,
                "new_adjacent_ties_from_finite_precision": new_adjacent_ties,
                "source_unique_values": int(np.unique(before).size),
                "transformed_unique_values": int(np.unique(after).size),
                "spearman": spearman,
                "maximum_abs_midrank_displacement": float(np.max(np.abs(before_rank - after_rank)) / len(before_rank)),
                "source_min": float(np.min(before)),
                "source_max": float(np.max(before)),
                "transformed_min": float(np.min(after)),
                "transformed_max": float(np.max(after)),
            })
    return pd.DataFrame(rows)


def validate_complete(directory: Path, required_files: list[str]) -> bool:
    complete_path = directory / "COMPLETE.json"
    if not complete_path.is_file():
        return False
    complete = json.loads(complete_path.read_text())
    if complete.get("status") != "complete":
        fail(f"Invalid completion marker: {complete_path}")
    for relative in required_files:
        path = directory / relative
        if not path.is_file():
            fail(f"Completion marker exists but output is missing: {path}")
    return True


def prepare_preprocessing(
    config: dict[str, Any], method_dir: Path, variant: str,
    source: np.ndarray, metadata: pd.DataFrame, markers: list[str], source_paths: list[Path],
) -> tuple[Path, Path, Path]:
    variant_dir = method_dir / variant
    preprocessing_dir = variant_dir / "preprocessing"
    input_path = preprocessing_dir / "data" / "combined_analysis_ready.csv.gz"
    marker_path = preprocessing_dir / "markers.txt"
    record_path = preprocessing_dir / "preprocessing_record.txt"
    required = [
        "data/combined_analysis_ready.csv.gz", "markers.txt", "preprocessing_record.txt",
        "source_to_method_input_rank_audit.csv", "manifest.json",
    ]
    if validate_complete(preprocessing_dir, required):
        manifest = json.loads((preprocessing_dir / "manifest.json").read_text())
        if manifest["source_cell_id_sha256"] != buffer_sha256(metadata["cell_id"].astype(str).to_numpy(dtype="U")):
            fail(f"Existing preprocessing cell order differs for {variant}")
        return input_path, marker_path, record_path
    if preprocessing_dir.exists() and any(preprocessing_dir.iterdir()):
        fail(f"Refusing to overwrite incomplete preprocessing directory: {preprocessing_dir}")

    transformed, scale_name, description = transform_matrix(source, variant)
    if transformed.shape != source.shape or not np.isfinite(transformed).all():
        fail(f"Invalid transformed matrix for {variant}")
    audit = preprocessing_rank_audit(source, transformed, metadata, markers)
    if int(audit["adjacent_order_inversions"].sum()) != 0:
        fail(f"Preprocessing introduced a within-dataset order inversion for {variant}")

    preprocessing_dir.mkdir(parents=True, exist_ok=True)
    marker_path.write_text("\n".join(markers) + "\n")
    record_lines = [
        f"benchmark={config['benchmark']}",
        f"variant={variant}",
        f"input_scale_name={scale_name}",
        f"operation={description}",
        f"source_scale_note={config['input']['source_scale_note']}",
        "scope=element-wise marker values; no cell-wise library-size normalization",
        "clipping=none",
        "shifting_for_nonnegative_values=none",
        "PCA=none",
        "labels_used=false",
        "Harmony_algorithm_layer_transformation=none",
        "source_files:",
        *[f"  {path}: sha256={sha256(path)}" for path in source_paths],
    ]
    record_path.write_text("\n".join(record_lines) + "\n")
    audit_path = preprocessing_dir / "source_to_method_input_rank_audit.csv"
    audit.to_csv(audit_path, index=False)
    input_frame = metadata.copy()
    for index, marker in enumerate(markers):
        input_frame[marker] = transformed[:, index]
    write_csv_gz(input_frame, input_path)
    roundtrip = pd.read_csv(
        input_path,
        dtype={"cell_id": str, "dataset": str, "unit_key": str},
        low_memory=False,
        float_precision="round_trip",
    )
    if not roundtrip[["cell_id", "dataset", "unit_key"]].equals(metadata):
        fail(f"Metadata changed during deterministic CSV round-trip for {variant}")
    if not np.array_equal(roundtrip[markers].to_numpy(np.float64), transformed):
        fail(f"Marker values changed during deterministic CSV round-trip for {variant}")
    manifest = {
        "schema_version": "final_xu.harmony_benchmark_preprocessing.v1",
        "status": "complete",
        "benchmark": config["benchmark"],
        "variant": variant,
        "cells": int(len(metadata)),
        "markers": markers,
        "dataset_order": config["expected"]["dataset_order"],
        "dataset_counts": config["expected"]["dataset_counts"],
        "labels_used": False,
        "formula": description,
        "source_files": [{"path": str(path), "sha256": sha256(path)} for path in source_paths],
        "source_cell_id_sha256": buffer_sha256(metadata["cell_id"].astype(str).to_numpy(dtype="U")),
        "outputs": {
            "combined_analysis_ready.csv.gz": sha256(input_path),
            "markers.txt": sha256(marker_path),
            "preprocessing_record.txt": sha256(record_path),
            "source_to_method_input_rank_audit.csv": sha256(audit_path),
        },
        "all_source_to_input_rank_vectors_equal": bool(audit["rank_vectors_exactly_equal"].all()),
        "no_source_to_input_order_inversions": bool((audit["adjacent_order_inversions"] == 0).all()),
        "finite_precision_new_adjacent_ties": int(audit["new_adjacent_ties_from_finite_precision"].sum()),
        "deterministic_csv_roundtrip_exact": True,
    }
    write_json(manifest, preprocessing_dir / "manifest.json")
    write_json({"status": "complete", "manifest_sha256": sha256(preprocessing_dir / "manifest.json")}, preprocessing_dir / "COMPLETE.json")
    return input_path, marker_path, record_path


def run_alignment(
    config: dict[str, Any], method_dir: Path, variant: str,
    input_path: Path, marker_path: Path, record_path: Path,
) -> Path:
    alignment_dir = method_dir / variant / "alignment"
    output_dir = alignment_dir / "data"
    required = [
        "harmony_adjusted_marker_coordinates.csv.gz", "cell_metadata_label_free.csv.gz",
        "marker_rank_audit.csv", "marker_rank_audit_by_sample.csv",
        "marker_value_shift_audit.csv", "manifest.json", "COMPLETE.json",
    ]
    if validate_complete(output_dir, required):
        return output_dir
    if output_dir.exists():
        fail(f"Refusing to overwrite incomplete alignment output: {output_dir}")
    alignment_dir.mkdir(parents=True, exist_ok=True)
    example = json.loads(HARMONY_EXAMPLE_CONFIG.read_text())
    scale_description = (
        "Frozen provider/source analysis-ready values supplied unchanged."
        if variant == "source_scale"
        else "ADTnorm official fixed element-wise transform asinh(1 + x/5)."
    )
    scale_name = "source_scale" if variant == "source_scale" else "ADTnorm_fixed_arcsinh_a1_b1over5_c0"
    runner_config = {
        "schema_version": example["schema_version"],
        "method_label": METHOD_LABEL,
        "cell_id_column": "cell_id",
        "batch_column": "dataset",
        "sample_column": "unit_key",
        "batch_levels": config["expected"]["dataset_order"],
        "input_scale": {
            "status": "analysis_ready",
            "name": scale_name,
            "description": scale_description,
            "preprocessing_record_file": str(record_path.resolve()),
            "preprocessing_record_sha256": sha256(record_path),
            "input_sha256": sha256(input_path),
            "marker_file_sha256": sha256(marker_path),
            "algorithm_applies_transformation": False,
        },
        "algorithm_parameters": example["algorithm_parameters"],
    }
    runner_config_path = alignment_dir / "config.json"
    write_json(runner_config, runner_config_path)
    command = [
        "Rscript", str(HARMONY_RUNNER), "--input", str(input_path),
        "--markers", str(marker_path), "--config", str(runner_config_path),
        "--output", str(output_dir),
    ]
    print(f"Running {config['benchmark']} / {variant} Harmony-direct", flush=True)
    subprocess.run(command, cwd=FINAL_XU, check=True)
    if not validate_complete(output_dir, required):
        fail(f"Harmony output did not complete for {config['benchmark']} / {variant}")
    return output_dir


def build_umap_initialization(
    config: dict[str, Any], expected_cells: int,
) -> tuple[str | np.ndarray, dict[str, Any]]:
    initialization = config["umap"]["initialization"]
    if initialization["kind"] == "spectral":
        return "spectral", {
            "kind": "spectral",
            "method_neutral": True,
            "random_state": int(config["umap"]["parameters"]["random_state"]),
        }
    if initialization["kind"] == "seeded_gaussian":
        coordinates = np.random.default_rng(int(initialization["seed"])).normal(
            0.0, float(initialization["sd"]), size=(expected_cells, 2)
        ).astype(np.float32)
        return coordinates, {
            "kind": "seeded Gaussian",
            "seed": int(initialization["seed"]),
            "sd": float(initialization["sd"]),
            "buffer_sha256": buffer_sha256(coordinates),
        }
    fail(f"Unsupported UMAP initialization: {initialization['kind']}")


def load_palette(config: dict[str, Any]) -> pd.DataFrame:
    display = config["display"]
    if "palette_file" in display:
        palette = pd.read_csv(resolve(display["palette_file"]), low_memory=False)
        required = {"celltype", "color", "legend_order", "draw_order_bottom_to_top", "point_size", "alpha", "zorder"}
        if not required.issubset(palette.columns):
            fail("Palette file lacks required display columns")
        palette = palette[list(required)].copy()
        if "draw_order_override" in display:
            override = [str(value) for value in display["draw_order_override"]]
            if set(override) != set(palette["celltype"].astype(str)):
                fail("Display draw-order override does not match the palette vocabulary")
            positions = {label: index for index, label in enumerate(override)}
            palette["draw_order_bottom_to_top"] = palette["celltype"].astype(str).map(positions)
            palette["zorder"] = palette["draw_order_bottom_to_top"] + 1
        return palette
    rows = []
    for label in display["label_order"]:
        style = display["celltype_style"][label]
        rows.append({
            "celltype": label,
            "color": style["color"],
            "legend_order": int(style["legend_order"]),
            "draw_order_bottom_to_top": int(style["draw_order_bottom_to_top"]),
            "point_size": float(style["point_size"]),
            "alpha": float(style["alpha"]),
            "zorder": int(style["zorder"]),
        })
    return pd.DataFrame(rows)


def style_axis(axis: plt.Axes) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_aspect("equal", adjustable="datalim")
    for spine in axis.spines.values():
        spine.set_visible(False)


def legend_handles(order: list[str], colors: dict[str, str]) -> list[Line2D]:
    return [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=colors[value],
               markeredgecolor="none", markersize=5, label=value)
        for value in order
    ]


def draw_celltypes(axis: plt.Axes, coordinates: np.ndarray, labels: np.ndarray, palette: pd.DataFrame) -> None:
    for row in palette.sort_values("draw_order_bottom_to_top").itertuples(index=False):
        selected = labels == str(row.celltype)
        if not selected.any():
            continue
        axis.scatter(
            coordinates[selected, 0], coordinates[selected, 1], s=float(row.point_size),
            color=str(row.color), alpha=float(row.alpha), linewidths=0,
            rasterized=True, zorder=int(row.zorder),
        )
    style_axis(axis)


def draw_datasets(axis: plt.Axes, coordinates: np.ndarray, datasets: np.ndarray, config: dict[str, Any]) -> None:
    display = config["display"]
    order = config["expected"]["dataset_order"]
    colors = display["dataset_colors"]
    if display.get("shuffle_dataset_points", False):
        permutation = np.random.default_rng(int(display.get("shuffle_seed", 42))).permutation(len(datasets))
        point_colors = np.asarray([colors[value] for value in datasets], dtype=object)
        axis.scatter(coordinates[permutation, 0], coordinates[permutation, 1], s=float(display["dataset_point_size"]),
                     c=point_colors[permutation], alpha=float(display["dataset_alpha"]), linewidths=0, rasterized=True)
    else:
        for dataset in order:
            selected = datasets == dataset
            axis.scatter(coordinates[selected, 0], coordinates[selected, 1], s=float(display["dataset_point_size"]),
                         c=colors[dataset], alpha=float(display["dataset_alpha"]), linewidths=0, rasterized=True)
    style_axis(axis)


def run_umap(
    config: dict[str, Any], method_dir: Path, variant: str,
    input_path: Path, alignment_output: Path, markers: list[str],
) -> Path:
    umap_dir = method_dir / variant / "umap"
    figure_stem = f"{config['benchmark']}_harmony_{variant}_umap"
    required = [
        "data/umap_coordinates.csv.gz", "data/umap_coordinates.npy",
        f"figures/{figure_stem}.png", f"figures/{figure_stem}.pdf", "umap_manifest.json",
    ]
    if validate_complete(umap_dir, required):
        return umap_dir
    if umap_dir.exists() and any(umap_dir.iterdir()):
        fail(f"Refusing to overwrite incomplete UMAP output: {umap_dir}")
    umap_dir.mkdir(parents=True, exist_ok=True)

    adjusted_path = alignment_output / "harmony_adjusted_marker_coordinates.csv.gz"
    adjusted = pd.read_csv(adjusted_path, dtype={"cell_id": str}, low_memory=False)
    inputs = pd.read_csv(input_path, usecols=["cell_id", "dataset", "unit_key"], dtype=str, low_memory=False)
    if list(adjusted.columns) != ["cell_id", *markers]:
        fail("Harmony output marker-axis contract changed")
    if not adjusted["cell_id"].equals(inputs["cell_id"]):
        fail("Harmony output cell order differs from frozen input")
    matrix = adjusted[markers].to_numpy(np.float64)
    keep = np.isfinite(matrix).all(axis=0) & (np.std(matrix, axis=0) > 1e-10)
    if int(keep.sum()) < 2 or not np.isfinite(matrix[:, keep]).all():
        fail("Harmony adjusted coordinates are invalid for UMAP")
    standardized = StandardScaler().fit_transform(matrix[:, keep]).astype(np.float32)
    initialization, initialization_record = build_umap_initialization(config, len(inputs))
    parameters = config["umap"]["parameters"]
    print(f"Fitting matched UMAP for {config['benchmark']} / {variant}: {len(inputs):,} cells x {int(keep.sum())} markers", flush=True)
    coordinates = umap.UMAP(
        n_neighbors=int(parameters["n_neighbors"]), min_dist=float(parameters["min_dist"]),
        metric=str(parameters["metric"]), random_state=int(parameters["random_state"]),
        transform_seed=int(parameters["transform_seed"]), n_jobs=int(parameters["n_jobs"]),
        low_memory=bool(parameters["low_memory"]),
        init=initialization.copy() if isinstance(initialization, np.ndarray) else initialization,
    ).fit_transform(standardized)
    coordinates = np.asarray(coordinates, dtype=np.float32)
    if coordinates.shape != (len(inputs), 2) or not np.isfinite(coordinates).all():
        fail("UMAP produced invalid coordinates")

    # Cell-type labels and palettes are deliberately opened only after UMAP is fixed.
    relabel_path = resolve(config["display"]["relabel_file"])
    labels = pd.read_csv(relabel_path, dtype={"cell_id": str, "dataset": str}, low_memory=False)
    if not labels["cell_id"].equals(inputs["cell_id"]):
        fail("Post-hoc relabel row order differs from Harmony output")
    if not labels["dataset"].astype(str).equals(inputs["dataset"].astype(str)):
        fail("Post-hoc relabel dataset identity differs")
    label_values = labels["celltype"].astype(str).to_numpy()
    palette = load_palette(config)
    if set(label_values) - set(palette["celltype"].astype(str)):
        fail(f"Palette is missing cell types: {sorted(set(label_values) - set(palette['celltype'].astype(str)))}")

    data_dir = umap_dir / "data"
    figure_dir = umap_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    coordinate_frame = inputs.copy()
    if "donor" in labels:
        coordinate_frame["donor"] = labels["donor"].astype(str).to_numpy()
    coordinate_frame["celltype"] = label_values
    coordinate_frame["UMAP1"] = coordinates[:, 0]
    coordinate_frame["UMAP2"] = coordinates[:, 1]
    coordinate_csv = data_dir / "umap_coordinates.csv.gz"
    coordinate_npy = data_dir / "umap_coordinates.npy"
    write_csv_gz(coordinate_frame, coordinate_csv, float_format="%.9g")
    np.save(coordinate_npy, coordinates, allow_pickle=False)

    display = config["display"]
    datasets = inputs["dataset"].astype(str).to_numpy()
    figure, axes = plt.subplots(1, 2, figsize=tuple(display["figure_size"]), dpi=180)
    figure.patch.set_facecolor("white")
    draw_celltypes(axes[0], coordinates, label_values, palette)
    draw_datasets(axes[1], coordinates, datasets, config)
    axes[0].set_title("Cell type (post hoc)", fontsize=11, fontweight="bold")
    axes[1].set_title("Dataset", fontsize=11, fontweight="bold")
    legend_palette = palette.sort_values("legend_order")
    label_order = legend_palette["celltype"].astype(str).tolist()
    label_colors = dict(zip(legend_palette["celltype"].astype(str), legend_palette["color"].astype(str)))
    dataset_order = config["expected"]["dataset_order"]
    dataset_colors = display["dataset_colors"]
    dataset_display = display.get("dataset_display", {value: value for value in dataset_order})
    figure.legend(handles=legend_handles(label_order, label_colors), loc="lower center",
                  bbox_to_anchor=(0.5, float(display["celltype_legend_y"])), ncol=int(display["celltype_legend_ncol"]),
                  frameon=False, fontsize=float(display["celltype_legend_fontsize"]), columnspacing=1.0, handletextpad=0.25)
    figure.legend(handles=legend_handles(dataset_order, dataset_colors), labels=[dataset_display[value] for value in dataset_order],
                  loc="lower center", bbox_to_anchor=(0.5, float(display["dataset_legend_y"])), ncol=3,
                  frameon=False, fontsize=float(display["dataset_legend_fontsize"]))
    variant_title = "source scale" if variant == "source_scale" else "ADTnorm fixed arcsinh"
    figure.suptitle(f"{config['display_name']} — Harmony-direct ({variant_title}; {len(markers)} shared markers)",
                    fontsize=13, fontweight="bold")
    figure.tight_layout(rect=(0, float(display["layout_bottom"]), 1, 0.94), w_pad=0.04)
    png_path = figure_dir / f"{figure_stem}.png"
    pdf_path = figure_dir / f"{figure_stem}.pdf"
    figure.savefig(png_path, dpi=320, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    plt.close(figure)

    manifest = {
        "schema_version": "final_xu.harmony_benchmark_umap.v1",
        "status": "complete",
        "benchmark": config["benchmark"],
        "variant": variant,
        "cells": int(len(inputs)),
        "markers_available": markers,
        "markers_used": [marker for marker, selected in zip(markers, keep) if selected],
        "pre_umap_processing": "pooled StandardScaler; no PCA",
        "labels_used_for_alignment": False,
        "labels_used_for_umap_fitting": False,
        "labels_loaded_after_umap_fitting": True,
        "umap_parameters": parameters,
        "initialization": initialization_record,
        "coordinate_buffer_sha256": buffer_sha256(coordinates),
        "sources": {
            "adjusted_marker_coordinates": {"path": str(adjusted_path), "sha256": sha256(adjusted_path)},
            "method_input": {"path": str(input_path), "sha256": sha256(input_path)},
            "posthoc_relabel": {"path": str(relabel_path), "sha256": sha256(relabel_path)},
        },
        "outputs": {
            coordinate_csv.name: sha256(coordinate_csv), coordinate_npy.name: sha256(coordinate_npy),
            png_path.name: sha256(png_path), pdf_path.name: sha256(pdf_path),
        },
        "runtime": {
            "python": platform.python_version(), "numpy": version("numpy"), "pandas": version("pandas"),
            "scikit-learn": version("scikit-learn"), "umap-learn": version("umap-learn"),
            "matplotlib": version("matplotlib"),
        },
    }
    write_json(manifest, umap_dir / "umap_manifest.json")
    write_json({"status": "complete", "manifest_sha256": sha256(umap_dir / "umap_manifest.json")}, umap_dir / "COMPLETE.json")
    return umap_dir


def write_summary(config: dict[str, Any], method_dir: Path, variants: list[str]) -> None:
    rows: list[dict[str, Any]] = []
    for variant in variants:
        preprocessing_dir = method_dir / variant / "preprocessing"
        alignment_dir = method_dir / variant / "alignment" / "data"
        rank_input = pd.read_csv(preprocessing_dir / "source_to_method_input_rank_audit.csv")
        rank_output = pd.read_csv(alignment_dir / "marker_rank_audit.csv")
        shift_output = pd.read_csv(alignment_dir / "marker_value_shift_audit.csv")
        rows.append({
            "benchmark": config["benchmark"],
            "method": METHOD_LABEL,
            "variant": variant,
            "cells": int(config["expected"]["cells"]),
            "markers": len(config["expected"]["markers"]),
            "all_source_to_input_rank_vectors_equal": bool(rank_input["rank_vectors_exactly_equal"].all()),
            "median_input_to_adjusted_spearman": float(rank_output["spearman"].median()),
            "minimum_input_to_adjusted_spearman": float(rank_output["spearman"].min()),
            "median_max_abs_midrank_displacement": float(rank_output["max_abs_midrank_displacement"].median()),
            "maximum_abs_midrank_displacement": float(rank_output["max_abs_midrank_displacement"].max()),
            "median_outside_input_range_fraction": float(shift_output["adjusted_outside_within_batch_input_range_fraction"].median()),
            "maximum_outside_input_range_fraction": float(shift_output["adjusted_outside_within_batch_input_range_fraction"].max()),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(method_dir / "harmony_variant_summary.csv", index=False)
    write_json({
        "status": "complete",
        "benchmark": config["benchmark"],
        "variants": variants,
        "summary_sha256": sha256(method_dir / "harmony_variant_summary.csv"),
    }, method_dir / "COMPLETE.json")


def run(config_path: Path, requested_variants: list[str]) -> None:
    config = json.loads(config_path.read_text())
    required_top = {"schema_version", "benchmark", "display_name", "input", "expected", "umap", "display"}
    if set(config) != required_top or config["schema_version"] != "final_xu.harmony_benchmark.v1":
        fail(f"Invalid benchmark config schema: {config_path}")
    method_dir = config_path.parent
    source, metadata, markers, source_paths = load_frozen_input(config)
    for variant in requested_variants:
        input_path, marker_path, record_path = prepare_preprocessing(
            config, method_dir, variant, source, metadata, markers, source_paths
        )
        alignment_output = run_alignment(config, method_dir, variant, input_path, marker_path, record_path)
        run_umap(config, method_dir, variant, input_path, alignment_output, markers)
    write_summary(config, method_dir, requested_variants)
    print(f"PASS: completed Harmony benchmark {config['benchmark']} variants={requested_variants}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--variant", choices=[*VARIANTS, "both"], default="both")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    config_path = arguments.config.resolve(strict=True)
    variants = list(VARIANTS) if arguments.variant == "both" else [arguments.variant]
    run(config_path, variants)


if __name__ == "__main__":
    main()
