#!/usr/bin/env python3
"""Build label-free, official-tutorial-scaled MaxFuse v2 BM inputs.

The row cohort is immutable and comes from ``benchmark_data/bone_marrow``.
Provider matrices are reopened only to recover each platform's full native
protein panel.  CD19 is removed before every transformation so the held-out
target cannot enter an active feature or a CLR denominator.
After each provider/paper transform, every dataset-native active marker is
sample-SD z-scored within dataset to reproduce MaxFuse's method-level
``sc.pp.scale`` input convention. Shared columns are sliced only afterward.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import platform
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from scipy.io import mmread


HERE = Path(__file__).resolve().parent
FINAL_XU = HERE.parents[2]
DEFAULT_RAW_ROOT = FINAL_XU / "raw_data" / "bone_marrow"
DEFAULT_BENCHMARK_ROOT = FINAL_XU / "benchmark_data" / "bone_marrow"
DEFAULT_OUTPUT_DIR = HERE / "processed"

ID_COLUMNS = ["cell_id", "dataset", "donor", "unit_key"]
HELDOUT = "CD19"
SHARED_MARKERS = [
    "CD117", "CD11B", "CD11C", "CD123", "CD14", "CD3", "CD33",
    "CD34", "CD38", "CD4", "CD45", "CD45RA", "CD61", "CD8",
]
EXPECTED = {
    "bm_codex": {"name": "BM CODEX", "cells": 30_000, "active": 48},
    "bm_cytof": {"name": "BM CyTOF", "cells": 30_000, "active": 31},
    "triana_abseq": {"name": "Triana AbSeq", "cells": 28_031, "active": 96},
}

# Only nomenclature is harmonized. Values and source feature order are kept.
MARKER_ALIASES = {
    "CD3e": "CD3",
    "CD11b": "CD11B",
    "CD11c": "CD11C",
}

CODEX_EXPORT_R = r'''
suppressPackageStartupMessages({
  library(SeuratObject)
  library(Matrix)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) {
  stop("usage: export.R SOURCE_RDS LOCKED_METADATA_CSV_GZ OUTPUT_CSV_GZ")
}
options(digits = 17)
object <- readRDS(normalizePath(args[[1]], mustWork = TRUE))
locked <- read.csv(gzfile(args[[2]]), stringsAsFactors = FALSE, check.names = FALSE)
if (!("cell_id" %in% colnames(locked))) stop("row lock lacks cell_id")
ids <- as.character(locked$cell_id)
if (anyDuplicated(ids)) stop("CODEX row-lock cell IDs are not unique")
if (!("CODEX" %in% names(object@assays))) stop("CODEX assay is absent")
counts <- GetAssayData(object, assay = "CODEX", slot = "counts")
if (nrow(counts) != 49L) stop(sprintf("expected 49 CODEX proteins; found %d", nrow(counts)))
missing <- setdiff(ids, colnames(counts))
if (length(missing) > 0L) stop(sprintf("CODEX source misses %d frozen IDs", length(missing)))
block <- t(as.matrix(counts[, ids, drop = FALSE]))
if (!identical(rownames(block), ids)) stop("CODEX provider extraction changed row order")
frame <- data.frame(cell_id = ids, block, check.names = FALSE, stringsAsFactors = FALSE)
connection <- gzfile(args[[3]], open = "wt")
write.table(
  frame, connection,
  sep = ",", row.names = FALSE, col.names = TRUE, quote = FALSE, na = "NA"
)
close(connection)
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def buffer_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return hashlib.sha256(array.tobytes()).hexdigest()


def sequence_sha256(values: pd.Series | np.ndarray | list[str]) -> str:
    payload = "".join(f"{value}\n" for value in map(str, values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def relative_to_final_xu(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(FINAL_XU.resolve()))
    except ValueError:
        return str(path)


def write_csv_gz(frame: pd.DataFrame, path: Path) -> None:
    """Write a deterministic, losslessly round-trippable float64 CSV.gz."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(
            filename="", fileobj=raw, mode="wb", compresslevel=6, mtime=0
        ) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
                frame.to_csv(text, index=False, float_format="%.17g", lineterminator="\n")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")


def canonical_active_marker(source_marker: str) -> str:
    marker = str(source_marker).strip()
    if marker.endswith("-AB"):
        marker = marker[:-3]
    return MARKER_ALIASES.get(marker, marker)


def validate_marker_names(source: list[str], active: list[str], expected_width: int) -> None:
    if len(source) != expected_width + 1:
        raise AssertionError(
            f"provider panel has {len(source)} proteins, expected {expected_width + 1}"
        )
    if source.count(HELDOUT) != 1:
        raise AssertionError(f"provider panel must contain CD19 exactly once: {source}")
    if len(active) != expected_width or HELDOUT in active:
        raise AssertionError("active panel width or CD19 exclusion changed")
    if len(set(active)) != len(active):
        raise AssertionError(f"canonical active marker names are not unique: {active}")
    missing_shared = [marker for marker in SHARED_MARKERS if marker not in active]
    if missing_shared:
        raise AssertionError(f"active panel lacks shared markers: {missing_shared}")


def seurat_clr_per_cell(values: np.ndarray) -> np.ndarray:
    """Seurat CLR equivalent for a cell-by-feature matrix, across features.

    For each cell vector x, Seurat's CLR denominator is
    exp(sum(log1p(x[x > 0])) / len(x)); zeros contribute zero to the same
    expression, so this is exp(mean(log1p(x))).
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("CLR input must be a nonempty cell-by-feature matrix")
    if not np.isfinite(values).all() or np.min(values) < 0:
        raise ValueError("Seurat CLR requires finite, nonnegative values")
    log_geometric_mean = np.log1p(values).mean(axis=1)
    denominator = np.exp(log_geometric_mean)
    transformed = np.log1p(values / denominator[:, None])
    if not np.isfinite(transformed).all():
        raise FloatingPointError("CLR produced non-finite values")
    return transformed


def marker_sample_zscore(
    values: np.ndarray,
    dataset: str,
    markers: list[str],
    provider_transform: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Apply the MaxFuse tutorial's dataset-marker scaling (sample SD)."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(markers):
        raise ValueError(f"{dataset}: invalid pre-z matrix or marker order")
    means = values.mean(axis=0)
    standard_deviations = values.std(axis=0, ddof=1)
    bad = (~np.isfinite(standard_deviations)) | (standard_deviations <= 0)
    if np.any(bad):
        raise ValueError(
            f"{dataset}: markers cannot be sample-SD z-scored: "
            f"{[marker for marker, failed in zip(markers, bad) if failed]}"
        )
    processed = (values - means) / standard_deviations
    parameters = pd.DataFrame({
        "dataset": dataset,
        "marker": markers,
        "provider_transform": provider_transform,
        "pre_z_dataset_mean": means,
        "pre_z_dataset_sample_std": standard_deviations,
        "zscore_ddof": 1,
    })
    return processed, parameters


def parse_fcs(path: Path) -> tuple[np.ndarray, list[str]]:
    """Read the event matrix and channel order from a local float FCS 3.0 file."""
    with path.open("rb") as handle:
        header = handle.read(58)
        text_start, text_end = int(header[10:18]), int(header[18:26])
        data_start = int(header[26:34])
        handle.seek(text_start)
        text = handle.read(text_end - text_start + 1).decode("latin1")
    fields = text.split(text[0])
    metadata = {
        fields[position]: fields[position + 1]
        for position in range(1, len(fields) - 1, 2)
        if fields[position]
    }
    n_events, n_parameters = int(metadata["$TOT"]), int(metadata["$PAR"])
    dtype = ">f4" if metadata["$BYTEORD"] == "4,3,2,1" else "<f4"
    if data_start == 0:
        data_start = int(metadata["$BEGINDATA"])
    with path.open("rb") as handle:
        handle.seek(data_start)
        values = np.fromfile(handle, dtype=dtype, count=n_events * n_parameters)
    if values.size != n_events * n_parameters:
        raise EOFError(f"{path}: truncated FCS data segment")
    channels = [metadata[f"$P{position}N"] for position in range(1, n_parameters + 1)]
    return values.reshape(n_events, n_parameters).astype(np.float64), channels


def verify_hash(path: Path, expected: str, role: str) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256(path)
    if observed != expected:
        raise AssertionError(
            f"{role} hash changed: expected {expected}, observed {observed}, path={path}"
        )
    return {
        "role": role,
        "path": relative_to_final_xu(path),
        "sha256": observed,
    }


def load_and_verify_frozen_inputs(
    benchmark_root: Path, raw_root: Path
) -> tuple[dict[str, pd.DataFrame], list[dict[str, str]], dict[str, Any]]:
    manifest_path = benchmark_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    frozen_manifest = json.loads(manifest_path.read_text())
    records: list[dict[str, str]] = [
        verify_hash(manifest_path, sha256(manifest_path), "frozen_manifest")
    ]

    locks: dict[str, pd.DataFrame] = {}
    for slug, spec in EXPECTED.items():
        relative = f"{slug}/locked_metadata.csv.gz"
        path = benchmark_root / relative
        records.append(
            verify_hash(path, frozen_manifest["files_sha256"][relative], f"{slug}.row_lock")
        )
        lock_columns = ["cell_id", "dataset", "donor"]
        if slug == "bm_cytof":
            lock_columns.append("source_fcs")
        locked = pd.read_csv(path, usecols=lock_columns, low_memory=False)
        required = {"cell_id", "dataset", "donor"}
        if required - set(locked):
            raise ValueError(f"{path} lacks row-lock fields: {sorted(required - set(locked))}")
        if len(locked) != spec["cells"]:
            raise AssertionError(f"{slug}: frozen row count changed")
        if locked["cell_id"].astype(str).duplicated().any():
            raise AssertionError(f"{slug}: duplicate frozen cell IDs")
        if locked["dataset"].astype(str).unique().tolist() != [spec["name"]]:
            raise AssertionError(f"{slug}: frozen dataset identity changed")
        locks[slug] = locked

    selected_path = benchmark_root / "selected_cells.csv.gz"
    records.append(
        verify_hash(
            selected_path,
            frozen_manifest["files_sha256"]["selected_cells.csv.gz"],
            "global_row_lock",
        )
    )
    selected = pd.read_csv(
        selected_path, usecols=["cell_id", "dataset", "donor"], low_memory=False
    )
    combined = pd.concat(
        [locks[slug][["cell_id", "dataset", "donor"]] for slug in EXPECTED],
        ignore_index=True,
    )
    if not combined.astype(str).equals(selected[["cell_id", "dataset", "donor"]].astype(str)):
        raise AssertionError("per-dataset row locks do not reproduce selected_cells.csv.gz")

    for relative, expected_hash in frozen_manifest["source_files_sha256"].items():
        records.append(
            verify_hash(raw_root / relative, expected_hash, f"provider_source:{relative}")
        )
    return locks, records, frozen_manifest


def identity_frame(locked: pd.DataFrame) -> pd.DataFrame:
    result = locked[["cell_id", "dataset", "donor"]].copy()
    result["cell_id"] = result["cell_id"].astype(str)
    result["dataset"] = result["dataset"].astype(str)
    result["donor"] = result["donor"].astype(str)
    result["unit_key"] = result["dataset"] + "|" + result["donor"]
    return result[ID_COLUMNS]


def extract_codex(
    locked: pd.DataFrame, raw_root: Path, temporary_root: Path
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    source_rds = raw_root / "BM_Codex" / "Normal_Bone_Marrow_CODEX_Atlas_Seurat_v2.rds"
    export_script = temporary_root / "export_codex_full_panel.R"
    export_script.write_text(CODEX_EXPORT_R)
    exported = temporary_root / "codex_full_panel.csv.gz"
    # Materialize only the verified IDs, avoiding labels and supporting a
    # relocated frozen benchmark without falling back to a different row lock.
    lock_path = temporary_root / "codex_row_lock.csv.gz"
    write_csv_gz(locked[["cell_id"]], lock_path)
    subprocess.run(
        ["Rscript", str(export_script), str(source_rds), str(lock_path), str(exported)],
        check=True,
        cwd=HERE,
    )
    frame = pd.read_csv(exported, low_memory=False)
    expected_ids = locked["cell_id"].astype(str).to_numpy()
    if not np.array_equal(frame["cell_id"].astype(str).to_numpy(), expected_ids):
        raise AssertionError("CODEX provider extraction violated the frozen row lock")
    source_markers = [str(column) for column in frame.columns[1:]]
    canonical = [canonical_active_marker(marker) for marker in source_markers]
    active_positions = [index for index, marker in enumerate(canonical) if marker != HELDOUT]
    active_markers = [canonical[index] for index in active_positions]
    validate_marker_names(canonical, active_markers, EXPECTED["bm_codex"]["active"])
    raw = frame.iloc[:, 1:].to_numpy(np.float64)[:, active_positions]
    processed = seurat_clr_per_cell(raw)
    return raw, processed, source_markers, active_markers


def extract_cytof(
    locked: pd.DataFrame, raw_root: Path
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    source_root = raw_root / "BM_CyTOF" / "levine_CyTOF_Annotated_Celltypes_fcs"
    channels = pd.read_csv(
        source_root / "Levine_CellTypesAnnotated_channels.csv", dtype=str
    ).fillna("")
    biological = channels.loc[
        channels["usedforPhenoGraphclustering"].str.lower() == "yes",
        ["channel", "marker"],
    ].copy()
    source_markers = biological["marker"].astype(str).tolist()
    canonical = [canonical_active_marker(marker) for marker in source_markers]
    active_positions = [index for index, marker in enumerate(canonical) if marker != HELDOUT]
    active_markers = [canonical[index] for index in active_positions]
    validate_marker_names(canonical, active_markers, EXPECTED["bm_cytof"]["active"])
    active_channels = biological["channel"].astype(str).to_numpy()[active_positions]

    if "source_fcs" not in locked:
        raise ValueError("CyTOF row lock lacks source_fcs")
    raw = np.empty((len(locked), len(active_markers)), dtype=np.float64)
    event_pattern = re.compile(r"::event_([1-9][0-9]*)$")
    for filename, row_index in locked.groupby("source_fcs", sort=False).groups.items():
        positions = np.asarray(list(row_index), dtype=np.int64)
        matrix, fcs_channels = parse_fcs(source_root / str(filename))
        lookup = {channel: index for index, channel in enumerate(fcs_channels)}
        missing_channels = [channel for channel in active_channels if channel not in lookup]
        if missing_channels:
            raise AssertionError(f"{filename}: missing biological channels {missing_channels}")
        cell_ids = locked.iloc[positions]["cell_id"].astype(str).tolist()
        event_numbers: list[int] = []
        prefix = f"{filename}::event_"
        for cell_id in cell_ids:
            match = event_pattern.search(cell_id)
            if not cell_id.startswith(prefix) or match is None:
                raise AssertionError(f"CyTOF cell ID does not encode source event: {cell_id}")
            event_numbers.append(int(match.group(1)))
        events = np.asarray(event_numbers, dtype=np.int64) - 1
        if np.min(events) < 0 or np.max(events) >= len(matrix):
            raise IndexError(f"{filename}: frozen event index is outside the provider FCS")
        channel_positions = [lookup[channel] for channel in active_channels]
        raw[positions] = matrix[events][:, channel_positions]

    transformed = np.arcsinh(raw / 5.0)
    return raw, transformed, source_markers, active_markers


def extract_triana(
    locked: pd.DataFrame, raw_root: Path
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    source_root = raw_root / "Triana_2021" / "exported"
    source_markers = [
        line.strip()
        for line in (source_root / "ab_features.tsv").read_text().splitlines()
        if line.strip()
    ]
    barcodes = [
        line.strip()
        for line in (source_root / "ab_barcodes.tsv").read_text().splitlines()
        if line.strip()
    ]
    metadata = pd.read_csv(
        source_root / "meta.csv", usecols=["barcode", "Batch"], low_memory=False
    )
    if barcodes != metadata["barcode"].astype(str).tolist():
        raise AssertionError("Triana matrix barcode order differs from provider metadata")
    canonical = [canonical_active_marker(marker) for marker in source_markers]
    active_positions = [index for index, marker in enumerate(canonical) if marker != HELDOUT]
    active_markers = [canonical[index] for index in active_positions]
    validate_marker_names(canonical, active_markers, EXPECTED["triana_abseq"]["active"])

    cell_lookup = pd.Index(barcodes)
    locked_ids = locked["cell_id"].astype(str).to_numpy()
    cell_positions = cell_lookup.get_indexer(locked_ids)
    if np.any(cell_positions < 0):
        raise AssertionError(
            f"Triana source misses {int(np.sum(cell_positions < 0))} frozen barcodes"
        )
    source_donors = metadata.iloc[cell_positions]["Batch"].astype(str).to_numpy()
    if not np.array_equal(source_donors, locked["donor"].astype(str).to_numpy()):
        raise AssertionError("Triana provider donor metadata violates the frozen row lock")

    counts = mmread(source_root / "ab_counts.mtx")
    counts = counts.tocsr() if sparse.issparse(counts) else sparse.csr_matrix(counts)
    if counts.shape != (len(source_markers), len(barcodes)):
        raise AssertionError(
            f"Triana matrix shape {counts.shape} != {(len(source_markers), len(barcodes))}"
        )
    raw = counts[active_positions][:, cell_positions].T.toarray().astype(np.float64)
    processed = seurat_clr_per_cell(raw)
    return raw, processed, source_markers, active_markers


def source_to_active_map(
    source_markers: list[str], active_markers: list[str]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    active_position = 0
    for source_position, source_marker in enumerate(source_markers, start=1):
        canonical = canonical_active_marker(source_marker)
        if canonical == HELDOUT:
            rows.append({
                "source_position": source_position,
                "source_marker": source_marker,
                "active_position": pd.NA,
                "active_marker": pd.NA,
                "status": "heldout_excluded_before_transform",
                "is_shared": False,
            })
            continue
        active_position += 1
        if active_markers[active_position - 1] != canonical:
            raise AssertionError("source-to-active marker map lost provider order")
        rows.append({
            "source_position": source_position,
            "source_marker": source_marker,
            "active_position": active_position,
            "active_marker": canonical,
            "status": "active",
            "is_shared": canonical in SHARED_MARKERS,
        })
    return pd.DataFrame(rows)


def numeric_audit(
    dataset: str,
    active_markers: list[str],
    raw: np.ndarray,
    processed: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame({
        "dataset": dataset,
        "marker": active_markers,
        "is_shared": [marker in SHARED_MARKERS for marker in active_markers],
        "raw_min": raw.min(axis=0),
        "raw_max": raw.max(axis=0),
        "raw_mean": raw.mean(axis=0),
        "processed_min": processed.min(axis=0),
        "processed_max": processed.max(axis=0),
        "processed_mean": processed.mean(axis=0),
        "processed_population_std": processed.std(axis=0, ddof=0),
        "processed_sample_std": processed.std(axis=0, ddof=1),
        "finite": np.isfinite(processed).all(axis=0),
    })


def validate_local_config(dataset_records: dict[str, dict[str, Any]]) -> None:
    config_path = HERE / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text())
    if config.get("result_subdir") != "maxfuse_v2":
        raise AssertionError("config.json must isolate outputs in result_subdir=maxfuse_v2")
    if config.get("shared_markers") != SHARED_MARKERS:
        raise AssertionError("config shared marker order differs from the builder")
    if config.get("feature_protocol", {}).get("active_mode") != "dataset_active_inputs":
        raise AssertionError("config does not enable dataset-native active inputs")
    if config.get("fit_sampling", {}).get("cap_per_dataset", "missing") is not None:
        raise AssertionError("MaxFuse v2 must not benchmark-level cap the frozen cohorts")
    for spec in config["datasets"]:
        record = dataset_records[spec["slug"]]
        if spec.get("active_markers") != record["active_markers"]:
            raise AssertionError(f"{spec['slug']}: config active marker order differs from builder")
        if int(spec["expected_cells"]) != record["n_cells"]:
            raise AssertionError(f"{spec['slug']}: config expected_cells changed")


def r_software_versions() -> dict[str, str]:
    expression = (
        'cat(paste(c(as.character(getRversion()), '
        'as.character(packageVersion("SeuratObject")), '
        'as.character(packageVersion("Matrix"))), collapse="\\n"))'
    )
    values = subprocess.check_output(["Rscript", "-e", expression], text=True).splitlines()
    if len(values) != 3:
        raise RuntimeError(f"unexpected R version output: {values}")
    return {"R": values[0], "SeuratObject": values[1], "Matrix": values[2]}


def build() -> dict[str, Any]:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    benchmark_root = args.benchmark_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir != DEFAULT_OUTPUT_DIR.resolve():
        raise ValueError(
            "This scoped MaxFuse v2 builder owns --output-dir; refusing to write "
            f"outside {DEFAULT_OUTPUT_DIR.resolve()}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    locks, source_records, frozen_manifest = load_and_verify_frozen_inputs(
        benchmark_root, raw_root
    )
    with tempfile.TemporaryDirectory(prefix=".bm_maxfuse_v2_", dir=output_dir.parent) as tmp:
        temporary_root = Path(tmp)
        stage = temporary_root / "processed"
        stage.mkdir()

        codex = extract_codex(locks["bm_codex"], raw_root, temporary_root)
        cytof = extract_cytof(locks["bm_cytof"], raw_root)
        triana = extract_triana(locks["triana_abseq"], raw_root)
        extracted: dict[str, tuple[np.ndarray, np.ndarray, list[str], list[str]]] = {
            "bm_codex": codex,
            "bm_cytof": cytof,
            "triana_abseq": triana,
        }
        provider_transforms = {
            "bm_codex": "CD19-excluded per-cell Seurat CLR",
            "bm_cytof": "CD19-excluded asinh(x/5)",
            "triana_abseq": "CD19-excluded per-cell Seurat CLR",
        }

        marker_audits: list[pd.DataFrame] = []
        numeric_audits: list[pd.DataFrame] = []
        zscore_audits: list[pd.DataFrame] = []
        row_audits: list[dict[str, Any]] = []
        dataset_records: dict[str, dict[str, Any]] = {}
        for slug, spec in EXPECTED.items():
            raw, provider_transformed, source_markers, active_markers = extracted[slug]
            processed, zscore_parameters = marker_sample_zscore(
                provider_transformed,
                spec["name"],
                active_markers,
                provider_transforms[slug],
            )
            zscore_audits.append(zscore_parameters)
            locked = locks[slug]
            if raw.shape != (spec["cells"], spec["active"]):
                raise AssertionError(f"{slug}: raw active shape changed: {raw.shape}")
            if processed.shape != raw.shape or not np.isfinite(processed).all():
                raise AssertionError(f"{slug}: processed active matrix is invalid")
            within_cell_variance = processed.var(axis=1, ddof=0)
            zero_within_cell_variance = int(np.sum(within_cell_variance <= 0))
            if zero_within_cell_variance:
                raise AssertionError(
                    f"{slug}: {zero_within_cell_variance} processed rows have zero "
                    "within-cell variance and would violate the full row lock"
                )
            shared_positions = [active_markers.index(marker) for marker in SHARED_MARKERS]
            shared = processed[:, shared_positions]
            if shared.shape != (spec["cells"], len(SHARED_MARKERS)):
                raise AssertionError(f"{slug}: shared extraction shape changed")

            identity = identity_frame(locked)
            active_frame = pd.concat(
                [identity, pd.DataFrame(processed, columns=active_markers)], axis=1
            )
            shared_frame = pd.concat(
                [identity, pd.DataFrame(shared, columns=SHARED_MARKERS)], axis=1
            )
            active_relative = f"{slug}/active_input_{spec['active']}.csv.gz"
            shared_relative = f"{slug}/shared_input_14.csv.gz"
            write_csv_gz(active_frame, stage / active_relative)
            write_csv_gz(shared_frame, stage / shared_relative)

            map_frame = source_to_active_map(source_markers, active_markers)
            map_frame.insert(0, "dataset", spec["name"])
            marker_audits.append(map_frame)
            numeric_audits.append(
                numeric_audit(spec["name"], active_markers, raw, processed)
            )
            row_digest = sequence_sha256(identity["cell_id"])
            row_audits.append({
                "dataset": spec["name"],
                "n_cells": len(identity),
                "n_donors": identity["donor"].nunique(),
                "first_cell_id": identity.iloc[0]["cell_id"],
                "last_cell_id": identity.iloc[-1]["cell_id"],
                "cell_id_order_sha256": row_digest,
                "active_width": len(active_markers),
                "shared_width": len(SHARED_MARKERS),
                "zero_within_cell_variance_rows": zero_within_cell_variance,
                "minimum_within_cell_population_variance": float(
                    within_cell_variance.min()
                ),
                "shared_extracted_from_processed_active": True,
            })
            dataset_records[slug] = {
                "name": spec["name"],
                "n_cells": len(identity),
                "n_donors": int(identity["donor"].nunique()),
                "provider_panel_width_including_cd19": len(source_markers),
                "active_width": len(active_markers),
                "shared_width": len(SHARED_MARKERS),
                "zero_within_cell_variance_rows": zero_within_cell_variance,
                "minimum_within_cell_population_variance": float(
                    within_cell_variance.min()
                ),
                "active_markers": active_markers,
                "shared_markers": SHARED_MARKERS,
                "active_input": active_relative,
                "shared_input": shared_relative,
                "cell_id_order_sha256": row_digest,
                "raw_active_float64_buffer_sha256": buffer_sha256(raw),
                "provider_transformed_active_float64_buffer_sha256": buffer_sha256(
                    provider_transformed
                ),
                "processed_active_float64_buffer_sha256": buffer_sha256(processed),
                "shared_float64_buffer_sha256": buffer_sha256(shared),
                "shared_extracted_from_same_processed_active_columns": True,
                "shared_extraction_max_absolute_difference": 0.0,
            }

        write_csv(pd.concat(marker_audits, ignore_index=True), stage / "marker_audit.csv")
        write_csv(pd.concat(numeric_audits, ignore_index=True), stage / "numeric_audit.csv")
        write_csv(pd.DataFrame(row_audits), stage / "row_lock_audit.csv")
        write_csv(
            pd.concat(zscore_audits, ignore_index=True),
            stage / "method_zscore_parameters.csv",
        )

        validate_local_config(dataset_records)
        generated = sorted(path for path in stage.rglob("*") if path.is_file())
        manifest = {
            "schema_version": "final_xu.bone_marrow.maxfuse_v2.inputs.v2",
            "benchmark": "bone_marrow",
            "purpose": (
                "label-free MaxFuse v2 native active inputs with provider/paper transforms "
                "followed by official-tutorial method-level marker scaling"
            ),
            "frozen_benchmark_package": frozen_manifest.get("package"),
            "frozen_total_cells": int(sum(spec["cells"] for spec in EXPECTED.values())),
            "row_lock": (
                "Exact cell IDs, dataset blocks, and order come from frozen locked_metadata "
                "and selected_cells; provider labels are never read into output matrices."
            ),
            "heldout_target": HELDOUT,
            "heldout_policy": (
                "CD19 is removed from each reliable full provider panel before transformation; "
                "it is absent from active inputs, shared inputs, and both CLR denominators."
            ),
            "shared_panel_policy": (
                "The 14 shared columns are direct column slices of the already processed "
                "dataset-native active matrix in the declared shared-marker order."
            ),
            "protocols": {
                "BM CODEX": (
                    "Provider CODEX counts slot; remove CD19; per-cell Seurat-style CLR "
                    "across the remaining 48 proteins. This intentional benchmark transform "
                    "is recomputed and is not the provider object's stored CODEX@data. "
                    "Then apply within-dataset marker sample-SD z-score (ddof=1) as the "
                    "MaxFuse method-level sc.pp.scale step."
                ),
                "BM CyTOF": (
                    "Provider cct_subtracted_normalized annotated FCS; select 32 biological "
                    "channels, remove CD19, apply asinh(x/5), then full frozen-dataset "
                    "per-marker sample-SD z-score (ddof=1), matching current scanpy.pp.scale."
                ),
                "Triana AbSeq": (
                    "Raw antibody UMI counts; remove CD19; per-cell Seurat-style CLR "
                    "across the remaining 96 proteins; then apply within-dataset marker "
                    "sample-SD z-score (ddof=1) as the MaxFuse method-level sc.pp.scale step."
                ),
            },
            "clr_post_zscore_applied": {
                "BM CODEX": True,
                "Triana AbSeq": True,
            },
            "cytof_cohort_scope": {
                "current_source": (
                    "Levine Benchmark Data Set 2: healthy H1/H2 with 32 surface markers"
                ),
                "transform": (
                    "Provider-normalized annotated FCS values followed by asinh(x/5) and "
                    "frozen-dataset marker sample-SD z-scoring (ddof=1)."
                ),
                "nonapplicable_paper_protocol": (
                    "The H1-H5 per-marker 99.5th-percentile scaling belongs to the distinct "
                    "16-surface-marker AML/signaling cohort and is not applied or treated as "
                    "a missing input for Benchmark Data Set 2."
                ),
            },
            "datasets": dataset_records,
            "source_records": source_records,
            "software_versions": {
                "Python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scipy": scipy.__version__,
                **r_software_versions(),
            },
            "recipe_sha256": sha256(Path(__file__).resolve()),
            "config_sha256": sha256(HERE / "config.json"),
            "pairwise_config_sha256": sha256(HERE / "pairwise_config.json"),
            "files_sha256": {
                str(path.relative_to(stage)): sha256(path) for path in generated
            },
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )

        # Publish only inside the v2 output tree. The v1 maxfuse directory is never touched.
        output_dir.mkdir(parents=True, exist_ok=True)
        # v1 wrote a CyTOF-only parameter table. v2 replaces it with one
        # method-level table covering all datasets, so do not leave an
        # untracked stale audit file in the regenerated package.
        (output_dir / "cytof_zscore_parameters.csv").unlink(missing_ok=True)
        for source in sorted(path for path in stage.rglob("*") if path.is_file()):
            target = output_dir / source.relative_to(stage)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, target)

    summary = {
        "output_dir": str(output_dir),
        "shapes": {
            slug: {
                "active": [record["n_cells"], record["active_width"]],
                "shared": [record["n_cells"], record["shared_width"]],
            }
            for slug, record in dataset_records.items()
        },
        "files_sha256": manifest["files_sha256"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return manifest


if __name__ == "__main__":
    build()
