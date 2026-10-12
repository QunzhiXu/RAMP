#!/usr/bin/env python3
"""Build canonical-benchmark, row-locked TNBC MaxFuse Archive inputs.

The frozen 62,701-cell TNBC benchmark owns cell selection and row order.  This
builder recovers the provider full active panels for exactly those cells, then
applies one predeclared transform independently to every marker within each
dataset::

    zscore(asinh(x / 1.0)), sample SD (ddof=1)

The sample-SD convention matches ``scanpy.pp.scale`` / ``scale_array`` with
``mean_var(correction=1)`` as used by the official MaxFuse notebook.  The
eleven shared columns are sliced directly from each processed active matrix.
In particular, Shiao marker values retain the canonical v1 arithmetic
mean of the nuclei and membrane compartment summaries; this builder does not
introduce a MaxFuse-specific compartment rule.

All persistent writes are confined to this directory's ``input/`` subdirectory.
Existing complete inputs are verified and reused.  A stale derived input tree
is rebuilt in a staging directory and replaced only after every row-lock,
round-trip, and hash audit has passed; provider raw rows are temporary and are
not retained.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any

import h5py
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
FINAL_XU = HERE.parents[2]
BENCHMARK = FINAL_XU / "benchmark_data" / "tnbc"
CANONICAL_LOCK = BENCHMARK / "selected_cells.csv.gz"
EXTRACTION_RECIPE = HERE / "extract_provider_rows.R"
TIETSCHER_RDS = (
    FINAL_XU
    / "raw_data"
    / "tnbc"
    / "Tietscher_2023"
    / "Protein_panel_SingleCellExperiment_object.rds"
)
MEYER_RDS = (
    FINAL_XU
    / "raw_data"
    / "tnbc"
    / "Meyer_2025"
    / "processed_r_objects"
    / "sce_ALL.rds"
)
SHIAO_COUNTS = (
    FINAL_XU
    / "raw_data"
    / "tnbc"
    / "Shiao_2024"
    / "TNBC_immune_CODEX_full_panel_counts.csv"
)
OUTPUT = HERE / "input"

SCHEMA = "final_xu.tnbc.maxfuse_archive.inputs.v1"
COFACTOR = 1.0
ZSCORE_DDOF = 1
EXPECTED_SHIAO_SOURCE_ROWS = 2_597_738

DATASETS = ["Tietscher IMC", "Meyer IMC", "Shiao CODEX"]
DATASET_SLUGS = {
    "Tietscher IMC": "tietscher_imc",
    "Meyer IMC": "meyer_imc",
    "Shiao CODEX": "shiao_codex",
}
EXPECTED_CELLS = {
    "Tietscher IMC": 20_000,
    "Meyer IMC": 22_701,
    "Shiao CODEX": 20_000,
}
SHARED_MARKERS = [
    "CD11C", "CD20", "CD3", "CD31", "CD4", "CD68", "CD8",
    "HLA_DR", "KI67", "PANCK", "SMA",
]

# Full provider biological panels.  The order is the provider channel order
# after only the predeclared acquisition/reference channels are removed.
EXPECTED_ACTIVE_MARKERS = {
    "Tietscher IMC": [
        "SMA", "ECAD", "PANCK", "CD38", "HLA_DR", "CD27", "CD303",
        "CD68", "LAG3", "CD20", "CD7", "CD45RA", "IDO1", "CD3",
        "LAMP3", "CD11C", "PD1", "TIM3", "MMP9", "GRANZYMEB",
        "PDL1", "TCF7", "CD45RO", "FOXP3", "ICOS", "CD8", "CAIX",
        "GITR", "ARGINASE", "CD140B", "CD40", "CD4", "CD31", "CD21",
        "HLAABC", "INOS", "CLEAVEDCP", "KI67", "CD15", "MPO",
    ],
    "Meyer IMC": [
        "H3K27ME3", "CK5", "FIBRONECTIN", "HLA_DR", "CK818", "CD15",
        "CD68", "KRT14", "SMA", "VIMENTIN", "CMYC", "HER2", "CD3",
        "PH3", "CD11C", "BCL2", "ER", "PR", "P53", "CD44", "AR",
        "CD45", "GATA3", "CD20", "CD8", "CAIX", "ECAD", "KI67",
        "EGFR", "S6", "CD4", "CD31", "PMTOR", "CK7", "PANCK",
        "CCAS3PARP",
    ],
    "Shiao CODEX": [
        "CD8", "CD4", "PANCK", "OX40L", "CD45", "CD20", "CD134",
        "CD68", "CD31", "CD103", "HLA_DR", "CXCR5", "IGG", "CD3",
        "KI67", "LAG3", "CXCL13", "IGA", "CD89", "PNAD", "PDL1",
        "PD1", "CD11C", "CD80", "CD69", "TIM3", "CD45RO", "CD40LG",
        "FOXP3", "CD64", "GRANZYMEB", "C1Q", "CD40", "CD45RA",
        "CD138", "IL7R", "IGM", "PDGFRB", "SMA",
    ],
}

EXPECTED_H5_SOURCE_MARKERS = {
    "Tietscher IMC": [
        "HH3", "SMA", "E-Cadherin", "panCK", "CD38", "HLA-DR", "CD27",
        "CD303", "CD68", "LAG-3", "CD20", "CD7", "CD45RA", "IDO1",
        "CD3", "LAMP3", "CD11c", "PD1", "Tim-3", "MMP9", "Granzyme_B",
        "CD274", "TCF7", "CD45RO", "FOXP3", "ICOS", "CD8a", "CAIX",
        "GITR", "Arginase", "CD140b", "CD40", "CD4", "CD31_vWF", "CD21",
        "HLA-ABC", "iNOS", "Cleaved_CP", "DNA1", "DNA2", "Ki-67",
        "CD15", "MPO",
    ],
    "Meyer IMC": [
        "HistoneH3", "H3K27me3", "CK5", "Fibronectin", "HLADR", "CK8_18",
        "CD15", "CD68", "KRT14", "SMA", "Vimentin", "cMYC", "HER2",
        "CD3", "pH3", "CD11c", "Bcl2", "ER", "PR", "p53", "CD44", "AR",
        "CD45", "GATA3", "CD20", "CD8a", "CAIX", "Ecad", "Ki67", "EGFR",
        "S6", "CD4", "CD31_vWF", "p_mTOR", "CK7", "panCK", "c_Cas3_PARP",
        "DNA_1", "DNA_2",
    ],
}

EXCLUDED_SOURCE_MARKERS = {
    "Tietscher IMC": {"HH3", "DNA1", "DNA2"},
    # H3K27me3 and pH3 are biological state channels and are intentionally
    # retained.  The historical 34-marker MaxFuse input excluded them.
    "Meyer IMC": {"HistoneH3", "DNA_1", "DNA_2"},
    "Shiao CODEX": {"DAPI"},
}

SHIAO_SOURCE_ROOTS = [
    "DAPI", "CD8", "CD4", "PanCytoK", "OX40L", "CD45", "CD20", "CD134",
    "CD68", "CD31", "CD103", "HLA-DR", "CXCR5", "IgG", "CD3e", "Ki-67",
    "LAG3", "CXCL13", "IgA", "CD89", "PNaD", "PD-L1", "PD-1", "CD11c",
    "CD80", "CD69", "TIM3", "CD45RO", "CD40LG", "FOXP3", "CD64", "GZMB",
    "C1q", "CD40", "CD45RA", "CD138", "IL7R", "IgM", "PDGFRb", "aSMA",
]
SHIAO_ID_COLUMN = "Unnamed: 0"

EXPECTED_SHA256 = {
    "canonical_cell_lock": "55ea28f02d14618b307f0309a7debda77381ec9769b9876c442d58c9a794dd1e",
    "tietscher_rds": "660634f9f96e64ab6c3158510ebe6631b4943218aaafd0993539bff6661e9928",
    "meyer_rds": "cbe2a4e3f857f08e8516c045dcea9bd098c1e35e8bb45438ab6962650b6e5974",
    "shiao_counts": "0697184090d5373c4d9ee7a052863504ff6984e9a88674c7158bdd3713361c96",
}

CANONICAL_IDENTITY_COLUMNS = [
    "cell_order",
    "cell_id",
    "source_cell_id",
    "source_index_zero_based",
    "dataset",
    "donor",
    "unit_key",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def buffer_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def string_rows_sha256(values: pd.Series | np.ndarray | list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def verify_expected_hash(path: Path, key: str) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256(path)
    expected = EXPECTED_SHA256[key]
    if observed != expected:
        raise AssertionError(
            f"Frozen source changed: {path}; expected {expected}, observed {observed}"
        )
    return {"path": str(path.resolve()), "sha256": observed}


def decode_strings(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in values
    ]


def compact_marker(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(value)).upper()


def canonical_marker(value: object) -> str:
    key = compact_marker(value)
    aliases = {
        "ASMA": "SMA",
        "ACTA2": "SMA",
        "CD3E": "CD3",
        "CD8A": "CD8",
        "CD31VWF": "CD31",
        "VWFCD31": "CD31",
        "HLADR": "HLA_DR",
        "HLADPDQDR": "HLA_DR",
        "PANCYTOK": "PANCK",
        "PANCYTOKERATIN": "PANCK",
        "ECADHERIN": "ECAD",
        "ECAD": "ECAD",
        "CD274": "PDL1",
        "GRANZYMEB": "GRANZYMEB",
        "GZMB": "GRANZYMEB",
        "CD279": "PD1",
        "CD279PD1": "PD1",
    }
    return aliases.get(key, key)


def deterministic_csv(frame: pd.DataFrame, path: Path, compressed: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not compressed:
        frame.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")
        return
    with path.open("wb") as raw:
        with gzip.GzipFile(
            filename="", fileobj=raw, mode="wb", compresslevel=6, mtime=0
        ) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
                frame.to_csv(
                    text, index=False, float_format="%.17g", lineterminator="\n"
                )


def load_canonical_lock() -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    source_records = {
        "canonical_cell_lock": verify_expected_hash(
            CANONICAL_LOCK, "canonical_cell_lock"
        ),
    }
    # The numerical build reads identity/order/source-location fields only.
    # Provider cell-type labels, coordinates, treatments, and outcomes are not
    # loaded into this process.
    metadata = pd.read_csv(
        CANONICAL_LOCK,
        usecols=CANONICAL_IDENTITY_COLUMNS,
        dtype={
            "cell_id": str,
            "source_cell_id": str,
            "dataset": str,
            "donor": str,
            "unit_key": str,
        },
        low_memory=False,
    )
    required = set(CANONICAL_IDENTITY_COLUMNS)
    if not required.issubset(metadata.columns):
        raise AssertionError(
            f"Canonical metadata lacks {sorted(required - set(metadata.columns))}"
        )
    if len(metadata) != 62_701:
        raise AssertionError(f"Canonical cell-lock row count changed: {len(metadata)}")
    if not np.array_equal(metadata["cell_order"].to_numpy(), np.arange(len(metadata))):
        raise AssertionError("Canonical cell_order no longer locks matrix row order")
    if metadata["cell_id"].astype(str).duplicated().any():
        raise AssertionError("Canonical cell_id values are not globally unique")
    if metadata["dataset"].astype(str).drop_duplicates().tolist() != DATASETS:
        raise AssertionError("Canonical dataset block order changed")
    observed_counts = (
        metadata["dataset"].astype(str).value_counts(sort=False).astype(int).to_dict()
    )
    if observed_counts != EXPECTED_CELLS:
        raise AssertionError(f"Canonical dataset sizes changed: {observed_counts}")
    return metadata, source_records


def active_from_source(
    dataset: str,
    values: np.ndarray,
    source_markers: list[str],
) -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
    if values.shape[1] != len(source_markers):
        raise AssertionError(f"{dataset}: provider matrix/marker mismatch")
    keep: list[int] = []
    active_markers: list[str] = []
    audit: list[dict[str, Any]] = []
    exclusions = EXCLUDED_SOURCE_MARKERS[dataset]
    for source_index, source_marker in enumerate(source_markers):
        excluded = source_marker in exclusions
        marker = canonical_marker(source_marker)
        active_index: int | None = None
        if not excluded:
            active_index = len(keep)
            keep.append(source_index)
            active_markers.append(marker)
        audit.append({
            "dataset": dataset,
            "slug": DATASET_SLUGS[dataset],
            "provider_channel_index": source_index,
            "active_index": active_index,
            "source_marker": source_marker,
            "canonical_marker": marker,
            "included_in_active": not excluded,
            "exclusion_reason": (
                "predeclared acquisition/reference channel" if excluded else ""
            ),
        })
    expected = EXPECTED_ACTIVE_MARKERS[dataset]
    if active_markers != expected:
        raise AssertionError(
            f"{dataset}: provider-full active panel changed; "
            f"expected {expected}, observed {active_markers}"
        )
    if len(set(active_markers)) != len(active_markers):
        raise AssertionError(f"{dataset}: canonical active markers are not unique")
    return np.asarray(values[:, keep], dtype=np.float64), active_markers, audit


def extract_provider_rows(destination: Path) -> dict[str, Path]:
    """Run the local read-only R bridge against provider RDS files.

    The bridge exports only the frozen rows and is created under a temporary
    directory.  These untransformed matrices are deleted after the processed
    inputs and audits have been installed.
    """

    if not EXTRACTION_RECIPE.is_file():
        raise FileNotFoundError(EXTRACTION_RECIPE)
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError("Rscript is required to read the provider SCE/RDS files")
    metadata_path = CANONICAL_LOCK
    command = [
        rscript,
        str(EXTRACTION_RECIPE),
        str(metadata_path),
        str(TIETSCHER_RDS),
        str(MEYER_RDS),
        str(destination),
    ]
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Provider-row extraction failed. R output follows:\n"
            + completed.stdout
        )
    print(completed.stdout, end="", flush=True)
    exports = {
        "Tietscher IMC": destination / "tietscher_provider_rows.h5",
        "Meyer IMC": destination / "meyer_provider_rows.h5",
    }
    missing = [str(path) for path in exports.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Provider extraction did not create: {missing}")
    return exports


def load_h5_dataset(
    dataset: str,
    path: Path,
    metadata: pd.DataFrame,
) -> tuple[np.ndarray, list[str], list[dict[str, Any]], dict[str, Any]]:
    expected_source_markers = EXPECTED_H5_SOURCE_MARKERS[dataset]
    with h5py.File(path, "r") as handle:
        required = {"X_features_by_cells", "source_markers", "source_cell_ids"}
        if not required.issubset(handle.keys()):
            raise AssertionError(f"{dataset}: provider HDF5 keys changed")
        stored = np.asarray(handle["X_features_by_cells"][:], dtype=np.float64)
        source_markers = decode_strings(handle["source_markers"][:])
        source_ids = decode_strings(handle["source_cell_ids"][:])
    if source_markers != expected_source_markers:
        raise AssertionError(f"{dataset}: provider HDF5 marker order changed")
    if stored.shape == (len(metadata), len(source_markers)):
        values = stored
        orientation = "cells_by_features"
    elif stored.shape == (len(source_markers), len(metadata)):
        values = stored.T
        orientation = "features_by_cells_transposed"
    else:
        raise AssertionError(f"{dataset}: cannot orient provider matrix {stored.shape}")
    expected_ids = metadata["source_cell_id"].astype(str).tolist()
    if source_ids != expected_ids:
        raise AssertionError(f"{dataset}: provider HDF5 row IDs differ from frozen rows")
    active, markers, marker_audit = active_from_source(dataset, values, source_markers)
    return active, markers, marker_audit, {
        "provider_shape": [int(values.shape[0]), int(values.shape[1])],
        "provider_marker_count": len(source_markers),
        "orientation": orientation,
        "source_cell_ids_match_frozen_order": True,
        "aggregation": "none; provider SCE counts-assay channel value",
    }


def selected_shiao_rows(
    selected_rows: np.ndarray,
    usecols: list[str],
) -> tuple[pd.DataFrame, int]:
    selected_rows = np.asarray(selected_rows, dtype=np.int64)
    if np.any(selected_rows < 0) or len(np.unique(selected_rows)) != len(selected_rows):
        raise AssertionError("Frozen Shiao source-row indices are invalid")
    sorted_rows = np.sort(selected_rows)
    pieces: list[pd.DataFrame] = []
    offset = 0
    for chunk in pd.read_csv(
        SHIAO_COUNTS,
        usecols=usecols,
        dtype={SHIAO_ID_COLUMN: str},
        chunksize=250_000,
        low_memory=False,
    ):
        left = np.searchsorted(sorted_rows, offset, side="left")
        right = np.searchsorted(sorted_rows, offset + len(chunk), side="left")
        if right > left:
            local_rows = sorted_rows[left:right] - offset
            piece = chunk.iloc[local_rows].copy()
            piece["_source_index_zero_based"] = sorted_rows[left:right]
            pieces.append(piece)
        offset += len(chunk)
    if offset != EXPECTED_SHIAO_SOURCE_ROWS:
        raise AssertionError(f"Shiao provider source row count changed: {offset}")
    if not pieces:
        raise AssertionError("No frozen Shiao rows were recovered")
    recovered = pd.concat(pieces, ignore_index=True)
    if len(recovered) != len(selected_rows):
        raise AssertionError(
            f"Recovered {len(recovered)} Shiao rows, expected {len(selected_rows)}"
        )
    recovered = recovered.set_index("_source_index_zero_based").loc[selected_rows]
    return recovered.reset_index(), offset


def load_shiao_dataset(
    metadata: pd.DataFrame,
) -> tuple[np.ndarray, list[str], list[dict[str, Any]], dict[str, Any]]:
    header = pd.read_csv(SHIAO_COUNTS, nrows=0).columns.astype(str).tolist()
    required_columns = [SHIAO_ID_COLUMN]
    for root in SHIAO_SOURCE_ROOTS:
        required_columns.extend(
            [f"{root}_nuclei_mean", f"{root}_membrane_mean"]
        )
    missing = [column for column in required_columns if column not in header]
    if missing:
        raise AssertionError(f"Shiao full-panel source lacks columns: {missing}")
    selected = metadata["source_index_zero_based"].to_numpy(np.int64)
    frame, source_rows = selected_shiao_rows(selected, required_columns)
    observed_ids = frame[SHIAO_ID_COLUMN].astype(str).to_numpy()
    expected_ids = metadata["source_cell_id"].astype(str).to_numpy()
    if not np.array_equal(observed_ids, expected_ids):
        mismatch = int(np.flatnonzero(observed_ids != expected_ids)[0])
        raise AssertionError(
            f"Shiao provider cell ID mismatch at frozen local row {mismatch}"
        )

    provider_columns: list[np.ndarray] = []
    for root in SHIAO_SOURCE_ROOTS:
        compartments = [f"{root}_nuclei_mean", f"{root}_membrane_mean"]
        # This is the canonical v1 definition for every Shiao channel, including
        # the shared eleven.  It is deliberately not marker-compartment specific.
        provider_columns.append(
            frame[compartments].mean(axis=1).to_numpy(np.float64)
        )
    provider_values = np.column_stack(provider_columns)
    active, markers, marker_audit = active_from_source(
        "Shiao CODEX", provider_values, SHIAO_SOURCE_ROOTS
    )
    for row in marker_audit:
        root = str(row["source_marker"])
        row["source_columns"] = (
            f"{root}_nuclei_mean;{root}_membrane_mean"
        )
        row["source_aggregation"] = "arithmetic_mean_nuclei_membrane"
    return active, markers, marker_audit, {
        "provider_shape": [source_rows, len(SHIAO_SOURCE_ROOTS)],
        "provider_marker_count": len(SHIAO_SOURCE_ROOTS),
        "selected_source_rows": int(len(frame)),
        "source_index_semantics": "zero-based CSV data row excluding header",
        "source_cell_ids_match_frozen_order": True,
        "aggregation": "arithmetic mean of nuclei_mean and membrane_mean",
        "canonical_v1_shared_definition_preserved": True,
        "maxfuse_only_compartment_override": False,
    }


def transform_dataset(
    dataset: str,
    raw: np.ndarray,
    markers: list[str],
    source_by_marker: dict[str, str],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    raw = np.asarray(raw, dtype=np.float64)
    if raw.shape != (EXPECTED_CELLS[dataset], len(markers)):
        raise AssertionError(f"{dataset}: active raw shape changed: {raw.shape}")
    if not np.isfinite(raw).all() or np.any(raw < 0):
        raise AssertionError(f"{dataset}: active raw values must be finite/nonnegative")
    asinh_values = np.arcsinh(raw / COFACTOR)
    means = asinh_values.mean(axis=0, dtype=np.float64)
    standard_deviations = asinh_values.std(axis=0, ddof=ZSCORE_DDOF)
    if np.any(~np.isfinite(standard_deviations)) or np.any(standard_deviations <= 1e-12):
        bad = [
            marker for marker, sd in zip(markers, standard_deviations) if sd <= 1e-12
        ]
        raise AssertionError(f"{dataset}: constant/invalid active markers: {bad}")
    processed = np.asarray(
        (asinh_values - means) / standard_deviations,
        dtype=np.float64,
        order="C",
    )
    if not np.isfinite(processed).all():
        raise AssertionError(f"{dataset}: processed active matrix is non-finite")
    post_means = processed.mean(axis=0)
    post_sds = processed.std(axis=0, ddof=ZSCORE_DDOF)
    if float(np.max(np.abs(post_means))) > 5e-12:
        raise AssertionError(f"{dataset}: processed marker means are not zero")
    if float(np.max(np.abs(post_sds - 1.0))) > 5e-12:
        raise AssertionError(f"{dataset}: processed marker SDs are not one")
    parameters: list[dict[str, Any]] = []
    for index, marker in enumerate(markers):
        parameters.append({
            "dataset": dataset,
            "slug": DATASET_SLUGS[dataset],
            "active_index": index,
            "marker": marker,
            "source_marker": source_by_marker[marker],
            "cofactor": COFACTOR,
            "nonlinear_transform": "asinh(x / cofactor)",
            "zscore_scope": "within_dataset_per_marker",
            "zscore_ddof": ZSCORE_DDOF,
            "asinh_mean": float(means[index]),
            "asinh_sample_sd": float(standard_deviations[index]),
            "raw_minimum": float(raw[:, index].min()),
            "raw_maximum": float(raw[:, index].max()),
            "processed_mean": float(post_means[index]),
            "processed_sample_sd": float(post_sds[index]),
            "processed_column_buffer_sha256_float64": buffer_sha256(
                processed[:, index]
            ),
        })
    return processed, parameters


def verify_existing_output(source_records: dict[str, dict[str, str]]) -> bool:
    manifest_path = OUTPUT / "input_manifest.json"
    if not OUTPUT.exists():
        return False
    if not manifest_path.is_file():
        print(f"INPUTS_STALE_REBUILD: manifest missing from {OUTPUT}", flush=True)
        return False
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != SCHEMA or manifest.get("status") != "complete":
        print(
            f"INPUTS_STALE_REBUILD: schema/status changed in {OUTPUT}",
            flush=True,
        )
        return False
    current_builder_hash = sha256(Path(__file__).resolve())
    if manifest.get("builder_sha256") != current_builder_hash:
        print(
            f"INPUTS_STALE_REBUILD: builder revision changed in {OUTPUT}",
            flush=True,
        )
        return False
    if manifest.get("source_files") != source_records:
        print(
            f"INPUTS_STALE_REBUILD: source contract changed in {OUTPUT}",
            flush=True,
        )
        return False
    for relative, expected_hash in manifest.get("output_sha256", {}).items():
        path = OUTPUT / relative
        if not path.is_file() or sha256(path) != expected_hash:
            print(
                f"INPUTS_STALE_REBUILD: generated file failed validation: {path}",
                flush=True,
            )
            return False
    print(f"INPUTS_ALREADY_VALID: {OUTPUT}", flush=True)
    return True


def install_staged_output(stage: Path) -> None:
    """Install a complete derived input tree, preserving rollback on failure."""

    backup = HERE / f".input.previous-{os.getpid()}"
    if backup.exists():
        raise RuntimeError(f"Unexpected stale input backup: {backup}")
    had_existing = OUTPUT.exists()
    if had_existing:
        os.replace(OUTPUT, backup)
    try:
        os.replace(stage, OUTPUT)
    except BaseException:
        if had_existing and backup.exists() and not OUTPUT.exists():
            os.replace(backup, OUTPUT)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def main() -> None:
    if OUTPUT.resolve().parent != HERE.resolve():
        raise AssertionError(f"Output confinement failed: {OUTPUT}")

    metadata, source_records = load_canonical_lock()
    source_records.update({
        "provider_extraction_recipe": {
            "path": str(EXTRACTION_RECIPE.resolve()),
            "sha256": sha256(EXTRACTION_RECIPE),
        },
        "tietscher_provider_sce": verify_expected_hash(
            TIETSCHER_RDS, "tietscher_rds"
        ),
        "meyer_provider_sce": verify_expected_hash(MEYER_RDS, "meyer_rds"),
        "shiao_provider_full_panel_counts": verify_expected_hash(
            SHIAO_COUNTS, "shiao_counts"
        ),
    })
    if verify_existing_output(source_records):
        return

    temporary = Path(tempfile.mkdtemp(prefix=".input.tmp-", dir=HERE))
    provider_temporary = Path(
        tempfile.mkdtemp(prefix=".provider-rows.tmp-", dir=HERE)
    )
    try:
        provider_exports = extract_provider_rows(provider_temporary)
        active_raw: dict[str, np.ndarray] = {}
        active_markers: dict[str, list[str]] = {}
        marker_audit_rows: list[dict[str, Any]] = []
        provider_audits: dict[str, dict[str, Any]] = {}

        for dataset in ("Tietscher IMC", "Meyer IMC"):
            selected = metadata.loc[metadata["dataset"].astype(str).eq(dataset)].copy()
            matrix, markers, audit, provider_audit = load_h5_dataset(
                dataset, provider_exports[dataset], selected
            )
            active_raw[dataset] = matrix
            active_markers[dataset] = markers
            marker_audit_rows.extend(audit)
            provider_audits[DATASET_SLUGS[dataset]] = provider_audit

        shiao_metadata = metadata.loc[
            metadata["dataset"].astype(str).eq("Shiao CODEX")
        ].copy()
        matrix, markers, audit, provider_audit = load_shiao_dataset(shiao_metadata)
        active_raw["Shiao CODEX"] = matrix
        active_markers["Shiao CODEX"] = markers
        marker_audit_rows.extend(audit)
        provider_audits["shiao_codex"] = provider_audit

        processed_blocks: dict[str, np.ndarray] = {}
        shared_blocks: list[np.ndarray] = []
        transform_rows: list[dict[str, Any]] = []
        row_lock_rows: list[dict[str, Any]] = []
        dataset_manifest: dict[str, dict[str, Any]] = {}

        for dataset in DATASETS:
            slug = DATASET_SLUGS[dataset]
            selected_mask = metadata["dataset"].astype(str).eq(dataset).to_numpy()
            selected = metadata.loc[selected_mask].copy()
            markers = active_markers[dataset]
            raw = active_raw[dataset]
            marker_to_index = {marker: index for index, marker in enumerate(markers)}
            missing_shared = [
                marker for marker in SHARED_MARKERS if marker not in marker_to_index
            ]
            if missing_shared:
                raise AssertionError(f"{dataset}: active panel lacks {missing_shared}")
            shared_indices = [marker_to_index[marker] for marker in SHARED_MARKERS]
            source_by_marker = {
                str(row["canonical_marker"]): str(row["source_marker"])
                for row in marker_audit_rows
                if row["dataset"] == dataset and bool(row["included_in_active"])
            }
            processed, parameters = transform_dataset(
                dataset, raw, markers, source_by_marker
            )
            shared = np.asarray(processed[:, shared_indices], dtype=np.float64, order="C")
            if not np.array_equal(shared, processed[:, shared_indices]):
                raise AssertionError(f"{dataset}: shared columns are not direct active slices")
            processed_blocks[dataset] = processed
            shared_blocks.append(shared)
            transform_rows.extend(parameters)

            dataset_dir = temporary / slug
            dataset_dir.mkdir()
            active_frame = pd.DataFrame(processed, columns=markers)
            active_frame.insert(0, "cell_id", selected["cell_id"].astype(str).to_numpy())
            shared_frame = selected[["cell_id", "dataset", "donor", "unit_key"]].copy()
            for index, marker in enumerate(SHARED_MARKERS):
                shared_frame[marker] = shared[:, index]
            active_path = dataset_dir / "active_processed.csv.gz"
            shared_path = dataset_dir / "shared_processed.csv.gz"
            deterministic_csv(active_frame, active_path, compressed=True)
            deterministic_csv(shared_frame, shared_path, compressed=True)

            active_roundtrip = pd.read_csv(
                active_path,
                dtype={"cell_id": str},
                float_precision="round_trip",
                low_memory=False,
            )
            shared_roundtrip = pd.read_csv(
                shared_path,
                dtype={"cell_id": str},
                float_precision="round_trip",
                low_memory=False,
            )
            expected_ids = selected["cell_id"].astype(str).to_numpy()
            if not np.array_equal(active_roundtrip["cell_id"].to_numpy(), expected_ids):
                raise AssertionError(f"{dataset}: active CSV row lock failed")
            if not np.array_equal(shared_roundtrip["cell_id"].to_numpy(), expected_ids):
                raise AssertionError(f"{dataset}: shared CSV row lock failed")
            active_values_roundtrip = active_roundtrip[markers].to_numpy(np.float64)
            shared_values_roundtrip = shared_roundtrip[SHARED_MARKERS].to_numpy(
                np.float64
            )
            if not np.array_equal(active_values_roundtrip, processed):
                raise AssertionError(f"{dataset}: active CSV float64 round trip changed")
            if not np.array_equal(shared_values_roundtrip, shared):
                raise AssertionError(f"{dataset}: shared CSV float64 round trip changed")
            if not np.array_equal(
                shared_values_roundtrip,
                active_values_roundtrip[:, shared_indices],
            ):
                raise AssertionError(f"{dataset}: persisted shared/active identity failed")

            cell_id_hash = string_rows_sha256(expected_ids)
            source_id_hash = string_rows_sha256(selected["source_cell_id"].astype(str))
            row_lock_rows.append({
                "dataset": dataset,
                "slug": slug,
                "cells": len(selected),
                "canonical_global_row_start": int(selected["cell_order"].min()),
                "canonical_global_row_stop_exclusive": int(selected["cell_order"].max()) + 1,
                "first_cell_id": str(expected_ids[0]),
                "last_cell_id": str(expected_ids[-1]),
                "cell_id_rows_sha256": cell_id_hash,
                "source_cell_id_rows_sha256": source_id_hash,
                "provider_ids_match_frozen_order": True,
                "active_shared_cell_ids_identical": True,
                "shared_direct_active_column_max_abs_difference": 0.0,
                "canonical_cell_lock_used_for_values": False,
            })
            dataset_manifest[slug] = {
                "dataset": dataset,
                "cells": len(selected),
                "active_shape": [int(processed.shape[0]), int(processed.shape[1])],
                "shared_shape": [int(shared.shape[0]), int(shared.shape[1])],
                "active_markers": markers,
                "shared_markers": SHARED_MARKERS,
                "shared_active_column_indices_zero_based": shared_indices,
                "active_buffer_sha256_float64": buffer_sha256(processed),
                "shared_buffer_sha256_float64": buffer_sha256(shared),
                "cell_id_rows_sha256": cell_id_hash,
                "canonical_cell_lock_used_for_values": False,
                "provider_audit": provider_audits[slug],
            }

        combined_shared = np.asarray(
            np.vstack(shared_blocks), dtype=np.float64, order="C"
        )
        if combined_shared.shape != (62_701, 11):
            raise AssertionError(f"Combined processed shared shape changed: {combined_shared.shape}")
        np.save(temporary / "shared_processed_float64.npy", combined_shared, allow_pickle=False)
        (temporary / "shared_markers.txt").write_text(
            "".join(f"{marker}\n" for marker in SHARED_MARKERS)
        )
        deterministic_csv(
            pd.DataFrame(transform_rows),
            temporary / "transform_parameters.csv",
            compressed=False,
        )
        deterministic_csv(
            pd.DataFrame(marker_audit_rows),
            temporary / "marker_source_audit.csv",
            compressed=False,
        )
        deterministic_csv(
            pd.DataFrame(row_lock_rows),
            temporary / "row_lock_audit.csv",
            compressed=False,
        )

        output_paths = sorted(
            path for path in temporary.rglob("*") if path.is_file()
        )
        output_hashes = {
            str(path.relative_to(temporary)): sha256(path) for path in output_paths
        }
        manifest = {
            "schema_version": SCHEMA,
            "status": "complete",
            "builder": str(Path(__file__).resolve()),
            "builder_sha256": sha256(Path(__file__).resolve()),
            "writes_confined_to": str(OUTPUT.resolve()),
            "cells": 62_701,
            "datasets_in_frozen_row_order": DATASETS,
            "shared_markers": SHARED_MARKERS,
            "source_files": source_records,
            "input_protocol": {
                "active_panel": "provider full biological panel for frozen cells",
                "provider_extraction": (
                    "local extract_provider_rows.R reads raw_data provider RDS counts "
                    "assays at canonical benchmark source indices; temporary raw-row "
                    "exports are deleted after the processed inputs are built"
                ),
                "historical_alignment_or_cache_used": False,
                "metadata_columns_loaded_for_numerical_build": (
                    CANONICAL_IDENTITY_COLUMNS
                ),
                "nonlinear_transform": "asinh(x / 1.0)",
                "zscore": (
                    "within each dataset independently, per marker, sample SD "
                    "(ddof=1)"
                ),
                "zscore_ddof": ZSCORE_DDOF,
                "zscore_reference": (
                    "scanpy 1.12.4 sc.pp.scale/scale_array mean_var(correction=1), "
                    "matching the official MaxFuse notebook convention"
                ),
                "shared_construction": (
                    "direct same-row column slice from the corresponding processed "
                    "active matrix"
                ),
                "shiao_compartment_rule": (
                    "canonical v1 arithmetic mean of nuclei_mean and membrane_mean"
                ),
                "maxfuse_only_compartment_override": False,
            },
            "datasets": dataset_manifest,
            "combined_shared_shape": [62_701, 11],
            "combined_shared_buffer_sha256_float64": buffer_sha256(combined_shared),
            "audit_files": [
                "transform_parameters.csv",
                "marker_source_audit.csv",
                "row_lock_audit.csv",
            ],
            "output_sha256": output_hashes,
            "software": {
                "python": __import__("platform").python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "h5py": h5py.__version__,
            },
        }
        (temporary / "input_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        install_staged_output(temporary)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        # These are untransformed selected provider rows.  They are an
        # implementation bridge only, never a persistent copy of raw data.
        shutil.rmtree(provider_temporary, ignore_errors=True)

    widths = {
        DATASET_SLUGS[dataset]: len(active_markers[dataset]) for dataset in DATASETS
    }
    print(
        f"INPUTS_BUILT: shared={combined_shared.shape}; active_widths={widths}; "
        f"shared_buffer_sha256_float64={buffer_sha256(combined_shared)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
