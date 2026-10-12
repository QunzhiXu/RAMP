#!/usr/bin/env python3
"""Create the frozen donor-balanced 140k Senosain CyTOF sample.

The script reads the published CyTOF patient index from Supplementary Table
S3, samples 2,000 events without replacement from each of the 70 patients,
and preserves all 77 FCS parameters exactly as stored in the provider's
normalized/denoised files.  No cell annotation, clinical outcome, SILA value,
or marker transformation is used in sampling.

Patient 13376 has two acquisition files.  They are treated as one pooled
patient-level event set, and only 2,000 events total are sampled from that
pool.  Patient 11840 is absent from the published clinical/assay index and is
therefore excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DATASET_ROOT = HERE.parent
FCS_DIR = DATASET_ROOT / "CyTOF"
WORKBOOK = DATASET_ROOT / "supplementary_tables.xlsx"

DEFAULT_SEED = 65
DEFAULT_CELLS_PER_PATIENT = 2_000
NESTED_SENSITIVITY_SIZE = 1_000

FILENAME_RE = re.compile(
    r"^(?P<date>\d{6})_(?P<run>\d+)_(?P<patient>\d+)_"
    r"(?P<legacy_label>[A-Za-z]+)_normalized_denoised\.FCS$"
)

OUTPUTS = {
    "matrix": HERE / "senosain_140k_provider_fcs77.npy",
    "biological35": HERE / "senosain_140k_raw35.npy",
    "metadata": HERE / "senosain_140k_cell_metadata.csv.gz",
    "channels": HERE / "senosain_140k_channels.csv",
    "markers35": HERE / "senosain_140k_markers35.csv",
    "patient_audit": HERE / "senosain_140k_patient_sampling_audit.csv",
    "source_audit": HERE / "senosain_140k_source_fcs_audit.csv",
    "manifest": HERE / "senosain_140k_manifest.json",
}

# The frozen 35-biological-marker order used by the existing lung story.
# Each entry is (FCS 1-based parameter index, channel, FCS description,
# canonical output marker). Exact channel and description assertions prevent a
# silent column shift if a different FCS release is substituted later.
BIOLOGICAL35 = [
    (15, "Pr141Di", "141Pr_EpCAM", "EPCAM"),
    (17, "Nd142Di", "142Nd_ccasp3", "CCASP3"),
    (18, "Nd143Di", "143Nd_TP53", "TP53"),
    (19, "Nd144Di", "144Nd_HLA-ABC", "HLA_ABC"),
    (20, "Nd145Di", "145Nd_CD31", "CD31"),
    (21, "Nd146Di", "146Nd_Thioredoxin", "THIOREDOXIN"),
    (22, "Sm147Di", "147Sm_b-CAT", "B_CATENIN"),
    (23, "Nd148Di", "148Nd_HER2", "HER2"),
    (24, "Sm149Di", "149Sm_p-STAT6", "P_STAT6"),
    (25, "Nd150Di", "150Nd_p-STAT5", "P_STAT5"),
    (26, "Eu151Di", "151Eu_TTF1", "TTF1"),
    (27, "Sm152Di", "152Sm_p-AKT", "P_AKT"),
    (28, "Eu153Di", "153Eu_Ki67", "KI67"),
    (29, "Sm154Di", "154Sm_CD45", "CD45"),
    (30, "Gd155Di", "155Gd_CD56", "CD56"),
    (31, "Gd156Di", "156Gd_Vimentin", "VIMENTIN"),
    (33, "Gd158Di", "158Gd_p-STAT3", "P_STAT3"),
    (34, "Tb159Di", "159Tb_CD4", "CD4"),
    (35, "Gd160Di", "160Gd_MDM2", "MDM2"),
    (37, "Dy161Di", "161Dy_Cytokeratin", "CYTOKERATIN"),
    (38, "Dy162Di", "162Dy_MET", "MET"),
    (39, "Dy163Di", "163Dy_TP63", "TP63"),
    (40, "Dy164Di", "164Dy_CK7", "CK7"),
    (41, "Ho165Di", "165Ho_EGFR", "EGFR"),
    (42, "Er166Di", "166Er_CD44", "CD44"),
    (43, "Er167Di", "167Er_p-ERK", "P_ERK"),
    (44, "Er168Di", "168Er_CD8", "CD8"),
    (45, "Tm169Di", "169Tm_CD24", "CD24"),
    (46, "Yb170Di", "170Yb_CD3", "CD3"),
    (47, "Yb171Di", "171Yb_CD11b", "CD11B"),
    (48, "Yb172Di", "172Yb_p-S6", "P_S6"),
    (49, "Yb173Di", "173Yb_CD90", "CD90"),
    (50, "Yb174Di", "174Yb_HLA-DR", "HLA_DR"),
    (51, "Lu175Di", "175Lu_PD-L1", "PD_L1"),
    (52, "Yb176Di", "176Yb_His-H3", "HISTONE_H3"),
]


@dataclass(frozen=True)
class FCSDescriptor:
    path: Path
    patient_id: str
    acquisition_date: str
    run_index: int
    legacy_filename_label: str
    version: str
    n_events: int
    n_parameters: int
    data_start: int
    data_end: int
    dtype: np.dtype
    text: dict[str, str]
    sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def header_integer(header: bytes, start: int, end: int) -> int:
    token = header[start:end].decode("ascii", errors="ignore").strip()
    return int(token) if token else 0


def parse_text_segment(segment: bytes) -> dict[str, str]:
    if not segment:
        raise ValueError("Empty FCS TEXT segment")
    delimiter = chr(segment[0])
    text = segment[1:].decode("latin1", errors="replace")
    tokens: list[str] = []
    current: list[str] = []
    index = 0
    while index < len(text):
        if text[index] == delimiter:
            if index + 1 < len(text) and text[index + 1] == delimiter:
                current.append(delimiter)
                index += 2
                continue
            tokens.append("".join(current))
            current = []
            index += 1
            continue
        current.append(text[index])
        index += 1
    if current:
        tokens.append("".join(current))
    if len(tokens) % 2:
        tokens = tokens[:-1]
    return dict(zip(tokens[0::2], tokens[1::2]))


def inspect_fcs(path: Path) -> FCSDescriptor:
    match = FILENAME_RE.match(path.name)
    if match is None:
        raise ValueError(f"Unexpected FCS filename: {path.name}")
    with path.open("rb") as handle:
        header = handle.read(58)
        version = header[:6].decode("ascii", errors="replace")
        if not version.startswith("FCS3."):
            raise ValueError(f"{path.name}: expected FCS3.x, found {version!r}")
        text_start = header_integer(header, 10, 18)
        text_end = header_integer(header, 18, 26)
        data_start = header_integer(header, 26, 34)
        data_end = header_integer(header, 34, 42)
        handle.seek(text_start)
        text = parse_text_segment(handle.read(text_end - text_start + 1))

    if data_start == 0:
        data_start = int(text["$BEGINDATA"])
        data_end = int(text["$ENDDATA"])
    n_events = int(text["$TOT"])
    n_parameters = int(text["$PAR"])
    if text.get("$DATATYPE") != "F":
        raise ValueError(f"{path.name}: only 32-bit floating-point FCS is supported")
    byte_order = text.get("$BYTEORD", "")
    if byte_order == "4,3,2,1":
        dtype = np.dtype(">f4")
    elif byte_order == "1,2,3,4":
        dtype = np.dtype("<f4")
    else:
        raise ValueError(f"{path.name}: unsupported byte order {byte_order!r}")
    bits = [int(text[f"$P{i}B"]) for i in range(1, n_parameters + 1)]
    if set(bits) != {32}:
        raise ValueError(f"{path.name}: expected all parameters to use 32 bits")
    expected_bytes = n_events * n_parameters * 4
    observed_bytes = data_end - data_start + 1
    if observed_bytes != expected_bytes:
        raise ValueError(
            f"{path.name}: DATA segment has {observed_bytes:,} bytes; "
            f"expected {expected_bytes:,}"
        )

    acquisition_date = datetime.strptime(match.group("date"), "%y%m%d").date().isoformat()
    return FCSDescriptor(
        path=path,
        patient_id=match.group("patient"),
        acquisition_date=acquisition_date,
        run_index=int(match.group("run")),
        legacy_filename_label=match.group("legacy_label"),
        version=version,
        n_events=n_events,
        n_parameters=n_parameters,
        data_start=data_start,
        data_end=data_end,
        dtype=dtype,
        text=text,
        sha256=sha256_file(path),
    )


def parameter_signature(descriptor: FCSDescriptor) -> tuple[tuple[str, str], ...]:
    return tuple(
        (
            descriptor.text.get(f"$P{i}N", "").strip(),
            descriptor.text.get(f"$P{i}S", "").strip(),
        )
        for i in range(1, descriptor.n_parameters + 1)
    )


def published_cytof_patient_ids() -> list[str]:
    table = pd.read_excel(
        WORKBOOK,
        sheet_name="Table S3",
        header=2,
        dtype={"Pt ID": str},
    )
    required = {"Pt ID", "CyTOF"}
    if not required.issubset(table.columns):
        raise ValueError(f"Table S3 is missing columns: {sorted(required - set(table.columns))}")
    cytof = pd.to_numeric(table["CyTOF"], errors="coerce").eq(1)
    patient_ids = table.loc[cytof, "Pt ID"].dropna().astype(str).str.strip().tolist()
    if len(patient_ids) != 70 or len(set(patient_ids)) != 70:
        raise ValueError(
            f"Expected 70 unique Table-S3 CyTOF patients; found "
            f"{len(patient_ids)} rows and {len(set(patient_ids))} unique IDs"
        )
    return sorted(patient_ids, key=int)


def channel_table(reference: FCSDescriptor) -> pd.DataFrame:
    rows = []
    for index in range(1, reference.n_parameters + 1):
        rows.append(
            {
                "matrix_column_zero_based": index - 1,
                "fcs_parameter_one_based": index,
                "channel": reference.text.get(f"$P{index}N", "").strip(),
                "marker": reference.text.get(f"$P{index}S", "").strip(),
                "bits": int(reference.text[f"$P{index}B"]),
                "range": reference.text.get(f"$P{index}R", ""),
                "exponent": reference.text.get(f"$P{index}E", ""),
            }
        )
    return pd.DataFrame(rows)


def biological35_table(channels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for output_index, (parameter, expected_channel, expected_description, marker) in enumerate(
        BIOLOGICAL35
    ):
        source = channels.iloc[parameter - 1]
        observed = (str(source.channel), str(source.marker))
        expected = (expected_channel, expected_description)
        if observed != expected:
            raise ValueError(
                f"FCS parameter {parameter} differs from frozen 35-marker map: "
                f"observed {observed}, expected {expected}"
            )
        rows.append(
            {
                "raw35_column_zero_based": output_index,
                "marker": marker,
                "fcs77_column_zero_based": parameter - 1,
                "fcs_parameter_one_based": parameter,
                "channel": expected_channel,
                "fcs_description": expected_description,
            }
        )
    return pd.DataFrame(rows)


def sample_patient(
    descriptors: list[FCSDescriptor],
    rng: np.random.Generator,
    cells_per_patient: int,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, int]]:
    descriptors = sorted(
        descriptors,
        key=lambda item: (item.acquisition_date, item.run_index, item.path.name),
    )
    total_events = sum(item.n_events for item in descriptors)
    if total_events < cells_per_patient:
        raise ValueError(
            f"Patient {descriptors[0].patient_id} has {total_events:,} events, "
            f"fewer than requested {cells_per_patient:,}"
        )

    # Generator.choice(..., replace=False, shuffle=True) returns an ordered
    # simple random sample.  Its first 1,000 rows form the nested 1k subset.
    chosen_global = rng.choice(
        total_events,
        size=cells_per_patient,
        replace=False,
        shuffle=True,
    )
    n_parameters = descriptors[0].n_parameters
    values = np.empty((cells_per_patient, n_parameters), dtype=np.float32)
    source_file = np.empty(cells_per_patient, dtype=object)
    source_row = np.empty(cells_per_patient, dtype=np.int64)
    source_total = np.empty(cells_per_patient, dtype=np.int64)
    acquisition_date = np.empty(cells_per_patient, dtype=object)
    run_index = np.empty(cells_per_patient, dtype=np.int64)
    legacy_label = np.empty(cells_per_patient, dtype=object)
    file_index = np.empty(cells_per_patient, dtype=np.int64)
    sampled_by_file: dict[str, int] = {}

    offset = 0
    for index, descriptor in enumerate(descriptors, start=1):
        positions = np.flatnonzero(
            (chosen_global >= offset) & (chosen_global < offset + descriptor.n_events)
        )
        local_rows = chosen_global[positions] - offset
        event_matrix = np.memmap(
            descriptor.path,
            dtype=descriptor.dtype,
            mode="r",
            offset=descriptor.data_start,
            shape=(descriptor.n_events, descriptor.n_parameters),
            order="C",
        )
        values[positions] = np.asarray(event_matrix[local_rows], dtype=np.float32)
        del event_matrix
        source_file[positions] = descriptor.path.name
        source_row[positions] = local_rows
        source_total[positions] = descriptor.n_events
        acquisition_date[positions] = descriptor.acquisition_date
        run_index[positions] = descriptor.run_index
        legacy_label[positions] = descriptor.legacy_filename_label
        file_index[positions] = index
        sampled_by_file[descriptor.path.name] = int(len(positions))
        offset += descriptor.n_events

    patient_id = descriptors[0].patient_id
    ranks = np.arange(1, cells_per_patient + 1, dtype=np.int64)
    metadata = pd.DataFrame(
        {
            "patient_id": patient_id,
            "dataset": "Senosain CyTOF",
            "unit_key": f"Senosain|{patient_id}",
            "sample_rank_within_patient": ranks,
            "nested_1000": ranks <= min(NESTED_SENSITIVITY_SIZE, cells_per_patient),
            "source_file": source_file,
            "source_file_index_within_patient": file_index,
            "source_row_zero_based": source_row,
            "source_total_events": source_total,
            "acquisition_date": acquisition_date,
            "run_index": run_index,
            "legacy_filename_label": legacy_label,
            "patient_source_file_count": len(descriptors),
        }
    )
    metadata.insert(
        0,
        "source_cell_id",
        [
            f"Senosain:{patient_id}:{name}:row{row}"
            for name, row in zip(source_file, source_row, strict=True)
        ],
    )
    return values, metadata, sampled_by_file


def refuse_overwrite(force: bool) -> None:
    existing = [path for path in OUTPUTS.values() if path.exists()]
    if existing and not force:
        rendered = "\n".join(f"  - {path.name}" for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing derived outputs without --force:\n" + rendered
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    refuse_overwrite(args.force)
    HERE.mkdir(parents=True, exist_ok=True)

    print("Reading the published Table-S3 CyTOF cohort...", flush=True)
    official_ids = published_cytof_patient_ids()
    official_set = set(official_ids)

    fcs_paths = sorted(FCS_DIR.glob("*.FCS"), key=lambda path: path.name)
    if len(fcs_paths) != 72:
        raise ValueError(f"Expected 72 local FCS files; found {len(fcs_paths)}")
    print("Inspecting and hashing 72 source FCS files...", flush=True)
    descriptors = [inspect_fcs(path) for path in fcs_paths]

    by_patient: dict[str, list[FCSDescriptor]] = {}
    for descriptor in descriptors:
        by_patient.setdefault(descriptor.patient_id, []).append(descriptor)
    missing = sorted(official_set - set(by_patient), key=int)
    if missing:
        raise ValueError(f"Published CyTOF patients missing FCS files: {missing}")
    extra = sorted(set(by_patient) - official_set, key=int)
    if extra != ["11840"]:
        raise ValueError(f"Expected only metadata-unmatched patient 11840; found {extra}")
    duplicate_ids = sorted(
        [patient for patient in official_ids if len(by_patient[patient]) > 1], key=int
    )
    if duplicate_ids != ["13376"]:
        raise ValueError(f"Expected only duplicated patient 13376; found {duplicate_ids}")

    included = [item for item in descriptors if item.patient_id in official_set]
    reference = included[0]
    signature = parameter_signature(reference)
    for descriptor in descriptors:
        if descriptor.n_parameters != reference.n_parameters:
            raise ValueError(f"{descriptor.path.name}: parameter count differs")
        if parameter_signature(descriptor) != signature:
            raise ValueError(f"{descriptor.path.name}: channel/marker signature differs")

    n_patients = len(official_ids)
    n_cells = n_patients * DEFAULT_CELLS_PER_PATIENT
    sampled_matrix = np.empty(
        (n_cells, reference.n_parameters), dtype=np.float32
    )
    metadata_parts: list[pd.DataFrame] = []
    patient_audit_rows: list[dict[str, object]] = []
    sampled_file_counts: dict[str, int] = {item.path.name: 0 for item in descriptors}
    rng = np.random.default_rng(args.seed)

    cursor = 0
    for position, patient_id in enumerate(official_ids, start=1):
        patient_descriptors = by_patient[patient_id]
        values, metadata, by_file = sample_patient(
            patient_descriptors, rng, DEFAULT_CELLS_PER_PATIENT
        )
        stop = cursor + DEFAULT_CELLS_PER_PATIENT
        sampled_matrix[cursor:stop] = values
        metadata.insert(0, "sampled_row_zero_based", np.arange(cursor, stop))
        metadata_parts.append(metadata)
        for name, count in by_file.items():
            sampled_file_counts[name] = count
        ordered = sorted(
            patient_descriptors,
            key=lambda item: (item.acquisition_date, item.run_index, item.path.name),
        )
        patient_audit_rows.append(
            {
                "patient_id": patient_id,
                "patient_order": position,
                "source_file_count": len(ordered),
                "source_files": "|".join(item.path.name for item in ordered),
                "source_event_counts": "|".join(str(item.n_events) for item in ordered),
                "available_events_pooled": sum(item.n_events for item in ordered),
                "sampled_cells_total": DEFAULT_CELLS_PER_PATIENT,
                "sampled_cells_by_file": "|".join(
                    str(by_file[item.path.name]) for item in ordered
                ),
                "sampling_replacement": False,
            }
        )
        cursor = stop
        if position == 1 or position % 10 == 0 or position == n_patients:
            print(f"  sampled {position:>2}/{n_patients} patients", flush=True)

    cell_metadata = pd.concat(metadata_parts, ignore_index=True)
    patient_audit = pd.DataFrame(patient_audit_rows)
    source_audit = pd.DataFrame(
        [
            {
                "source_file": item.path.name,
                "patient_id": item.patient_id,
                "official_table_s3_cytof_patient": item.patient_id in official_set,
                "included_in_140k": item.patient_id in official_set,
                "inclusion_reason": (
                    "published_Table_S3_CyTOF_patient"
                    if item.patient_id in official_set
                    else "excluded_metadata_unmatched_11840"
                ),
                "acquisition_date": item.acquisition_date,
                "run_index": item.run_index,
                "legacy_filename_label": item.legacy_filename_label,
                "fcs_version": item.version,
                "n_events": item.n_events,
                "n_parameters": item.n_parameters,
                "sampled_cells": sampled_file_counts[item.path.name],
                "sha256": item.sha256,
            }
            for item in descriptors
        ]
    ).sort_values(["patient_id", "acquisition_date", "run_index", "source_file"])
    channels = channel_table(reference)
    markers35 = biological35_table(channels)
    biological_indices = markers35["fcs77_column_zero_based"].to_numpy(dtype=int)
    sampled_raw35 = sampled_matrix[:, biological_indices].copy()

    if sampled_matrix.shape != (n_cells, reference.n_parameters):
        raise AssertionError("Unexpected sampled matrix shape")
    if len(cell_metadata) != n_cells:
        raise AssertionError("Matrix and metadata row counts differ")
    counts = cell_metadata.groupby("patient_id", sort=False).size()
    if len(counts) != 70 or not counts.eq(DEFAULT_CELLS_PER_PATIENT).all():
        raise AssertionError("Patient-level sampled cell counts are not exactly balanced")
    if cell_metadata["source_cell_id"].duplicated().any():
        raise AssertionError("A source event was selected more than once")
    if not np.isfinite(sampled_matrix).all():
        raise AssertionError("Sampled matrix contains NaN or infinite values")
    if sampled_raw35.shape != (n_cells, 35):
        raise AssertionError("Unexpected 35-marker matrix shape")
    if not np.isfinite(sampled_raw35).all() or np.any(sampled_raw35 < 0):
        raise AssertionError("35-marker matrix contains invalid values")
    patient_13376 = patient_audit.loc[patient_audit.patient_id.eq("13376")]
    if len(patient_13376) != 1 or int(patient_13376.iloc[0].source_file_count) != 2:
        raise AssertionError("Patient 13376 was not pooled from exactly two files")

    print("Writing derived outputs...", flush=True)
    np.save(OUTPUTS["matrix"], sampled_matrix, allow_pickle=False)
    np.save(OUTPUTS["biological35"], sampled_raw35, allow_pickle=False)
    cell_metadata.to_csv(
        OUTPUTS["metadata"],
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )
    channels.to_csv(OUTPUTS["channels"], index=False)
    markers35.to_csv(OUTPUTS["markers35"], index=False)
    patient_audit.to_csv(OUTPUTS["patient_audit"], index=False)
    source_audit.to_csv(OUTPUTS["source_audit"], index=False)

    output_hashes = {
        key: sha256_file(path)
        for key, path in OUTPUTS.items()
        if key != "manifest"
    }
    manifest = {
        "schema_version": "senosain-cytof-donor-balanced-sample-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "Senosain CyTOF",
        "source_state": "provider_normalized_denoised_FCS_not_instrument_raw",
        "sampling": {
            "seed": args.seed,
            "rng": "numpy.random.Generator(PCG64)",
            "patient_order": official_ids,
            "cells_per_patient": DEFAULT_CELLS_PER_PATIENT,
            "replacement": False,
            "nested_1000_definition": "sample_rank_within_patient <= 1000",
            "patient_13376": (
                "two acquisition files pooled as one patient; 2000 total sampled "
                "uniformly over pooled events"
            ),
        },
        "cohort": {
            "source": "supplementary_tables.xlsx, Table S3, CyTOF == 1",
            "n_patients": n_patients,
            "n_included_fcs_files": len(included),
            "excluded_patient_ids": ["11840"],
            "excluded_reason": "absent from published Table S2/Table S3 patient index",
        },
        "matrices": {
            "provider_fcs77": {
                "rows": int(sampled_matrix.shape[0]),
                "columns": int(sampled_matrix.shape[1]),
                "dtype": str(sampled_matrix.dtype),
                "all_values_finite": True,
                "minimum": float(sampled_matrix.min()),
                "maximum": float(sampled_matrix.max()),
                "values_transformed_by_sampling_script": False,
            },
            "biological_raw35": {
                "rows": int(sampled_raw35.shape[0]),
                "columns": int(sampled_raw35.shape[1]),
                "dtype": str(sampled_raw35.dtype),
                "all_values_finite": True,
                "all_values_nonnegative": True,
                "minimum": float(sampled_raw35.min()),
                "maximum": float(sampled_raw35.max()),
                "values_transformed_by_sampling_script": False,
                "marker_order": markers35["marker"].tolist(),
            },
        },
        "inputs": {
            "workbook": str(WORKBOOK),
            "workbook_sha256": sha256_file(WORKBOOK),
            "fcs_directory": str(FCS_DIR),
            "source_fcs_hashes": OUTPUTS["source_audit"].name,
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "outputs": {
            key: {"path": path.name, "sha256": output_hashes[key]}
            for key, path in OUTPUTS.items()
            if key != "manifest"
        },
        "annotation_present": False,
        "clinical_outcomes_or_sila_used": False,
    }
    OUTPUTS["manifest"].write_text(json.dumps(manifest, indent=2) + "\n")

    print(
        f"Complete: {n_patients} patients x {DEFAULT_CELLS_PER_PATIENT:,} cells "
        f"= {n_cells:,} rows; {reference.n_parameters} FCS parameters.",
        flush=True,
    )
    print(f"Output directory: {HERE}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
