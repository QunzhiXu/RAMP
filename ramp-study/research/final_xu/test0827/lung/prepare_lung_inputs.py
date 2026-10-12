#!/usr/bin/env python3
"""Freeze the audited 36- and 10-donor lung alignment inputs in test0827.

This script does not resample the primary archives and does not run RAMP.  It
normalizes the two previously audited, row-frozen lung inputs to the common
``test0827`` interface while checking every upstream file by SHA256.  Labels
are post-hoc metadata only; RAMP is allowed to read only ``dataset`` and
``unit_key``.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
FINAL_XU = HERE.parents[1]
SOURCE_ROOT = FINAL_XU / "story_lung_caf_bidirectional"

DATASETS = ["Senosain CyTOF", "Cords IMC", "Monkman CODEX"]
MARKERS = ["CD3", "CD4", "CD8", "CD31", "HLA_DR", "KI67", "VIMENTIN", "EPI"]
LABELS = [
    "Tumor/Epithelial",
    "CD4 T",
    "CD8 T",
    "B-like",
    "Myeloid",
    "Endothelial",
    "Mix",
]
DISPLAY_PALETTE = {
    "Tumor/Epithelial": "#D73027",
    "CD4 T": "#E78AC3",
    "CD8 T": "#D95F02",
    "B-like": "#1F78B4",
    "Myeloid": "#E6AB02",
    "Endothelial": "#17BECF",
    "Mix": "#9CA3AF",
}
DISPLAY_DRAW_ORDER = [
    "Tumor/Epithelial",
    "Mix",
    "B-like",
    "CD4 T",
    "CD8 T",
    "Endothelial",
    "Myeloid",
]


@dataclass(frozen=True)
class Case:
    slug: str
    source_slug: str
    case_name: str
    family: str
    donors_per_dataset: int
    selection_seed: int
    expected_c06_s01_mix: int
    sampling_design: dict[str, object]
    source_hashes: dict[str, str]
    cell_sampling_seeds: dict[str, int]


CASES = [
    Case(
        slug="alignment_36_donors",
        source_slug="test_alignment_36_donors",
        case_name="Lung three-platform alignment: 36 donors per dataset",
        # Use the historical current-RAMP family so the old current result is
        # reproducible; current/ver1/ver2 must all receive this exact string.
        family="lung_alignment_36x3_joint8_test",
        donors_per_dataset=36,
        selection_seed=20260826,
        expected_c06_s01_mix=628,
        sampling_design={
            "Senosain CyTOF": (
                "36 patients sampled with seed 20260827 from the 70-patient final "
                "annotated 140k pool; 20,000 cells allocated as 555/556 per patient"
            ),
            "Cords IMC": (
                "36 technically eligible patients: six random patients from each of "
                "TMA 86, 87, 88, 175, 176, 178; 20,000 cells allocated as 555/556 per patient"
            ),
            "Monkman CODEX": (
                "all 36 tissue-QC-passed cores; 20,000 cells allocated as 555/556 per core"
            ),
        },
        source_hashes={
            "input/raw_joint8.npy": "4058ada477160b935df1543a38cfabc12022367dda083091d24868613dac522b",
            "input/cell_metadata.csv.gz": "fce95adb69a2ba67930badada73f24cd29dc03eb5fd5ead8d64dce982ae284b2",
            "input/markers.txt": "067c8e1bfcd0d6292350ecaab8356f3223b0c9e5cf6a8d4f6826d141bd1f5ffb",
            "output/senosain_broad_celltype_labels.csv.gz": "47f3f96f1e365abafe6ae4879819aaf0278e0df6a8e690a706c690bf62cd8210",
            "frozen/selected_36x3_donors.csv": "99b56f7022b950e0d98e3b8d03a6a3db278f0b5019bee06c671d1a0674a1eadc",
            "frozen/selection_manifest.json": "ee0aab281f6c25290483b9c4491a2ce880bf7c2d42b9a00baf8de33876b687cc",
            "intermediate/cell_sampling_audit.csv": "cb5b041892f1bd418184bcebb825a6df1a5b0c5181a3c33a57228bd5445bd83d",
        },
        cell_sampling_seeds={
            "Senosain CyTOF": 20260837,
            "Cords IMC": 20260829,
            "Monkman CODEX": 20260839,
        },
    ),
    Case(
        slug="alignment_10_donors",
        source_slug="test_alignment_10_donors_20k_cells",
        case_name="Lung three-platform alignment: nested 10 donors per dataset",
        # Again retain the historical current-RAMP family and use it unchanged
        # for every method in the matched comparison.
        family="lung_alignment_nested_10x3_20k_joint8_current_ramp",
        donors_per_dataset=10,
        selection_seed=20260827,
        expected_c06_s01_mix=358,
        sampling_design={
            "Senosain CyTOF": (
                "10 patients randomly selected from the frozen parent 36; "
                "2,000 cells per patient without replacement"
            ),
            "Cords IMC": (
                "nested TMA-stratified selection covering all six parent TMAs, with a "
                "second patient in four strata; 2,000 cells per patient without replacement"
            ),
            "Monkman CODEX": (
                "10 cores randomly selected from the parent 36 after excluding 5D "
                "because it has fewer than 2,000 cells; 2,000 cells per core"
            ),
        },
        source_hashes={
            "input/raw_joint8.npy": "83cef15257b1a522cc11490a5b2b70e97379a469f2120b52b6b9cb1a4c231538",
            "input/cell_metadata.csv.gz": "0fc11d3c970df227c2f2a97c356340d5fdead4862de5750759608ba3fda0e42d",
            "input/markers.txt": "067c8e1bfcd0d6292350ecaab8356f3223b0c9e5cf6a8d4f6826d141bd1f5ffb",
            "output/senosain_broad_celltype_labels.csv.gz": "31ca3ecf2531390ba026228df2a0a45782fd140cdb6ed1d299d8098061f13a87",
            "frozen/selected_10x3_donors.csv": "34c4d1e784a8b46137fafd86fa400a7793dfeab8ca8887216d276ce02c1905e8",
            "frozen/selection_manifest.json": "1b66fe402cd8ccb59667bb8a60537aba3f89dd1b20c21742d562dec52ac405bf",
            "intermediate/cell_sampling_audit.csv": "63811e2741f9be0e483092d26a4653f76f321d2b6e07412c34a0643b8d71a3b9",
        },
        cell_sampling_seeds={
            "Senosain CyTOF": 20260838,
            "Cords IMC": 20260839,
            "Monkman CODEX": 20260840,
        },
    ),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_inside_lung(path: Path) -> None:
    try:
        path.resolve().relative_to(HERE)
    except ValueError as error:
        raise ValueError(f"write target escapes {HERE}: {path}") from error


def atomic_copy(source: Path, destination: Path) -> None:
    assert_inside_lung(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def atomic_save_npy(path: Path, values: np.ndarray) -> None:
    assert_inside_lung(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    os.replace(temporary, path)


def atomic_write_text(path: Path, value: str) -> None:
    assert_inside_lung(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def write_csv_gzip(frame: pd.DataFrame, path: Path) -> None:
    assert_inside_lung(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="", fileobj=raw_handle, mode="wb", compresslevel=6, mtime=0
        ) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                frame.to_csv(text, index=False)
    os.replace(temporary, path)


def validate_sources(case: Case, source: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in case.source_hashes.items():
        path = source / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        value = sha256(path)
        if value != expected:
            raise AssertionError(
                f"Frozen upstream changed: {path}; expected {expected}, observed {value}"
            )
        observed[str(path.resolve())] = value
    return observed


def build_metadata(case: Case, source: Path) -> pd.DataFrame:
    metadata = pd.read_csv(source / "input/cell_metadata.csv.gz", low_memory=False)
    labels = pd.read_csv(
        source / "output/senosain_broad_celltype_labels.csv.gz", low_memory=False
    )
    if len(metadata) != 60_000 or len(labels) != 60_000:
        raise AssertionError("Each lung case must contain exactly 60,000 input rows")
    for column in ("dataset", "unit_key", "donor_id", "source_cell_id"):
        if not np.array_equal(
            metadata[column].astype(str).to_numpy(), labels[column].astype(str).to_numpy()
        ):
            raise AssertionError(f"Metadata/label row mismatch in {column}")

    display_label = labels["senosain_broad_label"].astype(str)
    if set(display_label.unique()) - set(LABELS):
        raise AssertionError(f"Unexpected labels: {sorted(set(display_label.unique()) - set(LABELS))}")
    if "c06_s01_reassigned_to_mix" in metadata.columns:
        c06_s01 = metadata["c06_s01_reassigned_to_mix"].fillna(False).astype(bool)
    else:
        c06_s01 = (
            metadata["dataset"].astype(str).eq("Senosain CyTOF")
            & metadata["cell_type_raw"].astype(str).eq("CD4 T")
            & display_label.eq("Mix")
        )
    if int(c06_s01.sum()) != case.expected_c06_s01_mix:
        raise AssertionError(
            f"Expected {case.expected_c06_s01_mix} C06/S01 Mix cells, found {int(c06_s01.sum())}"
        )

    result = metadata.copy()
    result.insert(0, "row_id", np.arange(len(result), dtype=np.int64))
    result["donor"] = result["donor_id"].astype(str)
    result["display_label"] = display_label.to_numpy()
    result["c06_s01_reassigned_to_mix"] = c06_s01.to_numpy(bool)
    result["display_label_origin"] = result["dataset"].map(
        {
            "Senosain CyTOF": "project full-panel annotation; not a provider cell label",
            "Cords IMC": "provider SCE cell_category/cell_type harmonized post hoc",
            "Monkman CODEX": "provider H5AD cell_types harmonized post hoc",
        }
    )
    if result[["dataset", "unit_key", "donor", "display_label"]].isna().any().any():
        raise AssertionError("Missing required lung metadata")
    return result


def prepare_case(case: Case) -> None:
    source = SOURCE_ROOT / case.source_slug
    destination = HERE / case.slug
    input_dir = destination / "input"
    frozen_dir = destination / "frozen"
    source_sha = validate_sources(case, source)

    markers = [
        line.strip()
        for line in (source / "input/markers.txt").read_text().splitlines()
        if line.strip()
    ]
    if markers != MARKERS:
        raise AssertionError(f"Joint marker order changed: {markers}")
    matrix = np.load(source / "input/raw_joint8.npy", mmap_mode="r")
    if matrix.shape != (60_000, 8) or matrix.dtype != np.float64:
        raise AssertionError(f"Unexpected frozen matrix {matrix.shape}, {matrix.dtype}")
    if not np.isfinite(matrix).all():
        raise AssertionError("Non-finite values in frozen lung matrix")
    metadata = build_metadata(case, source)

    observed_cells = metadata.groupby("dataset", sort=False).size().astype(int).to_dict()
    observed_units = (
        metadata.groupby("dataset", sort=False)["unit_key"].nunique().astype(int).to_dict()
    )
    expected_cells = {dataset: 20_000 for dataset in DATASETS}
    expected_units = {dataset: case.donors_per_dataset for dataset in DATASETS}
    if list(dict.fromkeys(metadata["dataset"].astype(str))) != DATASETS:
        raise AssertionError("Dataset block order changed")
    if observed_cells != expected_cells or observed_units != expected_units:
        raise AssertionError((observed_cells, observed_units))

    # The sampled eight-marker matrix and marker file are copied byte-for-byte;
    # the complete primary archives are never copied into test0827.
    atomic_copy(source / "input/raw_joint8.npy", input_dir / "raw.npy")
    atomic_copy(source / "input/markers.txt", input_dir / "markers.txt")
    write_csv_gzip(metadata, input_dir / "metadata.csv.gz")
    atomic_save_npy(input_dir / "display_row_ids.npy", np.arange(60_000, dtype=np.int64))

    donor_name = f"selected_{case.donors_per_dataset}x3_donors.csv"
    atomic_copy(source / f"frozen/{donor_name}", frozen_dir / "donor_selection.csv")
    atomic_copy(
        source / "intermediate/cell_sampling_audit.csv",
        frozen_dir / "cell_sampling_audit.csv",
    )
    atomic_copy(
        source / "frozen/selection_manifest.json",
        frozen_dir / "source_selection_manifest.json",
    )

    label_counts = (
        metadata.groupby(["dataset", "display_label"], sort=False)
        .size()
        .rename("cells")
        .reset_index()
    )
    write_csv_gzip(label_counts, frozen_dir / "display_label_counts.csv.gz")

    output_paths = {
        "raw.npy": input_dir / "raw.npy",
        "metadata.csv.gz": input_dir / "metadata.csv.gz",
        "markers.txt": input_dir / "markers.txt",
        "display_row_ids.npy": input_dir / "display_row_ids.npy",
        "donor_selection.csv": frozen_dir / "donor_selection.csv",
        "cell_sampling_audit.csv": frozen_dir / "cell_sampling_audit.csv",
        "source_selection_manifest.json": frozen_dir / "source_selection_manifest.json",
        "display_label_counts.csv.gz": frozen_dir / "display_label_counts.csv.gz",
    }
    output_sha = {name: sha256(path) for name, path in output_paths.items()}
    if output_sha["raw.npy"] != case.source_hashes["input/raw_joint8.npy"]:
        raise AssertionError("raw.npy is not a byte-identical copy of the frozen sampled input")
    if output_sha["markers.txt"] != case.source_hashes["input/markers.txt"]:
        raise AssertionError("markers.txt is not byte-identical to the frozen source")

    manifest = {
        "case": case.case_name,
        "kind": "lung donor-depth sensitivity",
        "family": case.family,
        "family_policy": (
            "Pass this exact family string unchanged to current RAMP, ramp_ver1, and any "
            "test0827 candidate so their legacy hash partitions are matched."
        ),
        "exact_same_family_across_methods": True,
        "pair_fit_override": {"minimum": 7000, "per_unit": 0, "maximum": 7000},
        "pair_fit_override_reason": "reproduce the established lung-story runtime fit depth",
        "datasets": DATASETS,
        "cells": 60_000,
        "cells_per_dataset": 20_000,
        "donors_per_dataset": case.donors_per_dataset,
        "dataset_cells": observed_cells,
        "dataset_units": observed_units,
        "marker_count": len(MARKERS),
        "markers": MARKERS,
        "selection_seed": case.selection_seed,
        "cell_sampling_seeds": case.cell_sampling_seeds,
        "sampling": case.sampling_design,
        "selection_blinding": (
            "Clinical outcomes, expression, annotations, alignment, and UMAP were not used "
            "for donor selection; labels remain post-hoc only."
        ),
        "fit_metadata_columns": ["dataset", "unit_key"],
        "label_column": "display_label",
        "provider_labels_used_for_fitting": False,
        "display": {
            "cells": 60_000,
            "cells_per_dataset": 20_000,
            "row_policy": "all frozen fitting rows, in their original row order",
            "display_row_ids_file": "display_row_ids.npy",
            "label_vocabulary": LABELS,
            "palette": DISPLAY_PALETTE,
            "draw_order": DISPLAY_DRAW_ORDER,
            "mix_is_independent_label": True,
            "mix_color": DISPLAY_PALETTE["Mix"],
            "senosain_c06_s01_reassigned_to_mix": case.expected_c06_s01_mix,
        },
        "umap_n_neighbors": 30,
        "umap_min_dist": 0.12,
        "metric_cell_cap": 30_000,
        "input_status": (
            "Byte/hash-locked normalization of the previously audited frozen sample; this "
            "script intentionally does not draw a new random sample from the primary archives."
        ),
        "raw_input_caveat": (
            "raw.npy means the highest-available sampled source-scale matrix before any new "
            "test0827 transform. It is not a claim that all three platforms are instrument-raw "
            "or share a common physical scale."
        ),
        "input_scale": {
            "Senosain CyTOF": (
                "provider-normalized and denoised FCS values; no additional transform here"
            ),
            "Cords IMC": (
                "non-LS CellProfiler corrected mean intensities, clipped at zero and multiplied "
                "by 65535 to reconstruct the provider counts-assay scale"
            ),
            "Monkman CODEX": (
                "segmented cell intensities from 4301_cells.csv; no additional transform here"
            ),
        },
        "annotation_provenance": {
            "Senosain CyTOF": (
                "project-derived full-panel annotation; Senosain provides no per-cell provider label"
            ),
            "Cords IMC": "provider SCE annotations, harmonized after sampling",
            "Monkman CODEX": "provider H5AD cell_types, harmonized after sampling",
            "C06/S01": (
                f"{case.expected_c06_s01_mix} sampled Senosain cells conservatively retained "
                "as independent Mix rather than CD4 T"
            ),
        },
        "full_primary_raw_datasets_copied": False,
        "sampled_joint_marker_matrix_saved": True,
        "preparation_script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "source_sha256": source_sha,
        "output_sha256": output_sha,
        "all_new_files_within_test0827_lung": True,
    }
    atomic_write_text(
        input_dir / "input_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    print(
        f"{case.slug}: {len(metadata):,} cells, {case.donors_per_dataset} donors/dataset, "
        f"Mix={int(metadata.display_label.eq('Mix').sum()):,}",
        flush=True,
    )


def main() -> None:
    for case in CASES:
        prepare_case(case)
    print(f"Prepared {len(CASES)} lung cases under {HERE}", flush=True)


if __name__ == "__main__":
    main()
