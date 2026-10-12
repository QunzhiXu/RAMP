#!/usr/bin/env python3
"""Package the frozen 60,000-cell lung three-platform benchmark.

The accepted input was selected and sampled upstream.  This builder validates
that frozen input byte-for-byte, adds the original Cords provider annotations,
and packages source-scale marker values plus locked post-hoc metadata.  It does
not resample, normalize, align, cluster, or run UMAP.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path


sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
FINAL_XU = HERE.parents[1]
DEFAULT_SOURCE = FINAL_XU / "test0827/lung/alignment_10_donors"
DEFAULT_CORDS_LABELS = (
    HERE / "source_inputs/cords_sample_20k_provider_labels.csv.gz"
)

DATASETS = ["Senosain CyTOF", "Cords IMC", "Monkman CODEX"]
SLUGS = {
    "Senosain CyTOF": "senosain_cytof",
    "Cords IMC": "cords_imc",
    "Monkman CODEX": "monkman_codex",
}
TECHNOLOGY = {
    "Senosain CyTOF": "CyTOF",
    "Cords IMC": "IMC",
    "Monkman CODEX": "CODEX",
}
EXPECTED_CELLS = {dataset: 20_000 for dataset in DATASETS}
EXPECTED_UNITS = {dataset: 10 for dataset in DATASETS}
EXPECTED_DONORS = {
    "Senosain CyTOF": [
        "15326", "13769", "14958", "13538", "15569",
        "15083", "7984", "13376", "13651", "13014",
    ],
    "Cords IMC": [
        "86_28", "86_30", "87_257", "88_367", "88_452",
        "175_191", "175_43", "176_358", "178_457", "178_486",
    ],
    "Monkman CODEX": [
        "6D", "4B", "1E", "9A", "5E", "2B", "8C", "7B", "4E", "9B",
    ],
}
MARKERS = ["CD3", "CD4", "CD8", "CD31", "HLA_DR", "KI67", "VIMENTIN", "EPI"]
LABELS = [
    "Tumor/Epithelial", "CD4 T", "CD8 T", "B-like",
    "Myeloid", "Endothelial", "Mix",
]
JOINT_LABELS = [
    "Tumor/Epithelial", "CD4 T", "CD8 T", "B-like",
    "Myeloid", "Endothelial", "Non-joint",
]

EXPECTED_SOURCE_SHA256 = {
    "input/raw.npy": "83cef15257b1a522cc11490a5b2b70e97379a469f2120b52b6b9cb1a4c231538",
    "input/metadata.csv.gz": "9ccc1a4f05ed1fca115bf309bc2048da1413bd3184e5ead8dbf685e01645c349",
    "input/markers.txt": "067c8e1bfcd0d6292350ecaab8356f3223b0c9e5cf6a8d4f6826d141bd1f5ffb",
    "input/input_manifest.json": "340948780ce8d8f0067ef0d9a245efc81f40706ce9065d5e340d70d2f2163a7d",
    "frozen/donor_selection.csv": "34c4d1e784a8b46137fafd86fa400a7793dfeab8ca8887216d276ce02c1905e8",
    "frozen/cell_sampling_audit.csv": "63811e2741f9be0e483092d26a4653f76f321d2b6e07412c34a0643b8d71a3b9",
    "frozen/source_selection_manifest.json": "1b66fe402cd8ccb59667bb8a60537aba3f89dd1b20c21742d562dec52ac405bf",
}
EXPECTED_CORDS_LABELS_SHA256 = (
    "ae814004ab239a48c4738df975684e38f48a295439f8417c1e84cd19adb7ec72"
)

SOURCE_MARKERS = {
    "Senosain CyTOF": {
        "CD3": "CD3", "CD4": "CD4", "CD8": "CD8", "CD31": "CD31",
        "HLA_DR": "HLA_DR", "KI67": "KI67", "VIMENTIN": "VIMENTIN",
        "EPI": "CYTOKERATIN",
    },
    "Cords IMC": {
        "CD3": "Intensity_MeanIntensityCorrected_FullStackFiltered_c15",
        "CD4": "Intensity_MeanIntensityCorrected_FullStackFiltered_c33",
        "CD8": "Intensity_MeanIntensityCorrected_FullStackFiltered_c28",
        "CD31": "Intensity_MeanIntensityCorrected_FullStackFiltered_c34",
        "HLA_DR": "Intensity_MeanIntensityCorrected_FullStackFiltered_c6",
        "KI67": "Intensity_MeanIntensityCorrected_FullStackFiltered_c41",
        "VIMENTIN": "Intensity_MeanIntensityCorrected_FullStackFiltered_c24",
        "EPI": "Intensity_MeanIntensityCorrected_FullStackFiltered_c37",
    },
    "Monkman CODEX": {
        "CD3": "CD3e", "CD4": "CD4", "CD8": "CD8", "CD31": "CD31",
        "HLA_DR": "HLA-DR", "KI67": "Ki67", "VIMENTIN": "Vimentin",
        "EPI": "PanCK",
    },
}

INPUT_SCALE = {
    "Senosain CyTOF": (
        "provider-normalized and denoised FCS event values; no new transform"
    ),
    "Cords IMC": (
        "non-LS CellProfiler corrected mean intensities clipped at zero and "
        "multiplied by 65535 to reconstruct the provider counts-assay scale"
    ),
    "Monkman CODEX": (
        "segmented cell intensities from 4301_cells.csv; no new transform"
    ),
}

SELECTION = {
    "Senosain CyTOF": (
        "random 10 from frozen parent 36; 2,000 cells per patient"
    ),
    "Cords IMC": (
        "nested TMA-stratified random 10 covering all six parent TMAs; "
        "2,000 cells per patient"
    ),
    "Monkman CODEX": (
        "random 10 from parent 36 after excluding 5D (<2,000 cells); "
        "2,000 cells per core/patient"
    ),
}

MANAGED = [
    "input", "senosain_cytof", "cords_imc", "monkman_codex",
    "selected_cells.csv.gz", "donor_selection.csv", "sampling_counts.csv",
    "donor_cell_counts.csv", "celltype_counts.csv", "label_mapping_audit.csv",
    "dataset_manifest.csv", "marker_manifest.csv", "manifest.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--cords-labels", type=Path, default=DEFAULT_CORDS_LABELS)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv_gz(
    frame: pd.DataFrame,
    path: Path,
    *,
    float_format: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="", fileobj=raw_handle, mode="wb", compresslevel=6, mtime=0
        ) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                frame.to_csv(text, index=False, float_format=float_format)


def dataset_block_order(series: pd.Series) -> list[str]:
    values = series.astype(str).reset_index(drop=True)
    starts = values.ne(values.shift(fill_value="")).to_numpy()
    return values.loc[starts].tolist()


def derive_cords_label(values: pd.Series) -> np.ndarray:
    mapping = {
        "Bcell": "B-like",
        "Blood": "Endothelial",
        "CD4": "CD4 T",
        "CD8": "CD8 T",
        "HEV": "Endothelial",
        "Lymphatic": "Endothelial",
        "Myeloid": "Myeloid",
        "Neutrophil": "Myeloid",
        "normal": "Tumor/Epithelial",
        "hypoxic": "Tumor/Epithelial",
        "Collagen_CAF": "Mix",
        "IDO_CAF": "Mix",
        "Other": "Mix",
        "PDPN_CAF": "Mix",
        "SMA_CAF": "Mix",
        "dCAF": "Mix",
        "hypoxic_CAF": "Mix",
        "iCAF": "Mix",
        "mCAF": "Mix",
        "tpCAF": "Mix",
        "vCAF": "Mix",
    }
    result = values.astype(str).map(mapping)
    if result.isna().any():
        unexpected = sorted(values.loc[result.isna()].astype(str).unique())
        raise AssertionError(f"Unexpected Cords provider cell types: {unexpected}")
    return result.to_numpy(str)


def derive_monkman_label(values: pd.Series) -> np.ndarray:
    mapping = {
        "Tumour": "Tumor/Epithelial",
        "HLADR Tumour": "Tumor/Epithelial",
        "CD44 Tumour": "Tumor/Epithelial",
        "Proliferating Tumour": "Tumor/Epithelial",
        "B Cells": "B-like",
        "CD4 Cells": "CD4 T",
        "Effector CD4": "CD4 T",
        "Treg": "CD4 T",
        "CD8 Cells": "CD8 T",
        "Blood Vessels": "Endothelial",
        "Lymphatics": "Endothelial",
        "Monocytes": "Myeloid",
        "Granulocytes": "Myeloid",
        "Macrophages": "Myeloid",
        "Mast Cells": "Myeloid",
        "Stroma": "Mix",
        "Unclassified": "Mix",
        "Proliferating Lymphocytes": "Mix",
        "Lymphocytes": "Mix",
        "CCR7+ CD8/CD4 Cells": "Mix",
        "Artifact": "Mix",
    }
    result = values.astype(str).map(mapping)
    if result.isna().any():
        unexpected = sorted(values.loc[result.isna()].astype(str).unique())
        raise AssertionError(f"Unexpected Monkman provider cell types: {unexpected}")
    return result.to_numpy(str)


def validate_sources(source: Path, cords_labels: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in EXPECTED_SOURCE_SHA256.items():
        path = source / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        value = sha256(path)
        if value != expected:
            raise AssertionError(
                f"Accepted lung input changed: {relative}; expected {expected}, observed {value}"
            )
        observed[str(path.resolve())] = value
    if not cords_labels.is_file():
        raise FileNotFoundError(cords_labels)
    value = sha256(cords_labels)
    if value != EXPECTED_CORDS_LABELS_SHA256:
        raise AssertionError(
            "Accepted Cords provider labels changed: "
            f"expected {EXPECTED_CORDS_LABELS_SHA256}, observed {value}"
        )
    observed[str(cords_labels.resolve())] = value
    return observed


def load_and_validate(source: Path, cords_labels_path: Path):
    source_hashes = validate_sources(source, cords_labels_path)
    raw = np.load(source / "input/raw.npy", allow_pickle=False)
    metadata = pd.read_csv(source / "input/metadata.csv.gz", low_memory=False)
    markers = [
        value.strip()
        for value in (source / "input/markers.txt").read_text().splitlines()
        if value.strip()
    ]
    upstream = json.loads((source / "input/input_manifest.json").read_text())
    donor_selection = pd.read_csv(
        source / "frozen/donor_selection.csv", dtype={"donor_id": str}, low_memory=False
    )
    sampling = pd.read_csv(
        source / "frozen/cell_sampling_audit.csv",
        dtype={"donor_id": str},
        low_memory=False,
    )
    source_selection = json.loads(
        (source / "frozen/source_selection_manifest.json").read_text()
    )

    if raw.shape != (60_000, 8) or raw.dtype != np.float64:
        raise AssertionError((raw.shape, raw.dtype))
    if markers != MARKERS or not np.isfinite(raw).all() or np.any(raw < 0):
        raise AssertionError("Marker contract or source matrix validity changed")
    if len(metadata) != len(raw):
        raise AssertionError("Matrix/metadata row mismatch")
    if not np.array_equal(metadata["row_id"].to_numpy(), np.arange(len(metadata))):
        raise AssertionError("row_id no longer matches matrix order")
    if metadata["source_cell_id"].astype(str).duplicated().any():
        raise AssertionError("source_cell_id is not globally unique")
    if dataset_block_order(metadata["dataset"]) != DATASETS:
        raise AssertionError("Dataset blocks are not contiguous in canonical order")
    observed_cells = metadata.groupby("dataset", sort=False).size().astype(int).to_dict()
    observed_units = (
        metadata.groupby("dataset", sort=False)["unit_key"].nunique().astype(int).to_dict()
    )
    if observed_cells != EXPECTED_CELLS or observed_units != EXPECTED_UNITS:
        raise AssertionError((observed_cells, observed_units))
    for dataset in DATASETS:
        block = metadata.loc[metadata["dataset"].astype(str).eq(dataset)]
        observed = block["donor_id"].astype(str).drop_duplicates().tolist()
        if observed != EXPECTED_DONORS[dataset]:
            raise AssertionError(f"{dataset}: donor order changed: {observed}")
        if not block.groupby("donor_id", sort=False).size().eq(2_000).all():
            raise AssertionError(f"{dataset}: not every donor contributes 2,000 cells")

    if upstream.get("selection_seed") != 20_260_827:
        raise AssertionError("Selection seed changed")
    if source_selection.get("selection_seed") != 20_260_827:
        raise AssertionError("Source selection seed changed")
    if upstream.get("provider_labels_used_for_fitting") is not False:
        raise AssertionError("Provider labels were exposed during alignment fitting")

    cords_labels = pd.read_csv(
        cords_labels_path, dtype={"donor_id": str}, low_memory=False
    )
    required = {
        "sample_order", "source_cell_id", "donor_id",
        "cell_category", "cell_type", "cell_subtype",
    }
    if set(cords_labels.columns) != required or len(cords_labels) != 20_000:
        raise AssertionError("Unexpected Cords provider-label table")
    if cords_labels["source_cell_id"].astype(str).duplicated().any():
        raise AssertionError("Cords provider source_cell_id is not unique")
    cords_block = metadata.loc[metadata["dataset"].astype(str).eq("Cords IMC")]
    if not np.array_equal(
        cords_labels["sample_order"].to_numpy(), np.arange(1, 20_001)
    ):
        raise AssertionError("Cords provider sample_order changed")
    for column in ("source_cell_id", "donor_id"):
        if not np.array_equal(
            cords_labels[column].astype(str).to_numpy(),
            cords_block[column].astype(str).to_numpy(),
        ):
            raise AssertionError(f"Cords provider labels do not match frozen rows: {column}")

    return (
        raw,
        metadata,
        markers,
        upstream,
        donor_selection,
        sampling,
        source_selection,
        cords_labels,
        source_hashes,
    )


def canonical_metadata(metadata: pd.DataFrame, cords_labels: pd.DataFrame) -> pd.DataFrame:
    answer = metadata.copy()
    answer.insert(
        1,
        "cell_id",
        answer["dataset"].map(SLUGS).astype(str)
        + "::"
        + answer["source_cell_id"].astype(str),
    )
    if answer["cell_id"].duplicated().any():
        raise AssertionError("canonical cell_id is not unique")
    answer["technology"] = answer["dataset"].map(TECHNOLOGY)
    answer["donor"] = answer["donor_id"].astype(str)
    answer["specimen_id"] = answer["donor_id"].astype(str)
    answer["provider_cell_category"] = ""
    answer["provider_cell_type"] = ""
    answer["provider_cell_subtype"] = ""
    answer["annotation_label_original"] = answer["cell_type_raw"].astype(str)

    cords_mask = answer["dataset"].astype(str).eq("Cords IMC")
    answer.loc[cords_mask, "provider_cell_category"] = (
        cords_labels["cell_category"].astype(str).to_numpy()
    )
    answer.loc[cords_mask, "provider_cell_type"] = (
        cords_labels["cell_type"].astype(str).to_numpy()
    )
    answer.loc[cords_mask, "provider_cell_subtype"] = (
        cords_labels["cell_subtype"].astype(str).to_numpy()
    )
    answer.loc[cords_mask, "annotation_label_original"] = (
        cords_labels["cell_type"].astype(str).to_numpy()
    )

    sen_mask = answer["dataset"].astype(str).eq("Senosain CyTOF")
    monk_mask = answer["dataset"].astype(str).eq("Monkman CODEX")
    answer.loc[monk_mask, "provider_cell_type"] = (
        answer.loc[monk_mask, "cell_type_raw"].astype(str).to_numpy()
    )

    recomputed = np.empty(len(answer), dtype=object)
    sen = answer.loc[sen_mask, "cell_type_raw"].astype(str).to_numpy()
    sen_mix = answer.loc[sen_mask, "c06_s01_reassigned_to_mix"].astype(bool).to_numpy()
    sen[sen_mix] = "Mix"
    recomputed[sen_mask.to_numpy()] = sen
    recomputed[cords_mask.to_numpy()] = derive_cords_label(
        answer.loc[cords_mask, "provider_cell_type"]
    )
    recomputed[monk_mask.to_numpy()] = derive_monkman_label(
        answer.loc[monk_mask, "cell_type_raw"]
    )
    recomputed = np.asarray(recomputed, dtype=str)
    frozen = answer["display_label"].astype(str).to_numpy()
    if not np.array_equal(recomputed, frozen):
        mismatch = np.flatnonzero(recomputed != frozen)[:10]
        raise AssertionError(f"Recomputed display labels differ at rows {mismatch.tolist()}")
    if set(frozen) != set(LABELS):
        raise AssertionError(f"Unexpected display labels: {sorted(set(frozen))}")
    if int(answer["c06_s01_reassigned_to_mix"].astype(bool).sum()) != 358:
        raise AssertionError("Expected exactly 358 Senosain C06/S01 display-Mix cells")

    answer["celltype"] = frozen
    answer["celltype_joint_evaluation"] = np.where(
        answer["celltype"].eq("Mix"), "Non-joint", answer["celltype"]
    )
    answer["joint_evaluation"] = ~answer["celltype_joint_evaluation"].eq("Non-joint")
    answer["labels_used_for_alignment"] = False
    answer["celltype_relabel_version"] = "lung_broad_display_v1_60000"
    answer["celltype_provenance"] = answer["dataset"].map({
        "Senosain CyTOF": (
            "project full-panel annotation; C06/S01 display refinement; not provider ground truth"
        ),
        "Cords IMC": "provider SCE annotation harmonized post hoc",
        "Monkman CODEX": "provider H5AD cell_types harmonized post hoc",
    })
    return answer


def build_files(
    stage: Path,
    source: Path,
    cords_labels_path: Path,
    raw: np.ndarray,
    canonical: pd.DataFrame,
    upstream: dict,
    donor_selection: pd.DataFrame,
    sampling: pd.DataFrame,
    source_selection: dict,
    source_hashes: dict[str, str],
) -> None:
    input_dir = stage / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / "input/raw.npy", input_dir / "raw.npy")
    shutil.copyfile(source / "input/markers.txt", input_dir / "markers.txt")
    shutil.copyfile(
        source / "input/input_manifest.json",
        input_dir / "upstream_input_manifest.json",
    )
    write_csv_gz(canonical[["dataset", "unit_key"]], input_dir / "fit_metadata.csv.gz")
    write_csv_gz(canonical, input_dir / "locked_metadata.csv.gz")

    identity_columns = [
        "row_id", "cell_id", "source_cell_id", "source_row_zero_based",
        "dataset", "technology", "donor", "donor_id", "specimen_id", "unit_key",
    ]
    write_csv_gz(canonical[identity_columns], stage / "selected_cells.csv.gz")
    donor_selection.to_csv(stage / "donor_selection.csv", index=False)
    sampling.to_csv(stage / "sampling_counts.csv", index=False)

    donor_counts = (
        canonical.groupby(["dataset", "donor"], sort=False)
        .size().rename("cells").reset_index()
    )
    donor_counts.to_csv(stage / "donor_cell_counts.csv", index=False)
    celltype_counts = (
        canonical.groupby(["dataset", "celltype"], sort=False)
        .size().rename("cells").reset_index()
    )
    celltype_counts.to_csv(stage / "celltype_counts.csv", index=False)
    mapping = (
        canonical.groupby(
            [
                "dataset", "annotation_label_original", "provider_cell_category",
                "provider_cell_type", "provider_cell_subtype", "celltype",
                "celltype_joint_evaluation", "celltype_provenance",
            ],
            sort=False,
            dropna=False,
        )
        .size().rename("cells").reset_index()
    )
    mapping.to_csv(stage / "label_mapping_audit.csv", index=False)

    dataset_rows = []
    for dataset in DATASETS:
        mask = canonical["dataset"].astype(str).eq(dataset).to_numpy()
        block_meta = canonical.loc[mask].reset_index(drop=True)
        block_raw = raw[mask]
        slug = SLUGS[dataset]
        block_dir = stage / slug
        block_dir.mkdir(parents=True, exist_ok=True)
        inspectable = block_meta[
            ["row_id", "cell_id", "dataset", "donor", "unit_key"]
        ].copy()
        for marker_index, marker in enumerate(MARKERS):
            inspectable[marker] = block_raw[:, marker_index]
        alignment_path = block_dir / "alignment_input_8.csv.gz"
        write_csv_gz(inspectable, alignment_path, float_format="%.17g")
        write_csv_gz(block_meta, block_dir / "locked_metadata.csv.gz")
        roundtrip = pd.read_csv(alignment_path, usecols=MARKERS).to_numpy(np.float64)
        if not np.array_equal(roundtrip, block_raw):
            # pandas' decimal parser can differ from NumPy's binary float by a
            # few ulps even with 17 significant digits.  raw.npy remains the
            # canonical byte-exact matrix; this only validates the inspectable
            # CSV representation.
            if not np.allclose(roundtrip, block_raw, rtol=5e-13, atol=1e-15):
                raise AssertionError(f"{dataset}: marker CSV numeric round trip failed")
        dataset_rows.append({
            "dataset": dataset,
            "display_name": dataset,
            "technology": TECHNOLOGY[dataset],
            "patients_or_cores": int(block_meta["donor"].nunique()),
            "fit_units": int(block_meta["unit_key"].nunique()),
            "cells": int(len(block_meta)),
            "selection": SELECTION[dataset],
            "source_scale": INPUT_SCALE[dataset],
            "annotation_provenance": str(block_meta["celltype_provenance"].iloc[0]),
        })
    pd.DataFrame(dataset_rows).to_csv(stage / "dataset_manifest.csv", index=False)

    marker_rows = []
    for marker in MARKERS:
        for dataset in DATASETS:
            note = "highest-available provider/source measurement; no new transform"
            if dataset == "Cords IMC":
                note = "clipped at zero then multiplied by 65535 upstream"
            marker_rows.append({
                "canonical_marker": marker,
                "dataset": dataset,
                "source_column": SOURCE_MARKERS[dataset][marker],
                "role": "alignment feature",
                "note": note,
            })
    pd.DataFrame(marker_rows).to_csv(stage / "marker_manifest.csv", index=False)

    files = sorted(
        path for path in stage.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    manifest = {
        "package": "lung_three_platform_benchmark_v1_60000",
        "purpose": "paper-opening technical alignment benchmark; not a clinical matched cohort",
        "canonical_matrix": "input/raw.npy",
        "fit_metadata": "input/fit_metadata.csv.gz",
        "fit_metadata_columns": ["dataset", "unit_key"],
        "locked_metadata": "input/locked_metadata.csv.gz",
        "cells": 60_000,
        "markers": MARKERS,
        "dataset_order": DATASETS,
        "dataset_cells": EXPECTED_CELLS,
        "dataset_units": EXPECTED_UNITS,
        "dataset_donors_in_frozen_order": EXPECTED_DONORS,
        "selection": SELECTION,
        "selection_seed": 20_260_827,
        "selection_rng": "numpy.random.Generator(PCG64), with dataset-specific offsets",
        "cell_sampling_seeds": upstream["cell_sampling_seeds"],
        "cells_per_donor": 2_000,
        "sampling_without_replacement": True,
        "selection_used_expression_labels_outcomes_or_umap": False,
        "reroll_after_alignment_or_umap": False,
        "input_scale": INPUT_SCALE,
        "normalization_or_alignment_applied_by_builder": False,
        "provider_labels_available_to_fit": False,
        "display_labels": LABELS,
        "joint_evaluation_labels": JOINT_LABELS,
        "mix_policy": (
            "Mix is retained for display and mapped to Non-joint for shared-label evaluation; "
            "the 358 C06/S01 cells were refined after inspecting the established lung alignment"
        ),
        "source_selection_manifest": source_selection,
        "accepted_source": {
            "path": str(source.resolve()),
            "cords_provider_labels": str(cords_labels_path.resolve()),
            "sha256": source_hashes,
        },
        "recipe": {
            "build_benchmark.py": sha256(Path(__file__).resolve()),
            "README.md": sha256(HERE / "README.md"),
        },
        "software_versions": {
            "Python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "files_sha256": {
            str(path.relative_to(stage)): sha256(path) for path in files
        },
        "upstream_input_manifest": upstream,
    }
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def publish(stage: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stale_marker = output_dir / "relabel/PARENT_REBUILD_REQUIRED.txt"
    if (output_dir / "relabel/manifest.json").exists():
        stale_marker.parent.mkdir(parents=True, exist_ok=True)
        stale_marker.write_text(
            "The parent lung benchmark was rebuilt. Re-run relabel/build_relabel.py.\n"
        )
    for name in MANAGED:
        if name == "manifest.json":
            continue
        source = stage / name
        destination = output_dir / name
        if destination.is_dir():
            shutil.rmtree(destination)
        elif destination.exists():
            destination.unlink()
        os.replace(source, destination)
    destination = output_dir / "manifest.json"
    if destination.exists():
        destination.unlink()
    os.replace(stage / "manifest.json", destination)


def main() -> None:
    args = parse_args()
    (
        raw,
        metadata,
        markers,
        upstream,
        donor_selection,
        sampling,
        source_selection,
        cords_labels,
        source_hashes,
    ) = load_and_validate(args.source, args.cords_labels)
    del markers
    canonical = canonical_metadata(metadata, cords_labels)
    with tempfile.TemporaryDirectory(
        prefix=".lung_benchmark_stage_", dir=args.output_dir.parent
    ) as temporary:
        stage = Path(temporary) / "package"
        stage.mkdir()
        build_files(
            stage,
            args.source,
            args.cords_labels,
            raw,
            canonical,
            upstream,
            donor_selection,
            sampling,
            source_selection,
            source_hashes,
        )
        publish(stage, args.output_dir)
    print(
        "Built lung benchmark: 60,000 cells, 30 donors/cores, 8 markers, "
        f"output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
