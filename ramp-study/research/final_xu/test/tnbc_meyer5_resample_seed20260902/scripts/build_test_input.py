#!/usr/bin/env python3
"""Build a test-only TNBC input with a newly sampled five-patient Meyer block.

Tietscher and Shiao cells, values, labels, and within-dataset row order are
copied exactly from the canonical benchmark.  Five Meyer patients are selected
once from all 215 post-QC patients with an ordinary NumPy RNG; every cell from
the selected patients is retained.  No label, outcome, image, or alignment
result is used for sampling.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT_CASE = HERE.parent
CASE = Path(os.environ.get("RAMP_TNBC_TEST_CASE", str(DEFAULT_CASE))).resolve()
FINAL_XU = HERE.parents[2]
CANONICAL = FINAL_XU / "benchmark_data" / "tnbc"
MEYER_RDS = (
    FINAL_XU / "raw_data" / "tnbc" / "Meyer_2025"
    / "processed_r_objects" / "sce_ALL.rds"
)
EXPORTER = HERE / "export_meyer.R"
INPUT = CASE / "input"
AUDIT = CASE / "audit"

DONOR_SELECTION_SEED = int(os.environ.get("RAMP_TNBC_MEYER_SEED", "20260902"))
SELECTION_MODE = os.environ.get(
    "RAMP_TNBC_SELECTION_MODE", "unbiased_single_draw"
).strip()
SEED_SCREEN_MANIFEST_VALUE = os.environ.get("RAMP_TNBC_SEED_SCREEN_MANIFEST", "").strip()
SEED_SCREEN_MANIFEST = (
    Path(SEED_SCREEN_MANIFEST_VALUE).resolve() if SEED_SCREEN_MANIFEST_VALUE else None
)
DONORS_TO_SELECT = 5
DATASETS = ["Tietscher IMC", "Meyer IMC", "Shiao CODEX"]
MARKERS = [
    "CD11C", "CD20", "CD3", "CD31", "CD4", "CD68",
    "CD8", "HLA_DR", "KI67", "PANCK", "SMA",
]
MEYER_DIRECT = {
    "Neutrophil": "Myeloid",
    "HLA_DR": "Myeloid",
    "Monocyte": "Myeloid",
    "Endothelial": "Endothelial",
    "CD8_T_cell": "CD8 T",
    "Stromal": "Stroma",
    "CD4_T_cell": "CD4 T",
    "B_cell": "B lineage",
    "Immune_low": "Non-joint",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def buffer_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).view(np.uint8)).hexdigest()


def write_csv_gz(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="", fileobj=raw_handle, mode="wb", compresslevel=6, mtime=0
        ) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                frame.to_csv(text, index=False)
    os.replace(temporary, path)


def write_json(value: object, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def save_npy(values: np.ndarray, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
    os.replace(temporary, path)


def run_export(mode: str, out: Path, selected: Path | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    command = ["Rscript", str(EXPORTER), mode, str(MEYER_RDS), str(out)]
    if selected is not None:
        command.append(str(selected))
    subprocess.run(command, check=True, cwd=HERE)


def read_export(out: Path) -> tuple[np.ndarray, pd.DataFrame]:
    with h5py.File(out / "meyer_selected_counts.h5", "r") as handle:
        matrix = np.asarray(handle["X_features_by_cells"], dtype=np.float64)
        canonical = [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in np.asarray(handle["canonical_markers"])
        ]
        source = [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in np.asarray(handle["source_markers"])
        ]
    if canonical != MARKERS:
        raise AssertionError(f"Exported canonical marker order changed: {canonical}")
    expected_source = [
        "CD11c", "CD20", "CD3", "CD31_vWF", "CD4", "CD68",
        "CD8a", "HLADR", "Ki67", "panCK", "SMA",
    ]
    if source != expected_source:
        raise AssertionError(f"Exported Meyer source marker order changed: {source}")
    metadata = pd.read_csv(
        out / "meyer_selected_metadata.csv.gz",
        dtype={"source_cell_id": str, "donor": str, "sample_id": str},
        low_memory=False,
    )
    # rhdf5 presents the R feature-by-cell matrix to h5py in reversed
    # dimension order on this runtime.  Accept either physical orientation,
    # but resolve it solely from the locked marker and metadata dimensions.
    if matrix.shape == (len(MARKERS), len(metadata)):
        matrix = matrix.T
    elif matrix.shape != (len(metadata), len(MARKERS)):
        raise AssertionError((matrix.shape, len(metadata), len(MARKERS)))
    if matrix.shape != (len(metadata), len(MARKERS)):
        raise AssertionError((matrix.shape, len(metadata)))
    if not np.isfinite(matrix).all() or np.any(matrix < 0):
        raise AssertionError("Meyer selected source matrix is invalid")
    return matrix, metadata


def validate_provider_roundtrip(
    canonical_raw: np.ndarray, canonical_metadata: pd.DataFrame
) -> dict[str, object]:
    old = canonical_metadata[canonical_metadata["dataset"].eq("Meyer IMC")].copy()
    donors = pd.DataFrame({"donor": sorted(old["donor"].astype(str).unique())})
    with tempfile.TemporaryDirectory(prefix="tnbc-meyer-roundtrip-", dir=CASE) as temp:
        root = Path(temp)
        donor_path = root / "donors.csv"
        donors.to_csv(donor_path, index=False)
        run_export("selected", root, donor_path)
        observed, metadata = read_export(root)
    expected = canonical_raw[canonical_metadata["dataset"].eq("Meyer IMC").to_numpy()]
    if not np.array_equal(
        metadata["source_index_zero_based"].to_numpy(np.int64),
        old["source_index_zero_based"].to_numpy(np.int64),
    ):
        raise AssertionError("Meyer source-index round trip failed")
    if not np.array_equal(
        metadata["source_cell_id"].astype(str).to_numpy(),
        old["source_cell_id"].astype(str).to_numpy(),
    ):
        raise AssertionError("Meyer source-cell-ID round trip failed")
    maximum_error = float(np.max(np.abs(observed - expected)))
    if maximum_error != 0.0:
        raise AssertionError(f"Meyer counts assay does not reproduce canonical raw: {maximum_error}")
    return {
        "provider_assay": "counts",
        "cells_checked": int(len(old)),
        "source_index_exact": True,
        "source_cell_id_exact": True,
        "maximum_absolute_error": maximum_error,
    }


def meyer_broad_label(provider: pd.Series) -> pd.Series:
    values = provider.fillna("").astype(str)
    labels = values.map(MEYER_DIRECT)
    labels.loc[values.str.startswith("tumor_")] = "Tumor / epithelial"
    if labels.isna().any():
        raise AssertionError(
            f"Unmapped Meyer provider labels: {sorted(values[labels.isna()].unique())}"
        )
    return labels.astype(str)


def build_meyer_metadata(exported: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    provider = exported["provider_celltype"].fillna("").astype(str)
    celltype = meyer_broad_label(provider)
    rule = np.where(
        provider.str.startswith("tumor_"),
        "Meyer:celltype=tumor_*",
        "Meyer:celltype=" + provider,
    )
    base = pd.DataFrame(index=np.arange(len(exported)))
    base["cell_order"] = -1
    base["cell_id"] = "meyer_imc::" + exported["source_cell_id"].astype(str)
    base["source_cell_id"] = exported["source_cell_id"].astype(str)
    base["source_index_zero_based"] = exported["source_index_zero_based"].astype(np.int64)
    base["dataset"] = "Meyer IMC"
    base["donor"] = exported["donor"].astype(str)
    base["celltype"] = celltype
    base["celltype_provenance"] = "provider label, broad project harmonization"
    base["celltype_relabel_version"] = "tnbc_provider_broad_v1"
    base["technology"] = "IMC"
    base["donor_id"] = exported["donor"].astype(str)
    base["normalization_unit"] = exported["donor"].astype(str)
    base["acquisition_unit"] = exported["sample_id"].astype(str)
    base["sample_id"] = exported["sample_id"].astype(str)
    base["roi_id"] = exported["sample_id"].astype(str)
    base["x"] = exported["x"].to_numpy()
    base["y"] = exported["y"].to_numpy()
    base["provider_celltype"] = provider
    base["provider_cell_subtype"] = exported["provider_cell_subtype"].astype(str)
    base["provider_cell_class"] = exported["provider_cell_class"].astype(str)
    base["provider_tls"] = np.nan
    base["provider_ie"] = np.nan
    base["source_row_within_eligible_pool"] = exported["source_index_zero_based"].astype(np.int64)
    base["provider_treatment_code"] = np.nan
    base["provider_district"] = np.nan
    base["provider_pCR"] = np.nan
    base["provider_response_group"] = np.nan
    base["unit_key"] = "Meyer IMC|" + exported["donor"].astype(str)
    base["display_label"] = celltype
    base["relabel_rule"] = rule
    base["label_scope"] = np.where(
        celltype.eq("Non-joint"), "non_joint_uncertain", "shared_across_three_datasets"
    )
    base["labels_used_for_alignment"] = False
    missing = [column for column in columns if column not in base]
    if missing:
        raise AssertionError(f"Could not reconstruct canonical metadata fields: {missing}")
    return base[columns].copy()


def main() -> None:
    required = [
        CANONICAL / "input" / "raw.npy",
        CANONICAL / "input" / "locked_metadata.csv.gz",
        CANONICAL / "input" / "markers.txt",
        MEYER_RDS,
        EXPORTER,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    CASE.mkdir(parents=True, exist_ok=True)
    INPUT.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)

    markers = [
        value.strip()
        for value in (CANONICAL / "input" / "markers.txt").read_text().splitlines()
        if value.strip()
    ]
    if markers != MARKERS:
        raise AssertionError(f"Canonical marker contract changed: {markers}")
    canonical_raw = np.load(CANONICAL / "input" / "raw.npy", allow_pickle=False)
    canonical_metadata = pd.read_csv(
        CANONICAL / "input" / "locked_metadata.csv.gz", low_memory=False
    )
    if canonical_raw.shape != (len(canonical_metadata), len(MARKERS)):
        raise AssertionError("Canonical TNBC matrix/metadata shape changed")
    if list(pd.unique(canonical_metadata["dataset"].astype(str))) != DATASETS:
        raise AssertionError("Canonical TNBC dataset order changed")

    roundtrip = validate_provider_roundtrip(canonical_raw, canonical_metadata)

    seed_screen_record = None
    if SELECTION_MODE == "coverage_guided":
        if SEED_SCREEN_MANIFEST is None or not SEED_SCREEN_MANIFEST.is_file():
            raise FileNotFoundError("Coverage-guided mode requires a seed-screen manifest")
        seed_screen_record = json.loads(SEED_SCREEN_MANIFEST.read_text())
        if not seed_screen_record.get("not_an_unbiased_random_draw"):
            raise AssertionError("Seed screen is not explicitly marked post hoc")
        if int(seed_screen_record["winner"]["seed"]) != DONOR_SELECTION_SEED:
            raise AssertionError("Requested seed is not the locked seed-screen winner")
    elif SELECTION_MODE != "unbiased_single_draw":
        raise ValueError(f"Unknown selection mode: {SELECTION_MODE}")

    # Candidate export reads PID only.  This file freezes the complete post-QC
    # patient universe before any label or outcome field is used.
    run_export("candidates", AUDIT)
    candidates = pd.read_csv(
        AUDIT / "meyer_candidate_donor_counts.csv", dtype={"donor": str}
    ).sort_values("donor").reset_index(drop=True)
    if len(candidates) != 215 or int(candidates["cells"].sum()) != 928_993:
        raise AssertionError("Unexpected Meyer post-QC donor universe")
    rng = np.random.default_rng(DONOR_SELECTION_SEED)
    draw_order = rng.choice(
        candidates["donor"].to_numpy(dtype=object), DONORS_TO_SELECT, replace=False
    ).astype(str).tolist()
    selected_sorted = sorted(draw_order)
    selected = candidates[candidates["donor"].isin(selected_sorted)].copy()
    selected["draw_order"] = selected["donor"].map(
        {donor: order for order, donor in enumerate(draw_order, start=1)}
    )
    selected["selected_order_sorted"] = selected["donor"].map(
        {donor: order for order, donor in enumerate(selected_sorted, start=1)}
    )
    selected = selected.sort_values("draw_order")
    selected_path = AUDIT / "meyer_selected_donors.csv"
    selected.to_csv(selected_path, index=False)

    working = CASE / "working" / "new_meyer"
    if working.exists():
        shutil.rmtree(working)
    run_export("selected", working, selected_path)
    meyer_raw, exported = read_export(working)
    observed_donors = sorted(exported["donor"].astype(str).unique())
    if observed_donors != selected_sorted:
        raise AssertionError((observed_donors, selected_sorted))
    expected_selected_cells = int(selected["cells"].sum())
    if len(exported) != expected_selected_cells:
        raise AssertionError((len(exported), expected_selected_cells))

    old_dataset = canonical_metadata["dataset"].astype(str)
    tietscher_mask = old_dataset.eq("Tietscher IMC").to_numpy()
    shiao_mask = old_dataset.eq("Shiao CODEX").to_numpy()
    tietscher_raw = np.asarray(canonical_raw[tietscher_mask], dtype=np.float64)
    shiao_raw = np.asarray(canonical_raw[shiao_mask], dtype=np.float64)
    tietscher_metadata = canonical_metadata.loc[tietscher_mask].copy()
    shiao_metadata = canonical_metadata.loc[shiao_mask].copy()
    meyer_metadata = build_meyer_metadata(exported, canonical_metadata.columns.tolist())

    raw = np.vstack([tietscher_raw, meyer_raw, shiao_raw]).astype(np.float64, copy=False)
    metadata = pd.concat(
        [tietscher_metadata, meyer_metadata, shiao_metadata],
        ignore_index=True,
        sort=False,
    )
    metadata["cell_order"] = np.arange(len(metadata), dtype=np.int64)
    if raw.shape != (len(metadata), len(MARKERS)):
        raise AssertionError((raw.shape, len(metadata)))
    if metadata["cell_id"].astype(str).duplicated().any():
        raise AssertionError("Combined test cell IDs are duplicated")
    if list(pd.unique(metadata["dataset"].astype(str))) != DATASETS:
        raise AssertionError("Combined test dataset order changed")
    if not np.array_equal(raw[: len(tietscher_raw)], tietscher_raw):
        raise AssertionError("Tietscher values changed")
    if not np.array_equal(raw[-len(shiao_raw) :], shiao_raw):
        raise AssertionError("Shiao values changed")
    if not np.array_equal(
        metadata.iloc[: len(tietscher_metadata)]["cell_id"].astype(str).to_numpy(),
        tietscher_metadata["cell_id"].astype(str).to_numpy(),
    ):
        raise AssertionError("Tietscher identities/order changed")
    if not np.array_equal(
        metadata.iloc[-len(shiao_metadata) :]["cell_id"].astype(str).to_numpy(),
        shiao_metadata["cell_id"].astype(str).to_numpy(),
    ):
        raise AssertionError("Shiao identities/order changed")

    raw_path = INPUT / "raw.npy"
    metadata_path = INPUT / "metadata.csv.gz"
    fit_path = INPUT / "fit_metadata.csv.gz"
    markers_path = INPUT / "markers.txt"
    save_npy(raw, raw_path)
    write_csv_gz(metadata, metadata_path)
    write_csv_gz(metadata[["dataset", "unit_key"]], fit_path)
    markers_path.write_text("\n".join(MARKERS) + "\n", encoding="utf-8")

    donor_counts = (
        metadata.groupby(["dataset", "donor"], sort=False)
        .size().rename("cells").reset_index()
    )
    donor_counts.to_csv(AUDIT / "combined_donor_cell_counts.csv", index=False)
    celltype_counts = (
        metadata.groupby(["dataset", "celltype"], sort=False)
        .size().rename("cells").reset_index()
    )
    celltype_counts.to_csv(AUDIT / "combined_celltype_counts.csv", index=False)

    manifest = {
        "analysis": "TNBC Meyer five-patient resampling test",
        "writes_confined_to": str(CASE),
        "canonical_tietscher_unchanged": True,
        "canonical_shiao_unchanged": True,
        "meyer_source": str(MEYER_RDS),
        "meyer_source_sha256": sha256(MEYER_RDS),
        "meyer_source_assay": "counts",
        "external_preprocessing": False,
        "provider_roundtrip_against_current_25211_cells": roundtrip,
        "sampling": {
            "decision_frozen_before_labels_alignment_or_umap": True,
            "candidate_universe": "all 215 patients in provider post-QC sce_ALL.rds",
            "candidate_fields_used": ["PID", "source row identity"],
            "selection_mode": SELECTION_MODE,
            "clinical_filters": False,
            "provider_celltype_used_for_seed_selection": SELECTION_MODE == "coverage_guided",
            "rng": "numpy.random.default_rng",
            "seed": DONOR_SELECTION_SEED,
            "draw_order": draw_order,
            "selected_donors_sorted": selected_sorted,
            "all_cells_from_selected_donors_retained": True,
            "selected_meyer_cells": int(len(exported)),
            "reroll_after_results": False,
            "seed_screen_completed_before_alignment_or_umap": (
                SELECTION_MODE == "coverage_guided"
            ),
        },
        "datasets": DATASETS,
        "dataset_cells": metadata.groupby("dataset", sort=False).size().astype(int).to_dict(),
        "dataset_units": metadata.groupby("dataset", sort=False)["unit_key"].nunique().astype(int).to_dict(),
        "markers": MARKERS,
        "input_scale": "highest-available provider/source scale",
        "fit_metadata_columns": ["dataset", "unit_key"],
        "provider_labels_used_for_alignment": False,
        "provider_labels_used_for_sampling": SELECTION_MODE == "coverage_guided",
        "matrix_buffer_sha256": buffer_sha256(raw),
        "inputs_sha256": {
            "raw.npy": sha256(raw_path),
            "metadata.csv.gz": sha256(metadata_path),
            "fit_metadata.csv.gz": sha256(fit_path),
            "markers.txt": sha256(markers_path),
        },
        "canonical_sources_sha256": {
            "raw.npy": sha256(CANONICAL / "input" / "raw.npy"),
            "locked_metadata.csv.gz": sha256(
                CANONICAL / "input" / "locked_metadata.csv.gz"
            ),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "h5py": h5py.__version__,
        },
    }
    if seed_screen_record is not None:
        manifest["coverage_guided_seed_screen"] = {
            "path": str(SEED_SCREEN_MANIFEST),
            "sha256": sha256(SEED_SCREEN_MANIFEST),
            "selection_rule": seed_screen_record["selection_rule"],
            "seed_range_inclusive": seed_screen_record["seed_range_inclusive"],
            "seeds_scanned": seed_screen_record["seeds_scanned"],
            "winner": seed_screen_record["winner"],
        }
    write_json(manifest, INPUT / "input_manifest.json")

    # Keep only the compact sampled benchmark input; the temporary HDF5 copy is
    # redundant after raw.npy has been validated and published.
    shutil.rmtree(CASE / "working", ignore_errors=True)
    print(json.dumps({
        "selected_draw_order": draw_order,
        "selected_sorted": selected_sorted,
        "selected_cells": int(len(exported)),
        "dataset_cells": manifest["dataset_cells"],
        "output": str(INPUT),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
