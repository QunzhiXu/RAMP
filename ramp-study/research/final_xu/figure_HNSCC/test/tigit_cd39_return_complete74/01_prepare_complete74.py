#!/usr/bin/env python3
"""Prepare the complete-OS HNSCC PT74 RAMP input from copied complete data.

All 74 Haist patients with complete overall-survival information are retained;
there is no minimum PT-CD8 filter.  Exactly 1,000 provider-defined primary-
tumour (PT) CODEX cells are sampled per patient for alignment, and all 20,488
Rahim CITE-seq cells are retained.  Survival outcomes are not written to the
alignment metadata.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse


HERE = Path(__file__).resolve().parent
FIGURE_HNSCC = HERE.parents[1]
INPUT = HERE / "input"
HAIST_CELLS = FIGURE_HNSCC / "sampled_data/haist_codex/complete/HNSCCdiscovery_CODEX.csv"
RAHIM_ADT = FIGURE_HNSCC / "sampled_data/rahim_citeseq/complete/GSE212797_adata_ADT.h5ad"
RAHIM_RNA = FIGURE_HNSCC / "sampled_data/rahim_citeseq/complete/GSE212797_adata.h5ad"
SURVIVAL_INPUT = FIGURE_HNSCC / "survival_motivation/output/survival_analysis_input.csv"

SAMPLE_SEED = 20260823
CELLS_PER_HAIST_PATIENT = 1_000

MARKER_MAP = {
    "CD45": ("CD45", "CD45"),
    "CD3": ("CD3", "CD3"),
    "CD7": ("CD7-1", "CD7"),
    "CD4": ("CD4-1", "CD4"),
    "CD8": ("CD8", "CD8"),
    "CD25": ("CD25", "CD25"),
    "CD20": ("CD20", "CD20"),
    "CD38": ("CD38-1", "CD38"),
    "CD56": ("CD56--NCAM", "CD56"),
    "CD57": ("CD57", "CD57"),
    "CD16": ("CD16", "CD16"),
    "CD11c": ("CD11c", "CD11c"),
    "CD11b": ("CD11b", "CD11b"),
    "HLA-DR": ("HLA-DR", "HLA-DR"),
    "CD21": ("CD21", "CD21"),
    "CD31": ("CD31", "CD31"),
    "CD69": ("CD69-1", "CD69"),
    "ICOS": ("CD278--ICOS", "ICOS"),
    "PD1": ("CD279--PD-1", "PD1"),
    "CD45RO": ("CD45RO", "CD45RO"),
    "CD45RA": ("CD45RA", "CD45RA"),
    "PD-L1": ("CD274--B7-H1--PD-L1", "PD-L1"),
}
MARKERS = list(MARKER_MAP)

RAHIM_DISPLAY = {
    "CD4 T cell": "CD4 T", "CD8 T cell": "CD8 T", "B cell": "B",
    "Plasma cell": "Plasma", "NK/gd T cell": "NK / gdT",
    "Macrophage": "Myeloid", "cDC": "Myeloid", "pDC": "Myeloid",
    "Mast cell": "Myeloid", "Fibroblast": "Stromal",
    "Endothelial cell": "Vascular", "Salivary gland": "Epithelial / tumor",
    "Proliferating cell": "Other",
}
HAIST_DISPLAY = {
    "CD4T": "CD4 T", "Treg": "CD4 T", "TFH": "CD4 T", "CD8T": "CD8 T",
    "B": "B", "plasma": "Plasma", "NK": "NK / gdT", "gdT": "NK / gdT",
    "macrophages": "Myeloid", "DC": "Myeloid", "granulocyte": "Myeloid",
    "mast": "Myeloid", "stromal": "Stromal", "CAF": "Stromal",
    "VSMC_myofibroblast": "Stromal", "FDC": "Stromal",
    "vasculature": "Vascular", "LEC": "Vascular", "tumor": "Epithelial / tumor",
    "tumor_CA9": "Epithelial / tumor", "nerve": "Other",
}


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode() if isinstance(value, bytes) else str(value) for value in values],
        dtype=object,
    )


def categorical(handle: h5py.File, key: str) -> np.ndarray:
    levels = decode(handle[f"obs/__categories/{key}"][:])
    return levels[handle[f"obs/{key}"][:]]


def read_csr(handle: h5py.File) -> np.ndarray:
    group = handle["X"]
    matrix = sparse.csr_matrix(
        (group["data"][:], group["indices"][:], group["indptr"][:]),
        shape=(len(group["indptr"]) - 1, len(handle["var/_index"])),
    )
    return matrix.toarray().astype(np.float64, copy=False)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def selected_patients() -> pd.DataFrame:
    survival = pd.read_csv(
        SURVIVAL_INPUT, usecols=["CaseID", "pt_cd8_cells", "OS", "OS_status"]
    )
    survival["CaseID"] = pd.to_numeric(survival.CaseID, errors="raise").astype(int)
    survival["pt_cd8_cells"] = pd.to_numeric(
        survival.pt_cd8_cells, errors="raise"
    ).astype(int)
    if survival[["OS", "OS_status"]].isna().any().any():
        raise AssertionError("Complete-cohort input contains a missing OS outcome")
    if len(survival) != 74 or survival.CaseID.nunique() != 74:
        raise AssertionError(f"Expected 74 complete-OS patients, observed {len(survival)}")
    if int((survival.OS_status == "OS+").sum()) != 24:
        raise AssertionError("Expected 24 deaths in the complete-OS cohort")
    selected = survival[["CaseID", "pt_cd8_cells"]].sort_values("CaseID").reset_index(drop=True)
    if int(selected.pt_cd8_cells.min()) != 3 or int(selected.pt_cd8_cells.sum()) != 38_299:
        raise AssertionError("Complete-cohort PT-CD8 counts changed unexpectedly")
    return selected


def build_rahim() -> tuple[np.ndarray, pd.DataFrame]:
    with h5py.File(RAHIM_ADT, "r") as handle:
        cell_ids = decode(handle["obs/_index"][:])
        donors = categorical(handle, "patient")
        tissues = categorical(handle, "tissue")
        marker_names = decode(handle["var/_index"][:])
        counts = read_csr(handle)
    with h5py.File(RAHIM_RNA, "r") as handle:
        rna_ids = decode(handle["obs/_index"][:])
        rna_types = categorical(handle, "cell_type")

    type_lookup = dict(zip(rna_ids, rna_types))
    provider_types = np.asarray([type_lookup[cell_id] for cell_id in cell_ids], dtype=object)
    marker_index = {marker: index for index, marker in enumerate(marker_names)}
    values = np.column_stack(
        [counts[:, marker_index[MARKER_MAP[marker][0]]] for marker in MARKERS]
    ).astype(np.float64, copy=False)
    metadata = pd.DataFrame(
        {
            "cell_uid": [f"RAHIM|{cell_id}" for cell_id in cell_ids],
            "dataset": "GSE212797_ADT",
            "donor": donors.astype(str),
            "unit_key": "GSE212797_" + pd.Series(donors).astype(str),
            "tissue": tissues.astype(str),
            "provider_cell_type": provider_types.astype(str),
        }
    )
    metadata["display_label"] = metadata.provider_cell_type.map(RAHIM_DISPLAY)
    if len(metadata) != 20_488 or metadata.donor.nunique() != 3:
        raise AssertionError("Rahim input dimensions changed")
    if metadata.display_label.isna().any():
        missing = sorted(metadata.loc[metadata.display_label.isna(), "provider_cell_type"].unique())
        raise AssertionError(f"Unmapped Rahim labels: {missing}")
    return values, metadata


def build_haist(patients: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    eligible = set(patients.CaseID.astype(int))
    usecols = list(
        dict.fromkeys(
            [haist_name for _, haist_name in MARKER_MAP.values()]
            + ["UID", "Pat_ID", "Tissue_type_broad", "unique_region", "cell_types_mediumres"]
        )
    )
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(HAIST_CELLS, usecols=usecols, chunksize=150_000):
        patient_ids = pd.to_numeric(chunk.Pat_ID, errors="raise").astype(int)
        keep = patient_ids.isin(eligible) & chunk.Tissue_type_broad.astype(str).eq("PT")
        if keep.any():
            block = chunk.loc[keep].copy()
            block["Pat_ID"] = patient_ids.loc[keep].to_numpy(int)
            pieces.append(block)
    pt = pd.concat(pieces, ignore_index=True)
    pt["UID"] = pt.UID.astype(str)
    pt = pt.sort_values(["Pat_ID", "UID"], kind="stable").reset_index(drop=True)

    available = (
        pt.groupby("Pat_ID", sort=True)
        .agg(
            available_pt_cells=("UID", "size"),
            available_pt_cd8=("cell_types_mediumres", lambda x: int(x.astype(str).eq("CD8T").sum())),
            pt_rois=("unique_region", "nunique"),
        )
        .reset_index()
    )
    if set(available.Pat_ID) != eligible:
        raise AssertionError("At least one selected patient has no PT cells")
    if (available.available_pt_cells < CELLS_PER_HAIST_PATIENT).any():
        raise AssertionError("At least one patient has fewer than 1,000 PT cells")
    expected_cd8 = patients.set_index("CaseID").pt_cd8_cells.astype(int).sort_index()
    observed_cd8 = available.set_index("Pat_ID").available_pt_cd8.astype(int).sort_index()
    if not expected_cd8.equals(observed_cd8):
        raise AssertionError("PT-CD8 counts disagree with the complete-cohort table")

    rng = np.random.default_rng(SAMPLE_SEED)
    sampled_blocks = []
    for patient in sorted(eligible):
        group = pt.loc[pt.Pat_ID.eq(patient)]
        chosen = np.sort(rng.choice(len(group), CELLS_PER_HAIST_PATIENT, replace=False))
        sampled_blocks.append(group.iloc[chosen].copy())
    sampled = pd.concat(sampled_blocks, ignore_index=True)
    values = np.column_stack(
        [pd.to_numeric(sampled[MARKER_MAP[marker][1]], errors="raise") for marker in MARKERS]
    ).astype(np.float64, copy=False)
    metadata = pd.DataFrame(
        {
            "cell_uid": "HAIST|" + sampled.UID.astype(str),
            "dataset": "Haist_CODEX",
            "donor": sampled.Pat_ID.astype(str),
            "unit_key": "HAIST_" + sampled.Pat_ID.astype(str),
            "tissue": sampled.Tissue_type_broad.astype(str),
            "provider_cell_type": sampled.cell_types_mediumres.astype(str),
            "roi": sampled.unique_region.astype(str),
        }
    )
    metadata["display_label"] = metadata.provider_cell_type.map(HAIST_DISPLAY)
    if len(metadata) != 74_000 or metadata.donor.nunique() != 74:
        raise AssertionError("Haist sample is not exactly 74 patients x 1,000 PT cells")
    if set(metadata.tissue) != {"PT"}:
        raise AssertionError("Non-PT Haist cells entered the test")
    if metadata.display_label.isna().any():
        missing = sorted(metadata.loc[metadata.display_label.isna(), "provider_cell_type"].unique())
        raise AssertionError(f"Unmapped Haist labels: {missing}")
    available["sampled_pt_cells"] = CELLS_PER_HAIST_PATIENT
    return values, metadata, available


def main() -> None:
    INPUT.mkdir(parents=True, exist_ok=True)
    patients = selected_patients()
    rahim_values, rahim_metadata = build_rahim()
    haist_values, haist_metadata, haist_counts = build_haist(patients)
    raw = np.vstack([rahim_values, haist_values])
    metadata = pd.concat([rahim_metadata, haist_metadata], ignore_index=True)
    if raw.shape != (94_488, len(MARKERS)):
        raise AssertionError(f"Unexpected combined shape: {raw.shape}")
    if not np.isfinite(raw).all() or not metadata.cell_uid.is_unique:
        raise AssertionError("Combined input is non-finite or has duplicated cell IDs")

    raw_path = INPUT / "raw_shared_markers.float64.npy"
    metadata_path = INPUT / "cell_metadata.csv.gz"
    np.save(raw_path, raw)
    metadata.to_csv(metadata_path, index=False, compression="gzip")
    patients.to_csv(INPUT / "selected_74_patients.csv", index=False)
    haist_counts.to_csv(INPUT / "haist_pt_sampling_counts.csv", index=False)
    (INPUT / "markers.txt").write_text("\n".join(MARKERS) + "\n")
    manifest = {
        "purpose": "Complete-OS PT74 HNSCC RAMP input for locked TIGIT/CD39 stability test",
        "selection": {
            "Haist": "all 74 patients with complete OS; no minimum PT-CD8 threshold; all provider PT ROIs pooled per patient",
            "Rahim": "all cells from all three provider donors and both provider tissues",
        },
        "sampling": {
            "seed": SAMPLE_SEED,
            "Haist_cells_per_patient": CELLS_PER_HAIST_PATIENT,
            "scope": "within patient across all provider-defined PT ROIs",
            "cell_type_balancing": False,
            "ROI_selection": False,
        },
        "datasets": {
            "GSE212797_ADT": {"cells": 20_488, "units": 3},
            "Haist_CODEX": {"cells": 74_000, "units": 74},
        },
        "combined_shape": list(raw.shape),
        "shared_markers": MARKERS,
        "input_scale": "highest-available source-scale raw shared-marker values",
        "alignment_metadata": ["dataset", "unit_key"],
        "outcomes_written_to_alignment_input": False,
        "sha256": {
            "raw_shared_markers": sha256(raw_path),
            "cell_metadata": sha256(metadata_path),
        },
    }
    (INPUT / "input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
