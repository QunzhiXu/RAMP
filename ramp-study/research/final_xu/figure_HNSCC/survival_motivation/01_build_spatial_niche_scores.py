#!/usr/bin/env python3
"""Build PT-CD8 CAF/macrophage spatial scores directly from complete CODEX data.

This stage is deliberately outcome-blind. It reads only the complete CODEX
cell table copied under ``figure_HNSCC/sampled_data``. It does not read clinical
outcomes, RAMP output, transferred biomarkers, or any earlier result file.

For each provider-labelled CD8T cell i in a primary-tumor (PT) ROI r:

    local_i = (# CAF or macrophages among 15 nearest non-self cells) / 15
    background_r = (# CAF or macrophages in ROI r) / (N_r - 1)
    score_i = local_i - background_r

The patient score is the cell-weighted mean of score_i over every evaluable PT
CD8T cell in every PT ROI from that patient. No minimum CD8-cell threshold is
applied.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


HERE = Path(__file__).resolve().parent
COMPLETE_DATA = HERE.parent / "sampled_data" / "haist_codex" / "complete"
CELL_TABLE = COMPLETE_DATA / "HNSCCdiscovery_CODEX.csv"
OUT = HERE / "output"

TISSUE = "PT"
FOCAL_CELL_TYPE = "CD8T"
ARCHITECTURE_TYPES = frozenset({"CAF", "macrophages"})
K_NEIGHBORS = 15
CHUNK_SIZE = 150_000

USECOLS = [
    "UID",
    "unique_region",
    "x",
    "y",
    "Pat_ID",
    "Tissue_type_broad",
    "cell_types_mediumres",
    "neighb_name_new",
    "neighb_name_short",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv_gz(frame: pd.DataFrame, path: Path) -> None:
    """Write a deterministic gzip-compressed CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(fileobj=raw_handle, mode="wb", mtime=0) as handle:
            frame.to_csv(handle, index=False)


def process_roi(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]] | None:
    """Calculate cell-level scores and one summary row for a single ROI."""
    if frame.empty:
        return None
    if frame["unique_region"].nunique() != 1 or frame["Pat_ID"].nunique() != 1:
        raise AssertionError("An ROI block contains multiple ROI or patient identifiers")
    if not frame["Tissue_type_broad"].astype(str).eq(TISSUE).all():
        return None
    if len(frame) <= K_NEIGHBORS:
        return None

    cell_type = frame["cell_types_mediumres"].astype(str)
    focal_mask = cell_type.eq(FOCAL_CELL_TYPE).to_numpy()
    if not np.any(focal_mask):
        return None

    coordinates = frame[["x", "y"]].to_numpy(dtype=float)
    if not np.isfinite(coordinates).all():
        raise AssertionError(f"Non-finite spatial coordinate in ROI {frame['unique_region'].iloc[0]}")

    distances, neighbors = cKDTree(coordinates).query(
        coordinates[focal_mask], k=K_NEIGHBORS + 1, workers=1
    )
    if neighbors.ndim != 2 or neighbors.shape[1] != K_NEIGHBORS + 1:
        raise AssertionError("Unexpected nearest-neighbor result shape")
    if not np.allclose(distances[:, 0], 0.0, atol=1e-10):
        raise AssertionError("The focal CD8 cell was not the first neighbor")

    architecture = cell_type.isin(ARCHITECTURE_TYPES).to_numpy()
    local_count = architecture[neighbors[:, 1:]].sum(axis=1).astype(np.int16)
    local_fraction = local_count.astype(float) / K_NEIGHBORS

    # The focal cell is CD8T and therefore is not a CAF or macrophage. Removing
    # it leaves the architecture numerator unchanged while making the local and
    # background populations both explicitly non-self.
    background_fraction = float(architecture.sum() / (len(frame) - 1))
    residual = local_fraction - background_fraction

    keep_columns = [
        "UID",
        "unique_region",
        "Pat_ID",
        "x",
        "y",
        "neighb_name_new",
        "neighb_name_short",
    ]
    focal = frame.loc[focal_mask, keep_columns].copy()
    focal["local_caf_macrophage_count_k15"] = local_count
    focal["local_caf_macrophage_fraction_k15"] = local_fraction
    focal["roi_nonself_caf_macrophage_fraction"] = background_fraction
    focal["roi_adjusted_caf_macrophage_score"] = residual
    focal["in_caf_macrophage_rich_niche"] = residual > 0

    summary = {
        "CaseID": int(frame["Pat_ID"].iloc[0]),
        "unique_region": str(frame["unique_region"].iloc[0]),
        "total_cells": int(len(frame)),
        "pt_cd8_cells": int(focal_mask.sum()),
        "caf_cells": int(cell_type.eq("CAF").sum()),
        "macrophage_cells": int(cell_type.eq("macrophages").sum()),
        "roi_nonself_caf_macrophage_fraction": background_fraction,
        "mean_local_caf_macrophage_fraction_k15": float(local_fraction.mean()),
        "mean_roi_adjusted_caf_macrophage_score": float(residual.mean()),
        "median_roi_adjusted_caf_macrophage_score": float(np.median(residual)),
        "fraction_cd8_in_caf_macrophage_rich_niche": float(np.mean(residual > 0)),
    }
    return focal, summary


def stream_rois() -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Stream the complete table while preserving ROI boundaries."""
    cell_blocks: list[pd.DataFrame] = []
    roi_rows: list[dict[str, object]] = []
    carry: pd.DataFrame | None = None
    processed: set[str] = set()
    all_roi_count = 0

    def consume(frame: pd.DataFrame) -> None:
        nonlocal all_roi_count
        roi = str(frame["unique_region"].iloc[0])
        if roi in processed:
            raise AssertionError(f"ROI rows are not contiguous in the complete table: {roi}")
        processed.add(roi)
        all_roi_count += 1
        result = process_roi(frame)
        if result is not None:
            cells, summary = result
            cell_blocks.append(cells)
            roi_rows.append(summary)

    for chunk in pd.read_csv(CELL_TABLE, usecols=USECOLS, chunksize=CHUNK_SIZE):
        chunk["UID"] = chunk["UID"].astype(str)
        if carry is not None:
            chunk = pd.concat([carry, chunk], ignore_index=True)

        last_roi = str(chunk["unique_region"].iloc[-1])
        is_last_roi = chunk["unique_region"].astype(str).eq(last_roi)
        complete = chunk.loc[~is_last_roi]
        carry = chunk.loc[is_last_roi].copy()
        for _, frame in complete.groupby("unique_region", sort=False):
            consume(frame)

    if carry is not None and not carry.empty:
        consume(carry)
    if not cell_blocks:
        raise RuntimeError("No evaluable PT CD8T cells were found")

    cells = pd.concat(cell_blocks, ignore_index=True)
    rois = pd.DataFrame(roi_rows).sort_values(["CaseID", "unique_region"]).reset_index(drop=True)
    if cells["UID"].duplicated().any():
        raise AssertionError("Duplicate UID in the PT-CD8 output")
    return cells, rois, all_roi_count


def build_patient_scores(cells: pd.DataFrame, rois: pd.DataFrame) -> pd.DataFrame:
    """Aggregate all PT CD8 cells with equal cell weight within each patient."""
    patient = (
        cells.groupby("Pat_ID", as_index=False)
        .agg(
            pt_cd8_cells=("UID", "size"),
            evaluable_pt_rois=("unique_region", "nunique"),
            patient_spatial_niche_score=("roi_adjusted_caf_macrophage_score", "mean"),
            median_cell_spatial_niche_score=("roi_adjusted_caf_macrophage_score", "median"),
            mean_local_caf_macrophage_fraction=("local_caf_macrophage_fraction_k15", "mean"),
            cell_weighted_roi_background=("roi_nonself_caf_macrophage_fraction", "mean"),
            fraction_cd8_in_caf_macrophage_rich_niche=(
                "in_caf_macrophage_rich_niche",
                "mean",
            ),
        )
        .rename(columns={"Pat_ID": "CaseID"})
    )
    patient["CaseID"] = patient["CaseID"].astype(int)

    totals = (
        rois.groupby("CaseID", as_index=False)
        .agg(
            total_pt_cells=("total_cells", "sum"),
            total_caf_cells=("caf_cells", "sum"),
            total_macrophage_cells=("macrophage_cells", "sum"),
        )
    )
    patient = patient.merge(totals, on="CaseID", validate="one_to_one")
    patient["pooled_pt_caf_macrophage_fraction"] = (
        patient["total_caf_cells"] + patient["total_macrophage_cells"]
    ) / patient["total_pt_cells"]
    return patient.sort_values("CaseID").reset_index(drop=True)


def main() -> None:
    if not CELL_TABLE.is_file():
        raise FileNotFoundError(CELL_TABLE)
    OUT.mkdir(parents=True, exist_ok=True)

    cells, rois, all_roi_count = stream_rois()
    patients = build_patient_scores(cells, rois)

    cell_output = OUT / "pt_cd8_cell_spatial_niche_scores.csv.gz"
    roi_output = OUT / "pt_roi_spatial_niche_scores.csv"
    patient_output = OUT / "patient_spatial_niche_scores.csv"
    write_csv_gz(cells, cell_output)
    rois.to_csv(roi_output, index=False)
    patients.to_csv(patient_output, index=False)

    manifest = {
        "stage": "outcome-blind spatial-score construction",
        "input_cell_table": str(CELL_TABLE),
        "input_cell_table_sha256": sha256_file(CELL_TABLE),
        "other_inputs_read": [],
        "tissue": TISSUE,
        "focal_cell_type": FOCAL_CELL_TYPE,
        "architecture_cell_types": sorted(ARCHITECTURE_TYPES),
        "k_nearest_nonself_cells": K_NEIGHBORS,
        "cell_score": (
            "local k15 CAF/macrophage fraction minus same-ROI nonself "
            "CAF/macrophage fraction"
        ),
        "patient_score": "cell-weighted mean cell score across all evaluable PT CD8T cells",
        "minimum_pt_cd8_cell_threshold": None,
        "all_rois_scanned": int(all_roi_count),
        "evaluable_pt_rois": int(len(rois)),
        "patients_with_evaluable_pt_cd8": int(len(patients)),
        "pt_cd8_cells": int(len(cells)),
        "outputs": {
            "cell": str(cell_output),
            "roi": str(roi_output),
            "patient": str(patient_output),
        },
        "ramp_used": False,
        "molecular_markers_used": False,
        "clinical_or_survival_columns_read": False,
    }
    (OUT / "spatial_score_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
