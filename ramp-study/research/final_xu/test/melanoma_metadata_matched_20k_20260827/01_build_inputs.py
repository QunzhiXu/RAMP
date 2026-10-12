#!/usr/bin/env python3
"""Build a label/outcome-blind, metadata-defined melanoma alignment test.

The output contains 20,000 cells from each of Hickey CODEX, Hoch IMC, and
Ptacek MIBI.  Sampling uses only patient/sample metadata and an ordinary fixed
NumPy random seed.  Provider cell labels are attached only after rows have
been selected so they can be used for post-hoc UMAP coloring.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import platform
import sys
from pathlib import Path


sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
FINAL_XU = HERE.parents[1]
RAW_ROOT = FINAL_XU / "raw_data" / "melanoma"
OUT = HERE / "input"
ROOT_SEED = 20_260_827
TARGET_PER_DATASET = 20_000
CHUNK_SIZE = 400_000

DATASETS = (
    "MEL_Hickey2023_CODEX",
    "MEL_Hoch2021_IMC",
    "MEL_Ptacek2021_MIBI",
)
MARKERS = (
    "CD163", "CD3", "CD31", "CD68", "CD8", "HLA_DR",
    "LAG3", "PD1", "PDL1", "SMA", "SOX10", "VIMENTIN",
)
MARKER_MAP = {
    DATASETS[0]: dict(zip(MARKERS, (
        "CD163", "CD3", "CD31", "CD68", "CD8", "HLA-DR",
        "LAG3", "PD-1", "PD-L1", "aSMA", "SOX10/S100", "Vimentin",
    ))),
    DATASETS[1]: dict(zip(MARKERS, (
        "CD163", "CD3e", "CD31", "CD68", "CD8a", "HLA_DR",
        "LAG3", "PD_1", "PD_L1", "SMA", "SOX10", "VIM",
    ))),
    DATASETS[2]: dict(zip(MARKERS, (
        "CD163", "CD3", "CD31", "CD68", "CD8", "HLA DPDQDR",
        "LAG3", "PD-1", "PD-L1", "SMA", "SOX10", "Vimentin",
    ))),
}

FINE_LABEL_ORDER = (
    "Non-joint", "Epithelial (non-joint)", "CD4 T", "CD8 T",
    "Other T / CD8−", "NK", "Endothelial", "Myeloid", "Stroma", "Tumor",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv_gz(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="", fileobj=raw_handle, mode="wb", compresslevel=6, mtime=0
        ) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                frame.to_csv(text, index=False)


def source_record(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def true_mask(series: pd.Series) -> np.ndarray:
    return series.astype(str).str.lower().isin({"true", "1", "1.0"}).to_numpy()


def waterfill_quotas(
    capacities: dict[str, int], total: int, rng: np.random.Generator
) -> dict[str, int]:
    """Allocate as evenly as possible without exceeding any group capacity."""
    capacities = {str(key): int(value) for key, value in capacities.items()}
    if total < 0 or total > sum(capacities.values()):
        raise ValueError((total, capacities))
    keys = sorted(capacities)
    low, high = 0, max(capacities.values(), default=0)
    while low < high:
        middle = (low + high + 1) // 2
        if sum(min(capacities[key], middle) for key in keys) <= total:
            low = middle
        else:
            high = middle - 1
    quotas = {key: min(capacities[key], low) for key in keys}
    remaining = total - sum(quotas.values())
    candidates = [key for key in keys if quotas[key] < capacities[key]]
    if remaining > len(candidates):
        raise AssertionError("water-fill remainder is inconsistent")
    if remaining:
        chosen = rng.permutation(candidates)[:remaining]
        for key in chosen:
            quotas[str(key)] += 1
    if sum(quotas.values()) != total:
        raise AssertionError("water-fill did not reach requested total")
    return quotas


def proportional_quotas(
    capacities: dict[tuple[str, str, str], int],
    total: int,
    rng: np.random.Generator,
) -> dict[tuple[str, str, str], int]:
    """Largest-remainder allocation preserving acquisition-unit proportions."""
    keys = sorted(capacities, key=lambda value: tuple(map(str, value)))
    denominator = float(sum(capacities.values()))
    ideal = {key: capacities[key] * total / denominator for key in keys}
    quotas = {key: int(np.floor(ideal[key])) for key in keys}
    remaining = total - sum(quotas.values())
    tie_break = {key: float(value) for key, value in zip(keys, rng.random(len(keys)))}
    order = sorted(
        keys,
        key=lambda key: (-(ideal[key] - quotas[key]), tie_break[key]),
    )
    for key in order:
        if remaining == 0:
            break
        if quotas[key] < capacities[key]:
            quotas[key] += 1
            remaining -= 1
    if remaining or sum(quotas.values()) != total:
        raise AssertionError("proportional allocation failed")
    return quotas


def relabel_hickey(source: pd.Series) -> np.ndarray:
    values = source.fillna("").astype(str).to_numpy(dtype=str)
    result = np.full(len(values), "Non-joint", dtype=object)
    result[np.isin(values, ["CD4+ T", "CD4+ Treg"])] = "CD4 T"
    result[np.char.find(values, "CD8") >= 0] = "CD8 T"
    result[values == "Endothelial"] = "Endothelial"
    result[np.isin(values, [
        "DC", "CD163+ CD206+ Macrophage", "CD68+ Macrophage", "Neutrophil",
    ])] = "Myeloid"
    result[np.isin(values, [
        "Stromal", "CD73+ Stromal", "FAP+ Fibroblast", "PDPN+ Stromal",
        "Smooth Muscle",
    ])] = "Stroma"
    result[np.char.find(values, "Tumor") >= 0] = "Tumor"
    result[values == "Epithelial"] = "Epithelial (non-joint)"
    result[values == "NK"] = "NK"
    return result.astype(str)


def relabel_hoch(source: pd.Series) -> np.ndarray:
    mapping = {
        "CD8+ T cell": "CD8 T",
        "CD8- T cell": "Other T / CD8−",
        "Vasculature": "Endothelial",
        "Macrophage": "Myeloid",
        "Neutrophil": "Myeloid",
        "Stroma": "Stroma",
        "Tumor": "Tumor",
    }
    return source.fillna("").astype(str).map(mapping).fillna("Non-joint").to_numpy(str)


def derive_joint_labels(fine: np.ndarray) -> np.ndarray:
    result = np.full(len(fine), "Non-joint", dtype=object)
    result[np.isin(fine, ["CD4 T", "Other T / CD8−"])] = "CD8− T"
    result[fine == "CD8 T"] = "CD8 T"
    for label in ("Endothelial", "Myeloid", "Stroma", "Tumor"):
        result[fine == label] = label
    return result.astype(str)


def finalize_metadata(frame: pd.DataFrame, fine: np.ndarray) -> pd.DataFrame:
    answer = frame.copy().reset_index(drop=True)
    answer["celltype"] = np.asarray(fine, dtype=str)
    answer["celltype_joint_evaluation"] = derive_joint_labels(answer["celltype"].to_numpy(str))
    answer["joint_evaluation"] = answer["celltype_joint_evaluation"].ne("Non-joint")
    answer["is_nonjoint"] = answer["celltype"].isin(
        ["Non-joint", "Epithelial (non-joint)"]
    )
    if not set(answer["celltype"]).issubset(FINE_LABEL_ORDER):
        raise AssertionError("unexpected post-hoc cell label")
    return answer


def build_hickey(
    allocation_rng: np.random.Generator,
    selection_rng: np.random.Generator,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    dataset = DATASETS[0]
    folder = RAW_ROOT / "hickey_codex"
    raw_path = folder / "23_10_11_Melanoma_Marker_Cell_Neighborhood.csv"
    clinical_path = folder / "metadata.csv"
    clinical = pd.read_csv(clinical_path, dtype={"donor": "string"}, low_memory=False)
    pre = clinical.loc[clinical["Therapy"].astype(str).eq("Pre")].copy()
    if len(pre) != 6 or pre["Pat-ID"].nunique() != 6:
        raise AssertionError("Hickey pretreatment cohort is not six paired-study patients")
    specimen_to_patient = dict(zip(pre["donor"].astype(str), pre["Pat-ID"].astype(str)))
    selected_specimens = set(specimen_to_patient)

    counts: dict[tuple[str, str, str], int] = {}
    print("Hickey pass 1/2: counting pretreatment acquisition strata", flush=True)
    for chunk in pd.read_csv(
        raw_path,
        usecols=["donor", "filename", "region"],
        dtype={"donor": "string", "filename": "string", "region": "string"},
        chunksize=CHUNK_SIZE,
    ):
        chunk = chunk.loc[chunk["donor"].astype(str).isin(selected_specimens)]
        grouped = chunk.groupby(["donor", "filename", "region"], sort=False, dropna=False).size()
        for key, value in grouped.items():
            canonical_key = tuple(map(str, key))
            counts[canonical_key] = counts.get(canonical_key, 0) + int(value)
    patient_capacities = {
        patient: sum(value for key, value in counts.items() if specimen_to_patient[key[0]] == patient)
        for patient in sorted(set(specimen_to_patient.values()))
    }
    patient_quotas = waterfill_quotas(
        patient_capacities, TARGET_PER_DATASET, allocation_rng
    )
    stratum_quotas: dict[tuple[str, str, str], int] = {}
    for patient in sorted(patient_quotas):
        local = {
            key: value
            for key, value in counts.items()
            if specimen_to_patient[key[0]] == patient
        }
        stratum_quotas.update(
            proportional_quotas(local, patient_quotas[patient], allocation_rng)
        )
    selected_ordinals = {
        key: np.sort(selection_rng.choice(counts[key], quota, replace=False))
        for key, quota in stratum_quotas.items()
        if quota > 0
    }

    source_columns = list(MARKER_MAP[dataset].values()) + [
        "Unnamed: 0", "cellid", "donor", "filename", "region", "x", "y",
        "Cell_Type", "Cell_Type_Common", "Overall_Cell_Type",
    ]
    source_columns = list(dict.fromkeys(source_columns))
    seen = {key: 0 for key in counts}
    selected_parts: list[pd.DataFrame] = []
    source_offset = 0
    print("Hickey pass 2/2: reading selected source rows", flush=True)
    for chunk in pd.read_csv(
        raw_path,
        usecols=source_columns,
        dtype={"donor": "string", "filename": "string", "region": "string"},
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        local_positions: list[int] = []
        eligible = chunk["donor"].astype(str).isin(selected_specimens)
        if eligible.any():
            groups = chunk.loc[eligible].groupby(
                ["donor", "filename", "region"], sort=False, dropna=False
            ).indices
            eligible_positions = np.flatnonzero(eligible.to_numpy())
            # groupby.indices is relative to chunk.loc[eligible], not the original chunk.
            for raw_key, relative in groups.items():
                key = tuple(map(str, raw_key))
                positions = eligible_positions[np.asarray(relative, dtype=int)]
                start = seen[key]
                stop = start + len(positions)
                targets = selected_ordinals.get(key, np.empty(0, dtype=int))
                left = int(np.searchsorted(targets, start, side="left"))
                right = int(np.searchsorted(targets, stop, side="left"))
                if right > left:
                    local_positions.extend(
                        positions[(targets[left:right] - start).astype(int)].tolist()
                    )
                seen[key] = stop
        if local_positions:
            local_positions = sorted(local_positions)
            part = chunk.iloc[local_positions].copy()
            part["_source_index_zero_based"] = source_offset + np.asarray(local_positions)
            selected_parts.append(part)
        source_offset += len(chunk)
    if seen != counts:
        raise AssertionError("Hickey two-pass stratum counts changed between reads")
    chosen = pd.concat(selected_parts, ignore_index=True)
    if len(chosen) != TARGET_PER_DATASET:
        raise AssertionError(f"Hickey selected {len(chosen):,} cells")
    chosen = chosen.merge(pre, on="donor", how="left", validate="many_to_one")
    fine = relabel_hickey(chosen["Cell_Type"])
    metadata = pd.DataFrame({
        "source_cell_id": "hickey::" + chosen["Unnamed: 0"].astype(str),
        "source_index_zero_based": chosen["_source_index_zero_based"].astype(np.int64),
        "dataset": dataset,
        "technology": "CODEX",
        "patient_id": chosen["Pat-ID"].astype(str),
        "specimen_id": chosen["donor"].astype(str),
        "acquisition_id": chosen["filename"].astype(str) + "::region_" + chosen["region"].astype(str),
        "unit_key": dataset + "|patient_" + chosen["Pat-ID"].astype(str),
        "timepoint": chosen["Therapy"].astype(str),
        "stage": chosen["AJCC_stage_ED"].astype(str),
        "provider_celltype_original": chosen["Cell_Type"].astype(str),
        "provider_celltype_common": chosen["Cell_Type_Common"].astype(str),
        "provider_celltype_coarse": chosen["Overall_Cell_Type"].astype(str),
    })
    metadata = finalize_metadata(metadata, fine)
    raw = chosen[[MARKER_MAP[dataset][marker] for marker in MARKERS]].to_numpy(np.float32)
    audit_rows = []
    for key in sorted(counts):
        patient = specimen_to_patient[key[0]]
        audit_rows.append({
            "dataset": dataset,
            "patient_id": patient,
            "subunit": f"{key[1]}::region_{key[2]}",
            "eligible_cells": counts[key],
            "selected_cells": stratum_quotas[key],
            "patient_selected_cells": patient_quotas[patient],
        })
    return raw, metadata, pd.DataFrame(audit_rows), [
        source_record(raw_path), source_record(clinical_path)
    ]


def build_hoch(
    allocation_rng: np.random.Generator,
    selection_rng: np.random.Generator,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    dataset = DATASETS[1]
    folder = RAW_ROOT / "hoch_imc"
    metadata_path = folder / "HochSchulz_2022_Melanoma_metadata.csv"
    expression_path = folder / "HochSchulz_2022_Melanoma_exprs_cells_by_markers.csv"
    metadata_columns = [
        "cell_id_export", "patient_id", "image_number", "sample_block_id",
        "sample_location", "sample_MM_location", "sample_tissue_type",
        "patient_cancer_stage", "patient_treatment_status_before_surgery",
        "includeImage", "cell_type", "cell_type_rf", "cell_x", "cell_y",
    ]
    source = pd.read_csv(
        metadata_path,
        usecols=metadata_columns,
        dtype={
            "cell_id_export": "string", "patient_id": "string",
            "image_number": "string", "sample_block_id": "string",
        },
        low_memory=False,
    )
    if source["cell_id_export"].isna().any() or source["cell_id_export"].duplicated().any():
        raise AssertionError("Hoch metadata cell_id_export is not unique")
    eligible = (
        source["sample_tissue_type"].astype(str).eq("Melanoma")
        & source["patient_cancer_stage"].astype(str).eq("III")
        & source["patient_treatment_status_before_surgery"].astype(str).eq("naive")
    )
    cohort = source.loc[eligible].copy()
    if len(cohort) != 145_759 or cohort["patient_id"].nunique() != 17 or cohort["image_number"].nunique() != 23:
        raise AssertionError("Hoch metadata-defined cohort no longer matches 145,759/17/23")
    if not true_mask(cohort["includeImage"]).all():
        raise AssertionError("Hoch selected cohort contains includeImage=False")
    patient_counts = cohort.groupby("patient_id", sort=True).size().astype(int).to_dict()
    patient_quotas = waterfill_quotas(patient_counts, TARGET_PER_DATASET, allocation_rng)
    chosen_indices: list[int] = []
    audit_rows: list[dict[str, object]] = []
    for patient in sorted(patient_quotas):
        patient_rows = cohort.loc[cohort["patient_id"].astype(str).eq(patient)]
        image_counts = patient_rows.groupby("image_number", sort=True).size().astype(int).to_dict()
        image_quotas = waterfill_quotas(image_counts, patient_quotas[patient], allocation_rng)
        for image in sorted(image_quotas):
            candidates = patient_rows.index[
                patient_rows["image_number"].astype(str).eq(image)
            ].to_numpy(int)
            selected = selection_rng.choice(candidates, image_quotas[image], replace=False)
            chosen_indices.extend(selected.tolist())
            audit_rows.append({
                "dataset": dataset,
                "patient_id": patient,
                "subunit": f"image_{image}",
                "eligible_cells": len(candidates),
                "selected_cells": image_quotas[image],
                "patient_selected_cells": patient_quotas[patient],
            })
    chosen_indices = sorted(chosen_indices)
    chosen_meta = source.loc[chosen_indices].copy()
    chosen_meta["_source_index_zero_based"] = np.asarray(chosen_indices, dtype=np.int64)
    expression_columns = ["cell_id_export"] + [MARKER_MAP[dataset][marker] for marker in MARKERS]
    expression = pd.read_csv(
        expression_path,
        usecols=expression_columns,
        dtype={"cell_id_export": "string"},
        low_memory=False,
    )
    if expression["cell_id_export"].isna().any() or expression["cell_id_export"].duplicated().any():
        raise AssertionError("Hoch expression cell_id_export is not unique")
    if len(expression) != len(source):
        raise AssertionError("Hoch expression/metadata row counts differ")
    missing_from_expression = pd.Index(source["cell_id_export"]).difference(
        pd.Index(expression["cell_id_export"])
    )
    missing_from_metadata = pd.Index(expression["cell_id_export"]).difference(
        pd.Index(source["cell_id_export"])
    )
    if len(missing_from_expression) or len(missing_from_metadata):
        raise AssertionError("Hoch expression/metadata cell-ID sets differ")
    chosen = chosen_meta.merge(
        expression, on="cell_id_export", how="left", sort=False, validate="one_to_one"
    )
    if len(chosen) != TARGET_PER_DATASET:
        raise AssertionError(f"Hoch selected {len(chosen):,} cells")
    fine = relabel_hoch(chosen["cell_type"])
    metadata = pd.DataFrame({
        "source_cell_id": "hoch::" + chosen["cell_id_export"].astype(str),
        "source_index_zero_based": chosen["_source_index_zero_based"].astype(np.int64),
        "dataset": dataset,
        "technology": "IMC",
        "patient_id": chosen["patient_id"].astype(str),
        "specimen_id": chosen["sample_block_id"].astype(str),
        "acquisition_id": "image_" + chosen["image_number"].astype(str),
        "unit_key": dataset + "|patient_" + chosen["patient_id"].astype(str),
        "timepoint": chosen["patient_treatment_status_before_surgery"].astype(str),
        "stage": chosen["patient_cancer_stage"].astype(str),
        "provider_celltype_original": chosen["cell_type"].astype(str),
        "provider_celltype_common": chosen["cell_type_rf"].astype(str),
        "provider_celltype_coarse": chosen["cell_type"].astype(str),
    })
    metadata = finalize_metadata(metadata, fine)
    raw = chosen[[MARKER_MAP[dataset][marker] for marker in MARKERS]].to_numpy(np.float32)
    return raw, metadata, pd.DataFrame(audit_rows), [
        source_record(metadata_path), source_record(expression_path)
    ]


def build_ptacek(
    allocation_rng: np.random.Generator,
    selection_rng: np.random.Generator,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    dataset = DATASETS[2]
    folder = RAW_ROOT / "ptacek_mibi"
    raw_path = folder / "scaled_mibi_data_w_cell_gating.csv"
    patient_path = folder / "patient_info.csv"
    gate_columns = [
        "Tumor cells (SOX10+)", "Fibroblasts (Vimentin+)",
        "Smooth muscle/Myofibroblasts (Vimentin+SMA+)", "Immune Cells",
        "Myeloid cells", "B cells (CD20+)", "NK cells (CD3-CD56+)",
        "T cells (CD3+)", "Helper T Cells (CD3+CD4+)",
        "Cytotoxic T Cells (CD3+CD8+)",
    ]
    columns = ["fov", "label", "x_centroid", "y_centroid"] + list(
        MARKER_MAP[dataset].values()
    ) + gate_columns
    columns = list(dict.fromkeys(columns))
    source = pd.read_csv(raw_path, usecols=columns, dtype={"fov": "string"}, low_memory=False)
    source["_source_index_zero_based"] = np.arange(len(source), dtype=np.int64)
    patients = pd.read_csv(patient_path, dtype={"id": "string"}, low_memory=False)
    source = source.merge(patients, left_on="fov", right_on="id", how="left", validate="many_to_one")
    if source["gross_dx_stage"].isna().any():
        raise AssertionError("Ptacek FOV missing patient metadata")
    cohort = source.loc[source["gross_dx_stage"].astype(str).eq("III")].copy()
    if len(cohort) != 24_529 or cohort["fov"].nunique() != 23:
        raise AssertionError("Ptacek Stage III cohort no longer matches 24,529/23")
    fov_counts = cohort.groupby("fov", sort=True).size().astype(int).to_dict()
    quotas = waterfill_quotas(fov_counts, TARGET_PER_DATASET, allocation_rng)
    chosen_indices: list[int] = []
    audit_rows: list[dict[str, object]] = []
    for fov in sorted(quotas):
        candidates = cohort.index[cohort["fov"].astype(str).eq(fov)].to_numpy(int)
        selected = selection_rng.choice(candidates, quotas[fov], replace=False)
        chosen_indices.extend(selected.tolist())
        audit_rows.append({
            "dataset": dataset,
            "patient_id": fov,
            "subunit": fov,
            "eligible_cells": len(candidates),
            "selected_cells": quotas[fov],
            "patient_selected_cells": quotas[fov],
        })
    chosen = source.loc[sorted(chosen_indices)].copy().reset_index(drop=True)
    if len(chosen) != TARGET_PER_DATASET:
        raise AssertionError(f"Ptacek selected {len(chosen):,} cells")

    broad = np.full(len(chosen), "unknown", dtype=object)
    broad[true_mask(chosen["Immune Cells"])] = "other"
    stromal = true_mask(chosen["Fibroblasts (Vimentin+)"]) | true_mask(
        chosen["Smooth muscle/Myofibroblasts (Vimentin+SMA+)"]
    )
    broad[stromal] = "stroma"
    broad[true_mask(chosen["B cells (CD20+)"])] = "B_cell"
    broad[true_mask(chosen["T cells (CD3+)"])] = "T_cell"
    broad[true_mask(chosen["Myeloid cells"])] = "myeloid"
    broad[true_mask(chosen["Tumor cells (SOX10+)"])] = "tumor"

    fine = np.full(len(chosen), "Non-joint", dtype=object)
    fine[broad == "tumor"] = "Tumor"
    fine[broad == "myeloid"] = "Myeloid"
    fine[broad == "stroma"] = "Stroma"
    t_cells = broad == "T_cell"
    fine[t_cells] = "Other T / CD8−"
    cd4 = true_mask(chosen["Helper T Cells (CD3+CD4+)"])
    cd8 = true_mask(chosen["Cytotoxic T Cells (CD3+CD8+)"])
    if np.any(cd4 & cd8) or np.any((cd4 | cd8) & ~t_cells):
        raise AssertionError("Ptacek CD4/CD8 gates conflict with broad T-cell gate")
    fine[cd4] = "CD4 T"
    fine[cd8] = "CD8 T"
    metadata = pd.DataFrame({
        "source_cell_id": "ptacek::" + chosen["fov"].astype(str) + "::" + chosen["label"].astype(str),
        "source_index_zero_based": chosen["_source_index_zero_based"].astype(np.int64),
        "dataset": dataset,
        "technology": "MIBI",
        "patient_id": chosen["fov"].astype(str),
        "specimen_id": chosen["fov"].astype(str),
        "acquisition_id": chosen["fov"].astype(str),
        "unit_key": dataset + "|" + chosen["fov"].astype(str),
        "timepoint": "not provided",
        "stage": chosen["gross_dx_stage"].astype(str),
        "provider_celltype_original": pd.Series(broad).astype(str),
        "provider_celltype_common": pd.Series(broad).astype(str),
        "provider_celltype_coarse": pd.Series(broad).astype(str),
    })
    metadata = finalize_metadata(metadata, np.asarray(fine, dtype=str))
    raw = chosen[[MARKER_MAP[dataset][marker] for marker in MARKERS]].to_numpy(np.float32)
    return raw, metadata, pd.DataFrame(audit_rows), [
        source_record(raw_path), source_record(patient_path)
    ]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    children = np.random.SeedSequence(ROOT_SEED).spawn(6)
    rngs = [np.random.default_rng(child) for child in children]
    builders = (
        ("Hickey", build_hickey, rngs[0], rngs[1]),
        ("Hoch", build_hoch, rngs[2], rngs[3]),
        ("Ptacek", build_ptacek, rngs[4], rngs[5]),
    )
    matrices: list[np.ndarray] = []
    metadata_parts: list[pd.DataFrame] = []
    sampling_parts: list[pd.DataFrame] = []
    sources: dict[str, list[dict[str, object]]] = {}
    for name, builder, allocation_rng, selection_rng in builders:
        print(f"Building {name} metadata-defined 20,000-cell block", flush=True)
        matrix, metadata, sampling, source_files = builder(allocation_rng, selection_rng)
        matrices.append(matrix)
        metadata_parts.append(metadata)
        sampling_parts.append(sampling)
        sources[name] = source_files

    raw = np.vstack(matrices).astype(np.float32, copy=False)
    metadata = pd.concat(metadata_parts, ignore_index=True)
    metadata.insert(0, "row_id", np.arange(len(metadata), dtype=np.int64))
    sampling = pd.concat(sampling_parts, ignore_index=True)
    expected_shape = (TARGET_PER_DATASET * len(DATASETS), len(MARKERS))
    if raw.shape != expected_shape or len(metadata) != expected_shape[0]:
        raise AssertionError((raw.shape, len(metadata)))
    if not np.isfinite(raw).all():
        raise AssertionError("shared-marker matrix contains non-finite values")
    if metadata["source_cell_id"].duplicated().any():
        raise AssertionError("source cell IDs are not unique")
    if metadata.groupby("dataset", sort=False).size().to_dict() != {
        dataset: TARGET_PER_DATASET for dataset in DATASETS
    }:
        raise AssertionError("dataset blocks are not 20,000 cells each")
    expected_units = {DATASETS[0]: 6, DATASETS[1]: 17, DATASETS[2]: 23}
    if metadata.groupby("dataset", sort=False)["unit_key"].nunique().to_dict() != expected_units:
        raise AssertionError("unexpected fit-unit counts")

    raw_path = OUT / "raw.npy"
    metadata_path = OUT / "metadata.csv.gz"
    markers_path = OUT / "markers.txt"
    sampling_path = OUT / "sampling_counts.csv"
    labels_path = OUT / "posthoc_label_counts.csv"
    np.save(raw_path, raw)
    write_csv_gz(metadata, metadata_path)
    markers_path.write_text("\n".join(MARKERS) + "\n")
    sampling.to_csv(sampling_path, index=False)
    (
        metadata.groupby(["dataset", "celltype"], sort=False)
        .size().rename("cells").reset_index()
        .to_csv(labels_path, index=False)
    )
    script_path = Path(__file__).resolve()
    manifest = {
        "schema_version": "melanoma-metadata-matched-test-v1",
        "created_by": str(script_path),
        "created_by_sha256": sha256(script_path),
        "random_seed": ROOT_SEED,
        "randomization": "NumPy SeedSequence/default_rng; no hashing of cell IDs",
        "cells": int(len(metadata)),
        "markers": list(MARKERS),
        "marker_count": len(MARKERS),
        "dataset_order": list(DATASETS),
        "dataset_cells": metadata["dataset"].value_counts(sort=False).astype(int).to_dict(),
        "dataset_patients": metadata.groupby("dataset", sort=False)["patient_id"].nunique().astype(int).to_dict(),
        "fit_units": metadata.groupby("dataset", sort=False)["unit_key"].nunique().astype(int).to_dict(),
        "selection": {
            DATASETS[0]: (
                "All six paired-study patients; pretreatment specimen only; patient-balanced; "
                "within patient, acquisition-stratum proportional random sampling."
            ),
            DATASETS[1]: (
                "sample_tissue_type=Melanoma, patient_cancer_stage=III, "
                "patient_treatment_status_before_surgery=naive; all 17 eligible patients; "
                "patient-balanced, then image-balanced random sampling."
            ),
            DATASETS[2]: (
                "gross_dx_stage=III; all 23 eligible FOV/patients; capacity-aware "
                "patient-balanced random sampling."
            ),
        },
        "sampling_excluded_fields": [
            "cell type", "marker expression", "treatment response", "survival", "UMAP"
        ],
        "fit_metadata_columns": ["dataset", "unit_key"],
        "posthoc_provider_labels_used_for_fitting": False,
        "input_scale": (
            "Highest-available provider-processed values. No clipping, shifting, inverse "
            "transformation, asinh, or other preprocessing was applied by this builder."
        ),
        "marker_caveat": "Hickey SOX10/S100 is used as the shared SOX10 proxy.",
        "marker_source_columns": MARKER_MAP,
        "source_files": sources,
        "outputs": {
            "raw.npy": sha256(raw_path),
            "metadata.csv.gz": sha256(metadata_path),
            "markers.txt": sha256(markers_path),
            "sampling_counts.csv": sha256(sampling_path),
            "posthoc_label_counts.csv": sha256(labels_path),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    (OUT / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        "Wrote metadata-defined melanoma input: "
        f"{raw.shape[0]:,} cells x {raw.shape[1]} markers",
        flush=True,
    )


if __name__ == "__main__":
    main()
