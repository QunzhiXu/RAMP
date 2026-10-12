#!/usr/bin/env python3
"""Build the frozen melanoma spatial-transfer story cohort from raw data.

The cohort is selected exclusively from provider clinical metadata:

* Moldoveanu IMC: all 30 pretreatment samples with ``Cohort == ICI``.
* Engelbrecht CITE-seq: all 8 melanoma-positive sentinel lymph-node samples
  with ``Stage == StageIII``.

Cell labels, response labels, expression values, coordinates, and downstream
results are never used to decide which donors or cells enter the cohort.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
FINAL_XU = HERE.parents[2]
RAW = FINAL_XU / "raw_data/melanoma"
MOLD_ROOT = RAW / "moldoveanu_imc/Moldoveanu_2022_CyTOF_melanoma/Data"
ENG_ROOT = RAW / "engelbrecht_citeseq/raw"

SEED = 20260820
DISPLAY_N_PER_DATASET = 20_000

SHARED = [
    "PDL1", "CD45", "CD4", "CD20", "CD8", "CD45RA", "CD40",
    "CD3", "CD45RO", "HLA_DR", "CD14", "CD31", "CD29",
]

MOLD_SOURCE = {
    "PDL1": "PDL1",
    "CD45": "CD45",
    "CD4": "CD4",
    "CD20": "CD20",
    "CD8": "CD8a",
    "CD45RA": "CD45RA",
    "CD40": "CD40",
    "CD3": "CD3",
    "CD45RO": "CD45RO",
    "HLA_DR": "HLA.DR",
    "CD14": "CD14",
    "CD31": "CD31",
    "CD29": "CD29",
}

ENG_SOURCE = {
    "PDL1": "CD274-AB",
    "CD45": "CD45-AB",
    "CD4": "CD4-AB",
    "CD20": "CD20-AB",
    "CD8": "CD8-AB",
    "CD45RA": "CD45RA-AB",
    "CD40": "CD40-AB",
    "CD3": "CD3-AB",
    "CD45RO": "CD45RO-AB",
    "HLA_DR": "HLA-DR-AB",
    "CD14": "CD14-AB",
    "CD31": "CD31-AB",
    "CD29": "CD29-AB",
}

ENG_HELD_OUT = {
    "PD1": "CD279-AB",
    "TIGIT": "TIGIT-AB",
    "CD103": "CD103-AB",
    "CD69": "CD69-AB",
    "CD39": "CD39-AB",
    "CD107A": "CD107a-AB",
    "KLRG1": "KLRG1-AB",
    "CX3CR1": "CX3CR1-AB",
    "CD62L": "CD62L-AB",
}

MOLD_CLINICAL_COLUMNS = [
    "Sample_ID",
    "Age_last_alive",
    "Sex",
    "irRC",
    "Response",
    "Known_Deceased",
    "Survival_from_Rx_Start",
    "Primary_site",
    "Lymphocyte_Infiltration",
    "Drug",
    "CTLA4",
    "PD1",
    "Treatment",
    "Tissue_Source_Simplified",
    "Tissue_Source",
    "Primary_Site_Simplified",
    "Type",
]

ENG_METADATA_COLUMNS = [
    "Unnamed: 0",
    "Sample_ID",
    "Patient_ID",
    "Stage",
    "Batch",
    "clust.name",
    "Broad_celltype",
    "Broad_celltype2",
    "Broad_celltype3",
    "cell_type",
    "Broad_cell.type.1",
]


def write_csv_gz(frame: pd.DataFrame, path: Path) -> None:
    """Write deterministic gzip-compressed CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(fileobj=raw_handle, mode="wb", mtime=0) as handle:
            frame.to_csv(handle, index=False)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def donor_uniform_sample(donors: np.ndarray, target: int, seed: int) -> np.ndarray:
    """Select display cells uniformly across donors without using labels."""
    rng = np.random.default_rng(seed)
    donors = np.asarray(donors).astype(str)
    unique = np.array(sorted(pd.unique(donors)))
    target = min(int(target), len(donors))
    quota, remainder = divmod(target, len(unique))
    selected: list[np.ndarray] = []
    selected_mask = np.zeros(len(donors), dtype=bool)
    for i, donor in enumerate(unique):
        rows = np.flatnonzero(donors == donor)
        take = min(len(rows), quota + int(i < remainder))
        chosen = rng.choice(rows, size=take, replace=False)
        selected.append(chosen)
        selected_mask[chosen] = True
    result = np.concatenate(selected)
    if len(result) < target:
        available = np.flatnonzero(~selected_mask)
        extra = rng.choice(available, size=target - len(result), replace=False)
        result = np.concatenate([result, extra])
    rng.shuffle(result)
    return result


def build_moldoveanu() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clinical = pd.read_csv(MOLD_ROOT / "ST1_sample_info.txt", sep="\t")
    clinical = clinical.loc[clinical["Sample_ID"].notna()].drop_duplicates("Sample_ID")
    clinical["Sample_ID"] = clinical["Sample_ID"].astype(str)
    selected_clinical = clinical.loc[
        clinical["Cohort"].astype(str).eq("ICI"), MOLD_CLINICAL_COLUMNS
    ].copy()
    eligible = set(selected_clinical["Sample_ID"])
    if len(eligible) != 30:
        raise AssertionError(f"Expected 30 Moldoveanu ICI patients, observed {len(eligible)}")

    cell_columns = list(dict.fromkeys([
        "Location_Center_X",
        "Location_Center_Y",
        "Sample_ID",
        "Cell_ID",
        *MOLD_SOURCE.values(),
        "Ki67",
        "GranzymeB",
        "prob.Ki67",
        "Cluster.v2",
    ]))
    cells = pd.read_csv(
        MOLD_ROOT / "ICI_meanIntensity.tsv.gz",
        sep="\t",
        usecols=cell_columns,
    )
    cells["Sample_ID"] = cells["Sample_ID"].astype(str)
    cells = cells.loc[cells["Sample_ID"].isin(eligible)].reset_index(drop=True)
    if len(cells) != 112_497:
        raise AssertionError(f"Expected 112497 Moldoveanu ICI cells, observed {len(cells)}")
    cells["cell_uid"] = cells["Sample_ID"] + "::" + cells["Cell_ID"].astype(str)
    if not cells["cell_uid"].is_unique:
        raise AssertionError("Moldoveanu cell_uid is not unique")

    alignment = pd.DataFrame({
        "cell_uid": cells["cell_uid"],
        "dataset": "Moldoveanu IMC",
        "donor": cells["Sample_ID"],
    })
    for marker in SHARED:
        alignment[marker] = pd.to_numeric(cells[MOLD_SOURCE[marker]], errors="raise")

    locked = cells[[
        "cell_uid",
        "Sample_ID",
        "Cell_ID",
        "Location_Center_X",
        "Location_Center_Y",
        "Cluster.v2",
        "Ki67",
        "GranzymeB",
        "prob.Ki67",
    ]].merge(selected_clinical, on="Sample_ID", how="left", validate="many_to_one")
    return alignment, locked, selected_clinical


def load_engelbrecht_marker_matrix(
    selected_barcodes: list[str], required_rows: dict[str, str]
) -> pd.DataFrame:
    header = pd.read_csv(ENG_ROOT / "ADT_counts.csv", nrows=0).columns.astype(str).tolist()
    feature_column = header[0]
    all_barcodes = header[1:]
    selected_set = set(selected_barcodes)
    if not selected_set.issubset(all_barcodes):
        raise AssertionError("Selected Engelbrecht barcodes are absent from ADT_counts.csv")
    matrix = pd.read_csv(
        ENG_ROOT / "ADT_counts.csv",
        usecols=[feature_column, *selected_barcodes],
    )
    matrix[feature_column] = matrix[feature_column].astype(str)
    required = set(required_rows.values())
    matrix = matrix.loc[matrix[feature_column].isin(required)].set_index(feature_column)
    missing = required - set(matrix.index)
    if missing:
        raise AssertionError(f"Missing Engelbrecht ADT markers: {sorted(missing)}")
    return matrix


def build_engelbrecht() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metadata = pd.read_csv(
        ENG_ROOT / "cell_metadata.csv",
        usecols=ENG_METADATA_COLUMNS,
        low_memory=False,
    )
    metadata["source_cell_id"] = metadata["Unnamed: 0"].astype(str)
    metadata = metadata.drop(columns="Unnamed: 0")
    selected = metadata.loc[metadata["Stage"].astype(str).eq("StageIII")].reset_index(drop=True)
    if selected["Patient_ID"].astype(str).nunique() != 8:
        raise AssertionError("Expected 8 Engelbrecht Stage III patients")
    if len(selected) != 53_189:
        raise AssertionError(f"Expected 53189 Engelbrecht Stage III cells, observed {len(selected)}")
    if not selected["source_cell_id"].is_unique:
        raise AssertionError("Engelbrecht source_cell_id is not unique")

    matrix = load_engelbrecht_marker_matrix(
        selected["source_cell_id"].astype(str).tolist(),
        {**ENG_SOURCE, **ENG_HELD_OUT},
    )
    barcodes = selected["source_cell_id"].astype(str).tolist()
    alignment = pd.DataFrame({
        "cell_uid": barcodes,
        "dataset": "Engelbrecht CITE-seq",
        "donor": selected["Patient_ID"].astype(str),
    })
    for marker in SHARED:
        alignment[marker] = matrix.loc[ENG_SOURCE[marker], barcodes].to_numpy(np.float64)

    locked = selected.copy()
    for marker, source in ENG_HELD_OUT.items():
        locked[marker] = matrix.loc[source, barcodes].to_numpy(np.float64)
    return alignment, locked, selected[["Patient_ID", "Stage"]].drop_duplicates()


def main() -> None:
    mold_dir = HERE / "moldoveanu_imc"
    eng_dir = HERE / "engelbrecht_citeseq"
    for folder in (mold_dir, eng_dir):
        folder.mkdir(parents=True, exist_ok=True)

    mold_alignment, mold_locked, mold_patients = build_moldoveanu()
    eng_alignment, eng_locked, eng_patients = build_engelbrecht()

    outputs = {
        "moldoveanu_alignment": mold_dir / "alignment_input.csv.gz",
        "moldoveanu_locked": mold_dir / "locked_metadata.csv.gz",
        "engelbrecht_alignment": eng_dir / "alignment_input.csv.gz",
        "engelbrecht_locked": eng_dir / "locked_metadata.csv.gz",
    }
    write_csv_gz(mold_alignment, outputs["moldoveanu_alignment"])
    write_csv_gz(mold_locked, outputs["moldoveanu_locked"])
    write_csv_gz(eng_alignment, outputs["engelbrecht_alignment"])
    write_csv_gz(eng_locked, outputs["engelbrecht_locked"])

    display_mold = donor_uniform_sample(
        mold_alignment["donor"].to_numpy(), DISPLAY_N_PER_DATASET, SEED
    )
    display_eng = donor_uniform_sample(
        eng_alignment["donor"].to_numpy(), DISPLAY_N_PER_DATASET, SEED + 1
    )
    display = pd.concat([
        mold_alignment.loc[display_mold, ["cell_uid", "dataset", "donor"]],
        eng_alignment.loc[display_eng, ["cell_uid", "dataset", "donor"]],
    ], ignore_index=True)
    display["display_seed"] = np.where(
        display["dataset"].eq("Moldoveanu IMC"), SEED, SEED + 1
    )
    display_path = HERE / "display_sample_cells.csv.gz"
    write_csv_gz(display, display_path)
    outputs["display_sample"] = display_path

    donor_counts = pd.concat([
        mold_alignment.groupby(["dataset", "donor"], as_index=False).size(),
        eng_alignment.groupby(["dataset", "donor"], as_index=False).size(),
    ], ignore_index=True).rename(columns={"size": "n_cells"})
    donor_counts.to_csv(HERE / "donor_cell_counts.csv", index=False)
    pd.DataFrame({
        "canonical_marker": SHARED,
        "Moldoveanu_IMC_column": [MOLD_SOURCE[x] for x in SHARED],
        "Engelbrecht_CITEseq_ADT": [ENG_SOURCE[x] for x in SHARED],
    }).to_csv(HERE / "shared_marker_map.csv", index=False)

    checksums = {
        str(path.relative_to(HERE)): sha256(path)
        for path in outputs.values()
    }
    manifest = {
        "package": "melanoma_spatial_transfer_story",
        "created_by": Path(__file__).name,
        "selection_used_cell_labels": False,
        "selection_used_expression": False,
        "selection_used_outcomes": False,
        "selection": {
            "Moldoveanu IMC": {
                "rule": "Cohort == ICI",
                "clinical_context": "30 advanced-melanoma pretreatment samples",
                "n_donors": int(mold_alignment["donor"].nunique()),
                "n_cells": int(len(mold_alignment)),
            },
            "Engelbrecht CITE-seq": {
                "rule": "Stage == StageIII",
                "clinical_context": "8 melanoma-positive (MEL+) sentinel lymph nodes",
                "n_donors": int(eng_alignment["donor"].nunique()),
                "n_cells": int(len(eng_alignment)),
            },
        },
        "n_total_cells": int(len(mold_alignment) + len(eng_alignment)),
        "shared_markers": SHARED,
        "display_sampling": {
            "purpose": "UMAP display only; never used for cohort definition",
            "cells_per_dataset": DISPLAY_N_PER_DATASET,
            "strategy": "donor-uniform, label-blind, without replacement",
            "moldoveanu_seed": SEED,
            "engelbrecht_seed": SEED + 1,
        },
        "files_sha256": checksums,
        "clinical_rows": {
            "moldoveanu": int(len(mold_patients)),
            "engelbrecht": int(len(eng_patients)),
        },
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(json.dumps(manifest["selection"], indent=2))
    print(f"Total cells: {manifest['n_total_cells']}")
    print(f"Display cells: {len(display)}")


if __name__ == "__main__":
    main()
