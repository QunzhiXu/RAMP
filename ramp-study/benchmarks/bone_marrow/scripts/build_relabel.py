#!/usr/bin/env python3
"""Reproduce the cell-type relabeling used by the BM CD19 story.

This script never changes the canonical benchmark inputs.  It reads the three
frozen alignment matrices and their locked provider metadata, appends the
story-compatible ``celltype`` label, and writes relabeled copies below this
directory.
"""

from __future__ import annotations

import hashlib
import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from paths import new_output


# Bound only by main() after the input/output paths have been validated.
HERE = None
BENCHMARK_ROOT = None
DATA_DIR = None
VERSION = "bm_story_relabel_v1_2026-08-27"
EXPECTED_TOTAL = 88_031

DATASETS = (
    ("BM CODEX", "bm_codex"),
    ("BM CyTOF", "bm_cytof"),
    ("Triana AbSeq", "triana_abseq"),
)

LABEL_ORDER = (
    "Mature / CODEX B",
    "Pre / immature B",
    "Plasma cell",
    "CD4 T",
    "CD8 T",
    "Other T / NKT",
    "NK",
    "Monocyte",
    "Macrophage",
    "Dendritic",
    "Other myeloid",
    "Promyelocytic progenitors",
    "HSPC / progenitor",
    "CODEX erythroid",
    "Triana erythroid progenitor",
    "Megakaryocytic",
    "Endothelial",
    "Stromal / VSMC",
    "Unclassified / technical",
)

TRIANA_MATURE_B = {
    "CD11c+ memory B cells",
    "Class switched memory B cells",
    "Mature naive B cells",
    "Nonswitched memory B cells",
}
TRIANA_PRE_B = {
    "Immature B cells",
    "Pre-B cells",
    "Pre-pro-B cells",
    "Pro-B cells",
    "Small pre-B cell",
}
TRIANA_ERYTHROID = {
    "Early erythroid progenitor",
    "Late erythroid progenitor",
    "Aberrant erythroid",
}
TRIANA_PROMYELOCYTIC = {
    "Early promyelocytes",
    "Late promyelocytes",
    "Eosinophil-basophil-mast cell progenitors",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relabel_codex(frame: pd.DataFrame) -> np.ndarray:
    """Apply the frozen CODEX display-label mapping defined in this builder."""
    broad = frame["cluster_anno_l1"].astype(str).to_numpy()
    fine = frame["cluster_anno_l2"].astype(str).to_numpy()
    result = np.full(len(frame), "Unclassified / technical", dtype=object)
    mapping = {
        "Mature / CODEX B": ("B-Cells",),
        "Plasma cell": ("Plasma Cells",),
        "CD4 T": ("CD4+ T-Cell",),
        "CD8 T": ("CD8+ T-Cell",),
        "Monocyte": ("Monocytes",),
        "Dendritic": ("pDC",),
        "HSPC / progenitor": ("HSPC",),
        "CODEX erythroid": ("Erythroid",),
        "Megakaryocytic": ("Megakaryocyte",),
        "Endothelial": ("Endothelial",),
        "Stromal / VSMC": ("Stromal", "VSMC"),
    }
    for label, values in mapping.items():
        result[np.isin(broad, values)] = label
    result[np.isin(broad, ["Myeloid", "Macrophages"])] = "Other myeloid"
    result[(broad == "Macrophages") & (fine == "Macrophages")] = "Macrophage"
    return result.astype(str)


def relabel_cytof(labels: np.ndarray) -> np.ndarray:
    """Apply the frozen CyTOF display-label mapping defined in this builder."""
    mapping = {
        "CD16+ NK cells": "NK",
        "CD16- NK cells": "NK",
        "CD4 T cells": "CD4 T",
        "CD8 T cells": "CD8 T",
        "Mature B cells": "Mature / CODEX B",
        "PreBcells": "Pre / immature B",
        "Pre B cells": "Pre / immature B",
        "Monocytes": "Monocyte",
    }
    return np.asarray(
        [mapping.get(str(value), "Unclassified / technical") for value in labels],
        dtype=str,
    )


def relabel_triana(labels: np.ndarray) -> np.ndarray:
    """Apply the frozen Triana display-label mapping defined in this builder."""
    labels = np.asarray(labels, dtype=str)
    result = np.full(len(labels), "Unclassified / technical", dtype=object)
    result[np.isin(labels, list(TRIANA_MATURE_B))] = "Mature / CODEX B"
    result[np.isin(labels, list(TRIANA_PRE_B))] = "Pre / immature B"
    result[labels == "Plasma cells"] = "Plasma cell"
    result[np.char.startswith(labels, "CD4+")] = "CD4 T"
    result[labels == "CD69+PD-1+ memory CD4+ T cells"] = "CD4 T"
    result[np.char.startswith(labels, "CD8+")] = "CD8 T"
    result[np.isin(labels, ["GammaDelta T cells", "NK T cells"])] = "Other T / NKT"
    result[
        np.isin(
            labels,
            [
                "CD56brightCD16- NK cells",
                "CD56dimCD16+ NK cells",
                "NK cell progenitors",
            ],
        )
    ] = "NK"
    result[
        np.isin(labels, ["Classical Monocytes", "Non-classical monocytes"])
    ] = "Monocyte"
    result[
        np.isin(
            labels,
            [
                "Conventional dendritic cell 1",
                "Conventional dendritic cell 2",
                "Plasmacytoid dendritic cells",
                "Plasmacytoid dendritic cell progenitors",
            ],
        )
    ] = "Dendritic"
    result[labels == "Myelocytes"] = "Other myeloid"
    result[np.isin(labels, list(TRIANA_PROMYELOCYTIC))] = (
        "Promyelocytic progenitors"
    )
    result[
        np.isin(
            labels,
            ["HSCs & MPPs", "Lymphomyeloid prog", "Erythro-myeloid progenitors"],
        )
    ] = "HSPC / progenitor"
    result[np.isin(labels, list(TRIANA_ERYTHROID))] = (
        "Triana erythroid progenitor"
    )
    result[labels == "Megakaryocyte progenitors"] = "Megakaryocytic"
    return result.astype(str)


def provider_labels(
    dataset: str, metadata: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if dataset == "BM CODEX":
        l1 = metadata["cluster_anno_l1"].fillna("").astype(str).to_numpy()
        l2 = metadata["cluster_anno_l2"].fillna("").astype(str).to_numpy()
        coarse = metadata["cluster_anno_coarse"].fillna("").astype(str).to_numpy()
        return l1, l2, coarse, relabel_codex(metadata)

    l1 = metadata["provider_celltype"].fillna("").astype(str).to_numpy()
    empty = np.full(len(metadata), "", dtype=str)
    if dataset == "BM CyTOF":
        return l1, empty, empty, relabel_cytof(l1)
    if dataset == "Triana AbSeq":
        return l1, empty, empty, relabel_triana(l1)
    raise ValueError(f"Unknown dataset: {dataset}")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression="gzip")


def main() -> None:
    global HERE, BENCHMARK_ROOT, DATA_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    BENCHMARK_ROOT = args.benchmark_root.resolve(strict=True)
    HERE = new_output(args.output_dir, [BENCHMARK_ROOT])
    DATA_DIR = HERE / "data"
    temporary_root = Path(tempfile.mkdtemp(prefix="bm_relabel_", dir=HERE))
    try:
        temporary_data = temporary_root / "data"
        combined_parts: list[pd.DataFrame] = []
        metadata_parts: list[pd.DataFrame] = []
        audit_parts: list[pd.DataFrame] = []
        files_manifest: list[dict[str, object]] = []

        for dataset, slug in DATASETS:
            alignment_path = BENCHMARK_ROOT / slug / "alignment_input_14.csv.gz"
            metadata_path = BENCHMARK_ROOT / slug / "locked_metadata.csv.gz"
            alignment = pd.read_csv(alignment_path, low_memory=False)
            metadata = pd.read_csv(metadata_path, low_memory=False)

            if len(alignment) != len(metadata):
                raise RuntimeError(f"Row-count mismatch for {dataset}")
            if not alignment["cell_id"].astype(str).equals(
                metadata["cell_id"].astype(str)
            ):
                raise RuntimeError(f"Cell order mismatch for {dataset}")
            for column in ("dataset", "donor"):
                if not alignment[column].astype(str).equals(metadata[column].astype(str)):
                    raise RuntimeError(f"{column} mismatch for {dataset}")

            l1, l2, coarse, labels = provider_labels(dataset, metadata)
            if not set(labels).issubset(LABEL_ORDER):
                raise RuntimeError(f"Unexpected relabel output for {dataset}")

            relabeled = alignment.copy()
            relabeled["provider_label_l1"] = l1
            relabeled["provider_label_l2"] = l2
            relabeled["provider_label_coarse"] = coarse
            relabeled["celltype"] = labels
            relabeled["celltype_relabel_version"] = VERSION

            output_path = temporary_data / f"{slug}_relabeled.csv.gz"
            write_csv(relabeled, output_path)
            check = pd.read_csv(output_path, low_memory=False)
            if not alignment.equals(check[alignment.columns]):
                raise RuntimeError(f"Alignment values changed for {dataset}")
            if not np.array_equal(check["celltype"].astype(str).to_numpy(), labels):
                raise RuntimeError(f"Relabel verification failed for {dataset}")

            relabeled_metadata = relabeled[
                [
                    "cell_id",
                    "dataset",
                    "donor",
                    "unit_key",
                    "provider_label_l1",
                    "provider_label_l2",
                    "provider_label_coarse",
                    "celltype",
                    "celltype_relabel_version",
                ]
            ].copy()
            metadata_parts.append(relabeled_metadata)
            combined_parts.append(relabeled)
            audit_parts.append(
                relabeled_metadata[
                    [
                        "dataset",
                        "provider_label_l1",
                        "provider_label_l2",
                        "provider_label_coarse",
                        "celltype",
                    ]
                ]
            )
            files_manifest.append(
                {
                    "dataset": dataset,
                    "slug": slug,
                    "n_cells": int(len(relabeled)),
                    "alignment_input": str(alignment_path),
                    "alignment_input_sha256": sha256(alignment_path),
                    "locked_metadata": str(metadata_path),
                    "locked_metadata_sha256": sha256(metadata_path),
                    "relabeled_output": str(DATA_DIR / output_path.name),
                    "relabeled_output_sha256": sha256(output_path),
                }
            )

        combined = pd.concat(combined_parts, ignore_index=True)
        relabeled_metadata = pd.concat(metadata_parts, ignore_index=True)
        audit = pd.concat(audit_parts, ignore_index=True)
        if len(combined) != EXPECTED_TOTAL:
            raise RuntimeError(
                f"Expected {EXPECTED_TOTAL:,} cells, found {len(combined):,}"
            )
        if combined["cell_id"].duplicated().any():
            raise RuntimeError("Duplicate cell_id values in combined output")

        combined_path = temporary_data / "bone_marrow_88031_relabeled.csv.gz"
        metadata_output = temporary_root / "relabeled_metadata.csv.gz"
        write_csv(combined, combined_path)
        write_csv(relabeled_metadata, metadata_output)

        mapping = (
            audit.groupby(
                [
                    "dataset",
                    "provider_label_l1",
                    "provider_label_l2",
                    "provider_label_coarse",
                    "celltype",
                ],
                dropna=False,
            )
            .size()
            .rename("n_cells")
            .reset_index()
            .sort_values(["dataset", "celltype", "provider_label_l1"])
        )
        mapping.to_csv(temporary_root / "relabel_mapping_audit.csv", index=False)

        counts = (
            audit.groupby(["celltype", "dataset"], observed=False)
            .size()
            .unstack(fill_value=0)
            .reindex(LABEL_ORDER, fill_value=0)
        )
        counts["Total"] = counts.sum(axis=1)
        counts.to_csv(temporary_root / "relabel_counts.csv")

        mapping_source = Path(__file__).resolve()
        manifest = {
            "version": VERSION,
            "purpose": "Frozen BM story-compatible display relabeling",
            "n_cells": int(len(combined)),
            "label_order": list(LABEL_ORDER),
            "mapping_provenance": (
                "relabel_codex, relabel_cytof, and relabel_triana are frozen "
                "directly in this builder"
            ),
            "mapping_implementation_source": str(mapping_source),
            "mapping_implementation_sha256": sha256(mapping_source),
            "matrix_columns_preserved": True,
            "canonical_inputs_modified": False,
            "combined_output": {
                "path": str(DATA_DIR / combined_path.name),
                "sha256": sha256(combined_path),
            },
            "relabeled_metadata_output": {
                "path": str(HERE / metadata_output.name),
                "sha256": sha256(metadata_output),
            },
            "files": files_manifest,
        }
        with (temporary_root / "manifest.json").open("w") as stream:
            json.dump(manifest, stream, indent=2)

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        for path in temporary_data.iterdir():
            path.replace(DATA_DIR / path.name)
        for filename in (
            "relabeled_metadata.csv.gz",
            "relabel_mapping_audit.csv",
            "relabel_counts.csv",
            "manifest.json",
        ):
            (temporary_root / filename).replace(HERE / filename)

        print(counts.to_string())
        print(f"\nSaved {len(combined):,} relabeled cells to {DATA_DIR}")
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


if __name__ == "__main__":
    main()
