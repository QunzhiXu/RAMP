#!/usr/bin/env python3
"""Rebuild the frozen melanoma full active panels from provider source files.

This recipe is deliberately independent of ``test0827`` and of every prior
alignment, UMAP, and relabel output.  It reads only provider/source-scale
measurements under ``raw_data/melanoma`` and the identity/row locks under
``benchmark_data/melanoma``.  No cell-type or outcome column is loaded.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
FINAL_XU = HERE.parents[2]
BENCHMARK = FINAL_XU / "benchmark_data" / "melanoma"
RAW_ROOT = FINAL_XU / "raw_data" / "melanoma"
OUTPUT = HERE / "source_active"
EXPECTED_CELLS = 20_000
SHARED = [
    "CD163", "CD3", "CD31", "CD68", "CD8", "HLA_DR",
    "LAG3", "PD1", "PDL1", "SMA", "SOX10", "VIMENTIN",
]
DATASETS = [
    ("MEL_Hickey2023_CODEX", "hickey_codex"),
    ("MEL_Hoch2021_IMC", "hoch_imc"),
    ("MEL_Ptacek2021_MIBI", "ptacek_mibi"),
]

HICKEY_PANEL = [
    ("CCR7", "CCR7"), ("CD103", "CD103"), ("CD11B", "CD11b"),
    ("CD11C", "CD11c"), ("CD138", "CD138"), ("CD15", "CD15"),
    ("CD163", "CD163"), ("CD20", "CD20"), ("CD206", "CD206"),
    ("CD21", "CD21"), ("CD25", "CD25"), ("CD3", "CD3"),
    ("CD31", "CD31"), ("CD4", "CD4"), ("CD45", "CD45"),
    ("CD45RA", "CD45RA"), ("CD45RO", "CD45RO"),
    ("CD56", "CD56"), ("CD57", "CD57"), ("CD68", "CD68"),
    ("CD73", "CD73"), ("CD8", "CD8"), ("CLEC9A", "CLEC9A"),
    ("CXCL10", "CXCL10"), ("CXCL13", "CXCL13"),
    ("CXCL9", "CXCL9"), ("CXCR3", "CXCR3"), ("CXCR5", "CXCR5"),
    ("COLLAGENIV", "CollagenIV"), ("EOMES", "EOMES"),
    ("EZH2", "EZH2"), ("FAP", "FAP"), ("FOXP3", "FoxP3"),
    ("GATA3", "GATA3"), ("GRANZYMEB", "GranzymeB"),
    ("HLA_DR", "HLA-DR"), ("ICOS", "ICOS"), ("IL10", "IL10"),
    ("IRF8", "IRF8"), ("KI67", "Ki67"), ("LAG3", "LAG3"),
    ("LDHA", "LDH-A"), ("MHCI", "MHC-I"), ("MMP9", "MMP9"),
    ("MELANA", "Melan-A"), ("PD1", "PD-1"), ("PDL1", "PD-L1"),
    ("PTEN", "PTEN"), ("PODOPLANIN", "Podoplanin"),
    ("SOX10", "SOX10/S100"), ("TCF17", "TCF1/7"),
    ("TBET", "Tbet"), ("VIMENTIN", "Vimentin"), ("SMA", "aSMA"),
    ("BCATENIN", "bCatenin"), ("PERK12", "pERK1/2"),
]
HOCH_PANEL = [
    ("VIMENTIN", "VIM"), ("SMA", "SMA"), ("KRT5", "KRT5"),
    ("CD38", "CD38"), ("HLA_DR", "HLA_DR"), ("S100", "S100"),
    ("CDH11", "CDH11"), ("FAP", "FAP"), ("CD163", "CD163"),
    ("B2M", "B2M"), ("CD134", "CD134"), ("CD68", "CD68"),
    ("SLC2A12", "SLC2A12"), ("CD3", "CD3e"), ("LAG3", "LAG3"),
    ("PD1", "PD_1"), ("CCR2", "CCR2"), ("PDL1", "PD_L1"),
    ("CD8", "CD8a"), ("SOX10", "SOX10"), ("CD31", "CD31"),
    ("MELANA", "MLANA"), ("PRB", "p_Rb"), ("CPARP", "c_PARP"),
    ("CD15", "CD15"), ("MPO", "MPO"),
]
PTACEK_PANEL = [
    ("BETATUBULIN", "beta-tubulin"), ("CD11B", "CD11b"),
    ("CD11C", "CD11c"), ("CD14", "CD14"), ("CD163", "CD163"),
    ("CD20", "CD20"), ("CD3", "CD3"), ("CD31", "CD31"),
    ("CD4", "CD4"), ("CD45", "CD45"), ("CD45RO", "CD45RO"),
    ("CD56", "CD56"), ("CD68", "CD68"), ("CD8", "CD8"),
    ("FOXP3", "FOXP3"), ("GRANZYMEB", "Granzyme B"),
    ("HLACLASS1ABANDCNAKATPASE", "HLA class 1 A, B, and C, Na-K-ATPase"),
    ("HLA_DR", "HLA DPDQDR"), ("IDO1", "IDO-1"), ("KI67", "Ki-67"),
    ("LAG3", "LAG3"), ("PD1", "PD-1"), ("PDL1", "PD-L1"),
    ("PODOPLANIN", "Podoplanin"), ("SMA", "SMA"),
    ("SOX10", "SOX10"), ("TIM3", "TIM-3"), ("VIMENTIN", "Vimentin"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def buffer_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(values, dtype=np.float64).tobytes()
    ).hexdigest()


def deterministic_csv(frame: pd.DataFrame, path: Path) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
                frame.to_csv(text, index=False)


def source_identity(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "bytes": int(stat.st_size),
        "sha256": sha256(path),
    }


def publish(stage: Path, output: Path) -> None:
    if not output.exists():
        stage.rename(output)
        return
    backup = output.with_name(f".{output.name}.old-{next(tempfile._get_candidate_names())}")
    output.rename(backup)
    try:
        stage.rename(output)
    except BaseException:
        backup.rename(output)
        raise
    shutil.rmtree(backup)


def read_locks() -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray], dict[str, object]]:
    selected_path = BENCHMARK / "selected_cells.csv.gz"
    raw_path = BENCHMARK / "input" / "raw.npy"
    markers_path = BENCHMARK / "input" / "markers.txt"
    columns = [
        "row_id", "cell_id", "source_cell_id", "source_index_zero_based",
        "dataset", "donor", "unit_key",
    ]
    selected = pd.read_csv(
        selected_path,
        usecols=columns,
        dtype={"cell_id": str, "source_cell_id": str, "dataset": str},
        low_memory=False,
    )
    frozen = np.load(raw_path, mmap_mode="r", allow_pickle=False)
    markers = [line for line in markers_path.read_text().splitlines() if line]
    if markers != SHARED or frozen.shape != (60_000, len(SHARED)):
        raise AssertionError("melanoma canonical shared-marker contract changed")
    if not np.array_equal(selected["row_id"].to_numpy(), np.arange(60_000)):
        raise AssertionError("melanoma canonical row lock changed")
    if selected["cell_id"].duplicated().any():
        raise AssertionError("melanoma canonical cell IDs are not unique")
    order = selected.loc[
        selected["dataset"].ne(selected["dataset"].shift(fill_value="")), "dataset"
    ].tolist()
    if order != [dataset for dataset, _ in DATASETS]:
        raise AssertionError(f"melanoma dataset block order changed: {order}")

    blocks: dict[str, pd.DataFrame] = {}
    shared: dict[str, np.ndarray] = {}
    for dataset, slug in DATASETS:
        block = selected.loc[selected["dataset"].eq(dataset)].reset_index(drop=True)
        if len(block) != EXPECTED_CELLS:
            raise AssertionError(f"{dataset}: canonical cell count changed")
        inspectable_path = BENCHMARK / slug / "alignment_input_12.csv.gz"
        inspectable_ids = pd.read_csv(
            inspectable_path, usecols=["cell_id"], dtype=str
        )["cell_id"].to_numpy()
        if not np.array_equal(inspectable_ids, block["cell_id"].to_numpy()):
            raise AssertionError(f"{dataset}: inspectable/canonical row order differs")
        rows = block["row_id"].to_numpy(dtype=int)
        blocks[dataset] = block
        shared[dataset] = np.asarray(frozen[rows], dtype=np.float64)
    audit = {
        "selected_cells": source_identity(selected_path),
        "raw_shared": source_identity(raw_path),
        "shared_markers": source_identity(markers_path),
        "inspectable_row_locks": {
            slug: source_identity(BENCHMARK / slug / "alignment_input_12.csv.gz")
            for _, slug in DATASETS
        },
    }
    return blocks, shared, audit


def source_indices(canonical: pd.DataFrame, dataset: str) -> np.ndarray:
    numeric = pd.to_numeric(canonical["source_index_zero_based"], errors="raise")
    indices = numeric.to_numpy(dtype=np.int64)
    if not np.array_equal(numeric.to_numpy(np.float64), indices.astype(np.float64)):
        raise AssertionError(f"{dataset}: non-integral source indices")
    if np.any(indices < 0) or len(np.unique(indices)) != len(indices):
        raise AssertionError(f"{dataset}: invalid source indices")
    return indices


def save_dataset(
    *,
    dataset: str,
    slug: str,
    values: np.ndarray,
    panel: list[tuple[str, str]],
    observed_source_ids: np.ndarray,
    canonical: pd.DataFrame,
    expected_shared: np.ndarray,
    source_path: Path,
    source_scale: str,
    source_id_recipe: str,
    excluded_features: list[str],
    stage: Path,
) -> dict[str, object]:
    values = np.asarray(values, dtype=np.float64)
    markers = [canonical_name for canonical_name, _ in panel]
    source_markers = [source_name for _, source_name in panel]
    if values.shape != (EXPECTED_CELLS, len(markers)):
        raise AssertionError(f"{dataset}: source-active shape changed: {values.shape}")
    if len(set(markers)) != len(markers) or len(set(source_markers)) != len(source_markers):
        raise AssertionError(f"{dataset}: duplicate marker mapping")
    if not np.isfinite(values).all():
        raise AssertionError(f"{dataset}: non-finite source values")
    if np.any(values.std(axis=0, ddof=1) < 1e-10):
        raise AssertionError(f"{dataset}: constant source feature")

    observed_source_ids = np.asarray(observed_source_ids, dtype=str)
    expected_source_ids = canonical["source_cell_id"].astype(str).to_numpy()
    if not np.array_equal(observed_source_ids, expected_source_ids):
        mismatch = int(np.flatnonzero(observed_source_ids != expected_source_ids)[0])
        raise AssertionError(f"{dataset}: source identity mismatch at row {mismatch}")
    marker_to_index = {marker: index for index, marker in enumerate(markers)}
    if any(marker not in marker_to_index for marker in SHARED):
        raise AssertionError(f"{dataset}: source panel lacks shared marker")
    active_shared = values[:, [marker_to_index[marker] for marker in SHARED]]
    maximum_error = float(np.max(np.abs(active_shared - expected_shared)))
    if maximum_error > 1e-4:
        raise AssertionError(
            f"{dataset}: provider/canonical shared reconstruction error {maximum_error}"
        )

    destination = stage / slug
    destination.mkdir()
    np.save(destination / "active_source_float64.npy", values, allow_pickle=False)
    (destination / "active_markers.txt").write_text(
        "".join(f"{marker}\n" for marker in markers)
    )
    deterministic_csv(
        canonical[["cell_id", "source_cell_id"]].astype(str),
        destination / "cell_ids.csv.gz",
    )
    outputs = {
        path.name: sha256(path)
        for path in destination.iterdir()
        if path.is_file()
    }
    return {
        "dataset": dataset,
        "slug": slug,
        "cells": EXPECTED_CELLS,
        "active_features": len(markers),
        "markers": markers,
        "source_markers": source_markers,
        "source_scale": source_scale,
        "source_id_recipe": source_id_recipe,
        "excluded_source_features": excluded_features,
        "maximum_canonical_shared_reconstruction_error": maximum_error,
        "source_buffer_sha256_float64": buffer_sha256(values),
        "source_file": source_identity(source_path),
        "outputs": outputs,
    }


def build_hickey(
    canonical: pd.DataFrame, expected_shared: np.ndarray, stage: Path
) -> dict[str, object]:
    dataset, slug = DATASETS[0]
    path = RAW_ROOT / "hickey_codex" / "23_10_11_Melanoma_Marker_Cell_Neighborhood.csv"
    indices = source_indices(canonical, dataset)
    order = np.argsort(indices, kind="stable")
    sorted_indices = indices[order]
    source_markers = [source for _, source in HICKEY_PANEL]
    values = np.empty((EXPECTED_CELLS, len(source_markers)), dtype=np.float64)
    observed_ids = np.empty(EXPECTED_CELLS, dtype=object)
    filled = np.zeros(EXPECTED_CELLS, dtype=bool)
    offset = 0
    dtypes: dict[str, object] = {marker: np.float64 for marker in source_markers}
    dtypes["Unnamed: 0"] = "string"
    with pd.read_csv(
        path,
        usecols=["Unnamed: 0", *source_markers],
        dtype=dtypes,
        chunksize=250_000,
    ) as reader:
        for chunk in reader:
            end = offset + len(chunk)
            left = int(np.searchsorted(sorted_indices, offset, side="left"))
            right = int(np.searchsorted(sorted_indices, end, side="left"))
            if right > left:
                local = sorted_indices[left:right] - offset
                output_positions = order[left:right]
                selected = chunk.iloc[local]
                values[output_positions] = selected[source_markers].to_numpy(np.float64)
                observed_ids[output_positions] = (
                    "hickey::" + selected["Unnamed: 0"].astype(str)
                ).to_numpy()
                filled[output_positions] = True
            offset = end
            if offset > int(sorted_indices[-1]):
                break
    if not filled.all():
        raise AssertionError("Hickey source rows are missing")
    return save_dataset(
        dataset=dataset,
        slug=slug,
        values=values,
        panel=HICKEY_PANEL,
        observed_source_ids=observed_ids,
        canonical=canonical,
        expected_shared=expected_shared,
        source_path=path,
        source_scale="provider tissue-wise marker-z-normalized CODEX values",
        source_id_recipe="'hickey::' + Unnamed: 0; zero-based CSV row lock",
        excluded_features=["DAPI", "CCL5", "CD38"],
        stage=stage,
    )


def build_hoch(
    canonical: pd.DataFrame, expected_shared: np.ndarray, stage: Path
) -> dict[str, object]:
    dataset, slug = DATASETS[1]
    path = RAW_ROOT / "hoch_imc" / "HochSchulz_2022_Melanoma_exprs_cells_by_markers.csv"
    indices = source_indices(canonical, dataset)
    source_markers = [source for _, source in HOCH_PANEL]
    dtypes: dict[str, object] = {marker: np.float64 for marker in source_markers}
    dtypes["cell_id_export"] = "string"
    frame = pd.read_csv(
        path, usecols=["cell_id_export", *source_markers], dtype=dtypes
    )
    if indices.max() >= len(frame):
        raise AssertionError("Hoch source row lock exceeds source table")
    selected = frame.iloc[indices]
    return save_dataset(
        dataset=dataset,
        slug=slug,
        values=selected[source_markers].to_numpy(np.float64),
        panel=HOCH_PANEL,
        observed_source_ids=("hoch::" + selected["cell_id_export"].astype(str)).to_numpy(),
        canonical=canonical,
        expected_shared=expected_shared,
        source_path=path,
        source_scale="provider exprs assay: asinh(counts / 1)",
        source_id_recipe="'hoch::' + cell_id_export; zero-based CSV row lock",
        excluded_features=[
            "H3", "DNA1", "DNA2", "CCL4_mRNA", "CCL18_mRNA",
            "CXCL8_mRNA", "CXCL10_mRNA", "CXCL12_mRNA", "CXCL13_mRNA",
            "CCL2_mRNA", "CCL22_mRNA", "CXCL9_mRNA", "DapB_mRNA",
            "CCL8_mRNA", "CCL19_mRNA",
        ],
        stage=stage,
    )


def build_ptacek(
    canonical: pd.DataFrame, expected_shared: np.ndarray, stage: Path
) -> dict[str, object]:
    dataset, slug = DATASETS[2]
    path = RAW_ROOT / "ptacek_mibi" / "scaled_mibi_data_w_cell_gating.csv"
    indices = source_indices(canonical, dataset)
    source_markers = [source for _, source in PTACEK_PANEL]
    dtypes: dict[str, object] = {marker: np.float64 for marker in source_markers}
    dtypes.update({"fov": "string", "label": "string"})
    frame = pd.read_csv(
        path, usecols=["fov", "label", *source_markers], dtype=dtypes
    )
    if indices.max() >= len(frame):
        raise AssertionError("Ptacek source row lock exceeds source table")
    selected = frame.iloc[indices]
    observed_ids = (
        "ptacek::" + selected["fov"].astype(str) + "::" + selected["label"].astype(str)
    ).to_numpy()
    return save_dataset(
        dataset=dataset,
        slug=slug,
        values=selected[source_markers].to_numpy(np.float64),
        panel=PTACEK_PANEL,
        observed_source_ids=observed_ids,
        canonical=canonical,
        expected_shared=expected_shared,
        source_path=path,
        source_scale="provider denoised/normalized/scaled/arcsinh MIBI values",
        source_id_recipe="'ptacek::' + fov + '::' + segmentation label; zero-based CSV row lock",
        excluded_features=["dsDNA"],
        stage=stage,
    )


def verified_existing() -> bool:
    manifest_path = OUTPUT / "manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("recipe_sha256") != sha256(Path(__file__)):
        return False
    for relative, expected in manifest.get("outputs", {}).items():
        path = OUTPUT / relative
        if not path.is_file() or sha256(path) != expected:
            return False

    def identities_match(value: object) -> bool:
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                source = Path(str(value["path"]))
                if not source.is_file() or sha256(source) != value["sha256"]:
                    return False
            return all(identities_match(child) for child in value.values())
        if isinstance(value, list):
            return all(identities_match(child) for child in value)
        return True

    return identities_match(manifest.get("canonical_inputs", {})) and identities_match(
        manifest.get("datasets", [])
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="atomically rebuild verified derived files")
    args = parser.parse_args()
    if not args.force and verified_existing():
        print(f"verified existing source-active package: {OUTPUT}")
        return

    stage = Path(tempfile.mkdtemp(prefix=".source_active.stage-", dir=HERE))
    try:
        blocks, shared, canonical_audit = read_locks()
        records = [
            build_hickey(blocks[DATASETS[0][0]], shared[DATASETS[0][0]], stage),
            build_hoch(blocks[DATASETS[1][0]], shared[DATASETS[1][0]], stage),
            build_ptacek(blocks[DATASETS[2][0]], shared[DATASETS[2][0]], stage),
        ]
        outputs = {
            str(path.relative_to(stage)): sha256(path)
            for path in sorted(stage.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "schema_version": "final_xu.maxfuse_source_active.v2",
            "status": "complete",
            "benchmark": "melanoma",
            "selection": "exact canonical frozen cells and row order; no resampling",
            "labels_or_outcomes_read": False,
            "shared_markers": SHARED,
            "canonical_inputs": canonical_audit,
            "datasets": records,
            "recipe": str(Path(__file__).resolve()),
            "recipe_sha256": sha256(Path(__file__)),
            "outputs": outputs,
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        publish(stage, OUTPUT)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(f"rebuilt source-active package: {OUTPUT}")


if __name__ == "__main__":
    main()
