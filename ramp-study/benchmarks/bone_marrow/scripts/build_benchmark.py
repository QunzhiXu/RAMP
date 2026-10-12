#!/usr/bin/env python3
"""Build the frozen 88,031-cell three-platform bone-marrow benchmark.

The cohort is identical to the BM CD19 assay-repair story:

* Shovik CODEX: all 12 normal-BM donors x 2,500 cells = 30,000;
* Levine CyTOF: healthy H1/H2 x 15,000 cells = 30,000;
* Triana AbSeq: all young BM1/BM2/BM3 cells = 28,031.

This builder reads the highest-available provider matrices from
``final_xu/raw_data/bone_marrow``. It does not copy raw source data and does
not normalize, transform, cluster, align, or use downstream performance.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from scipy.io import mmread
from paths import new_output


HERE = Path(__file__).resolve().parent
SEED = 20260820

MARKERS14 = [
    "CD117", "CD11B", "CD11C", "CD123", "CD14", "CD3", "CD33",
    "CD34", "CD38", "CD4", "CD45", "CD45RA", "CD61", "CD8",
]
MARKERS15 = MARKERS14 + ["CD19"]
ID_COLUMNS = ["cell_id", "dataset", "donor", "unit_key"]
EXPECTED_CODEX_DONORS = [
    "SB67_NBM27_H10_CODEX_Mesmer",
    "SB67_NBM28_H14_CODEX_Mesmer",
    "SB67_NBM31_H32_CODEX_Mesmer",
    "SB67_NBM32_H37_CODEX_Mesmer",
    "SB67_NBM33_H36_CODEX_Mesmer",
    "SB67_NBM34_H38_CODEX_Mesmer",
    "SB67_NBM36_H26_CODEX_Mesmer",
    "SB67_NBM37_H35_CODEX_Mesmer",
    "SB67_NBM38_H33_CODEX_Mesmer",
    "SB67_NBM39_H41_CODEX_Mesmer",
    "SB67_NBM40_H39_CODEX_Mesmer",
    "SB67_NBM41_H27_CODEX_Mesmer",
]
EXPECTED_CYTOF_DONORS = ["H1", "H2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        required=True,
        help="External provider bone-marrow root; read-only, not copied",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New output directory; existing paths are rejected",
    )
    return parser.parse_args()


def write_csv_gz(frame: pd.DataFrame, path: Path) -> None:
    """Write byte-reproducible gzip CSV (fixed gzip timestamp)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as raw_handle:
            with gzip.GzipFile(fileobj=raw_handle, mode="wb", mtime=0) as handle:
                frame.to_csv(handle, index=False)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def raw_source_files(raw_root: Path) -> list[Path]:
    """Return every raw/provider file that can affect the frozen matrices."""
    cytof_root = raw_root / "BM_CyTOF" / "levine_CyTOF_Annotated_Celltypes_fcs"
    files = [
        raw_root / "BM_Codex" / "Normal_Bone_Marrow_CODEX_Atlas_Seurat_v2.rds",
        cytof_root / "Levine_CellTypesAnnotated_sampleID.csv",
        cytof_root / "Levine_CellTypesAnnotated_channels.csv",
        raw_root / "Triana_2021" / "exported" / "ab_counts.mtx",
        raw_root / "Triana_2021" / "exported" / "ab_features.tsv",
        raw_root / "Triana_2021" / "exported" / "ab_barcodes.tsv",
        raw_root / "Triana_2021" / "exported" / "meta.csv",
    ]
    files.extend(sorted(cytof_root.glob("*.fcs")))
    return files


def r_software_versions() -> dict[str, str]:
    expression = (
        'cat(paste(c(as.character(getRversion()), '
        'as.character(packageVersion("SeuratObject")), '
        'as.character(packageVersion("Matrix"))), collapse="\\n"))'
    )
    values = subprocess.check_output(
        ["Rscript", "-e", expression], text=True, cwd=HERE
    ).strip().splitlines()
    if len(values) != 3:
        raise RuntimeError(f"Unexpected R version output: {values}")
    return {"R": values[0], "SeuratObject": values[1], "Matrix": values[2]}


def canonical_marker(value: object) -> str:
    compact = "".join(character for character in str(value).upper() if character.isalnum())
    aliases = {"CD3E": "CD3", "CD3": "CD3", "CD8A": "CD8", "CD8": "CD8"}
    return aliases.get(compact, compact)


def exact_uniform_indices(groups: np.ndarray, per_group: int, seed: int) -> np.ndarray:
    groups = np.asarray(groups, dtype=str)
    selected: list[int] = []
    for position, name in enumerate(sorted(pd.unique(groups))):
        candidates = np.flatnonzero(groups == name)
        if len(candidates) < per_group:
            raise ValueError(f"{name}: {len(candidates):,} cells < quota {per_group:,}")
        rng = np.random.default_rng(seed + 104729 * (position + 1))
        selected.extend(rng.choice(candidates, per_group, replace=False).tolist())
    return np.sort(np.asarray(selected, dtype=int))


def parse_fcs(path: Path) -> tuple[np.ndarray, list[str]]:
    """Read the float event matrix needed from the local FCS 3.0 files."""
    with path.open("rb") as handle:
        header = handle.read(58)
        text_start, text_end = int(header[10:18]), int(header[18:26])
        data_start = int(header[26:34])
        handle.seek(text_start)
        text = handle.read(text_end - text_start + 1).decode("latin1")
    fields = text.split(text[0])
    meta = {
        fields[position]: fields[position + 1]
        for position in range(1, len(fields) - 1, 2)
        if fields[position]
    }
    n_events, n_parameters = int(meta["$TOT"]), int(meta["$PAR"])
    dtype = ">f4" if meta["$BYTEORD"] == "4,3,2,1" else "<f4"
    if data_start == 0:
        data_start = int(meta["$BEGINDATA"])
    with path.open("rb") as handle:
        handle.seek(data_start)
        values = np.fromfile(handle, dtype=dtype, count=n_events * n_parameters)
    channels = [meta[f"$P{i}N"] for i in range(1, n_parameters + 1)]
    return values.reshape(n_events, n_parameters).astype(np.float32), channels


def write_platform_inputs(source15: pd.DataFrame, locked: pd.DataFrame, out: Path) -> None:
    if source15.columns.tolist() != ID_COLUMNS + MARKERS15:
        raise AssertionError(f"Unexpected source columns for {out.name}")
    locked_core = ["cell_id", "dataset", "donor"]
    if not set(locked_core).issubset(locked.columns):
        raise AssertionError(f"Locked metadata lacks canonical fields for {out.name}")
    if not source15[locked_core].astype(str).equals(locked[locked_core].astype(str)):
        raise AssertionError(f"Matrix/metadata identity or order mismatch for {out.name}")
    write_csv_gz(source15, out / "source_input_15.csv.gz")
    write_csv_gz(source15[ID_COLUMNS + MARKERS14], out / "alignment_input_14.csv.gz")
    write_csv_gz(locked, out / "locked_metadata.csv.gz")


def build_codex(raw_root: Path, stage: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    source_rds = raw_root / "BM_Codex" / "Normal_Bone_Marrow_CODEX_Atlas_Seurat_v2.rds"
    out = stage / "bm_codex"
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["Rscript", str(HERE / "export_codex_30k.R"), str(source_rds), str(out)],
        check=True,
        cwd=HERE,
    )
    source15 = pd.read_csv(out / "source_input_15.csv", low_memory=False)
    locked = pd.read_csv(out / "locked_metadata.csv", low_memory=False)
    if len(source15) != 30_000 or len(locked) != 30_000:
        raise AssertionError("CODEX frozen sample is not 30,000 cells")
    observed_donors = sorted(source15["donor"].astype(str).unique().tolist())
    if observed_donors != EXPECTED_CODEX_DONORS:
        raise AssertionError(
            f"CODEX donor identity drift: expected {EXPECTED_CODEX_DONORS}, "
            f"observed {observed_donors}"
        )
    write_platform_inputs(source15, locked, out)
    (out / "source_input_15.csv").unlink()
    (out / "locked_metadata.csv").unlink()

    technical_labels = {"Artifact", "Autofluorescent", "Undetermined", "CD44+ Undetermined"}
    technical = (
        locked["cluster_anno_l2"].astype(str).isin(technical_labels)
        if "cluster_anno_l2" in locked
        else pd.Series(False, index=locked.index)
    )
    technical_counts = {
        str(key): int(value)
        for key, value in locked.loc[technical, "cluster_anno_l2"].value_counts().items()
    }
    technical_counts["total"] = int(technical.sum())
    technical_counts["retained_for_exact_story_reproduction"] = True
    return source15, technical_counts


def build_cytof(raw_root: Path, stage: Path) -> pd.DataFrame:
    source_root = raw_root / "BM_CyTOF" / "levine_CyTOF_Annotated_Celltypes_fcs"
    samples = pd.read_csv(source_root / "Levine_CellTypesAnnotated_sampleID.csv")
    observed_donors = sorted(samples["patient"].astype(str).unique().tolist())
    if observed_donors != EXPECTED_CYTOF_DONORS:
        raise AssertionError(
            f"CyTOF donor identity drift: expected {EXPECTED_CYTOF_DONORS}, "
            f"observed {observed_donors}"
        )
    files_per_donor = samples.groupby(samples["patient"].astype(str)).size().to_dict()
    if files_per_donor != {"H1": 7, "H2": 7}:
        raise AssertionError(f"Expected seven annotated FCS files per CyTOF donor: {files_per_donor}")
    channels = pd.read_csv(
        source_root / "Levine_CellTypesAnnotated_channels.csv", dtype=str
    ).fillna("")
    marker_by_channel = channels.set_index("channel")["marker"].astype(str).to_dict()

    matrices, donor_parts, label_parts, id_parts, file_parts = [], [], [], [], []
    for source in samples.itertuples(index=False):
        matrix, source_channels = parse_fcs(source_root / str(source.sample_id))
        lookup = {
            canonical_marker(marker_by_channel[channel]): index
            for index, channel in enumerate(source_channels)
        }
        missing = [marker for marker in MARKERS15 if marker not in lookup]
        if missing:
            raise RuntimeError(f"CyTOF missing markers: {missing}")
        matrices.append(matrix[:, [lookup[m] for m in MARKERS15]].astype(np.float64))
        n = len(matrix)
        donor_parts.append(np.repeat(str(source.patient), n))
        label_parts.append(np.repeat(str(source.complexcelltype), n))
        file_parts.append(np.repeat(str(source.sample_id), n))
        id_parts.append(np.asarray([
            f"{source.sample_id}::event_{event}" for event in range(1, n + 1)
        ], dtype=str))

    values = np.vstack(matrices)
    donors = np.concatenate(donor_parts)
    labels = np.concatenate(label_parts)
    source_files = np.concatenate(file_parts)
    cell_ids = np.concatenate(id_parts)
    selected = exact_uniform_indices(donors, 15_000, SEED + 1)

    source15 = pd.DataFrame(values[selected], columns=MARKERS15)
    source15.insert(0, "unit_key", "BM CyTOF|" + donors[selected])
    source15.insert(0, "donor", donors[selected])
    source15.insert(0, "dataset", "BM CyTOF")
    source15.insert(0, "cell_id", cell_ids[selected])
    locked = pd.DataFrame({
        "cell_id": cell_ids[selected],
        "dataset": "BM CyTOF",
        "donor": donors[selected],
        "source_fcs": source_files[selected],
        "provider_celltype": labels[selected],
    })
    write_platform_inputs(source15, locked, stage / "bm_cytof")
    return source15


def build_triana(raw_root: Path, stage: Path) -> pd.DataFrame:
    source_root = raw_root / "Triana_2021" / "exported"
    features = [
        line.strip() for line in (source_root / "ab_features.tsv").read_text().splitlines()
        if line.strip()
    ]
    barcodes = [
        line.strip() for line in (source_root / "ab_barcodes.tsv").read_text().splitlines()
        if line.strip()
    ]
    obs = pd.read_csv(source_root / "meta.csv", low_memory=False)
    if barcodes != obs["barcode"].astype(str).tolist():
        raise RuntimeError("Triana matrix/metadata order mismatch")
    donors = obs["Batch"].astype(str).to_numpy()
    selected = np.flatnonzero(np.isin(donors, ["BM1", "BM2", "BM3"]))
    if len(selected) != 28_031:
        raise RuntimeError(f"Expected 28,031 young Triana cells; found {len(selected):,}")

    counts = mmread(source_root / "ab_counts.mtx")
    counts = counts.T.tocsr() if sparse.issparse(counts) else sparse.csr_matrix(np.asarray(counts).T)
    lookup = {
        canonical_marker(name.removesuffix("-AB")): index
        for index, name in enumerate(features)
    }
    missing = [marker for marker in MARKERS15 if marker not in lookup]
    if missing:
        raise RuntimeError(f"Triana missing markers: {missing}")
    values = counts[selected][:, [lookup[m] for m in MARKERS15]].toarray().astype(np.float64)
    selected_barcodes = np.asarray(barcodes, dtype=str)[selected]

    source15 = pd.DataFrame(values, columns=MARKERS15)
    source15.insert(0, "unit_key", "Triana AbSeq|" + donors[selected])
    source15.insert(0, "donor", donors[selected])
    source15.insert(0, "dataset", "Triana AbSeq")
    source15.insert(0, "cell_id", selected_barcodes)
    locked = pd.DataFrame({
        "cell_id": selected_barcodes,
        "dataset": "Triana AbSeq",
        "donor": donors[selected],
        "provider_celltype": obs.iloc[selected]["CellTypes"].astype(str).to_numpy(),
    })
    write_platform_inputs(source15, locked, stage / "triana_abseq")
    return source15


def copy_new_file(source: Path, target: Path) -> None:
    """Copy without truncating existing files, hard links, or symlink targets."""
    source, target = Path(source), Path(target)
    with source.open("rb") as reader, target.open("xb") as writer:
        shutil.copyfileobj(reader, writer)
    shutil.copystat(source, target)


def copy_new_directory(source: Path, target: Path) -> None:
    """Create a new tree only; never replace or delete an existing tree."""
    shutil.copytree(source, target, copy_function=copy_new_file, dirs_exist_ok=False)


def publish(stage: Path, output_dir: Path) -> None:
    # The executing task has already claimed this directory with new_output().
    # Reject repeat publication; exclusive copy operations also reject entries
    # created after this preflight. Partial outputs remain for inspection.
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise ValueError(f"Output must be a claimed directory: {output_dir}")
    if any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to publish into nonempty output: {output_dir}")
    for relative in ["bm_codex", "bm_cytof", "triana_abseq"]:
        copy_new_directory(stage / relative, output_dir / relative)
    top_files = [
        "selected_cells.csv.gz", "donor_cell_counts.csv", "marker_manifest.csv",
        "dataset_manifest.csv",
    ]
    for relative in top_files:
        copy_new_file(stage / relative, output_dir / relative)
    # Publish the manifest last. An interrupted copy is incomplete, never a
    # replacement of a successful older run.
    copy_new_file(stage / "manifest.json", output_dir / "manifest.json")


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    output_dir = new_output(args.output_dir, [raw_root])
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    required = [
        raw_root / "BM_Codex" / "Normal_Bone_Marrow_CODEX_Atlas_Seurat_v2.rds",
        raw_root / "BM_CyTOF" / "levine_CyTOF_Annotated_Celltypes_fcs",
        raw_root / "Triana_2021" / "exported" / "ab_counts.mtx",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required source(s):\n" + "\n".join(missing))

    with tempfile.TemporaryDirectory(prefix="bm_benchmark_", dir=output_dir.parent) as temporary:
        stage = Path(temporary)
        codex, technical_counts = build_codex(raw_root, stage)
        cytof = build_cytof(raw_root, stage)
        triana = build_triana(raw_root, stage)
        frames = [codex, cytof, triana]

        selected = pd.concat(
            [frame[ID_COLUMNS[:3]] for frame in frames], ignore_index=True
        )
        write_csv_gz(selected, stage / "selected_cells.csv.gz")
        donor_counts = (
            selected.groupby(["dataset", "donor"], sort=True)
            .size().rename("n_cells").reset_index()
        )
        donor_counts.to_csv(stage / "donor_cell_counts.csv", index=False)

        marker_manifest = pd.DataFrame({
            "marker": MARKERS15,
            "role": ["alignment_input"] * 14 + ["failed_target"],
            "used_for_initial_alignment": [True] * 14 + [False],
            "observed_codex_value_allowed_in_initial_fit": [True] * 14 + [False],
        })
        marker_manifest.to_csv(stage / "marker_manifest.csv", index=False)

        dataset_manifest = pd.DataFrame([
            {
                "dataset": "BM CODEX", "assay": "CODEX",
                "source_scale": "provider-processed CODEX counts-slot cell-level intensity",
                "selected_donors": 12, "selected_cells": 30_000,
                "selection": "all donors; 2500 random cells per donor",
            },
            {
                "dataset": "BM CyTOF", "assay": "CyTOF",
                "source_scale": "provider cct_subtracted_normalized pre-gated FCS values",
                "selected_donors": 2, "selected_cells": 30_000,
                "selection": "H1/H2; 15000 random cells per donor within provider-pre-gated pool",
            },
            {
                "dataset": "Triana AbSeq", "assay": "AbSeq",
                "source_scale": "raw antibody UMI counts",
                "selected_donors": 3, "selected_cells": 28_031,
                "selection": "all cells from young BM1/BM2/BM3",
            },
        ])
        dataset_manifest.to_csv(stage / "dataset_manifest.csv", index=False)

        totals = selected.groupby("dataset").size().astype(int).to_dict()
        expected = {"BM CODEX": 30_000, "BM CyTOF": 30_000, "Triana AbSeq": 28_031}
        if totals != expected:
            raise AssertionError(f"Unexpected platform totals: {totals}")

        output_files = sorted(
            path for path in stage.rglob("*")
            if path.is_file() and path.name != "manifest.json"
        )
        manifest = {
            "package": "bone_marrow_three_platform_benchmark_v1",
            "source_root": str(raw_root),
            "sampling_seed": SEED,
            "n_total_cells": int(len(selected)),
            "platform_totals": totals,
            "sampling": {
                "BM CODEX": "all 12 normal-BM donors x 2,500 random cells",
                "BM CyTOF": "healthy H1/H2 x 15,000 random cells within provider-pre-gated pool",
                "Triana AbSeq": "all young BM1/BM2/BM3 cells",
            },
            "markers_retained_in_source_input": MARKERS15,
            "markers_used_for_initial_alignment": MARKERS14,
            "heldout_target": "CD19",
            "failed_codex_cd19_policy": (
                "retained only in source_input_15 for assay-failure QC; excluded from "
                "alignment_input_14 and forbidden in initial alignment/prediction"
            ),
            "sampling_used_expression_or_downstream_performance": False,
            "provider_labels_and_spatial_coordinates_locked": True,
            "codex_provider_technical_cells": technical_counts,
            "normalization_applied_by_builder": False,
            "source_files_sha256": {
                str(path.relative_to(raw_root)): sha256(path)
                for path in raw_source_files(raw_root)
            },
            "recipe_sha256": {
                "build_benchmark.py": sha256(Path(__file__).resolve()),
                "export_codex_30k.R": sha256(HERE / "export_codex_30k.R"),
            },
            "software_versions": {
                "Python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scipy": scipy.__version__,
                **r_software_versions(),
            },
            "files_sha256": {
                str(path.relative_to(stage)): sha256(path) for path in output_files
            },
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        publish(stage, output_dir)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
