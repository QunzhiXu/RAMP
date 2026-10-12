#!/usr/bin/env python3
"""Package the frozen 60,000-cell melanoma three-platform benchmark."""

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
DEFAULT_SOURCE_INPUT = (
    FINAL_XU / "test" / "melanoma_metadata_matched_20k_20260827" / "input"
)
DATASETS = [
    "MEL_Hickey2023_CODEX",
    "MEL_Hoch2021_IMC",
    "MEL_Ptacek2021_MIBI",
]
SLUGS = {
    DATASETS[0]: "hickey_codex",
    DATASETS[1]: "hoch_imc",
    DATASETS[2]: "ptacek_mibi",
}
DISPLAY = {
    DATASETS[0]: "Hickey CODEX",
    DATASETS[1]: "Hoch IMC",
    DATASETS[2]: "Ptacek MIBI",
}
EXPECTED_CELLS = {dataset: 20_000 for dataset in DATASETS}
EXPECTED_UNITS = {DATASETS[0]: 6, DATASETS[1]: 17, DATASETS[2]: 23}
EXPECTED_PATIENTS = EXPECTED_UNITS.copy()
MARKERS = [
    "CD163", "CD3", "CD31", "CD68", "CD8", "HLA_DR",
    "LAG3", "PD1", "PDL1", "SMA", "SOX10", "VIMENTIN",
]
SOURCE_MARKERS = {
    DATASETS[0]: dict(zip(MARKERS, [
        "CD163", "CD3", "CD31", "CD68", "CD8", "HLA-DR",
        "LAG3", "PD-1", "PD-L1", "aSMA", "SOX10/S100", "Vimentin",
    ])),
    DATASETS[1]: dict(zip(MARKERS, [
        "CD163", "CD3e", "CD31", "CD68", "CD8a", "HLA_DR",
        "LAG3", "PD_1", "PD_L1", "SMA", "SOX10", "VIM",
    ])),
    DATASETS[2]: dict(zip(MARKERS, [
        "CD163", "CD3", "CD31", "CD68", "CD8", "HLA DPDQDR",
        "LAG3", "PD-1", "PD-L1", "SMA", "SOX10", "Vimentin",
    ])),
}
SELECTION = {
    DATASETS[0]: (
        "all six paired-study patients; pretreatment specimen only; "
        "patient-balanced; acquisition-stratum proportional"
    ),
    DATASETS[1]: (
        "melanoma tissue; exact Stage III; treatment-naive before surgery; "
        "all 17 eligible patients; patient- then image-balanced"
    ),
    DATASETS[2]: (
        "exact gross Stage III; all 23 eligible FOV/patients; "
        "capacity-aware patient balancing"
    ),
}
ALLOWED_FINE_LABELS = {
    "Non-joint", "Epithelial (non-joint)", "CD4 T", "CD8 T",
    "Other T / CD8−", "NK", "Endothelial", "Myeloid", "Stroma", "Tumor",
}
EXPECTED_SOURCE_SHA256 = {
    "raw.npy": "eedadfd47dfe09daf1760ec96cb0d8bd5eb0483d8ab0a5274fc108450877a9da",
    "metadata.csv.gz": "df3edf9cb3ca97989e241802ff894aceeef7a9ea46647929db42605389c83775",
    "markers.txt": "9314e9a8dfe93fa305cda18ddd0ad6e63a1c4c22693106f2c7ec5cef43d1b5ec",
    "input_manifest.json": "807678f72e4428146c502086f472e5651abef457496e3eac3ef48c0ecb36d066",
    "sampling_counts.csv": "2d9fb6686fca3fe15466798999f6f5c86920999712f4d42c83d87f0e383bef1c",
    "posthoc_label_counts.csv": "dbba5dd0148a293fc98d6d7990c34fea12108bde00a994e0b076899d4cc22f99",
}
MANAGED_TOP_FILES = [
    "selected_cells.csv.gz", "sampling_counts.csv", "donor_cell_counts.csv",
    "celltype_counts.csv", "label_mapping_audit.csv", "dataset_manifest.csv",
    "marker_manifest.csv", "manifest.json",
]
RELABEL_STALE_MARKER = "PARENT_REBUILD_REQUIRED.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-input", type=Path, default=DEFAULT_SOURCE_INPUT)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
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
    """Return consecutive dataset blocks, not merely first-seen categories."""
    values = series.astype(str).reset_index(drop=True)
    starts = values.ne(values.shift(fill_value="")).to_numpy()
    return values.loc[starts].tolist()


def load_and_validate(source_input: Path):
    for name, expected in EXPECTED_SOURCE_SHA256.items():
        path = source_input / name
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256(path)
        if observed != expected:
            raise AssertionError(
                f"Accepted melanoma input changed: {name}; "
                f"expected {expected}, observed {observed}"
            )
    raw = np.load(source_input / "raw.npy", allow_pickle=False)
    metadata = pd.read_csv(source_input / "metadata.csv.gz", low_memory=False)
    markers = [
        value.strip()
        for value in (source_input / "markers.txt").read_text().splitlines()
        if value.strip()
    ]
    source_manifest = json.loads((source_input / "input_manifest.json").read_text())
    sampling = pd.read_csv(source_input / "sampling_counts.csv", low_memory=False)
    if raw.shape != (60_000, 12) or raw.dtype != np.float32:
        raise AssertionError((raw.shape, raw.dtype))
    if markers != MARKERS or not np.isfinite(raw).all():
        raise AssertionError("marker contract or matrix finiteness changed")
    if len(metadata) != len(raw):
        raise AssertionError("matrix/metadata row mismatch")
    if not np.array_equal(metadata["row_id"].to_numpy(), np.arange(len(metadata))):
        raise AssertionError("row_id no longer matches matrix order")
    if metadata["source_cell_id"].astype(str).duplicated().any():
        raise AssertionError("source_cell_id is not globally unique")
    if dataset_block_order(metadata["dataset"]) != DATASETS:
        raise AssertionError("dataset blocks are not contiguous in canonical order")
    cells = metadata.groupby("dataset", sort=False).size().astype(int).to_dict()
    units = metadata.groupby("dataset", sort=False)["unit_key"].nunique().astype(int).to_dict()
    patients = metadata.groupby("dataset", sort=False)["patient_id"].nunique().astype(int).to_dict()
    if cells != EXPECTED_CELLS or units != EXPECTED_UNITS or patients != EXPECTED_PATIENTS:
        raise AssertionError((cells, units, patients))
    if not set(metadata["celltype"].astype(str)).issubset(ALLOWED_FINE_LABELS):
        raise AssertionError("unexpected post-hoc cell type")
    hickey = metadata["dataset"].astype(str).eq(DATASETS[0])
    hoch = metadata["dataset"].astype(str).eq(DATASETS[1])
    ptacek = metadata["dataset"].astype(str).eq(DATASETS[2])
    if set(metadata.loc[hickey, "timepoint"].astype(str)) != {"Pre"}:
        raise AssertionError("Hickey is not pretreatment-only")
    if set(metadata.loc[hoch, "stage"].astype(str)) != {"III"} or set(
        metadata.loc[hoch, "timepoint"].astype(str)
    ) != {"naive"}:
        raise AssertionError("Hoch is not exact Stage III/treatment-naive")
    if set(metadata.loc[ptacek, "stage"].astype(str)) != {"III"}:
        raise AssertionError("Ptacek is not exact gross Stage III")
    sampled_counts = (
        sampling.groupby("dataset", sort=False)["selected_cells"].sum().astype(int).to_dict()
    )
    if sampled_counts != EXPECTED_CELLS:
        raise AssertionError(f"sampling audit does not sum to dataset totals: {sampled_counts}")
    patient_audit = (
        sampling.groupby(["dataset", "patient_id"], sort=False)["selected_cells"]
        .sum().sort_index()
    )
    patient_metadata = (
        metadata.groupby(["dataset", "patient_id"], sort=False)
        .size().sort_index()
    )
    patient_audit.index = patient_audit.index.set_levels(
        [patient_audit.index.levels[0].astype(str), patient_audit.index.levels[1].astype(str)]
    )
    patient_metadata.index = patient_metadata.index.set_levels(
        [patient_metadata.index.levels[0].astype(str), patient_metadata.index.levels[1].astype(str)]
    )
    if not patient_audit.equals(patient_metadata.astype(int)):
        raise AssertionError("sampling audit and locked patient depths differ")
    if source_manifest.get("random_seed") != 20_260_827:
        raise AssertionError("sampling seed changed")
    if source_manifest.get("posthoc_provider_labels_used_for_fitting") is not False:
        raise AssertionError("provider labels were exposed during fitting")
    return raw, metadata, markers, source_manifest, sampling


def canonical_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    answer = metadata.copy()
    answer.insert(1, "cell_id", answer["source_cell_id"].astype(str))
    answer.insert(7, "donor", answer["patient_id"].astype(str))
    answer["celltype_provenance"] = "provider-derived upstream; frozen before alignment"
    answer["celltype_relabel_version"] = "melanoma_provider_fine_v2_60000"
    answer["labels_used_for_alignment"] = False
    if answer["cell_id"].duplicated().any():
        raise AssertionError("canonical cell_id is not unique")
    return answer


def build_files(
    stage: Path,
    source_input: Path,
    raw: np.ndarray,
    metadata: pd.DataFrame,
    source_manifest: dict,
    sampling: pd.DataFrame,
) -> None:
    canonical = canonical_metadata(metadata)
    input_dir = stage / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_input / "raw.npy", input_dir / "raw.npy")
    shutil.copyfile(source_input / "markers.txt", input_dir / "markers.txt")
    shutil.copyfile(
        source_input / "input_manifest.json",
        input_dir / "upstream_input_manifest.json",
    )
    write_csv_gz(canonical[["dataset", "unit_key"]], input_dir / "fit_metadata.csv.gz")
    write_csv_gz(canonical, input_dir / "locked_metadata.csv.gz")

    identity_columns = [
        "row_id", "cell_id", "source_cell_id", "source_index_zero_based",
        "dataset", "technology", "donor", "patient_id", "specimen_id",
        "acquisition_id", "unit_key",
    ]
    write_csv_gz(canonical[identity_columns], stage / "selected_cells.csv.gz")
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
                "dataset", "provider_celltype_original", "provider_celltype_common",
                "provider_celltype_coarse", "celltype", "celltype_joint_evaluation",
            ],
            sort=False,
            dropna=False,
        )
        .size().rename("cells").reset_index()
    )
    mapping.to_csv(stage / "label_mapping_audit.csv", index=False)

    dataset_rows = []
    for dataset in DATASETS:
        selected = canonical["dataset"].astype(str).eq(dataset).to_numpy()
        block_meta = canonical.loc[selected].reset_index(drop=True)
        block_raw = raw[selected]
        slug = SLUGS[dataset]
        block_dir = stage / slug
        block_dir.mkdir(parents=True, exist_ok=True)
        inspectable = block_meta[
            ["row_id", "cell_id", "dataset", "donor", "unit_key"]
        ].copy()
        for marker_index, marker in enumerate(MARKERS):
            inspectable[marker] = block_raw[:, marker_index]
        alignment_path = block_dir / "alignment_input_12.csv.gz"
        write_csv_gz(inspectable, alignment_path, float_format="%.9g")
        write_csv_gz(block_meta, block_dir / "locked_metadata.csv.gz")
        roundtrip = pd.read_csv(alignment_path, usecols=MARKERS).to_numpy(np.float64)
        if not np.allclose(roundtrip, block_raw, rtol=2e-8, atol=2e-8):
            raise AssertionError(f"{dataset}: inspectable marker CSV changed values")
        dataset_rows.append({
            "dataset": dataset,
            "display_name": DISPLAY[dataset],
            "technology": str(block_meta["technology"].iloc[0]),
            "patients": int(block_meta["patient_id"].nunique()),
            "fit_units": int(block_meta["unit_key"].nunique()),
            "specimens": int(block_meta["specimen_id"].nunique()),
            "acquisitions": int(block_meta["acquisition_id"].nunique()),
            "cells": int(len(block_meta)),
            "selection": SELECTION[dataset],
        })
    pd.DataFrame(dataset_rows).to_csv(stage / "dataset_manifest.csv", index=False)

    marker_rows = []
    for marker in MARKERS:
        for dataset in DATASETS:
            marker_rows.append({
                "canonical_marker": marker,
                "dataset": dataset,
                "source_column": SOURCE_MARKERS[dataset][marker],
                "note": (
                    "SOX10 proxy combines SOX10/S100"
                    if dataset == DATASETS[0] and marker == "SOX10"
                    else "highest-available provider-processed value"
                ),
            })
    pd.DataFrame(marker_rows).to_csv(stage / "marker_manifest.csv", index=False)

    files = [
        path for path in stage.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    ]
    manifest = {
        "package": "melanoma_three_platform_benchmark_v2_60000",
        "supersedes": "fixed-ID 61,221-cell melanoma benchmark",
        "accepted_source_input": {
            "path": str(source_input.resolve()),
            "sha256": EXPECTED_SOURCE_SHA256,
        },
        "canonical_matrix": "input/raw.npy",
        "fit_metadata": "input/fit_metadata.csv.gz",
        "fit_metadata_columns": ["dataset", "unit_key"],
        "locked_metadata": "input/locked_metadata.csv.gz",
        "cells": 60_000,
        "markers": MARKERS,
        "dataset_order": DATASETS,
        "dataset_cells": EXPECTED_CELLS,
        "dataset_patients": EXPECTED_PATIENTS,
        "dataset_units": EXPECTED_UNITS,
        "selection": SELECTION,
        "sampling_seed": 20_260_827,
        "sampling_rng": "NumPy SeedSequence/default_rng; no cell-ID hashing",
        "sampling_used_expression_labels_outcomes_or_umap": False,
        "reroll_after_alignment_or_umap": False,
        "input_scale": "highest-available provider-processed source scale",
        "normalization_or_alignment_applied_by_builder": False,
        "provider_labels_available_to_fit": False,
        "celltype_policy": "provider-derived fine labels plus joint evaluation labels, post hoc only",
        "marker_caveat": "Hickey SOX10/S100 is the shared SOX10 proxy",
        "upstream_selection_manifest": source_manifest,
        "files_sha256": {
            str(path.relative_to(stage)): sha256(path) for path in sorted(files)
        },
        "recipe_sha256": {
            "build_benchmark.py": sha256(Path(__file__).resolve()),
            "README.md": sha256(HERE / "README.md"),
        },
        "software_versions": {
            "Python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    (stage / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def validate_stage(stage: Path) -> None:
    raw = np.load(stage / "input" / "raw.npy", allow_pickle=False)
    locked = pd.read_csv(stage / "input" / "locked_metadata.csv.gz", low_memory=False)
    fit = pd.read_csv(stage / "input" / "fit_metadata.csv.gz", low_memory=False)
    if raw.shape != (60_000, 12) or len(locked) != len(raw) or len(fit) != len(raw):
        raise AssertionError("staged package dimensions differ")
    if fit.columns.tolist() != ["dataset", "unit_key"]:
        raise AssertionError("fit metadata exposes extra columns")
    if not np.array_equal(locked["row_id"].to_numpy(), np.arange(len(raw))):
        raise AssertionError("staged locked metadata row order differs")
    if dataset_block_order(locked["dataset"]) != DATASETS:
        raise AssertionError("staged dataset blocks are not contiguous")
    manifest = json.loads((stage / "manifest.json").read_text())
    for relative, expected in manifest["files_sha256"].items():
        if sha256(stage / relative) != expected:
            raise AssertionError(f"staged hash differs: {relative}")


def replace_directory(source: Path, target: Path) -> None:
    """Replace one managed directory while retaining the old copy on failure."""
    replacement = Path(tempfile.mkdtemp(prefix=f".{target.name}.new.", dir=target.parent))
    backup = target.parent / f".{target.name}.old.{os.getpid()}"
    try:
        shutil.copytree(source, replacement, dirs_exist_ok=True)
        if backup.exists():
            shutil.rmtree(backup)
        if target.exists():
            os.replace(target, backup)
        try:
            os.replace(replacement, target)
        except Exception:
            if backup.exists() and not target.exists():
                os.replace(backup, target)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if replacement.exists():
            shutil.rmtree(replacement)


def replace_file(source: Path, target: Path) -> None:
    """Publish a managed file with an atomic same-filesystem rename."""
    temporary = target.parent / f".{target.name}.new.{os.getpid()}"
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def update_relabel_stale_marker(output_dir: Path) -> None:
    """Make a preserved child relabel visibly stale when its parent changes."""
    relabel_dir = output_dir / "relabel"
    child_manifest_path = relabel_dir / "manifest.json"
    marker = relabel_dir / RELABEL_STALE_MARKER
    if not child_manifest_path.is_file():
        return
    try:
        child = json.loads(child_manifest_path.read_text())
        child_parent_hash = child["source"]["parent_manifest_sha256"]
    except (KeyError, TypeError, json.JSONDecodeError):
        child_parent_hash = None
    current_parent_hash = sha256(output_dir / "manifest.json")
    if child_parent_hash != current_parent_hash:
        marker.write_text(
            "The parent melanoma benchmark changed. Rebuild this relabel package with\n"
            "python3 final_xu/benchmark_data/melanoma/relabel/build_relabel.py\n"
        )
    elif marker.exists():
        marker.unlink()


def publish(stage: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Leave recipe files, legacy/, and relabel/ untouched. Replace each managed
    # directory without retaining stale files, and publish the manifest last.
    for name in ["input", *SLUGS.values()]:
        replace_directory(stage / name, output_dir / name)
    for name in MANAGED_TOP_FILES:
        if name == "manifest.json":
            continue
        replace_file(stage / name, output_dir / name)
    replace_file(stage / "manifest.json", output_dir / "manifest.json")
    update_relabel_stale_marker(output_dir)


def main() -> None:
    args = parse_args()
    source_input = args.source_input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    raw, metadata, _, source_manifest, sampling = load_and_validate(source_input)
    with tempfile.TemporaryDirectory(
        prefix=".melanoma_benchmark_", dir=output_dir.parent
    ) as temporary:
        stage = Path(temporary) / "package"
        stage.mkdir(parents=True)
        build_files(stage, source_input, raw, metadata, source_manifest, sampling)
        validate_stage(stage)
        publish(stage, output_dir)
        validate_stage(output_dir)
    print(
        f"Published melanoma benchmark: 60,000 cells x 12 markers to {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
