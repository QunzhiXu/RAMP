#!/usr/bin/env python3
"""Package the frozen 62,701-cell TNBC three-platform benchmark.

The accepted source input keeps the frozen Tietscher IMC and Shiao CODEX blocks
and replaces Meyer IMC with the coverage-informed five-patient draw selected by
the locked seed-screen rule.  This builder does not normalize, transform, align,
cluster, or derive new cell types.  In particular, the exploratory SMA-based
Stroma split is not part of this benchmark package.
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
DEFAULT_SOURCE_INPUT = (
    FINAL_XU
    / "test"
    / "tnbc_meyer5_coverage_guided_seed20260960"
    / "input"
)

DATASETS = ["Tietscher IMC", "Meyer IMC", "Shiao CODEX"]
SLUGS = {
    "Tietscher IMC": "tietscher_imc",
    "Meyer IMC": "meyer_imc",
    "Shiao CODEX": "shiao_codex",
}
EXPECTED_CELLS = {
    "Tietscher IMC": 20_000,
    "Meyer IMC": 22_701,
    "Shiao CODEX": 20_000,
}
EXPECTED_TOTAL = sum(EXPECTED_CELLS.values())
EXPECTED_UNITS = {
    "Tietscher IMC": 1,
    "Meyer IMC": 5,
    "Shiao CODEX": 5,
}
EXPECTED_DONOR_COUNTS = {
    "Tietscher IMC": {"BB165": 20_000},
    "Meyer IMC": {
        "B12.02518": 3_364,
        "B12.64192": 3_289,
        "B13.50280": 5_594,
        "B15.54100": 4_300,
        "B15.60754": 6_154,
    },
    "Shiao CODEX": {
        "h05": 4_000,
        "h09": 4_000,
        "h12": 4_000,
        "h17": 4_000,
        "h30": 4_000,
    },
}
MARKERS = [
    "CD11C", "CD20", "CD3", "CD31", "CD4", "CD68", "CD8",
    "HLA_DR", "KI67", "PANCK", "SMA",
]
ALLOWED_BROAD_LABELS = {
    "Non-joint", "B lineage", "CD4 T", "CD8 T", "Endothelial",
    "Myeloid", "NK", "Stroma", "Tumor / epithelial",
}
EXPECTED_SOURCE_SHA256 = {
    "raw.npy": "f4b8b9e7e20fe80c005fc3fef549b6a2f5e60b0b03c1d016f8449b0625ff5c10",
    "metadata.csv.gz": "b03e8169de792cfa8afcc9975bbbfb097cd602aeee762ba5317f3a0166ab8465",
    "markers.txt": "39bb80c51c6b7b246d915f2f524def088b056c51a9a083e2db44ea7fe79cce77",
    "input_manifest.json": "aea2e9ffcef68a2d50d9e5b4fc9eec80a8d4ad09e931b311e220690eeca6ec35",
}
EXPECTED_MEYER_SAMPLING = {
    "rng": "numpy.random.default_rng",
    "seed": 20260960,
    "selection_mode": "coverage_guided",
    "selected_donors_sorted": [
        "B12.02518", "B12.64192", "B13.50280", "B15.54100", "B15.60754",
    ],
    "draw_order": [
        "B13.50280", "B15.54100", "B12.64192", "B15.60754", "B12.02518",
    ],
    "selected_meyer_cells": 22_701,
    "all_cells_from_selected_donors_retained": True,
    "provider_celltype_used_for_seed_selection": True,
    "seed_screen_completed_before_alignment_or_umap": True,
    "clinical_filters": False,
}
SHIAO_SAMPLING = {
    "donor_selection_rng": "numpy.default_rng",
    "donor_selection_seed": 20260827,
    "selected_donors_sorted": ["h05", "h09", "h12", "h17", "h30"],
    "cells_per_donor": 4_000,
    "cell_selection_rng": "one numpy.default_rng stream in sorted selected-donor order",
    "cell_selection_seed": 20260828,
    "reroll_after_results": False,
    "celltype_used_for_selection": False,
    "response_or_pcr_used_for_selection": False,
}
SOURCE_MARKERS = {
    "Tietscher IMC": {
        "CD11C": "CD11c", "CD20": "CD20", "CD3": "CD3",
        "CD31": "CD31_vWF", "CD4": "CD4", "CD68": "CD68",
        "CD8": "CD8a", "HLA_DR": "HLA-DR", "KI67": "Ki-67",
        "PANCK": "panCK", "SMA": "SMA",
    },
    "Meyer IMC": {
        "CD11C": "CD11c", "CD20": "CD20", "CD3": "CD3",
        "CD31": "CD31_vWF", "CD4": "CD4", "CD68": "CD68",
        "CD8": "CD8a", "HLA_DR": "HLADR", "KI67": "Ki67",
        "PANCK": "panCK", "SMA": "SMA",
    },
    "Shiao CODEX": {
        "CD11C": "mean(CD11c_nuclei_mean,CD11c_membrane_mean)",
        "CD20": "mean(CD20_nuclei_mean,CD20_membrane_mean)",
        "CD3": "mean(CD3e_nuclei_mean,CD3e_membrane_mean)",
        "CD31": "mean(CD31_nuclei_mean,CD31_membrane_mean)",
        "CD4": "mean(CD4_nuclei_mean,CD4_membrane_mean)",
        "CD68": "mean(CD68_nuclei_mean,CD68_membrane_mean)",
        "CD8": "mean(CD8_nuclei_mean,CD8_membrane_mean)",
        "HLA_DR": "mean(HLA-DR_nuclei_mean,HLA-DR_membrane_mean)",
        "KI67": "mean(Ki-67_nuclei_mean,Ki-67_membrane_mean)",
        "PANCK": "mean(PanCytoK_nuclei_mean,PanCytoK_membrane_mean)",
        "SMA": "mean(aSMA_nuclei_mean,aSMA_membrane_mean)",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-input",
        type=Path,
        default=DEFAULT_SOURCE_INPUT,
        help=f"accepted frozen input directory (default: {DEFAULT_SOURCE_INPUT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE,
        help=f"benchmark output directory (default: {HERE})",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


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


def load_and_validate(source_input: Path):
    for name, expected in EXPECTED_SOURCE_SHA256.items():
        path = source_input / name
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256(path)
        if observed != expected:
            raise AssertionError(
                f"Accepted TNBC source changed: {name}; expected {expected}, observed {observed}"
            )

    raw = np.load(source_input / "raw.npy", allow_pickle=False)
    metadata = pd.read_csv(source_input / "metadata.csv.gz", low_memory=False)
    source_manifest = json.loads((source_input / "input_manifest.json").read_text())
    markers = [
        value.strip()
        for value in (source_input / "markers.txt").read_text().splitlines()
        if value.strip()
    ]
    if markers != MARKERS:
        raise AssertionError(f"Marker order changed: {markers}")
    if raw.shape != (EXPECTED_TOTAL, len(MARKERS)) or len(metadata) != EXPECTED_TOTAL:
        raise AssertionError((raw.shape, len(metadata)))
    if not np.isfinite(raw).all() or np.any(raw < 0):
        raise AssertionError("Source matrix must be finite and nonnegative")
    if not np.array_equal(metadata["cell_order"].to_numpy(), np.arange(len(metadata))):
        raise AssertionError("cell_order no longer matches matrix rows")
    if list(pd.unique(metadata["dataset"].astype(str))) != DATASETS:
        raise AssertionError("Dataset order changed")
    cells = metadata.groupby("dataset", sort=False).size().astype(int).to_dict()
    units = (
        metadata.groupby("dataset", sort=False)["unit_key"]
        .nunique().astype(int).to_dict()
    )
    if cells != EXPECTED_CELLS or units != EXPECTED_UNITS:
        raise AssertionError(f"Cohort composition changed: cells={cells}, units={units}")
    for dataset in DATASETS:
        observed = (
            metadata.loc[metadata["dataset"].eq(dataset), "donor_id"]
            .astype(str).value_counts(sort=False).astype(int).to_dict()
        )
        if observed != EXPECTED_DONOR_COUNTS[dataset]:
            raise AssertionError(f"{dataset} donor counts changed: {observed}")
    labels = set(metadata["display_label"].astype(str))
    if not labels.issubset(ALLOWED_BROAD_LABELS):
        raise AssertionError(f"Unexpected derived/fine labels: {sorted(labels)}")
    if metadata["labels_used_for_alignment"].astype(bool).any():
        raise AssertionError("Provider labels were marked as alignment inputs")

    if source_manifest.get("canonical_tietscher_unchanged") is not True:
        raise AssertionError("Tietscher block was not certified unchanged")
    if source_manifest.get("canonical_shiao_unchanged") is not True:
        raise AssertionError("Shiao block was not certified unchanged")
    if source_manifest.get("provider_labels_used_for_alignment") is not False:
        raise AssertionError("Provider labels were exposed to alignment")
    sampling = source_manifest["sampling"]
    for key, expected in EXPECTED_MEYER_SAMPLING.items():
        if sampling.get(key) != expected:
            raise AssertionError(f"Sampling field changed: {key}={sampling.get(key)}")
    return raw.astype(np.float64, copy=False), metadata, source_manifest


def canonical_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    answer = metadata.copy()
    slug = answer["dataset"].map(SLUGS)
    if slug.isna().any():
        raise AssertionError("Missing dataset slug")
    expected_cell_id = slug.astype(str) + "::" + answer["source_cell_id"].astype(str)
    if "cell_id" in answer and not answer["cell_id"].astype(str).equals(expected_cell_id):
        raise AssertionError("Upstream canonical cell_id changed")
    answer["cell_id"] = expected_cell_id
    answer["donor"] = answer["donor_id"].astype(str)
    answer["celltype"] = answer["display_label"].astype(str)
    answer["celltype_provenance"] = "provider label, broad project harmonization"
    # The broad mapping itself is unchanged; preserve its original version so
    # the frozen Tietscher and Shiao metadata blocks remain byte-reproducible.
    answer["celltype_relabel_version"] = "tnbc_provider_broad_v1"
    if answer["cell_id"].duplicated().any():
        raise AssertionError("Canonical cell_id is not globally unique")
    return answer


def build_files(stage: Path, source_input: Path, raw, metadata, source_manifest):
    canonical = canonical_metadata(metadata)
    (stage / "input").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_input / "raw.npy", stage / "input" / "raw.npy")
    shutil.copyfile(source_input / "markers.txt", stage / "input" / "markers.txt")
    shutil.copyfile(
        source_input / "input_manifest.json",
        stage / "input" / "upstream_input_manifest.json",
    )
    write_csv_gz(
        canonical[["dataset", "unit_key"]],
        stage / "input" / "fit_metadata.csv.gz",
    )
    write_csv_gz(canonical, stage / "input" / "locked_metadata.csv.gz")

    identity_columns = [
        "cell_order", "cell_id", "source_cell_id", "dataset", "donor",
        "unit_key", "source_index_zero_based", "roi_id",
    ]
    write_csv_gz(canonical[identity_columns], stage / "selected_cells.csv.gz")

    donor_counts = (
        canonical.groupby(["dataset", "donor"], sort=False)
        .size().rename("cells").reset_index()
    )
    donor_counts.to_csv(stage / "donor_cell_counts.csv", index=False)
    label_counts = (
        canonical.groupby(["dataset", "celltype"], sort=False)
        .size().rename("cells").reset_index()
    )
    label_counts.to_csv(stage / "celltype_counts.csv", index=False)
    mapping = (
        canonical.groupby(
            ["dataset", "provider_celltype", "provider_cell_subtype", "celltype", "relabel_rule"],
            dropna=False, sort=False,
        )
        .size().rename("cells").reset_index()
    )
    mapping.to_csv(stage / "label_mapping_audit.csv", index=False)

    dataset_rows = []
    selection = {
        "Tietscher IMC": (
            "strict-TNBC donor BB165; 20,000 frozen cells across 7 ROIs; "
            "historical frozen rows preserved"
        ),
        "Meyer IMC": (
            "all 22,701 cells from five coverage-informed patients; seed 20260960 "
            "won a locked 100-seed screen for balanced B-lineage/endothelial coverage"
        ),
        "Shiao CODEX": (
            "provider treatment code 1, full-panel; seed 20260827 selected five donors; "
            "seed 20260828 sampled 4,000 cells per donor without replacement"
        ),
    }
    for dataset in DATASETS:
        local = canonical[canonical["dataset"].eq(dataset)]
        dataset_rows.append({
            "dataset": dataset,
            "directory": SLUGS[dataset],
            "technology": str(local["technology"].iloc[0]),
            "cells": int(len(local)),
            "donors": int(local["donor"].nunique()),
            "units": int(local["unit_key"].nunique()),
            "rois": int(local["roi_id"].nunique(dropna=True)),
            "source_scale": "highest-available provider/source measurement",
            "selection": selection[dataset],
        })

        rows = local["cell_order"].to_numpy(np.int64)
        source = local[["cell_id", "source_cell_id", "dataset", "donor", "unit_key"]].copy()
        for index, marker in enumerate(MARKERS):
            source[marker] = raw[rows, index]
        destination = stage / SLUGS[dataset]
        write_csv_gz(
            source,
            destination / "alignment_input_11.csv.gz",
            float_format="%.17g",
        )
        write_csv_gz(local, destination / "locked_metadata.csv.gz")
        check = pd.read_csv(
            destination / "alignment_input_11.csv.gz",
            low_memory=False,
            float_precision="round_trip",
        )
        if not np.array_equal(check["cell_id"].astype(str), source["cell_id"].astype(str)):
            raise AssertionError(f"{dataset}: source CSV cell order changed")
        if not np.array_equal(check[MARKERS].to_numpy(np.float64), raw[rows]):
            raise AssertionError(f"{dataset}: source CSV values changed")

    pd.DataFrame(dataset_rows).to_csv(stage / "dataset_manifest.csv", index=False)
    marker_rows = []
    for marker in MARKERS:
        marker_rows.append({
            "marker": marker,
            "used_for_alignment": True,
            "Tietscher_IMC_source": SOURCE_MARKERS["Tietscher IMC"][marker],
            "Meyer_IMC_source": SOURCE_MARKERS["Meyer IMC"][marker],
            "Shiao_CODEX_source": SOURCE_MARKERS["Shiao CODEX"][marker],
        })
    pd.DataFrame(marker_rows).to_csv(stage / "marker_manifest.csv", index=False)

    generated = sorted(
        path for path in stage.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    manifest = {
        "package": "tnbc_three_platform_benchmark_v2",
        "cells": EXPECTED_TOTAL,
        "datasets": DATASETS,
        "dataset_cells": EXPECTED_CELLS,
        "dataset_units": EXPECTED_UNITS,
        "markers": MARKERS,
        "canonical_matrix": "input/raw.npy",
        "fit_metadata": "input/fit_metadata.csv.gz",
        "locked_metadata": "input/locked_metadata.csv.gz",
        "input_scale": "native/highest-available provider source scale",
        "normalization_or_alignment_applied_by_builder": False,
        "fit_metadata_columns": ["dataset", "unit_key"],
        "provider_labels_available_to_fit": False,
        "celltype_policy": "provider labels with broad cross-dataset harmonization",
        "exploratory_sma_stroma_relabel_included": False,
        "meyer_sampling": source_manifest["sampling"],
        "meyer_coverage_guided_seed_screen": source_manifest["coverage_guided_seed_screen"],
        "shiao_sampling": SHIAO_SAMPLING,
        "selection_caveats": {
            "Tietscher IMC": "one strict-TNBC donor; donor and platform are confounded",
            "Meyer IMC": "five-patient coverage-informed set; provider labels were used only to choose the set, never for alignment",
            "Shiao CODEX": "five of ten eligible treatment-code-1 full-panel donors; 4,000 cells each",
        },
        "accepted_source_input": {
            "path": str(source_input.resolve()),
            "sha256": EXPECTED_SOURCE_SHA256,
        },
        "source_provenance": {
            "canonical_sources_sha256": source_manifest.get("canonical_sources_sha256", {}),
            "meyer_source": source_manifest.get("meyer_source"),
            "meyer_source_sha256": source_manifest.get("meyer_source_sha256"),
            "meyer_source_assay": source_manifest.get("meyer_source_assay"),
        },
        "software_versions": {
            "Python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "recipe_sha256": {
            "build_benchmark.py": sha256(HERE / "build_benchmark.py"),
            "README.md": sha256(HERE / "README.md") if (HERE / "README.md").is_file() else None,
        },
        "files_sha256": {
            str(path.relative_to(stage)): sha256(path) for path in generated
        },
    }
    write_text(stage / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def publish(stage: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    generated_directories = ["input", *SLUGS.values()]
    generated_files = [
        "selected_cells.csv.gz", "donor_cell_counts.csv", "celltype_counts.csv",
        "label_mapping_audit.csv", "dataset_manifest.csv", "marker_manifest.csv",
        "manifest.json",
    ]
    for name in generated_directories:
        destination = output / name
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(stage / name, destination)
    for name in generated_files:
        os.replace(stage / name, output / name)


def main() -> None:
    args = parse_args()
    source_input = args.source_input.resolve()
    output = args.output_dir.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    raw, metadata, source_manifest = load_and_validate(source_input)
    stage = Path(tempfile.mkdtemp(prefix=".tnbc-benchmark-build-", dir=output.parent))
    try:
        build_files(stage, source_input, raw, metadata, source_manifest)
        publish(stage, output)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    print(json.dumps({
        "output": str(output),
        "cells": int(len(metadata)),
        "dataset_cells": EXPECTED_CELLS,
        "meyer_donors": sorted(EXPECTED_DONOR_COUNTS["Meyer IMC"]),
        "shiao_donors": sorted(EXPECTED_DONOR_COUNTS["Shiao CODEX"]),
        "markers": MARKERS,
        "sma_derived_labels_included": False,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
