#!/usr/bin/env python3
"""Run canonical RAMP 2.3.0 on the frozen lung-cancer benchmark.

The study-specific 7,000-cell pair-fit depth is applied temporarily to the
canonical implementation.  RAMP sees only the source-scale marker matrix and
the two-column fitting metadata.  Post-hoc relabel metadata is opened only
after fitting, algorithm audits, and accepted-output regression checks finish.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import platform
import shutil
import sys
from importlib.metadata import version
from pathlib import Path


sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
FINAL_XU = HERE.parents[3]
BENCHMARK_ROOT = FINAL_XU / "benchmark_data" / "lung_cancer"
RAW_SOURCE = BENCHMARK_ROOT / "input" / "raw.npy"
FIT_METADATA_SOURCE = BENCHMARK_ROOT / "input" / "fit_metadata.csv.gz"
MARKERS_SOURCE = BENCHMARK_ROOT / "input" / "markers.txt"
RELABEL_METADATA_SOURCE = BENCHMARK_ROOT / "relabel" / "relabeled_metadata.csv.gz"

CANONICAL_ALGORITHM = FINAL_XU / "algorithm" / "ramp" / "ramp_ver2.py"
ALGORITHM_ROOT = FINAL_XU / "algorithm"
ACCEPTED_ALIGNED = (
    FINAL_XU
    / "test0827"
    / "lung"
    / "alignment_10_donors"
    / "output"
    / "ver2"
    / "aligned.npy"
)
ACCEPTED_BASELINE = (
    FINAL_XU
    / "test0827"
    / "lung"
    / "alignment_10_donors"
    / "output"
    / "kmeanslow_global.npy"
)

INPUT_DIR = HERE / "input"
DATA_DIR = HERE / "data"

DATASETS = ["Senosain CyTOF", "Cords IMC", "Monkman CODEX"]
MARKERS = ["CD3", "CD4", "CD8", "CD31", "HLA_DR", "KI67", "VIMENTIN", "EPI"]
EXPECTED_DATASET_CELLS = {dataset: 20_000 for dataset in DATASETS}
EXPECTED_DATASET_UNITS = {dataset: 10 for dataset in DATASETS}
EXPECTED_TOTAL = 60_000
FAMILY = "lung_alignment_nested_10x3_20k_joint8_current_ramp"
PAIR_FIT_POLICY = {"minimum": 7_000, "per_unit": 0, "maximum": 7_000}

EXPECTED_INPUT_SHA256 = {
    "raw": "83cef15257b1a522cc11490a5b2b70e97379a469f2120b52b6b9cb1a4c231538",
    "fit_metadata": "994131e9b2a9d5e72f5f1fade93724af420d509ff67c6f2f010b20e407cbfd91",
    "markers": "067c8e1bfcd0d6292350ecaab8356f3223b0c9e5cf6a8d4f6826d141bd1f5ffb",
    "relabel_metadata": "c909b809cf8690a8f8cc99d8808367e5d1a5ff5abc9ac517abb5351ab82e35c3",
}
EXPECTED_CANONICAL_SHA256 = (
    "2501055cf269e383179b602bd99ff39171c8d283bc6332cd4143eed30c4a68c1"
)
EXPECTED_ACCEPTED_SHA256 = {
    "aligned": "45b947761a6582a0e790b4eb0ce497aaa5bb21ecbdc24d9110614798f94ebed4",
    "baseline": "1b5f6b9a571ae07b17259c863535dc6f5625d0959d5d5c47525475ae4d79ded6",
}
EXPECTED_OUTPUT_SHA256 = {
    "ramp_aligned.npy": "c986e48c562fd2cff877cfd044f53ce16545268d28c22f45c68131e1a2c4e7cf",
    "kmeanslow_global.npy": "9fc5a8f81094705794cf6e6bde3e5fc2685cc52e8401f47bdf3c149519f79912",
    "ramp_aligned_float64.npy": EXPECTED_ACCEPTED_SHA256["aligned"],
    "kmeanslow_global_float64.npy": EXPECTED_ACCEPTED_SHA256["baseline"],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_output_path(path: Path) -> None:
    try:
        path.resolve().relative_to(HERE)
    except ValueError as error:
        raise ValueError(f"Refusing to write outside {HERE}: {path}") from error


def write_csv_gz(
    frame: pd.DataFrame,
    path: Path,
    *,
    float_format: str | None = None,
) -> None:
    assert_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="",
            fileobj=raw_handle,
            mode="wb",
            compresslevel=6,
            mtime=0,
        ) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                frame.to_csv(text, index=False, float_format=float_format)


def write_json(payload: dict[str, object], path: Path) -> None:
    assert_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def audit_dictionary(frame: pd.DataFrame) -> dict[str, str]:
    return dict(zip(frame["audit_key"].astype(str), frame["audit_value"].astype(str)))


def boolean_values(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype=bool)
    return series.astype(str).str.lower().isin({"true", "1", "1.0"}).to_numpy()


def validate_source_contract() -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    for path in (RAW_SOURCE, FIT_METADATA_SOURCE, MARKERS_SOURCE, CANONICAL_ALGORITHM):
        if not path.is_file():
            raise FileNotFoundError(path)
    observed_hashes = {
        "raw": sha256(RAW_SOURCE),
        "fit_metadata": sha256(FIT_METADATA_SOURCE),
        "markers": sha256(MARKERS_SOURCE),
    }
    if observed_hashes != {
        key: EXPECTED_INPUT_SHA256[key] for key in observed_hashes
    }:
        raise AssertionError(f"Frozen lung input hash changed: {observed_hashes}")
    if sha256(CANONICAL_ALGORITHM) != EXPECTED_CANONICAL_SHA256:
        raise AssertionError("Canonical RAMP 2.3.0 source hash changed")

    raw = np.load(RAW_SOURCE, allow_pickle=False)
    fit_metadata = pd.read_csv(FIT_METADATA_SOURCE, low_memory=False)
    markers = [
        value.strip()
        for value in MARKERS_SOURCE.read_text().splitlines()
        if value.strip()
    ]
    if raw.dtype != np.float64 or raw.shape != (EXPECTED_TOTAL, len(MARKERS)):
        raise AssertionError(f"Lung matrix contract changed: {raw.shape}, {raw.dtype}")
    if not np.isfinite(raw).all():
        raise AssertionError("Lung source-scale matrix contains non-finite values")
    if markers != MARKERS:
        raise AssertionError(f"Lung marker order changed: {markers}")
    if fit_metadata.columns.tolist() != ["dataset", "unit_key"]:
        raise AssertionError(
            f"Fit metadata leaked extra columns: {fit_metadata.columns.tolist()}"
        )
    if len(fit_metadata) != EXPECTED_TOTAL:
        raise AssertionError("Lung fit-metadata row count changed")
    if list(pd.unique(fit_metadata["dataset"].astype(str))) != DATASETS:
        raise AssertionError("Lung dataset order changed")
    observed_cells = fit_metadata.groupby("dataset", sort=False).size().to_dict()
    if observed_cells != EXPECTED_DATASET_CELLS:
        raise AssertionError(f"Lung dataset counts changed: {observed_cells}")
    observed_units = (
        fit_metadata.groupby("dataset", sort=False)["unit_key"].nunique().to_dict()
    )
    if observed_units != EXPECTED_DATASET_UNITS:
        raise AssertionError(f"Lung unit counts changed: {observed_units}")
    if fit_metadata.groupby("unit_key", sort=False)["dataset"].nunique().max() != 1:
        raise AssertionError("A lung unit_key occurs in more than one dataset")
    return raw, fit_metadata, markers


def run_canonical_ramp(raw: np.ndarray, fit_metadata: pd.DataFrame, markers: list[str]):
    if str(ALGORITHM_ROOT) not in sys.path:
        sys.path.insert(0, str(ALGORITHM_ROOT))
    from ramp import ramp_ver2 as canonical_impl

    source_path = Path(canonical_impl.__file__).resolve()
    if source_path != CANONICAL_ALGORITHM.resolve():
        raise AssertionError(f"Unexpected RAMP implementation imported: {source_path}")
    if canonical_impl.__version__ != "2.3.0" or int(canonical_impl.SEED) != 42:
        raise AssertionError(
            f"Unexpected canonical RAMP identity: {canonical_impl.__version__}, "
            f"seed={canonical_impl.SEED}"
        )

    previous = {
        "minimum": canonical_impl.PAIR_FIT_MINIMUM,
        "per_unit": canonical_impl.PAIR_FIT_PER_UNIT,
        "maximum": canonical_impl.PAIR_FIT_MAXIMUM,
    }
    try:
        canonical_impl.PAIR_FIT_MINIMUM = PAIR_FIT_POLICY["minimum"]
        canonical_impl.PAIR_FIT_PER_UNIT = PAIR_FIT_POLICY["per_unit"]
        canonical_impl.PAIR_FIT_MAXIMUM = PAIR_FIT_POLICY["maximum"]
        result = canonical_impl.fit_transform(
            raw,
            fit_metadata[["dataset", "unit_key"]].copy(),
            markers,
            datasets=DATASETS,
            family=FAMILY,
        )
    finally:
        canonical_impl.PAIR_FIT_MINIMUM = previous["minimum"]
        canonical_impl.PAIR_FIT_PER_UNIT = previous["per_unit"]
        canonical_impl.PAIR_FIT_MAXIMUM = previous["maximum"]
    return result, canonical_impl


def validate_algorithm_result(result) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    aligned = np.asarray(result.aligned, dtype=np.float64)
    baseline = np.asarray(result.baseline, dtype=np.float64)
    expected_shape = (EXPECTED_TOTAL, len(MARKERS))
    if aligned.shape != expected_shape or baseline.shape != expected_shape:
        raise AssertionError("RAMP output shape changed")
    if not np.isfinite(aligned).all() or not np.isfinite(baseline).all():
        raise AssertionError("RAMP produced non-finite output")
    if int(result.strict_rank_inversions) != 0:
        raise AssertionError("RAMP reported a strict within-unit rank inversion")

    coupling = result.coupling_audit
    if set(coupling["family"].astype(str)) != {FAMILY}:
        raise AssertionError("RAMP coupling audit family changed")
    for column, expected in (
        ("pair_fit_minimum", PAIR_FIT_POLICY["minimum"]),
        ("pair_fit_per_unit", PAIR_FIT_POLICY["per_unit"]),
        ("pair_fit_maximum", PAIR_FIT_POLICY["maximum"]),
        ("pair_fit_effective", PAIR_FIT_POLICY["maximum"]),
        ("graph_k_nonself", 15),
    ):
        if set(coupling[column].astype(int)) != {expected}:
            raise AssertionError(f"Unexpected {column}: {set(coupling[column])}")

    profile_minimums = set(result.profile_audit["curve_minimum_bin_cells"].astype(int))
    if profile_minimums != {15}:
        raise AssertionError(f"Curve regularization changed: {profile_minimums}")

    algorithm_audit = audit_dictionary(result.algorithm_audit)
    expected_audit = {
        "algorithm_version": "2.3.0",
        "mapping_operator_calibration": "bounded_profile_mapping",
        "mapping_operator_final": "bounded_profile_mapping",
        "cell_marker_values_over_maximum": "0",
        "curve_minimum_cells_per_bin": "15",
        "pair_fit_maximum": "7000",
        "graph_k_nonself": "15",
        "transition_includes_self": "False",
    }
    for key, expected in expected_audit.items():
        if algorithm_audit.get(key) != expected:
            raise AssertionError(
                f"Algorithm audit changed for {key}: {algorithm_audit.get(key)!r}"
            )
    displacement = np.abs(aligned - baseline)
    if np.any(displacement > 0.75 + 1e-10):
        raise AssertionError("RAMP exceeded MAX_DISPLACEMENT=0.75")
    return aligned, baseline, algorithm_audit


def compare_accepted(aligned: np.ndarray, baseline: np.ndarray) -> dict[str, object]:
    comparisons: dict[str, object] = {}
    for name, path, formal in (
        ("aligned", ACCEPTED_ALIGNED, aligned),
        ("baseline", ACCEPTED_BASELINE, baseline),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        observed_hash = sha256(path)
        if observed_hash != EXPECTED_ACCEPTED_SHA256[name]:
            raise AssertionError(f"Accepted lung {name} hash changed")
        accepted = np.load(path, allow_pickle=False)
        if accepted.dtype != np.float64 or accepted.shape != formal.shape:
            raise AssertionError(f"Accepted lung {name} contract changed")
        maximum_error = float(np.max(np.abs(accepted - formal)))
        if maximum_error != 0.0 or not np.array_equal(accepted, formal):
            raise AssertionError(
                f"Canonical lung {name} differs from accepted test: {maximum_error}"
            )
        comparisons[name] = {
            "path": str(path),
            "sha256": observed_hash,
            "maximum_absolute_error_float64": maximum_error,
            "elementwise_identical": True,
        }
    return comparisons


def save_algorithm_outputs(
    result,
    aligned: np.ndarray,
    baseline: np.ndarray,
    canonical_impl,
    accepted_comparison: dict[str, object],
) -> dict[str, Path]:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for destination in (
        INPUT_DIR / "raw.npy",
        INPUT_DIR / "fit_metadata.csv.gz",
        INPUT_DIR / "markers.txt",
    ):
        assert_output_path(destination)
    shutil.copyfile(RAW_SOURCE, INPUT_DIR / "raw.npy")
    shutil.copyfile(FIT_METADATA_SOURCE, INPUT_DIR / "fit_metadata.csv.gz")
    shutil.copyfile(MARKERS_SOURCE, INPUT_DIR / "markers.txt")

    paths = {
        "aligned": DATA_DIR / "ramp_aligned.npy",
        "baseline": DATA_DIR / "kmeanslow_global.npy",
        "aligned_float64": DATA_DIR / "ramp_aligned_float64.npy",
        "baseline_float64": DATA_DIR / "kmeanslow_global_float64.npy",
    }
    for path in paths.values():
        assert_output_path(path)
    np.save(paths["aligned"], aligned.astype(np.float32))
    np.save(paths["baseline"], baseline.astype(np.float32))
    np.save(paths["aligned_float64"], aligned)
    np.save(paths["baseline_float64"], baseline)

    audit_frames = {
        "coupling_audit.csv": result.coupling_audit,
        "strength_audit.csv": result.strength_audit,
        "evidence_audit.csv": result.evidence_audit,
        "reference_weight_audit.csv": result.reference_weight_audit,
        "profile_audit.csv": result.profile_audit,
        "algorithm_audit.csv": result.algorithm_audit,
    }
    for filename, frame in audit_frames.items():
        path = DATA_DIR / filename
        assert_output_path(path)
        frame.to_csv(path, index=False)

    observed_output_hashes = {
        "ramp_aligned.npy": sha256(paths["aligned"]),
        "kmeanslow_global.npy": sha256(paths["baseline"]),
        "ramp_aligned_float64.npy": sha256(paths["aligned_float64"]),
        "kmeanslow_global_float64.npy": sha256(paths["baseline_float64"]),
    }
    if observed_output_hashes != EXPECTED_OUTPUT_SHA256:
        raise AssertionError(f"Canonical lung output hashes changed: {observed_output_hashes}")

    helper_source = Path(canonical_impl._legacy.__file__).resolve()
    ramp_manifest_path = DATA_DIR / "ramp_manifest.json"
    ramp_manifest = {
        "algorithm": "RAMP",
        "algorithm_version": canonical_impl.__version__,
        "algorithm_seed": int(canonical_impl.SEED),
        "algorithm_source": {
            "file": str(CANONICAL_ALGORITHM),
            "sha256": sha256(CANONICAL_ALGORITHM),
        },
        "algorithm_helper_source": {
            "file": str(helper_source),
            "sha256": sha256(helper_source),
        },
        "family": FAMILY,
        "datasets": DATASETS,
        "dataset_count": len(DATASETS),
        "dataset_cells": EXPECTED_DATASET_CELLS,
        "dataset_units": EXPECTED_DATASET_UNITS,
        "cells": EXPECTED_TOTAL,
        "markers": MARKERS,
        "marker_count": len(MARKERS),
        "pair_fit_policy": PAIR_FIT_POLICY,
        "pair_fit_policy_reason": "frozen historical lung-story runtime fit depth",
        "strict_rank_inversions": int(result.strict_rank_inversions),
        "metadata_columns_read_for_fitting": ["dataset", "unit_key"],
        "celltype_labels_read_for_fitting": False,
        "accepted_test_comparison": accepted_comparison,
        "inputs": {
            "raw": str((INPUT_DIR / "raw.npy").resolve()),
            "raw_sha256": sha256(INPUT_DIR / "raw.npy"),
            "metadata": str((INPUT_DIR / "fit_metadata.csv.gz").resolve()),
            "metadata_sha256": sha256(INPUT_DIR / "fit_metadata.csv.gz"),
            "markers": str((INPUT_DIR / "markers.txt").resolve()),
            "markers_sha256": sha256(INPUT_DIR / "markers.txt"),
        },
        "outputs": {
            **observed_output_hashes,
            **{
                filename: sha256(DATA_DIR / filename)
                for filename in audit_frames
            },
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "scipy": version("scipy"),
            "scikit_learn": version("scikit-learn"),
        },
    }
    write_json(ramp_manifest, ramp_manifest_path)
    paths["ramp_manifest"] = ramp_manifest_path
    return paths


def load_posthoc_metadata(fit_metadata: pd.DataFrame) -> pd.DataFrame:
    # This is intentionally the first read of any label-bearing source.
    if not RELABEL_METADATA_SOURCE.is_file():
        raise FileNotFoundError(RELABEL_METADATA_SOURCE)
    if sha256(RELABEL_METADATA_SOURCE) != EXPECTED_INPUT_SHA256["relabel_metadata"]:
        raise AssertionError("Frozen lung relabel metadata hash changed")
    labels = pd.read_csv(RELABEL_METADATA_SOURCE, low_memory=False)
    required = [
        "row_id",
        "cell_id",
        "source_cell_id",
        "dataset",
        "donor_id",
        "donor",
        "unit_key",
        "technology",
        "celltype",
        "celltype_joint_evaluation",
        "joint_evaluation",
        "labels_used_for_alignment",
        "is_mix_display",
        "celltype_relabel_version",
        "celltype_provenance",
    ]
    missing = [column for column in required if column not in labels]
    if missing:
        raise ValueError(f"Missing lung relabel columns: {missing}")
    if len(labels) != EXPECTED_TOTAL:
        raise AssertionError("Lung relabel row count changed")
    if labels["cell_id"].astype(str).duplicated().any():
        raise AssertionError("Duplicate lung cell IDs")
    if not np.array_equal(
        labels["row_id"].to_numpy(np.int64), np.arange(EXPECTED_TOTAL, dtype=np.int64)
    ):
        raise AssertionError("Lung relabel row IDs changed")
    for column in ("dataset", "unit_key"):
        if not fit_metadata[column].astype(str).equals(labels[column].astype(str)):
            raise AssertionError(f"Lung relabel {column} row order changed")
    if boolean_values(labels["labels_used_for_alignment"]).any():
        raise AssertionError("A lung relabel row claims labels were used for alignment")
    if list(pd.unique(labels["dataset"].astype(str))) != DATASETS:
        raise AssertionError("Lung relabel dataset order changed")
    expected_labels = {
        "Tumor/Epithelial",
        "CD4 T",
        "CD8 T",
        "B-like",
        "Myeloid",
        "Endothelial",
        "Mix",
    }
    if set(labels["celltype"].astype(str)) != expected_labels:
        raise AssertionError("Lung relabel vocabulary changed")
    mix = labels["celltype"].astype(str).eq("Mix").to_numpy()
    if not np.array_equal(mix, boolean_values(labels["is_mix_display"])):
        raise AssertionError("Lung Mix display flag changed")
    return labels[required].copy()


def save_posthoc_tables(
    display: pd.DataFrame,
    aligned: np.ndarray,
    baseline: np.ndarray,
) -> dict[str, Path]:
    metadata_path = INPUT_DIR / "cell_metadata.csv.gz"
    write_csv_gz(display, metadata_path)

    aligned_frame = display.copy()
    baseline_frame = display.copy()
    aligned_compact = aligned.astype(np.float32)
    baseline_compact = baseline.astype(np.float32)
    for marker_index, marker in enumerate(MARKERS):
        aligned_frame[marker] = aligned_compact[:, marker_index]
        baseline_frame[marker] = baseline_compact[:, marker_index]
    aligned_csv = DATA_DIR / "ramp_aligned.csv.gz"
    baseline_csv = DATA_DIR / "kmeanslow_global.csv.gz"
    write_csv_gz(aligned_frame, aligned_csv, float_format="%.9g")
    write_csv_gz(baseline_frame, baseline_csv, float_format="%.9g")

    for name, path, expected in (
        ("aligned", aligned_csv, aligned_compact),
        ("baseline", baseline_csv, baseline_compact),
    ):
        check = pd.read_csv(path, low_memory=False, float_precision="round_trip")
        if not np.array_equal(check["row_id"].to_numpy(np.int64), display["row_id"]):
            raise AssertionError(f"{name} CSV row order changed")
        if not np.array_equal(check[MARKERS].to_numpy(np.float32), expected):
            raise AssertionError(f"{name} CSV values differ from compact NPY")
    return {
        "cell_metadata.csv.gz": metadata_path,
        "ramp_aligned.csv.gz": aligned_csv,
        "kmeanslow_global.csv.gz": baseline_csv,
    }


def main() -> None:
    raw, fit_metadata, markers = validate_source_contract()
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(
        "Running canonical RAMP 2.3.0 on 60,000 lung cells x 8 markers "
        "with frozen pair-fit=7,000/dataset",
        flush=True,
    )
    result, canonical_impl = run_canonical_ramp(raw, fit_metadata, markers)
    aligned, baseline, algorithm_audit = validate_algorithm_result(result)
    accepted_comparison = compare_accepted(aligned, baseline)
    algorithm_paths = save_algorithm_outputs(
        result,
        aligned,
        baseline,
        canonical_impl,
        accepted_comparison,
    )

    # Labels are unlocked only after fitting and every algorithm-only check above.
    display = load_posthoc_metadata(fit_metadata)
    posthoc_paths = save_posthoc_tables(display, aligned, baseline)

    output_paths = {
        "ramp_aligned.npy": DATA_DIR / "ramp_aligned.npy",
        "kmeanslow_global.npy": DATA_DIR / "kmeanslow_global.npy",
        "ramp_aligned_float64.npy": DATA_DIR / "ramp_aligned_float64.npy",
        "kmeanslow_global_float64.npy": DATA_DIR / "kmeanslow_global_float64.npy",
        "ramp_manifest.json": algorithm_paths["ramp_manifest"],
        "coupling_audit.csv": DATA_DIR / "coupling_audit.csv",
        "strength_audit.csv": DATA_DIR / "strength_audit.csv",
        "evidence_audit.csv": DATA_DIR / "evidence_audit.csv",
        "reference_weight_audit.csv": DATA_DIR / "reference_weight_audit.csv",
        "profile_audit.csv": DATA_DIR / "profile_audit.csv",
        "algorithm_audit.csv": DATA_DIR / "algorithm_audit.csv",
        "ramp_aligned.csv.gz": posthoc_paths["ramp_aligned.csv.gz"],
        "kmeanslow_global.csv.gz": posthoc_paths["kmeanslow_global.csv.gz"],
    }
    manifest = {
        "analysis": "lung_cancer_ramp_alignment",
        "cells": EXPECTED_TOTAL,
        "markers": MARKERS,
        "datasets": DATASETS,
        "dataset_cells": EXPECTED_DATASET_CELLS,
        "dataset_units": EXPECTED_DATASET_UNITS,
        "family": FAMILY,
        "algorithm_seed": int(canonical_impl.SEED),
        "pair_fit_policy": PAIR_FIT_POLICY,
        "pair_fit_policy_reason": "reproduce the accepted historical lung-story fit depth",
        "source_scale_input": True,
        "external_preprocessing_before_ramp": False,
        "ramp_internal_preprocessing": "KMEANSLOW",
        "labels_used_for_fitting": False,
        "labels_loaded_after_alignment_and_algorithm_audit": True,
        "fit_metadata_columns": ["dataset", "unit_key"],
        "canonical_algorithm_version": canonical_impl.__version__,
        "canonical_algorithm_source": {
            "file": str(CANONICAL_ALGORITHM),
            "sha256": sha256(CANONICAL_ALGORITHM),
        },
        "strict_rank_inversions": int(result.strict_rank_inversions),
        "algorithm_audit": algorithm_audit,
        "accepted_test_comparison": accepted_comparison,
        "sources": {
            "raw": {"path": str(RAW_SOURCE), "sha256": sha256(RAW_SOURCE)},
            "fit_metadata": {
                "path": str(FIT_METADATA_SOURCE),
                "sha256": sha256(FIT_METADATA_SOURCE),
            },
            "markers": {
                "path": str(MARKERS_SOURCE),
                "sha256": sha256(MARKERS_SOURCE),
            },
            "relabel_metadata_posthoc": {
                "path": str(RELABEL_METADATA_SOURCE),
                "sha256": sha256(RELABEL_METADATA_SOURCE),
            },
        },
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__)),
        },
        "inputs": {
            "raw.npy": sha256(INPUT_DIR / "raw.npy"),
            "fit_metadata.csv.gz": sha256(INPUT_DIR / "fit_metadata.csv.gz"),
            "markers.txt": sha256(INPUT_DIR / "markers.txt"),
            "cell_metadata.csv.gz": sha256(INPUT_DIR / "cell_metadata.csv.gz"),
        },
        "outputs": {name: sha256(path) for name, path in output_paths.items()},
        "precision": {
            "compact_npy_and_csv": "float32",
            "downstream_umap_npy": "canonical float64",
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "scipy": version("scipy"),
            "scikit_learn": version("scikit-learn"),
        },
    }
    write_json(manifest, HERE / "alignment_manifest.json")
    print(f"Saved canonical lung alignment under {DATA_DIR}", flush=True)


if __name__ == "__main__":
    main()
