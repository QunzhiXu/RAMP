#!/usr/bin/env python3
"""Run the two formal RAMP 2.3.0 alignments for the BM CD19 story.

Conditions
----------
1. ``without_cd19``: the 14 healthy shared markers.
2. ``with_wrong_observed_cd19``: the 14 markers plus the known failed observed
   CODEX CD19 channel; this is a negative control only.
3. ``with_predicted_cd19``: the same 14 markers plus observed CD19 in the two
   reference platforms and label-free predicted CD19 in CODEX.

The predicted CODEX CD19 values are already on the KMEANSLOW scale. They are
therefore inserted into the completed KMEANSLOW matrix and passed directly to
the RAMP post-KMEANSLOW backend; they are never preprocessed a second time.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
STORY = HERE.parent
SAMPLED = STORY / "sampled_data"
PREDICTION = STORY / "prediction" / "output"
OUTPUT = HERE / "output"
CACHE = HERE / ".cache"
ALGORITHM_ROOT = STORY.parent / "algorithm"

for folder in [OUTPUT, CACHE / "numba", CACHE / "joblib", CACHE / "tmp"]:
    folder.mkdir(parents=True, exist_ok=True)
for name, value in {
    "NUMBA_CACHE_DIR": str(CACHE / "numba"),
    "JOBLIB_TEMP_FOLDER": str(CACHE / "joblib"),
    "TMPDIR": str(CACHE / "tmp"),
    "LOKY_MAX_CPU_COUNT": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}.items():
    os.environ.setdefault(name, value)
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd


sys.path.insert(0, str(ALGORITHM_ROOT))
from ramp import ramp_ver2 as ramp230  # noqa: E402


DATA_SPECS = [
    ("BM CODEX", "bm_codex"),
    ("BM CyTOF", "bm_cytof"),
    ("Triana AbSeq", "triana_abseq"),
]
DATASETS = [item[0] for item in DATA_SPECS]
EXPECTED_COUNTS = {"BM CODEX": 30_000, "BM CyTOF": 30_000, "Triana AbSeq": 28_031}
EXPECTED_CELLS = 88_031
MARKERS14 = [
    "CD117", "CD11B", "CD11C", "CD123", "CD14", "CD3", "CD33",
    "CD34", "CD38", "CD4", "CD45", "CD45RA", "CD61", "CD8",
]
MARKERS15 = MARKERS14 + ["CD19"]
IDENTITY_COLUMNS = ["cell_id", "dataset", "donor", "unit_key"]


@dataclass
class AlignmentResult:
    name: str
    family: str
    entrypoint: str
    markers: list[str]
    baseline: np.ndarray
    aligned: np.ndarray
    coupling_audit: pd.DataFrame
    strength_audit: pd.DataFrame
    evidence_audit: pd.DataFrame
    reference_weight_audit: pd.DataFrame
    profile_audit: pd.DataFrame
    algorithm_audit: pd.DataFrame
    strict_rank_inversions: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv_gz(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(
        path,
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )


def load_inputs() -> tuple[np.ndarray, pd.DataFrame, dict[str, str]]:
    matrices: list[np.ndarray] = []
    metadata_parts: list[pd.DataFrame] = []
    hashes: dict[str, str] = {}
    for dataset, slug in DATA_SPECS:
        path = SAMPLED / slug / "source_input_15.csv.gz"
        frame = pd.read_csv(path, low_memory=False)
        if not frame["dataset"].astype(str).eq(dataset).all():
            raise AssertionError(f"Dataset mismatch in {path}")
        if frame["cell_id"].astype(str).duplicated().any():
            raise AssertionError(f"Duplicate cell IDs in {path}")
        values = frame[MARKERS15].to_numpy(np.float64)
        if not np.isfinite(values).all():
            raise AssertionError(f"Non-finite values in {path}")
        matrices.append(values)
        metadata_parts.append(frame[IDENTITY_COLUMNS].copy())
        hashes[str(path.resolve())] = sha256_file(path)

    raw15 = np.vstack(matrices)
    metadata = pd.concat(metadata_parts, ignore_index=True)
    counts = metadata.groupby("dataset", sort=False).size().astype(int).to_dict()
    if counts != EXPECTED_COUNTS or len(metadata) != EXPECTED_CELLS:
        raise AssertionError(f"Unexpected frozen cohort: {counts}; n={len(metadata)}")
    if list(pd.unique(metadata["dataset"].astype(str))) != DATASETS:
        raise AssertionError("Dataset or row order drift")

    # Relabeled cell types are joined only for saved metadata and later UMAP
    # display. They are never passed into either RAMP fit.
    relabel_path = SAMPLED / "relabel" / "relabeled_metadata.csv.gz"
    relabel = pd.read_csv(relabel_path, low_memory=False)
    key_columns = ["dataset", "cell_id"]
    if relabel.duplicated(key_columns).any():
        raise AssertionError("Duplicate keys in relabeled metadata")
    lookup = relabel.set_index(key_columns)
    ordered = lookup.loc[pd.MultiIndex.from_frame(metadata[key_columns])].reset_index()
    if not ordered[key_columns].astype(str).equals(metadata[key_columns].astype(str)):
        raise AssertionError("Relabeled metadata identity/order mismatch")
    metadata["celltype"] = ordered["celltype"].astype(str).to_numpy()
    metadata["provider_label_l1"] = ordered["provider_label_l1"].astype(str).to_numpy()
    metadata["provider_label_l2"] = ordered["provider_label_l2"].astype(str).to_numpy()
    hashes[str(relabel_path.resolve())] = sha256_file(relabel_path)
    return raw15, metadata, hashes


def from_public(name: str, family: str, markers: list[str], result: Any) -> AlignmentResult:
    return AlignmentResult(
        name=name,
        family=family,
        entrypoint="ramp.ramp_ver2.fit_transform",
        markers=markers,
        baseline=result.baseline,
        aligned=result.aligned,
        coupling_audit=result.coupling_audit,
        strength_audit=result.strength_audit,
        evidence_audit=result.evidence_audit,
        reference_weight_audit=result.reference_weight_audit,
        profile_audit=result.profile_audit,
        algorithm_audit=result.algorithm_audit,
        strict_rank_inversions=int(result.strict_rank_inversions),
    )


def load_and_validate_prediction(
    metadata: pd.DataFrame,
    observed_baseline15: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    prediction_path = PREDICTION / "codex_cd19_prediction.npz"
    prediction_metadata_path = PREDICTION / "codex_prediction_metadata.csv.gz"
    prediction_manifest_path = PREDICTION / "manifest.json"
    payload = np.load(prediction_path, allow_pickle=False)
    prediction_metadata = pd.read_csv(prediction_metadata_path, low_memory=False)
    codex_rows = np.flatnonzero(metadata["dataset"].astype(str).eq("BM CODEX"))
    predicted = payload["predicted_kmeanslow"].astype(np.float64)

    checks = {
        "codex_rows_exact": bool(np.array_equal(payload["codex_rows"], codex_rows)),
        "metadata_identity_exact": bool(
            prediction_metadata[IDENTITY_COLUMNS].astype(str).reset_index(drop=True).equals(
                metadata.iloc[codex_rows][IDENTITY_COLUMNS].astype(str).reset_index(drop=True)
            )
        ),
        "prediction_count": int(len(predicted)),
        "prediction_finite": bool(np.isfinite(predicted).all()),
        "observed_codex_cd19_kmeanslow_max_abs_difference": float(
            np.max(
                np.abs(
                    payload["observed_kmeanslow"].astype(np.float64)
                    - observed_baseline15[codex_rows, 14]
                ),
                initial=0.0,
            )
        ),
        "prediction_scale": "KMEANSLOW; inserted without a second preprocessing pass",
        "prediction_source": str(prediction_path.resolve()),
        "prediction_sha256": sha256_file(prediction_path),
        "prediction_metadata_sha256": sha256_file(prediction_metadata_path),
        "prediction_manifest_sha256": sha256_file(prediction_manifest_path),
    }
    checks["passed"] = bool(
        checks["codex_rows_exact"]
        and checks["metadata_identity_exact"]
        and checks["prediction_count"] == len(codex_rows)
        and checks["prediction_finite"]
        and checks["observed_codex_cd19_kmeanslow_max_abs_difference"] <= 1e-6
    )
    if not checks["passed"]:
        raise AssertionError(f"Predicted CD19 compatibility failed: {checks}")
    return predicted, codex_rows, checks


def fit_with_predicted_cd19(
    raw15: np.ndarray,
    metadata: pd.DataFrame,
    baseline14: np.ndarray,
) -> tuple[AlignmentResult, dict[str, Any]]:
    units = metadata["unit_key"].astype(str).to_numpy()
    dataset_values = metadata["dataset"].astype(str).to_numpy()
    observed_baseline15_full_precision = ramp230.kmeanslow_global(
        raw15, units, dataset_values, MARKERS15
    )
    # The audited prediction-to-alignment interface serializes the completed
    # KMEANSLOW matrix as float32. Recreate that explicit boundary locally so
    # this self-contained formal run is numerically identical to the selected
    # RAMP 2.3.0 test, rather than depending on a test artifact.
    observed_baseline15 = observed_baseline15_full_precision.astype(np.float32).astype(np.float64)
    serialized_baseline14 = baseline14.astype(np.float32).astype(np.float64)
    first14_delta = float(
        np.max(np.abs(observed_baseline15[:, :14] - serialized_baseline14), initial=0.0)
    )
    if first14_delta > 0.0:
        raise AssertionError(f"The shared-marker KMEANSLOW baseline changed: {first14_delta}")

    predicted, codex_rows, compatibility = load_and_validate_prediction(
        metadata, observed_baseline15
    )
    completed_baseline = observed_baseline15.copy()
    completed_baseline[codex_rows, 14] = predicted

    # `raw` is retained as the rank/matching view. The fit itself receives the
    # completed KMEANSLOW matrix below, so predicted CD19 is not transformed.
    completed_matching_view = raw15.copy()
    completed_matching_view[codex_rows, 14] = predicted
    validated_raw, fit_metadata, markers, datasets = ramp230._validated_metadata(
        completed_matching_view,
        metadata[["dataset", "unit_key"]],
        MARKERS15,
        DATASETS,
    )
    family = "bm_prediction_story_completed15"
    benchmark = ramp230.Benchmark(
        family=family,
        raw=validated_raw,
        metadata=fit_metadata,
        markers=markers,
        datasets=datasets,
        reference=datasets[0],
    )
    (
        aligned,
        coupling,
        strength,
        evidence,
        weights,
        profile_audit,
        compatibility_audit,
    ) = ramp230.fit_v40_v21(benchmark, completed_baseline)

    displacement = np.abs(aligned - completed_baseline)
    if np.any(displacement > ramp230.MAX_DISPLACEMENT + 1e-10):
        raise AssertionError("Predicted-CD19 fit exceeded the RAMP 2.3.0 displacement cap")
    inversions = ramp230.rank_inversion_count(completed_baseline, aligned, fit_metadata)
    matching_inversions = ramp230.rank_inversion_count(validated_raw, aligned, fit_metadata)
    if inversions or matching_inversions:
        raise AssertionError(
            f"Predicted-CD19 fit produced rank inversions: {inversions}, {matching_inversions}"
        )
    algorithm_audit = ramp230._algorithm_audit(
        completed_baseline, aligned, compatibility_audit
    )
    extra = pd.DataFrame(
        [
            ("entrypoint", "ramp.ramp_ver2.fit_v40_v21"),
            (
                "low_level_entrypoint_reason",
                "predicted CODEX CD19 is already KMEANSLOW-scale; avoid a second preprocessing pass",
            ),
            ("matching_view_rank_inversions", int(matching_inversions)),
        ],
        columns=["audit_key", "audit_value"],
    )
    algorithm_audit = pd.concat([algorithm_audit, extra], ignore_index=True)
    compatibility["first14_baseline_max_abs_difference"] = first14_delta
    compatibility["completed_baseline_numeric_contract"] = (
        "KMEANSLOW matrix serialized to float32 at the prediction-to-alignment boundary, "
        "then widened to float64 for RAMP fitting"
    )
    return (
        AlignmentResult(
            name="with_predicted_cd19",
            family=family,
            entrypoint="ramp.ramp_ver2.fit_v40_v21 (explicit post-KMEANSLOW backend)",
            markers=MARKERS15,
            baseline=completed_baseline,
            aligned=aligned,
            coupling_audit=coupling,
            strength_audit=strength,
            evidence_audit=evidence,
            reference_weight_audit=weights,
            profile_audit=profile_audit,
            algorithm_audit=algorithm_audit,
            strict_rank_inversions=int(inversions),
        ),
        compatibility,
    )


def save_result(result: AlignmentResult, metadata: pd.DataFrame) -> dict[str, Any]:
    folder = OUTPUT / result.name
    folder.mkdir(parents=True, exist_ok=True)
    baseline_npy = folder / "kmeanslow_baseline.npy"
    aligned_npy = folder / "ramp_aligned.npy"
    baseline_csv = folder / "kmeanslow_baseline.csv.gz"
    aligned_csv = folder / "ramp_aligned.csv.gz"
    np.save(baseline_npy, result.baseline.astype(np.float32))
    np.save(aligned_npy, result.aligned.astype(np.float32))
    identity = metadata[IDENTITY_COLUMNS].reset_index(drop=True)
    write_csv_gz(
        pd.concat(
            [identity, pd.DataFrame(result.baseline, columns=result.markers)], axis=1
        ),
        baseline_csv,
    )
    write_csv_gz(
        pd.concat(
            [identity, pd.DataFrame(result.aligned, columns=result.markers)], axis=1
        ),
        aligned_csv,
    )

    audits = {
        "coupling_audit.csv": result.coupling_audit,
        "strength_audit.csv": result.strength_audit,
        "evidence_audit.csv": result.evidence_audit,
        "reference_weight_audit.csv": result.reference_weight_audit,
        "profile_audit.csv": result.profile_audit,
        "algorithm_audit.csv": result.algorithm_audit,
    }
    for name, frame in audits.items():
        frame.to_csv(folder / name, index=False)

    displacement = np.abs(result.aligned - result.baseline)
    files = [baseline_npy, aligned_npy, baseline_csv, aligned_csv]
    files.extend(folder / name for name in audits)
    manifest = {
        "condition": result.name,
        "family": result.family,
        "algorithm": "RAMP",
        "algorithm_version": ramp230.__version__,
        "canonical_source": str(Path(ramp230.__file__).resolve()),
        "canonical_source_sha256": sha256_file(Path(ramp230.__file__)),
        "entrypoint": result.entrypoint,
        "cells": int(len(result.aligned)),
        "markers": result.markers,
        "marker_count": int(len(result.markers)),
        "datasets": DATASETS,
        "reference_dataset": DATASETS[0],
        "labels_or_coordinates_used_for_fitting": False,
        "strict_rank_inversions": int(result.strict_rank_inversions),
        "observed_maximum_displacement": float(displacement.max(initial=0.0)),
        "cell_marker_values_over_0_75": int(
            np.sum(displacement > ramp230.MAX_DISPLACEMENT + 1e-10)
        ),
        "outputs": {
            path.name: sha256_file(path)
            for path in files
        },
    }
    write_json(folder / "manifest.json", manifest)
    return manifest


def main() -> None:
    if ramp230.__version__ != "2.3.0":
        raise AssertionError(f"Expected RAMP 2.3.0, imported {ramp230.__version__}")
    expected_source = (ALGORITHM_ROOT / "ramp" / "ramp_ver2.py").resolve()
    if Path(ramp230.__file__).resolve() != expected_source:
        raise AssertionError(f"Unexpected RAMP source: {ramp230.__file__}")

    raw15, metadata, source_hashes = load_inputs()
    fit_metadata = metadata[["dataset", "unit_key"]].copy()

    print("Fit RAMP 2.3.0: 14 shared markers without CD19", flush=True)
    public14 = ramp230.fit_transform(
        raw15[:, :14],
        fit_metadata,
        MARKERS14,
        datasets=DATASETS,
        family="bm_prediction_story_observed14",
    )
    without = from_public(
        "without_cd19", "bm_prediction_story_observed14", MARKERS14, public14
    )

    print("Fit RAMP 2.3.0: 15 observed markers with failed CODEX CD19 negative control", flush=True)
    wrong_public = ramp230.fit_transform(
        raw15,
        fit_metadata,
        MARKERS15,
        datasets=DATASETS,
        family="bm_prediction_story_wrong_observed15",
    )
    wrong_observed = from_public(
        "with_wrong_observed_cd19",
        "bm_prediction_story_wrong_observed15",
        MARKERS15,
        wrong_public,
    )

    print("Fit RAMP 2.3.0: completed 15 markers with predicted CODEX CD19", flush=True)
    with_predicted, prediction_checks = fit_with_predicted_cd19(
        raw15, metadata, without.baseline
    )

    print("Save aligned matrices and audits", flush=True)
    manifests = {
        without.name: save_result(without, metadata),
        wrong_observed.name: save_result(wrong_observed, metadata),
        with_predicted.name: save_result(with_predicted, metadata),
    }
    metadata_path = OUTPUT / "cell_metadata.csv.gz"
    write_csv_gz(metadata, metadata_path)
    prediction_checks_path = OUTPUT / "predicted_cd19_compatibility.json"
    write_json(prediction_checks_path, prediction_checks)
    top_manifest = {
        "analysis": "three-condition BM CD19 prediction story alignment",
        "status": "complete",
        "algorithm": {
            "name": "RAMP",
            "version": ramp230.__version__,
            "canonical_source": str(expected_source),
            "canonical_source_sha256": sha256_file(expected_source),
        },
        "cells": EXPECTED_CELLS,
        "dataset_cells": EXPECTED_COUNTS,
        "dataset_order": DATASETS,
        "conditions": manifests,
        "predicted_cd19_policy": (
            "predicted CODEX CD19 is already KMEANSLOW-scale and is inserted "
            "without a second preprocessing pass; reference CD19 remains observed; "
            "the completed KMEANSLOW interface is serialized as float32"
        ),
        "wrong_observed_cd19_policy": (
            "separate negative control only; the known failed observed CODEX CD19 "
            "channel is excluded from both main conditions"
        ),
        "labels_or_coordinates_used_for_fitting": False,
        "source_hashes": source_hashes,
        "prediction_compatibility": prediction_checks,
        "runtime": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "scipy": version("scipy"),
            "scikit_learn": version("scikit-learn"),
        },
        "outputs": {
            "cell_metadata.csv.gz": sha256_file(metadata_path),
            "predicted_cd19_compatibility.json": sha256_file(prediction_checks_path),
        },
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__)),
        },
    }
    write_json(OUTPUT / "manifest.json", top_manifest)
    print(json.dumps(top_manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
