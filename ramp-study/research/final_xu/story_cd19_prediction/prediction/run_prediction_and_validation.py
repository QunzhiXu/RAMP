#!/usr/bin/env python3
"""Fit the frozen masked-marker model and predict failed CODEX CD19.

The predictor sees only the 14 shared, healthy markers. CD19 is available as a
training target in Triana AbSeq and BM CyTOF, but the failed observed CODEX
channel, provider labels, and spatial coordinates are never supplied.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from importlib.metadata import version
from pathlib import Path


HERE = Path(__file__).resolve().parent
CACHE = HERE / ".cache"
CACHE.mkdir(parents=True, exist_ok=True)
for name, value in {
    "LOKY_MAX_CPU_COUNT": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "MPLCONFIGDIR": str(CACHE / "matplotlib"),
    "NUMBA_CACHE_DIR": str(CACHE / "numba"),
    "XDG_CACHE_HOME": str(CACHE / "xdg"),
    "JOBLIB_TEMP_FOLDER": str(CACHE / "joblib"),
}.items():
    os.environ.setdefault(name, value)
for folder in ["matplotlib", "numba", "xdg", "joblib"]:
    (CACHE / folder).mkdir(parents=True, exist_ok=True)
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch


STORY = HERE.parent
SAMPLE = STORY / "sampled_data"
ALGORITHM = STORY.parent / "algorithm"
sys.path.insert(0, str(ALGORITHM))
from ramp import ramp_ver2 as ramp  # noqa: E402
import masked_marker_model as model_definition  # noqa: E402


OUTPUT = HERE / "output"
MARKERS14 = list(model_definition.NORMAL_MARKERS)
MARKERS15 = MARKERS14 + ["CD19"]
DATA_FOLDERS = ["bm_codex", "triana_abseq", "bm_cytof"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_data() -> tuple[np.ndarray, pd.DataFrame]:
    matrices: list[np.ndarray] = []
    metadata: list[pd.DataFrame] = []
    for folder in DATA_FOLDERS:
        frame = pd.read_csv(SAMPLE / folder / "source_input_15.csv.gz")
        matrices.append(frame[MARKERS15].to_numpy(np.float64))
        metadata.append(frame[["cell_id", "dataset", "donor", "unit_key"]].copy())
    return np.vstack(matrices), pd.concat(metadata, ignore_index=True)


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.0
    value = float(spearmanr(x, y).statistic)
    return value if np.isfinite(value) else 0.0


def high_state(values: np.ndarray, marker: str, unit: str) -> np.ndarray:
    fit = KMeans(
        n_clusters=2, n_init=20, max_iter=500,
        random_state=model_definition.stable_seed("HIGH", marker, unit),
    ).fit(values[:, None])
    return fit.labels_ == int(np.argmax(fit.cluster_centers_.ravel()))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if ramp.__version__ != "2.3.0":
        raise AssertionError(f"Expected RAMP 2.3.0, imported {ramp.__version__}")
    expected_ramp = (ALGORITHM / "ramp" / "ramp_ver2.py").resolve()
    if Path(ramp.__file__).resolve() != expected_ramp:
        raise AssertionError(f"Unexpected RAMP source: {ramp.__file__}")
    if model_definition.ramp is not ramp:
        raise AssertionError("Prediction model is not using canonical RAMP 2.3.0")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)

    raw15, metadata = load_data()
    units = metadata.unit_key.astype(str).to_numpy()
    datasets = metadata.dataset.astype(str).to_numpy()

    print("KMEANSLOW preprocessing", flush=True)
    baseline14 = ramp.kmeanslow_global(raw15[:, :14], units, datasets, MARKERS14)
    processed_cd19 = ramp.kmeanslow_global(raw15[:, 14:15], units, datasets, ["CD19"])[:, 0]
    platform_ids = metadata.dataset.astype(str).map(model_definition.PLATFORM_TO_ID).to_numpy(int)
    codex_rows = np.flatnonzero(platform_ids == model_definition.CODEX_ID)

    cd19_target = processed_cd19.copy()
    cd19_target[codex_rows] = np.nan
    target_values = np.column_stack([baseline14, cd19_target]).astype(np.float32)
    features = StandardScaler().fit_transform(baseline14).astype(np.float32)
    balance = model_definition.target_balance_weights(target_values, metadata)
    fit_indices = model_definition.donor_balanced_fit_indices(metadata)

    print("Training masked-marker model", flush=True)
    model = model_definition.train_model(
        features, target_values, balance, platform_ids, fit_indices
    )
    torch.save(model.state_dict(), OUTPUT / "masked_marker_model.pt")

    print("Predicting CODEX CD19", flush=True)
    predicted_cd19, triana_cd19, cytof_cd19 = model_definition.predict_cd19(
        model, features, codex_rows
    )
    np.savez_compressed(
        OUTPUT / "codex_cd19_prediction.npz",
        codex_rows=codex_rows,
        observed_kmeanslow=processed_cd19[codex_rows].astype(np.float32),
        predicted_kmeanslow=predicted_cd19.astype(np.float32),
        triana_component=triana_cd19.astype(np.float32),
        cytof_component=cytof_cd19.astype(np.float32),
    )
    metadata.iloc[codex_rows].to_csv(
        OUTPUT / "codex_prediction_metadata.csv.gz",
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )

    print("Ground-truth validation on 14 observed CODEX markers", flush=True)
    predictions = np.empty((len(codex_rows), len(MARKERS14)), dtype=np.float32)
    records: list[dict[str, object]] = []
    codex_units = metadata.iloc[codex_rows].unit_key.astype(str).to_numpy()
    for marker_index, marker in enumerate(MARKERS14):
        print(f"  {marker_index + 1:02d}/14 {marker}", flush=True)
        prediction, _, _ = model_definition.predict_normal_marker(
            model, features, codex_rows, marker_index
        )
        predictions[:, marker_index] = prediction.astype(np.float32)
        truth = baseline14[codex_rows, marker_index]
        for unit in sorted(np.unique(codex_units)):
            local = codex_units == unit
            state = high_state(truth[local], marker, unit)
            records.append({
                "marker": marker,
                "codex_donor": unit,
                "n_cells": int(np.sum(local)),
                "spearman": safe_spearman(truth[local], prediction[local]),
                "high_state_auroc": float(roc_auc_score(state.astype(int), prediction[local])),
                "mae": float(np.mean(np.abs(truth[local] - prediction[local]))),
            })
    np.save(OUTPUT / "codex_heldout_marker_predictions.npy", predictions)
    validation = pd.DataFrame(records)
    validation.to_csv(OUTPUT / "heldout_marker_validation_by_donor.csv", index=False)
    summary = (
        validation.groupby("marker", as_index=False)
        .agg(
            median_donor_spearman=("spearman", "median"),
            mean_high_state_auroc=("high_state_auroc", "mean"),
            median_mae=("mae", "median"),
        )
    )
    summary.to_csv(OUTPUT / "heldout_marker_validation_summary.csv", index=False)
    manifest = {
        "analysis": "BM CODEX CD19 completion and observed-marker holdout validation",
        "sample": "final_xu/story_cd19_prediction/sampled_data",
        "algorithm": {
            "name": "RAMP",
            "version": ramp.__version__,
            "canonical_source": str(Path(ramp.__file__).resolve()),
            "canonical_source_sha256": sha256_file(Path(ramp.__file__)),
        },
        "prediction_input": "KMEANSLOW_GLOBAL values of 14 healthy shared markers",
        "prediction_target": "KMEANSLOW_GLOBAL CD19 in Triana AbSeq and BM CyTOF only",
        "failed_codex_cd19_used_for_training_or_selection": False,
        "cell_labels_or_spatial_coordinates_used": False,
        "fit_cells_per_platform": model_definition.FIT_CAP_PER_PLATFORM,
        "source_average": "0.5 Triana-like + 0.5 CyTOF-like",
        "validation": "each healthy CODEX marker masked and predicted by the identical two-source rule",
        "median_marker_spearman": float(summary.median_donor_spearman.median()),
        "mean_marker_high_state_auroc": float(summary.mean_high_state_auroc.mean()),
        "cells": int(len(metadata)),
        "dataset_cells": metadata.groupby("dataset", sort=False).size().astype(int).to_dict(),
        "inputs": {
            folder: {
                "path": str((SAMPLE / folder / "source_input_15.csv.gz").resolve()),
                "sha256": sha256_file(SAMPLE / folder / "source_input_15.csv.gz"),
            }
            for folder in DATA_FOLDERS
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "scipy": version("scipy"),
            "scikit_learn": version("scikit-learn"),
            "torch": version("torch"),
            "device": "cpu",
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        },
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(summary.sort_values("median_donor_spearman", ascending=False).to_string(index=False), flush=True)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
