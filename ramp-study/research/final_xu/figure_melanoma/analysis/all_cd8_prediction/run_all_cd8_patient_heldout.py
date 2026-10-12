#!/usr/bin/env python3
"""Predict the fixed spatial CD8 target among *all* spatial CD8 T cells.

Prediction pool
---------------
Moldoveanu IMC cells annotated by the provider as either ``Tc.ae`` or
``Tc.naive``.

Positive label
--------------
``Tc.ae`` AND ``prob.Ki67 >= 0.5`` AND nearest melanoma centroid distance
``<= 30 um``.

The provider ``Tc.ae`` label is therefore used only to construct the source
ground truth.  It is not used as an inference-time pre-filter.  Each patient is
held out in turn and scored by a model trained on the remaining patients using
only the frozen clean15 RAMP-aligned shared-marker matrix.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
FIGURE_ROOT = HERE.parents[1]
FINAL_XU = FIGURE_ROOT.parent
ALIGNMENT_ROOT = HERE.parent / "alignment/output"

META_PATH = FIGURE_ROOT / "inputs/spatial_cell_metadata.csv.gz"
ALIGNMENT_META_PATH = ALIGNMENT_ROOT / "cell_metadata.csv.gz"
RAMP_PATH = ALIGNMENT_ROOT / "ramp_aligned.npy"
MANIFEST_PATH = ALIGNMENT_ROOT / "manifest.json"

DISTANCE_UM = 30.0
KI67_CUTOFF = 0.5
CD8_LABELS = ("Tc.ae", "Tc.naive")
TOP_FRACTIONS = (0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30)
MIN_EVAL_CELLS = 20
SEED = 20260829

os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(HERE / ".xdg-cache"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv_gz(frame: pd.DataFrame, path: Path) -> None:
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(fileobj=raw_handle, mode="wb", mtime=0) as handle:
            frame.to_csv(handle, index=False)


def fit_model(x: np.ndarray, y: np.ndarray) -> object:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.5,
            class_weight="balanced",
            max_iter=5000,
            solver="liblinear",
            random_state=SEED,
        ),
    )
    model.fit(x, y)
    return model


def call_top_fraction(score: np.ndarray, fraction: float) -> np.ndarray:
    n = max(1, int(np.ceil(fraction * len(score))))
    called = np.zeros(len(score), dtype=bool)
    called[np.argsort(-score, kind="mergesort")[:n]] = True
    return called


def call_metrics(y: np.ndarray, called: np.ndarray) -> dict[str, float | int]:
    y = y.astype(bool)
    called = called.astype(bool)
    tp = int((y & called).sum())
    fp = int((~y & called).sum())
    fn = int((y & ~called).sum())
    prevalence = float(y.mean())
    precision = float(tp / called.sum()) if called.sum() else np.nan
    recall = float(tp / y.sum()) if y.sum() else np.nan
    f1 = float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "n_called": int(called.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "enrichment": float(precision / prevalence) if prevalence > 0 else np.nan,
    }


def load_all_cd8() -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    metadata = pd.read_csv(META_PATH, low_memory=False)
    alignment_metadata = pd.read_csv(ALIGNMENT_META_PATH, low_memory=False)
    ramp = np.load(RAMP_PATH).astype(np.float64)
    if len(metadata) != len(alignment_metadata) or len(metadata) != len(ramp):
        raise AssertionError("Metadata and frozen alignment row counts differ")
    for left, right in (
        (metadata["cell_uid_ordered"], alignment_metadata["cell_uid"]),
        (metadata["dataset"], alignment_metadata["dataset"]),
        (metadata["donor"], alignment_metadata["donor"]),
    ):
        if not left.astype(str).equals(right.astype(str)):
            raise AssertionError("Metadata and frozen alignment row order differ")

    manifest = json.loads(MANIFEST_PATH.read_text())
    markers = list(manifest["shared_markers"])
    if ramp.shape[1] != len(markers):
        raise AssertionError("Marker manifest does not match RAMP matrix")

    focal = (
        metadata["dataset"].astype(str).eq("Moldoveanu IMC")
        & metadata["Cluster.v2"].astype(str).isin(CD8_LABELS)
    ).to_numpy()
    required_target_fields = metadata.loc[
        focal, ["prob.Ki67", "distance_to_nearest_tumor"]
    ]
    if required_target_fields.isna().any().any():
        raise AssertionError("Every spatial all-CD8 cell must have Ki67 and tumor distance")
    cells = metadata.loc[focal].copy().reset_index().rename(columns={"index": "global_row"})
    cells["patient"] = cells["donor"].astype(str)
    cells["is_tcae"] = cells["Cluster.v2"].astype(str).eq("Tc.ae")
    cells["is_tcnaive"] = cells["Cluster.v2"].astype(str).eq("Tc.naive")
    cells["target_30um"] = (
        cells["is_tcae"].to_numpy(bool)
        & (cells["prob.Ki67"].to_numpy(float) >= KI67_CUTOFF)
        & (cells["distance_to_nearest_tumor"].to_numpy(float) <= DISTANCE_UM)
    )
    return cells, ramp[focal], markers


def run() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    cells, x, markers = load_all_cd8()
    patient = cells["patient"].to_numpy(str)
    target = cells["target_30um"].to_numpy(int)
    patients = sorted(pd.unique(patient))

    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for number, held_out_patient in enumerate(patients, start=1):
        test = patient == held_out_patient
        train = ~test
        model = fit_model(x[train], target[train])
        score = model.predict_proba(x[test])[:, 1]
        y = target[test]
        local = cells.loc[test]

        n = int(test.sum())
        n_target = int(y.sum())
        prevalence = float(y.mean())
        row: dict[str, object] = {
            "held_out_patient": held_out_patient,
            "n_all_cd8": n,
            "n_tcae": int(local["is_tcae"].sum()),
            "n_tcnaive": int(local["is_tcnaive"].sum()),
            "n_target": n_target,
            "prevalence": prevalence,
            "evaluable_discrimination": bool(n >= MIN_EVAL_CELLS and 0 < n_target < n),
            "AUROC": np.nan,
            "AUPRC": np.nan,
            "AUPRC_over_prevalence": np.nan,
        }
        if len(np.unique(y)) == 2:
            auroc = float(roc_auc_score(y, score))
            auprc = float(average_precision_score(y, score))
            row.update({
                "AUROC": auroc,
                "AUPRC": auprc,
                "AUPRC_over_prevalence": float(auprc / prevalence),
            })

        calls: dict[str, np.ndarray] = {}
        for fraction in TOP_FRACTIONS:
            label = f"top{100 * fraction:g}".replace(".", "p")
            called = call_top_fraction(score, fraction)
            calls[label] = called
            for key, value in call_metrics(y, called).items():
                row[f"{label}_{key}"] = value
        metric_rows.append(row)

        fold = local[[
            "global_row",
            "cell_uid_ordered",
            "patient",
            "Cluster.v2",
            "is_tcae",
            "is_tcnaive",
            "prob.Ki67",
            "distance_to_nearest_tumor",
            "target_30um",
        ]].copy()
        fold["model_score"] = score
        for label, called in calls.items():
            fold[f"called_{label}"] = called
        prediction_frames.append(fold)
        print(
            f"{number:02d}/{len(patients)} {held_out_patient}: all-CD8={n}, "
            f"Tc.ae={int(local['is_tcae'].sum())}, Tc.naive={int(local['is_tcnaive'].sum())}, "
            f"target={n_target}",
            flush=True,
        )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    evaluable = metrics.loc[metrics["evaluable_discrimination"]]
    summary = {
        "analysis": "patient-held-out prediction of fixed spatial target among all CD8",
        "prediction_pool": "Moldoveanu Cluster.v2 in {Tc.ae, Tc.naive}",
        "ground_truth": "Tc.ae AND prob.Ki67 >= 0.5 AND nearest melanoma centroid <= 30 um",
        "inference_features": "frozen clean15 RAMP-aligned shared-marker values only",
        "model": "StandardScaler + LogisticRegression(C=0.5, class_weight=balanced)",
        "n_all_cd8": int(len(cells)),
        "n_tcae": int(cells["is_tcae"].sum()),
        "n_tcnaive": int(cells["is_tcnaive"].sum()),
        "n_targets": int(cells["target_30um"].sum()),
        "n_patients": int(cells["patient"].nunique()),
        "n_evaluable_patients": int(len(evaluable)),
        "median_AUROC": float(evaluable["AUROC"].median()),
        "median_AUPRC": float(evaluable["AUPRC"].median()),
        "median_AUPRC_over_prevalence": float(evaluable["AUPRC_over_prevalence"].median()),
        "top_fractions": list(TOP_FRACTIONS),
        "shared_markers": markers,
        "software": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "inputs": {
            "metadata": str(META_PATH),
            "alignment_metadata": str(ALIGNMENT_META_PATH),
            "ramp_aligned": str(RAMP_PATH),
            "alignment_manifest": str(MANIFEST_PATH),
            "ramp_sha256": sha256(RAMP_PATH),
        },
    }
    return metrics, predictions, summary


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    metrics, predictions, summary = run()
    metrics.to_csv(HERE / "patient_metrics.csv", index=False)
    write_csv_gz(predictions, HERE / "outer_patient_predictions.csv.gz")
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
