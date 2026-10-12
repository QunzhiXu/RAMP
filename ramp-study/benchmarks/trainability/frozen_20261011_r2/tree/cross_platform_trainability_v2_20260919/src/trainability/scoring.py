"""Scoring-only functions. These receive answers after predictions are frozen."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import confusion_matrix

from .core import file_sha256, output_path, require, unique_ids, write_json


def classification_score(truth, prediction, patient_ids, classes, *, include_sufficient_stats=False) -> dict:
    truth, prediction, patients = map(np.asarray, (truth, prediction, patient_ids))
    classes = unique_ids(classes, "ontology classes")
    require(truth.ndim == 1 and truth.shape == prediction.shape == patients.shape
            and len(truth) > 0, "Classification scoring shape/support mismatch")
    require(set(truth) <= set(classes) and set(prediction) <= set(classes), "Label outside ontology")
    require(all(str(p) not in {"", "None", "nan"} for p in patients), "Missing patient identity")
    matrices = []
    for patient in np.unique(patients):
        m = patients == patient
        matrices.append(confusion_matrix(truth[m], prediction[m], labels=classes)/m.sum())
    cm = np.mean(matrices, axis=0)
    tp = np.diag(cm)
    den = cm.sum(axis=0) + cm.sum(axis=1)
    f1 = np.divide(2*tp, den, out=np.zeros(len(classes)), where=den > 0)
    support = cm.sum(axis=1)
    recall = np.divide(tp, support, out=np.zeros(len(classes)), where=support > 0)
    result=dict(macro_f1=float(f1.mean()), balanced_accuracy=float(recall.mean()),
                accuracy=float(tp.sum()), confusion_matrix=cm.tolist(),
                patients=len(matrices), cells=len(truth), classes=list(classes),
                zero_division=0)
    if include_sufficient_stats:
        result['patient_confusion_matrices']=[dict(patient=str(p),matrix=m.tolist())
            for p,m in zip(np.unique(patients),matrices)]
    return result


def regression_score(measured, prediction, units, patient_ids, *, minimum_cells=30) -> dict:
    measured, prediction = np.asarray(measured, dtype=float), np.asarray(prediction, dtype=float)
    units, patients = np.asarray(units, dtype=str), np.asarray(patient_ids, dtype=str)
    require(measured.ndim == 1 and measured.shape == prediction.shape == units.shape == patients.shape
            and len(measured) > 0, "Regression scoring shape/support mismatch")
    require(minimum_cells >= 2, "Invalid minimum correlation support")
    require(np.isfinite(measured).all(), "Nonfinite truth must be resolved in frozen eligibility")
    require(all(x not in {"", "None", "nan"} for x in patients), "Missing patient identity")
    require(all(x not in {"", "None", "nan"} for x in units), "Missing unit identity")
    # Do not selectively remove failed cells/units to produce a survivor score.
    if not np.isfinite(prediction).all():
        return dict(status="run_failure", reason="nonfinite_predictions", units=[])
    rows = []
    for unit in np.unique(units):
        m = units == unit
        pp = np.unique(patients[m])
        require(len(pp) == 1, "Unit belongs to multiple patients")
        y, pred = measured[m], prediction[m]
        target = (rankdata(y, method="average")-.5)/len(y)
        mae = float(np.abs(np.clip(pred, 0, 1)-target).mean())
        row = dict(unit=str(unit), patient=str(pp[0]), cells=int(m.sum()),
                   mae=mae, rho=None, effective_rho=None)
        if len(y) < minimum_cells:
            row["rho_status"] = "insufficient_truth_support"
        elif np.ptp(y) == 0:
            row["rho_status"] = "constant_truth"
        elif np.ptp(pred) == 0:
            row.update(rho_status="constant_prediction", effective_rho=0.0)
        else:
            rho = float(spearmanr(y, pred).statistic)
            require(np.isfinite(rho), "Unexpected undefined correlation")
            row.update(rho_status="ok", rho=rho, effective_rho=rho)
        rows.append(row)

    def patient_mean(key, eligible_only=False):
        values = {}
        for r in rows:
            if r[key] is None or (eligible_only and r["effective_rho"] is None):
                continue
            values.setdefault(r["patient"], []).append(r[key])
        means = [float(np.mean(v)) for v in values.values()]
        return (float(np.mean(means)) if means else None), len(means)

    rho, n_rho = patient_mean("effective_rho")
    mae, n_mae = patient_mean("mae")
    mae_support, _ = patient_mean("mae", True)
    return dict(status="ok", effective_spearman=rho, mae=mae, mae_rho_support=mae_support,
                rho_patients=n_rho, mae_patients=n_mae, units=rows,
                rho_status_counts=dict(Counter(r["rho_status"] for r in rows)))


def freeze_predictions(path, cell_ids, predictions, provenance: dict) -> dict:
    """Write-once prediction artifact. No target answer argument exists."""
    path = output_path(path)
    ids = unique_ids(cell_ids)
    pred = np.asarray(predictions)
    require(pred.ndim in (1, 2) and len(pred) == len(ids), "Prediction identity mismatch")
    require(pred.dtype.kind in "fiuUS", "Object prediction arrays are forbidden")
    if pred.dtype.kind in "fiu":
        require(np.isfinite(pred).all(), "Nonfinite predictions cannot be frozen as valid")
    sidecar = path.with_suffix(".json")
    require(not path.exists() and not sidecar.exists(), "Prediction artifact already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        np.savez_compressed(f, cell_ids=np.asarray(ids, dtype=str), predictions=pred)
    receipt = dict(prediction_file=str(path), sha256=file_sha256(path),
                   n_cells=len(ids), provenance=provenance, target_answers_included=False)
    write_json(sidecar, receipt)
    return receipt


def load_frozen_predictions(receipt: dict, expected_cell_ids) -> np.ndarray:
    path = Path(receipt["prediction_file"])
    require(file_sha256(path) == receipt["sha256"], "Frozen predictions changed")
    ids = unique_ids(expected_cell_ids)
    with np.load(path, allow_pickle=False) as data:
        actual = unique_ids(data["cell_ids"])
        require(set(actual) == set(ids), "Prediction coverage differs from fixed evaluation IDs")
        index = {x: n for n, x in enumerate(actual)}
        return data["predictions"][[index[i] for i in ids]]
