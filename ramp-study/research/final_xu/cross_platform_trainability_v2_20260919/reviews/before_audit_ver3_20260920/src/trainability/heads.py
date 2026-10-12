"""Common prediction heads; selection accepts source validation only.

MLPs use explicit validation each epoch, never sklearn's hidden internal split.
All fit/validation rows have already been charged to the supervision ledger.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, asdict
import time
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

from .core import require, unique_ids


@dataclass(frozen=True)
class Candidate:
    family: str
    scale: str = "identity"
    regularization: float = 1.0
    neighbors: int = 15
    weights: str = "uniform"
    width: int = 32
    learning_rate: float = 0.001
    max_epochs: int = 100
    patience: int = 15
    batch_size: int = 128

    def validate(self):
        require(self.family in {"linear", "knn", "mlp"}, "Unknown head family")
        require(self.scale in {"identity", "standard"}, "Unknown head scaler")
        require(np.isfinite(self.regularization) and self.regularization > 0, "Invalid regularization")
        require(self.neighbors > 0 and self.width > 0 and self.max_epochs > 0
                and self.patience > 0 and self.batch_size > 0, "Invalid head capacity/budget")
        require(self.weights in {"uniform", "distance"}, "Invalid kNN weights")
        require(np.isfinite(self.learning_rate) and self.learning_rate > 0, "Invalid learning rate")


class ConstantClassifier:
    def __init__(self, label):
        self.label = label
        self.classes_ = np.asarray([label])

    def predict(self, x):
        return np.repeat(self.label, len(x))

    def predict_proba(self, x):
        return np.ones((len(x), 1))


@dataclass
class FittedHead:
    model: object
    scaler: object | None
    task: str
    classes: tuple[str, ...]
    selection: dict
    candidates: list[dict]

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        require(x.ndim == 2 and np.isfinite(x).all(), "Invalid prediction features")
        return self.model.predict(x if self.scaler is None else self.scaler.transform(x))

    def predict_proba(self, x):
        require(self.task == "classification", "Regression has no class probabilities")
        x = np.asarray(x, dtype=float)
        require(x.ndim == 2 and np.isfinite(x).all(), "Invalid prediction features")
        if self.scaler is not None:
            x = self.scaler.transform(x)
        raw = self.model.predict_proba(x)
        out = np.zeros((len(x), len(self.classes)))
        for j, label in enumerate(self.model.classes_):
            out[:, self.classes.index(str(label))] = raw[:, j]
        return out


def parameter_accounting(model, family: str, fit_x: np.ndarray) -> dict:
    if isinstance(model, ConstantClassifier):
        return dict(trainable_head_parameters=0, constant_classifier_fallback=True)
    if family == "knn":
        return dict(trainable_head_parameters=None, stored_training_cells=len(fit_x),
                    feature_storage_bytes=int(fit_x.nbytes),
                    note="Feature storage only; not a total-memory measurement")
    if family == "mlp":
        count = sum(x.size for x in model.coefs_ + model.intercepts_)
    else:
        count = np.asarray(model.coef_).size + np.asarray(model.intercept_).size
    return dict(trainable_head_parameters=int(count), constant_classifier_fallback=False)


def fit_head(fit_x, fit_y, validation_x, validation_y, *, fit_ids, validation_ids,
             task: str, candidates: list[Candidate], source_validation_score,
             classes=(), seed: int = 0) -> FittedHead:
    """Score callable receives (source validation y, predictions), higher is better.

    Supply patient-weighted macro-F1 for J1/J3 or unit/patient-weighted
    correlation for J2. The caller must not close over target answers.
    There are deliberately no target data/label arguments for model selection.
    """
    require(task in {"classification", "regression"}, "Unknown prediction task")
    fi, vi = unique_ids(fit_ids), unique_ids(validation_ids)
    require(not set(fi).intersection(vi), "Source fit/validation overlap")
    fx, vx = np.asarray(fit_x, dtype=float), np.asarray(validation_x, dtype=float)
    fy, vy = np.asarray(fit_y), np.asarray(validation_y)
    require(fx.ndim == vx.ndim == 2 and fx.shape[1] == vx.shape[1] and fx.shape[1] > 0,
            "Head feature shape mismatch")
    require(len(fx) == len(fi) == len(fy) and len(vx) == len(vi) == len(vy)
            and len(fi) > 0 and len(vi) > 0, "Supervised identities do not match rows")
    require(fy.ndim == vy.ndim == 1, "Targets must be one-dimensional")
    require(np.isfinite(fx).all() and np.isfinite(vx).all(), "Nonfinite head input")
    classes = unique_ids(classes, "ontology classes")
    if task == "classification":
        require(bool(classes) and set(fy) <= set(classes) and set(vy) <= set(classes),
                "Source labels outside fixed ontology")
    else:
        fy, vy = fy.astype(float), vy.astype(float)
        require(np.isfinite(fy).all() and np.isfinite(vy).all(), "Nonfinite regression targets")
    require(bool(candidates), "No source-selection candidates")
    records, best = [], None
    seen = set()
    for candidate in candidates:
        candidate.validate()
        effective = asdict(candidate)
        if candidate.family == "knn":
            effective["neighbors"] = min(candidate.neighbors, len(fi))
        key = tuple(sorted(effective.items()))
        if key in seen:
            continue
        seen.add(key)
        start = time.perf_counter()
        scaler = StandardScaler().fit(fx) if candidate.scale == "standard" else None
        x = fx if scaler is None else scaler.transform(fx)
        v = vx if scaler is None else scaler.transform(vx)
        single = task == "classification" and len(np.unique(fy)) == 1
        if single:
            model = ConstantClassifier(fy[0])
        elif candidate.family == "linear":
            model = (LogisticRegression(C=candidate.regularization, max_iter=2000, random_state=seed)
                     if task == "classification" else Ridge(alpha=candidate.regularization))
        elif candidate.family == "knn":
            cls = KNeighborsClassifier if task == "classification" else KNeighborsRegressor
            model = cls(n_neighbors=effective["neighbors"], weights=candidate.weights, n_jobs=1)
        else:
            cls = MLPClassifier if task == "classification" else MLPRegressor
            model = cls(hidden_layer_sizes=(candidate.width,), random_state=seed,
                        batch_size=min(len(fi), candidate.batch_size), early_stopping=False,
                        learning_rate_init=candidate.learning_rate, alpha=candidate.regularization)
        epochs, selected_epoch = 0, None
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            if candidate.family == "mlp" and not single:
                local_best, stale = None, 0
                for epoch in range(candidate.max_epochs):
                    if task == "classification":
                        model.partial_fit(x, fy, classes=np.asarray(classes))
                    else:
                        model.partial_fit(x, fy)
                    epochs += 1
                    score = source_validation_score(vy, model.predict(v))
                    require(score is not None and np.isfinite(score),
                            "Source validation not evaluable; cannot select using target answers")
                    if local_best is None or score > local_best[0]:
                        local_best = (float(score), deepcopy(model), epoch+1)
                        stale = 0
                    else:
                        stale += 1
                    if stale >= candidate.patience:
                        break
                score, model, selected_epoch = local_best
            else:
                if not single:
                    model.fit(x, fy)
                score = source_validation_score(vy, model.predict(v))
                require(score is not None and np.isfinite(score), "Source validation not evaluable")
        record = dict(candidate=effective, source_validation_score=float(score),
                      seconds=time.perf_counter()-start, epochs_executed=epochs,
                      selected_epoch=selected_epoch, warnings=[str(w.message) for w in captured],
                      **parameter_accounting(model, candidate.family, x))
        records.append(record)
        if best is None or score > best[0]:
            best = (score, model, scaler, record)
    return FittedHead(best[1], best[2], task, classes, best[3], records)
