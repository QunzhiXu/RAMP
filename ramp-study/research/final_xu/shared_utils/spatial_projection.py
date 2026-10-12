#!/usr/bin/env python3
"""Shared utilities for continuous cross-platform molecular projection."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


SEED = 20260821


def write_csv_gz(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(filename=str(path), mode="wb", mtime=0) as handle:
        frame.to_csv(handle, index=False)


def within_unit_percentile(values: np.ndarray, units: np.ndarray) -> np.ndarray:
    """Average-tie empirical percentiles, independently within each donor."""
    values = np.asarray(values, float)
    units = np.asarray(units, str)
    out = np.full(len(values), np.nan, float)
    for unit in pd.unique(units):
        rows = np.flatnonzero(units == unit)
        finite = rows[np.isfinite(values[rows])]
        if len(finite) == 0:
            continue
        if len(finite) == 1:
            out[finite] = 0.5
        else:
            out[finite] = (rankdata(values[finite], method="average") - 1.0) / (
                len(finite) - 1.0
            )
    return out


def canonical_features(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    keep = np.nanstd(values, axis=0) > 1e-10
    if not np.any(keep):
        raise ValueError("No nonconstant shared features")
    return StandardScaler().fit_transform(values[:, keep]).astype(np.float32)


def weighted_knn_predict(
    reference_x: np.ndarray,
    reference_y: np.ndarray,
    query_x: np.ndarray,
    *,
    k: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse-distance weighted kNN; returns prediction and effective N."""
    reference_x = np.asarray(reference_x, float)
    query_x = np.asarray(query_x, float)
    reference_y = np.asarray(reference_y, float)
    finite = np.isfinite(reference_y) & np.isfinite(reference_x).all(axis=1)
    reference_x, reference_y = reference_x[finite], reference_y[finite]
    if len(reference_x) < 2:
        raise ValueError("Fewer than two finite reference cells")
    k_use = min(int(k), len(reference_x))
    nn = NearestNeighbors(n_neighbors=k_use, metric="euclidean", n_jobs=1)
    distance, index = nn.fit(reference_x).kneighbors(query_x)
    positive = distance[distance > 1e-12]
    scale = float(np.median(positive)) if len(positive) else 1.0
    weight = np.exp(-distance / max(scale, 1e-12))
    weight /= np.maximum(weight.sum(axis=1, keepdims=True), 1e-12)
    prediction = np.sum(weight * reference_y[index], axis=1)
    effective_n = 1.0 / np.maximum(np.sum(weight * weight, axis=1), 1e-12)
    return prediction, effective_n


def donor_cv_predictions(
    features: np.ndarray,
    target: np.ndarray,
    donors: np.ndarray,
    *,
    eligible: np.ndarray | None = None,
    k: int = 30,
) -> np.ndarray:
    donors = np.asarray(donors, str)
    eligible = np.ones(len(target), bool) if eligible is None else np.asarray(eligible, bool)
    out = np.full(len(target), np.nan)
    for donor in pd.unique(donors):
        query = eligible & (donors == donor) & np.isfinite(target)
        train = eligible & (donors != donor) & np.isfinite(target)
        if query.sum() == 0 or train.sum() < 10:
            continue
        out[query], _ = weighted_knn_predict(
            features[train], target[train], features[query], k=k
        )
    return out


def validation_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    units: np.ndarray,
    *,
    eligible: np.ndarray | None = None,
    top_fraction: float = 0.25,
) -> pd.DataFrame:
    observed = np.asarray(observed, float)
    predicted = np.asarray(predicted, float)
    units = np.asarray(units, str)
    eligible = np.ones(len(observed), bool) if eligible is None else np.asarray(eligible, bool)
    records = []
    for unit in pd.unique(units):
        use = eligible & (units == unit) & np.isfinite(observed) & np.isfinite(predicted)
        if use.sum() < 20:
            continue
        obs, pred = observed[use], predicted[use]
        rho = spearmanr(obs, pred).statistic if np.nanstd(obs) > 0 and np.nanstd(pred) > 0 else np.nan
        threshold = np.quantile(obs, 1.0 - top_fraction)
        truth = obs >= threshold
        if truth.all() or (~truth).all():
            auroc = auprc = np.nan
        else:
            auroc = roc_auc_score(truth, pred)
            auprc = average_precision_score(truth, pred)
        ptop = pred >= np.quantile(pred, 1.0 - top_fraction)
        enrichment = (
            float(np.mean(obs[ptop]) - np.mean(obs[~ptop])) if ptop.any() and (~ptop).any() else np.nan
        )
        records.append({
            "unit": unit,
            "n": int(use.sum()),
            "spearman": float(rho),
            "auroc_observed_top_quartile": float(auroc),
            "auprc_observed_top_quartile": float(auprc),
            "observed_percentile_difference_predicted_top_vs_rest": enrichment,
        })
    return pd.DataFrame(records)


def nearest_class_distance(
    metadata: pd.DataFrame,
    query_mask: np.ndarray,
    reference_mask: np.ndarray,
    *,
    unit_col: str,
    x_col: str,
    y_col: str,
) -> np.ndarray:
    """Nearest reference-class distance within the same physical image/unit."""
    query_mask = np.asarray(query_mask, bool)
    reference_mask = np.asarray(reference_mask, bool)
    out = np.full(len(metadata), np.nan)
    for unit, block in metadata.groupby(unit_col, sort=False):
        rows = block.index.to_numpy(int)
        q = rows[query_mask[rows]]
        r = rows[reference_mask[rows]]
        if len(q) == 0 or len(r) == 0:
            continue
        ref_xy = metadata.loc[r, [x_col, y_col]].to_numpy(float)
        qry_xy = metadata.loc[q, [x_col, y_col]].to_numpy(float)
        valid_r = np.isfinite(ref_xy).all(axis=1)
        valid_q = np.isfinite(qry_xy).all(axis=1)
        if valid_r.sum() == 0 or valid_q.sum() == 0:
            continue
        distance, _ = NearestNeighbors(n_neighbors=1).fit(ref_xy[valid_r]).kneighbors(qry_xy[valid_q])
        out[q[valid_q]] = distance[:, 0]
    return out


def top_rest_summary(
    score: np.ndarray,
    outcome: np.ndarray,
    units: np.ndarray,
    *,
    eligible: np.ndarray | None = None,
    top_fraction: float = 0.25,
    outcome_name: str = "outcome",
) -> pd.DataFrame:
    score, outcome, units = np.asarray(score, float), np.asarray(outcome, float), np.asarray(units, str)
    eligible = np.ones(len(score), bool) if eligible is None else np.asarray(eligible, bool)
    records = []
    for unit in pd.unique(units):
        use = eligible & (units == unit) & np.isfinite(score) & np.isfinite(outcome)
        if use.sum() < 20:
            continue
        cutoff = np.quantile(score[use], 1.0 - top_fraction)
        high = use & (score >= cutoff)
        rest = use & (score < cutoff)
        records.append({
            "unit": unit,
            "n": int(use.sum()),
            "score_cutoff": float(cutoff),
            f"{outcome_name}_high": float(np.mean(outcome[high])),
            f"{outcome_name}_rest": float(np.mean(outcome[rest])),
            f"{outcome_name}_difference": float(np.mean(outcome[high]) - np.mean(outcome[rest])),
        })
    return pd.DataFrame(records)


def spatial_scatter(
    axis: plt.Axes,
    frame: pd.DataFrame,
    score: np.ndarray,
    *,
    x: str,
    y: str,
    background: str = "#D7DCE2",
    cmap: str = "inferno",
    title: str = "",
    focus: np.ndarray | None = None,
) -> None:
    score = np.asarray(score, float)
    focus = np.isfinite(score) if focus is None else np.asarray(focus, bool) & np.isfinite(score)
    axis.scatter(frame[x], frame[y], s=0.7, c=background, alpha=0.42, linewidths=0, rasterized=True)
    if focus.any():
        lo, hi = np.nanquantile(score[focus], [0.05, 0.95])
        axis.scatter(
            frame.loc[focus, x], frame.loc[focus, y], s=2.5,
            c=score[focus], cmap=cmap, vmin=lo, vmax=hi,
            alpha=0.88, linewidths=0, rasterized=True,
        )
    axis.set_title(title, fontsize=10, fontweight="bold")
    axis.set_aspect("equal", adjustable="box")
    axis.set(xticks=[], yticks=[])
    for spine in axis.spines.values():
        spine.set_visible(False)


def save_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

