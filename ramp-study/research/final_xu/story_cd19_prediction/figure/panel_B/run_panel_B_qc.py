#!/usr/bin/env python3
"""Recompute label-free marker QC on the frozen BM story sample.

Each target marker is hidden in turn.  ExtraTrees models trained on either of
the two other platforms predict the hidden marker from within-donor ranks of
the remaining markers.  Prediction disagreement and raw-distribution anomaly
diagnostics are converted to within-donor percentiles and aggregated into the
QC alarm score.  Cell labels and spatial coordinates are never loaded.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesRegressor


HERE = Path(__file__).resolve().parent
STORY = HERE.parents[1]
SAMPLE = STORY / "sampled_data"
ALGORITHM = STORY.parent / "algorithm"
sys.path.insert(0, str(ALGORITHM))
from ramp import ramp_ver2 as ramp  # noqa: E402


OUT = HERE / "data"
SEED = 20260814
EVAL_CAP_PER_UNIT = 2_500
TRAIN_CAP_PER_SOURCE = 4_000
N_TREES = 140
TOP_FRACTION = 0.10
MARKERS = [
    "CD117", "CD11B", "CD11C", "CD123", "CD14", "CD3", "CD33", "CD34",
    "CD38", "CD4", "CD45", "CD45RA", "CD61", "CD8", "CD19",
]
DATASETS = ["BM CODEX", "BM CyTOF", "Triana AbSeq"]
SHAPE_FEATURES = [
    "raw_upper_range_ratio", "raw_upper_tail_ratio", "raw_kmeans_separation",
    "raw_max_pile_fraction", "raw_tie_fraction",
]
REPLICATE_SCALE_FEATURES = ["raw_std", "raw_q99_q01", "raw_q99_q50"]


def stable_rng(*parts: object) -> np.random.Generator:
    payload = "|".join(map(str, (SEED, *parts))).encode()
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") % (2**32 - 1)
    return np.random.default_rng(value)


def load_data() -> tuple[np.ndarray, pd.DataFrame]:
    folders = ["bm_codex", "bm_cytof", "triana_abseq"]
    matrices: list[np.ndarray] = []
    metadata: list[pd.DataFrame] = []
    for folder in folders:
        frame = pd.read_csv(SAMPLE / folder / "source_input_15.csv.gz")
        missing = [marker for marker in MARKERS if marker not in frame]
        if missing:
            raise ValueError(f"{folder} is missing {missing}")
        matrices.append(frame[MARKERS].to_numpy(np.float64))
        metadata.append(frame[["cell_id", "dataset", "donor", "unit_key"]].copy())
    raw = np.vstack(matrices)
    md = pd.concat(metadata, ignore_index=True)
    if len(raw) != len(md):
        raise RuntimeError("Matrix/metadata length mismatch")
    return raw, md


def deterministic_sample(rows: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    rows = np.asarray(rows, dtype=int)
    return np.sort(rows if len(rows) <= cap else rng.choice(rows, cap, replace=False))


def donor_balanced_local_indices(metadata: pd.DataFrame, dataset: str, cap: int) -> np.ndarray:
    groups = list(metadata.loc[metadata.dataset.eq(dataset)].groupby("unit_key", sort=True))
    quota = cap // len(groups)
    selected: list[np.ndarray] = []
    leftovers: list[np.ndarray] = []
    for gi, (_, frame) in enumerate(groups):
        rows = stable_rng("train", dataset, gi).permutation(frame.index.to_numpy(int))
        selected.append(rows[: min(quota, len(rows))])
        leftovers.append(rows[min(quota, len(rows)):])
    answer = np.concatenate(selected)
    remaining = cap - len(answer)
    if remaining > 0:
        pool = np.concatenate(leftovers)
        answer = np.concatenate([
            answer,
            stable_rng("train_remainder", dataset).permutation(pool)[:remaining],
        ])
    return np.sort(answer)


def within_unit_ranks(values: np.ndarray, metadata: pd.DataFrame) -> np.ndarray:
    answer = np.empty_like(values, dtype=np.float32)
    for _, frame in metadata.groupby("unit_key", sort=True):
        rows = frame.index.to_numpy(int)
        for j in range(values.shape[1]):
            answer[rows, j] = ((rankdata(values[rows, j], method="average") - 0.5) / len(rows)).astype(np.float32)
    return answer


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.0
    value = float(spearmanr(x, y).statistic)
    return value if np.isfinite(value) else 0.0


def qc_metrics(observed: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    observed_rank = (rankdata(observed, method="average") - 0.5) / len(observed)
    predicted_rank = (rankdata(prediction, method="average") - 0.5) / len(prediction)
    high = predicted_rank >= 1.0 - TOP_FRACTION
    low = predicted_rank <= TOP_FRACTION
    high_deficit = float(prediction[high].mean() - observed[high].mean())
    low_excess = float(observed[low].mean() - prediction[low].mean())
    return {
        "rho": safe_spearman(prediction, observed),
        "mae": float(np.mean(np.abs(prediction - observed))),
        "tail_mismatch": max(0.0, high_deficit, low_excess),
        "top_overlap": float(np.mean(observed_rank[high] >= 1.0 - TOP_FRACTION)),
    }


def raw_shape_features(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    q01, q50, q75, q90, q99 = np.quantile(x, [0.01, 0.50, 0.75, 0.90, 0.99])
    iqr_upper = max(float(q75 - q50), 1e-9)
    shoulder = max(float(q90 - q50), 1e-9)
    fit = KMeans(n_clusters=2, n_init=20, random_state=42).fit(x[:, None])
    centers = np.sort(fit.cluster_centers_.ravel())
    scale = max(float(np.std(x, ddof=0)), 1e-9)
    tolerance = max(1e-12, 1e-9 * max(1.0, abs(float(np.max(x)))))
    return {
        "raw_upper_range_ratio": float((q99 - q50) / iqr_upper),
        "raw_upper_tail_ratio": float((q99 - q90) / shoulder),
        "raw_kmeans_separation": float((centers[1] - centers[0]) / scale),
        "raw_max_pile_fraction": float(np.mean(np.abs(x - np.max(x)) <= tolerance)),
        "raw_tie_fraction": float(1.0 - len(np.unique(x)) / len(x)),
        "raw_std": float(np.std(x, ddof=0)),
        "raw_q99_q01": float(q99 - q01),
        "raw_q99_q50": float(q99 - q50),
    }


def compute_shape_table(raw: np.ndarray, metadata: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for unit, frame in metadata.groupby("unit_key", sort=True):
        idx = frame.index.to_numpy(int)
        dataset = str(frame.dataset.iloc[0])
        for j, marker in enumerate(MARKERS):
            records.append({"dataset": dataset, "unit_key": unit, "marker": marker, **raw_shape_features(raw[idx, j])})
    table = pd.DataFrame(records)
    table["raw_shape_anomaly"] = 0.0
    table["replicate_shape_anomaly"] = 0.0
    for marker, marker_frame in table.groupby("marker", sort=False):
        marker_rows = marker_frame.index
        feature_scores = []
        for feature in SHAPE_FEATURES:
            values = marker_frame[feature].to_numpy(float)
            center = float(np.median(values))
            mad = max(float(np.median(np.abs(values - center))), 1e-6)
            feature_scores.append(np.abs(0.6745 * (values - center) / mad))
        feature_scores = np.sort(np.vstack(feature_scores).T, axis=1)
        table.loc[marker_rows, "raw_shape_anomaly"] = feature_scores[:, -2:].mean(axis=1)
        for idx in marker_rows:
            row = table.loc[idx]
            refs = table[
                table.dataset.eq(row.dataset) & table.marker.eq(marker)
                & ~table.unit_key.eq(row.unit_key)
            ]
            if refs.empty:
                continue
            values = []
            for feature in REPLICATE_SCALE_FEATURES:
                target = max(abs(float(row[feature])), 1e-9)
                reference = max(float(np.median(np.abs(refs[feature].to_numpy(float)))), 1e-9)
                values.append(abs(np.log(target / reference)))
            table.loc[idx, "replicate_shape_anomaly"] = float(np.sort(values)[-2:].mean())
    return table


def add_alarm_scores(frame: pd.DataFrame) -> pd.DataFrame:
    answer = frame.copy()
    anomaly_columns = {
        "rho_anomaly": -answer["rho"],
        "mae_anomaly": answer["mae"],
        "tail_anomaly": answer["tail_mismatch"],
        "overlap_anomaly": -answer["top_overlap"],
        "shape_anomaly": answer["raw_shape_anomaly"],
        "replicate_anomaly": answer["replicate_shape_anomaly"],
    }
    percentile_cols: list[str] = []
    for name, values in anomaly_columns.items():
        answer[name] = values
        column = name + "_percentile"
        answer[column] = answer.groupby("unit_key")[name].rank(pct=True, method="average")
        percentile_cols.append(column)
    percentiles = np.sort(answer[percentile_cols].to_numpy(float), axis=1)
    answer["qc_alarm"] = percentiles[:, -2:].mean(axis=1)
    answer["qc_rank_within_unit"] = (
        answer.groupby("unit_key")["qc_alarm"].rank(ascending=False, method="min").astype(int)
    )
    return answer


def fit_predictions(processed: np.ndarray, metadata: pd.DataFrame) -> pd.DataFrame:
    sample_global: list[np.ndarray] = []
    for unit in sorted(metadata.unit_key.unique()):
        rows = metadata.index[metadata.unit_key.eq(unit)].to_numpy(int)
        sample_global.append(deterministic_sample(rows, EVAL_CAP_PER_UNIT, stable_rng("eval", unit)))
    sampled_idx = np.sort(np.concatenate(sample_global))
    values = np.asarray(processed[sampled_idx], dtype=np.float32)
    sampled_md = metadata.loc[sampled_idx].copy().reset_index(drop=True)
    ranks = within_unit_ranks(values, sampled_md)
    platform = sampled_md.dataset.astype(str).to_numpy()
    records: list[dict[str, object]] = []

    for marker_index, marker in enumerate(MARKERS):
        print(f"LOMO {marker_index + 1:02d}/{len(MARKERS)}: {marker}", flush=True)
        keep = np.arange(len(MARKERS)) != marker_index
        for target_dataset in DATASETS:
            query_rows = np.flatnonzero(platform == target_dataset)
            predictions: list[np.ndarray] = []
            for source_dataset in [x for x in DATASETS if x != target_dataset]:
                train_rows = donor_balanced_local_indices(sampled_md, source_dataset, TRAIN_CAP_PER_SOURCE)
                model = ExtraTreesRegressor(
                    n_estimators=N_TREES, min_samples_leaf=12, max_features=0.8,
                    n_jobs=-1,
                    random_state=int(stable_rng("forest", target_dataset, source_dataset, marker).integers(0, 2**31 - 1)),
                )
                model.fit(ranks[train_rows][:, keep], values[train_rows, marker_index])
                predictions.append(model.predict(ranks[query_rows][:, keep]).astype(np.float32))
            prediction = 0.5 * (predictions[0] + predictions[1])
            query_md = sampled_md.iloc[query_rows]
            for unit, frame in query_md.groupby("unit_key", sort=True):
                local_rows = frame.index.to_numpy(int)
                positions = np.searchsorted(query_rows, local_rows)
                records.append({
                    "dataset": target_dataset,
                    "unit_key": unit,
                    "marker": marker,
                    "n_cells": len(frame),
                    **qc_metrics(values[local_rows, marker_index], prediction[positions]),
                })
    return pd.DataFrame(records)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if ramp.__version__ != "2.3.0":
        raise AssertionError(f"Expected RAMP 2.3.0, imported {ramp.__version__}")
    expected_ramp = (ALGORITHM / "ramp" / "ramp_ver2.py").resolve()
    if Path(ramp.__file__).resolve() != expected_ramp:
        raise AssertionError(f"Unexpected RAMP source: {ramp.__file__}")
    raw, metadata = load_data()
    processed = ramp.kmeanslow_global(
        raw,
        metadata.unit_key.astype(str).to_numpy(),
        metadata.dataset.astype(str).to_numpy(),
        MARKERS,
    )
    shape = compute_shape_table(raw, metadata)
    observed = fit_predictions(processed, metadata)
    observed = observed.merge(
        shape[["unit_key", "marker", "raw_shape_anomaly", "replicate_shape_anomaly"]],
        on=["unit_key", "marker"], how="left", validate="one_to_one",
    )
    observed = add_alarm_scores(observed)
    observed.to_csv(OUT / "all_marker_qc_scores.csv", index=False)

    codex = observed[observed.dataset.eq("BM CODEX")].copy()
    summary = (
        codex.groupby("marker", as_index=False)
        .agg(
            mean_qc_alarm=("qc_alarm", "mean"),
            median_rank=("qc_rank_within_unit", "median"),
            top1_donors=("qc_rank_within_unit", lambda x: int(np.sum(np.asarray(x) == 1))),
            top3_donors=("qc_rank_within_unit", lambda x: int(np.sum(np.asarray(x) <= 3))),
        )
        .sort_values(["top1_donors", "top3_donors", "mean_qc_alarm"], ascending=False)
    )
    summary.to_csv(OUT / "codex_marker_qc_summary.csv", index=False)
    top3 = summary.marker.head(3).tolist()
    codex[codex.marker.isin(top3)].to_csv(OUT / "codex_top3_marker_donor_scores.csv", index=False)
    manifest = {
        "sample": "final_xu/story_cd19_prediction/sampled_data",
        "ramp_version": ramp.__version__,
        "ramp_source": str(Path(ramp.__file__).resolve()),
        "codex_sampling": "12 donors x 2,500 cells",
        "cell_labels_used": False,
        "spatial_coordinates_used": False,
        "target_marker_hidden_from_each_predictor": True,
        "top3_markers": top3,
        "cd19_top1_donors": int(summary.loc[summary.marker.eq("CD19"), "top1_donors"].iloc[0]),
        "qc_score": "mean of the two largest within-donor anomaly percentiles",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
