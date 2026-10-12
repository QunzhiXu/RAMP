#!/usr/bin/env python3
"""Compare wrong-CD19 and predicted-CD19 ARI with the benchmark protocol."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
STORY = HERE.parent
FINAL_XU = HERE.parents[1]
DATA = HERE / "data"
CACHE = HERE / ".cache" / "ari"
for folder in [DATA, CACHE]:
    folder.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("RAMP_EVALUATION_CACHE", str(CACHE))

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


EVALUATOR_PATH = (
    FINAL_XU
    / "benchmark_results"
    / "comparison_evaluation"
    / "scripts"
    / "evaluate_benchmark.py"
)
SPEC = importlib.util.spec_from_file_location("ramp_benchmark_evaluator", EVALUATOR_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load benchmark evaluator: {EVALUATOR_PATH}")
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)


CONDITIONS = {
    "wrong_observed_cd19": (
        STORY
        / "alignment"
        / "output"
        / "with_wrong_observed_cd19"
        / "ramp_aligned.npy"
    ),
    "predicted_cd19": (
        STORY
        / "alignment"
        / "output"
        / "with_predicted_cd19"
        / "ramp_aligned.npy"
    ),
}
METADATA_PATH = STORY / "alignment" / "output" / "cell_metadata.csv.gz"
CELLS_PER_DATASET = 5_000


def evaluable_labels(celltype: np.ndarray, dataset: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    joint_flag = np.asarray(
        [
            not bool(EVALUATOR.INVALID_LABEL_PATTERN.search(label))
            for label in celltype
        ],
        dtype=bool,
    )
    frame = pd.DataFrame(
        {"celltype": celltype[joint_flag], "dataset": dataset[joint_flag]}
    )
    counts = frame["celltype"].value_counts()
    present = frame.groupby("celltype")["dataset"].nunique()
    levels = present[(present >= 2) & (counts >= 30)].index.astype(str).to_numpy()
    return joint_flag & np.isin(celltype, levels), levels


def main() -> None:
    metadata = pd.read_csv(METADATA_PATH, low_memory=False)
    representations = {
        condition: EVALUATOR.standardized(np.load(path), condition)
        for condition, path in CONDITIONS.items()
    }
    records: list[dict[str, object]] = []
    for seed in EVALUATOR.SEEDS:
        rows = EVALUATOR.sample_rows(metadata, seed, CELLS_PER_DATASET)
        local = metadata.iloc[rows].reset_index(drop=True)
        celltype = local["celltype"].fillna("Unannotated").astype(str).to_numpy()
        dataset = local["dataset"].astype(str).to_numpy()
        valid_joint, joint_levels = evaluable_labels(celltype, dataset)
        target = max(2, len(joint_levels))

        for condition, representation in representations.items():
            index, distance = EVALUATOR.knn(representation[rows], 90)
            graph_sparse = EVALUATOR.graph_from_knn(index, distance, 15)
            graph = EVALUATOR.igraph_from_sparse(graph_sparse)
            for algorithm in ["leiden", "louvain"]:
                clusters, resolution, count, valid = EVALUATOR.target_k_partition(
                    graph, target, seed, algorithm
                )
                records.append(
                    {
                        "condition": condition,
                        "sampling_seed": seed,
                        "algorithm": algorithm,
                        "cells": len(rows),
                        "joint_celltypes": len(joint_levels),
                        "clusters": count,
                        "resolution": resolution,
                        "cluster_count_valid": valid,
                        "celltype_ari": (
                            float(adjusted_rand_score(celltype[valid_joint], clusters[valid_joint]))
                            if valid
                            else np.nan
                        ),
                        "dataset_ari": (
                            max(0.0, float(adjusted_rand_score(dataset, clusters)))
                            if valid
                            else np.nan
                        ),
                    }
                )
        print(f"completed seed {seed}", flush=True)

    replicates = pd.DataFrame(records)
    replicates.to_csv(DATA / "wrong_vs_predicted_ari_replicates.csv", index=False)
    summary = (
        replicates.groupby(["condition", "algorithm"], as_index=False)
        .agg(
            celltype_ari_mean=("celltype_ari", "mean"),
            celltype_ari_sd=("celltype_ari", "std"),
            dataset_ari_mean=("dataset_ari", "mean"),
            dataset_ari_sd=("dataset_ari", "std"),
            valid_resamples=("cluster_count_valid", "sum"),
        )
    )
    summary.to_csv(DATA / "wrong_vs_predicted_ari_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
