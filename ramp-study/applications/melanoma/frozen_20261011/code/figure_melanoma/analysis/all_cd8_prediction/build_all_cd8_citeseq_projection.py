#!/usr/bin/env python3
"""Train on all spatial CD8 cells and project the state to all CITE-seq CD8 cells."""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
FIGURE_ROOT = HERE.parents[1]
FINAL_XU = FIGURE_ROOT.parent
ALIGNMENT_ROOT = HERE.parent / "alignment/output"

ALIGNMENT_META = ALIGNMENT_ROOT / "cell_metadata.csv.gz"
RAMP_PATH = ALIGNMENT_ROOT / "ramp_aligned.npy"
ALIGNMENT_MANIFEST = ALIGNMENT_ROOT / "manifest.json"
ANALYSIS_META = FIGURE_ROOT / "inputs/spatial_cell_metadata.csv.gz"
QUERY_META = (
    FIGURE_ROOT
    / "inputs/cohort_package/engelbrecht_citeseq/locked_metadata.csv.gz"
)

OUT = HERE / "citeseq_projection"
SEED = 20260829
CD8_LABELS = ("Tc.ae", "Tc.naive")
HELD_OUT_PROTEINS = ("PD1", "TIGIT", "CD103", "CD69", "CD39", "CD107A", "KLRG1", "CX3CR1", "CD62L")

os.environ.setdefault("NUMBA_CACHE_DIR", str(FIGURE_ROOT / ".numba-cache"))
os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(OUT / ".xdg-cache"))


def write_csv_gz(frame: pd.DataFrame, path: Path) -> None:
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(fileobj=raw_handle, mode="wb", mtime=0) as handle:
            frame.to_csv(handle, index=False)


def model() -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.5,
            class_weight="balanced",
            max_iter=5000,
            solver="liblinear",
            random_state=SEED,
        ),
    )


def patient_percentiles(values: np.ndarray, patients: np.ndarray) -> np.ndarray:
    result = np.empty(len(values), dtype=float)
    for patient in sorted(pd.unique(patients)):
        use = patients == patient
        ranks = pd.Series(values[use]).rank(method="average").to_numpy(float)
        result[use] = ranks / (int(use.sum()) + 1.0)
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    identity = pd.read_csv(ALIGNMENT_META, low_memory=False)
    analysis = pd.read_csv(ANALYSIS_META, low_memory=False)
    query = pd.read_csv(QUERY_META, low_memory=False)
    ramp = np.load(RAMP_PATH).astype(np.float64)
    if len(identity) != len(analysis) or len(identity) != len(ramp):
        raise AssertionError("Clean15 metadata and aligned array row counts differ")
    if not identity["cell_uid"].astype(str).equals(analysis["cell_uid_ordered"].astype(str)):
        raise AssertionError("Analysis metadata and clean15 alignment row order differ")

    source_mask = identity["dataset"].astype(str).eq("Moldoveanu IMC").to_numpy()
    query_mask = identity["dataset"].astype(str).eq("Engelbrecht CITE-seq").to_numpy()
    if int(query_mask.sum()) != len(query):
        raise AssertionError("Full CITE-seq metadata row count differs from clean15 alignment")
    query_identity = identity.loc[query_mask, "cell_uid"].astype(str).reset_index(drop=True)
    if not query_identity.equals(query["source_cell_id"].astype(str).reset_index(drop=True)):
        raise AssertionError("CITE-seq metadata and clean15 alignment row order differ")

    source_all_cd8 = (
        source_mask
        & analysis["Cluster.v2"].astype(str).isin(CD8_LABELS).to_numpy()
    )
    source_cells = analysis.loc[source_all_cd8]
    if source_cells[["prob.Ki67", "distance_to_nearest_tumor"]].isna().any().any():
        raise AssertionError("Every spatial all-CD8 cell must have Ki67 and tumor distance")
    y = (
        source_cells["Cluster.v2"].astype(str).eq("Tc.ae").to_numpy()
        & (source_cells["prob.Ki67"].to_numpy(float) >= 0.5)
        & (source_cells["distance_to_nearest_tumor"].to_numpy(float) <= 30.0)
    ).astype(int)
    x_source = ramp[source_all_cd8]
    if int(y.sum()) != 478 or len(y) != 5649:
        raise AssertionError("Unexpected fixed all-CD8 training population")

    fitted = model()
    fitted.fit(x_source, y)

    query_values = ramp[query_mask]
    query_cd8 = query["Broad_cell.type.1"].astype(str).eq("CD8_T").to_numpy()
    x_query_cd8 = query_values[query_cd8]
    score = fitted.predict_proba(x_query_cd8)[:, 1]
    query_cd8_meta = query.loc[query_cd8].copy().reset_index(drop=True)
    patients = query_cd8_meta["Patient_ID"].astype(str).to_numpy()
    percentile = patient_percentiles(score, patients)

    # UMAP is a display embedding of the full clean15-aligned CITE-seq CD8 pool.
    import umap

    standardized = StandardScaler().fit_transform(x_query_cd8).astype(np.float32)
    coordinates = umap.UMAP(
        n_neighbors=30,
        min_dist=0.20,
        n_components=2,
        metric="euclidean",
        random_state=SEED,
        transform_seed=SEED,
        n_jobs=1,
    ).fit_transform(standardized).astype(np.float32)

    output_columns = [
        "source_cell_id",
        "Sample_ID",
        "Patient_ID",
        "Broad_cell.type.1",
        "clust.name",
        "cell_type",
    ] + [protein for protein in HELD_OUT_PROTEINS if protein in query_cd8_meta.columns]
    output = query_cd8_meta[output_columns].copy()
    output["umap_1"] = coordinates[:, 0]
    output["umap_2"] = coordinates[:, 1]
    output["all_cd8_spatial_state_score"] = score
    output["within_patient_score_percentile"] = percentile
    write_csv_gz(output, OUT / "engelbrecht_all_cd8_projection.csv.gz")
    np.save(OUT / "engelbrecht_all_cd8_umap.npy", coordinates)
    np.save(OUT / "engelbrecht_all_cd8_score.npy", score.astype(np.float32))

    classifier = fitted.named_steps["logisticregression"]
    markers = list(json.loads(ALIGNMENT_MANIFEST.read_text())["shared_markers"])
    coefficients = pd.DataFrame({"marker": markers, "coefficient": classifier.coef_[0]})
    coefficients.to_csv(OUT / "classifier_coefficients.csv", index=False)

    manifest = {
        "source": "Moldoveanu IMC",
        "query": "Engelbrecht CITE-seq",
        "source_prediction_pool": "all provider CD8 cells: Tc.ae + Tc.naive",
        "source_ground_truth": "Tc.ae AND prob.Ki67 >= 0.5 AND nearest melanoma centroid <= 30 um",
        "source_training_cells": int(len(y)),
        "source_training_targets": int(y.sum()),
        "source_training_patients": int(analysis.loc[source_all_cd8, "donor"].astype(str).nunique()),
        "query_prediction_pool": "Broad_cell.type.1 == CD8_T",
        "query_cd8_cells_scored": int(len(output)),
        "query_patients": int(output["Patient_ID"].astype(str).nunique()),
        "representation": "15 shared proteins after frozen clean15 RAMP alignment",
        "shared_markers": markers,
        "classifier": "StandardScaler + class-balanced logistic regression (C=0.5)",
        "query_label_use": "defines the deployment population only; no Tc.ae label is required",
        "umap": {
            "input": "standardized clean15 RAMP-aligned values in all CITE-seq CD8_T cells",
            "n_neighbors": 30,
            "min_dist": 0.20,
            "metric": "euclidean",
            "seed": SEED,
        },
        "outputs": {
            "table": str(OUT / "engelbrecht_all_cd8_projection.csv.gz"),
            "umap": str(OUT / "engelbrecht_all_cd8_umap.npy"),
            "score": str(OUT / "engelbrecht_all_cd8_score.npy"),
            "coefficients": str(OUT / "classifier_coefficients.csv"),
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
