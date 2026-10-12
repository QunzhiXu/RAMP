#!/usr/bin/env python3
"""Project a locked TIGIT/CD39 Rahim CD8 state back to Haist CODEX.

This phase is outcome blind.  It reads the existing PT70 RAMP 2.3.0
alignment, constructs the source state as the unweighted mean of within-Rahim-
donor percentile ranks of strict-no-LAG3 paper-CLR TIGIT and CD39, and transfers
that fixed state to every primary-tumour CD8 cell in the 70 selected Haist
patients.  It then constructs spatial-niche features without opening survival.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import binomtest, rankdata, spearmanr, wilcoxon
from scipy.spatial import cKDTree


HERE = Path(__file__).resolve().parent
FIGURE_HNSCC = HERE.parents[1]
PT70 = HERE.parent / "pt70_full_90488_umap"
OUT = HERE / "output"
FORWARD_SCRIPT = PT70 / "04_project_niche_score_to_rahim.py"
ADT_H5AD = (
    FIGURE_HNSCC
    / "sampled_data/rahim_citeseq/complete/GSE212797_adata_ADT.h5ad"
)
EXPOSURE = (
    FIGURE_HNSCC
    / "survival_motivation/output/pt_cd8_cell_spatial_niche_scores.csv.gz"
)
SCREEN = (
    PT70
    / "marker_rerun/screen_no_lag3/marker_screen_all_107_all_variants.csv"
)
PROJECTED_NICHE = (
    PT70
    / "marker_rerun/projection/rahim_tumor_cd8_projected_scores.csv.gz"
)

PROGRAM_MARKERS = ("TIGIT--VSTM3", "CD39")
PROGRAM_LABEL = "TIGIT/CD39"
HOLDOUT_MARKER = "CD223--LAG-3"
TOP_FRACTION = 0.05
EXPECTED_CANDIDATES = 107
K_TRANSFER = 30


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode() if isinstance(value, bytes) else str(value) for value in values],
        dtype=object,
    )


def percentile_midrank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    return (rankdata(values, method="average") - 0.5) / len(values)


def write_csv_gz(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="", fileobj=raw_handle, mode="wb", compresslevel=6, mtime=0
        ) as handle:
            frame.to_csv(handle, index=False)
    os.replace(temporary, path)


def lock_high_confidence_pair() -> dict:
    screen = pd.read_csv(SCREEN)
    if len(screen) != EXPECTED_CANDIDATES:
        raise AssertionError(f"Expected {EXPECTED_CANDIDATES} screened ADTs")
    cutoff = EXPECTED_CANDIDATES * TOP_FRACTION
    high_confidence = screen.loc[
        screen.passes_frozen_rule_pre_redundancy.astype(bool)
        & screen.selected_after_redundancy_pruning.astype(bool)
        & screen.robust_median_rank_primary_variants.le(cutoff)
    ].sort_values(
        ["robust_median_rank_primary_variants", "paper_clr__rank_107"],
        kind="stable",
    )
    observed = tuple(high_confidence.marker_raw.astype(str))
    if observed != PROGRAM_MARKERS:
        raise AssertionError(
            f"Top-5% robust program changed: expected {PROGRAM_MARKERS}, observed {observed}"
        )
    return {
        "selection_rule": (
            "pass the frozen donor/technical/delete-one-donor sign/redundancy gates, "
            "then retain robust median ranks in the top 5% of all 107 candidates"
        ),
        "delete_one_donor_definition": (
            "association is recomputed in each remaining two-donor subset; this is "
            "a sign-robustness check, not train-on-two/predict-the-held-out-donor CV"
        ),
        "top_fraction": TOP_FRACTION,
        "numeric_rank_cutoff": cutoff,
        "markers_raw": list(PROGRAM_MARKERS),
        "marker_rows": high_confidence[
            [
                "marker_raw",
                "marker",
                "robust_median_rank_primary_variants",
                "paper_clr__rank_107",
                "isotype_adjusted__rank_107",
                "global_adt_loo_adjusted__rank_107",
                "robust_min_lodo_all_primary_variants",
            ]
        ].to_dict(orient="records"),
        "screen_sha256": sha256(SCREEN),
    }


def read_rahim_pair(
    cell_ids: np.ndarray, donors: np.ndarray
) -> tuple[pd.DataFrame, np.ndarray]:
    with h5py.File(ADT_H5AD, "r") as handle:
        ids = decode(handle["obs/_index"][:])
        markers = decode(handle["var/_index"][:])
        group = handle["X"]
        matrix = sparse.csr_matrix(
            (group["data"][:], group["indices"][:], group["indptr"][:]),
            shape=(len(group["indptr"]) - 1, len(markers)),
        )
    row_lookup = pd.Series(np.arange(len(ids), dtype=int), index=ids)
    if row_lookup.index.has_duplicates:
        raise AssertionError("Rahim ADT cell IDs are not unique")
    rows = row_lookup.reindex(cell_ids).to_numpy()
    if pd.isna(rows).any():
        raise AssertionError("Aligned Rahim tumour-CD8 cells are missing from ADT H5AD")
    rows = rows.astype(int)

    nonisotype = np.asarray(["isotype" not in marker.lower() for marker in markers])
    background = nonisotype & (markers != HOLDOUT_MARKER)
    if int(background.sum()) != 129:
        raise AssertionError("Strict no-LAG3 CLR background is no longer 129 ADTs")
    marker_lookup = {marker: index for index, marker in enumerate(markers)}
    missing = sorted(set(PROGRAM_MARKERS) - set(marker_lookup))
    if missing:
        raise KeyError(f"Rahim ADT is missing locked markers: {missing}")

    counts = matrix[rows].toarray().astype(np.float64, copy=False)
    if np.any(counts < 0) or not np.allclose(counts, np.round(counts)):
        raise AssertionError("Rahim ADT X is not the expected raw-count matrix")
    logged = np.log1p(counts)
    center = logged[:, background].mean(axis=1)
    ranks = np.empty((len(rows), len(PROGRAM_MARKERS)), np.float64)
    clr_values = np.empty_like(ranks)
    for marker_index, marker in enumerate(PROGRAM_MARKERS):
        clr = logged[:, marker_lookup[marker]] - center
        clr_values[:, marker_index] = clr
        for donor in sorted(pd.unique(donors), key=str):
            local = np.flatnonzero(donors == donor)
            ranks[local, marker_index] = percentile_midrank(clr[local])
    score = ranks.mean(axis=1)
    source = pd.DataFrame(
        {
            "cell_uid": cell_ids,
            "donor": donors,
            "tigit_paper_clr": clr_values[:, 0],
            "cd39_paper_clr": clr_values[:, 1],
            "tigit_within_donor_rank": ranks[:, 0],
            "cd39_within_donor_rank": ranks[:, 1],
            "tigit_cd39_score": score,
        }
    )
    return source, score


def predict_by_source_donor(
    source_x: np.ndarray,
    target_x: np.ndarray,
    donors: np.ndarray,
    response: np.ndarray,
    weighted_predict_fn,
) -> tuple[np.ndarray, np.ndarray]:
    columns = []
    donor_order = sorted(pd.unique(donors), key=str)
    for donor in donor_order:
        rows = np.flatnonzero(donors == donor)
        tree = cKDTree(source_x[rows])
        columns.append(weighted_predict_fn(tree, response[rows], target_x))
    donor_predictions = np.column_stack(columns)
    return np.median(donor_predictions, axis=1), donor_predictions


def build_patient_features(cells: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for patient, frame in cells.groupby("patient", sort=True):
        predicted = frame.predicted_tigit_cd39.to_numpy(float)
        state_rank = percentile_midrank(predicted)
        spatial = frame.spatial_y.to_numpy(float)
        spatial_rank = percentile_midrank(spatial)
        niche = spatial > 0
        outside = ~niche
        if min(int(niche.sum()), int(outside.sum())) == 0:
            raise AssertionError(f"Patient {patient} has an empty niche comparison")
        high = state_rank >= 0.75
        low = state_rank <= 0.25
        rows.append(
            {
                "patient": str(patient),
                "pt_cd8_cells": int(len(frame)),
                "niche_cells": int(niche.sum()),
                "outside_cells": int(outside.sum()),
                "niche_positive_fraction": float(niche.mean()),
                "niche_mean_pair_rank": float(state_rank[niche].mean()),
                "outside_mean_pair_rank": float(state_rank[outside].mean()),
                "niche_minus_outside_mean_pair_rank": float(
                    state_rank[niche].mean() - state_rank[outside].mean()
                ),
                "niche_minus_outside_mean_pair_raw": float(
                    predicted[niche].mean() - predicted[outside].mean()
                ),
                "pair_spatial_spearman": float(spearmanr(predicted, spatial).statistic),
                "high_minus_low_mean_spatial_rank": float(
                    spatial_rank[high].mean() - spatial_rank[low].mean()
                ),
                # Pre-existing joint-location summary used by the historical
                # return analysis: how adverse is the spatial environment of
                # the patient's top-quartile returned molecular-state cells?
                "high_pair_mean_spatial_y": float(spatial[high].mean()),
                "low_pair_mean_spatial_y": float(spatial[low].mean()),
                "high_pair_mean_spatial_rank": float(spatial_rank[high].mean()),
                "high_pair_niche_fraction": float(niche[high].mean()),
                "overall_mean_predicted_pair": float(predicted.mean()),
                "overall_median_predicted_pair": float(np.median(predicted)),
                "patient_spatial_niche_score": float(spatial.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("patient", key=lambda x: x.astype(int))


def summarize_enrichment(features: pd.DataFrame) -> dict:
    delta = features.niche_minus_outside_mean_pair_raw.to_numpy(float)
    rank_delta = features.niche_minus_outside_mean_pair_rank.to_numpy(float)
    rho = features.pair_spatial_spearman.to_numpy(float)
    wilcox = wilcoxon(delta, alternative="greater", zero_method="wilcox")
    positives = int(np.sum(delta > 0))
    sign = binomtest(positives, len(delta), p=0.5, alternative="greater")
    rank_wilcox = wilcoxon(
        rank_delta, alternative="greater", zero_method="wilcox"
    )
    rank_positives = int(np.sum(rank_delta > 0))
    rank_sign = binomtest(
        rank_positives, len(rank_delta), p=0.5, alternative="greater"
    )
    return {
        "patients": int(len(features)),
        "primary_feature": "niche_minus_outside_mean_pair_raw",
        "primary_feature_definition": (
            "within each patient, mean predicted TIGIT/CD39 score among PT-CD8 "
            "cells with ROI-adjusted CAF/macrophage exposure > 0 minus the "
            "corresponding mean outside that niche"
        ),
        "median_niche_minus_outside_raw": float(np.median(delta)),
        "positive_patients": positives,
        "one_sided_wilcoxon_p": float(wilcox.pvalue),
        "one_sided_exact_sign_p": float(sign.pvalue),
        "rank_sensitivity": {
            "median_niche_minus_outside_rank": float(np.median(rank_delta)),
            "positive_patients": rank_positives,
            "one_sided_wilcoxon_p": float(rank_wilcox.pvalue),
            "one_sided_exact_sign_p": float(rank_sign.pvalue),
        },
        "median_pair_spatial_spearman": float(np.median(rho)),
        "positive_spearman_patients": int(np.sum(rho > 0)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lock = lock_high_confidence_pair()
    pt = import_file("pt70_forward_helpers", FORWARD_SCRIPT)

    raw = np.load(PT70 / "input/raw_shared_markers.float64.npy", mmap_mode="r")
    aligned = np.load(PT70 / "alignment/aligned_ramp_ver2.float32.npy", mmap_mode="r")
    metadata = pd.read_csv(
        PT70 / "input/cell_metadata.csv.gz",
        dtype={"cell_uid": str, "dataset": str, "donor": str, "unit_key": str},
        low_memory=False,
    )
    selected = set(
        pd.read_csv(
            PT70 / "input/selected_70_patients.csv", usecols=["CaseID"]
        ).CaseID.astype(str)
    )
    curves = pt.build_haist_curves(raw, aligned, metadata)
    haist, haist_aligned = pt.load_full_haist_pt_cd8(selected, curves)

    rahim_rows = np.flatnonzero(
        metadata.dataset.eq("GSE212797_ADT")
        & metadata.tissue.astype(str).eq("tumor")
        & metadata.provider_cell_type.astype(str).eq("CD8 T cell")
    )
    rahim_meta = metadata.iloc[rahim_rows].copy().reset_index(drop=True)
    rahim_ids = rahim_meta.cell_uid.str.removeprefix("RAHIM|").to_numpy(str)
    rahim_donors = rahim_meta.donor.to_numpy(str)
    rahim_aligned = np.asarray(aligned[rahim_rows], np.float64)
    if len(rahim_meta) != 4_788:
        raise AssertionError("Expected 4,788 Rahim tumour-CD8 cells")

    source, source_score = read_rahim_pair(rahim_ids, rahim_donors)
    projected = pd.read_csv(
        PROJECTED_NICHE, dtype={"cell_uid": str, "donor": str}
    )[["cell_uid", "donor", "projected_roi_adjusted_exposure"]]
    source = source.merge(projected, on=["cell_uid", "donor"], validate="one_to_one")
    source_donor_results = []
    for donor, frame in source.groupby("donor", sort=True):
        source_donor_results.append(
            {
                "donor": str(donor),
                "cells": int(len(frame)),
                "spearman_pair_vs_projected_niche": float(
                    spearmanr(
                        frame.tigit_cd39_score,
                        frame.projected_roi_adjusted_exposure,
                    ).statistic
                ),
            }
        )
    source_donor_results = pd.DataFrame(source_donor_results)

    haist_patients = haist.Pat_ID.to_numpy(str)
    mean, scale = pt.equal_donor_patient_scaler(
        rahim_aligned, rahim_donors, haist_aligned, haist_patients
    )
    source_z = (rahim_aligned - mean) / scale
    target_z = (haist_aligned - mean) / scale
    predicted, donor_predictions = predict_by_source_donor(
        source_z, target_z, rahim_donors, source_score, pt.weighted_predict
    )

    exposure = pd.read_csv(
        EXPOSURE,
        usecols=[
            "UID",
            "Pat_ID",
            "unique_region",
            "x",
            "y",
            "roi_adjusted_caf_macrophage_score",
            "in_caf_macrophage_rich_niche",
        ],
        dtype={"UID": str, "Pat_ID": str},
    ).rename(
        columns={
            "UID": "cell_uid",
            "Pat_ID": "patient",
            "roi_adjusted_caf_macrophage_score": "spatial_y",
        }
    )
    target = haist[["UID", "Pat_ID"]].rename(
        columns={"UID": "cell_uid", "Pat_ID": "patient"}
    )
    target["patient"] = target.patient.astype(str)
    target = target.merge(
        exposure,
        on=["cell_uid", "patient"],
        how="left",
        validate="one_to_one",
    )
    if target.spatial_y.isna().any() or len(target) != 38_194:
        raise AssertionError("The 38,194 target PT-CD8 cells did not join exactly")
    target["predicted_tigit_cd39"] = predicted
    donor_order = sorted(pd.unique(rahim_donors), key=str)
    for column, donor in enumerate(donor_order):
        target[f"predicted_from_rahim_{donor}"] = donor_predictions[:, column]
    target["predicted_pair_rank_within_patient"] = target.groupby(
        "patient", sort=False
    ).predicted_tigit_cd39.transform(percentile_midrank)
    target["spatial_y_rank_within_patient"] = target.groupby(
        "patient", sort=False
    ).spatial_y.transform(percentile_midrank)

    features = build_patient_features(target)
    enrichment = summarize_enrichment(features)
    source_path = OUT / "rahim_tumor_cd8_locked_tigit_cd39.csv.gz"
    target_path = OUT / "haist70_pt_cd8_tigit_cd39_return.csv.gz"
    feature_path = OUT / "haist70_tigit_cd39_patient_features.csv"
    source_effect_path = OUT / "rahim_pair_vs_projected_niche_by_donor.csv"
    write_csv_gz(source, source_path)
    write_csv_gz(target, target_path)
    features.to_csv(feature_path, index=False)
    source_donor_results.to_csv(source_effect_path, index=False)

    manifest = {
        "status": "outcome-blind TIGIT/CD39 lock, return, and niche-feature construction",
        "survival_outcome_read": False,
        "program": lock,
        "source_state": (
            "unweighted mean of within-Rahim-donor percentile midranks of "
            "strict-no-LAG3 paper-CLR TIGIT and CD39"
        ),
        "return_alignment": "existing PT70 canonical RAMP 2.3.0 alignment",
        "return_predictor": (
            "distance-weighted k=30 Rahim-to-Haist predictor per Rahim donor; "
            "median across the three donor-specific predictions"
        ),
        "source_cells": int(len(source)),
        "source_donors": donor_order,
        "target_cells": int(len(target)),
        "target_patients": int(features.patient.nunique()),
        "niche_definition": "ROI-adjusted local k15 CAF/macrophage exposure > 0",
        "enrichment": enrichment,
        "inputs_sha256": {
            "alignment": sha256(PT70 / "alignment/aligned_ramp_ver2.float32.npy"),
            "metadata": sha256(PT70 / "input/cell_metadata.csv.gz"),
            "screen": sha256(SCREEN),
            "projected_niche": sha256(PROJECTED_NICHE),
            "spatial_exposure": sha256(EXPOSURE),
        },
        "outputs_sha256": {
            "source": sha256(source_path),
            "target": sha256(target_path),
            "patient_features": sha256(feature_path),
            "source_associations": sha256(source_effect_path),
        },
        "boundary": (
            "The pair is fixed before survival is opened. Because the PT70 cohort "
            "contributed to the upstream spatial-score projection and marker screen, "
            "the returned niche result is an internal closed-loop test, not an "
            "independent external validation."
        ),
    }
    (OUT / "PAIR_LOCK_AND_RETURN_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    (OUT / "NICHE_ENRICHMENT_SUMMARY.json").write_text(
        json.dumps(enrichment, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
