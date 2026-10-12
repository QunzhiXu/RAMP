#!/usr/bin/env python3
"""Combine the four completed 31-metric benchmark evaluations."""

from __future__ import annotations

import json
import hashlib
import os
import tempfile
from pathlib import Path

CACHE = Path(
    os.environ.get(
        "RAMP_EVALUATION_CACHE",
        str(Path(tempfile.gettempdir()) / "ramp-comparison-evaluation-cache"),
    )
)
for variable, subdirectory in (("MPLCONFIGDIR", "matplotlib"), ("XDG_CACHE_HOME", "xdg")):
    destination = CACHE / subdirectory
    destination.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(variable, str(destination))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "summary"
CASES = ("bone_marrow", "tnbc", "melanoma", "lung_cancer")
PRIMARY_FAMILIES = ("RAMP", "MaxFuse", "cyCombine", "Harmony-direct", "fastMNN", "CytoNorm")
EXPECTED_SCHEMA = "final_xu.comparison_evaluation.v2"
INCREMENTAL_SCHEMA = "final_xu.comparison_evaluation.v3_incremental"
EXPECTED_RESAMPLES = 20
EXPECTED_CELLS_PER_DATASET = 5000
KEY_METRICS = (
    "leiden_celltype_ari",
    "leiden_dataset_ari",
    "louvain_celltype_ari",
    "louvain_dataset_ari",
    "highdim_celltype_asw_macro",
    "highdim_dataset_asw_conditional",
    "celltype_knn_purity",
    "dataset_ilisi_norm",
    "conditional_kbet_acceptance",
    "crossplatform_knn_macro_f1",
    "native_knn_jaccard",
    "crossplatform_mnn_coverage",
    "umap_celltype_asw_macro",
    "umap_dataset_abs_asw",
)


def markdown_table(frame: pd.DataFrame, columns: list[str], limit: int = 50) -> str:
    local = frame.loc[:, columns].head(limit).copy()
    for column in local.select_dtypes(include=["float", "float32", "float64"]):
        local[column] = local[column].map(lambda value: f"{value:.3f}" if pd.notna(value) else "NA")
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in local.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    mapped = series.astype(str).str.lower().map({"true": True, "false": False})
    if mapped.isna().any():
        raise ValueError("include_in_method_rank contains non-boolean values")
    return mapped.astype(bool)


def validate_case(
    case: str,
    local: pd.DataFrame,
    inventory: pd.DataFrame,
    registry: pd.DataFrame,
    manifest: dict[str, object],
    checkpoint: dict[str, object],
) -> None:
    if manifest.get("schema_version") not in (EXPECTED_SCHEMA, INCREMENTAL_SCHEMA):
        raise ValueError(f"{case}: unexpected evaluation schema")
    if manifest.get("benchmark") != case:
        raise ValueError(f"{case}: manifest benchmark mismatch")
    metrics = registry["metric"].astype(str).tolist()
    if manifest.get("metric_registry") != metrics or manifest.get("metric_count") != len(metrics):
        raise ValueError(f"{case}: manifest metric registry mismatch")
    seeds = [int(value) for value in manifest.get("sampling_seeds", [])]
    if len(seeds) != EXPECTED_RESAMPLES or len(set(seeds)) != EXPECTED_RESAMPLES:
        raise ValueError(f"{case}: expected {EXPECTED_RESAMPLES} unique resampling seeds")
    if manifest.get("cells_per_dataset_per_resample") != EXPECTED_CELLS_PER_DATASET:
        raise ValueError(f"{case}: unexpected cells-per-dataset setting")
    fingerprint = str(manifest.get("run_fingerprint", ""))
    if not fingerprint or "run_fingerprint" not in local:
        raise ValueError(f"{case}: missing run fingerprint")
    if manifest.get("schema_version") == INCREMENTAL_SCHEMA:
        source_runs = manifest.get("method_run_fingerprints", {})
        if set(source_runs) != set(local["method"].astype(str)):
            raise ValueError(f"{case}: missing method-specific source fingerprints")
        expected = local["method"].map(source_runs).astype(str)
        if not expected.equals(local["run_fingerprint"].astype(str)):
            raise ValueError(f"{case}: a reused or added metric row has the wrong source fingerprint")
        configuration = checkpoint.get("configuration", {})
        sources = configuration.get("incremental_source_files", [])
        if not sources or configuration.get("method_run_fingerprints") != source_runs:
            raise ValueError(f"{case}: incomplete incremental source provenance")
        expected_fingerprint = hashlib.sha256(
            json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if expected_fingerprint != fingerprint:
            raise ValueError(f"{case}: incremental assembly fingerprint mismatch")
        for source in sources:
            if sha256_file(Path(source["path"])) != source["sha256"]:
                raise ValueError(f"{case}: incremental evaluation source changed: {source['path']}")
    elif set(local["run_fingerprint"].astype(str)) != {fingerprint}:
        raise ValueError(f"{case}: replicate fingerprint differs from manifest")
    if checkpoint.get("run_fingerprint") != fingerprint:
        raise ValueError(f"{case}: checkpoint fingerprint differs from completion manifest")
    checkpoint_config = checkpoint.get("configuration", {})
    if (
        checkpoint_config.get("benchmark") != case
        or checkpoint_config.get("schema_version") != manifest.get("schema_version")
        or checkpoint_config.get("methods") != manifest.get("methods")
        or checkpoint_config.get("sampling_seeds") != manifest.get("sampling_seeds")
    ):
        raise ValueError(f"{case}: checkpoint configuration differs from completion manifest")
    methods = manifest.get("methods")
    if not isinstance(methods, list) or set(methods) != set(inventory["method"].astype(str)):
        raise ValueError(f"{case}: method inventory differs from manifest")
    if inventory["method"].duplicated().any():
        raise ValueError(f"{case}: duplicate methods in inventory")
    rank_inventory = inventory[as_bool(inventory["include_in_method_rank"])]
    family_counts = rank_inventory.groupby("method_family").size().to_dict()
    expected_family_counts = {
        "RAMP": 1,
        "cyCombine": 1,
        "fastMNN": 1,
        "Harmony-direct": 1,
        "MaxFuse": 3,
        "CytoNorm": 1,
    }
    if family_counts != expected_family_counts:
        raise ValueError(f"{case}: primary method-family inventory differs: {family_counts}")
    expected = {(str(method), seed) for method in methods for seed in seeds}
    observed_rows = list(zip(local["method"].astype(str), local["sampling_seed"].astype(int)))
    if len(observed_rows) != len(set(observed_rows)) or set(observed_rows) != expected:
        raise ValueError(f"{case}: incomplete or duplicate method-by-seed grid")
    missing_columns = sorted(set(metrics) - set(local.columns))
    if missing_columns:
        raise ValueError(f"{case}: missing metric columns {missing_columns}")
    leiden_metrics = [metric for metric in metrics if metric.startswith("leiden_")]
    louvain_metrics = [metric for metric in metrics if metric.startswith("louvain_")]
    cluster_metrics = set(leiden_metrics + louvain_metrics)
    for prefix, local_metrics in (("leiden", leiden_metrics), ("louvain", louvain_metrics)):
        valid = local[f"{prefix}_cluster_count_valid"].astype(bool)
        if local.loc[valid, local_metrics].isna().any().any():
            raise ValueError(f"{case}: valid {prefix} rows contain NA cluster metrics")
        if local.loc[~valid, local_metrics].notna().any().any():
            raise ValueError(f"{case}: invalid {prefix} rows contain numeric cluster metrics")
    noncluster = [metric for metric in metrics if metric not in cluster_metrics]
    if local[noncluster].isna().any().any():
        raise ValueError(f"{case}: non-clustering metrics contain NA")
    maxfuse = inventory[inventory["method_family"] == "MaxFuse"]
    if len(maxfuse) != 3 or not maxfuse["source_path"].astype(str).str.contains(
        "/maxfuse_archive/", regex=False
    ).all():
        raise ValueError(f"{case}: reportable MaxFuse inputs are not the three archive hubs")
    if not bool(manifest.get("labels_used_posthoc_for_evaluation")) or bool(
        manifest.get("labels_used_for_alignment")
    ):
        raise ValueError(f"{case}: label-use policy differs")


def main() -> None:
    SUMMARY.mkdir(parents=True, exist_ok=True)
    completion_manifest = SUMMARY / "manifest.json"
    incomplete_marker = SUMMARY / "RUN_INCOMPLETE"
    completion_manifest.unlink(missing_ok=True)
    incomplete_marker.write_text(
        "Cross-benchmark summary is running or failed before completion.\n"
    )
    replicates = []
    inventories = []
    registry = None
    manifests = []
    checkpoints = []
    common_protocol = None
    for case in CASES:
        case_root = ROOT / case
        required = [
            case_root / "metric_battery_replicates.csv",
            case_root / "metric_registry.csv",
            case_root / "method_inventory.csv",
            case_root / "manifest.json",
            case_root / "checkpoint_manifest.json",
        ]
        if missing := [str(path) for path in required if not path.exists()]:
            raise FileNotFoundError(f"{case}: incomplete evaluation: {missing}")
        local = pd.read_csv(case_root / "metric_battery_replicates.csv")
        inventory = pd.read_csv(case_root / "method_inventory.csv")
        local_registry = pd.read_csv(case_root / "metric_registry.csv")
        if registry is None:
            registry = local_registry
        elif not registry.equals(local_registry):
            raise ValueError(f"{case}: metric registry differs")
        manifest = json.loads((case_root / "manifest.json").read_text())
        checkpoint = json.loads((case_root / "checkpoint_manifest.json").read_text())
        validate_case(case, local, inventory, local_registry, manifest, checkpoint)
        protocol = manifest.get("common_umap_protocol")
        if common_protocol is None:
            common_protocol = protocol
        elif protocol != common_protocol:
            raise ValueError(f"{case}: common UMAP protocol differs")
        local["benchmark"] = case
        replicates.append(local)
        inventories.append(inventory)
        manifests.append(manifest)
        checkpoints.append(checkpoint)
    assert registry is not None
    evaluator_hashes = {
        str(checkpoint.get("configuration", {}).get("evaluator_sha256", ""))
        for checkpoint in checkpoints
    }
    if len(evaluator_hashes) != 1 or "" in evaluator_hashes:
        raise ValueError("the four cases were not generated by the same evaluator code")
    all_replicates = pd.concat(replicates, ignore_index=True)
    inventory = pd.concat(inventories, ignore_index=True)
    all_replicates.to_csv(
        SUMMARY / "all_benchmark_metric_replicates.csv.gz", index=False, compression="gzip"
    )
    inventory.to_csv(SUMMARY / "all_benchmark_method_inventory.csv", index=False)
    registry.to_csv(SUMMARY / "metric_registry.csv", index=False)

    # Attach method family and primary/sensitivity status.
    metadata = inventory[["benchmark", "method", "method_family", "variant", "role", "include_in_method_rank"]].drop_duplicates()
    metadata["include_in_method_rank"] = as_bool(metadata["include_in_method_rank"])
    all_replicates = all_replicates.merge(metadata, on=["benchmark", "method"], how="left", validate="many_to_one")

    # A method-family comparison must not count three MaxFuse hubs as three
    # independent algorithms.  Collapse hubs by the per-seed median.  Harmony's
    # fixed-arcsinh run remains a sensitivity and is excluded from the primary
    # family table; source_scale is the primary Harmony-direct arm.
    primary = all_replicates[
        all_replicates["include_in_method_rank"]
        & all_replicates["method_family"].isin(PRIMARY_FAMILIES)
    ].copy()
    id_columns = ["benchmark", "sampling_seed", "method_family"]
    metric_columns = registry["metric"].tolist()
    family_rows = []
    for keys, block in primary.groupby(id_columns, sort=False):
        record = dict(zip(id_columns, keys))
        for metric in metric_columns:
            values = block[metric]
            record[metric] = float(values.median()) if values.notna().all() else float("nan")
        family_rows.append(record)
    family_replicates = pd.DataFrame(family_rows)
    expected_family_grid = {
        (case, int(seed), family)
        for case, block in all_replicates.groupby("benchmark")
        for seed in block["sampling_seed"].astype(int).unique()
        for family in PRIMARY_FAMILIES
    }
    observed_family_grid = set(
        family_replicates[id_columns].itertuples(index=False, name=None)
    )
    if expected_family_grid != observed_family_grid or family_replicates.duplicated(
        id_columns
    ).any():
        raise ValueError("primary method-family by seed grid is incomplete or duplicated")

    exclusion_rows = []
    for (benchmark, seed), indices in family_replicates.groupby(
        ["benchmark", "sampling_seed"]
    ).groups.items():
        indices = list(indices)
        for metric in metric_columns:
            missing_families = family_replicates.loc[
                indices, ["method_family", metric]
            ].loc[lambda frame: frame[metric].isna(), "method_family"].astype(str).tolist()
            if missing_families:
                family_replicates.loc[indices, metric] = np.nan
                exclusion_rows.append(
                    {
                        "benchmark": benchmark,
                        "sampling_seed": int(seed),
                        "metric": metric,
                        "missing_method_families": ";".join(sorted(missing_families)),
                        "policy": "excluded_for_all_method_families",
                    }
                )
    pd.DataFrame(
        exclusion_rows,
        columns=[
            "benchmark",
            "sampling_seed",
            "metric",
            "missing_method_families",
            "policy",
        ],
    ).to_csv(SUMMARY / "matched_metric_exclusions.csv", index=False)
    family_replicates.to_csv(SUMMARY / "primary_method_family_replicates.csv", index=False)

    long_rows = []
    rank_rows = []
    for row in registry.itertuples(index=False):
        metric = row.metric
        case_means = (
            family_replicates.groupby(["benchmark", "method_family"])[metric]
            .agg(mean="mean", sd="std", n_resamples="count")
            .reset_index()
        )
        valid_counts = case_means.groupby("benchmark")["n_resamples"].nunique()
        if (valid_counts != 1).any() or (case_means["n_resamples"] == 0).any():
            raise ValueError(f"{metric}: method families do not share matched valid resamples")
        case_means["metric"] = metric
        case_means["metric_family"] = row.family
        case_means["direction"] = row.direction
        long_rows.append(case_means)
        for benchmark, block in case_means.groupby("benchmark"):
            ranks = block["mean"].rank(ascending=(row.direction == "lower"), method="average")
            for (_, record), rank in zip(block.iterrows(), ranks):
                rank_rows.append(
                    {
                        "benchmark": benchmark,
                        "metric": metric,
                        "metric_family": row.family,
                        "direction": row.direction,
                        "method_family": record["method_family"],
                        "mean": record["mean"],
                        "rank": rank,
                    }
                )
    family_summary = pd.concat(long_rows, ignore_index=True)
    family_summary.to_csv(SUMMARY / "primary_method_family_metric_summary.csv", index=False)
    ranks = pd.DataFrame(rank_rows)
    ranks.to_csv(SUMMARY / "primary_method_family_metric_ranks.csv", index=False)

    raw_case_metric_overall = (
        ranks.groupby("method_family", as_index=False)
        .agg(
            mean_rank_across_124_case_metrics=("rank", "mean"),
            median_rank=("rank", "median"),
            first_place_count=("rank", lambda x: int(np.sum(np.asarray(x) == 1))),
            top_two_count=("rank", lambda x: int(np.sum(np.asarray(x) <= 2))),
            n_case_metrics=("rank", "size"),
        )
        .sort_values("mean_rank_across_124_case_metrics")
    )
    balanced_units = (
        ranks.groupby(["benchmark", "metric_family", "method_family"], as_index=False)
        .agg(mean_rank_within_metric_family=("rank", "mean"), n_metrics=("metric", "nunique"))
    )
    balanced_units.to_csv(
        SUMMARY / "primary_method_family_balanced_rank_units.csv", index=False
    )
    family_balanced = (
        balanced_units.groupby("method_family", as_index=False)
        .agg(
            family_balanced_mean_rank=("mean_rank_within_metric_family", "mean"),
            family_balanced_median_rank=("mean_rank_within_metric_family", "median"),
            n_benchmark_metric_families=("mean_rank_within_metric_family", "size"),
        )
    )
    overall = family_balanced.merge(
        raw_case_metric_overall,
        on="method_family",
        how="inner",
        validate="one_to_one",
    ).sort_values(["family_balanced_mean_rank", "mean_rank_across_124_case_metrics"])
    overall.to_csv(SUMMARY / "primary_method_family_overall_rank.csv", index=False)
    family_rank = (
        ranks.groupby(["metric_family", "method_family"], as_index=False)
        .agg(mean_rank=("rank", "mean"), median_rank=("rank", "median"), n_case_metrics=("rank", "size"))
        .sort_values(["metric_family", "mean_rank"])
    )
    family_rank.to_csv(SUMMARY / "primary_method_family_rank_by_metric_family.csv", index=False)

    key = family_summary[family_summary["metric"].isin(KEY_METRICS)].copy()
    key.to_csv(SUMMARY / "key_metric_summary.csv", index=False)

    method_order = overall["method_family"].tolist()
    metric_order = registry["metric"].tolist()
    rank_mean = ranks.groupby(["metric", "method_family"])["rank"].mean().unstack()
    matrix = rank_mean.reindex(index=metric_order, columns=method_order).to_numpy(dtype=float)
    figure, ax = plt.subplots(figsize=(9.8, 14.5))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis_r", vmin=1, vmax=len(method_order))
    ax.set_xticks(np.arange(len(method_order)), method_order, rotation=25, ha="right", fontsize=10)
    ax.set_yticks(np.arange(len(metric_order)), metric_order, fontsize=7.8)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center", fontsize=7.1,
                    color="white" if matrix[row, column] >= 3.5 else "#111827")
    ax.tick_params(length=0)
    ax.set_title(
        "Mean method-family rank across four benchmarks (31 metrics)",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    colorbar = figure.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("Mean rank (1 = best)")
    figure.tight_layout()
    figure.savefig(SUMMARY / "all_31_metrics_mean_rank_heatmap.png", dpi=240, bbox_inches="tight")
    figure.savefig(SUMMARY / "all_31_metrics_mean_rank_heatmap.pdf", bbox_inches="tight")
    plt.close(figure)

    # Key metrics are shown as case-specific ranks because raw values from
    # distinct metric families have different scales and directions.
    key_ranks = ranks[ranks["metric"].isin(KEY_METRICS)].copy()
    key_matrix = (
        key_ranks.groupby(["benchmark", "metric", "method_family"])["rank"]
        .mean()
        .unstack()
        .reindex(columns=method_order)
    )
    labels = [f"{case} | {metric}" for case, metric in key_matrix.index]
    figure, ax = plt.subplots(figsize=(10.5, max(10, 0.27 * len(labels) + 3)))
    image = ax.imshow(key_matrix.to_numpy(), aspect="auto", cmap="viridis_r", vmin=1, vmax=len(method_order))
    ax.set_xticks(np.arange(len(method_order)), method_order, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels, fontsize=7.3)
    ax.tick_params(length=0)
    ax.set_title("Selected biology, mixing, transfer, geometry, and UMAP metrics", fontweight="bold", pad=14)
    figure.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="Rank (1 = best)")
    figure.tight_layout()
    figure.savefig(SUMMARY / "key_metric_rank_heatmap.png", dpi=240, bbox_inches="tight")
    figure.savefig(SUMMARY / "key_metric_rank_heatmap.pdf", bbox_inches="tight")
    plt.close(figure)

    result = [
        "# Four-benchmark comparison: 31-metric battery",
        "",
        "Each benchmark generated 20 matched resamples of 5,000 cells per dataset. If a clustering arm fails its frozen target-count tolerance, that benchmark × seed × metric is excluded for every method family; `n_resamples` records the matched effective count.",
        "The UMAP-derived metrics use separately fitted spectral UMAPs with the same frozen parameters and no labels; representation dimensionality remains method-specific.",
        "",
        "## Overall family-balanced descriptive rank",
        "",
        markdown_table(
            overall,
            [
                "method_family",
                "family_balanced_mean_rank",
                "family_balanced_median_rank",
                "mean_rank_across_124_case_metrics",
                "first_place_count",
                "top_two_count",
                "n_benchmark_metric_families",
            ],
        ),
        "",
        "## Interpretation",
        "",
        "- MaxFuse's three frozen hub orientations are collapsed by the per-seed median before method-family ranking; all three remain available in each benchmark's variant-level files.",
        "- Harmony-direct source-scale is the primary Harmony arm; fixed-arcsinh is reported as a sensitivity but excluded from the primary family rank.",
        "- Unaligned source values are retained as a baseline but excluded from method ranks.",
        "- CytofRUV is not assigned metrics because all four benchmarks lack a true cross-batch replicated reference specimen.",
        "- CytoNorm 2.0.6 uses batch aggregates without dedicated controls and frozen benchmark-only preprocessing. Some FlowSOM clusters remain platform-specific; its results are an exploratory cross-platform application. Its preprocessing differs from historical cyCombine provider layers, so differences compare full pipelines, not correction formulas alone.",
        "- CytoNorm was added incrementally. Existing metric values and UMAPs were retained, with original per-method run fingerprints; the pre-CytoNorm tables and reporting code are archived under archive/pre_cytonorm_20260909/.",
        "- The headline descriptive rank first averages metrics within each benchmark × metric-family block, then gives those blocks equal weight; the raw 124 case-metric mean is retained as a secondary column.",
        "- The overall rank remains descriptive; biology, mixing, transfer, geometry, matching, and visual results should also be read separately.",
    ]
    (ROOT / "RESULTS.md").write_text("\n".join(result) + "\n")
    output_paths = sorted(
        [
            path
            for path in SUMMARY.iterdir()
            if path.is_file() and path.name not in {"manifest.json", "RUN_INCOMPLETE"}
        ]
        + [ROOT / "RESULTS.md"],
        key=lambda path: str(path),
    )
    manifest = {
        "schema_version": "final_xu.comparison_evaluation_summary.v2",
        "summary_script_sha256": sha256_file(Path(__file__).resolve()),
        "benchmarks": list(CASES),
        "metrics": metric_order,
        "metric_count": len(metric_order),
        "case_metric_count": len(metric_order) * len(CASES),
        "primary_method_families": list(PRIMARY_FAMILIES),
        "maxfuse_hub_aggregation": "per-seed median across three frozen hub orientations",
        "harmony_primary": "source_scale",
        "harmony_sensitivity": "adtnorm_fixed_arcsinh",
        "resamples_per_benchmark": EXPECTED_RESAMPLES,
        "cells_per_dataset_per_resample": EXPECTED_CELLS_PER_DATASET,
        "run_fingerprints": {
            str(manifest["benchmark"]): str(manifest["run_fingerprint"])
            for manifest in manifests
        },
        "common_umap_protocol": common_protocol,
        "overall_rank_policy": "equal weight per benchmark x metric-family block; raw 124 case-metric mean retained secondarily",
        "output_files": [
            {
                "path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in output_paths
        ],
    }
    temporary_manifest = completion_manifest.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary_manifest.replace(completion_manifest)
    incomplete_marker.unlink()
    print("combined summary complete", flush=True)


if __name__ == "__main__":
    main()
