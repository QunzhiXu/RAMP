#!/usr/bin/env python3
"""Publish the four frozen CytoNorm runs and incrementally rebuild comparison tables.

Run from any directory with Python 3. The full 20-seed evaluations must already
be complete under algorithm_test/cytonorm_validation/results. No fitting or
benchmark-data writes occur here. Existing metric CSV rows are retained byte
for byte; per-row source fingerprints are not replaced by the assembly hash.
"""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
RESULTS = Path(__file__).resolve().parent
PROJECT = RESULTS.parent
ALGORITHM = PROJECT / "algorithm/CytoNorm"
SOURCE = PROJECT / "algorithm_test/cytonorm_validation/results"
COMPARISON = RESULTS / "comparison_evaluation"
ARCHIVE = COMPARISON / "archive/pre_cytonorm_20260909"
STAGE = PROJECT / "algorithm_test/cytonorm_publication"
CASES = ("bone_marrow", "tnbc", "melanoma", "lung_cancer")
INTERNAL_METHOD = "CytoNorm_2.0.6_aggregate"
VARIANT = "v2.0.6_batch_aggregates_benchmark_only_preprocessing"
SCHEMA = "final_xu.comparison_evaluation.v3_incremental"
SCOPE = (
    "Exploratory cross-platform application using batch aggregates without dedicated controls. "
    "Frozen preprocessing uses benchmark_data only and differs from historical cyCombine "
    "provider layers; results compare full pipelines. Platform-specific FlowSOM clusters "
    "limit the shared-population assumption. No parameter selection used evaluation labels or scores."
)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


adapter = module("cytonorm_publication_adapter", ALGORITHM / "benchmark.py")
adapter.configure(STAGE)
import numpy as np
import pandas as pd

sha = adapter.sha
write_json = adapter.write_json
core = module("frozen_metric_core", PROJECT / "algorithm_test/vendor/evaluation_reference.py")
summary = module("incremental_summary", COMPARISON / "scripts/summarize_all.py")


def read_json(path):
    return json.loads(path.read_text())


def file_record(path, label=None):
    record = {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha(path)}
    if label is not None:
        record["label"] = label
    return record


def verify_records(records):
    for record in records:
        if sha(record["path"]) != record["sha256"]:
            raise ValueError(f"Source checksum changed: {record['path']}")


def verify_tree(root, name="COMPLETE.json"):
    record = read_json(root / name)
    if not record.get("outputs"):
        raise ValueError(f"Empty completion record: {root}")
    for relative, digest in record["outputs"].items():
        if sha(root / relative) != digest:
            raise ValueError(f"Completed file changed: {root / relative}")
    return record


def finish_tree(root, **fields):
    write_json(root / "COMPLETE.json", {**fields, "outputs": {
        str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*"))
        if p.is_file() and p != root / "COMPLETE.json"
    }})


def check_archive_and_core():
    for record in read_json(ARCHIVE / "SNAPSHOT.json")["files"]:
        if sha(ARCHIVE / record["path"]) != record["sha256"]:
            raise ValueError(f"Pre-CytoNorm snapshot changed: {record['path']}")
    old = ARCHIVE / "scripts/evaluate_benchmark.py"
    if sha(old) != sha(PROJECT / "algorithm_test/vendor/evaluation_reference.py"):
        raise ValueError("Old and CytoNorm metric implementations differ")
    current_tree = ast.parse((COMPARISON / "scripts/evaluate_benchmark.py").read_text())
    old_tree = ast.parse(old.read_text())
    functions = lambda tree: {x.name: ast.dump(x) for x in tree.body
                              if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)) and x.name != "load_case"}
    if functions(old_tree) != functions(current_tree):
        raise ValueError("Metric functions changed in the formal evaluator")


def validate_source(case, design):
    root = SOURCE / case
    adapter.fit(case, SOURCE, design)  # Existing COMPLETE files: checksum-only; never refits.
    completion = verify_tree(root / "evaluation")
    if (completion["status"], completion["resamples"], completion["metrics"]) != ("full_protocol", 20, 31):
        raise ValueError(f"{case}: full evaluation not complete")
    provenance = read_json(root / "evaluation/provenance.json")
    expected = dict(provenance)
    fingerprint = expected.pop("fingerprint")
    if adapter.fingerprint(expected) != fingerprint or completion["fingerprint"] != fingerprint:
        raise ValueError(f"{case}: evaluation fingerprint mismatch")
    if (provenance["reference_sha256"] != sha(PROJECT / "algorithm_test/vendor/evaluation_reference.py")
            or provenance["evaluator_sha256"] != sha(ALGORITHM / "evaluate.py")
            or provenance["fit_completion_sha256"] != sha(root / "alignment/COMPLETE.json")
            or provenance["seeds"] != list(core.SEEDS)
            or provenance["UMAP"] != core.UMAP_PROTOCOL):
        raise ValueError(f"{case}: evaluation protocol/code mismatch")
    verify_records(provenance["source_files"])
    label_path = PROJECT / "benchmark_data" / case / "relabel/relabeled_metadata.csv.gz"
    if sha(label_path) != provenance["labels_sha256"]:
        raise ValueError(f"{case}: labels changed")
    source, markers, _ = adapter.load_source(case, design["cases"][case])
    native_root = RESULTS / case / "ramp/alignment/input"
    native_meta = pd.read_csv(native_root / "cell_metadata.csv.gz", dtype=str)
    labels = pd.read_csv(label_path, dtype=str)
    columns = [x for x in ("cell_id", "dataset", "unit_key", "celltype", "celltype_joint_evaluation", "joint_evaluation") if x in labels]
    for column in columns:
        pd.testing.assert_series_equal(labels[column].fillna(""), native_meta[column].fillna(""), check_names=False)
    if not np.array_equal(core.standardized(source[markers].to_numpy(), "Unaligned"),
                          core.standardized(np.load(native_root / "raw.npy"), "Unaligned")):
        raise ValueError(f"{case}: native-neighbour evaluation baseline differs")
    xy = np.load(root / "evaluation/umap/coordinates" / f"{INTERNAL_METHOD}.npy")
    if xy.shape != (len(source), 2) or not np.isfinite(xy).all():
        raise ValueError(f"{case}: incomplete/nonfinite UMAP")
    return provenance


def publish_method(case, provenance):
    destination = RESULTS / case / "cytonorm"
    if destination.exists():
        prior = verify_tree(destination)
        if prior["evaluation_fingerprint"] != provenance["fingerprint"]:
            raise ValueError(f"{case}: refusing to overwrite a different published run")
        return destination
    staging = STAGE / "methods" / case
    if staging.exists():
        shutil.rmtree(staging)  # Dedicated disposable publication staging only.
    root = SOURCE / case
    shutil.copytree(root / "prepared", staging / "preprocessing")
    shutil.copytree(root / "alignment", staging / "alignment/data")
    shutil.copytree(root / "evaluation", staging / "evaluation")
    shutil.copy2(root / "run_config.json", staging / "alignment/executed_config.json")
    (staging / "umap").mkdir()
    for suffix in ("png", "pdf"):
        shutil.copy2(root / "evaluation" / f"umap.{suffix}", staging / "umap" / f"umap.{suffix}")
    (staging / "README.md").write_text(
        f"# {case} — CytoNorm 2.0.6\n\n"
        "四组冻结拟合之后，使用统一 31 指标 × 20 种子评估；UMAP 包含全部冻结细胞。\n\n"
        "- `alignment/data/`：校正矩阵、模型、逐细胞聚类、样条及覆盖诊断。\n"
        "- `preprocessing/`：仅来自 benchmark_data 的固定输入及变换记录。\n"
        "- `evaluation/`：完整指标、逐次结果、全量 UMAP 坐标及运行来源。\n"
        "- `umap/`：可直接查看的 PNG / PDF。\n\n"
        "这是使用 batch aggregate 的跨平台探索性应用；部分聚类仅包含一个平台。"
        "预处理与历史 cyCombine 的完整 provider 数据层存在差异，比较反映整条流程。\n\n"
        "已执行配置中的 algorithm_test 路径保留为真实运行记录；发布时未重新拟合。"
        "重跑请从 final_xu 根目录使用 algorithm/CytoNorm/README.md 的命令。\n"
    )
    write_json(staging / "manifest.json", {
        "schema_version": "final_xu.cytonorm_publication.v1", "benchmark": case,
        "method": "CytoNorm", "internal_evaluation_method": INTERNAL_METHOD,
        "version": "2.0.6", "variant": VARIANT, "scope": SCOPE,
        "source_run": str(root), "publication_operation": "byte-identical copy; no refit",
        "evaluation_fingerprint": provenance["fingerprint"],
        "source_completion_files": [file_record(root / sub / "COMPLETE.json") for sub in ("alignment", "evaluation")],
        "source_inputs": provenance["source_files"],
        "integrity": read_json(root / "alignment/integrity.json"),
        "publisher_sha256": sha(Path(__file__)),
    })
    finish_tree(staging, status="COMPLETE", evaluation_fingerprint=provenance["fingerprint"])
    verify_tree(staging)
    staging.rename(destination)
    return destination


def stage_comparison(case, method_root, provenance):
    old_root = ARCHIVE / case
    output = STAGE / "comparison_evaluation" / case
    output.mkdir(parents=True, exist_ok=True)
    old_path = old_root / "metric_battery_replicates.csv"
    old = pd.read_csv(old_path, float_precision="round_trip")
    added = pd.read_csv(method_root / "evaluation/metric_replicates.csv", float_precision="round_trip")
    added = added.rename(columns={"fingerprint": "run_fingerprint", "cells": "n_cells"})
    if set(added.method) != {INTERNAL_METHOD} or set(added.sampling_seed) != set(core.SEEDS) or len(added) != 20:
        raise ValueError(f"{case}: incomplete CytoNorm replicate grid")
    if set(added.run_fingerprint) != {provenance["fingerprint"]} or set(added.n_cells) != {15000}:
        raise ValueError(f"{case}: invalid CytoNorm replicate provenance")
    added["method"] = "CytoNorm"
    added["n_joint_celltypes"] = added.target_cluster_count.astype(int)
    for seed, block in old.groupby("sampling_seed"):
        if set(block.n_joint_celltypes) != set(added.loc[added.sampling_seed == seed, "n_joint_celltypes"]):
            raise ValueError(f"{case}: post-hoc joint-label selection differs")
    added = added.loc[:, old.columns].sort_values("sampling_seed")
    # Preserve every existing row's literal bytes and run fingerprint.
    combined_path = output / old_path.name
    combined_path.write_bytes(old_path.read_bytes() + added.to_csv(index=False, header=False).encode())
    combined = pd.read_csv(combined_path)
    roundtrip = pd.read_csv(combined_path, float_precision="round_trip")
    pd.testing.assert_frame_equal(old, roundtrip[roundtrip.method != "CytoNorm"].reset_index(drop=True), check_exact=True)
    inventory = pd.read_csv(old_root / "method_inventory.csv")
    old_manifest = read_json(old_root / "manifest.json")
    umap_source = method_root / "evaluation/umap"
    umap_manifest = read_json(umap_source / "manifests" / f"{INTERNAL_METHOD}.json")
    inventory = pd.concat([inventory, pd.DataFrame([{
        "benchmark": case, "method": "CytoNorm", "method_family": "CytoNorm", "variant": VARIANT,
        "role": "comparator", "include_in_method_rank": True, "output_type": "corrected_marker_matrix",
        "n_cells": old_manifest["n_cells_full"], "n_dimensions_raw": old_manifest["n_markers"],
        "n_dimensions_evaluated": umap_manifest["n_dimensions"],
        "source_path": str(method_root / "alignment/data/corrected_expression.csv.gz"),
    }])], ignore_index=True)
    inventory.to_csv(output / "method_inventory.csv", index=False)
    core.summarize(combined, inventory, output)
    # Preserve previously reported means/SDs exactly as well as raw replicates.
    local_summary = pd.read_csv(output / "metric_battery_summary.csv", float_precision="round_trip")
    (output / "metric_battery_summary.csv").write_bytes(
        (old_root / "metric_battery_summary.csv").read_bytes()
        + local_summary[local_summary.method == "CytoNorm"].to_csv(index=False, header=False).encode())
    for directory in ("coordinates", "manifests"):
        (output / "umap" / directory).mkdir(parents=True, exist_ok=True)
    for suffix in ("npy", "csv.gz"):
        shutil.copy2(umap_source / "coordinates" / f"{INTERNAL_METHOD}.{suffix}",
                     output / "umap/coordinates" / f"CytoNorm.{suffix}")
    umap_manifest.update(method="CytoNorm", original_method=INTERNAL_METHOD,
                         original_manifest_sha256=sha(umap_source / "manifests" / f"{INTERNAL_METHOD}.json"))
    write_json(output / "umap/manifests/CytoNorm.json", umap_manifest)
    source_records = [file_record(old_root / name) for name in
                      ("metric_battery_replicates.csv", "metric_battery_summary.csv", "manifest.json", "checkpoint_manifest.json", "method_inventory.csv")]
    source_records += [file_record(method_root / name) for name in
                       ("COMPLETE.json", "manifest.json", "evaluation/metric_replicates.csv", "evaluation/provenance.json", "evaluation/COMPLETE.json")]
    source_records.append(file_record(ARCHIVE / "scripts/evaluate_benchmark.py"))
    configuration = read_json(old_root / "checkpoint_manifest.json")["configuration"]
    source_runs = {method: old_manifest["run_fingerprint"] for method in old_manifest["methods"]}
    source_runs["CytoNorm"] = provenance["fingerprint"]
    new_matrix = file_record(method_root / "alignment/data/corrected_expression.csv.gz", "CytoNorm")
    new_manifest = file_record(method_root / "manifest.json", "CytoNorm_provenance")
    configuration.update(schema_version=SCHEMA, methods=inventory.method.tolist(),
                         method_run_fingerprints=source_runs, incremental_source_files=source_records,
                         inventory=inventory.to_dict("records"), publisher_sha256=sha(Path(__file__)),
                         cytonorm_evaluator_sha256=provenance["evaluator_sha256"],
                         evaluator_hash_meaning="Frozen numerical evaluator used by original and added runs; loader-only extension audited separately",
                         assembly_policy="Reuse unchanged old replicates and add frozen CytoNorm run; labels and standardized native baselines verified identical")
    configuration["source_files"] += [new_matrix, new_manifest]
    configuration["umap"].append({k: umap_manifest[k] for k in ("method", "representation_sha256", "coordinate_sha256", "protocol")})
    fingerprint = core.sha256_json(configuration)
    write_json(output / "checkpoint_manifest.json", {"run_fingerprint": fingerprint, "configuration": configuration})
    old_manifest.update(schema_version=SCHEMA, run_fingerprint=fingerprint, methods=inventory.method.tolist(),
                        method_run_fingerprints=source_runs, cytonorm_scope=SCOPE,
                        incremental_source_compatibility="Identical metric core, labels, cell order, standardized native baseline, sampling and UMAP protocols verified")
    write_json(output / "manifest.json", old_manifest)
    pd.concat([pd.read_csv(old_root / "input_output_file_hashes.csv"), pd.DataFrame([new_matrix, new_manifest])], ignore_index=True).to_csv(
        output / "input_output_file_hashes.csv", index=False)
    summary.validate_case(case, combined, inventory, pd.read_csv(output / "metric_registry.csv"), old_manifest,
                          read_json(output / "checkpoint_manifest.json"))
    return {"benchmark": case, "cells": old_manifest["n_cells_full"], "markers": old_manifest["n_markers"],
            "resamples": len(added), "metrics": 31, "numeric_metric_values": int(added[old_manifest["metric_registry"]].notna().sum().sum()),
            "missing_metric_values": int(added[old_manifest["metric_registry"]].isna().sum().sum()),
            "old_replicate_rows_unchanged": len(old), "status": "COMPLETE"}


def main():
    check_archive_and_core()
    design = read_json(ALGORITHM / "config.json")
    # Fail before any publication unless all four complete evaluations exist.
    for case in CASES:
        if not (SOURCE / case / "evaluation/COMPLETE.json").exists():
            raise ValueError(f"{case}: full 20-seed evaluation is still incomplete")
        if not (SOURCE / case / "alignment/COMPLETE.json").exists():
            raise ValueError(f"{case}: frozen alignment is missing")
    old_umap_files = [file_record(p) for case in CASES for p in (COMPARISON / case / "umap").rglob("*")
                      if p.is_file() and not p.name.startswith("CytoNorm.")]
    old_input_files = [r for case in CASES for r in read_json(ARCHIVE / case / "checkpoint_manifest.json")["configuration"]["source_files"]]
    verify_records(old_input_files)
    audit = []
    for case in CASES:
        provenance = validate_source(case, design)
        method_root = publish_method(case, provenance)
        audit.append(stage_comparison(case, method_root, provenance))
        print(case, "staged and validated", flush=True)
    summary.ROOT = STAGE / "comparison_evaluation"
    summary.SUMMARY = summary.ROOT / "summary"
    summary.main()
    summary_manifest = read_json(summary.SUMMARY / "manifest.json")
    for record in summary_manifest["output_files"]:
        if sha(summary.ROOT / record["path"]) != record["sha256"]:
            raise ValueError("Staged summary checksum mismatch")
    # Mark multi-file publication incomplete until the final summary manifest.
    marker = COMPARISON / "summary/RUN_INCOMPLETE"
    marker.write_text("Publishing validated CytoNorm incremental evaluation.\n")
    (COMPARISON / "summary/manifest.json").unlink(missing_ok=True)
    for p in sorted(summary.ROOT.rglob("*")):
        if not p.is_file() or p == summary.SUMMARY / "manifest.json":
            continue
        dest = COMPARISON / p.relative_to(summary.ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp = dest.with_name(dest.name + ".tmp")
        shutil.copy2(p, temp)
        temp.replace(dest)
    verify_records(old_umap_files)
    verify_records(old_input_files)
    for case in CASES:
        if not (COMPARISON / case / "metric_battery_replicates.csv").read_bytes().startswith(
                (ARCHIVE / case / "metric_battery_replicates.csv").read_bytes()):
            raise ValueError(f"{case}: prior metric rows changed during publication")
    shutil.copy2(summary.SUMMARY / "manifest.json", COMPARISON / "summary/manifest.json")
    marker.unlink()
    pd.DataFrame(audit).to_csv(RESULTS / "cytonorm_benchmark_summary.csv", index=False)
    write_json(STAGE / "PUBLICATION_VERIFICATION.json", {
        "status": "COMPLETE", "cases": audit, "original_umap_files_unchanged": old_umap_files,
        "original_method_inputs_unchanged": old_input_files,
        "summary_manifest_sha256": sha(COMPARISON / "summary/manifest.json"),
        "publisher_sha256": sha(Path(__file__)),
    })
    print("All four CytoNorm benchmarks published and comparisons rebuilt.", flush=True)


if __name__ == "__main__":
    main()
