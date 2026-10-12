"""Post-hoc validation using the existing frozen 31-metric / UMAP protocol."""
from __future__ import annotations
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import time

from benchmark import HERE, PROJECT, DATA, sha, write_json, fingerprint, load_source, verify_complete


def evaluate(case, output, design, resamples):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    root = output / case
    for name in design["cases"]:
        verify_complete(output / name / "alignment")
    source, markers, records = load_source(case, design["cases"][case])
    corrected = pd.read_csv(root / "alignment/corrected_expression.csv.gz", dtype={"cell_id": str}, float_precision="round_trip")
    if corrected.cell_id.tolist() != source.cell_id.tolist() or list(corrected) != ["cell_id", *markers]:
        raise ValueError("Corrected matrix row/marker order mismatch")
    label_path = DATA / case / "relabel/relabeled_metadata.csv.gz"
    header = pd.read_csv(label_path, nrows=0).columns
    cols = [c for c in ["cell_id", "dataset", "unit_key", "celltype", "celltype_joint_evaluation", "joint_evaluation"] if c in header]
    meta = pd.read_csv(label_path, usecols=cols, dtype=str)
    for c in ["cell_id", "dataset", "unit_key"]:
        if meta[c].tolist() != source[c].tolist():
            raise ValueError(f"Post-hoc {c} order differs")
    reference = PROJECT / "algorithm_test/vendor/evaluation_reference.py"
    spec = importlib.util.spec_from_file_location("cytonorm_metric_reference", reference)
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    method = "CytoNorm_2.0.6_aggregate"
    dest = root / ("evaluation" if resamples == 20 else "validation_metrics_1seed")
    dest.mkdir(exist_ok=True)
    x = ev.standardized(corrected[markers].to_numpy(), method)
    native = ev.standardized(source[markers].to_numpy(), "Unaligned")
    cfg = {"case": case, "method": method, "fit_completion_sha256": sha(root / "alignment/COMPLETE.json"),
           "source_files": records, "labels_sha256": sha(label_path), "reference_sha256": sha(reference),
           "evaluator_sha256": sha(__file__), "seeds": list(ev.SEEDS[:resamples]),
           "cells_per_dataset": 5000, "UMAP": ev.UMAP_PROTOCOL,
           "status": "full_protocol" if resamples == 20 else "integration_check_not_full_benchmark",
           "packages": {p: importlib.metadata.version(p) for p in
                        ["numpy", "pandas", "scipy", "scikit-learn", "umap-learn", "numba", "matplotlib", "igraph", "leidenalg"]}}
    key = fingerprint(cfg)
    record = dest / "provenance.json"
    if record.exists() and json.loads(record.read_text())["fingerprint"] != key:
        raise ValueError("Evaluation code/inputs changed; choose a fresh evaluation output")
    if not record.exists():
        write_json(record, {"fingerprint": key, **cfg})
    xy = ev.fit_or_load_umap(x, method, dest)
    label = "celltype_joint_evaluation" if "celltype_joint_evaluation" in meta else "celltype"
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), layout="constrained")
    order = np.random.default_rng(42).permutation(len(meta))
    for ax, column, title in zip(axes, ["dataset", "celltype"], ["Dataset", "Post-hoc cell type"]):
        values = meta[column].fillna("Unannotated").to_numpy()
        levels = list(pd.unique(values))
        cmap = plt.get_cmap("tab10" if column == "dataset" else "tab20")
        colors = {name: cmap(i % cmap.N) for i, name in enumerate(levels)}
        ax.scatter(xy[order, 0], xy[order, 1], s=0.45, alpha=.5, linewidths=0,
                   c=[colors[v] for v in values[order]], rasterized=True)
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        handles = [plt.Line2D([], [], marker="o", linestyle="", color=colors[v], label=v, markersize=4) for v in levels]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, -.02), ncol=2, fontsize=7, frameon=False)
    fig.suptitle(f"{case} · CytoNorm 2.0.6 · all {len(meta):,} frozen cells", fontsize=13)
    fig.savefig(dest / "umap.png", dpi=180)
    fig.savefig(dest / "umap.pdf")
    plt.close(fig)
    chunks = dest / "replicates"
    chunks.mkdir(exist_ok=True)
    for seed in ev.SEEDS[:resamples]:
        path = chunks / f"{seed}.json"
        if path.exists():
            if json.loads(path.read_text())["fingerprint"] != key:
                raise ValueError("Stale metric replicate")
            continue
        rows = ev.sample_rows(meta, seed, 5000)
        local = meta.iloc[rows].reset_index(drop=True)
        celltype = local[label].fillna("Unannotated").to_numpy()
        dataset = local.dataset.to_numpy(); unit = local.unit_key.to_numpy()
        joint = (local.joint_evaluation.str.lower().isin(["true", "1"]).to_numpy()
                 if "joint_evaluation" in local else
                 np.array([not bool(ev.INVALID_LABEL_PATTERN.search(s)) for s in celltype]))
        counts = pd.Series(celltype[joint]).value_counts()
        present = pd.DataFrame({"celltype": celltype[joint], "dataset": dataset[joint]}).groupby("celltype").dataset.nunique()
        levels = present[(present >= 2) & (counts >= 30)].index.to_numpy()
        valid = joint & np.isin(celltype, levels)
        if len(levels) < 2:
            raise ValueError("Insufficient joint labels")
        start = time.time()
        print(case, "metrics", seed, flush=True)
        scores = ev.evaluate_one(x[rows], native[rows], celltype, dataset, unit, valid, seed, xy[rows])
        write_json(path, {"fingerprint": key, "benchmark": case, "method": method, "sampling_seed": seed,
                         "cells": len(rows), "seconds": time.time() - start,
                         **{k: float(v) if np.isfinite(v) else None for k, v in scores.items()}})
    reps = pd.DataFrame([json.loads((chunks / f"{s}.json").read_text()) for s in ev.SEEDS[:resamples]])
    reps.to_csv(dest / "metric_replicates.csv", index=False)
    summary = [{"metric": m, "family": family, "direction": direction, "definition": definition,
                "mean": reps[m].mean(), "sd": reps[m].std(ddof=1),
                "valid_resamples": int(reps[m].notna().sum())} for m, family, direction, definition in ev.REGISTRY]
    pd.DataFrame(summary).to_csv(dest / "metric_summary.csv", index=False)
    for r in records + [{"path": str(label_path), "sha256": cfg["labels_sha256"]}]:
        if sha(r["path"]) != r["sha256"]:
            raise ValueError("Benchmark data changed during evaluation")
    write_json(dest / "COMPLETE.json", {"fingerprint": key, "status": cfg["status"],
        "all_frozen_cells_in_umap": len(xy), "metrics": len(ev.REGISTRY), "resamples": resamples,
        "inputs_unchanged": True,
        "outputs": {str(p.relative_to(dest)): sha(p) for p in dest.rglob("*") if p.is_file() and p.name != "COMPLETE.json"}})
    print(case, "EVALUATION COMPLETE", flush=True)
