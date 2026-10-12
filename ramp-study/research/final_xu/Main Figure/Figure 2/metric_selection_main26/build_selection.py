"""Version the user-requested main-26 selection; leave frozen evaluations intact."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EXCLUDED = {
    "umap_unit_abs_asw": "User-requested exclusion of UMAP donor/ROI absolute silhouette after reviewing its scores; post-hoc reporting change.",
    "celltype_knn_purity": "User-requested joint exclusion of kNN purity and LISI purity after reviewing their scores; post-hoc reporting change.",
    "celltype_lisi_purity": "User-requested joint exclusion of kNN purity and LISI purity after reviewing their scores; post-hoc reporting change.",
    "native_knn_jaccard": "Earlier user-requested exclusion of geometry from the main metric set.",
    "celltype_graph_connectivity": "User-requested exclusion after reviewing graph-connectivity results; post-hoc reporting change.",
}
FAMILY_NAMES = {
    "biology": "细胞类型结构",
    "mixing": "平台／donor 混合",
    "transfer": "跨平台标签转移",
    "matching": "跨平台匹配覆盖",
    "visual": "UMAP 表现",
}


def read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    old_root = ROOT / "benchmark_results/comparison_evaluation/summary"
    registry_path = old_root / "metric_registry.csv"
    registry = read_csv(registry_path)
    selected = [r for r in registry if r["metric"] not in EXCLUDED]
    assert len(registry) == 31 and len(selected) == 26
    selected_names = {r["metric"] for r in selected}
    counts = Counter(r["family"] for r in selected)
    assert dict(counts) == {"biology": 5, "mixing": 16, "transfer": 2, "matching": 1, "visual": 2}
    sources = {str(registry_path.relative_to(ROOT)): sha(registry_path)}
    old_ranks_path = old_root / "primary_method_family_metric_ranks.csv"
    sources[str(old_ranks_path.relative_to(ROOT))] = sha(old_ranks_path)
    records = []
    for row in read_csv(old_ranks_path):
        if row["metric"] in selected_names:
            records.append({
                "benchmark": row["benchmark"], "method": row["method_family"],
                "metric": row["metric"], "family": row["metric_family"],
                "direction": row["direction"], "mean": row["mean"], "rank": row["rank"],
            })
    for story in ["story_lung_HLA_ABC", "figure_melanoma", "figure_HNSCC"]:
        path = ROOT / story / "Comparison/maxfuse_full_panel/evaluation/metric_method_ranks_all31.csv"
        sources[str(path.relative_to(ROOT))] = sha(path)
        for row in read_csv(path):
            if row["metric"] in selected_names:
                records.append({key: row[key] for key in records[0]})
    previous_path = HERE.parent / "metric_selection_main27/metric_ranks_7_benchmarks_main27.csv"
    previous = read_csv(previous_path)
    assert records == [row for row in previous if row["metric"] != "umap_unit_abs_asw"]
    assert {"umap_celltype_asw_macro", "umap_dataset_abs_asw", "highdim_unit_asw_conditional"} <= selected_names
    sources[str(previous_path.relative_to(ROOT))] = sha(previous_path)
    cases = {r["benchmark"] for r in records}
    assert len(cases) == 7 and len(records) == 7 * 6 * 26
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["benchmark"], row["method"])].append(row)
    rankings = []
    for (case, method), rows in grouped.items():
        assert len(rows) == 26 and {r["metric"] for r in rows} == selected_names
        families = defaultdict(list)
        for row in rows:
            families[row["family"]].append(float(row["rank"]))
        rankings.append({
            "benchmark": case, "method": method, "n_metrics": 26,
            "mean_rank_26": mean(float(r["rank"]) for r in rows),
            "family_balanced_mean_rank": mean(mean(v) for v in families.values()),
        })
    rankings.sort(key=lambda r: (r["benchmark"], r["mean_rank_26"]))
    distribution = [{"family": f, "label": FAMILY_NAMES[f], "n_metrics": n,
                     "share_percent": round(100 * n / 26, 2)} for f, n in counts.items()]
    write_csv(HERE / "metric_registry_main26.csv", selected)
    write_csv(HERE / "metric_family_counts.csv", distribution)
    write_csv(HERE / "metric_ranks_7_benchmarks_main26.csv", records)
    write_csv(HERE / "method_ranking_by_benchmark_main26.csv", rankings)
    policy = {
        "selection_id": "main26_without_geometry_connectivity_local_purity_or_umap_donor_asw",
        "status": "current_main_metric_selection",
        "n_metrics": 26, "excluded": EXCLUDED,
        "included_metrics": [r["metric"] for r in selected],
        "change_type": "post_hoc_user_requested_reporting_subset",
        "raw_31_metric_evaluations_preserved": True,
        "models_and_umaps_refit": False, "previous_figure_images_redrawn": False,
        "scope": "same selection for all seven benchmark cases and all six methods",
        "source_sha256": sources,
    }
    (HERE / "metric_selection.json").write_text(json.dumps(policy, ensure_ascii=False, indent=2) + "\n")
    lines = ["# 当前主评价：26 项指标", "",
             "本次仅在 27 项清单基础上移除 UMAP donor/ROI 绝对 silhouette（`umap_unit_abs_asw`）。UMAP 细胞类型和平台 silhouette 保留，高维同类细胞内 donor 混合 ASW 也保留。此前排除的四项延续不变。原始 31 项得分、30／29／27 项历史排名、模型、UMAP 和已有 Figure 图片均保留。此目录是新的主评价清单及基于已有单项排名重汇总的七组结果。", "",
             "此次是查看结果后的指标选择变更；没有将其写成预先固定的方案。七组、六种方法使用相同的 26 项清单。", "",
             "| 类别 | 指标数 | 单项等权时占比 |", "|---|---:|---:|"]
    for row in distribution:
        lines.append(f"| {row['label']} | {row['n_metrics']} | {row['share_percent']:.2f}% |")
    lines += ["| 合计 | 26 | 100% |", "",
              "以上占比适用于直接平均全部单项名次；五类等权汇总时，各类别仍各占 20%。移除图连通性和两项局部纯度后，biology 从最初 8 项减至 5 项，mixing 仍为 16 项；UMAP 类从 3 项减至 2 项。", "",
              "## 完整保留清单", ""]
    for family in counts:
        lines += [f"### {FAMILY_NAMES[family]}（{counts[family]} 项）", ""]
        for row in selected:
            if row["family"] == family:
                lines.append(f"- `{row['metric']}`：{row['definition']}")
        lines.append("")
    lines += ["## 文件", "",
              "- `metric_selection.json`：当前选择、排除理由及来源哈希。",
              "- `metric_registry_main26.csv`：26 项名称、类别、方向及定义。",
              "- `metric_family_counts.csv`：类别数量与占比。",
              "- `metric_ranks_7_benchmarks_main26.csv`：七组 × 六方法 × 26 项得分与单项名次。",
              "- `method_ranking_by_benchmark_main26.csv`：各 benchmark 的 26 项平均名次及五类等权平均名次；",
              "- `build_selection.py`：复现主清单，仅读取已有评价结果。",
              "- `VERIFICATION.json`：与上一版相比仅移除一项的核验。", ""]
    (HERE / "README.md").write_text("\n".join(lines))
    assert all(sha(ROOT / path) == digest for path, digest in sources.items())
    verification = {"status": "passed", "exact_main27_minus_umap_unit_abs_asw": True,
                    "unchanged_remaining_scores_and_ranks": True,
                    "other_two_umap_metrics_retained": True,
                    "highdim_donor_mixing_asw_retained": True,
                    "benchmarks": 7, "methods": 6, "metrics": 26,
                    "rank_rows": len(records), "source_files_unchanged": True}
    (HERE / "VERIFICATION.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps({"status": "passed", "metrics": 26, "counts": counts,
                      "benchmarks": sorted(cases), "rank_rows": len(records),
                      "source_files_unchanged": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
