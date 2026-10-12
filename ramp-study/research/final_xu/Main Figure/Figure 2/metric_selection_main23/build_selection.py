"""Apply the user-agreed metric removals to frozen scores; retain prior versions."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "metric_selection_main26"
REMOVED = {
    "highdim_dataset_asw_conditional": "User requested removal of the two mixing ASW metrics after reviewing results.",
    "highdim_unit_asw_conditional": "User requested removal of the two mixing ASW metrics after reviewing results.",
    "crossplatform_mnn_coverage": "Earlier user request to remove the matching-coverage category and retain four categories; applied here after the intervening discussion.",
}
LABELS = {"biology": "细胞类型结构", "mixing": "平台／donor 混合",
          "transfer": "跨平台标签转移", "visual": "UMAP 表现"}
FAMILY_WEIGHTS = {family: 0.25 for family in LABELS}


def read(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sources = [PREVIOUS / "metric_registry_main26.csv",
               PREVIOUS / "metric_ranks_7_benchmarks_main26.csv",
               PREVIOUS / "metric_selection.json"]
    hashes = {str(p): sha(p) for p in sources}
    old_registry, old_rows = read(sources[0]), read(sources[1])
    old_policy = json.loads(sources[2].read_text())
    registry = [r for r in old_registry if r["metric"] not in REMOVED]
    rows = [r for r in old_rows if r["metric"] not in REMOVED]
    counts = Counter(r["family"] for r in registry)
    assert dict(counts) == {"biology": 5, "mixing": 14, "transfer": 2, "visual": 2}
    assert len(old_registry) == 26 and len(registry) == 23
    names = {r["metric"] for r in registry}
    assert {"highdim_celltype_asw_macro", "umap_celltype_asw_macro", "umap_dataset_abs_asw"} <= names
    assert names.isdisjoint(REMOVED)
    grouped = defaultdict(list)
    family_scores = defaultdict(list)
    for row in rows:
        grouped[(row["benchmark"], row["method"])].append(row)
        family_scores[(row["benchmark"], row["method"], row["family"])].append(float(row["rank"]))
    assert len(grouped) == 7 * 6 and len(rows) == 7 * 6 * 23
    case_rankings = []
    for (case, method), block in grouped.items():
        assert len(block) == 23 and {r["metric"] for r in block} == names
        means = [mean(v) for (c, m, f), v in family_scores.items() if c == case and m == method]
        assert len(means) == 4
        case_rankings.append({"benchmark": case, "method": method, "n_metrics": 23,
                              "mean_rank_23": mean(float(r["rank"]) for r in block),
                              "family_balanced_mean_rank": mean(means)})
    overall = []
    for weighting, column in [("family_equal", "family_balanced_mean_rank"), ("metric_equal", "mean_rank_23")]:
        methods = defaultdict(list)
        for row in case_rankings:
            methods[row["method"]].append(row[column])
        scores = {method: mean(values) for method, values in methods.items()}
        for method, score in sorted(scores.items(), key=lambda pair: pair[1]):
            assert len(methods[method]) == 7
            overall.append({"weighting": weighting, "method": method, "mean_rank": score,
                            "overall_position": 1 + sum(v < score - 1e-12 for v in scores.values())})
    distribution = [{"family": f, "label": LABELS[f], "n_metrics": n,
                     "share_percent": round(100 * n / 23, 2),
                     "primary_weight_percent": 100 * FAMILY_WEIGHTS[f]} for f, n in counts.items()]
    primary_cases = []
    for row in case_rankings:
        score = row["family_balanced_mean_rank"]
        peers = [r["family_balanced_mean_rank"] for r in case_rankings if r["benchmark"] == row["benchmark"]]
        primary_cases.append({"benchmark": row["benchmark"], "method": row["method"],
                              "weighting": "family_equal", "mean_rank": score,
                              "position": 1 + sum(v < score - 1e-12 for v in peers)})
    primary_overall = [r for r in overall if r["weighting"] == "family_equal"]
    assert sum(FAMILY_WEIGHTS.values()) == 1.0
    write(HERE / "metric_registry_main23.csv", registry)
    write(HERE / "metric_ranks_7_benchmarks_main23.csv", rows)
    write(HERE / "metric_family_counts.csv", distribution)
    write(HERE / "method_ranking_by_benchmark_main23.csv", sorted(case_rankings, key=lambda r: (r["benchmark"], r["mean_rank_23"])))
    write(HERE / "overall_method_ranking_main23.csv", overall)
    write(HERE / "primary_method_ranking_by_benchmark.csv", sorted(primary_cases, key=lambda r: (r["benchmark"], r["mean_rank"])))
    write(HERE / "primary_overall_method_ranking.csv", primary_overall)
    excluded = dict(old_policy["excluded"])
    excluded.update(REMOVED)
    policy = {"selection_id": "main23_four_categories_without_mixing_asw",
              "status": "current_main_metric_selection", "n_metrics": 23,
              "included_metrics": [r["metric"] for r in registry], "excluded": excluded,
              "removed_from_previous_selection": REMOVED,
              "change_type": "post_hoc_user_requested_reporting_subset",
              "primary_weighting": "family_equal",
              "sensitivity_weighting": "metric_equal",
              "primary_family_weights": FAMILY_WEIGHTS,
              "benchmark_weighting": "equal_across_seven_benchmarks",
              "aggregation": "Within each benchmark, average the original per-metric ranks within each family, then average the four family means; finally average the seven benchmark scores. Lower is better. Do not re-rank the family means before averaging.",
              "weighting_choice": "Post hoc user choice after reviewing results, to prevent the 14 mixing metrics from dominating the 23-metric aggregate; applied uniformly to every method and benchmark.",
              "raw_31_metric_evaluations_preserved": True,
              "models_and_umaps_refit": False, "previous_figure_images_redrawn": False,
              "scope": "same selection for seven benchmark cases and all six methods",
              "previous_selection": str(PREVIOUS), "source_sha256": hashes,
              "original_source_sha256": old_policy.get("source_sha256", {})}
    (HERE / "metric_selection.json").write_text(json.dumps(policy, ensure_ascii=False, indent=2) + "\n")
    lines = ["# 当前主评价：四类等权、23 项指标", "",
             "主汇总采用四类等权，每类占 25%；单项等权保留为敏感性对照。先在每个 benchmark 内对各类所含指标的原始名次取平均，再平均四个类别得分，最后对七组 benchmark 等权平均。平均名次越低越好；不对类别均值再次排名，不按细胞数加权。", "",
             "本次从实际保存的 26 项清单移除两项混合 ASW，并落实此前已决定、但讨论中尚未写入文件的跨平台匹配覆盖删除，共移除 3 项。保留高维 cell-type silhouette、UMAP cell-type silhouette 和 UMAP 平台绝对 silhouette。", "",
             "移除项目为 `highdim_dataset_asw_conditional`、`highdim_unit_asw_conditional`、`crossplatform_mnn_coverage`。此前排除的项目延续不变；原始 31 项得分和历史清单均保留。所有修改都是查看结果后的报告选择，未标为预先固定方案。", "",
             "尚未执行此前讨论的 NMI、iLISI 或混合熵删减，也未采用五项混合度候选方案。", "",
             "| 类别 | 指标数 | 主汇总权重 | 单项等权对照权重 |", "|---|---:|---:|---:|"]
    for row in distribution:
        lines.append(f"| {row['label']} | {row['n_metrics']} | {row['primary_weight_percent']:.0f}% | {row['share_percent']:.2f}% |")
    lines += ["| 合计 | 23 | 100% | 100% |", "",
              "采用类别等权是查看结果后的报告选择，目的是避免混合度的 14 项指标因数量较多而主导综合得分；同一权重统一用于所有方法和七组 benchmark。该方案不消除类内指标之间的相关性。移除整个匹配类别会改变类别权重，因此不能把新旧类别等权总分差异全解释为模型性能变化。", ""]
    for weighting, heading in [("family_equal", "主结果：四类等权"), ("metric_equal", "敏感性对照：23 项单项等权")]:
        lines += [f"## 七组综合排名 · {heading}", "",
                  "| 方法 | 平均名次 ↓ | 综合位置 |", "|---|---:|---:|"]
        for row in overall:
            if row["weighting"] == weighting:
                lines.append(f"| {row['method']} | {row['mean_rank']:.4f} | {row['overall_position']} |")
        lines.append("")
    lines += ["", "## 保留清单", ""]
    for family, count in counts.items():
        lines += [f"### {LABELS[family]}（{count} 项）", ""]
        lines += [f"- `{r['metric']}`：{r['definition']}" for r in registry if r["family"] == family]
        lines.append("")
    lines += ["## 文件", "",
              "- `metric_registry_main23.csv`：保留名称、类别、方向和定义。",
              "- `metric_ranks_7_benchmarks_main23.csv`：七组、六方法的保留得分与原单项名次。",
              "- `method_ranking_by_benchmark_main23.csv`：逐组汇总。",
              "- `overall_method_ranking_main23.csv`：两种汇总方式的七组总体排名。",
              "- `primary_method_ranking_by_benchmark.csv`：主口径（四类等权）的逐组得分和位置。",
              "- `primary_overall_method_ranking.csv`：主口径的七组总体排名。",
              "- `metric_family_counts.csv`：各类指标数量、单项等权占比及主汇总权重。",
              "- `metric_selection.json`：具体移除记录和来源哈希。",
              "- `VERIFICATION.json`：仅移除指定三项、其余得分不变的核验。",
              "- `build_selection.py`：复现脚本，不重跑模型、UMAP 或底层指标。", ""]
    (HERE / "README.md").write_text("\n".join(lines))
    assert all(sha(Path(p)) == digest for p, digest in hashes.items())
    for p, digest in old_policy.get("source_sha256", {}).items():
        assert sha(HERE.parents[2] / p) == digest
    verification = {"status": "passed", "metrics": 23, "categories": 4,
                    "benchmarks": 7, "methods": 6, "rows": len(rows),
                    "only_three_agreed_metrics_removed_from_main26": True,
                    "retained_scores_and_individual_ranks_unchanged": True,
                    "previous_source_files_unchanged": True,
                    "primary_weighting": "family_equal",
                    "primary_family_weights": FAMILY_WEIGHTS,
                    "sensitivity_weighting": "metric_equal",
                    "models_umaps_and_previous_figures_unchanged": True}
    (HERE / "VERIFICATION.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification))


if __name__ == "__main__":
    main()
