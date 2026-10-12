"""Compare two reporting subsets without changing any fitted result or metric."""
import csv
import json
import hashlib
from collections import defaultdict
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
LABELS = {
    "bone_marrow": "Bone marrow", "tnbc": "TNBC",
    "lung_cancer": "原 Lung benchmark", "melanoma": "原 Melanoma benchmark",
    "lung_story": "Lung story", "melanoma_story": "Melanoma story",
    "hnscc_story": "HNSCC story",
}


def read(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def winner(left, right):
    return "tie" if abs(left - right) < 1e-12 else ("RAMP" if left < right else "MaxFuse")


def main():
    totals, cases = [], []
    source_hashes = {}
    for count in [29, 27]:
        path = HERE.parent / f"metric_selection_main{count}/method_ranking_by_benchmark_main{count}.csv"
        source_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = read(path)
        for weighting, key in [("metric_equal", f"mean_rank_{count}"), ("family_equal", "family_balanced_mean_rank")]:
            methods = defaultdict(list)
            by_case = defaultdict(dict)
            for row in rows:
                methods[row["method"]].append(float(row[key]))
                by_case[row["benchmark"]][row["method"]] = float(row[key])
            scores = {method: mean(values) for method, values in methods.items()}
            for method, score in scores.items():
                assert len(methods[method]) == 7
                totals.append({"n_metrics": count, "weighting": weighting, "method": method,
                               "mean_rank": score, "overall_position": 1 + sum(v < score - 1e-12 for v in scores.values())})
            for case, local in by_case.items():
                r, m = local["RAMP"], local["MaxFuse"]
                best = min(local.values())
                cases.append({"n_metrics": count, "weighting": weighting, "benchmark": case,
                              "ramp_mean_rank": r, "maxfuse_mean_rank": m,
                              "ramp_minus_maxfuse": r - m, "pairwise_winner": winner(r, m),
                              "best_of_six": ";".join(sorted(k for k, v in local.items() if abs(v - best) < 1e-12))})
    write(HERE / "overall_ranking_main29_vs_main27.csv", totals)
    write(HERE / "ramp_vs_maxfuse_by_benchmark_main29_vs_main27.csv", cases)
    old_rows = read(HERE.parent / "metric_selection_main29/metric_ranks_7_benchmarks_main29.csv")
    selected_rows = read(HERE / "metric_ranks_7_benchmarks_main27.csv")
    excluded = {"celltype_knn_purity", "celltype_lisi_purity"}
    expected = [row for row in old_rows if row["metric"] not in excluded]
    assert selected_rows == expected
    lines = ["# 同时移除两项局部纯度：29 项与 27 项对照", "",
             "两套汇总均使用已有单项名次，先在每个 benchmark 内汇总，再对七组等权。单项等权与五类等权分别保留；没有根据结果只展示较有利的一种。", "",
             "| 清单 | 汇总方式 | RAMP 平均名次 ↓ | MaxFuse 平均名次 ↓ |", "|---|---|---:|---:|"]
    for count in [29, 27]:
        for weighting in ["metric_equal", "family_equal"]:
            block = {r["method"]: r for r in totals if r["n_metrics"] == count and r["weighting"] == weighting}
            label = "单项等权" if weighting == "metric_equal" else "五类等权"
            lines.append(f"| {count} | {label} | {block['RAMP']['mean_rank']:.4f} | {block['MaxFuse']['mean_rank']:.4f} |")
    lines += ["", "27 项下，RAMP 在两种汇总方式的七组总体排名均为第一，但并非每组均优于 MaxFuse。", ""]
    for weighting in ["metric_equal", "family_equal"]:
        block = [r for r in cases if r["n_metrics"] == 27 and r["weighting"] == weighting]
        counts = {key: sum(r["pairwise_winner"] == key for r in block) for key in ["RAMP", "MaxFuse", "tie"]}
        label = "单项等权" if weighting == "metric_equal" else "五类等权"
        lines += [f"## {label}：27 项逐组比较", "",
                  f"RAMP 领先 {counts['RAMP']} 组，MaxFuse 领先 {counts['MaxFuse']} 组，持平 {counts['tie']} 组。", "",
                  "| Benchmark | RAMP ↓ | MaxFuse ↓ | 两者较好 | 六方法第一名 |",
                  "|---|---:|---:|---|---|"]
        for row in block:
            lines.append(f"| {LABELS[row['benchmark']]} | {row['ramp_mean_rank']:.4f} | {row['maxfuse_mean_rank']:.4f} | {row['pairwise_winner']} | {row['best_of_six']} |")
        lines.append("")
    lines += ["HNSCC 的五类等权汇总中，RAMP 与 MaxFuse 只差约 0.0058 个平均名次，且该组第一名是 fastMNN；这不足以表述为 RAMP 对 MaxFuse 的明显优势。平均名次差不代表原始效应量或统计显著性。", "",
              "删除依据为查看结果后的用户选择，所有原始指标及 29 项版本仍保留。本比较只描述现有数据及选定指标下的汇总结果。", ""]
    (HERE / "RANKING_COMPARISON.md").write_text("\n".join(lines))
    checks = {"status": "passed", "exact_29_minus_two_subset": True,
              "retained_scores_and_ranks_unchanged": True, "benchmark_count": 7,
              "method_count": 6, "metric_count": 27, "source_sha256": source_hashes}
    (HERE / "VERIFICATION.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks))


if __name__ == "__main__":
    main()
