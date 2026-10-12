"""Render task-level boxplots from frozen primary endpoints; no fitting or selection."""
from pathlib import Path
import argparse
import hashlib
import json
import os

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

METHODS = ["ramp", "unaligned", "harmony_direct", "cycombine", "cytonorm",
           "fastmnn", "maxfuse", "clipped_zscore", "minmax", "percentile"]
LABELS = {"ramp": "RAMP", "unaligned": "Unaligned", "harmony_direct": "Harmony direct",
          "cycombine": "cyCombine", "cytonorm": "CytoNorm", "fastmnn": "fastMNN",
          "maxfuse": "MaxFuse", "clipped_zscore": "Clipped z-score",
          "minmax": "Min–max", "percentile": "Percentile"}
COLORS = {"unaligned": "#656565", "ramp": "#007C78", "clipped_zscore": "#A07617",
          "percentile": "#8557A4", "minmax": "#AB7474", "harmony_direct": "#A4A44D",
          "cycombine": "#4C8DAE", "cytonorm": "#C77638", "fastmnn": "#5266AB",
          "maxfuse": "#AF5580"}
FAMILIES = ["linear", "knn", "mlp"]
TASKS = ["bone_marrow", "tnbc", "melanoma", "lung_cancer", "lung_story",
         "melanoma_story", "hnscc_story"]
SETTINGS = {
    "B": dict(kind="J1", title="Cell-type annotation", count=7,
              metrics=["balanced_accuracy", "macro_f1"],
              labels=["Balanced accuracy ↑", "Macro-F1 ↑"],
              limits=[(0, 1), (0, 1)], ticks=[[0, .25, .5, .75, 1]] * 2,
              stem="panel_b_cell_type_annotation"),
    "C": dict(kind="J2", title="Missing-marker prediction", count=6,
              metrics=["effective_spearman", "mae"],
              labels=["Spearman ρ ↑", "Relative-expression MAE ↓"],
              limits=[(0, .55), (.15, .45)], ticks=[[0, .1, .2, .3, .4, .5], [.15, .25, .35, .45]],
              stem="panel_c_missing_marker_prediction"),
}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_scores():
    summary = Path((PROJECT / "results/formal_v0_5/LATEST_SUMMARY.txt").read_text().strip())
    report = json.loads((summary / "REPORT.json").read_text())
    assert report["status"] == "complete"
    assert report["passed"] == report["jobs"] == 53320
    assert report["failed"] == report["not_run"] == 0
    raw = pd.read_json(summary / "primary_means.json")
    rows = raw[raw.task.ne("ALL")].copy()
    assert set(rows.method) == set(METHODS)
    assert set(rows.family) == set(FAMILIES)
    keys = ["task_type", "task", "metric", "method"]
    assert not rows.duplicated(keys + ["family"]).any()
    assert rows.groupby(keys).family.nunique().eq(3).all()
    assert np.isfinite(rows.score).all()
    benchmark = rows.groupby(keys, as_index=False).agg(score=("score", "mean"),
                                                        model_count=("family", "nunique"))
    # Confirm two equivalent equal-weight aggregation paths agree with the
    # released overall endpoint means. Job counts are never used as weights.
    calculated = benchmark.groupby(["task_type", "metric", "method"]).score.mean()
    released = raw[raw.task.eq("ALL")].groupby(["task_type", "metric", "method"]).score.mean()
    assert calculated.index.equals(released.index)
    error = float(np.abs(calculated - released).max())
    assert error < 1e-12, error
    return summary, rows, benchmark, error


def draw_panel(fig, letter, frame):
    cfg = SETTINGS[letter]
    ink = "#202C32"
    fig.text(.024, .938, letter, fontsize=20, fontweight="bold", va="center", color=ink)
    fig.text(.073, .938, cfg["title"], fontsize=17, fontweight="bold", va="center", color=ink)
    fig.text(.973, .929, f"{cfg['count']} benchmarks", fontsize=10.1,
             fontweight="bold", ha="right", va="center", color=ink)
    axes = [fig.add_axes([.225, .165, .330, .635]),
            fig.add_axes([.622, .165, .350, .635])]
    task_order = [t for t in TASKS if t in set(frame.task)]
    offsets = dict(zip(task_order, np.linspace(-.16, .16, len(task_order))))
    for col, (ax, metric, title, bounds, ticks) in enumerate(zip(
            axes, cfg["metrics"], cfg["labels"], cfg["limits"], cfg["ticks"])):
        ax.set_title(title, fontsize=11.7, fontweight="bold", pad=8, color=ink)
        ax.axhspan(-.45, .45, facecolor=COLORS["ramp"], alpha=.055, zorder=0)
        for j, method in enumerate(METHODS):
            d = frame[(frame.method == method) & (frame.metric == metric)].set_index("task").loc[task_order]
            values = d.score.to_numpy()
            assert len(values) == cfg["count"]
            assert np.all((values >= bounds[0]) & (values <= bounds[1])), (letter, metric, method)
            color = COLORS[method]
            bp = ax.boxplot([values], positions=[j], orientation="horizontal", widths=.56,
                            patch_artist=True, whis=1.5, showfliers=False, manage_ticks=False,
                            boxprops=dict(facecolor=color, edgecolor=color, linewidth=1.0),
                            medianprops=dict(color=color, linewidth=1.6),
                            whiskerprops=dict(color=color, linewidth=.9),
                            capprops=dict(color=color, linewidth=.9))
            bp["boxes"][0].set_facecolor((*plt.matplotlib.colors.to_rgb(color), .14))
            # All individual values are drawn, including values outside whiskers.
            ax.scatter(values, [j + offsets[t] for t in task_order], s=18,
                       facecolor=color, edgecolor="white", linewidth=.35, alpha=.83, zorder=4)
            ax.scatter([values.mean()], [j], s=37, marker="D", facecolor="white",
                       edgecolor=ink, linewidth=1.05, zorder=6)
        ax.set_xlim(*bounds)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{v:g}" for v in ticks], fontsize=9.5, fontweight="bold")
        ax.set_ylim(len(METHODS)-.55, -.55)
        ax.set_yticks(range(len(METHODS)))
        if col == 0:
            ax.set_yticklabels([LABELS[m] for m in METHODS], fontsize=10.1, fontweight="bold")
            ax.get_yticklabels()[0].set_color(COLORS["ramp"])
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0, pad=7)
        ax.tick_params(axis="x", length=3, color="#64717A", pad=4)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#E8ECEE", linewidth=.65)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#6F7B83")
        ax.spines["bottom"].set_linewidth(.8)
    handles = [Line2D([], [], linestyle="none", marker="o", color="#697780",
                      markersize=4.6, label="Benchmark (3-model mean)"),
               Line2D([], [], linestyle="none", marker="D", markerfacecolor="white",
                      markeredgecolor=ink, markersize=4.8, label="Overall mean")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.55, .027),
               ncol=2, frameon=False, prop={"size": 9.5, "weight": "bold"},
               handletextpad=.55, columnspacing=1.7)


def write_panel_data(letter, raw, benchmark, summary, error):
    cfg = SETTINGS[letter]
    folder = HERE / f"Panel {letter}"
    folder.mkdir(exist_ok=True)
    mask = (benchmark.task_type == cfg["kind"]) & benchmark.metric.isin(cfg["metrics"])
    frame = benchmark[mask].copy()
    expected_tasks = set(TASKS) - ({"melanoma"} if letter == "C" else set())
    assert set(frame.task) == expected_tasks
    assert len(frame) == cfg["count"] * len(METHODS) * len(cfg["metrics"])
    frame.to_csv(folder / "per_benchmark_scores.csv", index=False)
    raw[(raw.task_type == cfg["kind"]) & raw.metric.isin(cfg["metrics"])].to_csv(
        folder / "per_model_scores.csv", index=False)
    means = frame.groupby(["metric", "method"], as_index=False).agg(
        mean=("score", "mean"), median=("score", "median"), n_benchmarks=("score", "size"))
    means.to_csv(folder / "overall_means.csv", index=False)
    provenance = dict(panel=letter, plot="horizontal boxplots with every benchmark point and overall mean",
                      source=str(summary / "primary_means.json"),
                      source_sha256=sha(summary / "primary_means.json"),
                      source_report_sha256=sha(summary / "REPORT.json"),
                      renderer_sha256=sha(Path(__file__)),
                      benchmark_count=cfg["count"], benchmark_tasks=sorted(expected_tasks),
                      methods=METHODS, model_families=FAMILIES, metrics=cfg["metrics"],
                      aggregation="Original within-model task endpoints, then equal mean of all 3 model families per benchmark; equal benchmark weights for overall means.",
                      max_difference_from_released_means=error,
                      box="Q1 to Q3; median line; 1.5 IQR whiskers; every observation explicitly overlaid",
                      no_significance_test=True, no_refitting=True,
                      scope="Descriptive comparison; benchmark configurations share some original cohorts.")
    (folder / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return folder, frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", choices=["B", "C", "both"], default="both")
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "Arial", "font.size": 10,
                         "text.color": "#202C32", "axes.labelcolor": "#202C32",
                         "xtick.color": "#202C32", "ytick.color": "#202C32",
                         "svg.fonttype": "none", "pdf.fonttype": 42,
                         "savefig.facecolor": "white"})
    summary, raw, benchmark, error = load_scores()
    selected = ["B", "C"] if args.panel == "both" else [args.panel]
    frames = {}
    for letter in selected:
        folder, frame = write_panel_data(letter, raw, benchmark, summary, error)
        frames[letter] = frame
        fig = plt.figure(figsize=(7.80, 4.32), facecolor="white")
        draw_panel(fig, letter, frame)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(folder / f"{SETTINGS[letter]['stem']}.{ext}", dpi=320)
        plt.close(fig)
    if args.panel == "both":
        folder = HERE / "Preview BC"
        folder.mkdir(exist_ok=True)
        fig = plt.figure(figsize=(15.60, 4.32), facecolor="white")
        for subfig, letter in zip(fig.subfigures(1, 2, wspace=0), selected):
            draw_panel(subfig, letter, frames[letter])
        for ext in ("png", "pdf", "svg"):
            fig.savefig(folder / f"panels_bc_boxplots.{ext}", dpi=260)
        plt.close(fig)
    print(json.dumps({"panels": selected, "refit": False,
                      "max_aggregation_error": error, "source": str(summary)}, indent=2))


if __name__ == "__main__":
    main()
