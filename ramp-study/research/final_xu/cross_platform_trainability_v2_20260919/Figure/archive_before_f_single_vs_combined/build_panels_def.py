"""Plot frozen learning/capacity/cohort experiments and assemble a six-panel draft."""
from pathlib import Path
import hashlib
import json
import os
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
REVIEW = PROJECT / "reports/formal_v0_5_20260921"
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))
os.environ.setdefault("MPLBACKEND", "Agg")

import fitz
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from build_panels_bc import METHODS, LABELS, COLORS

PANEL_HEIGHT = 4.32
COLUMN_WIDTHS = [7.10, 6.90, 6.90]
INK = "#202C32"
FAMILIES = ["linear", "knn", "mlp"]
FAMILY_LABELS = {"linear": "Linear", "knn": "kNN", "mlp": "Neural network"}
SETTINGS = {
    "D": dict(title="Label efficiency", task="J1", scope="learning", resource="fit_budget",
              values=[20, 100, 500, 2000], benchmarks=7,
              xlabel="Labeled training cells", footnote="512 validation cells at every budget",
              stem="panel_d_label_efficiency", limits=(.24, .58), ticks=[.3, .4, .5]),
    "E": dict(title="Model capacity", task="J1", scope="capacity", resource="width",
              values=[8, 32, 128], benchmarks=7,
              xlabel="Hidden-layer width", footnote="500 training cells · 512 validation cells",
              stem="panel_e_model_capacity", limits=(.24, .58), ticks=[.3, .4, .5]),
    "F": dict(title="Multi-cohort learning", task="J3", scope=None, resource="arm",
              values=["A-only", "B-only", "A+B"], benchmarks=5,
              xlabel="Labeled source cohorts", footnote="Matched training and validation budgets",
              stem="panel_f_multi_cohort_learning", limits=(.36, .71), ticks=[.4, .5, .6, .7]),
}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def prepare_data():
    summary = Path((PROJECT / "results/formal_v0_5/LATEST_SUMMARY.txt").read_text().strip())
    report = json.loads((summary / "REPORT.json").read_text())
    suite_path = PROJECT / "config/formal_v0_5/SUITE.json"
    suite = json.loads(suite_path.read_text())
    profile_path = PROJECT / "config/formal_v0_5/PROFILE.json"
    profile = json.loads(profile_path.read_text())
    assert report["status"] == "complete" and report["jobs"] == report["passed"] == 53320
    assert sha(suite_path) == report["suite_sha256"]
    jobs = {}
    for shard in suite["shards"]:
        path = Path(shard["config"])
        assert sha(path) == shard["sha256"]
        for job in json.loads(path.read_text())["jobs"]:
            if ((job["task_type"] == "J1" and job["family"] == "mlp")
                    or (job["task_type"] == "J3" and job["family"] in FAMILIES)):
                assert job["id"] not in jobs
                jobs[job["id"]] = job
    metrics = pd.read_csv(summary / "metrics.csv", low_memory=False).fillna({"marker": "", "arm": ""})
    selected = ((metrics.task_type.eq("J1") & metrics.family.eq("mlp"))
                | (metrics.task_type.eq("J3") & metrics.family.isin(FAMILIES)))
    metrics = metrics[selected & metrics.metric.eq("macro_f1")].copy()
    assert set(metrics.job_id) <= set(jobs)
    metrics["scopes"] = metrics.job_id.map(lambda key: jobs[key]["analysis_scopes"])
    all_tables = {}
    source_records = [{"path": str(p), "sha256": sha(p)} for p in
                      [summary / "metrics.csv", summary / "REPORT.json", suite_path, profile_path]]

    for letter, cfg in SETTINGS.items():
        folder = HERE / f"Panel {letter}"
        folder.mkdir(exist_ok=True)
        d = metrics[metrics.task_type.eq(cfg["task"])].copy()
        if cfg["scope"]:
            d = d[d.scopes.map(lambda x: cfg["scope"] in x)].copy()
        resource = cfg["resource"]
        assert set(d[resource]) == set(cfg["values"])
        assert d.task.nunique() == cfg["benchmarks"]
        assert set(d.method) == set(METHODS)
        assert set(d.seed) == set(profile["seeds"])
        if letter == "D":
            assert d.width.eq(profile["learning_width"]).all()
            assert (d.total_supervision == d.fit_budget + profile["validation_cells"]).all()
        elif letter == "E":
            assert d.fit_budget.eq(profile["primary_fit_cells"]).all()
            assert d.total_supervision.eq(profile["primary_fit_cells"] + profile["validation_cells"]).all()
        else:
            assert set(d.family) == set(FAMILIES)
            assert d.loc[d.family.eq("mlp"), "width"].eq(profile["learning_width"]).all()
            assert d.groupby(["task", "direction", "seed"]).total_supervision.nunique().eq(1).all()

        family_keys = ["family"] if letter == "F" else []
        keys = family_keys + ["method", "task", "direction", resource]
        grouped = d.groupby(keys, as_index=False).agg(score=("score", "mean"),
                    n_repeats=("score", "size"), seeds=("seed", "nunique"),
                    min_parameters=("parameters", "min"), max_parameters=("parameters", "max"))
        assert grouped.n_repeats.eq(3).all() and grouped.seeds.eq(3).all()
        assert grouped.groupby(family_keys + ["method", "task", "direction"])[resource].nunique().eq(len(cfg["values"])).all()
        support = {method: set(map(tuple, g[family_keys + ["task", "direction", resource]].to_numpy()))
                   for method, g in grouped.groupby("method")}
        assert all(v == support["ramp"] for v in support.values())
        if letter == "F":
            family_support = [set(map(tuple, g[["task", "direction", resource]].to_numpy()))
                              for _, g in grouped.groupby("family")]
            assert all(v == family_support[0] for v in family_support)
        tasks = grouped.groupby(family_keys + ["method", "task", resource], as_index=False).score.mean()
        means = tasks.groupby(family_keys + ["method", resource], as_index=False).score.mean()
        index = family_keys + [resource]
        levels = (pd.MultiIndex.from_product([FAMILIES, cfg["values"]], names=index)
                  if family_keys else cfg["values"])
        table = means.pivot(index=index, columns="method", values="score").loc[levels, METHODS]
        # Independently reproduce the previously released descriptive analysis.
        if letter in ("D", "E"):
            reference_path = REVIEW / f"J1_{cfg['scope']}_common_support.csv"
            reference = pd.read_csv(reference_path, index_col=0).loc[cfg["values"], METHODS]
        else:
            reference_path = REVIEW / "J3_endpoints.csv"
            r = pd.read_csv(reference_path)
            reference = r.pivot(index=index, columns="method", values="score").loc[levels, METHODS]
        error = float(np.abs(table.to_numpy() - reference.to_numpy()).max())
        assert error < 1e-12
        assert np.isfinite(table.to_numpy()).all()
        assert table.to_numpy().min() >= cfg["limits"][0]
        assert table.to_numpy().max() <= cfg["limits"][1]
        means.to_csv(folder / "curve_means.csv", index=False)
        tasks.to_csv(folder / "per_benchmark_scores.csv", index=False)
        grouped.to_csv(folder / "per_direction_scores.csv", index=False)
        d.drop(columns=["scopes"]).to_csv(folder / "plotted_job_metrics.csv", index=False)
        if letter == "E":
            d.groupby(["method", "task", "width"], as_index=False).agg(
                min_parameters=("parameters", "min"), max_parameters=("parameters", "max")
            ).to_csv(folder / "parameter_counts.csv", index=False)
        provenance = dict(panel=letter, sources=source_records,
                          reference={"path": str(reference_path), "sha256": sha(reference_path)},
                          renderer_sha256=sha(Path(__file__)),
                          families=FAMILIES if letter == "F" else ["mlp"],
                          display_names=[FAMILY_LABELS[f] for f in (FAMILIES if letter == "F" else ["mlp"])],
                          metric="macro_f1", methods=METHODS, resource=resource, levels=cfg["values"],
                          benchmark_tasks=sorted(d.task.unique().tolist()),
                          benchmarks=int(d.task.nunique()), directions=int(d[["task", "direction"]].drop_duplicates().shape[0]),
                          seeds=profile["seeds"], matched_direction_support=True,
                          aggregation="Within each model: mean over seeds, then directions within benchmark, then equal benchmark means",
                          y_limits=cfg["limits"], max_error_vs_existing_report=error,
                          all_curve_values_visible=True, training_or_frozen_results_modified=False,
                          uncertainty="Descriptive mean curves, no confidence intervals or significance claims")
        (folder / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n")
        all_tables[letter] = table
    return all_tables


def handles():
    return [Line2D([], [], color=COLORS[m], marker="o", markersize=4,
                   linewidth=2.4 if m == "ramp" else 1.25, label=LABELS[m]) for m in METHODS]


def draw_panel(fig, letter, table, own_legend=True):
    cfg = SETTINGS[letter]
    fig.text(.024, .938, letter, fontsize=20, fontweight="bold", va="center", color=INK)
    fig.text(.073, .938, cfg["title"], fontsize=17, fontweight="bold", va="center", color=INK)
    fig.text(.972, .929, f"{cfg['benchmarks']} benchmarks", fontsize=10.1,
             fontweight="bold", ha="right", va="center", color=INK)
    if letter == "F":
        draw_cohort_facets(fig, table, own_legend)
        return
    fig.text(.555, .825, "Neural network", fontsize=11.7, fontweight="bold", ha="center", color=INK)
    bottom = .305 if own_legend else .225
    ax = fig.add_axes([.125, bottom, .835, .465 if own_legend else .545])
    xs = np.arange(3) if letter == "F" else np.array(cfg["values"])
    for method in [m for m in METHODS if m != "ramp"] + ["ramp"]:
        ax.plot(xs, table[method].to_numpy(), marker="o",
                markersize=5.8 if method == "ramp" else 4.0,
                linewidth=2.65 if method == "ramp" else 1.25,
                color=COLORS[method], alpha=1 if method == "ramp" else .88,
                markeredgecolor="white", markeredgewidth=.5,
                zorder=5 if method == "ramp" else 3)
    ax.set_ylim(*cfg["limits"])
    ax.set_yticks(cfg["ticks"])
    ax.set_ylabel("Macro-F1 ↑", fontsize=12, fontweight="bold", labelpad=7)
    ax.set_xlabel(cfg["xlabel"], fontsize=11.7, fontweight="bold", labelpad=7)
    if letter in ("D", "E"):
        ax.set_xscale("log", base=10 if letter == "D" else 2)
        ax.set_xticks(cfg["values"], [f"{x:,}" for x in cfg["values"]])
        ax.set_xlim(min(xs)/1.23, max(xs)*1.23)
        ax.minorticks_off()
    else:
        ax.set_xticks(xs, cfg["values"])
        ax.set_xlim(-.14, 2.14)
    ax.tick_params(labelsize=10.3, length=3, pad=4)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
    ax.grid(axis="y", color="#E5E9EC", linewidth=.75)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color("#6F7B83")
        ax.spines[side].set_linewidth(.8)
    fig.text(.555, .147 if own_legend else .053, cfg["footnote"], ha="center",
             fontsize=10.1, fontweight="bold", color=INK)
    if own_legend:
        legend = fig.legend(handles=handles(), loc="lower center", bbox_to_anchor=(.51, .009),
                            ncol=5, frameon=False, prop={"size": 9.1, "weight": "bold"},
                            handlelength=1.5, handletextpad=.45, columnspacing=1.35, labelspacing=.6)
        legend.get_texts()[0].set_color(COLORS["ramp"])


def draw_cohort_facets(fig, table, own_legend):
    """Show each prediction model separately on the same macro-F1 scale."""
    cfg = SETTINGS["F"]
    bottom = .305 if own_legend else .225
    gap = .040
    width = (.835 - 2*gap)/3
    axes = []
    for i, family in enumerate(FAMILIES):
        left = .125 + i*(width + gap)
        fig.text(left + width/2, .825, FAMILY_LABELS[family], fontsize=11.7,
                 fontweight="bold", ha="center", color=INK)
        ax = fig.add_axes([left, bottom, width, .465 if own_legend else .545],
                          sharey=axes[0] if axes else None)
        axes.append(ax)
        for method in [m for m in METHODS if m != "ramp"] + ["ramp"]:
            ax.plot(np.arange(3), table.loc[family, method].to_numpy(), marker="o",
                    markersize=5.0 if method == "ramp" else 3.3,
                    linewidth=2.4 if method == "ramp" else 1.1,
                    color=COLORS[method], alpha=1 if method == "ramp" else .88,
                    markeredgecolor="white", markeredgewidth=.45,
                    zorder=5 if method == "ramp" else 3)
        ax.set_ylim(*cfg["limits"])
        ax.set_yticks(cfg["ticks"])
        ax.set_xlim(-.17, 2.17)
        ax.set_xticks(np.arange(3), cfg["values"])
        ax.tick_params(axis="x", labelsize=9.0, length=3, pad=4)
        ax.tick_params(axis="y", labelsize=10.3, length=3 if i == 0 else 0,
                       pad=4, labelleft=i == 0)
        if i == 0:
            ax.set_ylabel("Macro-F1 ↑", fontsize=12, fontweight="bold", labelpad=7)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight("bold")
        ax.grid(axis="y", color="#E5E9EC", linewidth=.75)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("bottom", "left"):
            ax.spines[side].set_color("#6F7B83")
            ax.spines[side].set_linewidth(.8)
        if i:
            ax.spines["left"].set_visible(False)
    fig.text(.555, .199 if own_legend else .119, cfg["xlabel"], ha="center",
             fontsize=11.7, fontweight="bold", color=INK)
    fig.text(.555, .147 if own_legend else .053, cfg["footnote"], ha="center",
             fontsize=10.1, fontweight="bold", color=INK)
    if own_legend:
        legend = fig.legend(handles=handles(), loc="lower center", bbox_to_anchor=(.51, .009),
                            ncol=5, frameon=False, prop={"size": 9.1, "weight": "bold"},
                            handlelength=1.5, handletextpad=.45, columnspacing=1.35, labelspacing=.6)
        legend.get_texts()[0].set_color(COLORS["ramp"])


def save_panels(tables):
    for letter, table in tables.items():
        folder, cfg = HERE / f"Panel {letter}", SETTINGS[letter]
        for own_legend, suffix in [(True, ""), (False, "_for_combination")]:
            width = COLUMN_WIDTHS[ord(letter)-ord('D')]
            fig = plt.figure(figsize=(width, PANEL_HEIGHT), facecolor="white")
            draw_panel(fig, letter, table, own_legend=own_legend)
            for ext in ("png", "pdf", "svg"):
                fig.savefig(folder / f"{cfg['stem']}{suffix}.{ext}", dpi=300)
            plt.close(fig)
    folder = HERE / "Shared legend"
    folder.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(sum(COLUMN_WIDTHS), .39), facecolor="white")
    fig.legend(handles=handles(), loc="center", ncol=10, frameon=False,
               prop={"size": 11, "weight": "bold"}, handlelength=1.8,
               handletextpad=.6, columnspacing=2)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(folder / f"methods.{ext}", dpi=220)
    plt.close(fig)


def combine_vector_files(sources, placements, width, height, output):
    """Compose scientific vector panels without rasterizing their text/marks."""
    doc = fitz.open()
    page = doc.new_page(width=width, height=height)
    for stem, rect in zip(sources, placements):
        with fitz.open(stem.with_suffix(".pdf")) as source:
            page.show_pdf_page(fitz.Rect(*rect), source, 0)
    doc.save(output.with_suffix(".pdf"), garbage=4, deflate=True)
    page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).save(output.with_suffix(".png"))
    doc.close()

    # Build a matching SVG with namespaced element ids and unchanged viewBoxes.
    ns = "http://www.w3.org/2000/svg"
    ET.register_namespace("", ns)
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    root = ET.Element(f"{{{ns}}}svg", width=f"{width}pt", height=f"{height}pt",
                      viewBox=f"0 0 {width} {height}", version="1.1")
    ET.SubElement(root, f"{{{ns}}}rect", x="0", y="0", width=str(width), height=str(height), fill="white")
    for i, (stem, rect) in enumerate(zip(sources, placements)):
        nested = ET.fromstring(stem.with_suffix(".svg").read_text())
        ids = {el.get("id"): f"panel{i}_{el.get('id')}" for el in nested.iter() if el.get("id")}
        for el in nested.iter():
            if el.get("id"):
                el.set("id", ids[el.get("id")])
            for key, value in list(el.attrib.items()):
                if value.startswith("#") and value[1:] in ids:
                    el.set(key, "#" + ids[value[1:]])
                elif "url(#" in value:
                    for old, new in ids.items():
                        value = value.replace(f"url(#{old})", f"url(#{new})")
                    el.set(key, value)
        x0, y0, x1, y1 = rect
        nested.set("x", str(x0)); nested.set("y", str(y0))
        nested.set("width", str(x1-x0)); nested.set("height", str(y1-y0))
        root.append(nested)
    ET.ElementTree(root).write(output.with_suffix(".svg"), encoding="utf-8", xml_declaration=True)
    (output.parent / "LAYOUT.json").write_text(json.dumps({
        "canvas_points": [width, height], "vector_panels_preserved": True,
        "draft_layout": True,
        "sources": [{"path": str(s.with_suffix('.pdf')), "sha256": sha(s.with_suffix('.pdf')),
                     "rectangle": r} for s, r in zip(sources, placements)],
    }, indent=2) + "\n")


def assemble():
    widths, ph = [w*72 for w in COLUMN_WIDTHS], PANEL_HEIGHT*72
    gap, row_gap, legend_h = 4.0, 2.0, .39*72
    width = sum(widths) + 2*gap
    top = [HERE / "Panel A/panel_a_cross_platform_learning",
           HERE / "Panel B/panel_b_cell_type_annotation",
           HERE / "Panel C/panel_c_missing_marker_prediction"]
    lower = [HERE / f"Panel {letter}" / (SETTINGS[letter]["stem"] + "_for_combination") for letter in ("D", "E", "F")]
    legend = HERE / "Shared legend/methods"
    for s in top + lower + [legend]:
        assert s.with_suffix(".pdf").exists() and s.with_suffix(".svg").exists()
    for name, rows in [("Preview DEF", [lower]), ("Combined ABCDEF", [top, lower])]:
        folder = HERE / name
        folder.mkdir(exist_ok=True)
        placements, sources = [], []
        for row_i, row in enumerate(rows):
            y = row_i*(ph+row_gap)
            for col_i, stem in enumerate(row):
                x = sum(widths[:col_i]) + col_i*gap
                placements.append([x, y, x+widths[col_i], y+ph]); sources.append(stem)
        y = len(rows)*ph + (len(rows)-1)*row_gap
        sources.append(legend); placements.append([gap, y, width-gap, y+legend_h])
        stem = "panels_def" if len(rows) == 1 else "figure_cross_platform_learning"
        combine_vector_files(sources, placements, width, y+legend_h, folder / stem)


def main():
    plt.rcParams.update({"font.family": "Arial", "font.size": 10, "svg.fonttype": "none",
                         "pdf.fonttype": 42, "savefig.facecolor": "white", "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
    tables = prepare_data()
    save_panels(tables)
    assemble()
    print(json.dumps({"panels": list(tables), "models_refitted": False,
                      "combined": str(HERE / 'Combined ABCDEF/figure_cross_platform_learning.pdf')}, indent=2))


if __name__ == "__main__":
    main()
