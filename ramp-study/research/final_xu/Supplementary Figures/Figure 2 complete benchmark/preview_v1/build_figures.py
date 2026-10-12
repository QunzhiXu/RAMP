"""Independent S8/S9 previews from the frozen main-23 integration scores."""
from pathlib import Path
import hashlib
import json
import os

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
os.environ.setdefault('MPLCONFIGDIR', str(HERE / '.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FuncFormatter
from matplotlib.text import Text
import numpy as np
import pandas as pd

OUT = HERE / 'output'
(OUT / 'pdf').mkdir(parents=True, exist_ok=True)
WIDTH, HEIGHT = 210, 297
INK, GREEN = '#202B30', '#008866'
METHODS = ['Harmony-direct', 'cyCombine', 'CytoNorm', 'fastMNN', 'MaxFuse', 'RAMP']
CASES = ['bone_marrow', 'tnbc', 'melanoma', 'lung_cancer', 'lung_story', 'melanoma_story', 'hnscc_story']
CASE_NAMES = ['Bone marrow', 'TNBC', 'Melanoma 1', 'Lung benchmark', 'Lung HLA-ABC', 'Melanoma 2', 'HNSCC']
COLORS = ['#397CA8', '#D99632', '#9873AE', '#469886', '#B65C70', '#59679A', '#8A7551']
SHAPES = ['o', 's', '^', 'v', 'P', 'h', 'X']
TITLES = {
    'leiden_celltype_ari': 'Leiden\nCell-type ARI',
    'leiden_celltype_nmi': 'Leiden\nCell-type NMI',
    'louvain_celltype_ari': 'Louvain\nCell-type ARI',
    'louvain_celltype_nmi': 'Louvain\nCell-type NMI',
    'highdim_celltype_asw_macro': 'Cell-type silhouette\nAligned space',
    'crossplatform_knn_macro_f1': 'Cross-platform label transfer\nMacro-F1',
    'crossplatform_knn_balanced_accuracy': 'Cross-platform label transfer\nBalanced accuracy',
    'umap_celltype_asw_macro': 'Cell-type silhouette\nUMAP',
    'umap_dataset_abs_asw': 'Absolute dataset silhouette\nUMAP',
    'leiden_dataset_ari': 'Leiden\nDataset ARI',
    'leiden_dataset_nmi': 'Leiden\nDataset NMI',
    'leiden_unit_ari': 'Leiden\nDonor/unit ARI',
    'leiden_unit_nmi': 'Leiden\nDonor/unit NMI',
    'louvain_dataset_ari': 'Louvain\nDataset ARI',
    'louvain_dataset_nmi': 'Louvain\nDataset NMI',
    'louvain_unit_ari': 'Louvain\nDonor/unit ARI',
    'louvain_unit_nmi': 'Louvain\nDonor/unit NMI',
    'dataset_ilisi_norm': 'Dataset\nNormalized iLISI',
    'unit_ilisi_norm': 'Donor/unit\nNormalized iLISI',
    'dataset_entropy_norm': 'Dataset\nEntropy',
    'conditional_kbet_acceptance': 'Conditional\nkBET-like acceptance',
    'cross_dataset_30nn_fraction': 'Cross-dataset\nNeighbor fraction',
    'conditional_cross_dataset_30nn_fraction': 'Same-type cross-dataset\nNeighbor fraction',
}
plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 8, 'font.weight': 'bold',
    'axes.labelweight': 'bold', 'axes.linewidth': .5,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.7,
    'pdf.fonttype': 42, 'svg.fonttype': 'none',
    'text.color': INK, 'axes.labelcolor': INK,
    'xtick.color': INK, 'ytick.color': INK,
})

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

INPUTS = {str((BASE / n).resolve()): sha(BASE / n) for n in [
    'metric_scores_main23.csv', 'metric_registry_main23.csv', 'PANEL_PLAN.csv', 'DESIGN_MANIFEST.json']}
DATA = pd.read_csv(BASE / 'metric_scores_main23.csv', float_precision='round_trip')
PLAN = pd.read_csv(BASE / 'PANEL_PLAN.csv').fillna({'block': ''})
REGISTRY = pd.read_csv(BASE / 'metric_registry_main23.csv')
assert len(DATA) == 966 and len(PLAN) == 23 and PLAN.metric.is_unique
assert set(DATA.metric) == set(PLAN.metric) == set(REGISTRY.metric)
assert set(DATA.benchmark) == set(CASES) and set(DATA.method) == set(METHODS)
assert not DATA.duplicated(['benchmark', 'method', 'metric']).any()
assert DATA.groupby(['metric', 'method']).size().eq(7).all()
assert np.isfinite(DATA['mean']).all()

def text(fig, x, y, content, **kw):
    return fig.text(x / WIDTH, 1 - y / HEIGHT, content, va='top', **kw)

def legend(fig):
    handles = [Line2D([], [], ls='', marker=shape, ms=4.3, color=color,
                      markeredgecolor='white', markeredgewidth=.35, label=label)
               for shape, color, label in zip(SHAPES, COLORS, CASE_NAMES)]
    handles += [Line2D([], [], ls='', marker='D', ms=4.3, color=INK,
                       markerfacecolor='none', label='Mean')]
    return fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, 1 - 26 / HEIGHT),
                      ncol=4, frameon=False, prop={'size': 7.8, 'weight': 'bold'},
                      columnspacing=1.45, handletextpad=.5, borderaxespad=0, labelspacing=.65)

def header(fig, which):
    name = ('Cell-type structure, transfer and UMAP' if which == 'S8'
            else 'Platform and donor mixing')
    n = 9 if which == 'S8' else 14
    text(fig, 9, 8, f'{which}  {name}', fontsize=14, weight='bold')
    text(fig, 9, 18, f'{n} metrics  |  7 benchmarks  |  6 methods', fontsize=8.7)
    legend(fig)
    text(fig, 105, 289, '↑ Higher is better    ↓ Lower is better', fontsize=7.8, ha='center')

def limits(values):
    low, high = float(values.min()), float(values.max())
    spread = max(high - low, .02)
    return low - .07 * spread, high + .07 * spread

def metric_panel(fig, item, x, y, w, h, first_column):
    g = DATA[DATA.metric.eq(item.metric)]
    arrays = [g[g.method.eq(m)].set_index('benchmark').loc[CASES, 'mean'].to_numpy() for m in METHODS]
    ax = fig.add_axes([x / WIDTH, 1 - (y + h) / HEIGHT, w / WIDTH, h / HEIGHT])
    box = ax.boxplot(arrays, positions=np.arange(6), orientation='horizontal', widths=.52,
                     patch_artist=True, showfliers=False, manage_ticks=False,
                     medianprops={'color': '#3D4B50', 'lw': .85},
                     whiskerprops={'color': '#869196', 'lw': .65},
                     capprops={'color': '#869196', 'lw': .65},
                     boxprops={'edgecolor': '#869196', 'lw': .65})
    for method, artist in zip(METHODS, box['boxes']):
        artist.set_facecolor('#C5E0D7' if method == 'RAMP' else '#EEF1F2')
    jitter = np.linspace(-.27, .27, 7)
    records, summaries = [], []
    for i, (method, values) in enumerate(zip(METHODS, arrays)):
        for j, (case, value) in enumerate(zip(CASES, values)):
            ax.scatter([value], [i + jitter[j]], s=15 if w > 40 else 12,
                       marker=SHAPES[j], c=COLORS[j], edgecolors='white', linewidths=.3,
                       alpha=.94, zorder=4)
            records.append(dict(figure=item.figure, panel=item.panel, metric=item.metric,
                                method=method, benchmark=case, score=float(value),
                                y_position=float(i + jitter[j])))
        mean = float(np.mean(values))
        ax.scatter([mean], [i], marker='D', s=22, facecolors='none',
                   edgecolors=INK, linewidths=.8, zorder=6)
        summaries.append(dict(figure=item.figure, panel=item.panel, metric=item.metric,
                              method=method, mean=mean, median=float(np.median(values)),
                              q1=float(np.quantile(values, .25)), q3=float(np.quantile(values, .75)), n=7))
    low, high = limits(g['mean'])
    ax.set_xlim(low, high)
    ax.set_ylim(5.60, -.60)
    ax.set_yticks(range(6), METHODS if first_column else [''] * 6)
    ax.tick_params(axis='y', length=0, pad=4)
    if first_column:
        ax.get_yticklabels()[-1].set_color(GREEN)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:.2f}'.rstrip('0').rstrip('.') if abs(value) > 1e-10 else '0'))
    ax.tick_params(axis='x', length=2, pad=2, width=.5)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color('#869196')
    ax.grid(axis='x', color='#E7EBED', linewidth=.5)
    ax.set_axisbelow(True)
    if low < 0 < high:
        ax.axvline(0, color='#A7AFB3', ls=(0, (2, 2)), lw=.6, zorder=1)
    arrow = '↑' if item.direction == 'higher' else '↓'
    text(fig, x, y - 11, item.panel, fontsize=9.2, weight='bold')
    text(fig, x + (6.2 if len(item.panel) == 2 else 5.2), y - 11,
         TITLES[item.metric] + ' ' + arrow, fontsize=8.2, weight='bold', linespacing=1.2)
    return records, summaries, dict(panel=item.panel, metric=item.metric, x_mm=x, y_mm=y,
                                   width_mm=w, height_mm=h, score_limits=[low, high],
                                   points=42, boxplots=6, mean_diamonds=6)

def check_text(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    W, H = fig.canvas.get_width_height()
    bounds = []
    objects = list(fig.texts)
    for legend_artist in fig.legends:
        objects.extend(legend_artist.get_texts())
    for ax in fig.axes:
        for axis, interval in [(ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())]:
            low, high = sorted(interval)
            for tick in axis.get_major_ticks():
                if low <= tick.get_loc() <= high:
                    objects.extend([tick.label1, tick.label2])
    for obj in objects:
        if not obj.get_visible() or not obj.get_text():
            continue
        bbox = obj.get_window_extent(renderer)
        assert bbox.x0 >= -1 and bbox.y0 >= -1 and bbox.x1 <= W + 1 and bbox.y1 <= H + 1, obj.get_text()
        bounds.append(dict(text=obj.get_text(), bounds=bbox.extents.tolist()))
    return bounds

def build(which):
    fig = plt.figure(figsize=(WIDTH / 25.4, HEIGHT / 25.4), dpi=160, facecolor='white')
    header(fig, which)
    if which == 'S9':
        text(fig, 9, 42, 'A  Cluster association', fontsize=10.5)
        text(fig, 9, 166, 'B  Neighborhood mixing', fontsize=10.5)
    records, summaries, panels = [], [], []
    for item in PLAN[PLAN.figure.eq(which)].itertuples(index=False):
        if which == 'S8':
            ncol, gap, h = 3, 9, 56
            y = [55, 132, 209][item.row - 1]
        elif item.block == 'A':
            ncol, gap, h = 4, 6, 36
            y = [59, 112][item.row - 1]
        else:
            ncol, gap, h = 3, 9, 36
            y = [187, 242][item.row - 1]
        w = (200 - 33 - (ncol - 1) * gap) / ncol
        x = 33 + (item.column - 1) * (w + gap)
        a, b, layout = metric_panel(fig, item, x, y, w, h, item.column == 1)
        records.extend(a); summaries.extend(b); panels.append(layout)
    checks = check_text(fig)
    name = 'supplement_s8_structure_transfer_umap' if which == 'S8' else 'supplement_s9_platform_donor_mixing'
    for extension in ['png', 'svg', 'pdf']:
        path = (OUT / 'pdf' if extension == 'pdf' else OUT) / f'{name}.{extension}'
        fig.savefig(path, dpi=300, facecolor='white', metadata={'Creator': 'RAMP supplementary figures'} if extension == 'pdf' else None)
    pd.DataFrame(records).to_csv(HERE / f'{which}_plotted_values.csv', index=False)
    pd.DataFrame(summaries).to_csv(HERE / f'{which}_boxplot_summaries.csv', index=False)
    plt.close(fig)
    return dict(figure=which, stem=name, panels=panels, plotted_values=len(records), text_bounds=checks)

if __name__ == '__main__':
    results = [build('S8'), build('S9')]
    assert sum(r['plotted_values'] for r in results) == 966
    for path, expected in INPUTS.items():
        assert sha(path) == expected
    outputs = {str(p.relative_to(HERE)): sha(p) for p in OUT.rglob('*') if p.is_file()}
    manifest = dict(status='independent_preview', formal_supplement_modified=False,
                    primary_reporting_metrics=23, benchmarks=7, methods=6, scores=966,
                    figure_size_mm=[WIDTH, HEIGHT], methods_order=METHODS,
                    case_order=CASES, case_colors=dict(zip(CASES, COLORS)),
                    source_hashes=INPUTS, source_inputs_unchanged=True,
                    script_sha256=sha(__file__), figures=results, output_sha256=outputs)
    (HERE / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Rendered S8 and S9; all 966 scores retained.')
