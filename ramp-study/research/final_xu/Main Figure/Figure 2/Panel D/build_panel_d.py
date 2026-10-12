"""Summarize all frozen benchmark metrics with explicit category and overall ranks."""
from pathlib import Path
import os
import json
import hashlib

HERE = Path(__file__).resolve().parent
FIGURE = HERE.parent
MAIN = FIGURE.parent
ROOT = MAIN.parent
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(HERE/'.cache/matplotlib'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle

DPI = 210
ACCENT = '#BF4945'
INK = '#252B31'
CASES = ['bone_marrow', 'lung_cancer', 'melanoma', 'tnbc']
TITLES = {'bone_marrow': 'Bone marrow', 'lung_cancer': 'Lung cancer',
          'melanoma': 'Melanoma', 'tnbc': 'TNBC'}
FAMILIES = ['biology', 'mixing', 'transfer', 'geometry', 'matching', 'visual']
LABELS = ['Cell type', 'Mixing', 'Transfer', 'Geometry', 'Matching', 'UMAP', 'Overall']
READS = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(name):
    path = ROOT/'benchmark_results/comparison_evaluation/summary'/name
    READS[str(path)] = sha(path)
    return pd.read_csv(path)


def load_data():
    blocks = read_csv('primary_method_family_balanced_rank_units.csv')
    overall = read_csv('primary_method_family_overall_rank.csv')
    registry = read_csv('metric_registry.csv')
    assert len(registry) == 31 and set(registry.family) == set(FAMILIES)
    assert not blocks.duplicated(['benchmark', 'metric_family', 'method_family']).any()
    assert len(blocks) == 4*6*6 and set(blocks.benchmark) == set(CASES)
    expected_counts = registry.groupby('family').size()
    assert (blocks.n_metrics == blocks.metric_family.map(expected_counts)).all()
    overall = overall.sort_values('family_balanced_mean_rank').reset_index(drop=True)
    methods = overall.method_family.tolist()
    check = blocks.groupby('method_family').mean_rank_within_metric_family.mean().loc[methods]
    assert np.allclose(check, overall.family_balanced_mean_rank, atol=1e-12)
    overall['overall_position'] = overall.family_balanced_mean_rank.rank(method='min').astype(int)
    by_case = blocks.groupby(['benchmark', 'method_family']).mean_rank_within_metric_family.mean().rename('mean_rank').reset_index()
    by_case['overall_position'] = by_case.groupby('benchmark').mean_rank.rank(method='min').astype(int)
    blocks['category_position'] = blocks.groupby(['benchmark', 'metric_family']).mean_rank_within_metric_family.rank(method='min').astype(int)
    assert overall.loc[0, 'method_family'] == 'RAMP'
    assert by_case.query("method_family == 'RAMP'").overall_position.eq(1).all()
    return blocks, by_case, overall, methods


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'svg.fonttype': 'none', 'axes.linewidth': .55,
                         'xtick.major.width': .55, 'ytick.major.width': .55})
    blocks, by_case, overall, methods = load_data()
    reference = FIGURE/'Panel C/manifest.json'
    READS[str(reference)] = sha(reference)
    ref = json.loads(reference.read_text())
    width = ref['layout']['figure_width_in']
    height = ref['layout']['figure_height_in']
    fig = plt.figure(figsize=(width, height), dpi=DPI, facecolor='white')
    fig.text(.16/width, 1-.29/height, 'D', fontsize=17, weight='bold')
    fig.text(.56/width, 1-.29/height, 'Overall benchmark performance', fontsize=13.5, weight='bold')
    fig.text(1-.14/width, 1-.13/height, '31 METRICS  |  6 METHODS',
             fontsize=11.5, weight='bold', ha='right', va='top')
    fig.text(.56/width, 1-.66/height, 'Category ranks and overall rank within each benchmark', fontsize=10, color=INK)
    cmap = ListedColormap(['#236B80', '#719CA9', '#AAC4CF', '#CEDDE4', '#E5EDF1', '#F4F7F9'])
    norm = BoundaryNorm(np.arange(.5, 7.5, 1), cmap.N)
    image = None
    for i, case in enumerate(CASES):
        row, col = divmod(i, 2)
        left = [1.32, 6.03][col]
        bottom = height-[3.13, 6.03][row]
        ax = fig.add_axes([left/width, bottom/height, 3.22/width, 1.95/height])
        category = blocks.query('benchmark == @case').pivot(index='method_family', columns='metric_family', values='category_position').loc[methods, FAMILIES]
        local = by_case.query('benchmark == @case').set_index('method_family').loc[methods]
        values = np.column_stack([category.to_numpy(), local.overall_position.to_numpy()])
        image = ax.imshow(values, cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')
        ax.set_xticks(np.arange(7), LABELS, fontsize=8.5, rotation=45,
                      ha='right', rotation_mode='anchor')
        ax.tick_params(axis='x', length=0, pad=7)
        ax.set_yticks(np.arange(6), methods, fontsize=9.2)
        ax.tick_params(axis='y', length=0, pad=6)
        ax.set_xticks(np.arange(-.5, 7, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 6, 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=1.6)
        ax.tick_params(which='minor', bottom=False, left=False)
        ax.set_title(TITLES[case], loc='left', fontsize=12, weight='bold', pad=12)
        for r, method in enumerate(methods):
            for c in range(7):
                value = int(values[r, c])
                ax.text(c, r, str(value), ha='center', va='center', fontsize=11,
                        color='white' if value <= 2 else INK,
                        weight='bold' if value == 1 or c == 6 else 'normal')
            if method == 'RAMP':
                tick = ax.get_yticklabels()[r]
                tick.set_color(ACCENT); tick.set_weight('bold')
                ax.add_patch(Rectangle((-.5, r-.5), 7, 1, fill=False,
                                       edgecolor=ACCENT, linewidth=1.45, zorder=5, clip_on=False))
        # Separate the within-benchmark composite ranking from the six categories.
        ax.axvline(5.5, color='white', linewidth=4.5, zorder=3)
        ax.add_patch(Rectangle((5.5, -.5), 1, 6, fill=False, edgecolor='#506E7B',
                               linewidth=.8, zorder=4, clip_on=False))
        ax.get_xticklabels()[-1].set_weight('bold')
        for spine in ax.spines.values(): spine.set_visible(False)
    cax = fig.add_axes([3.50/width, 3.05/height, 3.8/width, .14/height])
    colorbar = fig.colorbar(image, cax=cax, orientation='horizontal', ticks=np.arange(1, 7))
    colorbar.outline.set_visible(False)
    cax.tick_params(length=0, labelsize=8, pad=3)
    fig.text(1.32/width, 3.115/height, 'Rank  (1 = best)', fontsize=9.3, va='center')
    fig.text(.56/width, 2.69/height, 'Overall ranking', fontsize=12, weight='bold')
    fig.text(9.25/width, 2.69/height, 'Equal weight per benchmark and category',
             fontsize=9, ha='right', color=INK)
    ax = fig.add_axes([1.80/width, .57/height, 7.12/width, 1.93/height])
    values = overall.family_balanced_mean_rank.to_numpy()
    y = np.arange(len(methods))
    ax.axhspan(-.45, .45, color='#FBEFED', zorder=0)
    ax.barh(y, values, height=.55, color=[ACCENT]+['#AEBAC3']*5, linewidth=0, zorder=2)
    ax.set_yticks(y, [f'#{p}  {m}' for p, m in zip(overall.overall_position, methods)], fontsize=10.5)
    ax.get_yticklabels()[0].set_color(ACCENT)
    ax.get_yticklabels()[0].set_weight('bold')
    ax.tick_params(axis='y', length=0, pad=9)
    ax.set_ylim(5.58, -.58)
    ax.set_xlim(0, 5)
    ax.set_xticks(np.arange(0, 6))
    ax.tick_params(axis='x', labelsize=8.5, length=3)
    ax.xaxis.grid(True, color='#E8EDF0', linewidth=.5, zorder=0)
    ax.set_axisbelow(True)
    for j, value in enumerate(values):
        ax.text(value+.075, j, f'{value:.2f}', va='center', fontsize=10.5,
                color=ACCENT if j == 0 else INK, weight='bold' if j == 0 else 'normal')
    for side in ['top', 'right', 'left']: ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('#8C8C8C')
    ax.set_xlabel('Mean rank (lower is better)', fontsize=10, weight='bold', labelpad=5)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = list(fig.texts)
    for axes in fig.axes:
        texts += [t for t in [axes.title, axes._left_title, axes.xaxis.label, axes.yaxis.label,
                              *axes.texts, *axes.get_xticklabels(), *axes.get_yticklabels()]
                  if t.get_text() and t.get_visible()]
    for text in texts:
        box = text.get_window_extent(renderer)
        assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fig.bbox.width+1 and box.y1 <= fig.bbox.height+1, (text.get_text(), box)
    for suffix in ['.png', '.svg']:
        fig.savefig(HERE/('panel_d_overall_benchmark_performance'+suffix), dpi=DPI)
    plt.close(fig)
    blocks.to_csv(HERE/'category_ranks.csv', index=False)
    by_case.to_csv(HERE/'benchmark_overall_ranks.csv', index=False)
    overall.to_csv(HERE/'overall_ranking.csv', index=False)
    assert all(sha(path) == digest for path, digest in READS.items())
    report = {
        'panel_label': 'D', 'input_sha256': READS, 'inputs_unchanged': True,
        'title': 'Overall benchmark performance', 'method_order': methods,
        'top_right_label': {'text': '31 METRICS  |  6 METHODS', 'fontsize_pt': 11.5, 'weight': 'bold'},
        'benchmarks': CASES, 'categories': FAMILIES, 'n_metrics': 31,
        'category_metric_counts': blocks.groupby('metric_family').n_metrics.first().to_dict(),
        'heatmap_values': 'Ordinal method position after averaging constituent metric ranks within a benchmark/category; 1 is best.',
        'heatmap_overall_column': 'Ordinal method position after averaging the six category mean ranks within each benchmark.',
        'overall_bar_values': 'Existing family_balanced_mean_rank: equal weight across all 24 benchmark/category blocks.',
        'ties': 'Minimum rank for ties; tied category leaders all shown as 1.',
        'highlight': 'All category leaders use rank-1 color and bold digits; RAMP row additionally outlined in red; overall first method highlighted.',
        'overall_first': methods[0], 'ramp_first_in_all_four_benchmarks': True,
        'selection': 'All 31 registered metrics and all six primary method families included; no new exclusions or weights.',
        'figure_width_in': width, 'figure_height_in': height, 'dpi': DPI,
    }
    (HERE/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print(HERE/'panel_d_overall_benchmark_performance.png')


if __name__ == '__main__': main()
