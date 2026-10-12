"""Two cell-type preservation metrics across four frozen benchmarks."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from figure_style import *
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

METRICS = [
    ('leiden_celltype_ari', 'Cell-type ARI'),
    ('umap_celltype_asw_macro', 'UMAP cell-type silhouette'),
]
BENCHMARK_COLORS = ['#397CA8', '#E2A13D', '#9772B4', '#429984']
BENCHMARK_LABELS = ['Bone marrow', 'TNBC', 'Melanoma', 'Lung']


def main():
    setup()
    data = pd.read_csv(track(ROOT / 'benchmark_results/comparison_evaluation/summary/primary_method_family_metric_summary.csv'))
    selected = data[data.metric.isin([m for m, _ in METRICS])].copy()
    assert set(selected.method_family) == set(METHODS)
    assert set(selected.benchmark) == set(CASES)
    assert selected.n_resamples.eq(20).all() and len(selected) == 48
    assert not selected.duplicated(['metric', 'benchmark', 'method_family']).any()
    selected.to_csv(HERE / 'benchmark_metric_values.csv', index=False)
    fig = new_panel('C', 'Cell-type preservation')
    means = []
    for row, (metric, title) in enumerate(METRICS):
        ax = fig.add_axes([.093, .53-row*.43, .887, .295])
        local = selected[selected.metric.eq(metric)]
        table = local.pivot(index='method_family', columns='benchmark', values='mean').loc[METHODS, CASES]
        values = table.to_numpy()
        positions = np.arange(len(METHODS))
        artist = ax.boxplot(list(values), positions=positions, widths=.48, patch_artist=True,
                            showfliers=False, manage_ticks=False,
                            medianprops={'color': '#4A5155', 'linewidth': .8},
                            whiskerprops={'color': '#61696D', 'linewidth': .7},
                            capprops={'color': '#61696D', 'linewidth': .7},
                            boxprops={'edgecolor': '#61696D', 'linewidth': .7})
        for i, patch in enumerate(artist['boxes']):
            patch.set_facecolor('#C5E0D9' if METHODS[i] == 'RAMP' else '#E9EEF0')
        for j, color in enumerate(BENCHMARK_COLORS):
            ax.scatter(positions+[-.14,-.045,.045,.14][j], values[:,j],
                       s=19, facecolors=color, edgecolors='white', linewidths=.35, zorder=4)
        mean_values = values.mean(axis=1)
        assert METHODS[int(np.argmax(mean_values))] == 'RAMP'
        ax.scatter(positions, mean_values, marker='D', s=21, color='#172327',
                   edgecolors='white', linewidths=.4, zorder=5)
        for method, value in zip(METHODS, mean_values):
            means.append({'metric': metric, 'method': method, 'equal_benchmark_mean': float(value), 'n_benchmarks': 4})
        ax.set_title(title, loc='left', fontsize=10.5, weight='bold', pad=6)
        if row == 0:
            handles = [Line2D([], [], marker='o', linestyle='none', color=color, label=label, markersize=4)
                       for color, label in zip(BENCHMARK_COLORS, BENCHMARK_LABELS)]
            handles.append(Line2D([], [], marker='D', linestyle='none', color='#172327', label='Mean', markersize=4))
            ax.legend(handles=handles, loc='lower right', bbox_to_anchor=(1,1.017), ncol=5,
                      frameon=False, fontsize=8.3, borderaxespad=0, handlelength=.8,
                      handletextpad=.3, columnspacing=.7)
        ax.set_xticks(positions, METHODS)
        ax.tick_params(axis='x', labelsize=9, length=0, pad=4)
        ax.get_xticklabels()[-1].set_weight('bold')
        ax.get_xticklabels()[-1].set_color('#087F5B')
        ax.set_xlim(-.5,5.5)
        if row == 0:
            ax.set_ylim(0,.70); ax.set_yticks([0,.2,.4,.6]); ax.set_ylabel('ARI', fontsize=9.5, weight='bold')
        else:
            ax.set_ylim(-.105,.255); ax.set_yticks([-.1,0,.1,.2]); ax.set_ylabel('Silhouette', fontsize=9.5, weight='bold')
        ax.tick_params(axis='y', labelsize=8.6, length=2)
        ax.grid(axis='y', color='#E7E7E7', linewidth=.55)
        ax.set_axisbelow(True)
        for spine in ax.spines.values(): spine.set_color('#8C8C8C')
    pd.DataFrame(means).to_csv(HERE / 'benchmark_metric_means.csv', index=False)
    save(fig, HERE, 'panel_c_benchmark_performance', {
        'panel': 'C', 'metrics': [m for m, _ in METRICS], 'methods': METHODS,
        'benchmarks': CASES, 'all_metrics_higher_is_better': True,
        'display': 'Two vertically stacked boxplots; one colored point per benchmark mean, black diamond denotes the equal-benchmark mean. Boxes are median/IQR with 1.5 IQR whiskers; all four benchmark values are shown as points.',
        'selection': 'User selected the two cell-type preservation metrics after the four-metric mean review. RAMP ranks first in their equal-benchmark means; it ranks second for the omitted dataset iLISI and cross-platform F1. Complete four-metric results remain in mean_review and the archived heatmap.',
        'boxplot_n': 4, 'resamples_are_not_independent_benchmarks': True,
        'aggregation': 'Frozen mean of 20 matched resamples; MaxFuse hub runs already averaged within each resample in source family summary.',
    })


if __name__ == '__main__': main()
