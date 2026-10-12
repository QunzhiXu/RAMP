"""Four metrics, four frozen benchmarks, six formal method families."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from figure_style import *
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize

METRICS = [
    ('leiden_celltype_ari', 'Cell-type ARI'),
    ('umap_celltype_asw_macro', 'UMAP cell-type silhouette'),
    ('dataset_ilisi_norm', 'Dataset mixing (iLISI)'),
    ('crossplatform_knn_macro_f1', 'Cross-platform transfer (F1)'),
]


def main():
    setup()
    data = pd.read_csv(track(ROOT / 'benchmark_results/comparison_evaluation/summary/primary_method_family_metric_summary.csv'))
    selected = data[data.metric.isin([m for m, _ in METRICS])].copy()
    assert set(selected.method_family) == set(METHODS)
    assert set(selected.benchmark) == set(CASES)
    assert selected.n_resamples.eq(20).all() and len(selected) == 96
    assert not selected.duplicated(['metric', 'benchmark', 'method_family']).any()
    selected.to_csv(HERE / 'benchmark_metric_values.csv', index=False)
    fig = new_panel('C', 'Benchmark performance')
    cmap = LinearSegmentedColormap.from_list('figure_teal', ['#F3F6F7', '#94C4C3', '#07565A'])
    bounds = []
    for k, (metric, title) in enumerate(METRICS):
        col, row = k % 2, k // 2
        ax = fig.add_axes([.137+col*.501, .53-row*.438, .35, .293])
        local = selected[selected.metric.eq(metric)]
        table = local.pivot(index='method_family', columns='benchmark', values='mean').loc[METHODS, CASES]
        values = table.to_numpy()
        norm = Normalize(vmin=float(values.min()), vmax=float(values.max()))
        ax.imshow(values, cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')
        for i in range(len(METHODS)):
            for j in range(len(CASES)):
                value = values[i, j]
                color = 'white' if norm(value) > .64 else '#1D3033'
                ax.text(j, i, f'{value:.2f}', ha='center', va='center', fontsize=8.8,
                        color=color, weight='bold' if i == np.argmax(values[:, j]) else 'normal')
        ax.set_title(title, fontsize=10.5, weight='bold', pad=5)
        ax.set_xticks(range(4), ['Bone\nmarrow', 'TNBC', 'Melanoma', 'Lung'])
        ax.tick_params(axis='x', labelsize=8.4, length=0, pad=4)
        ax.set_yticks(range(6), METHODS)
        ax.tick_params(axis='y', labelsize=8.5, length=0, pad=4)
        for label in ax.get_yticklabels():
            if label.get_text() == 'RAMP': label.set_weight('bold')
        ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 6, 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=.75)
        ax.tick_params(which='minor', length=0)
        for spine in ax.spines.values(): spine.set_visible(False)
        bounds.append({'metric': metric, 'vmin': norm.vmin, 'vmax': norm.vmax})
    save(fig, HERE, 'panel_c_benchmark_performance', {
        'panel': 'C', 'metrics': [m for m, _ in METRICS], 'methods': METHODS,
        'benchmarks': CASES, 'all_metrics_higher_is_better': True,
        'display': 'Numbers are original scores, not within-column ranks; darker means higher within each metric; bold marks maximum within benchmark.',
        'color_limits': bounds, 'metric_scales_independent': True,
        'aggregation': 'Frozen mean of 20 matched resamples; MaxFuse hub runs already averaged within each resample in source family summary.',
    })


if __name__ == '__main__': main()
