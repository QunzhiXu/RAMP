"""Compare all 14 frozen marker-holdout tasks on the same 30,000 CODEX cells."""
from runtime import BASE, STORY, save, sha, now
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, str(STORY / 'prediction'))
import masked_marker_model as model_def

OUT = BASE / 'all_donor_marker_review0913'
for directory in [OUT, OUT / 'data', OUT / 'figures']:
    directory.mkdir(parents=True, exist_ok=True)
RAW = STORY / 'sampled_data/bm_codex/source_input_15.csv.gz'
MANIFEST = BASE / 'validation_subpanels/all_marker_validation/COMPLETION.json'
PREVIOUS = BASE / 'validation_subpanels/donor_mean_four_metrics/source_metrics_by_marker_and_donor.csv'
ID = ['cell_id', 'dataset', 'donor', 'unit_key']
METRICS = ['spearman', 'pearson', 'auroc', 'average_precision']
MARKERS = list(model_def.NORMAL_MARKERS)
SOURCES = {}


def track(path):
    SOURCES[str(path)] = sha(path)
    return path


def calculate(truth, prediction, high):
    return {'spearman': float(spearmanr(truth, prediction).statistic),
            'pearson': float(pearsonr(truth, prediction).statistic),
            'auroc': float(roc_auc_score(high, prediction)),
            'average_precision': float(average_precision_score(high, prediction)),
            'n_cells': len(truth), 'n_high': int(high.sum()), 'high_fraction': float(high.mean())}


def evaluate():
    origins = json.loads(track(MANIFEST).read_text())['origins']
    raw = pd.read_csv(track(RAW), usecols=ID + MARKERS)
    previous = pd.read_csv(track(PREVIOUS)).set_index(['marker', 'donor'])
    assert len(raw) == 30000 and raw.cell_id.is_unique
    assert raw.groupby('unit_key').size().eq(2500).all() and raw.unit_key.nunique() == 12
    arrays, donor_rows, summary_rows, discrepancies = {}, [], [], []
    for marker in MARKERS:
        origin = origins[marker]
        root = Path(origin['metrics_path']).parent
        prediction_path = track(root / f'{marker}_prediction.npz')
        assert sha(prediction_path) == origin['prediction_sha256']
        md = pd.read_csv(track(root / 'codex_prediction_metadata.csv.gz'))
        assert raw[ID].astype(str).equals(md[ID].astype(str))
        with np.load(prediction_path) as z:
            prediction = z['prediction']
        # Reproduce the fixed evaluation transformation at its original precision.
        truth = model_def.ramp.kmeanslow_global(raw[[marker]].to_numpy(float),
                md.unit_key.to_numpy(), md.dataset.to_numpy(), [marker])[:, 0]
        assert prediction.shape == truth.shape == (30000,)
        assert np.isfinite(prediction).all() and np.isfinite(truth).all()
        high = np.zeros(len(truth), dtype=bool)
        local_rows = []
        for donor, ix in md.groupby('unit_key', sort=True).indices.items():
            fit = KMeans(n_clusters=2, n_init=20, max_iter=500,
                         random_state=model_def.stable_seed('HIGH', marker, donor)).fit(truth[ix, None])
            high[ix] = fit.labels_ == int(np.argmax(fit.cluster_centers_.ravel()))
            row = {'marker': marker, 'donor': donor, **calculate(truth[ix], prediction[ix], high[ix])}
            old = previous.loc[(marker, donor)]
            delta = max(abs(row[m] - old[m]) for m in METRICS)
            assert delta < 1e-10, (marker, donor, delta)
            discrepancies.append(delta)
            donor_rows.append(row)
            local_rows.append(row)
        within = pd.DataFrame(local_rows)
        summary = {'marker': marker, 'n_donors': 12, **calculate(truth, prediction, high)}
        for metric in METRICS:
            summary[f'mean_donor_{metric}'] = float(within[metric].mean())
            summary[f'min_donor_{metric}'] = float(within[metric].min())
            summary[f'max_donor_{metric}'] = float(within[metric].max())
        summary_rows.append(summary)
        arrays[marker] = {'measured': truth, 'predicted': prediction, 'measured_high': high}
        print(f'{marker}: pooled rho={summary["spearman"]:.4f}; '
              f'donor mean rho={summary["mean_donor_spearman"]:.4f}; all four prior metrics reproduced', flush=True)
    summary = pd.DataFrame(summary_rows).sort_values('spearman', ascending=False).reset_index(drop=True)
    summary.insert(0, 'pooled_spearman_rank', np.arange(1, len(summary) + 1))
    summary.to_csv(OUT / 'data/all_14_markers_pooled_ranking.csv', index=False)
    donors = pd.DataFrame(donor_rows)
    donors.to_csv(OUT / 'data/all_14_markers_by_donor.csv', index=False)
    # Retain both levels of aggregation; do not replace the existing lower four-metric panel.
    raw[ID].to_csv(OUT / 'data/scatter_cell_metadata.csv.gz', index=False)
    np.savez_compressed(OUT / 'data/all_14_markers_scatter_values.npz',
                        **{f'{m}__{field}': value for m, a in arrays.items() for field, value in a.items()})
    comparisons = []
    for marker in MARKERS:
        if marker == 'CD3':
            continue
        a = donors[donors.marker.eq(marker)].set_index('donor')
        b = donors[donors.marker.eq('CD3')].set_index('donor')
        difference = a.spearman - b.spearman
        comparisons.append({'marker': marker, 'comparator': 'CD3',
            'donors_better_spearman': int((difference > 0).sum()),
            'donors_lower_spearman': int((difference < 0).sum()),
            'mean_paired_spearman_difference': float(difference.mean())})
    pd.DataFrame(comparisons).to_csv(OUT / 'data/paired_donor_comparisons_vs_CD3.csv', index=False)
    save(OUT / 'EVALUATION.json', {'time_utc': now(), 'status': 'PASS', 'source_sha256': SOURCES,
        'markers': MARKERS, 'n_cells_per_marker': 30000, 'n_donors': 12, 'cells_per_donor': 2500,
        'no_retraining': True, 'no_new_cell_sampling': True,
        'permanent_marker_holdout': True,
        'same_input_rows_all_markers': True,
        'max_difference_from_168_existing_donor_metric_rows': max(discrepancies),
        'pooled_definition': 'Metric computed once on all 30,000 measured/predicted pairs for this marker',
        'high_state_definition': 'Concatenate the original within-donor measured K=2 KMeans high-state labels',
        'rank_rule': 'Descending pooled Spearman across all 14 completed tasks',
        'no_cell_type_labels_used_for_metrics': True})
    return arrays, summary


def scatter(ax, marker, arrays, summary):
    row = summary.set_index('marker').loc[marker]
    a = arrays[marker]
    points = ax.scatter(a['measured'], a['predicted'], s=1.8, alpha=.18,
                        color='#397CAB', edgecolors='none', rasterized=True)
    for dim, values in [('x', a['measured']), ('y', a['predicted'])]:
        low, high = float(values.min()), float(values.max())
        margin = .045 * (high-low)
        getattr(ax, f'set_{dim}lim')(low-margin, high+margin)
    ax.set_title(marker, loc='left', fontsize=12, fontweight='bold', pad=6)
    ax.set_title(f'Spearman ρ = {row.spearman:.2f}', loc='right', fontsize=11, fontweight='bold', pad=6)
    ax.set_xlabel(f'Measured {marker}', fontsize=10)
    ax.set_ylabel(f'Predicted {marker}', fontsize=10)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.spines[['top', 'right']].set_visible(False)
    for side in ['bottom', 'left']:
        ax.spines[side].set(color='#9BA6AF', linewidth=.7)
    ax.tick_params(labelsize=9, length=2, pad=3)
    assert np.array_equal(np.asarray(points.get_offsets()),
                          np.column_stack([a['measured'], a['predicted']]))


def export(fig, name):
    for ext in ['png', 'svg', 'pdf']:
        fig.savefig(OUT / 'figures' / f'{name}.{ext}', dpi=300, facecolor='white')


def grid(markers, columns, name, arrays, summary):
    rows = int(np.ceil(len(markers)/columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.25*columns, 3.25*rows+.45), squeeze=False)
    fig.subplots_adjust(left=.065, right=.985, bottom=.10 if rows==1 else .06,
                        top=1-.65/(3.25*rows+.45), wspace=.32, hspace=.53)
    fig.suptitle('All 12 CODEX donors · 30,000 cells per marker', fontsize=14,
                 fontweight='bold', y=.975)
    for ax, marker in zip(axes.flat, markers):
        scatter(ax, marker, arrays, summary)
    for ax in list(axes.flat)[len(markers):]:
        ax.axis('off')
    export(fig, name)
    plt.close(fig)


def plot(arrays, summary):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'text.color': '#28343E', 'axes.labelcolor': '#45515B',
                         'xtick.color': '#45515B', 'ytick.color': '#45515B'})
    grid(['CD3', 'CD4', 'CD8'], 3, 'CD3_CD4_CD8_all_donors', arrays, summary)
    grid(['CD3', 'CD4', 'CD8', 'CD45', 'CD117', 'CD11B'], 3,
         'six_marker_examples_all_donors', arrays, summary)
    grid(summary.marker.tolist(), 3, 'all_14_markers_ranked_scatter', arrays, summary)
    # Candidate Panel E panels share the established four-metric summary below.
    import assemble_cd19_layout_v13 as layout
    module = layout.e_module
    candidates = list(dict.fromkeys(['CD3', 'CD4', 'CD8', summary.iloc[0].marker]))
    for marker in candidates:
        frame = pd.read_csv(OUT / 'data/scatter_cell_metadata.csv.gz')
        frame['measured'] = arrays[marker]['measured']
        frame['predicted'] = arrays[marker]['predicted']
        row = summary.set_index('marker').loc[marker]
        module.examples[marker] = {'cells': frame, 'donor': 'all 12 donors', 'short': '',
                                  'rho': row.spearman, 'pearson': row.pearson}
        module.checks[marker] = {}
        fig = plt.figure(figsize=(4.40, 3.60), facecolor='white')
        ax, lower = module.panel_e(fig, (.05, .05, 4.30, 3.50), marker=marker)
        ax.set_title(marker, loc='left', fontsize=10.5, fontweight='bold', pad=5)
        ax.collections[0].set_sizes([1.8])
        ax.collections[0].set_alpha(.18)
        assert len(ax.collections[0].get_offsets()) == 30000
        export(fig, f'Panel_E_{marker}_all_donors')
        plt.close(fig)


if __name__ == '__main__':
    arrays, summary = evaluate()
    plot(arrays, summary)
    assert all(sha(path) == digest for path, digest in SOURCES.items())
    save(OUT / 'PLOT_AUDIT.json', {'status': 'PASS', 'time_utc': now(), 'script_sha256': sha(__file__),
        'all_points_shown': True, 'points_per_scatter': 30000,
        'full_axis_ranges': True, 'point_size': 1.8, 'point_alpha': .18,
        'all_14_marker_results_saved': True, 'bottom_four_metrics_unchanged': True,
        'no_main_figure_replacement': True, 'source_files_unchanged': True})
    print(summary[['marker', 'spearman', 'pearson', 'auroc', 'average_precision',
                   'mean_donor_spearman']].round(4).to_string(index=False), flush=True)
    print(OUT, flush=True)
