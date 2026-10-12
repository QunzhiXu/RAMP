"""Adopt the all-donor CD8 scatter while retaining all-marker donor summaries."""
from runtime import BASE, save, sha, now
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import assemble_cd19_layout_v13 as previous

OUT = BASE / 'figure_layout_v14'
for directory in [OUT, OUT / 'panels', OUT / 'data']:
    directory.mkdir(parents=True, exist_ok=True)
REVIEW = BASE / 'all_donor_marker_review0913'
v, v8 = previous.v, previous.v8
e_module = previous.e_module
frame = pd.read_csv(REVIEW / 'data/scatter_cell_metadata.csv.gz')
with np.load(REVIEW / 'data/all_14_markers_scatter_values.npz') as z:
    frame['measured'] = z['CD8__measured']
    frame['predicted'] = z['CD8__predicted']
assert len(frame) == 30000 and frame.cell_id.is_unique
assert frame.groupby('unit_key').size().eq(2500).all()
rho = float(spearmanr(frame.measured, frame.predicted).statistic)
pearson = float(pearsonr(frame.measured, frame.predicted).statistic)
row = pd.read_csv(REVIEW / 'data/all_14_markers_pooled_ranking.csv').set_index('marker').loc['CD8']
assert abs(rho - row.spearman) < 1e-12
full_metrics = pd.read_csv(BASE / 'validation_subpanels/donor_mean_four_metrics/source_metrics_by_marker_and_donor.csv')
bottom = pd.read_csv(BASE / 'validation_subpanels/donor_mean_four_metrics/donor_mean_metrics.csv').set_index('donor')
metric_columns = ['spearman', 'pearson', 'auroc', 'average_precision']
assert len(full_metrics) == 168 and full_metrics.marker.nunique() == 14
assert full_metrics.groupby('donor').marker.nunique().eq(14).all()
recalculated = full_metrics.groupby('donor')[metric_columns].mean()
assert np.allclose(recalculated, bottom.loc[recalculated.index, metric_columns], atol=1e-12, rtol=0)


def panel_e(fig, rect, letter='E'):
    e_module.examples['CD8'] = {'cells': frame, 'donor': 'all 12 donors', 'short': '',
                               'rho': rho, 'pearson': pearson}
    e_module.checks['CD8'] = {}
    ax, lower = e_module.panel_e(fig, rect, letter=letter, marker='CD8')
    ax.set_title('CD8', loc='left', fontsize=10.5, fontweight='bold', pad=5)
    ax.collections[0].set_sizes([1.8])
    ax.collections[0].set_alpha(.18)
    assert np.array_equal(np.asarray(ax.collections[0].get_offsets()), frame[['measured', 'predicted']].to_numpy())
    assert sum(len(c.get_offsets()) for c in lower.collections) == 48
    return ax, lower


def export(fig, name):
    for ext in ['png', 'svg', 'pdf']:
        fig.savefig(OUT / f'{name}.{ext}', dpi=300, facecolor='white')


def main():
    plt.rcParams['pdf.fonttype'] = 42
    g = previous.previous.panel_g
    panels = [
        (v.panel_a, (.15, .13, 2.50, 2.55)),
        (v8.v6.panel_b, (2.76, .13, 5.70, 2.55)),
        (v8.v3.panel_c, (8.57, .13, 5.78, 2.55)),
        (v8.v3.panel_d, (.15, 2.79, 5.45, 3.50)),
        (panel_e, (5.695, 2.79, 4.30, 3.50)),
        (v8.v7.panel_f, (10.09, 2.79, 4.26, 3.50)),
        (g, (.15, 6.40, 6.72, 3.95)),
        (v8.v3.panel_h, (7.00, 6.40, 7.35, 3.95)),
    ]
    fig = plt.figure(figsize=(14.5, 10.5), facecolor='white')
    for draw, rect in panels:
        draw(fig, rect)
    export(fig, 'cd19_story_three_row_v14')
    fig.savefig(OUT / 'second_row_preview.png', dpi=300, facecolor='white',
                bbox_inches=Bbox.from_extents(.10, 10.5-6.34, 14.42, 10.5-2.74))
    plt.close(fig)
    for draw, name, size in [(panel_e, 'E_CD8_all_donors', (4.40, 3.60)),
                             (g, 'G_continuous_matched_color_upper_1.5', (6.82, 4.05))]:
        fig = plt.figure(figsize=size, facecolor='white')
        draw(fig, (.05, .05, size[0]-.1, size[1]-.1))
        export(fig, f'panels/{name}')
        plt.close(fig)
    frame.to_csv(OUT / 'data/CD8_scatter_cells.csv.gz', index=False)
    bottom.to_csv(OUT / 'data/donor_mean_all_14_markers.csv')
    full_metrics.to_csv(OUT / 'data/all_14_markers_by_donor.csv', index=False)
    evaluation = json.loads((REVIEW / 'EVALUATION.json').read_text())
    prior = json.loads((BASE / 'figure_layout_v13/LAYOUT_AUDIT.json').read_text())
    sources = {**prior['source_sha256'], **evaluation['source_sha256']}
    for p in [REVIEW / 'data/all_14_markers_scatter_values.npz',
              REVIEW / 'data/scatter_cell_metadata.csv.gz',
              REVIEW / 'data/all_14_markers_pooled_ranking.csv']:
        sources[str(p)] = sha(p)
    assert all(sha(p) == digest for p, digest in sources.items())
    save(OUT / 'LAYOUT_AUDIT.json', {
        'status': 'PASS', 'time_utc': now(), 'script_sha256': sha(__file__), 'source_sha256': sources,
        'main_scatter_marker': 'CD8', 'n_scatter_cells': 30000, 'n_scatter_donors': 12,
        'scatter_spearman': rho, 'scatter_pearson': pearson,
        'donor_code_absent_from_plot': True, 'all_scatter_points_shown': True,
        'lower_summary': {'n_markers': 14, 'n_donors': 12, 'points_per_metric': 12,
            'marker_names': sorted(full_metrics.marker.unique().tolist()),
            'aggregation': 'Unweighted arithmetic mean of all 14 within-donor marker scores',
            'original_values_reproduced': True, 'markers_selected_by_performance': False},
        'D_marker_unchanged': 'CD3', 'G_unchanged_from_v12': True,
        'G_display_bounds': [.5, 1.5], 'all_input_files_unchanged': True,
        'no_model_or_UMAP_refit': True})
    print(f'PASS: CD8 pooled rho={rho:.10f}; lower panel retains all 14 markers for each of 12 donors.')
    print(OUT / 'cd19_story_three_row_v14.png')


if __name__ == '__main__':
    main()
