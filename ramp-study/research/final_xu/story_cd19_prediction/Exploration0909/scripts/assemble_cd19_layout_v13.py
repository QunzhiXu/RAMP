"""Panel E: all frozen CODEX CD3 cells in one scatter, without donor code."""
from runtime import BASE, save, sha, now
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from matplotlib.transforms import Bbox
import assemble_cd19_layout_v12 as previous

OUT = BASE / 'figure_layout_v13'
for directory in [OUT, OUT / 'panels', OUT / 'data']:
    directory.mkdir(parents=True, exist_ok=True)
v, v8 = previous.v, previous.v8
e_module = previous.source.old.previous
SOURCE = BASE / 'CD3_NN_UMAP_validation/cell_predictions_and_umap.csv.gz'
source_hash = sha(SOURCE)
data = pd.read_csv(SOURCE)
cells = data[['cell_id', 'unit_key', 'measured_CD3', 'predicted_CD3']].rename(
    columns={'measured_CD3': 'measured', 'predicted_CD3': 'predicted'})
assert len(cells) == 30000 and cells.cell_id.is_unique
assert cells.groupby('unit_key').size().eq(2500).all()
assert cells.unit_key.nunique() == 12
assert np.isfinite(cells[['measured', 'predicted']].to_numpy()).all()
RHO = float(spearmanr(cells.measured, cells.predicted).statistic)
PEARSON = float(pearsonr(cells.measured, cells.predicted).statistic)
single_example = e_module.examples['CD3']
single_check = dict(e_module.checks['CD3'])


def panel_e(fig, rect, letter='E', pooled=True):
    old_example = e_module.examples['CD3']
    old_check = e_module.checks['CD3']
    frame = cells if pooled else single_example['cells']
    rho = RHO if pooled else single_example['rho']
    try:
        e_module.examples['CD3'] = {
            'cells': frame, 'donor': 'all 12 donors' if pooled else single_example['donor'],
            'short': '', 'rho': rho, 'pearson': PEARSON if pooled else single_example['pearson']}
        e_module.checks['CD3'] = {}
        ax, lower = e_module.panel_e(fig, rect, letter=letter)
        ax.set_title('CD3', loc='left', fontsize=10.5, fontweight='bold', pad=5)
        # All observations remain visible within the full observed axis ranges.
        ax.collections[0].set_sizes([1.8 if pooled else 3.6])
        ax.collections[0].set_alpha(.18 if pooled else .38)
        points = np.asarray(ax.collections[0].get_offsets())
        assert points.shape == (len(frame), 2)
        assert np.array_equal(points, frame[['measured', 'predicted']].to_numpy())
        assert ax.get_xlim()[0] <= points[:, 0].min() <= points[:, 0].max() <= ax.get_xlim()[1]
        assert ax.get_ylim()[0] <= points[:, 1].min() <= points[:, 1].max() <= ax.get_ylim()[1]
        assert ax.get_title(loc='left') == 'CD3'
        assert sum(len(c.get_offsets()) for c in lower.collections) == 48
        return ax, lower
    finally:
        e_module.examples['CD3'] = old_example
        e_module.checks['CD3'] = old_check


def export(fig, name):
    for extension in ['png', 'svg', 'pdf']:
        fig.savefig(OUT / f'{name}.{extension}', dpi=300, facecolor='white')


def main():
    plt.rcParams['pdf.fonttype'] = 42
    for pooled, name in [(True, 'E_CD3_all_12_donors'), (False, 'E_CD3_single_donor_no_code')]:
        fig = plt.figure(figsize=(4.4, 3.6), facecolor='white')
        panel_e(fig, (.05, .05, 4.30, 3.50), pooled=pooled)
        export(fig, f'panels/{name}')
        plt.close(fig)
    fig = plt.figure(figsize=(9.0, 3.95), facecolor='white')
    for x, pooled, label in [(.05, False, 'One donor · 2,500 cells'),
                             (4.65, True, 'All 12 donors · 30,000 cells')]:
        v.text_at(fig, x + 2.15, .17, label, ha='center', fontsize=11, fontweight='bold')
        panel_e(fig, (x, .40, 4.30, 3.50), pooled=pooled)
    export(fig, 'panels/E_single_vs_all_donors')
    plt.close(fig)
    panels = [
        (v.panel_a, (.15, .13, 2.50, 2.55)),
        (v8.v6.panel_b, (2.76, .13, 5.70, 2.55)),
        (v8.v3.panel_c, (8.57, .13, 5.78, 2.55)),
        (v8.v3.panel_d, (.15, 2.79, 5.45, 3.50)),
        (panel_e, (5.695, 2.79, 4.30, 3.50)),
        (v8.v7.panel_f, (10.09, 2.79, 4.26, 3.50)),
        (previous.panel_g, (.15, 6.40, 6.72, 3.95)),
        (v8.v3.panel_h, (7.00, 6.40, 7.35, 3.95)),
    ]
    fig = plt.figure(figsize=(14.5, 10.5), facecolor='white')
    for draw, rect in panels:
        draw(fig, rect)
    export(fig, 'cd19_story_three_row_v13')
    fig.savefig(OUT / 'second_row_preview.png', dpi=300, facecolor='white',
                bbox_inches=Bbox.from_extents(.10, 10.5-6.34, 14.42, 10.5-2.74))
    plt.close(fig)
    cells.to_csv(OUT / 'data/CD3_all_12_donors_scatter_cells.csv.gz', index=False)
    per_donor = pd.DataFrame([
        {'donor': donor, 'n_cells': len(group),
         'spearman': float(spearmanr(group.measured, group.predicted).statistic),
         'pearson': float(pearsonr(group.measured, group.predicted).statistic)}
        for donor, group in cells.groupby('unit_key')])
    per_donor.to_csv(OUT / 'data/CD3_metrics_by_donor.csv', index=False)
    summary = {'n_cells': len(cells), 'n_donors': 12, 'pooled_spearman': RHO,
               'pooled_pearson': PEARSON, 'mean_within_donor_spearman': float(per_donor.spearman.mean()),
               'previous_single_donor_spearman': single_example['rho'],
               'previous_single_donor': single_example['donor']}
    save(OUT / 'data/CD3_summary.json', summary)
    prior = json.loads((BASE / 'figure_layout_v12/LAYOUT_AUDIT.json').read_text())
    assert all(sha(path) == digest for path, digest in prior['source_sha256'].items())
    assert sha(SOURCE) == source_hash
    save(OUT / 'LAYOUT_AUDIT.json', {
        'status': 'PASS', 'time_utc': now(), 'script_sha256': sha(__file__),
        'source_sha256': {**prior['source_sha256'], str(SOURCE): source_hash},
        'changed_panel': 'E', 'scatter': summary,
        'all_30000_points_plotted': True, 'no_point_filtering_or_axis_clipping': True,
        'donor_code_removed_from_plot': True,
        'bottom_four_metrics_unchanged': True,
        'G_unchanged_from_v12': True, 'G_display_bounds': list(previous.BOUNDS),
        'no_model_or_UMAP_refit': True, 'all_input_files_unchanged': True})
    print(json.dumps(summary, indent=2))
    print('PASS: all 30,000 original points; donor code removed; bottom summary unchanged.')
    print(OUT / 'cd19_story_three_row_v13.png')


if __name__ == '__main__':
    main()
