"""Panel E: a measured/predicted scatter above the existing donor summary."""
from runtime import BASE, STORY, save, sha, now
import json
import re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox
import assemble_cd19_layout_v8 as previous

v = previous.v
OUT = BASE / 'figure_layout_v9'
for directory in [OUT, OUT / 'panels', OUT / 'data']:
    directory.mkdir(parents=True, exist_ok=True)

METRICS = ['spearman', 'pearson', 'auroc', 'average_precision']
COLORS = ['#397CAB', '#5A9993', '#A27BA9', '#CF9856']
SCORES_PATH = BASE / 'validation_subpanels/donor_mean_four_metrics/source_metrics_by_marker_and_donor.csv'
scores = pd.read_csv(SCORES_PATH)
new_sources = {str(SCORES_PATH): sha(SCORES_PATH)}
examples = {}
checks = {}


def track(path):
    new_sources[str(path)] = sha(path)
    return path


def prepare_examples():
    manifest_path = track(STORY / 'alignment/output/without_cd19/manifest.json')
    manifest = json.loads(manifest_path.read_text())
    baseline_path = track(STORY / 'alignment/output/without_cd19/kmeanslow_baseline.npy')
    assert sha(baseline_path) == manifest['outputs']['kmeanslow_baseline.npy']
    aligned_md = pd.read_csv(track(STORY / 'alignment/output/cell_metadata.csv.gz'))
    codex = aligned_md.dataset.eq('BM CODEX').to_numpy()
    baseline = np.load(baseline_path, mmap_mode='r')
    for marker in ['CD3', 'CD8']:
        if marker == 'CD3':
            frame = v.cells[['cell_id', 'unit_key', 'measured_CD3', 'predicted_CD3']].copy()
            frame = frame.rename(columns={'measured_CD3': 'measured', 'predicted_CD3': 'predicted'})
        else:
            root = BASE / 'validation_subpanels/all_marker_validation' / marker
            md = pd.read_csv(track(root / 'codex_prediction_metadata.csv.gz'))
            npz_path = track(root / f'{marker}_prediction.npz')
            completion = json.loads(track(BASE / 'validation_subpanels/all_marker_validation/COMPLETION.json').read_text())
            assert sha(npz_path) == completion['origins'][marker]['prediction_sha256']
            with np.load(npz_path) as data:
                pred = data['prediction']
            assert aligned_md.loc[codex, ['cell_id', 'unit_key']].reset_index(drop=True).equals(md[['cell_id', 'unit_key']])
            frame = md[['cell_id', 'unit_key']].copy()
            frame['measured'] = baseline[codex, manifest['markers'].index(marker)].astype(float)
            frame['predicted'] = pred
        assert len(frame) == 30000 and frame.cell_id.is_unique
        evaluated = []
        for donor, group in frame.groupby('unit_key'):
            row = scores[(scores.marker == marker) & (scores.donor == donor)].iloc[0]
            rho = float(spearmanr(group.measured, group.predicted).statistic)
            pearson = float(pearsonr(group.measured, group.predicted).statistic)
            assert abs(rho - row.spearman) < 1e-10
            # The saved baseline is float32; the original validation used float64.
            assert abs(pearson - row.pearson) < 1e-8
            evaluated.append({'marker': marker, 'donor': donor, 'spearman': rho,
                              'pearson': pearson, 'n_cells': len(group)})
        ranking = pd.DataFrame(evaluated).sort_values(['spearman', 'donor'], ascending=[False, True])
        ranking.to_csv(OUT / 'data' / f'{marker}_donor_ranking.csv', index=False)
        best = ranking.iloc[0]
        chosen = frame[frame.unit_key.eq(best.donor)].copy()
        assert len(chosen) == 2500
        chosen.to_csv(OUT / 'data' / f'{marker}_scatter_cells.csv.gz', index=False)
        short = re.search(r'_(H\d+)_', best.donor).group(1)
        examples[marker] = {'cells': chosen, 'donor': best.donor, 'short': short,
                            'rho': float(best.spearman), 'pearson': float(best.pearson)}
        checks[marker] = {
            'marker': marker, 'selected_donor': best.donor, 'donor_short_label': short,
            'spearman': float(best.spearman), 'pearson': float(best.pearson),
            'n_cells': len(chosen), 'n_donors_considered': len(ranking),
            'selection': 'Highest existing within-donor Spearman for this requested example marker',
            'all_selected_donor_cells_shown': True, 'expression_scale': 'KMEANSLOW',
            'prediction_protocol': 'Frozen permanent-marker-holdout neural network',
            'all_12_existing_donor_correlations_reproduced': True,
        }


def panel_e(fig, rect, letter='E', marker='CD3'):
    x, y, w, h = rect
    v.title(fig, rect, letter, 'Prediction Accuracy')
    example = examples[marker]
    data = example['cells']
    ax = v.ax_at(fig, x + .58, y + .62, w - .70, 1.04)
    ax.scatter(data.measured, data.predicted, s=3.6, color=COLORS[0], alpha=.38,
               edgecolors='none', rasterized=True)
    for dim, values in [('x', data.measured), ('y', data.predicted)]:
        low, high = float(values.min()), float(values.max())
        margin = .045 * (high - low)
        getattr(ax, f'set_{dim}lim')(low - margin, high + margin)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.set_xlabel(f'Measured {marker}', fontsize=9.4, labelpad=2)
    ax.set_ylabel(f'Predicted {marker}', fontsize=9.4, labelpad=3)
    ax.tick_params(labelsize=8.5, length=2, pad=2)
    ax.spines[['top', 'right']].set_visible(False)
    for side in ['bottom', 'left']:
        ax.spines[side].set(color='#9BA6AF', linewidth=.7)
    ax.set_title(f'{marker} · {example["short"]}', loc='left', fontsize=10.5, fontweight='bold', pad=5)
    ax.set_title(f'Spearman ρ = {example["rho"]:.2f}', loc='right', fontsize=10.3, fontweight='bold', pad=5)

    lower = v.ax_at(fig, x + .58, y + 2.23, w - .70, .82)
    lower.set_title('14 markers · 12 donors', loc='center', fontsize=10.5, fontweight='bold', pad=6)
    values = [v.donor_scores[m].to_numpy() for m in METRICS]
    bp = lower.boxplot(values, positions=np.arange(4), widths=.48, patch_artist=True,
                       whis=1.5, showfliers=False,
                       medianprops={'color': '#2D3942', 'linewidth': 1.35},
                       whiskerprops={'linewidth': .9}, capprops={'linewidth': .9})
    rng = np.random.default_rng(42)
    for i, (color, group) in enumerate(zip(COLORS, values)):
        bp['boxes'][i].set(facecolor=to_rgba(color, .17), edgecolor=color, linewidth=1.05)
        for item in bp['whiskers'][2*i:2*i+2] + bp['caps'][2*i:2*i+2]:
            item.set_color(color)
        lower.scatter(i + rng.uniform(-.145, .145, len(group)), group, s=17,
                      color=color, edgecolor='white', linewidth=.45, alpha=.95, zorder=4)
    lower.set(xlim=(-.55, 3.55), ylim=(.4, .9))
    lower.set_xticks(range(4), ['Spearman ρ', 'Pearson r', 'AUROC', 'AUPRC\n(AP)'],
                    fontsize=8.8, fontweight='bold')
    lower.set_yticks([.4, .6, .8, .9])
    lower.tick_params(axis='x', length=0, pad=5)
    lower.tick_params(axis='y', length=0, labelsize=8.5, pad=3)
    lower.yaxis.grid(True, color='#E8EBEE', linewidth=.6)
    lower.set_axisbelow(True)
    lower.spines[['top', 'right', 'left']].set_visible(False)
    lower.spines['bottom'].set(color='#B9C2C9', linewidth=.7)
    lower.set_ylabel('Mean score', fontsize=9.4, labelpad=6)
    assert all(len(collection.get_offsets()) == 12 for collection in lower.collections)
    assert sum(len(collection.get_offsets()) for collection in lower.collections) == 48
    assert all(np.all((group >= .4) & (group <= .9)) for group in values)
    checks[marker]['scatter_axis_limits'] = {'x': list(ax.get_xlim()), 'y': list(ax.get_ylim())}
    checks[marker]['scatter_point_count'] = len(ax.collections[0].get_offsets())
    return ax, lower


def export(fig, name):
    for extension in ['png', 'svg', 'pdf']:
        fig.savefig(OUT / f'{name}.{extension}', dpi=300, facecolor='white')


def main():
    prepare_examples()
    plt.rcParams['pdf.fonttype'] = 42
    panels = [
        (v.panel_a, (.15, .13, 2.50, 2.55)),
        (previous.v6.panel_b, (2.76, .13, 5.70, 2.55)),
        (previous.v3.panel_c, (8.57, .13, 5.78, 2.55)),
        (previous.v3.panel_d, (.15, 2.79, 5.45, 3.50)),
        (panel_e, (5.695, 2.79, 4.30, 3.50)),
        (previous.v7.panel_f, (10.09, 2.79, 4.26, 3.50)),
        (previous.panel_g, (.15, 6.40, 6.72, 3.95)),
        (previous.v3.panel_h, (7.00, 6.40, 7.35, 3.95)),
    ]
    prior_path = BASE / 'figure_layout_v8/LAYOUT_AUDIT.json'
    prior = json.loads(track(prior_path).read_text())
    rectangles = {draw.__name__: list(rect) for draw, rect in panels}
    assert rectangles == prior['panel_rectangles_inches_from_top_left']
    fig = plt.figure(figsize=(14.5, 10.5), facecolor='white')
    for draw, rect in panels:
        draw(fig, rect)
    export(fig, 'cd19_story_three_row_v9')
    fig.savefig(OUT / 'second_row_preview.png', dpi=300, facecolor='white',
                bbox_inches=Bbox.from_extents(.10, 10.5-6.34, 14.42, 10.5-2.74))
    plt.close(fig)
    for marker in ['CD3', 'CD8']:
        fig = plt.figure(figsize=(4.40, 3.60), facecolor='white')
        panel_e(fig, (.05, .05, 4.30, 3.50), marker=marker)
        export(fig, f'panels/E_scatter_and_accuracy_{marker}')
        plt.close(fig)
    fig = plt.figure(figsize=(9.0, 3.6), facecolor='white')
    panel_e(fig, (.05, .05, 4.3, 3.5), marker='CD3')
    panel_e(fig, (4.65, .05, 4.3, 3.5), marker='CD8')
    export(fig, 'panels/E_CD3_vs_CD8_options')
    plt.close(fig)
    source_hashes = {**v.input_hashes, **previous.metric_hashes, **new_sources}
    assert all(sha(path) == value for path, value in source_hashes.items())
    v.donor_scores.to_csv(OUT / 'data/donor_mean_metrics_unchanged.csv', index=False)
    save(OUT / 'LAYOUT_AUDIT.json', {
        'time_utc': now(), 'script_sha256': sha(__file__), 'source_sha256': source_hashes,
        'canvas_inches': [14.5, 10.5], 'panel_rectangles_inches_from_top_left': rectangles,
        'changed_panel': 'E', 'main_scatter_marker': 'CD3', 'alternative_marker': 'CD8',
        'examples': checks,
        'lower_summary': {'same_values_boxes_and_jitter_as_v8': True, 'n_markers': 14,
                          'n_donors': 12, 'points_per_metric': 12, 'n_points': 48,
                          'aggregation': 'Equal mean of 14 marker scores per donor',
                          'y_limits': [.4, .9], 'axis_height_inches': .82},
        'remaining_panels_match_v8': True, 'no_model_or_UMAP_refit': True,
        'all_input_files_unchanged': True,
        'G_version': 'Existing v8 scatter retained in this Panel-E layout test; real-DAPI assets remain separate',
    })
    print(json.dumps(checks, indent=2))
    print(OUT / 'panels/E_scatter_and_accuracy_CD3.png')


if __name__ == '__main__':
    main()
