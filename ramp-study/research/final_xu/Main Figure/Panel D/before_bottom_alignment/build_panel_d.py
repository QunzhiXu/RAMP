"""Two reviewed biological contrasts on frozen marker outputs."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from figure_style import *
import numpy as np
import pandas as pd
from scipy.stats import rankdata, mannwhitneyu
from matplotlib.patches import Patch

DISPLAY_METHODS = ['Raw', 'Harmony-direct', 'cyCombine', 'CytoNorm', 'RAMP']
EXAMPLES = [
    {'case': 'tnbc', 'dataset': 'Meyer IMC', 'unit': 'Meyer IMC|B12.64192', 'marker': 'CD8',
     'target': 'CD8 T', 'other': 'CD4 T', 'other_label': 'CD4 T',
     'title': 'CD8 | Meyer IMC · B12.64192', 'drivers': ['Harmony-direct', 'cyCombine']},
    {'case': 'bone_marrow', 'dataset': 'BM CyTOF', 'unit': 'BM CyTOF|H1', 'marker': 'CD4',
     'target': 'CD4 T', 'other': 'Mature / CODEX B', 'other_label': 'B cells',
     'title': 'CD4 | Bone marrow · CyTOF H1', 'drivers': ['CytoNorm']},
]
COLORS = ['#297AA4', '#D36C58']


def load(example):
    case, marker = example['case'], example['marker']
    base = ROOT / 'benchmark_results' / case
    meta = pd.read_csv(track(base / 'ramp/alignment/input/cell_metadata.csv.gz'), low_memory=False)
    markers = track(base / 'ramp/alignment/input/markers.txt').read_text().splitlines()
    assert meta.cell_id.is_unique
    matrices = {}
    for label, file in [
        ('Raw', 'ramp/alignment/input/raw.npy'),
        ('Harmony-direct', 'harmony/source_scale/alignment/data/harmony_adjusted_marker_coordinates.csv.gz'),
        ('cyCombine', 'cycombine/alignment/data/corrected_expression.csv.gz'),
        ('CytoNorm', 'cytonorm/alignment/data/corrected_expression.csv.gz'),
        ('RAMP', 'ramp/alignment/data/ramp_aligned.npy'),
    ]:
        path = track(base / file)
        if path.suffix == '.npy':
            values = np.load(path, mmap_mode='r')[:, markers.index(marker)]
        else:
            table = pd.read_csv(path, usecols=['cell_id', marker], dtype={'cell_id': str})
            assert table.cell_id.is_unique and set(table.cell_id) == set(meta.cell_id.astype(str))
            values = table.set_index('cell_id').loc[meta.cell_id.astype(str), marker].to_numpy()
        assert len(values) == len(meta) and np.isfinite(values).all()
        matrices[label] = values
    mask = meta.dataset.eq(example['dataset']).to_numpy()
    if example['unit'] is not None: mask &= meta.unit_key.eq(example['unit']).to_numpy()
    assert mask.any(), f'No cells matched dataset/unit: {example["dataset"]!r} / {example["unit"]!r}'
    return meta.loc[mask].reset_index(drop=True), {key: val[mask] for key, val in matrices.items()}


def main():
    setup()
    fig = new_panel('D', 'Artifacts in comparison methods')
    stats, comparisons = [], []
    for row, example in enumerate(EXAMPLES):
        meta, outputs = load(example)
        target = meta.celltype.eq(example['target']).to_numpy()
        other = meta.celltype.eq(example['other']).to_numpy()
        assert target.sum() >= 50 and other.sum() >= 50
        ax = fig.add_axes([.093, .53-row*.43, .887, .295])
        for method in example['drivers']:
            driver = DISPLAY_METHODS.index(method)
            ax.axvspan(driver-.45, driver+.45, facecolor='#FBEAE4', zorder=0)
        for x, method in enumerate(DISPLAY_METHODS):
            values = outputs[method]
            ranked = rankdata(values, method='average')/(len(values)+1)
            arrays = [ranked[target], ranked[other]]
            artist = ax.boxplot(arrays, positions=[x-.16, x+.16], widths=.24,
                                patch_artist=True, showfliers=False, manage_ticks=False,
                                medianprops={'color': '#20252A', 'linewidth': .9},
                                whiskerprops={'color': '#4D555A', 'linewidth': .65},
                                capprops={'color': '#4D555A', 'linewidth': .65},
                                boxprops={'edgecolor': '#4D555A', 'linewidth': .65})
            for patch, color in zip(artist['boxes'], COLORS): patch.set_facecolor(color)
            for label, vals in zip([example['target'], example['other']], arrays):
                q1, median, q3 = np.quantile(vals, [.25, .5, .75])
                stats.append({'case': example['case'], 'dataset': example['dataset'],
                              'unit_key': example['unit'], 'marker': example['marker'],
                              'method': method, 'celltype': label, 'n_cells': len(vals),
                              'scope_cells': len(meta), 'mean_percentile': float(vals.mean()),
                              'q1': q1, 'median': median, 'q3': q3})
            auc = mannwhitneyu(values[target], values[other], alternative='greater').statistic/(target.sum()*other.sum())
            comparisons.append({'case': example['case'], 'scope': example['unit'] or example['dataset'],
                                'marker': example['marker'], 'method': method,
                                'target_vs_other_auc': float(auc)})
        ax.set_title(example['title'], loc='left', fontsize=10.5, weight='bold', pad=6)
        ax.legend(handles=[Patch(facecolor=COLORS[0], label=example['target']),
                           Patch(facecolor=COLORS[1], label=example['other_label'])],
                  loc='lower right', bbox_to_anchor=(1, 1.017), ncol=2, frameon=False,
                  fontsize=8.7, borderaxespad=0, handlelength=.9, handletextpad=.35,
                  columnspacing=.9)
        ax.set_xlim(-.5, 4.5); ax.set_ylim(0, 1)
        ax.set_xticks(range(5), DISPLAY_METHODS)
        ax.tick_params(axis='x', labelsize=9, length=0, pad=4)
        for label, name in zip(ax.get_xticklabels(), DISPLAY_METHODS):
            if name in example['drivers']: label.set_color('#A64736'); label.set_weight('bold')
            if name == 'RAMP': label.set_weight('bold')
        ax.set_yticks([0, .5, 1]); ax.tick_params(axis='y', labelsize=8.6, length=2)
        ax.set_ylabel(f'{example["marker"]} percentile', fontsize=9.5, weight='bold')
        ax.grid(axis='y', color='#E7E7E7', linewidth=.55, zorder=0)
        ax.set_axisbelow(True)
        for spine in ax.spines.values(): spine.set_color('#8C8C8C')
    pd.DataFrame(stats).to_csv(HERE / 'artifact_box_statistics.csv', index=False)
    pd.DataFrame(comparisons).to_csv(HERE / 'artifact_auc.csv', index=False)
    save(fig, HERE, 'panel_d_marker_artifacts', {
        'panel': 'D', 'examples': EXAMPLES, 'methods': DISPLAY_METHODS,
        'percentile': 'average_rank/(n+1), separately in each output using ALL cells in the stated dataset/unit; no positivity gate',
        'plot': 'Median/IQR boxes, 1.5 IQR whiskers; outlier points omitted; selected driver columns lightly shaded',
        'selection': 'Selected current sample examples: Meyer IMC B12.64192 CD8 T/CD4 T for Harmony-direct and cyCombine; BM CyTOF H1 CD4 T/B cells for CytoNorm. These are selected samples, not pooled patient results.',
        'Meyer_CD8_note': 'Cell-type specificity loss, not complete high/low inversion. All five current Meyer samples show lower CD8 T/CD4 T AUC under cyCombine; see cycombine_review for the all-sample plot.',
        'H1_CD4_note': 'Reviewed specificity loss; not a hit under the earlier strict automatic AUC <=0.40 or negative percentile >=0.60 rule.',
    })


if __name__ == '__main__': main()
