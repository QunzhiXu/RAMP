#!/usr/bin/env python3
"""One-axis protein/RNA comparison from the frozen donor-wise 30% groups."""
from pathlib import Path
import hashlib
import json
import os
import shutil

HERE = Path(__file__).resolve().parent
TRIAL = HERE.parent
BIG = HERE.parents[1]
SCAN = BIG / 'panel_E/group_fraction_scan_20260910'
os.environ.setdefault('MPLCONFIGDIR', str(TRIAL / '.cache/matplotlib'))
os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

WIDTH, HEIGHT = 3.60, 3.45
INK, GRID = '#182027', '#E7EBEE'
LOW, HIGH = '#2B6CB0', '#C9433C'
DONORS = ['126', 'indiv_1', 'indiv_2']
FEATURES = [
    ('ADT', 'CD223--LAG-3', 'LAG-3'),
    ('ADT', 'CD127--IL-7Ra', 'CD127'),
    ('ADT', 'CD62L', 'CD62L'),
    ('RNA', 'CTLA4', 'CTLA4'),
    ('RNA', 'CXCL13', 'CXCL13'),
    ('RNA', 'IL7R', 'IL7R'),
]
INPUTS = {name: SCAN / f'{name}.csv' for name in
          ['donor_contrasts', 'equal_donor_contrasts', 'group_counts']}


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    hashes = {str(p): sha(p) for p in INPUTS.values()}
    backup = TRIAL / 'backup_before_polish/panel_E'
    backup.mkdir(parents=True, exist_ok=True)
    for name in ['panel_E_molecular_30pct_two_columns.png',
                 'panel_E_molecular_30pct_two_columns.pdf',
                 'panel_E_molecular_30pct_two_columns.svg',
                 'panel_E_molecular_30pct_two_columns_values.csv', 'README.md']:
        original, target = BIG / 'panel_E' / name, backup / name
        if target.exists():
            assert sha(target) == sha(original)
        else:
            shutil.copy2(original, target)
    donors = pd.read_csv(INPUTS['donor_contrasts'], dtype={'donor': str})
    donors = donors.loc[donors.fraction.eq(.30)].copy()
    summary = pd.read_csv(INPUTS['equal_donor_contrasts'])
    summary = summary.loc[summary.fraction.eq(.30)].set_index(['modality', 'marker'])
    counts = pd.read_csv(INPUTS['group_counts'], dtype={'donor': str})
    counts = counts.loc[counts.fraction.eq(.30)].copy()
    assert len(donors) == 18 and set(donors.donor) == set(DONORS)
    assert counts.n_low.sum() == 1439 and counts.n_high.sum() == 1437
    assert counts.n_total.sum() == 4788
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'svg.fonttype': 'none', 'text.color': INK})
    fig = plt.figure(figsize=(WIDTH, HEIGHT), facecolor='white')
    fig.text(.025, .975, 'E', fontsize=23, weight='bold', va='top', ha='left')
    fig.text(.140, .948, 'CD39/TIGIT state phenotype', fontsize=10.8,
             weight='bold', va='top', ha='left')
    fig.legend(handles=[Line2D([], [], color=color, marker='o', markersize=4.2,
                              linestyle='none', label=label)
                        for color, label in [(LOW, 'Lower 30%'), (HIGH, 'Higher 30%')]],
               loc='center', bbox_to_anchor=(.575, .850), ncol=2, frameon=False,
               fontsize=9.1, handletextpad=.40, handlelength=.7, columnspacing=1.05)
    ax = fig.add_axes([.163, .170, .813, .565])
    centers = np.array([0, 1, 2, 3.35, 4.35, 5.35])
    offsets = np.array([-.095, 0., .095])
    records, scatter_artists, box_records = [], [], []
    for center, (modality, marker, label) in zip(centers, FEATURES):
        block = donors.loc[donors.modality.eq(modality) & donors.marker.eq(marker)].set_index('donor').loc[DONORS]
        assert len(block) == 3 and block.direction_consistent.all()
        expected = summary.loc[(modality, marker)]
        np.testing.assert_allclose(block[['low', 'high']].mean().to_numpy(),
                                   expected[['low', 'high']].to_numpy(float), atol=1e-12, rtol=0)
        for group, displacement, color in [('low', -.19, LOW), ('high', .19, HIGH)]:
            values = block[group].to_numpy(float)
            assert np.isfinite(values).all() and ((0 <= values) & (values <= 100)).all()
            x = center + displacement
            artists = ax.boxplot([values], positions=[x], widths=.29, patch_artist=True,
                manage_ticks=False, showfliers=False, whis=(0, 100),
                boxprops={'facecolor': color, 'alpha': .18, 'edgecolor': color, 'linewidth': .9},
                medianprops={'color': color, 'linewidth': 1.2},
                whiskerprops={'color': color, 'linewidth': .8},
                capprops={'color': color, 'linewidth': .8})
            points = ax.scatter(x+offsets, values, s=11, facecolor=color,
                                edgecolor='white', linewidth=.35, zorder=4)
            scatter_artists.append((points, values))
            box_records.append({'modality': modality, 'marker': marker, 'group': group,
                'minimum': float(values.min()), 'q1': float(np.quantile(values, .25)),
                'median': float(np.median(values)), 'q3': float(np.quantile(values, .75)),
                'maximum': float(values.max()), 'arithmetic_mean': float(values.mean())})
            unit = 'mean expression percentile' if modality == 'ADT' else 'RNA detected (%)'
            for donor, value in zip(DONORS, values):
                records.append({'modality': modality, 'marker': marker, 'label': label,
                                'donor': donor, 'group': group, 'value': float(value), 'unit': unit})
    ax.set(xlim=(-.65, 6.0), ylim=(0, 100), yticks=[0, 25, 50, 75, 100])
    ax.set_xticks(centers, [f[2] for f in FEATURES])
    ax.tick_params(axis='x', labelsize=8.0, length=0, pad=7)
    ax.tick_params(axis='y', labelsize=8.0, length=2.5, pad=3)
    for tick, (modality, _, _) in zip(ax.get_xticklabels(), FEATURES):
        tick.set_fontstyle('italic' if modality == 'RNA' else 'normal')
    ax.set_ylabel('Percentile / detected (%)', fontsize=9.0, labelpad=5)
    ax.yaxis.grid(True, color=GRID, linewidth=.6)
    ax.set_axisbelow(True)
    ax.axvline(2.675, color='#D2D9DE', linewidth=.7, zorder=0)
    ax.text(1., 103.5, 'Protein', ha='center', va='bottom', fontsize=9.8,
            fontweight='bold', clip_on=False)
    ax.text(4.35, 103.5, 'RNA', ha='center', va='bottom', fontsize=9.8,
            fontweight='bold', clip_on=False)
    ax.spines[['top', 'right']].set_visible(False)
    for side in ['left', 'bottom']:
        ax.spines[side].set_color('#7A838A')
        ax.spines[side].set_linewidth(.8)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [*fig.texts, *ax.texts, *ax.get_xticklabels(), *ax.get_yticklabels(), ax.yaxis.label]
    texts += [t for legend in fig.legends for t in legend.get_texts()]
    for text in texts:
        bb = text.get_window_extent(renderer)
        assert bb.x0 >= -.1 and bb.y0 >= -.1 and bb.x1 <= fig.bbox.width+.1 and bb.y1 <= fig.bbox.height+.1, text.get_text()
    xboxes = [t.get_window_extent(renderer) for t in ax.get_xticklabels()]
    assert all(a.x1+1 < b.x0 for a, b in zip(xboxes, xboxes[1:])), 'Marker labels overlap'
    assert sum(len(p.get_offsets()) for p, _ in scatter_artists) == 36
    for points, values in scatter_artists:
        np.testing.assert_array_equal(points.get_offsets()[:, 1], values)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(HERE / f'panel_E_grouped_boxes.{ext}', dpi=320, facecolor='white')
    plt.close(fig)
    pd.DataFrame(records).to_csv(HERE / 'plotted_donor_values.csv', index=False)
    pd.DataFrame(box_records).to_csv(HERE / 'boxplot_summaries.csv', index=False)
    counts.to_csv(HERE / 'group_counts.csv', index=False)
    assert all(sha(Path(p)) == h for p, h in hashes.items())
    audit = {'status': 'PASS', 'input_sha256': hashes, 'inputs_unchanged': True,
        'figure_inches': [WIDTH, HEIGHT], 'n_markers': 6, 'n_boxes': 12,
        'n_donors_per_box': 3, 'n_donor_points': 36,
        'group_fraction': .30, 'lower_cells': 1439, 'higher_cells': 1437,
        'protein_quantity': 'Within-donor group mean expression percentile (0-100)',
        'rna_quantity': 'Within-donor group RNA detection percentage (raw.X > 0)',
        'box_center': 'Median across three donor summaries',
        'box': '25th to 75th percentile, linear interpolation',
        'whiskers': 'Full minimum-to-maximum range of three donors',
        'original_equal_donor_means_verified': True, 'no_refitting_or_new_group_selection': True,
        'colors': {'lower': LOW, 'higher': HIGH}}
    (HERE / 'PLOT_AUDIT.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps({'status': 'PASS', 'boxes': 12, 'donor_points': 36, 'width_inches': WIDTH}))
    print(HERE / 'panel_E_grouped_boxes.png')


if __name__ == '__main__':
    main()
