#!/usr/bin/env python3
"""Two-row Panel E: six measured features, donor-wise bottom/top 30%."""
from pathlib import Path
import hashlib
import json
import os

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'group_fraction_scan_20260910'
os.environ.setdefault('MPLCONFIGDIR', str(HERE / '.cache' / 'matplotlib'))
os.environ.setdefault('MPLBACKEND', 'Agg')

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

INPUTS = {name: SOURCE / name for name in ['equal_donor_contrasts.csv', 'donor_contrasts.csv', 'group_counts.csv', 'run_manifest.json']}
FRACTION = .30
ROWS = [
    ('ADT', 'Protein', [('CD223--LAG-3', 'LAG-3'), ('CD127--IL-7Ra', 'CD127'), ('CD62L', 'CD62L')], 'Mean expression percentile'),
    ('RNA', 'RNA', [('CTLA4', 'CTLA4'), ('CXCL13', 'CXCL13'), ('IL7R', 'IL7R')], 'RNA detected (%)'),
]
LOW, HIGH, INK, GRID = '#2775B6', '#D95F02', '#182027', '#E7EBEE'
STEM = 'panel_E_molecular_30pct_two_rows'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    hashes = {name: sha(path) for name, path in INPUTS.items()}
    summary = pd.read_csv(INPUTS['equal_donor_contrasts.csv'])
    summary = summary[summary.fraction.eq(FRACTION)].set_index(['modality', 'marker'])
    donors = pd.read_csv(INPUTS['donor_contrasts.csv'], dtype={'donor': str})
    donors = donors[donors.fraction.eq(FRACTION)]
    counts = pd.read_csv(INPUTS['group_counts.csv'], dtype={'donor': str})
    counts = counts[counts.fraction.eq(FRACTION)]
    assert len(summary) == 6 and len(counts) == 3
    assert set(counts.donor) == {'126', 'indiv_1', 'indiv_2'}
    assert (counts[['n_low', 'n_high', 'n_middle']].sum(axis=1) == counts.n_total).all()
    totals = counts[['n_total', 'n_low', 'n_high', 'n_middle']].sum().astype(int)
    assert totals.to_dict() == {'n_total': 4788, 'n_low': 1439, 'n_high': 1437, 'n_middle': 1912}
    exported = []
    for modality, title, markers, xlabel in ROWS:
        for marker, label in markers:
            s = summary.loc[(modality, marker)]
            b = donors[donors.modality.eq(modality) & donors.marker.eq(marker)]
            assert len(b) == 3 and s.donors_consistent == 3
            np.testing.assert_allclose(b[['low', 'high']].mean(), s[['low', 'high']].astype(float), rtol=0, atol=1e-12)
            assert 0 <= s.low <= 100 and 0 <= s.high <= 100
            exported.append(dict(modality=modality, marker=marker, label=label,
                                 lower=s.low, higher=s.high, delta=s.delta,
                                 unit='mean_expression_percentile' if modality == 'ADT' else 'RNA_detection_percent',
                                 group_fraction=FRACTION, aggregation='equal_mean_of_three_donors'))
    table = pd.DataFrame(exported)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'svg.fonttype': 'none'})
    fig = plt.figure(figsize=(4.65, 3.45), facecolor='white')
    fig.text(.025, .976, 'E', ha='left', va='top', fontsize=23, weight='bold', color=INK)
    fig.text(.145, .947, 'CD39/TIGIT STATE PHENOTYPE', ha='left', va='top',
             fontsize=11.9, weight='bold', color=INK)
    fig.legend(handles=[Line2D([0], [0], marker='o', linestyle='', color=color,
                              markersize=5.5, label=label, markeredgewidth=0)
                        for label, color in [('Lower 30%', LOW), ('Higher 30%', HIGH)]],
               loc='center', bbox_to_anchor=(.57, .849), ncol=2,
               frameon=False, fontsize=9.5, handlelength=.7, handletextpad=.45, columnspacing=1.5)
    axes = []
    for (modality, title, markers, xlabel), bottom in zip(ROWS, [.555, .125]):
        ax = fig.add_axes([.185, bottom, .775, .218])
        axes.append(ax)
        b = table[table.modality.eq(modality)]
        for y, row in zip([2, 1, 0], b.itertuples(index=False)):
            ax.plot([row.lower, row.higher], [y, y], lw=1.65, color='#BFC8CF', solid_capstyle='round', zorder=2)
            ax.scatter(row.lower, y, s=34, color=LOW, edgecolors='white', linewidths=.5, zorder=3)
            ax.scatter(row.higher, y, s=34, color=HIGH, edgecolors='white', linewidths=.5, zorder=3)
        ax.set(xlim=(-3, 103), ylim=(-.55, 2.55), xticks=[0, 25, 50, 75, 100], yticks=[2, 1, 0])
        ax.set_yticklabels([label for marker, label in markers], fontsize=9.7)
        ax.tick_params(axis='y', length=0, pad=5)
        ax.tick_params(axis='x', length=2.5, pad=2.5, labelsize=8.5, color='#7A838A')
        ax.set_title(title, fontsize=10.4, weight='bold', color=INK, loc='left', pad=5)
        ax.set_xlabel(xlabel, fontsize=9, labelpad=3)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.spines['bottom'].set_color('#7A838A')
        ax.spines['bottom'].set_linewidth(.65)
        ax.grid(axis='x', color=GRID, lw=.7)
        ax.set_axisbelow(True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = list(fig.texts)
    for ax in axes:
        texts += [ax._left_title, ax.xaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()
    texts += fig.legends[0].get_texts()
    for item in texts:
        box = item.get_window_extent(renderer)
        assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height, item.get_text()
    for fmt in ['png', 'pdf', 'svg']:
        fig.savefig(HERE / f'{STEM}.{fmt}', dpi=320, facecolor='white')
    plt.close(fig)
    table.to_csv(HERE / f'{STEM}_values.csv', index=False)
    assert hashes == {name: sha(path) for name, path in INPUTS.items()}
    manifest = dict(inputs={name: dict(path=str(INPUTS[name]), sha256=value) for name, value in hashes.items()},
                    inputs_unchanged=True, size_inches=[4.65, 3.45], layout='Top: protein; bottom: RNA',
                    group_definition='Measured CD39/TIGIT joint-score midrank <=0.30 and >=0.70 within each donor',
                    counts=totals.to_dict(), donors=counts.to_dict('records'),
                    markers=table.to_dict('records'),
                    plotted_statistic='Equal-weight mean of the three donor group means',
                    inferential_testing='None; descriptive molecular features',
                    original_panel_and_combined_figure_unchanged=True)
    (HERE / f'{STEM}_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(table.to_string(index=False))
    print(HERE / f'{STEM}.pdf')


if __name__ == '__main__':
    main()
