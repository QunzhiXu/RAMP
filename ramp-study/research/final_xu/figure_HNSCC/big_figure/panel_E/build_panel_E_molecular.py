#!/usr/bin/env python3
"""Compact protein/RNA phenotype panel for the revised HNSCC story.

Reads fixed measured TIGIT/CD39 lower/upper donor quintile results. Protein
expression percentiles and RNA detection percentages are separate endpoints.
All displayed values are equally weighted means across the three donors.
"""
from pathlib import Path
import os
import json
import hashlib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / 'story_test_hnscc/ending_exploration_20260910'
SUMMARY = SOURCE / 'individual_marker_equal_donor_contrasts.csv'
DONOR = SOURCE / 'individual_marker_donor_contrasts.csv'
os.environ.setdefault('MPLCONFIGDIR', str(HERE / '.cache' / 'matplotlib'))
os.environ.setdefault('MPLBACKEND', 'Agg')

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

INK = '#182027'
MUTED = '#65717C'
LOW = '#2775B6'
HIGH = '#D95F02'
GRID = '#E7EBEE'
CONNECTOR = '#BFC8CF'
PROTEINS = [('CD137', 'CD137'), ('CD103', 'CD103'),
            ('CD223--LAG-3', 'LAG-3'), ('CD244--2B4', 'CD244'),
            ('CD28-1', 'CD28'), ('CD127--IL-7Ra', 'CD127'), ('CD62L', 'CD62L')]
GENES = [(g, g) for g in ['CXCL13', 'CTLA4', 'TNFRSF9', 'LAG3', 'LAYN', 'GZMB', 'IL7R']]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    hashes = {str(p): sha(p) for p in [SUMMARY, DONOR]}
    summary = pd.read_csv(SUMMARY).set_index(['modality', 'marker'])
    donor = pd.read_csv(DONOR, dtype={'donor': str})
    selected = []
    for modality, markers, columns in [
        ('ADT', PROTEINS, ['low_mean_percentile', 'high_mean_percentile']),
        ('RNA', GENES, ['low_detection_fraction', 'high_detection_fraction']),
    ]:
        for marker, label in markers:
            row = summary.loc[(modality, marker)]
            block = donor.loc[donor.modality.eq(modality) & donor.marker.eq(marker)]
            assert len(block) == 3 and set(block.donor) == {'126', 'indiv_1', 'indiv_2'}
            assert np.isfinite(row[columns].astype(float)).all()
            assert np.allclose(block[columns].mean().values, row[columns].values.astype(float),
                               atol=1e-12, rtol=0)
            low, high = [float(row[c]) * 100 for c in columns]
            assert 0 <= low <= 100 and 0 <= high <= 100
            selected.append({'modality': modality, 'marker': marker, 'display_name': label,
                             'lower': low, 'higher': high, 'donors': 3,
                             'unit': 'mean_expression_percentile' if modality == 'ADT'
                                     else 'RNA_detection_percent'})
    table = pd.DataFrame(selected)

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'svg.fonttype': 'none'})
    fig = plt.figure(figsize=(4.65, 3.45), facecolor='white')
    fig.text(.025, .976, 'E', fontsize=23, weight='bold', ha='left', va='top', color=INK)
    fig.text(.145, .950, 'CD8 MOLECULAR PHENOTYPE', fontsize=12.1,
             weight='bold', ha='left', va='top', color=INK)
    fig.text(.53, .823, 'Measured CD39/TIGIT state', fontsize=9.8,
             ha='center', va='center', color=INK)
    fig.legend(handles=[Line2D([0], [0], marker='o', linestyle='', color=color,
                               markeredgewidth=0, markersize=5.3, label=name)
                        for name, color in [('Lower', LOW), ('Higher', HIGH)]],
               loc='center', bbox_to_anchor=(.53, .750), ncol=2,
               frameon=False, fontsize=9.5, columnspacing=1.4,
               handlelength=.8, handletextpad=.40)

    axes = [fig.add_axes([.16, .183, .320, .452]),
            fig.add_axes([.705, .183, .270, .452])]
    for ax, modality, title, xlabel in zip(
        axes, ['ADT', 'RNA'], ['Protein', 'RNA'],
        ['Mean expression\npercentile', 'RNA detected (%)']
    ):
        rows = table.loc[table.modality.eq(modality)]
        ypos = np.arange(len(rows))[::-1]
        for y, row in zip(ypos, rows.itertuples(index=False)):
            ax.plot([row.lower, row.higher], [y, y], color=CONNECTOR,
                    lw=1.35, solid_capstyle='round', zorder=2)
            ax.scatter(row.lower, y, s=29, color=LOW, edgecolors='white',
                       linewidths=.45, zorder=4)
            ax.scatter(row.higher, y, s=29, color=HIGH, edgecolors='white',
                       linewidths=.45, zorder=4)
        # Pad the bounded 0-100 scale so low-detection dots are fully visible.
        ax.set(xlim=(-4, 104), ylim=(-.55, len(rows)-.45))
        ax.set_xticks([0, 50, 100])
        ax.set_yticks(ypos, rows.display_name.tolist())
        ax.tick_params(axis='y', length=0, pad=4, labelsize=8.7)
        ax.tick_params(axis='x', length=2.5, pad=3, labelsize=8.4, color='#7A838A')
        ax.set_title(title, fontsize=10.5, weight='bold', color=INK, pad=8)
        ax.set_xlabel(xlabel, fontsize=9.0, labelpad=5)
        ax.grid(axis='x', color=GRID, lw=.65)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.spines['bottom'].set_color('#7A838A')
        ax.spines['bottom'].set_linewidth(.7)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = list(fig.texts)
    for ax in axes:
        texts += [ax.title, ax.xaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()
    for text in texts:
        bounds = text.get_window_extent(renderer)
        assert bounds.x0 >= 0 and bounds.y0 >= 0, text.get_text()
        assert bounds.x1 <= fig.bbox.width and bounds.y1 <= fig.bbox.height, text.get_text()
    for fmt in ['png', 'pdf', 'svg']:
        path = HERE / f'panel_E_molecular.{fmt}'
        fig.savefig(path, dpi=320, facecolor='white')
        print(path)
    plt.close(fig)
    assert hashes == {str(p): sha(p) for p in [SUMMARY, DONOR]}
    table.to_csv(HERE / 'panel_E_molecular_values.csv', index=False)
    metadata = {
        'inputs_sha256': hashes, 'inputs_unchanged': True,
        'group_definition': 'Within each donor, measured TIGIT/CD39 midrank <=0.2 versus >=0.8',
        'aggregation': 'Equal weight for each of the three donor means',
        'protein_measure': 'Within-donor expression percentile of pair-free CLR ADT, multiplied by 100',
        'rna_measure': 'Within-group fraction with measured RNA raw.X >0, multiplied by 100',
        'population': '4788 Rahim tumour CD8 T cells; groups are quintile subsets',
        'protein_rows': len(PROTEINS), 'rna_rows': len(GENES),
        'marker_scope': 'The 7 protein and 7 RNA contrasts discussed in the existing ending README; not all 107 screening candidates',
        'size_inches': [4.65, 3.45],
    }
    (HERE / 'panel_E_molecular_manifest.json').write_text(json.dumps(metadata, indent=2)+'\n')


if __name__ == '__main__':
    main()
