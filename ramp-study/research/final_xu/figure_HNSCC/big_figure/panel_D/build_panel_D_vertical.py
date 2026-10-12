#!/usr/bin/env python3
"""Standalone, two-row-spanning HNSCC protein-discovery panel.

Uses the frozen 107-protein screen and donor associations for all six named
candidates. Only presentation changes; the original panel is retained.
"""
from pathlib import Path
import os
import json
import hashlib

HERE = Path(__file__).resolve().parent
SCREEN = HERE.parent.parent / 'biomarker_discovery' / 'screen'
INPUTS = [SCREEN / 'marker_screen_all_107_all_variants.csv',
          SCREEN / 'marker_donor_effects_long.csv']
os.environ.setdefault('MPLCONFIGDIR', str(HERE / '.cache' / 'matplotlib'))
os.environ.setdefault('MPLBACKEND', 'Agg')

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

INK = '#182027'
MUTED = '#65717C'
ORANGE = '#D95F02'
PALE = '#FFF0DF'
GRAY = '#BFC7CD'
TEAL = '#238A8D'
GRID = '#E5E9ED'
DONORS = {'126': '#0072B2', 'indiv_1': '#D55E00', 'indiv_2': '#009E73'}
DONOR_NAMES = {'126': 'Donor 1', 'indiv_1': 'Donor 2', 'indiv_2': 'Donor 3'}
OFFSETS = {'126': .28, 'indiv_1': -.28, 'indiv_2': 0.0}
MARKERS = {'TIGIT--VSTM3': 'TIGIT', 'CD39': 'CD39'}
SECONDARY = {'CD224': 'CD224', 'CD134--OX40': 'OX40',
             'TCR-A--TCR-B': 'TCRαβ', 'CD83-1': 'CD83'}
CONSISTENCY_MARKERS = {**MARKERS, **SECONDARY}
HEIGHT = 6.98


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    before = {str(p): digest(p) for p in INPUTS}
    screen = pd.read_csv(INPUTS[0])
    effects = pd.read_csv(INPUTS[1], dtype={'donor': str})
    final = screen.loc[screen.selected_compact_top5pct].copy()
    donors = effects.loc[effects.variant.eq('paper_clr') &
                         effects.marker_raw.isin(CONSISTENCY_MARKERS)].copy()
    assert len(screen) == 107 and not screen.marker_raw.duplicated().any()
    assert set(final.marker_raw) == set(MARKERS)
    assert int(screen.selected_after_redundancy_pruning.sum()) == 8
    assert len(donors) == 18 and set(donors.donor) == set(DONORS)
    assert not donors.duplicated(['marker_raw', 'donor']).any()
    assert donors.groupby('marker_raw').donor.nunique().eq(3).all()
    assert (donors.association > 0).all()

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'svg.fonttype': 'none'})
    fig = plt.figure(figsize=(4.65, HEIGHT), facecolor='white')
    fig.text(.025, .978, 'D', fontsize=23, weight='bold', color=INK,
             ha='left', va='top')
    fig.text(.145, .966, 'BIOMARKER DISCOVERY', fontsize=13.5,
             weight='bold', color=INK, ha='left', va='top')

    fig.text(.150, .899, 'Top-ranked candidates', fontsize=11.6,
             weight='bold', color=INK, ha='left', va='center')
    fig.text(.150, .868, '107 proteins screened', fontsize=9.3,
             color=MUTED, ha='left', va='center')
    rank_ax = fig.add_axes([.150, .570, .810, .271])
    rank_col = 'robust_median_rank_primary_variants'
    assoc_col = 'paper_clr__mean_donor_association'
    shown = screen.loc[screen[rank_col] <= 10].copy()
    other = shown.loc[~shown.selected_compact_top5pct]
    secondary = other.loc[other.marker_raw.isin(SECONDARY)]
    assert len(secondary) == 4 and secondary.selected_after_redundancy_pruning.all()
    assert not secondary.selected_compact_top5pct.any()
    cutoff = .05 * len(screen)
    rank_ax.axvspan(0, cutoff, color=PALE, zorder=0)
    rank_ax.axvline(cutoff, color=ORANGE, ls=(0, (4, 3)), lw=1.0, zorder=1)
    rank_ax.scatter(other[rank_col], other[assoc_col], s=31,
                    color=GRAY, edgecolors='white', linewidths=.4, zorder=3)
    rank_ax.scatter(secondary[rank_col], secondary[assoc_col], s=41,
                    color=TEAL, edgecolors='white', linewidths=.5, zorder=4)
    rank_ax.scatter(final[rank_col], final[assoc_col], s=66,
                    color=ORANGE, edgecolors='white', linewidths=.65, zorder=5)
    positions = {'TIGIT--VSTM3': (3.70, .226), 'CD39': (4.0, .282)}
    for marker, name in MARKERS.items():
        row = final.loc[final.marker_raw.eq(marker)].iloc[0]
        rank_ax.annotate(name, (row[rank_col], row[assoc_col]),
                         xytext=positions[marker], fontsize=10.3,
                         weight='bold', color=ORANGE, ha='left', va='center',
                         arrowprops={'arrowstyle': '-', 'lw': .85,
                                     'color': ORANGE, 'shrinkA': 2, 'shrinkB': 5},
                         zorder=6)
    # These are existing post-pruning candidates, not additional final selections.
    secondary_positions = {'CD224': (6.08, .245), 'CD134--OX40': (7.30, .202),
                           'TCR-A--TCR-B': (7.85, .110), 'CD83-1': (8.75, .274)}
    for marker, name in SECONDARY.items():
        row = secondary.loc[secondary.marker_raw.eq(marker)].iloc[0]
        rank_ax.annotate(name, (row[rank_col], row[assoc_col]),
                         xytext=secondary_positions[marker], fontsize=9.8,
                         weight='bold', color=TEAL, ha='left', va='center',
                         arrowprops={'arrowstyle': '-', 'lw': .8, 'color': TEAL,
                                     'shrinkA': 2, 'shrinkB': 4}, zorder=6)
    rank_ax.text(.30, .054, 'Top 5%', color=ORANGE, fontsize=9.0,
                 weight='bold', ha='left', va='bottom')
    rank_ax.set(xlim=(0, 10.25), ylim=(.045, .305))
    rank_ax.set_xticks([0, 2, 4, 6, 8, 10])
    rank_ax.set_yticks([.05, .10, .15, .20, .25, .30])
    rank_ax.set_xlabel('Robust protein rank', fontsize=10.2, labelpad=5)
    rank_ax.set_ylabel('Association with niche score', fontsize=10.0, labelpad=6)
    rank_ax.tick_params(labelsize=9.0, length=3, color='#7A838A')
    rank_ax.spines[['top', 'right']].set_visible(False)
    for side in ['left', 'bottom']:
        rank_ax.spines[side].set_color('#7A838A')
        rank_ax.spines[side].set_linewidth(.8)

    # The second plot is positioned for the lower row of the combined figure.
    fig.text(.150, .487, 'Consistency across donors', fontsize=11.6,
             weight='bold', color=INK, ha='left', va='center')
    handles = [Line2D([0], [0], marker='o', linestyle='', color=color,
                      markeredgewidth=0, markersize=5.3, label=DONOR_NAMES[d])
               for d, color in DONORS.items()]
    handles.append(Line2D([0], [0], marker='D', linestyle='', color=INK,
                          markeredgewidth=0, markersize=5.0, label='Mean'))
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.138, .466),
               frameon=False, ncol=2, fontsize=9.0, handletextpad=.35,
               columnspacing=1.1, borderaxespad=0, labelspacing=.45)
    donor_ax = fig.add_axes([.150, .105, .810, .302])
    row_positions = list(reversed(range(len(CONSISTENCY_MARKERS))))
    for (marker, name), y in zip(CONSISTENCY_MARKERS.items(), row_positions):
        block = donors.loc[donors.marker_raw.eq(marker)]
        donor_ax.hlines(y, 0, block.association.max(), color='#CDD4DA',
                        linewidth=1.1, zorder=1)
        for row in block.itertuples(index=False):
            donor_ax.scatter(row.association, y + OFFSETS[row.donor],
                             s=51, color=DONORS[row.donor],
                             edgecolors='white', linewidths=.6, zorder=3)
        donor_ax.scatter(block.association.mean(), y, s=51, marker='D',
                         color=INK, edgecolors='white', linewidths=.6, zorder=4)
    donor_ax.axvline(0, color='#8B9399', ls=(0, (4, 3)), lw=.85)
    donor_ax.set(xlim=(-.012, .355), ylim=(-.55, 5.55))
    # Keep the zero reference and measured positions; emphasize sign consistency.
    donor_ax.set_xticks([0], ['0'])
    donor_ax.set_yticks(row_positions, list(CONSISTENCY_MARKERS.values()))
    for tick, marker in zip(donor_ax.get_yticklabels(), CONSISTENCY_MARKERS):
        tick.set_color(ORANGE if marker in MARKERS else TEAL)
        tick.set_fontweight('bold')
    donor_ax.set_xlabel('Positive association with\ntransferred niche score',
                        fontsize=10.0, labelpad=6)
    donor_ax.tick_params(axis='x', labelsize=9.0, length=3, color='#7A838A')
    donor_ax.tick_params(axis='y', labelsize=10.3, length=0, pad=6)
    donor_ax.grid(False)
    donor_ax.set_axisbelow(True)
    donor_ax.spines[['top', 'right', 'left']].set_visible(False)
    donor_ax.spines['bottom'].set_color('#7A838A')
    donor_ax.spines['bottom'].set_linewidth(.8)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = list(fig.texts)
    for ax in [rank_ax, donor_ax]:
        texts += list(ax.texts) + [ax.xaxis.label, ax.yaxis.label]
        texts += ax.get_xticklabels() + ax.get_yticklabels()
    for text in texts:
        if not text.get_visible() or not text.get_text():
            continue
        bounds = text.get_window_extent(renderer)
        assert bounds.x0 >= 0 and bounds.y0 >= 0, text.get_text()
        assert bounds.x1 <= fig.bbox.width and bounds.y1 <= fig.bbox.height, text.get_text()
    for fmt in ('png', 'pdf', 'svg'):
        out = HERE / f'panel_D_vertical.{fmt}'
        fig.savefig(out, dpi=320, facecolor='white')
        print(out)
    plt.close(fig)
    after = {str(p): digest(p) for p in INPUTS}
    assert before == after, 'An input changed during rendering'
    report = {'input_sha256': before, 'inputs_unchanged': True,
              'screened_proteins': len(screen), 'displayed_rank_limit': 10,
              'displayed_proteins': len(shown), 'compact_rank_cutoff': cutoff,
              'selected_markers': list(MARKERS.values()),
              'secondary_labels': SECONDARY,
              'consistency_markers': CONSISTENCY_MARKERS,
              'consistency_donor_points': len(donors),
              'consistency_mean_points': len(CONSISTENCY_MARKERS),
              'secondary_label_rule': 'The four other post-pruning candidates within the existing displayed rank <=10; final selection unchanged',
              'donor_associations': donors[['marker_raw', 'donor', 'association']].to_dict('records'),
              'donor_axis': {'limits': [-.012, .355], 'visible_ticks': [0],
                             'display': 'Positive association; zero reference and original quantitative point positions retained'},
              'size_inches': [4.65, HEIGHT]}
    (HERE / 'panel_D_vertical_manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    donor_values = donors[['marker_raw', 'donor', 'association']].copy()
    donor_values['marker_label'] = donor_values.marker_raw.map(CONSISTENCY_MARKERS)
    donor_values['donor_label'] = donor_values.donor.map(DONOR_NAMES)
    donor_values['equal_donor_mean'] = donor_values.groupby('marker_raw').association.transform('mean')
    donor_values.to_csv(HERE / 'panel_D_vertical_donor_values.csv', index=False)


if __name__ == '__main__':
    main()
