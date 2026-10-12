"""Inspect CD4/CD8 correspondence on the existing full three-platform RAMP UMAP."""
from runtime import BASE
import hashlib
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PLATFORMS = {'Senosain CyTOF': 'CyTOF', 'Zheng mIF': 'mIF', 'Cords IMC': 'IMC'}
COLORS = {'CyTOF': '#487c9d', 'mIF': '#368f83', 'IMC': '#c35b75'}
LINEAGE_COLORS = {'CD4 T': '#3978b5', 'CD8 T': '#e17c39'}
MAPPING = {
    ('Senosain CyTOF', 'CD4 T'): 'CD4 T',
    ('Senosain CyTOF', 'CD8 T'): 'CD8 T',
    ('Cords IMC', 'CD4'): 'CD4 T',
    ('Cords IMC', 'CD8'): 'CD8 T',
    ('Zheng mIF', 'Th'): 'CD4 T',
    ('Zheng mIF', 'Treg'): 'CD4 T',
    ('Zheng mIF', 'Tc'): 'CD8 T',
    ('Zheng mIF', 'DNT'): 'DNT',
}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'svg.fonttype': 'none', 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.labelcolor': '#43515a',
                     'axes.edgecolor': '#9ba8b0', 'axes.linewidth': .7,
                     'savefig.facecolor': 'white'})

md = pd.read_csv(BASE/'data/CELL_METADATA.csv.gz', low_memory=False)
u = np.load(BASE/'data/RAMP_UMAP.npy')
assert u.shape == (84000, 2) and md.cell_key.is_unique and len(md) == len(u)
assert np.isfinite(u).all()
assert md.groupby(['dataset', 'patient_id']).size().eq(1000).all()
md['platform'] = md.dataset.map(PLATFORMS)
md['T_lineage'] = [MAPPING.get((ds, cell), 'Other cells')
                   for ds, cell in zip(md.dataset, md.provider_cell_type)]
assert md.loc[md.broad_type.eq('T cell'), 'T_lineage'].ne('Other cells').all()
assert md.T_lineage.value_counts()['CD4 T'] == 6214
assert md.T_lineage.value_counts()['CD8 T'] == 5922
assert md.T_lineage.value_counts()['DNT'] == 587

mapping = pd.DataFrame([{'dataset': k[0], 'provider_cell_type': k[1],
                         'display_lineage': v} for k, v in MAPPING.items()])
mapping.to_csv(BASE/'data/T_CELL_MAPPING.csv', index=False)
counts = md[md.T_lineage.ne('Other cells')].groupby(
    ['dataset', 'platform', 'T_lineage']).size().rename('n_cells').reset_index()
counts.to_csv(BASE/'results/CELL_COUNTS.csv', index=False)
export = md[['cell_key', 'dataset', 'platform', 'patient_id',
             'provider_cell_type', 'broad_type', 'T_lineage']].copy()
export[['RAMP_UMAP1', 'RAMP_UMAP2']] = u
export.to_csv(BASE/'data/CELL_COORDINATES_AND_LINEAGE.csv.gz', index=False,
              compression={'method': 'gzip', 'compresslevel': 3, 'mtime': 0})

lo, hi = u.min(axis=0), u.max(axis=0)
pad = (hi-lo)*.035
limits = [[float(lo[i]-pad[i]), float(hi[i]+pad[i])] for i in range(2)]


def clean(ax, title):
    ax.set(xlim=limits[0], ylim=limits[1], xlabel='UMAP 1', ylabel='UMAP 2')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title, fontsize=12.5, fontweight='bold', pad=10)


def background(ax):
    ax.scatter(u[:, 0], u[:, 1], c='#cbd0d5', s=.45, alpha=.13,
               linewidths=0, rasterized=True)


def highlight(ax, mask, values, palette, size=2.2):
    ix = np.random.default_rng(42).permutation(np.flatnonzero(mask))
    colors = pd.Series(values).map(palette).to_numpy()[ix]
    assert not pd.isna(colors).any()
    ax.scatter(u[ix, 0], u[ix, 1], c=colors, s=size, alpha=.74,
               linewidths=0, rasterized=True)


def legend_handles(palette):
    return [Line2D([], [], marker='o', linestyle='', markersize=5.5,
                   color=color, label=label) for label, color in palette.items()]


def save(fig, name):
    fig.savefig(BASE/f'{name}.png', dpi=300)
    fig.savefig(BASE/f'{name}.svg', dpi=300)
    plt.close(fig)


fig, axes = plt.subplots(1, 3, figsize=(10.8, 4.15))
fig.subplots_adjust(left=.055, right=.99, bottom=.22, top=.80, wspace=.19)
for ax in axes: background(ax)
highlight(axes[0], md.T_lineage.isin(LINEAGE_COLORS), md.T_lineage,
          LINEAGE_COLORS, size=1.8)
for ax, lineage in zip(axes[1:], ['CD4 T', 'CD8 T']):
    highlight(ax, md.T_lineage.eq(lineage), md.platform, COLORS)
for ax, title in zip(axes, ['CD4 vs CD8', 'CD4 T · by platform', 'CD8 T · by platform']):
    clean(ax, title)
fig.suptitle('CD4 / CD8 T-cell alignment', x=.035, y=.985, ha='left',
             fontsize=17, fontweight='bold', color='#243745')
fig.legend(handles=legend_handles(LINEAGE_COLORS), loc='lower center',
           bbox_to_anchor=(.205, .045), ncol=2, frameon=False, fontsize=10,
           handletextpad=.35, columnspacing=1.0)
fig.legend(handles=legend_handles(COLORS), loc='lower center',
           bbox_to_anchor=(.674, .045), ncol=3, frameon=False, fontsize=10,
           handletextpad=.35, columnspacing=1.4)
save(fig, 'CD4_CD8_alignment')

fig, axes = plt.subplots(2, 3, figsize=(10.8, 7.0))
fig.subplots_adjust(left=.10, right=.985, bottom=.09, top=.86, wspace=.18, hspace=.24)
for row, lineage in enumerate(['CD4 T', 'CD8 T']):
    for col, platform in enumerate(['CyTOF', 'mIF', 'IMC']):
        ax = axes[row, col]
        mask = md.T_lineage.eq(lineage) & md.platform.eq(platform)
        background(ax)
        highlight(ax, mask, md.platform, COLORS, size=2.7)
        clean(ax, platform if row == 0 else '')
        if row == 0: ax.title.set_color(COLORS[platform])
        ax.text(.03, .96, f'n = {int(mask.sum()):,}', transform=ax.transAxes,
                ha='left', va='top', fontsize=10, color='#43515a')
        if col == 0:
            ax.text(-.20, .5, lineage, transform=ax.transAxes, rotation=90,
                    ha='center', va='center', fontsize=13, fontweight='bold')
fig.suptitle('CD4 / CD8 distribution in each platform', x=.035, y=.965,
             ha='left', fontsize=17, fontweight='bold', color='#243745')
save(fig, 'CD4_CD8_platform_facets')

source_records = json.loads((BASE/'provenance/SOURCES.json').read_text())
for item in source_records:
    assert hashlib.sha256((BASE/item['local']).read_bytes()).hexdigest() == item['sha256']
    from pathlib import Path
    assert hashlib.sha256(Path(item['source']).read_bytes()).hexdigest() == item['sha256']

report = {'status': 'PASS', 'n_all_cells': len(md),
          'lineage_counts': {k: int(v) for k, v in md.T_lineage.value_counts().items()},
          'n_cells_per_platform': md.platform.value_counts().to_dict(),
          'coordinate_refit': False, 'spatial_coordinates_used': False,
          'same_existing_RAMP_coordinates_in_every_view': True,
          'display_limits': limits, 'Zheng_CD4_mapping': ['Th', 'Treg'],
          'Zheng_CD8_mapping': ['Tc'], 'Zheng_DNT_separate': True,
          'source_files_unchanged': True, 'new_expression_gates_applied': False,
          'point_order_seed': 42,
          'outputs': ['CD4_CD8_alignment.png', 'CD4_CD8_alignment.svg',
                      'CD4_CD8_platform_facets.png', 'CD4_CD8_platform_facets.svg']}
(BASE/'provenance/VERIFICATION.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(report, ensure_ascii=False), flush=True)
