"""Compare frozen benchmark UMAPs for figure selection; no fitting or subsampling."""
from pathlib import Path
import os
import json
import hashlib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
os.environ.setdefault('MPLCONFIGDIR', str(HERE / '.cache/matplotlib'))
os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CASES = {'bone_marrow': 'Bone marrow', 'tnbc': 'TNBC',
         'melanoma': 'Melanoma', 'lung_cancer': 'Lung cancer'}
METHODS = {'Unaligned': 'Raw', 'Harmony-direct_source_scale': 'Harmony-direct',
           'cyCombine': 'cyCombine', 'RAMP': 'RAMP'}
COLORS = {
    'CD4 T': '#CC79A7', 'CD8 T': '#D55E00', 'Other T / NKT': '#6F8FC5',
    'Other T / CD8−': '#6F8FC5', 'Mature / CODEX B': '#0072B2',
    'B lineage': '#0072B2', 'B-like': '#0072B2', 'Pre / immature B': '#56B4E9',
    'Plasma cell': '#543C99', 'NK': '#009E73', 'Myeloid': '#D4AC0D',
    'Monocyte': '#D4AC0D', 'Other myeloid': '#8C6D31', 'Dendritic': '#B07AA1',
    'Macrophage': '#7F2704', 'HSPC / progenitor': '#18A999',
    'Promyelocytic progenitors': '#BCBD22', 'Megakaryocytic': '#E78AC3',
    'CODEX erythroid': '#D62728', 'Triana erythroid progenitor': '#FB6A4A',
    'Endothelial': '#17BECF', 'Stroma': '#9467BD', 'Stromal / VSMC': '#9467BD',
    'Tumor': '#B2182B', 'Tumor / epithelial': '#B2182B',
    'Tumor/Epithelial': '#B2182B', 'Epithelial (non-joint)': '#F2C14E',
    'Non-joint': '#BDBDBD', 'Unclassified / technical': '#BDBDBD', 'Mix': '#BDBDBD',
}
DATASET_COLORS = ['#2775B6', '#E58A2B', '#38A169']


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_case(case, sources):
    p = ROOT / f'benchmark_results/{case}/ramp/alignment/input/cell_metadata.csv.gz'
    sources[str(p)] = sha(p)
    meta = pd.read_csv(p, low_memory=False)
    if 'row_id' in meta:
        assert np.array_equal(meta.row_id.to_numpy(), np.arange(len(meta)))
    assert meta.cell_id.is_unique
    assert set(meta.celltype.astype(str)).issubset(COLORS)
    coords = {}
    for method in METHODS:
        p = ROOT / f'benchmark_results/comparison_evaluation/{case}/umap/coordinates/{method}.csv.gz'
        sources[str(p)] = sha(p)
        frame = pd.read_csv(p).sort_values('row_id')
        assert np.array_equal(frame.row_id.to_numpy(), np.arange(len(meta)))
        xy = frame[['UMAP1', 'UMAP2']].to_numpy()
        assert np.isfinite(xy).all()
        coords[method] = xy
    return meta, coords


def scatter(ax, meta, xy, color_by):
    if color_by == 'Dataset':
        palette = dict(zip(meta.dataset.unique(), DATASET_COLORS))
        values = meta.dataset.astype(str)
    else:
        palette = COLORS
        values = meta.celltype.astype(str)
    colors = np.asarray([palette[v] for v in values])
    order = np.random.default_rng(42).permutation(len(meta))
    ax.scatter(xy[order, 0], xy[order, 1], c=colors[order], s=.48,
               alpha=.68, edgecolors='none', rasterized=True)
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    margin = (hi - lo) * .025
    ax.set_xlim(lo[0]-margin[0], hi[0]+margin[0])
    ax.set_ylim(lo[1]-margin[1], hi[1]+margin[1])
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()


def legend(ax, labels, colors, ncol, fontsize=8):
    handles = [Line2D([], [], color=c, marker='o', linestyle='', markersize=4,
                      markeredgewidth=0, label=l) for l, c in zip(labels, colors)]
    ax.legend(handles=handles, loc='center', ncol=ncol, frameon=False,
              fontsize=fontsize, columnspacing=1.1, handletextpad=.3)
    ax.set_axis_off()


def raw_ramp_overview(loaded):
    fig = plt.figure(figsize=(16, 14), facecolor='white')
    grid = fig.add_gridspec(4, 4, left=.012, right=.995, bottom=.07, top=.965,
                           hspace=.52, wspace=.07)
    for i, (case, title) in enumerate(CASES.items()):
        meta, coords = loaded[case]
        axes = []
        for j, (method, color_by) in enumerate([('Unaligned','Dataset'),('RAMP','Dataset'),
                                               ('Unaligned','Cell type'),('RAMP','Cell type')]):
            ax = fig.add_subplot(grid[i,j]); axes.append(ax)
            scatter(ax, meta, coords[method], color_by)
            ax.set_title(f'{title} | {METHODS[method]}\n{color_by}', fontsize=10.5, weight='bold', pad=5)
        y = grid[i,0].get_position(fig).y0
        la = fig.add_axes([.01, y-.055, .98, .039])
        labels = meta.celltype.astype(str).value_counts().index.tolist()
        legend(la, labels, [COLORS[l] for l in labels], min(9, len(labels)), 7.5)
    fig.savefig(HERE / 'raw_ramp_four_benchmarks.png', dpi=180)
    plt.close(fig)


def method_sheets(loaded):
    for case, (meta, coords) in loaded.items():
        fig = plt.figure(figsize=(9, 12), facecolor='white')
        fig.suptitle(f'{CASES[case]} | {len(meta):,} cells | frozen common-protocol UMAP',
                     fontsize=13, weight='bold', y=.99)
        grid = fig.add_gridspec(4, 2, top=.94, bottom=.14, left=.02, right=.98,
                               hspace=.16, wspace=.05)
        for i, (method, title) in enumerate(METHODS.items()):
            for j, color_by in enumerate(['Dataset', 'Cell type']):
                ax = fig.add_subplot(grid[i,j])
                scatter(ax, meta, coords[method], color_by)
                ax.set_title(f'{title} | {color_by}', fontsize=10, weight='bold')
        labels = meta.celltype.astype(str).value_counts().index.tolist()
        la = fig.add_axes([.01, .01, .98, .092])
        legend(la, labels, [COLORS[l] for l in labels], 4, 8)
        da = fig.add_axes([.01, .10, .98, .025])
        legend(da, list(meta.dataset.unique()), DATASET_COLORS, 3, 8)
        fig.savefig(HERE / f'{case}_four_methods.png', dpi=180)
        plt.close(fig)


def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans'})
    sources = {}
    loaded = {case: load_case(case, sources) for case in CASES}
    raw_ramp_overview(loaded)
    method_sheets(loaded)
    assert all(sha(Path(p)) == value for p, value in sources.items())
    report = {'input_sha256': sources, 'inputs_unchanged': True,
              'cells': {case: len(meta) for case,(meta,_) in loaded.items()},
              'alignment_refit': False, 'umap_refit': False, 'display_subsampling': False,
              'point_size': .48, 'alpha': .68, 'random_draw_order_seed': 42,
              'coloring': 'Frozen original cell-type labels; explicit review palette, no label merging',
              'methods': METHODS}
    (HERE / 'review_manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(HERE / 'raw_ramp_four_benchmarks.png')


if __name__ == '__main__':
    main()
