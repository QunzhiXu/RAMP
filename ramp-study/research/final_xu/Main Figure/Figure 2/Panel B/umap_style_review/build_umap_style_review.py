"""Compare lung UMAP rendering styles using the same frozen cells and coordinates."""
from pathlib import Path
import os
import sys
import json

HERE = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(HERE/'.cache/matplotlib'))
os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, str(HERE.parent))
import build_lung_cancer as base
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ORIGINAL_SCATTER = base.draw_scatter
ORIGINAL_ORDER = list(base.ORDER)
ORIGINAL_PALETTE = dict(base.PALETTE)
ORDER = ['Mix', 'B-like', 'Myeloid', 'Tumor/Epithelial', 'Endothelial', 'CD4 T', 'CD8 T']
PALETTE = {**ORIGINAL_PALETTE, 'Mix': '#CCD0D5'}
STYLE = {
    'scatter_dataset_size': .20, 'scatter_dataset_alpha': .90,
    'scatter_celltype_size': .38, 'scatter_celltype_alpha': .95,
    'mix_size': .18, 'mix_alpha': .32,
    'coordinate_margin_fraction': .025,
}


def clean_scatter(ax, meta, xy, color_by):
    if color_by == 'Dataset':
        palette = dict(zip(meta.dataset.unique(), base.DATASET_COLORS))
        colors = np.asarray([palette[label] for label in meta.dataset])
        order = np.random.default_rng(42).permutation(len(meta))
        ax.scatter(xy[order, 0], xy[order, 1], c=colors[order],
                   s=STYLE['scatter_dataset_size'], alpha=STYLE['scatter_dataset_alpha'],
                   edgecolors='none', rasterized=True)
    else:
        labels = meta.celltype.to_numpy()
        for z, label in enumerate(ORDER):
            idx = np.random.default_rng(42).permutation(np.flatnonzero(labels == label))
            size = STYLE['mix_size'] if label == 'Mix' else STYLE['scatter_celltype_size']
            alpha = STYLE['mix_alpha'] if label == 'Mix' else STYLE['scatter_celltype_alpha']
            ax.scatter(xy[idx, 0], xy[idx, 1], c=PALETTE[label], s=size, alpha=alpha,
                       edgecolors='none', rasterized=True, zorder=z+1)
    low, high = xy.min(axis=0), xy.max(axis=0)
    margin = (high-low)*STYLE['coordinate_margin_fraction']
    ax.set_xlim(low[0]-margin[0], high[0]+margin[0])
    ax.set_ylim(low[1]-margin[1], high[1]+margin[1])
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()


def comparison(meta, coords):
    # Same physical size as the full figure, with the same frozen coordinate extent.
    reference = json.loads((base.MAIN/'Panel B/tnbc_bandwidth_review/adtnorm_style_review/manifest.json').read_text())
    row_height = reference['layout']['umap_axes'][0]['height_in']
    slot_width = reference['layout']['umap_axes'][0]['frame_width_in']
    width = .38 + 2*slot_width + .14 + .14
    height = .76 + 2*row_height + .31 + .56
    fig = plt.figure(figsize=(width, height), dpi=210, facecolor='white')
    fig.text(.10/width, 1-.20/height, 'Lung cancer | RAMP', fontsize=12.5, weight='bold')
    geometry = {}
    for col, (title, func) in enumerate([('Current', ORIGINAL_SCATTER), ('Refined', clean_scatter)]):
        left = .38 + col*(slot_width+.14)
        fig.text((left+slot_width/2)/width, 1-.55/height, title,
                 ha='center', fontsize=11.5, weight='bold')
        for row, color_by in enumerate(['Dataset', 'Cell type']):
            bottom = height-.76-(row+1)*row_height-row*.31
            xy = coords['RAMP']
            aspect = np.ptp(xy[:, 0])/np.ptp(xy[:, 1])
            actual_height = min(row_height, slot_width/aspect)
            actual_width = actual_height*aspect
            ax = fig.add_axes([(left+(slot_width-actual_width)/2)/width,
                               (bottom+(row_height-actual_height)/2)/height,
                               actual_width/width, actual_height/height])
            func(ax, meta, xy, color_by)
            geometry[f'{title}_{color_by}'] = [list(ax.get_xlim()), list(ax.get_ylim())]
            fig.add_artist(Rectangle((left/width, bottom/height), slot_width/width,
                                     row_height/height, transform=fig.transFigure,
                                     fill=False, edgecolor=base.FRAME, linewidth=.45))
            if col == 0:
                fig.text(.15/width, (bottom+row_height/2)/height, color_by,
                         rotation=90, ha='center', va='center', fontsize=10.5, weight='bold')
    labels = meta.celltype.value_counts().index.tolist()
    ax = fig.add_axes([.02, .035/height, .96, .42/height])
    base.legend(ax, labels, [PALETTE[label] for label in labels], 3, 7.3)
    for key in ['Dataset', 'Cell type']:
        assert geometry['Current_'+key] == geometry['Refined_'+key]
    fig.savefig(HERE/'lung_umap_current_vs_refined.png', dpi=210)
    fig.savefig(HERE/'lung_umap_current_vs_refined.svg', dpi=210)
    plt.close(fig)


def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none'})
    sources = {}
    meta, coords = base.load_case('lung_cancer', sources)
    comparison(meta, coords)
    # Reuse the full panel composition; all new output is isolated in this folder.
    base.HERE = HERE
    base.draw_scatter = clean_scatter
    base.ORDER = ORDER
    base.PALETTE = PALETTE
    base.main()
    for suffix in ['.png', '.svg']:
        (HERE/('lung_cancer_ridge_umap_figure1_style'+suffix)).rename(
            HERE/('lung_cancer_ridge_umap_refined'+suffix))
    manifest_path = HERE/'manifest.json'
    report = json.loads(manifest_path.read_text())
    report.update(STYLE)
    report['display_variant'] = 'Mix at bottom; green B-like below yellow Myeloid; stronger color; finer dataset dots'
    report['reference_panel'] = str(HERE.parent/'lung_cancer_ridge_umap_figure1_style.png')
    report['celltype_draw_order_before'] = ORIGINAL_ORDER
    report['celltype_palette_before'] = ORIGINAL_PALETTE
    report['mix_cell_count'] = int(meta.celltype.eq('Mix').sum())
    report['mix_fraction'] = float(meta.celltype.eq('Mix').mean())
    report['source_panel_preserved'] = True
    manifest_path.write_text(json.dumps(report, indent=2)+'\n')
    assert all(base.sha(Path(path)) == digest for path, digest in sources.items())
    print(HERE/'lung_umap_current_vs_refined.png')
    print(HERE/'lung_cancer_ridge_umap_refined.png')


if __name__ == '__main__':
    main()
