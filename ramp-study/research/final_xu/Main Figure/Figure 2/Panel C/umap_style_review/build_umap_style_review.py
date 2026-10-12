"""Polish melanoma UMAP display without changing frozen data or geometry."""
from pathlib import Path
import os
import sys
import json

HERE = Path(__file__).resolve().parent
HERE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(HERE/'.cache/matplotlib'))
os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, str(HERE.parent))
import build_melanoma as base
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ORIGINAL_SCATTER = base.draw_scatter
ORIGINAL_ORDER = list(base.ORDER)
PALETTE = dict(base.PALETTE)
ORDER = ['Non-joint', 'Epithelial (non-joint)', 'Stroma', 'Tumor', 'Myeloid',
         'Endothelial', 'Other T / CD8−', 'NK', 'CD4 T', 'CD8 T']
VARIANTS = {
    'layered': {'size': .38, 'alpha': .95},
    'fine': {'size': .28, 'alpha': .95},
}
SELECTED = 'layered'


def refined_scatter(ax, meta, xy, color_by, variant=None):
    if color_by == 'Dataset':
        ORIGINAL_SCATTER(ax, meta, xy, color_by)
        return
    style = VARIANTS[variant or SELECTED]
    labels = meta.celltype.to_numpy()
    for z, label in enumerate(ORDER):
        idx = np.random.default_rng(42).permutation(np.flatnonzero(labels == label))
        size, alpha = (.18, .32) if label in base.BACKGROUND_LABELS else (style['size'], style['alpha'])
        ax.scatter(xy[idx, 0], xy[idx, 1], c=PALETTE[label], s=size, alpha=alpha,
                   edgecolors='none', rasterized=True, zorder=z+1)
    low, high = xy.min(axis=0), xy.max(axis=0)
    margin = (high-low)*.025
    ax.set_xlim(low[0]-margin[0], high[0]+margin[0])
    ax.set_ylim(low[1]-margin[1], high[1]+margin[1])
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()


def comparison(meta, coords, methods, variants, output):
    reference = json.loads((base.MAIN/'Panel B/tnbc_bandwidth_review/adtnorm_style_review/manifest.json').read_text())
    row_height = reference['layout']['umap_axes'][0]['height_in']
    slot_width = reference['layout']['umap_axes'][0]['frame_width_in']
    left_margin, gap, header, footer = .46, .12, .67, .64
    if len(variants) == 2:
        footer = .95
    width = left_margin+len(variants)*slot_width+(len(variants)-1)*gap+.12
    height = header+len(methods)*row_height+(len(methods)-1)*.14+footer
    fig = plt.figure(figsize=(width, height), dpi=210, facecolor='white')
    fig.text(.10/width, 1-.21/height, 'Melanoma | cell type', fontsize=12.5, weight='bold')
    title_map = {'current': 'Current', 'layered': 'Reordered layers', 'fine': 'Finer dots'}
    geometry = {}
    for col, variant in enumerate(variants):
        left = left_margin+col*(slot_width+gap)
        fig.text((left+slot_width/2)/width, 1-.49/height, title_map[variant],
                 fontsize=10.5, weight='bold', ha='center')
        for row, method in enumerate(methods):
            bottom = height-header-(row+1)*row_height-row*.14
            xy = coords[method]
            aspect = np.ptp(xy[:, 0])/np.ptp(xy[:, 1])
            actual_height = min(row_height, slot_width/aspect)
            actual_width = actual_height*aspect
            ax = fig.add_axes([(left+(slot_width-actual_width)/2)/width,
                               (bottom+(row_height-actual_height)/2)/height,
                               actual_width/width, actual_height/height])
            if variant == 'current':
                ORIGINAL_SCATTER(ax, meta, xy, 'Cell type')
            else:
                refined_scatter(ax, meta, xy, 'Cell type', variant)
            geometry[(method, variant)] = [list(ax.get_xlim()), list(ax.get_ylim())]
            fig.add_artist(Rectangle((left/width, bottom/height), slot_width/width,
                                     row_height/height, fill=False, edgecolor=base.FRAME,
                                     linewidth=.45, transform=fig.transFigure))
            if col == 0:
                fig.text(.21/width, (bottom+row_height/2)/height, base.METHODS[method],
                         rotation=90, ha='center', va='center', fontsize=10.5, weight='bold')
    labels = meta.celltype.value_counts().index.tolist()
    legend_height = .47 if len(variants) == 3 else .75
    ax = fig.add_axes([.02, .06/height, .96, legend_height/height])
    base.legend(ax, labels, [PALETTE[label] for label in labels], 5 if len(variants)==3 else 2, 7.2)
    for method in methods:
        for variant in variants:
            assert geometry[(method, variant)] == geometry[(method, variants[0])]
    fig.canvas.draw()
    legend = ax.get_legend().get_window_extent(fig.canvas.get_renderer())
    assert legend.x0>=0 and legend.x1<=fig.bbox.width and legend.y0>=0
    for suffix in ['.png', '.svg']:
        fig.savefig(HERE/(output+suffix), dpi=210)
    plt.close(fig)


def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none'})
    sources = {}
    meta, coords = base.load_case('melanoma', sources)
    comparison(meta, coords, ['RAMP'], ['current', 'layered', 'fine'], 'melanoma_umap_style_options')
    comparison(meta, coords, list(base.METHODS), ['current', SELECTED], 'melanoma_umap_current_vs_refined')
    base.HERE = HERE
    base.draw_scatter = refined_scatter
    base.ORDER = ORDER
    base.main()
    for suffix in ['.png', '.svg']:
        (HERE/('melanoma_ridge_umap_figure1_style'+suffix)).rename(
            HERE/('melanoma_ridge_umap_refined'+suffix))
    path = HERE/'manifest.json'
    report = json.loads(path.read_text())
    report.update({
        'display_variant': SELECTED,
        'style_options': VARIANTS,
        'celltype_draw_order_before': ORIGINAL_ORDER,
        'draw_order_bottom_to_top': ORDER,
        'scatter_celltype_size': VARIANTS[SELECTED]['size'],
        'scatter_celltype_alpha': VARIANTS[SELECTED]['alpha'],
        'palette_changed': False,
        'dataset_display_changed': False,
        'source_panel': str(HERE.parent/'melanoma_ridge_umap_figure1_style.png'),
        'source_panel_sha256': base.sha(HERE.parent/'melanoma_ridge_umap_figure1_style.png'),
        'source_panel_preserved': True,
    })
    path.write_text(json.dumps(report, indent=2)+'\n')
    assert all(base.sha(Path(path)) == digest for path, digest in sources.items())
    print(HERE/'melanoma_umap_style_options.png')
    print(HERE/'melanoma_ridge_umap_refined.png')


if __name__ == '__main__':
    main()
