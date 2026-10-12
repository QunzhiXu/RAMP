"""Restyle frozen TNBC ridges and UMAPs using ADTnorm-like column proportions."""
from pathlib import Path
import json
import os
import sys

HERE = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(HERE / '.cache/matplotlib'))
os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, str(HERE.parent))
from build_refined_tnbc import load_case, load_matrices, sha, refined_scatter, PALETTE
from build_myeloid_bottom_tnbc import ORDER
from build_review import METHODS, DATASET_COLORS, legend
from build_ridge_review import SELECTED, tick
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator
from scipy.ndimage import gaussian_filter1d

FRAME = '#8C8C8C'
PEAK_HEIGHT = .70
FILL_ALPHA = .82
DATASET_LABELS = {'Tietscher IMC': 'IMC1', 'Meyer IMC': 'IMC2', 'Shiao CODEX': 'CODEX'}


def ridge(ax, values, meta, method, marker, audit):
    # Preserve the previous display range and density estimates exactly.
    low, high = np.quantile(values, [.005, .995])
    if high <= low:
        high = low + 1
    datasets = meta.dataset.astype(str).to_numpy()
    units = meta.unit_key.astype(str).to_numpy()
    palette = dict(zip(meta.dataset.unique(), DATASET_COLORS))
    groups = [(d, u) for d in palette for u in sorted(set(units[datasets == d]))]
    level, prev = 0., None
    for i, (dataset, unit) in enumerate(reversed(groups)):
        if prev is not None and dataset != prev:
            level += .36
        vals = values[(datasets == dataset) & (units == unit)]
        counts, edges = np.histogram(vals, bins=160, range=(low, high))
        smooth = gaussian_filter1d(counts.astype(float), sigma=6, mode='constant')
        if smooth.max() > 0:
            smooth /= smooth.max()
        x = (edges[1:] + edges[:-1]) / 2
        y = level + PEAK_HEIGHT * smooth
        # Draw each complete ridge above the lower one to make slight overlap clear.
        ax.fill_between(x, level, y, color=palette[dataset], alpha=FILL_ALPHA,
                        linewidth=0, zorder=2*i+1)
        ax.plot(x, y, color='#46505A', linewidth=.55, zorder=2*i+2)
        audit.append({'benchmark': 'tnbc', 'method': method, 'marker': marker,
                      'smoothing_sigma_bins': 6, 'dataset': dataset, 'unit_key': unit,
                      'n_cells': len(vals), 'display_low': float(low),
                      'display_high': float(high),
                      'n_below_display': int((vals < low).sum()),
                      'n_above_display': int((vals > high).sum())})
        level += .57
        prev = dataset
    ax.set_xlim(low, high)
    ax.set_ylim(-.09, level - .57 + PEAK_HEIGHT + .04)
    if low < 0 < high:
        ax.axvline(0, color='#666666', linewidth=.55,
                   linestyle=(0, (2, 3)), zorder=100, alpha=.7)
    ax.set_yticks([])
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.xaxis.set_major_formatter(FuncFormatter(tick))
    ax.tick_params(axis='x', labelsize=8, length=2.1, pad=2, width=.5)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(.45)
        spine.set_color(FRAME)


def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none'})
    sources, audit = {}, []
    meta, coords = load_case('tnbc', sources)
    markers, matrices = load_matrices('tnbc', meta, sources)
    previous = json.loads((HERE.parent / 'tight_rows_manifest.json').read_text())
    assert sources == previous['input_sha256']
    row_height = previous['layout']['row_height_in']
    row_gap, header, footer = .16, .90, .60
    height = header + 4*row_height + 3*row_gap + footer
    ridge_left, ridge_width, ridge_gap = .56, 2.02*.65, .10
    ridge_right = ridge_left + 3*ridge_width + 2*ridge_gap
    umap_gap = .10
    umap_widths = {m: row_height*np.ptp(xy[:, 0])/np.ptp(xy[:, 1])
                   for m, xy in coords.items()}
    slot_width = max(umap_widths.values())
    pair_left = ridge_right + .12
    width = pair_left + 2*slot_width + umap_gap + .14
    fig = plt.figure(figsize=(width, height), facecolor='white')
    fig.text(.16/width, 1-.29/height, 'B', fontsize=17,
             weight='bold', va='baseline')
    fig.text(ridge_left/width, 1-.29/height, 'Cross-platform alignment in TNBC',
             fontsize=13.5, weight='bold', va='baseline')
    fig.text(1-.14/width, 1-.13/height,
             f'{len(meta):,} cells  |  {meta.unit_key.nunique()} sample units',
             fontsize=9, ha='right', va='top')
    da = fig.add_axes([.04, 1-.54/height, .95, .20/height])
    legend(da, [DATASET_LABELS[d] for d in meta.dataset.unique()], DATASET_COLORS, 3, 9)
    geometry = []
    text_artists = list(fig.texts)
    for row, method in enumerate(METHODS):
        bottom = height-header-(row+1)*row_height-row*row_gap
        for col, marker in enumerate(SELECTED['tnbc']):
            left = ridge_left+col*(ridge_width+ridge_gap)
            ax = fig.add_axes([left/width, bottom/height,
                               ridge_width/width, row_height/height])
            vals = matrices[method][:, markers.index(marker)]
            displayed_method = method
            if method == 'Unaligned':
                vals = np.arcsinh(vals)
                displayed_method = 'Unaligned_arcsinh_display'
            ridge(ax, vals, meta, displayed_method, marker, audit)
            if row == 0:
                ax.set_title(marker, fontsize=11.5, weight='bold', pad=8)
            if col == 0:
                label = 'Raw\n(arcsinh)' if method == 'Unaligned' else METHODS[method]
                ax.set_ylabel(label, fontsize=11.5, weight='bold', labelpad=8)
        for col, color_by in enumerate(['Dataset', 'Cell type']):
            slot_left = pair_left+col*(slot_width+umap_gap)
            actual_width = umap_widths[method]
            # Keep the previous physical scatter size; align frames across all rows.
            left = slot_left + (slot_width-actual_width)/2
            ax = fig.add_axes([left/width, bottom/height,
                               actual_width/width, row_height/height])
            refined_scatter(ax, meta, coords[method], color_by, draw_order=ORDER)
            fig.add_artist(Rectangle((slot_left/width, bottom/height),
                                     slot_width/width, row_height/height,
                                     fill=False, edgecolor=FRAME, linewidth=.45,
                                     transform=fig.transFigure, zorder=100))
            if row == 0:
                text_artists.append(fig.text((slot_left+slot_width/2)/width,
                                             (bottom+row_height+.11)/height,
                                             color_by, ha='center', va='bottom',
                                             fontsize=11.5, weight='bold'))
            geometry.append({'method': method, 'color_by': color_by,
                             'left_in': left, 'width_in': actual_width,
                             'height_in': row_height, 'frame_left_in': slot_left,
                             'frame_width_in': slot_width})
    labels = meta.celltype.astype(str).value_counts().index.tolist()
    la = fig.add_axes([.02, .055/height, .97, .34/height])
    legend(la, labels, [PALETTE[label] for label in labels], 5, 8.5)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Ensure the compact headers, row labels and legends fit inside the canvas.
    text_artists += [t for ax in fig.axes for t in [ax.title, ax.yaxis.label]
                     if t.get_text()]
    for ax in fig.axes:
        if ax.get_legend() is not None:
            text_artists.append(ax.get_legend())
    for artist in text_artists:
        bounds = artist.get_window_extent(renderer)
        assert bounds.x0 >= -1 and bounds.y0 >= -1, (str(artist), bounds)
        assert bounds.x1 <= fig.bbox.width+1 and bounds.y1 <= fig.bbox.height+1
    png = HERE / 'tnbc_ridge_umap_adtnorm_style.png'
    fig.savefig(png, dpi=210)
    fig.savefig(HERE / 'tnbc_ridge_umap_adtnorm_style.svg', dpi=210)
    plt.close(fig)
    pd.DataFrame(audit).to_csv(HERE / 'ridge_display_audit.csv', index=False)
    assert all(sha(Path(p)) == digest for p, digest in sources.items())
    old_geometry = previous['layout']['umap_axes']
    for before, after in zip(old_geometry, geometry):
        assert before['method'] == after['method']
        assert np.isclose(before['width_in'], after['width_in'])
        assert np.isclose(before['height_in'], after['height_in'])
    report = {
        'panel_label': 'B', 'panel_title': 'Cross-platform alignment in TNBC',
        'input_sha256': sources, 'inputs_unchanged': True,
        'reference_pdf': '/Users/qxu6/Downloads/Papers/ADTnorm.pdf',
        'reference_figure': 'Figure 1B: column proportions, ridge silhouette and thin frames',
        'previous_version': '../tnbc_ridge_umap_sigma6_tight_rows.png',
        'cells': len(meta), 'sample_units': int(meta.unit_key.nunique()),
        'dataset_display_labels': DATASET_LABELS,
        'all_cells_retained': True, 'umap_refit': False,
        'umap_physical_size_preserved': True, 'umap_display_rules_unchanged': True,
        'celltype_draw_order_bottom_to_top': ORDER,
        'ridge_sigma_bins': 6, 'ridge_bins': 160, 'ridge_range_quantiles': [.005, .995],
        'raw_ridge_display': 'arcsinh(x/1)', 'density_estimation_unchanged': True,
        'ridge_curve_height': PEAK_HEIGHT, 'previous_ridge_curve_height': .49,
        'ridge_height_normalization': 'Each curve divided by its own peak; not cell abundance',
        'ridge_fill_alpha': FILL_ALPHA, 'ridge_outline_color': '#46505A',
        'frame_color': FRAME, 'frame_linewidth_pt': .45,
        'layout': {'figure_width_in': width, 'figure_height_in': height,
                   'previous_width_in': previous['layout']['figure_width_in'],
                   'ridge_width_in': ridge_width, 'previous_ridge_width_in': 2.02,
                   'ridge_gap_in': ridge_gap, 'row_gap_in': row_gap,
                   'umap_frame_gap_in': umap_gap, 'umap_axes': geometry},
    }
    (HERE / 'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print(png, flush=True)
    print(json.dumps({'figure_inches': [width, height],
                      'width_reduction_percent': 100*(1-width/previous['layout']['figure_width_in']),
                      'source_files_verified': len(sources)}, indent=2))


if __name__ == '__main__':
    main()
