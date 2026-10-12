"""Review frozen marker matrices with all sample units; write only in this folder."""
from pathlib import Path
import json
import math

from build_review import (HERE, ROOT, CASES, METHODS, COLORS, DATASET_COLORS,
                          sha, load_case, scatter, legend, raw_ramp_overview)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from scipy.ndimage import gaussian_filter1d

SELECTED = {
    'bone_marrow': ['CD3', 'CD4', 'CD8'],
    'tnbc': ['CD3', 'CD8', 'PANCK'],
    'melanoma': ['CD3', 'CD8', 'SOX10'],
    'lung_cancer': ['CD3', 'VIMENTIN', 'EPI'],
}
PRETTY_DATASET = {
    'MEL_Hickey2023_CODEX': 'Hickey CODEX',
    'MEL_Hoch2021_IMC': 'Hoch IMC',
    'MEL_Ptacek2021_MIBI': 'Ptacek MIBI',
}


def load_matrices(case, meta, sources):
    base = ROOT / 'benchmark_results' / case
    marker_path = base / 'ramp/alignment/input/markers.txt'
    sources[str(marker_path)] = sha(marker_path)
    markers = marker_path.read_text().splitlines()
    paths = {
        'Unaligned': base / 'ramp/alignment/input/raw.npy',
        'Harmony-direct_source_scale': base / 'harmony/source_scale/alignment/data/harmony_adjusted_marker_coordinates.csv.gz',
        'cyCombine': base / 'cycombine/alignment/data/corrected_expression.csv.gz',
        'RAMP': base / 'ramp/alignment/data/ramp_aligned.npy',
    }
    out = {}
    for method, path in paths.items():
        sources[str(path)] = sha(path)
        if path.suffix == '.npy':
            matrix = np.load(path, mmap_mode='r')
        else:
            frame = pd.read_csv(path, low_memory=False)
            ids = meta.cell_id.astype(str).to_numpy()
            frame['cell_id'] = frame.cell_id.astype(str)
            assert frame.cell_id.is_unique
            assert len(frame) == len(meta) and set(frame.cell_id) == set(ids)
            frame = frame.set_index('cell_id').loc[ids]
            matrix = frame[markers].to_numpy()
        assert matrix.shape == (len(meta), len(markers)), (case, method, matrix.shape)
        assert np.isfinite(matrix).all()
        out[method] = matrix
    return markers, out


def tick(value, _pos):
    if abs(value) >= 1000:
        return f'{value/1000:g}k'
    return f'{value:g}'


def ridge(ax, values, meta, case, method, marker, audit, sigma_bins=2):
    low, high = np.quantile(values, [.005, .995])
    if high <= low:
        high = low + 1
    datasets = meta.dataset.astype(str).to_numpy()
    units = meta.unit_key.astype(str).to_numpy()
    palette = dict(zip(meta.dataset.unique(), DATASET_COLORS))
    groups = [(d, u) for d in palette for u in sorted(set(units[datasets == d]))]
    level = 0.
    prev = None
    for dataset, unit in reversed(groups):
        if prev is not None and dataset != prev:
            level += .36
        vals = values[(datasets == dataset) & (units == unit)]
        # Do not clip tails into edge bins: doing so would manufacture edge peaks.
        counts, edges = np.histogram(vals, bins=160, range=(low, high))
        smooth = gaussian_filter1d(counts.astype(float), sigma=sigma_bins, mode='constant')
        if smooth.max() > 0:
            smooth /= smooth.max()
        x = (edges[1:] + edges[:-1])/2
        y = level + .49*smooth
        ax.fill_between(x, level, y, color=palette[dataset], alpha=.37, linewidth=0)
        ax.plot(x, y, color=palette[dataset], linewidth=.58)
        audit.append({'benchmark': case, 'method': method, 'marker': marker,
                      'smoothing_sigma_bins': sigma_bins,
                      'dataset': dataset, 'unit_key': unit, 'n_cells': len(vals),
                      'display_low': float(low), 'display_high': float(high),
                      'n_below_display': int((vals < low).sum()),
                      'n_above_display': int((vals > high).sum())})
        level += .57
        prev = dataset
    ax.set_xlim(low, high)
    ax.set_ylim(-.09, level+.03)
    if low < 0 < high:
        ax.axvline(0, color='#666666', linewidth=.6, linestyle=(0, (2, 3)), zorder=0)
    ax.set_yticks([])
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.xaxis.set_major_formatter(FuncFormatter(tick))
    ax.tick_params(axis='x', labelsize=8, length=2.3, pad=2)
    for side in ['left', 'right', 'top']:
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_linewidth(.6)


def dataset_legend(fig, meta, y, fontsize=9, box_height=.027):
    da = fig.add_axes([.04, y, .95, box_height])
    labels = [PRETTY_DATASET.get(d, d) for d in meta.dataset.unique()]
    legend(da, labels, DATASET_COLORS, 3, fontsize)


def combined(case, meta, coords, markers, matrices, audit, sigma_bins=2, output_path=None,
             raw_display='linear', compact=False, scatter_func=None, cell_palette=None,
             compact_rows=False):
    assert raw_display in ['linear', 'arcsinh']
    draw_scatter = scatter if scatter_func is None else scatter_func
    cell_colors = COLORS if cell_palette is None else cell_palette
    height = 11.8
    row_height = height * (.90-.13) / (4 + 3*.20)
    row_gap = row_height*.20
    grid_top, grid_bottom, row_hspace = .90, .13, .20
    if compact_rows:
        assert compact
        row_gap = .16
        header, footer = .90, .60
        height = header + 4*row_height + 3*row_gap + footer
        grid_top, grid_bottom = 1-header/height, footer/height
        row_hspace = row_gap/row_height
    if compact:
        # Size each UMAP axis to its actual data aspect, then pack each pair.
        # Equal-aspect axes inside wide GridSpec cells otherwise create hidden gaps.
        ridge_left, ridge_width, ridge_gap = .64, 2.02, .18
        ridge_right = ridge_left + 3*ridge_width + 2*ridge_gap
        umap_gap = .10
        umap_widths = {m: row_height*np.ptp(xy[:,0])/np.ptp(xy[:,1])
                       for m,xy in coords.items()}
        pair_width = 2*max(umap_widths.values()) + umap_gap
        pair_left = ridge_right + .12
        width = pair_left + pair_width + .14
    else:
        width = 14
    fig = plt.figure(figsize=(width, height), facecolor='white')
    title_y = 1-.10/height if compact_rows else .982
    count_y = 1-.13/height if compact_rows else .98
    fig.text(.055, title_y, CASES[case], fontsize=18, weight='bold', va='top')
    fig.text(.985, count_y, f'{len(meta):,} cells  |  {meta.unit_key.nunique()} sample units',
             fontsize=10, ha='right', va='top')
    if compact_rows:
        dataset_legend(fig,meta,1-.54/height,box_height=.20/height)
    else:
        dataset_legend(fig, meta, .926)
    if compact:
        row_grid = fig.add_gridspec(4,1,left=0,right=1,top=grid_top,bottom=grid_bottom,
                                   hspace=row_hspace)
    else:
        grid = fig.add_gridspec(4, 5, left=.055, right=.99, top=.90, bottom=.13,
                               width_ratios=[1,1,1,1.25,1.25], hspace=.20, wspace=.19)
    geometry = []
    for i, method in enumerate(METHODS):
        if compact:
            row_box = row_grid[i,0].get_position(fig)
        for j, marker in enumerate(SELECTED[case]):
            if compact:
                ax = fig.add_axes([(ridge_left+j*(ridge_width+ridge_gap))/width,
                                   row_box.y0,ridge_width/width,row_box.height])
            else:
                ax = fig.add_subplot(grid[i,j])
            values = matrices[method][:,markers.index(marker)]
            displayed_method = method
            if method == 'Unaligned' and raw_display == 'arcsinh':
                values = np.arcsinh(values)
                displayed_method = 'Unaligned_arcsinh_display'
                if not compact_rows:
                    ax.set_xlabel('arcsinh(source value)', fontsize=8)
            ridge(ax, values, meta,
                  case, displayed_method, marker, audit, sigma_bins=sigma_bins)
            if i == 0:
                ax.set_title(marker, fontsize=12, weight='bold', pad=10)
            if j == 0:
                row_label = METHODS[method]
                if compact_rows and method == 'Unaligned' and raw_display == 'arcsinh':
                    row_label = 'Raw\n(arcsinh)'
                ax.set_ylabel(row_label, fontsize=12, weight='bold', labelpad=11)
        for j, color_by in enumerate(['Dataset', 'Cell type'], start=3):
            if compact:
                w = umap_widths[method]
                start = pair_left+(pair_width-(2*w+umap_gap))/2
                ax = fig.add_axes([(start+(j-3)*(w+umap_gap))/width,
                                   row_box.y0,w/width,row_box.height])
                geometry.append({'method':method,'color_by':color_by,
                                 'left_in':start+(j-3)*(w+umap_gap),
                                 'width_in':w,'height_in':row_height})
            else:
                ax = fig.add_subplot(grid[i,j])
            draw_scatter(ax, meta, coords[method], color_by)
            if i == 0:
                ax.set_title(color_by, fontsize=12, weight='bold', pad=10)
    labels = meta.celltype.astype(str).value_counts().index.tolist()
    legend_box = [.02,.055/height,.97,.34/height] if compact_rows else [.02,.022,.97,.078]
    la = fig.add_axes(legend_box)
    legend(la, labels, [cell_colors[l] for l in labels], 5, 8.5)
    path = output_path or HERE / f'{case}_ridge_umap.png'
    fig.savefig(path, dpi=190)
    plt.close(fig)
    print(path, flush=True)
    return {'figure_width_in':width,'figure_height_in':height,
            'compact':compact,'umap_axes':geometry,
            'umap_axes_gap_in':umap_gap if compact else None,
            'compact_rows':compact_rows,'row_height_in':row_height,'row_gap_in':row_gap}


def ramp_atlas(case, meta, markers, matrices, audit):
    nrows = math.ceil(len(markers)/4)
    fig = plt.figure(figsize=(13.5, 3.2*nrows+1.05), facecolor='white')
    fig.text(.025, .99, f'{CASES[case]} | RAMP | All {len(markers)} shared markers',
             fontsize=17, weight='bold', va='top')
    dataset_legend(fig, meta, .932 if nrows == 4 else .913)
    grid = fig.add_gridspec(nrows, 4, left=.035, right=.99, top=.895 if nrows == 4 else .87,
                           bottom=.04, hspace=.28, wspace=.24)
    for j, marker in enumerate(markers):
        ax = fig.add_subplot(grid[j//4,j%4])
        ridge(ax, matrices['RAMP'][:,j], meta, case, 'RAMP', marker, audit)
        ax.set_title(marker, fontsize=12, weight='bold', pad=7)
    path = HERE / f'{case}_ramp_all_markers.png'
    fig.savefig(path, dpi=175)
    plt.close(fig)
    print(path, flush=True)


def main():
    plt.rcParams.update({'font.family':'DejaVu Sans'})
    sources, audit, loaded = {}, [], {}
    for case in CASES:
        meta, coords = load_case(case, sources)
        loaded[case] = (meta, coords)
        markers, matrices = load_matrices(case, meta, sources)
        combined(case, meta, coords, markers, matrices, audit)
        ramp_atlas(case, meta, markers, matrices, audit)
    raw_ramp_overview(loaded)
    pd.DataFrame(audit).drop_duplicates(['benchmark','method','marker','dataset','unit_key']).to_csv(
        HERE / 'ridge_display_audit.csv', index=False)
    assert all(sha(Path(p)) == digest for p, digest in sources.items())
    report = {'input_sha256': sources, 'inputs_unchanged': True,
              'alignment_refit': False, 'umap_refit': False, 'display_subsampling': False,
              'ridge_unit': 'one curve per dataset × unit_key; all frozen benchmark cells',
              'selected_markers': SELECTED, 'all_ramp_markers_in_atlases': True,
              'ridge_display_quantiles': [.005,.995],
              'ridge_x_scale': 'linear, separately per method × marker; shared by every sample in that subplot',
              'ridge_histogram_bins':160, 'ridge_smoothing_sigma_bins':2,
              'ridge_smoothing_mode':'constant', 'tail_values_clipped_into_edge_bins':False,
              'ridge_height':'Each curve normalized by its own maximum; does not encode cell abundance',
              'posthoc_sample_recentering_or_scaling':False,
              'zero_reference':'Dashed x=0 guide where in range; not a biological negative gate'}
    (HERE / 'ridge_review_manifest.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
