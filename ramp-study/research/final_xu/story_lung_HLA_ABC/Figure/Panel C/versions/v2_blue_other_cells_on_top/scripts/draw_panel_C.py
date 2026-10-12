"""Show fixed UMAP with blue Other cells above vCAF above Other CAF."""
from runtime import BASE, STORY
from pathlib import Path
import hashlib
import json
import shutil

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, ConnectionPatch
from matplotlib.lines import Line2D

COLORS = {'Other cells': '#2b8cbe', 'Other CAF': '#9869b0', 'vCAF': '#e54285'}
DRAW_ORDER = ['Other CAF', 'vCAF', 'Other cells']
STYLE = {
    'vCAF': {'size': 2.4, 'alpha': .85},
    'Other CAF': {'size': .80, 'alpha': .64},
    'Other cells': {'size': .38, 'alpha': .42},
}
SOURCES = {
    'data/RAMP_UMAP.npy': STORY / 'umap/results/RAMP_UMAP.npy',
    'data/CELL_METADATA.csv.gz': STORY / 'data_umap/data/CELL_METADATA.csv.gz',
    'provenance/PANEL_B_VERIFICATION.json': STORY / 'Figure/Panel B/provenance/VERIFICATION.json',
    'provenance/Cords2024_paper.md': STORY.parent / 'raw_data/lung_cancer/Cords2024/Paper_Cords_2024.md',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def main():
    for folder in ['data', 'provenance']:
        (BASE / folder).mkdir(exist_ok=True)
    sources = []
    for relative, source in SOURCES.items():
        destination = BASE / relative
        if destination.exists():
            assert sha(destination) == sha(source), source
        else:
            shutil.copyfile(source, destination)
        sources.append({'source': str(source), 'local': relative, 'sha256': sha(source)})
    save_json(BASE / 'provenance/SOURCES.json', sources)

    md = pd.read_csv(BASE / 'data/CELL_METADATA.csv.gz', low_memory=False)
    u = np.load(BASE / 'data/RAMP_UMAP.npy')
    assert u.shape == (84000, 2) and len(md) == 84000 and md.cell_key.is_unique
    assert np.isfinite(u).all()
    cords = md.dataset.eq('Cords IMC')
    vcaf = cords & md.provider_cell_type.eq('vCAF')
    other_caf = cords & md.cell_category.eq('Fibroblast') & ~vcaf
    md['display_group'] = 'Other cells'
    md.loc[other_caf, 'display_group'] = 'Other CAF'
    md.loc[vcaf, 'display_group'] = 'vCAF'
    counts = {key: int(md.display_group.eq(key).sum()) for key in COLORS}
    assert counts == {'Other cells': 78794, 'Other CAF': 4560, 'vCAF': 646}
    assert md.loc[vcaf, 'patient_id'].nunique() == 27
    assert not md.loc[~cords, 'display_group'].isin(['vCAF', 'Other CAF']).any()

    metadata = md[['cell_key', 'dataset', 'patient_id', 'provider_cell_type',
                   'cell_category', 'broad_type', 'display_group']].copy()
    metadata[['UMAP1', 'UMAP2']] = u
    metadata.to_csv(BASE / 'data/PLOT_DATA.csv.gz', index=False,
                    compression={'method': 'gzip', 'compresslevel': 3, 'mtime': 0})
    md.groupby(['dataset', 'display_group']).size().rename('n_cells').reset_index().to_csv(
        BASE / 'data/GROUP_COUNTS.csv', index=False)

    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 8, 'svg.fonttype': 'none',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.labelcolor': '#43515a', 'axes.edgecolor': '#9ba8b0',
        'axes.linewidth': .7, 'savefig.facecolor': 'white',
    })
    reference = json.loads((BASE / 'provenance/PANEL_B_VERIFICATION.json').read_text())
    limits = reference['coordinate_limits']['aligned_both_views']
    plot_w, plot_h = reference['plot_size_inches']
    bottom = reference['plot_bottom_inches']
    width, height, left = 3.67, 2.75, .22
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([left / width, bottom / height, plot_w / width, plot_h / height])
    ax.set(xlim=limits[0], ylim=limits[1], xlabel='UMAP 1', ylabel='UMAP 2')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('Author annotations', fontsize=10, fontweight='bold', pad=7, color='#243745')
    fig.text(.017, .960, 'C', fontsize=22, fontweight='bold', va='top', color='#243745')
    fig.text(.150, .957, 'vCAF in IMC', fontsize=12.3, fontweight='bold',
             va='top', color='#243745')

    # From top to bottom: Other cells > vCAF > Other CAF; categories are disjoint.
    artists, all_indices = [], []
    for zorder, group in enumerate(DRAW_ORDER, start=1):
        indices = np.random.default_rng(42).permutation(np.flatnonzero(md.display_group.eq(group)))
        style = STYLE[group]
        artist = ax.scatter(u[indices, 0], u[indices, 1], c=COLORS[group],
                            s=style['size'], alpha=style['alpha'], linewidths=0,
                            rasterized=True, zorder=zorder)
        assert np.array_equal(np.asarray(artist.get_offsets()), u[indices])
        artists.append(artist); all_indices.append(indices)
    assert np.array_equal(np.sort(np.concatenate(all_indices)), np.arange(len(md)))
    assert [artist.get_zorder() for artist in artists] == [1, 2, 3]

    # This ellipse is a visual callout, not a cell selection rule or a new phenotype gate.
    center, ellipse_w, ellipse_h = (3.85, 9.92), 4.0, 3.25
    ring = Ellipse(center, ellipse_w, ellipse_h, facecolor='none',
                   edgecolor='#49545d', linewidth=.9, linestyle=(0, (3, 2)), zorder=4)
    ax.add_patch(ring)
    arrow = ConnectionPatch(
        xyA=(5.80, 10.20), coordsA=ax.transData,
        xyB=(2.46 / width, 1.99 / height), coordsB=fig.transFigure,
        arrowstyle='-|>', mutation_scale=9, linewidth=.9,
        color='#49545d', shrinkA=0, shrinkB=0, clip_on=False, zorder=5,
    )
    fig.add_artist(arrow)
    text_x = 2.53 / width
    definition = [
        (1.99, 'vCAF', 11, 'bold', COLORS['vCAF']),
        (1.76, 'Vascular CAF', 8.3, 'normal', '#243745'),
        (1.54, 'CD146⁺ / CD34⁻', 8.8, 'bold', '#243745'),
        (1.31, 'Cords et al., 2024', 7, 'normal', '#67747c'),
    ]
    for y, text, size, weight, color in definition:
        fig.text(text_x, y / height, text, fontsize=size, fontweight=weight,
                 color=color, ha='left', va='center')
    legend = fig.legend(
        handles=[Line2D([], [], marker='o', linestyle='', color=color,
                        markersize=4.5, label=label) for label, color in COLORS.items()],
        loc='center left', bbox_to_anchor=(text_x, .66 / height),
        ncol=1, frameon=False, fontsize=7.8, handlelength=.85,
        handletextpad=.40, labelspacing=.60, borderaxespad=0, borderpad=0,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in [*fig.texts, legend]:
        box = text.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        assert box.x0 >= 0 and box.x1 <= 1 and box.y0 >= 0 and box.y1 <= 1, box
    for ext in ['png', 'svg']:
        fig.savefig(BASE / f'panel_C_vcaf.{ext}', dpi=300)
    plt.close(fig)

    in_ring = (((u[:, 0] - center[0]) / (ellipse_w / 2)) ** 2 +
               ((u[:, 1] - center[1]) / (ellipse_h / 2)) ** 2 <= 1)
    for record in sources:
        assert sha(Path(record['source'])) == sha(BASE / record['local']) == record['sha256']
    report = {
        'status': 'PASS', 'version': 'v2_blue_other_cells_on_top',
        'n_cells': len(md), 'group_counts': counts, 'vCAF_n_patients': 27,
        'annotation_source': 'Original Cords IMC CAF labels only',
        'other_cells_includes': 'All remaining pooled reference cells, including non-IMC cells',
        'layer_order_bottom_to_top': DRAW_ORDER, 'group_style': STYLE,
        'layer_order_top_to_bottom': list(reversed(DRAW_ORDER)), 'group_colors': COLORS,
        'coordinate_refit': False, 'all_cells_shown': True,
        'scatter_coordinates_exact': True, 'same_limits_and_plot_size_as_Panel_B': True,
        'figure_size_inches': [width, height], 'coordinate_limits': limits,
        'new_marker_gating': False, 'source_annotations_unchanged': True,
        'ellipse': {'center': center, 'width': ellipse_w, 'height': ellipse_h,
                    'role': 'Visual callout only; not used to select or define vCAF',
                    'vCAF_inside': int((in_ring & vcaf.to_numpy()).sum()),
                    'vCAF_outside': int((~in_ring & vcaf.to_numpy()).sum())},
        'paper_definition': 'vCAF (CD146+/CD34-) among author-annotated CAFs',
        'paper_url': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC10929690/',
        'text_and_legend_inside_canvas': True,
    }
    save_json(BASE / 'provenance/VERIFICATION.json', report)
    save_json(BASE / 'provenance/MANIFEST.json', [
        {'file': str(path.relative_to(BASE)), 'sha256': sha(path)}
        for path in sorted(BASE.rglob('*')) if path.is_file()
        and path.relative_to(BASE).parts[0] not in {'cache', 'tmp', 'versions'}
        and path.name != 'MANIFEST.json'
    ])
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
