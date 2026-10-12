"""Draw aligned UMAPs with seven display groups and CD4 above fibroblast/myeloid."""
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
from matplotlib.lines import Line2D

PLATFORMS = {'Senosain CyTOF': 'CyTOF', 'Zheng mIF': 'mIF', 'Cords IMC': 'IMC'}
PLATFORM_COLORS = {'CyTOF': '#487c9d', 'mIF': '#368f83', 'IMC': '#c35b75'}
TYPE_COLORS = {
    'Tumor / epithelial': '#cf5748',
    'CD4 T': '#3978b5',
    'CD8 T': '#e17c39',
    'Myeloid / APC': '#d0a02e',
    'Fibroblast / stroma': '#c35199',
    'Endothelial': '#46aab6',
    'Others': '#bcbfc5',
}
DISPLAY_GROUPS = {
    'Other / uncertain': 'Others',
    'B cell': 'Others',
    'NK cell': 'Others',
    'Benign epithelium': 'Others',
}
T_MAPPING = {
    ('Senosain CyTOF', 'CD4 T'): 'CD4 T',
    ('Senosain CyTOF', 'CD8 T'): 'CD8 T',
    ('Cords IMC', 'CD4'): 'CD4 T',
    ('Cords IMC', 'CD8'): 'CD8 T',
    ('Zheng mIF', 'Th'): 'CD4 T',
    ('Zheng mIF', 'Treg'): 'CD4 T',
    ('Zheng mIF', 'Tc'): 'CD8 T',
    ('Zheng mIF', 'DNT'): 'Others',
}
SOURCES = {
    'data/RAMP_UMAP.npy': 'umap/results/RAMP_UMAP.npy',
    'data/CELL_METADATA.csv.gz': 'data_umap/data/CELL_METADATA.csv.gz',
    'provenance/SELECTED_DONORS.csv': 'data_umap/data/SELECTED_DONORS.csv',
    'provenance/MARKER_MAPPING.csv': 'data_umap/data/MARKER_MAPPING.csv',
    'provenance/RAMP_UMAP_PARAMETERS.json': 'umap/results/RAMP_UMAP.json',
    'provenance/SOURCE_UMAP_VERIFICATION.json': 'umap/results/VERIFICATION.json',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def stage_inputs():
    for name in ['data', 'results', 'provenance']:
        (BASE / name).mkdir(exist_ok=True)
    record_file = BASE / 'provenance/SOURCES.json'
    if record_file.exists():
        records = [item for item in json.loads(record_file.read_text())
                   if item['local'] in SOURCES]
        assert {item['local'] for item in records} == set(SOURCES)
        for item in records:
            assert sha(BASE / item['local']) == item['sha256'], item['local']
        save_json(record_file, records)
        return records
    records = []
    for local, relative_source in SOURCES.items():
        source, destination = STORY / relative_source, BASE / local
        digest = sha(source)
        if destination.exists():
            assert sha(destination) == digest, destination
        else:
            shutil.copyfile(source, destination)
        assert sha(destination) == digest
        records.append({'local': local, 'source': str(source), 'sha256': digest})
    save_json(record_file, records)
    return records


def handles(palette):
    return [Line2D([], [], color=color, marker='o', linestyle='',
                   markersize=4.5, label=label) for label, color in palette.items()]


def bounds(u):
    lo, hi = u.min(axis=0), u.max(axis=0)
    padding = (hi - lo) * .035
    return [(float(a), float(b)) for a, b in zip(lo - padding, hi + padding)]


def style_axis(ax, u, title):
    limits = bounds(u)
    ax.set(xlim=limits[0], ylim=limits[1], xlabel='UMAP 1', ylabel='UMAP 2')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title, fontsize=10, fontweight='bold', pad=7, color='#243745')


def scatter(ax, u, values, colors, ix):
    assert len(ix) == len(u) and len(np.unique(ix)) == len(u)
    mapped = pd.Series(values).map(colors).to_numpy()[ix]
    assert not pd.isna(mapped).any()
    return ax.scatter(u[ix, 0], u[ix, 1], c=mapped, s=.60, alpha=.65,
                      linewidths=0, rasterized=True)


def main():
    source_records = stage_inputs()
    md = pd.read_csv(BASE / 'data/CELL_METADATA.csv.gz', low_memory=False)
    after = np.load(BASE / 'data/RAMP_UMAP.npy')
    assert after.shape == (84000, 2)
    assert len(md) == len(after) and md.cell_key.is_unique
    assert np.isfinite(after).all()
    assert md.groupby(['dataset', 'patient_id']).size().eq(1000).all()
    assert md.groupby('dataset').patient_id.nunique().eq(28).all()
    assert md.groupby('dataset').size().eq(28000).all()

    md['platform'] = md.dataset.map(PLATFORMS)
    md['display_cell_type'] = md.broad_type.replace(DISPLAY_GROUPS)
    t_mask = md.broad_type.eq('T cell')
    md.loc[t_mask, 'display_cell_type'] = [
        T_MAPPING[(ds, label)] for ds, label in zip(
            md.loc[t_mask, 'dataset'], md.loc[t_mask, 'provider_cell_type'])
    ]
    assert md.display_cell_type.isin(TYPE_COLORS).all()
    dnt = md.dataset.eq('Zheng mIF') & md.provider_cell_type.eq('DNT')
    assert dnt.sum() == 587 and md.loc[dnt, 'display_cell_type'].eq('Others').all()
    expected_counts = {
        'Tumor / epithelial': 28580, 'Myeloid / APC': 15351,
        'Fibroblast / stroma': 11847, 'CD4 T': 6214, 'CD8 T': 5922,
        'Endothelial': 5252, 'Others': 10834,
    }
    assert md.display_cell_type.value_counts().to_dict() == expected_counts
    pooled = md.broad_type.isin(['B cell', 'NK cell', 'Benign epithelium'])
    assert pooled.sum() == 2667 and md.loc[pooled, 'display_cell_type'].eq('Others').all()
    md.groupby(['dataset', 'platform', 'display_cell_type']).size().rename(
        'n_cells').reset_index().to_csv(BASE / 'results/CELL_COUNTS.csv', index=False)
    md.groupby(['dataset', 'provider_cell_type', 'broad_type', 'display_cell_type'],
               dropna=False).size().rename('n_cells').reset_index().to_csv(
                   BASE / 'data/DISPLAY_GROUPING.csv', index=False)
    pd.DataFrame([
        {'dataset': ds, 'provider_cell_type': label, 'display_cell_type': display}
        for (ds, label), display in T_MAPPING.items()
    ]).to_csv(BASE / 'data/T_CELL_DISPLAY_MAPPING.csv', index=False)
    export = md[['cell_key', 'dataset', 'platform', 'patient_id',
                 'provider_cell_type', 'broad_type', 'display_cell_type']].copy()
    export[['RAMP_UMAP1', 'RAMP_UMAP2']] = after
    export.to_csv(BASE / 'data/PLOT_DATA.csv.gz', index=False,
                  compression={'method': 'gzip', 'compresslevel': 3, 'mtime': 0})

    order = np.random.default_rng(42).permutation(len(md))
    types = md.display_cell_type.to_numpy()[order]
    bottom = types == 'Tumor / epithelial'
    upper_labels = ['Fibroblast / stroma', 'Myeloid / APC', 'CD4 T', 'Endothelial']
    middle = ~(bottom | np.isin(types, upper_labels))
    masks = [bottom, middle] + [types == label for label in upper_labels]
    type_order = np.concatenate([order[mask] for mask in masks])
    rendered_types = md.display_cell_type.to_numpy()[type_order]
    for lower in ['Fibroblast / stroma', 'Myeloid / APC']:
        assert np.flatnonzero(rendered_types == lower).max() < np.flatnonzero(
            rendered_types == 'CD4 T').min()
    layers = [
        {'label': label, 'n_cells': int(mask.sum())}
        for label, mask in zip(['Tumor / epithelial', 'Other identities', *upper_labels], masks)
    ]

    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 8, 'svg.fonttype': 'none',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.labelcolor': '#43515a', 'axes.edgecolor': '#9ba8b0',
        'axes.linewidth': .7, 'savefig.facecolor': 'white',
    })
    # Explicit inch geometry reserves only the space actually needed by each side legend.
    figure_w, figure_h = 6.55, 2.75
    plot_w, plot_bottom = 2.12, .25
    limits = bounds(after)
    plot_h = plot_w * ((limits[1][1] - limits[1][0]) /
                       (limits[0][1] - limits[0][0]))
    plot_left = [.22, 3.17]
    fig = plt.figure(figsize=(figure_w, figure_h))
    axes = [fig.add_axes([left / figure_w, plot_bottom / figure_h,
                          plot_w / figure_w, plot_h / figure_h])
            for left in plot_left]
    plots = [
        scatter(axes[0], after, md.platform, PLATFORM_COLORS, order),
        scatter(axes[1], after, md.display_cell_type, TYPE_COLORS, type_order),
    ]
    for ax, title in zip(axes, ['Platform', 'Cell type']):
        style_axis(ax, after, title)
    fig.text(.011, .960, 'B', fontsize=22, fontweight='bold', va='top', color='#243745')
    fig.text(.084, .957, 'Cross-platform alignment', fontsize=12.3,
             fontweight='bold', va='top', color='#243745')

    # Both legends are vertical, to the right, and use identical typography and spacing.
    legend_options = dict(loc='center left', bbox_to_anchor=(1.02, .5),
                          ncol=1, frameon=False, fontsize=7.8, handlelength=.85,
                          handletextpad=.40, labelspacing=.60, borderaxespad=0,
                          borderpad=0)
    platform_legend = axes[0].legend(handles=handles(PLATFORM_COLORS), **legend_options)
    cell_legend = axes[1].legend(handles=handles(TYPE_COLORS), **legend_options)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist in [platform_legend, cell_legend, *fig.texts]:
        box = artist.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= 1 and box.y1 <= 1, box
    first_legend_box = platform_legend.get_window_extent(renderer)
    second_y_label_box = axes[1].yaxis.label.get_window_extent(renderer)
    assert first_legend_box.x1 + 1 < second_y_label_box.x0
    for ax, legend in zip(axes, [platform_legend, cell_legend]):
        assert legend.get_window_extent(renderer).x0 > ax.get_window_extent(renderer).x1
    assert np.array_equal(np.asarray(plots[0].get_offsets()), after[order])
    assert np.array_equal(np.asarray(plots[1].get_offsets()), after[type_order])
    assert axes[0].get_xlim() == axes[1].get_xlim()
    assert axes[0].get_ylim() == axes[1].get_ylim()
    for ext in ['png', 'svg']:
        fig.savefig(BASE / f'panel_B_alignment.{ext}', dpi=300)
    plt.close(fig)

    source_checks = []
    for record in source_records:
        assert sha(BASE / record['local']) == record['sha256']
        source = Path(record['source'])
        if source.exists():
            assert sha(source) == record['sha256'], source
        source_checks.append({'source': str(source), 'checked': source.exists()})
    report = {
        'status': 'PASS', 'version': 'v4_CD4_top_simplified_groups', 'n_views': 2,
        'views': ['Aligned UMAP by platform', 'Aligned UMAP by cell type'],
        'baseline_displayed': False, 'n_cells_per_panel': 84000,
        'n_patients_per_dataset': 28, 'n_cells_per_patient': 1000,
        'display_cell_type_counts': expected_counts,
        'DNT_display_rule': 'Zheng DNT -> Others; original annotations unchanged',
        'DNT_n_cells': int(dnt.sum()), 'source_annotations_preserved': True,
        'additional_Others_groups': ['B cell', 'NK cell', 'Benign epithelium'],
        'additional_Others_n_cells': int(pooled.sum()), 'n_display_groups': len(TYPE_COLORS),
        'CD4_above_fibroblast_and_myeloid': True,
        'new_gating': False, 'coordinate_refit': False, 'prediction_changes': False,
        'plotted_coordinates_exact': True, 'all_cells_plotted_in_every_view': True,
        'after_platform_and_type_axes_identical': True,
        'coordinate_limits': {'aligned_both_views': bounds(after)},
        'layer_order_bottom_to_top': layers, 'draw_order_seed': 42,
        'platform_palette': PLATFORM_COLORS, 'cell_type_palette': TYPE_COLORS,
        'source_checks': source_checks, 'text_and_legend_within_canvas': True,
        'figure_size_inches': [figure_w, figure_h], 'dpi': 300,
        'plot_left_inches': plot_left, 'plot_bottom_inches': plot_bottom,
        'plot_size_inches': [plot_w, plot_h],
        'legend_position': 'right of each UMAP, vertically centered, one column',
        'legend_font_size_pt': 7.8, 'side_legends_do_not_overlap_plots_or_labels': True,
        'outputs': ['panel_B_alignment.png', 'panel_B_alignment.svg'],
    }
    save_json(BASE / 'provenance/VERIFICATION.json', report)
    manifest = []
    for path in sorted(BASE.rglob('*')):
        relative = path.relative_to(BASE)
        if not path.is_file() or relative.parts[0] in {'cache', 'tmp', 'cd4_cd8_check', 'versions'}:
            continue
        if relative.as_posix() == 'provenance/MANIFEST.json':
            continue
        manifest.append({'file': relative.as_posix(), 'bytes': path.stat().st_size,
                         'sha256': sha(path)})
    save_json(BASE / 'provenance/MANIFEST.json', manifest)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
