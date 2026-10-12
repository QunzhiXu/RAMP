"""Draw three existing UMAP views; split CD4/CD8 and pool DNT into Others for display."""
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
    'B cell': '#6686cd',
    'Myeloid / APC': '#d0a02e',
    'Fibroblast / stroma': '#c35199',
    'Endothelial': '#46aab6',
    'NK cell': '#8ba249',
    'Benign epithelium': '#e6b1a1',
    'Others': '#bcbfc5',
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
    'data/baseline_UMAP.npy': 'umap/results/baseline_UMAP.npy',
    'data/RAMP_UMAP.npy': 'umap/results/RAMP_UMAP.npy',
    'data/CELL_METADATA.csv.gz': 'data_umap/data/CELL_METADATA.csv.gz',
    'provenance/SELECTED_DONORS.csv': 'data_umap/data/SELECTED_DONORS.csv',
    'provenance/MARKER_MAPPING.csv': 'data_umap/data/MARKER_MAPPING.csv',
    'provenance/baseline_UMAP_PARAMETERS.json': 'umap/results/baseline_UMAP.json',
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
        records = json.loads(record_file.read_text())
        for item in records:
            assert sha(BASE / item['local']) == item['sha256'], item['local']
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
                   markersize=5.5, label=label) for label, color in palette.items()]


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
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10, color='#243745')


def scatter(ax, u, values, colors, ix):
    assert len(ix) == len(u) and len(np.unique(ix)) == len(u)
    mapped = pd.Series(values).map(colors).to_numpy()[ix]
    assert not pd.isna(mapped).any()
    return ax.scatter(u[ix, 0], u[ix, 1], c=mapped, s=.60, alpha=.65,
                      linewidths=0, rasterized=True)


def main():
    source_records = stage_inputs()
    md = pd.read_csv(BASE / 'data/CELL_METADATA.csv.gz', low_memory=False)
    before = np.load(BASE / 'data/baseline_UMAP.npy')
    after = np.load(BASE / 'data/RAMP_UMAP.npy')
    assert before.shape == after.shape == (84000, 2)
    assert len(md) == len(after) and md.cell_key.is_unique
    assert np.isfinite(before).all() and np.isfinite(after).all()
    assert md.groupby(['dataset', 'patient_id']).size().eq(1000).all()
    assert md.groupby('dataset').patient_id.nunique().eq(28).all()
    assert md.groupby('dataset').size().eq(28000).all()

    md['platform'] = md.dataset.map(PLATFORMS)
    md['display_cell_type'] = md.broad_type.replace({'Other / uncertain': 'Others'})
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
        'Endothelial': 5252, 'B cell': 1000, 'NK cell': 849,
        'Benign epithelium': 818, 'Others': 8167,
    }
    assert md.display_cell_type.value_counts().to_dict() == expected_counts
    md.groupby(['dataset', 'platform', 'display_cell_type']).size().rename(
        'n_cells').reset_index().to_csv(BASE / 'results/CELL_COUNTS.csv', index=False)
    pd.DataFrame([
        {'dataset': ds, 'provider_cell_type': label, 'display_cell_type': display}
        for (ds, label), display in T_MAPPING.items()
    ]).to_csv(BASE / 'data/T_CELL_DISPLAY_MAPPING.csv', index=False)
    export = md[['cell_key', 'dataset', 'platform', 'patient_id',
                 'provider_cell_type', 'broad_type', 'display_cell_type']].copy()
    export[['baseline_UMAP1', 'baseline_UMAP2']] = before
    export[['RAMP_UMAP1', 'RAMP_UMAP2']] = after
    export.to_csv(BASE / 'data/PLOT_DATA.csv.gz', index=False,
                  compression={'method': 'gzip', 'compresslevel': 3, 'mtime': 0})

    order = np.random.default_rng(42).permutation(len(md))
    types = md.display_cell_type.to_numpy()[order]
    bottom = types == 'Tumor / epithelial'
    upper_labels = ['Fibroblast / stroma', 'Myeloid / APC', 'Endothelial']
    middle = ~(bottom | np.isin(types, upper_labels))
    masks = [bottom, middle] + [types == label for label in upper_labels]
    type_order = np.concatenate([order[mask] for mask in masks])
    layers = [
        {'label': label, 'n_cells': int(mask.sum())}
        for label, mask in zip(['Tumor / epithelial', 'Other identities', *upper_labels], masks)
    ]

    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 9, 'svg.fonttype': 'none',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.labelcolor': '#43515a', 'axes.edgecolor': '#9ba8b0',
        'axes.linewidth': .7, 'savefig.facecolor': 'white',
    })
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 4.45))
    fig.subplots_adjust(left=.055, right=.99, bottom=.26, top=.81, wspace=.19)
    plots = [
        scatter(axes[0], before, md.platform, PLATFORM_COLORS, order),
        scatter(axes[1], after, md.platform, PLATFORM_COLORS, order),
        scatter(axes[2], after, md.display_cell_type, TYPE_COLORS, type_order),
    ]
    for ax, u, title in zip(axes, [before, after, after],
                            ['Before RAMP', 'After RAMP', 'Cell type']):
        style_axis(ax, u, title)
    fig.text(.014, .966, 'B', fontsize=22, fontweight='bold', va='top', color='#243745')
    fig.text(.057, .962, 'Cross-platform alignment', fontsize=17,
             fontweight='bold', va='top', color='#243745')

    # The platform legend belongs to the first two views; cell identities span two rows below.
    platform_legend = fig.legend(
        handles=handles(PLATFORM_COLORS), loc='lower center', bbox_to_anchor=(.358, .142),
        ncol=3, frameon=False, fontsize=9.5, handletextpad=.35, columnspacing=1.6,
    )
    # Matplotlib fills columns first; interleave to preserve the intended reading order.
    labels = list(TYPE_COLORS)
    legend_order = [labels[i] for pair in zip(range(5), range(5, 10)) for i in pair]
    cell_legend = fig.legend(
        handles=handles({key: TYPE_COLORS[key] for key in legend_order}),
        loc='lower center', bbox_to_anchor=(.52, .008), ncol=5,
        frameon=False, fontsize=8.5, handletextpad=.30, columnspacing=1.45, labelspacing=.65,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist in [platform_legend, cell_legend, *fig.texts]:
        box = artist.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= 1 and box.y1 <= 1, box
    assert np.array_equal(np.asarray(plots[0].get_offsets()), before[order])
    assert np.array_equal(np.asarray(plots[1].get_offsets()), after[order])
    assert np.array_equal(np.asarray(plots[2].get_offsets()), after[type_order])
    assert axes[1].get_xlim() == axes[2].get_xlim()
    assert axes[1].get_ylim() == axes[2].get_ylim()
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
        'status': 'PASS', 'n_cells_per_panel': 84000,
        'n_patients_per_dataset': 28, 'n_cells_per_patient': 1000,
        'display_cell_type_counts': expected_counts,
        'DNT_display_rule': 'Zheng DNT -> Others; original annotations unchanged',
        'DNT_n_cells': int(dnt.sum()), 'source_annotations_preserved': True,
        'new_gating': False, 'coordinate_refit': False, 'prediction_changes': False,
        'plotted_coordinates_exact': True, 'all_cells_plotted_in_every_view': True,
        'after_platform_and_type_axes_identical': True,
        'coordinate_limits': {'before': bounds(before), 'after': bounds(after)},
        'layer_order_bottom_to_top': layers, 'draw_order_seed': 42,
        'platform_palette': PLATFORM_COLORS, 'cell_type_palette': TYPE_COLORS,
        'source_checks': source_checks, 'text_and_legend_within_canvas': True,
        'figure_size_inches': [10.8, 4.45], 'dpi': 300,
        'outputs': ['panel_B_alignment.png', 'panel_B_alignment.svg'],
    }
    save_json(BASE / 'provenance/VERIFICATION.json', report)
    manifest = []
    for path in sorted(BASE.rglob('*')):
        relative = path.relative_to(BASE)
        if not path.is_file() or relative.parts[0] in {'cache', 'tmp', 'cd4_cd8_check'}:
            continue
        if relative.as_posix() == 'provenance/MANIFEST.json':
            continue
        manifest.append({'file': relative.as_posix(), 'bytes': path.stat().st_size,
                         'sha256': sha(path)})
    save_json(BASE / 'provenance/MANIFEST.json', manifest)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
