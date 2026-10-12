"""Fixed original Cords full-panel CAF UMAP with a vCAF definition callout."""
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

COLORS = {'Other CAF': '#9869b0', 'vCAF': '#e54285'}
ANALYSIS = BASE / 'caf_fullpanel'
SOURCES = {
    'data/FULL41_UMAP.npy': ANALYSIS / 'results/FULL41_UMAP.npy',
    'data/FULL41_COORDINATES.csv.gz': ANALYSIS / 'results/FULL41_COORDINATES.csv.gz',
    'provenance/FULL41_UMAP_CONFIG.json': ANALYSIS / 'results/FULL41_UMAP_CONFIG.json',
    'provenance/FULL41_VERIFICATION.json': ANALYSIS / 'provenance/VERIFICATION.json',
    'provenance/PANEL_B_VERIFICATION.json': STORY / 'Figure/Panel B/provenance/VERIFICATION.json',
    'provenance/Cords2024_paper.md': STORY.parent / 'raw_data/lung_cancer/Cords2024/Paper_Cords_2024.md',
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')

def main():
    sources = []
    for relative, source in SOURCES.items():
        destination = BASE / relative
        if destination.exists():
            assert sha(destination) == sha(source), source
        else:
            shutil.copyfile(source, destination)
        sources.append({'source': str(source), 'local': relative, 'sha256': sha(source)})
    save_json(BASE / 'provenance/SOURCES.json', sources)
    md = pd.read_csv(BASE / 'data/FULL41_COORDINATES.csv.gz',
                     dtype={'patient_id': str, 'roi_id': str})
    u = np.load(BASE / 'data/FULL41_UMAP.npy')
    config = json.loads((BASE / 'provenance/FULL41_UMAP_CONFIG.json').read_text())
    assert json.loads((BASE / 'provenance/FULL41_VERIFICATION.json').read_text())['status'] == 'PASS'
    assert config['coordinate_sha256'] == sha(BASE / 'data/FULL41_UMAP.npy')
    assert config['n_proteins'] == 41 and not config['RAMP_used_for_full_panel']
    assert u.shape == (5206, 2) and len(md) == 5206 and md.cell_key.is_unique
    assert np.isfinite(u).all()
    np.testing.assert_allclose(md[['UMAP1', 'UMAP2']].to_numpy(), u, atol=1e-6, rtol=0)
    counts = {key: int(md.display_group.eq(key).sum()) for key in COLORS}
    assert counts == {'Other CAF': 4560, 'vCAF': 646}
    vcaf = md.display_group.eq('vCAF').to_numpy()
    assert np.array_equal(vcaf, md.provider_cell_type.eq('vCAF').to_numpy())
    assert md.patient_id.nunique() == 28 and md.loc[vcaf, 'patient_id'].nunique() == 27
    md[['UMAP1', 'UMAP2']] = u
    md.to_csv(BASE / 'data/PLOT_DATA.csv.gz', index=False,
              compression={'method': 'gzip', 'compresslevel': 3, 'mtime': 0})
    md.groupby('display_group').size().rename('n_cells').reset_index().to_csv(
        BASE / 'data/GROUP_COUNTS.csv', index=False)
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 8, 'svg.fonttype': 'none',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.labelcolor': '#43515a', 'axes.edgecolor': '#9ba8b0',
        'axes.linewidth': .7, 'savefig.facecolor': 'white',
    })
    reference = json.loads((BASE / 'provenance/PANEL_B_VERIFICATION.json').read_text())
    plot_h = reference['plot_size_inches'][1]
    bottom = reference['plot_bottom_inches']
    lo, hi = u.min(0), u.max(0)
    pad = (hi - lo) * .04
    limits = np.column_stack((lo - pad, hi + pad))
    # Match B's height and preserve this embedding's own aspect ratio.
    plot_w = plot_h * float((limits[0, 1] - limits[0, 0]) /
                            (limits[1, 1] - limits[1, 0]))
    width, height, left = 3.40, 2.75, .30
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([left / width, bottom / height, plot_w / width, plot_h / height])
    ax.set(xlim=limits[0], ylim=limits[1], xlabel='UMAP 1', ylabel='UMAP 2')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('41-protein CAF UMAP', fontsize=10, fontweight='bold', pad=7, color='#243745')
    fig.text(.017, .960, 'C', fontsize=22, fontweight='bold', va='top', color='#243745')
    fig.text(.150, .957, 'vCAF in IMC', fontsize=12.3, fontweight='bold', va='top', color='#243745')
    all_indices = []
    for zorder, (group, color) in enumerate(COLORS.items(), start=1):
        indices = np.random.default_rng(42).permutation(np.flatnonzero(md.display_group.eq(group)))
        artist = ax.scatter(u[indices, 0], u[indices, 1], c=color, s=1.9,
                            alpha=.78, linewidths=0, rasterized=True, zorder=zorder)
        assert np.array_equal(np.asarray(artist.get_offsets()), u[indices])
        all_indices.append(indices)
    assert np.array_equal(np.sort(np.concatenate(all_indices)), np.arange(len(md)))
    # This visual callout never changes source labels or cell selection.
    center, ellipse_w, ellipse_h, angle = (7.15, .68), 4.3, 3.25, 28
    ring = Ellipse(center, ellipse_w, ellipse_h, angle=angle, facecolor='none',
                   edgecolor='#49545d', linewidth=.9, linestyle=(0, (3, 2)), zorder=4)
    ax.add_patch(ring)
    fig.add_artist(ConnectionPatch(
        xyA=(9.05, 1.48), coordsA=ax.transData,
        xyB=(2.25 / width, 1.03 / height), coordsB=fig.transFigure,
        arrowstyle='-|>', mutation_scale=9, linewidth=.9,
        color='#49545d', shrinkA=0, shrinkB=0, clip_on=False, zorder=5))
    text_x = 2.29 / width
    for y, text, size, weight, color in [
        (1.03, 'vCAF', 11, 'bold', COLORS['vCAF']),
        (.80, 'Vascular CAF', 8.3, 'normal', '#243745'),
        (.58, 'CD146⁺ / CD34⁻', 8.8, 'bold', '#243745'),
        (.35, 'Cords et al., 2024', 7, 'normal', '#67747c'),
    ]:
        fig.text(text_x, y / height, text, fontsize=size, fontweight=weight,
                 color=color, ha='left', va='center')
    legend = fig.legend(
        handles=[Line2D([], [], marker='o', linestyle='', color=color,
                        markersize=4.5, label=label) for label, color in COLORS.items()],
        loc='center left', bbox_to_anchor=(text_x, 1.66 / height),
        frameon=False, fontsize=7.8, handlelength=.85,
        handletextpad=.40, labelspacing=.60, borderaxespad=0, borderpad=0)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in [*fig.texts, legend, ax.title, ax.xaxis.label, ax.yaxis.label]:
        box = text.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        assert box.x0 >= 0 and box.x1 <= 1 and box.y0 >= 0 and box.y1 <= 1, box
    for ext in ['png', 'svg']:
        fig.savefig(BASE / f'panel_C_vcaf.{ext}', dpi=300)
    plt.close(fig)
    theta = np.deg2rad(angle)
    delta = u - center
    local_x = delta[:, 0] * np.cos(theta) + delta[:, 1] * np.sin(theta)
    local_y = -delta[:, 0] * np.sin(theta) + delta[:, 1] * np.cos(theta)
    in_ring = (local_x / (ellipse_w / 2)) ** 2 + (local_y / (ellipse_h / 2)) ** 2 <= 1
    for record in sources:
        assert sha(Path(record['source'])) == sha(BASE / record['local']) == record['sha256']
    report = {
        'status': 'PASS', 'version': 'v3_full41_CAF_definition',
        'n_cells': len(md), 'group_counts': counts, 'n_patients': 28, 'vCAF_n_patients': 27,
        'annotation_source': 'Original Cords IMC CAF labels',
        'embedding_source': 'Original Cords 41-protein CAF-only UMAP; fixed existing coordinates',
        'coordinate_refit': False, 'RAMP_alignment_used_for_this_UMAP': False,
        'layer_order_bottom_to_top': list(COLORS), 'group_colors': COLORS,
        'point_size': 1.9, 'point_alpha': .78, 'all_cells_shown': True,
        'scatter_coordinates_exact': True, 'same_plot_height_and_baseline_as_Panel_B': True,
        'plot_size_inches': [plot_w, plot_h], 'plot_bottom_inches': bottom,
        'figure_size_inches': [width, height], 'coordinate_limits': limits.tolist(),
        'new_marker_gating': False, 'source_annotations_unchanged': True,
        'ellipse': {'center': center, 'width': ellipse_w, 'height': ellipse_h, 'angle': angle,
                    'role': 'Visual callout only; not used to select or define vCAF',
                    'vCAF_inside': int((in_ring & vcaf).sum()),
                    'vCAF_outside': int((~in_ring & vcaf).sum())},
        'paper_definition': 'vCAF (CD146+/CD34-) among author-annotated CAFs',
        'paper_url': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC10929690/',
        'text_and_legend_inside_canvas': True,
    }
    save_json(BASE / 'provenance/VERIFICATION.json', report)
    save_json(BASE / 'provenance/MANIFEST.json', [
        {'file': str(path.relative_to(BASE)), 'sha256': sha(path)}
        for path in sorted(BASE.rglob('*')) if path.is_file()
        and path.relative_to(BASE).parts[0] not in
        {'cache', 'tmp', 'versions', 'caf_umap', 'caf_fullpanel'}
        and path.name != 'MANIFEST.json'])
    print(json.dumps(report, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
