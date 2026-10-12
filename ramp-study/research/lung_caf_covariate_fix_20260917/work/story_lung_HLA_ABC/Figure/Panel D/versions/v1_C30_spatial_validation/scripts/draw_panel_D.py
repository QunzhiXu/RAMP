"""Panel D: the approved Zheng C30 measured/predicted spatial validation pair."""
from runtime import BASE, STORY
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

VALIDATION = STORY / 'validation'
SOURCES = {
    'data/C30_CELLS.csv.gz': VALIDATION / 'results/C30_CELLS.csv.gz',
    'provenance/C30_DISPLAY.json': VALIDATION / 'results/C30_DISPLAY.json',
    'provenance/C30_SELECTION.json': VALIDATION / 'reference/C30_SELECTION.json',
    'provenance/C30_COLOR_SUMMARY.json': VALIDATION / 'reference/C30_COLOR_SUMMARY.json',
    'provenance/VALIDATION_RUN_MANIFEST.json': VALIDATION / 'results/RUN_MANIFEST.json',
    'provenance/PANEL_B_VERIFICATION.json': STORY / 'Figure/Panel B/provenance/VERIFICATION.json',
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')

def main():
    records = []
    for relative, source in SOURCES.items():
        local = BASE / relative
        if local.exists():
            assert sha(source) == sha(local), source
        else:
            shutil.copyfile(source, local)
        records.append({'source': str(source), 'local': relative, 'sha256': sha(source)})
    save_json(BASE / 'provenance/SOURCES.json', records)
    previous = json.loads((BASE / 'provenance/C30_DISPLAY.json').read_text())
    manifest = json.loads((BASE / 'provenance/VALIDATION_RUN_MANIFEST.json').read_text())
    # The copied C30 file is the existing frozen validation result.
    assert sha(BASE / 'data/C30_CELLS.csv.gz') in json.dumps(manifest)
    md = pd.read_csv(BASE / 'data/C30_CELLS.csv.gz')
    assert len(md) == previous['n_plotted_each_panel'] == 10719
    assert md.patient_id.eq('P_26').all() and md.core_id.eq('C_30').all()
    assert md.CellID.is_unique and md.source_row_id.is_unique
    cols = ['observed_HLA_A_patient_rank', 'RAMP_k30']
    assert np.isfinite(md[['x_um', 'y_um', *cols]].to_numpy()).all()
    rho = float(spearmanr(md[cols[0]], md[cols[1]]).statistic)
    assert abs(rho - previous['cell_spearman']) < 1e-12
    seed = int.from_bytes(hashlib.sha256(b'C_30').digest()[:8], 'little')
    plot = md.iloc[np.random.default_rng(seed).permutation(len(md))].copy()
    assert hashlib.sha256(plot.source_row_id.to_numpy(np.int64).tobytes()).hexdigest() == previous['source_row_ids_sha256']
    xy = plot[['x_um', 'y_um']].to_numpy()
    lo, hi = xy.min(0), xy.max(0)
    center = (lo + hi) / 2
    half = (hi - lo).max() / 2 + 65
    xlim = [center[0] - half, center[0] + half]
    ylim = [center[1] + half, center[1] - half]
    np.testing.assert_allclose(xlim, previous['xlim'], rtol=0, atol=1e-9)
    np.testing.assert_allclose(ylim, previous['ylim'], rtol=0, atol=1e-9)
    vmin, vmax = previous['color_vmin'], previous['color_vmax']
    assert [vmin, vmax] == [.25, .60]
    norm = Normalize(vmin, vmax, clip=True)
    plot[['source_row_id', 'CellID', 'patient_id', 'core_id', 'cell_type',
          'x_um', 'y_um', *cols]].to_csv(BASE / 'data/PLOT_DATA.csv.gz', index=False,
          compression={'method': 'gzip', 'compresslevel': 3, 'mtime': 0})

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'svg.fonttype': 'none', 'savefig.facecolor': 'white',
                         'text.color': '#243745', 'axes.labelcolor': '#43515a'})
    ref = json.loads((BASE / 'provenance/PANEL_B_VERIFICATION.json').read_text())
    side = ref['plot_size_inches'][1]
    bottom = ref['plot_bottom_inches']
    width, height = 4.90, 2.75
    fig = plt.figure(figsize=(width, height))
    fig.text(.017, .960, 'D', fontsize=22, fontweight='bold', va='top')
    fig.text(.50 / width, .957, 'Validation in mIF', fontsize=12.3, fontweight='bold', va='top')
    fig.text(3.12 / width, .949, f'C30 · ρ = {rho:.3f}', fontsize=8.0, va='top', color='#56646d')
    # Preserve the original marker size relative to the physical plotting area.
    original_side = 7.2 * (.81 - .225)
    point_size = previous['point_size_pt2'] * (side / original_side) ** 2
    axes, artists, scale_labels = [], [], []
    for left, col, title in zip([.14, 2.20], cols, ['Measured HLA-A', 'Predicted HLA-A proxy']):
        ax = fig.add_axes([left / width, bottom / height, side / width, side / height])
        artist = ax.scatter(xy[:, 0], xy[:, 1], c=plot[col].to_numpy(), cmap='viridis',
                            norm=norm, s=point_size, linewidths=0, edgecolors='none',
                            alpha=1, rasterized=True)
        ax.set(xlim=xlim, ylim=ylim)
        ax.set_aspect('equal'); ax.axis('off')
        ax.set_title(title, fontsize=10, fontweight='bold', pad=7, color='#243745')
        bx, by = xlim[0] + 30, ylim[0] - 27
        ax.plot([bx, bx + 100], [by, by], color='#29313B', linewidth=1.2, solid_capstyle='butt')
        scale_labels.append(ax.text(bx + 50, by - 12, '100 µm', ha='center', va='bottom',
                                    fontsize=6.2, color='#29313B'))
        assert np.array_equal(np.asarray(artist.get_offsets()), xy)
        assert np.array_equal(np.asarray(artist.get_array()), plot[col].to_numpy())
        axes.append(ax); artists.append(artist)
    assert np.array_equal(artists[0].get_offsets(), artists[1].get_offsets())
    assert axes[0].get_xlim() == axes[1].get_xlim() and axes[0].get_ylim() == axes[1].get_ylim()
    cax = fig.add_axes([4.23 / width, .62 / height, .085 / width, 1.12 / height])
    cb = fig.colorbar(artists[0], cax=cax, orientation='vertical', ticks=[.25, .4, .6])
    cb.ax.set_yticklabels(['≤0.25', '0.4', '≥0.6'])
    cb.set_label('Relative rank', fontsize=8, labelpad=5)
    cb.ax.tick_params(labelsize=7.2, length=2, pad=2)
    cb.outline.set_linewidth(.45)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in [*fig.texts, *scale_labels, *(ax.title for ax in axes),
                 *cb.ax.get_yticklabels(), cb.ax.yaxis.label]:
        box = text.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        assert box.x0 >= 0 and box.x1 <= 1 and box.y0 >= 0 and box.y1 <= 1, (text.get_text(), box)
    for ext in ['png', 'svg']:
        fig.savefig(BASE / f'panel_D_validation.{ext}', dpi=300)
    plt.close(fig)

    for record in records:
        assert sha(Path(record['source'])) == sha(BASE / record['local']) == record['sha256']
    report = {
        'status': 'PASS', 'version': 'v1_C30_spatial_validation',
        'patient_id': 'P_26', 'core_id': 'C_30', 'n_cells_per_plot': len(md),
        'plot_titles': ['Measured HLA-A', 'Predicted HLA-A proxy'],
        'measured_column': cols[0], 'predicted_column': cols[1],
        'prediction_meaning': 'Transferred Senosain HLA-ABC within-patient rank, used as HLA-A proxy',
        'cell_spearman': rho, 'coordinates': 'Original Zheng spatial coordinates in micrometers',
        'same_cells_coordinates_order_and_limits_as_approved_C30': True,
        'xlim': xlim, 'ylim': ylim, 'y_axis_inverted_as_original': True,
        'new_prediction_or_refit': False, 'spatial_smoothing': False,
        'cell_selection_changed': False, 'numeric_values_changed': False,
        'cmap': 'viridis', 'color_range': [vmin, vmax], 'shared_color_normalization': True,
        'color_clipping_labelled_at_endpoints': True, 'new_core_or_color_search': False,
        'point_size_pt2': point_size, 'point_size_scaled_with_plot_area': True,
        'figure_size_inches': [width, height], 'plot_size_inches': [side, side],
        'plot_bottom_inches': bottom, 'same_plot_height_and_baseline_as_B_C': True,
        'scale_bar_um': 100, 'all_text_inside_canvas': True, 'source_files_unchanged': True,
    }
    save_json(BASE / 'provenance/VERIFICATION.json', report)
    save_json(BASE / 'provenance/MANIFEST.json', [
        {'file': str(p.relative_to(BASE)), 'sha256': sha(p)} for p in sorted(BASE.rglob('*'))
        if p.is_file() and p.relative_to(BASE).parts[0] not in {'cache', 'tmp', 'versions'}
        and p.name != 'MANIFEST.json'])
    print(json.dumps(report, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
