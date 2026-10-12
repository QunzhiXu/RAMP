"""Recompute UMAP on all sampled Cords CAF using aligned nine-marker expression."""
from runtime import BASE, STORY
from pathlib import Path
import importlib.metadata as im
import json
import hashlib
import time
import shutil

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def main():
    for name in ['data', 'results', 'provenance']:
        (BASE / name).mkdir(exist_ok=True)
    files = {
        'expression': STORY / 'data_umap/reference/RAMP_SHARED9.npy',
        'metadata': STORY / 'data_umap/data/CELL_METADATA.csv.gz',
        'markers': STORY / 'data_umap/data/MARKER_MAPPING.csv',
        'parent_parameters': STORY / 'umap/results/RAMP_UMAP.json',
    }
    hashes = {key: {'path': str(path), 'sha256': sha(path)} for key, path in files.items()}
    md = pd.read_csv(files['metadata'], low_memory=False)
    x = np.load(files['expression'])
    assert x.shape == (84000, 9) and len(md) == 84000 and md.cell_key.is_unique
    assert np.isfinite(x).all()
    mask = md.dataset.eq('Cords IMC') & md.cell_category.eq('Fibroblast')
    ids = np.flatnonzero(mask)
    sub = md.iloc[ids].copy().reset_index(drop=True)
    raw = x[ids]
    sub['display_group'] = np.where(sub.provider_cell_type.eq('vCAF'), 'vCAF', 'Other CAF')
    counts = sub.display_group.value_counts().to_dict()
    assert len(sub) == 5206 and counts == {'Other CAF': 4560, 'vCAF': 646}
    assert sub.patient_id.nunique() == 28
    params = json.loads(files['parent_parameters'].read_text())['parameters']
    scaler = StandardScaler()
    scaled = scaler.fit_transform(raw)
    assert scaled.shape == (5206, 9) and np.isfinite(scaled).all()

    np.save(BASE / 'data/PARENT_ROW_INDEX.npy', ids)
    np.save(BASE / 'data/CAF_RAMP_SHARED9.npy', raw)
    np.save(BASE / 'data/CAF_STANDARDIZED.npy', scaled)
    sub.to_csv(BASE / 'data/CAF_METADATA.csv.gz', index=False,
                compression={'method': 'gzip', 'compresslevel': 3, 'mtime': 0})
    shutil.copyfile(files['markers'], BASE / 'provenance/MARKER_MAPPING.csv')
    save_json(BASE / 'provenance/SOURCES.json', hashes)

    u_path = BASE / 'results/CAF_UMAP.npy'
    config_path = BASE / 'results/UMAP_CONFIG.json'
    if u_path.exists():
        config = json.loads(config_path.read_text())
        assert config['input_sha256'] == sha(BASE / 'data/CAF_STANDARDIZED.npy')
        assert config['parameters'] == params
        u = np.load(u_path)
        assert sha(u_path) == config['coordinate_sha256']
    else:
        print('Fitting one CAF-only UMAP: 5,206 CAF, shared nine markers, seed 42.', flush=True)
        import umap
        start = time.time()
        u = umap.UMAP(**params).fit_transform(scaled)
        seconds = time.time() - start
        np.save(u_path, u)
        config = {
            'parameters': params, 'n_cells': len(sub), 'n_markers': 9,
            'preprocessing': 'StandardScaler fitted to the 5,206 selected CAF; no PCA',
            'input_sha256': sha(BASE / 'data/CAF_STANDARDIZED.npy'),
            'coordinate_sha256': sha(u_path),
            'scale_mean': scaler.mean_.tolist(), 'scale_sd': scaler.scale_.tolist(),
            'seconds': seconds, 'RAMP_refit': False, 'UMAP_refit': True,
            'old_UMAP_coordinates_used': False, 'labels_used_for_embedding': False,
            'HLA_values_used': False, 'CD146_CD34_used_as_features': False,
            'parameter_search': False,
            'environment': {name: im.version(name) for name in
                            ['numpy', 'scikit-learn', 'umap-learn', 'numba', 'pynndescent']},
        }
        save_json(config_path, config)
        print(f'UMAP completed in {seconds:.1f} seconds.', flush=True)
    assert u.shape == (5206, 2) and np.isfinite(u).all()
    export = sub[['cell_key', 'dataset', 'patient_id', 'provider_cell_type', 'display_group']].copy()
    export[['CAF_UMAP1', 'CAF_UMAP2']] = u
    export.to_csv(BASE / 'results/CAF_UMAP_COORDINATES.csv.gz', index=False,
                  compression={'method': 'gzip', 'compresslevel': 3, 'mtime': 0})

    colors = {'Other CAF': '#9869b0', 'vCAF': '#e54285'}
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 9, 'svg.fonttype': 'none',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.labelcolor': '#43515a', 'axes.edgecolor': '#9ba8b0',
        'axes.linewidth': .7, 'savefig.facecolor': 'white',
    })
    fig, ax = plt.subplots(figsize=(5.4, 3.65))
    fig.subplots_adjust(left=.09, right=.74, top=.82, bottom=.14)
    lo, hi = u.min(0), u.max(0)
    pad = (hi - lo) * .04
    ax.set(xlim=(lo[0] - pad[0], hi[0] + pad[0]),
           ylim=(lo[1] - pad[1], hi[1] + pad[1]), xlabel='UMAP 1', ylabel='UMAP 2')
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal', adjustable='box')
    collections = []
    for zorder, (group, color) in enumerate(colors.items(), start=1):
        indices = np.random.default_rng(42).permutation(np.flatnonzero(sub.display_group.eq(group)))
        artist = ax.scatter(u[indices, 0], u[indices, 1], c=color, s=2.8,
                            alpha=.78, linewidths=0, rasterized=True, zorder=zorder)
        assert np.array_equal(np.asarray(artist.get_offsets()), u[indices])
        collections.append(artist)
    assert sum(len(artist.get_offsets()) for artist in collections) == len(sub)
    fig.text(.065, .967, 'CAF-only UMAP', fontsize=15, fontweight='bold',
             color='#243745', va='top')
    fig.text(.065, .875, 'Cords IMC · 5,206 CAF', fontsize=9.5, color='#56646d')
    legend = ax.legend(
        handles=[Line2D([], [], marker='o', linestyle='', color=color, markersize=5,
                        label=f'{group} (n = {counts[group]:,})') for group, color in colors.items()],
        loc='center left', bbox_to_anchor=(1.025, .5), frameon=False,
        fontsize=8.5, handlelength=.9, handletextpad=.35, labelspacing=.8,
        borderaxespad=0, borderpad=0,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist in [*fig.texts, legend]:
        box = artist.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        assert box.x0 >= 0 and box.x1 <= 1 and box.y0 >= 0 and box.y1 <= 1, box
    for ext in ['png', 'svg']:
        fig.savefig(BASE / f'CAF_UMAP.{ext}', dpi=300)
    plt.close(fig)

    for record in hashes.values():
        assert sha(record['path']) == record['sha256']
    report = {
        'status': 'PASS', 'n_cells': len(sub), 'n_patients': 28, 'group_counts': counts,
        'subset_exact': True, 'all_selected_CAF_plotted': True,
        'input': 'Existing RAMP-aligned nine-marker expression subset to Cords CAF',
        'new_UMAP': True, 'new_RAMP': False, 'old_two_dimensional_coordinates_used': False,
        'cell_labels_used_for_UMAP_fit': False, 'HLA_values_used': False,
        'draw_order_bottom_to_top': list(colors), 'source_files_unchanged': True,
        'same_point_size_and_alpha_for_both_groups': True,
        'outputs': ['CAF_UMAP.png', 'CAF_UMAP.svg', 'results/CAF_UMAP.npy'],
    }
    save_json(BASE / 'provenance/VERIFICATION.json', report)
    save_json(BASE / 'provenance/MANIFEST.json', [
        {'file': str(path.relative_to(BASE)), 'sha256': sha(path)}
        for path in sorted(BASE.rglob('*')) if path.is_file()
        and path.relative_to(BASE).parts[0] not in {'cache', 'tmp'}
        and path.name != 'MANIFEST.json'
    ])
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
