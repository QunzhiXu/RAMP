"""Compare the same CAF cells using the original full Cords protein panel."""
from runtime import BASE, STORY
from pathlib import Path
import json
import hashlib
import time
import importlib.metadata as im

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PRIOR = BASE.parent / 'caf_umap'
COLORS = {'Other CAF': '#9869b0', 'vCAF': '#e54285'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def neighbor_metrics(x, md, space, method):
    indices = NearestNeighbors(n_neighbors=30, n_jobs=1).fit(x).kneighbors(return_distance=False)
    assert indices.shape == (len(md), 30)
    assert not np.equal(indices, np.arange(len(md))[:, None]).any()
    labels = md.display_group.to_numpy()
    patient = md.patient_id.to_numpy()
    same_type = (labels[indices] == labels[:, None]).mean(1)
    same_patient = (patient[indices] == patient[:, None]).mean(1)
    cell = pd.DataFrame({'cell_key': md.cell_key, 'patient_id': patient,
                         'display_group': labels, 'same_type_neighbor_fraction': same_type,
                         'same_patient_neighbor_fraction': same_patient})
    cell.to_csv(BASE / 'results' / f'{method}_{space}_NEIGHBORS.csv.gz', index=False,
                compression={'method': 'gzip', 'compresslevel': 3, 'mtime': 0})
    rows = []
    for group in COLORS:
        subset = cell[cell.display_group.eq(group)]
        rows.append({'method': method, 'space': space, 'group': group, 'k': 30,
                     'n_cells': len(subset), 'n_patients': subset.patient_id.nunique(),
                     'same_type_fraction': subset.same_type_neighbor_fraction.mean(),
                     'patient_equal_mean_same_type_fraction': subset.groupby('patient_id').same_type_neighbor_fraction.mean().mean(),
                     'same_patient_fraction': subset.same_patient_neighbor_fraction.mean()})
    return rows


def draw(ax, u, md, title, show_legend=True):
    lo, hi = u.min(0), u.max(0)
    pad = (hi - lo) * .04
    ax.set(xlim=(lo[0] - pad[0], hi[0] + pad[0]),
           ylim=(lo[1] - pad[1], hi[1] + pad[1]), xlabel='UMAP 1', ylabel='UMAP 2')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=9, color='#243745')
    all_ix = []
    for zorder, (group, color) in enumerate(COLORS.items(), start=1):
        ix = np.random.default_rng(42).permutation(np.flatnonzero(md.display_group.eq(group)))
        artist = ax.scatter(u[ix, 0], u[ix, 1], c=color, s=2.8,
                            alpha=.78, linewidths=0, rasterized=True, zorder=zorder)
        assert np.array_equal(np.asarray(artist.get_offsets()), u[ix])
        all_ix.extend(ix)
    assert np.array_equal(np.sort(all_ix), np.arange(len(md)))
    if show_legend:
        return ax.legend(
            handles=[Line2D([], [], marker='o', linestyle='', color=color,
                            markersize=5, label=group) for group, color in COLORS.items()],
            loc='center left', bbox_to_anchor=(1.025, .5), frameon=False,
            fontsize=8.5, handlelength=.9, handletextpad=.35, labelspacing=.8,
            borderaxespad=0, borderpad=0)


def save_plot(fig, name):
    fig.canvas.draw()
    for ext in ['png', 'svg']:
        fig.savefig(BASE / f'{name}.{ext}', dpi=300)
    plt.close(fig)


def main():
    export_report = json.loads((BASE / 'provenance/EXPORT_VERIFICATION.json').read_text())
    assert export_report['status'] == 'PASS'
    request = pd.read_csv(BASE / 'data/CELL_REQUEST.csv', dtype={'patient_id': str, 'roi_id': str})
    prior_md = pd.read_csv(PRIOR / 'data/CAF_METADATA.csv.gz', dtype={'patient_id': str, 'roi_id': str})
    assert request.cell_key.equals(prior_md.cell_key)
    assert len(request) == 5206
    md = request.copy()
    md['display_group'] = np.where(md.provider_cell_type.eq('vCAF'), 'vCAF', 'Other CAF')
    assert md.display_group.value_counts().to_dict() == {'Other CAF': 4560, 'vCAF': 646}

    assay_records = export_report['assays']
    assays = {}
    for name in ['counts', 'c_counts', 'c_counts_asinh']:
        record = assay_records[name]
        assert sha(BASE / record['file']) == record['sha256']
        frame = pd.read_csv(BASE / record['file'])
        assert frame.cell_key.equals(md.cell_key)
        assert frame.source_row_id.equals(md.source_row_id)
        assays[name] = frame
    marker_info = pd.read_csv(BASE / 'data/ALL_MARKERS.csv')
    included = marker_info.loc[marker_info.used_as_protein, 'provider_marker'].tolist()
    assert len(included) == 41 and {'CD146', 'CD34'}.issubset(included)
    author_values = assays['c_counts_asinh'][included].to_numpy()
    clipped_counts = assays['c_counts'][included].to_numpy()
    transform_error = float(np.max(np.abs(author_values - np.arcsinh(clipped_counts))))
    assert transform_error < 1e-10

    # Cross-check the new provider extraction against the existing nine raw marker values.
    mapping = ['CD3', 'CD4', 'CD8a', 'vWF + CD31', 'Ki-67', 'Vimentin',
               'Pan Cytokeratin + Keratin Epithelial', 'CD45RA + CD45R0', 'HLA-DR']
    parent_indices = np.load(PRIOR / 'data/PARENT_ROW_INDEX.npy')
    reference_raw = np.load(STORY / 'data_umap/data/RAW_SHARED9.npy')[parent_indices]
    raw_error = float(np.max(np.abs(assays['counts'][mapping].to_numpy() - reference_raw)))
    assert raw_error < 1e-8, raw_error

    scaler = StandardScaler()
    full = scaler.fit_transform(author_values)
    assert full.shape == (5206, 41) and np.isfinite(full).all()
    assert np.all(scaler.var_ > 0)
    np.save(BASE / 'data/FULL41_AUTHOR_ASINH.npy', author_values)
    np.save(BASE / 'data/FULL41_STANDARDIZED.npy', full)
    pd.DataFrame({'column_index': np.arange(41), 'provider_marker': included,
                  'mean': scaler.mean_, 'scale': scaler.scale_}).to_csv(
                      BASE / 'data/UMAP_MARKERS.csv', index=False)
    prior_config = json.loads((PRIOR / 'results/UMAP_CONFIG.json').read_text())
    params = prior_config['parameters']
    coordinate_file = BASE / 'results/FULL41_UMAP.npy'
    config_file = BASE / 'results/FULL41_UMAP_CONFIG.json'
    if coordinate_file.exists():
        config = json.loads(config_file.read_text())
        assert config['input_sha256'] == sha(BASE / 'data/FULL41_STANDARDIZED.npy')
        assert config['parameters'] == params
        assert config['coordinate_sha256'] == sha(coordinate_file)
        u = np.load(coordinate_file)
    else:
        print('Fitting UMAP: same 5,206 CAF, original 41-protein panel, seed 42.', flush=True)
        import umap
        start = time.time()
        u = umap.UMAP(**params).fit_transform(full)
        np.save(coordinate_file, u)
        config = {
            'parameters': params, 'n_cells': 5206, 'n_proteins': 41,
            'provider_assay': 'c_counts_asinh',
            'preprocessing': 'Author-provided censored counts + asinh(cofactor=1); StandardScaler across selected CAF',
            'RAMP_used_for_full_panel': False, 'RAMP_refit': False,
            'HLA_predictions_changed': False, 'labels_used_for_embedding': False,
            'excluded_channels': ['Iridium_191', 'Iridium_193'],
            'all_other_protein_channels_included': True,
            'CD146_CD34_included': True, 'parameter_search': False,
            'seconds': time.time() - start,
            'input_sha256': sha(BASE / 'data/FULL41_STANDARDIZED.npy'),
            'coordinate_sha256': sha(coordinate_file),
            'environment': {name: im.version(name) for name in
                            ['numpy', 'scikit-learn', 'umap-learn', 'numba', 'pynndescent']},
        }
        save_json(config_file, config)
        print(f"UMAP complete in {config['seconds']:.1f}s.", flush=True)
    assert u.shape == (5206, 2) and np.isfinite(u).all()
    xy = md[['cell_key', 'source_row_id', 'patient_id', 'roi_id', 'provider_cell_type', 'display_group']].copy()
    xy[['UMAP1', 'UMAP2']] = u
    xy.to_csv(BASE / 'results/FULL41_COORDINATES.csv.gz', index=False,
              compression={'method': 'gzip', 'compresslevel': 3, 'mtime': 0})

    print('Comparing 30-neighbor cell-type enrichment in expression and UMAP space.', flush=True)
    old_x = np.load(PRIOR / 'data/CAF_STANDARDIZED.npy')
    old_u = np.load(PRIOR / 'results/CAF_UMAP.npy')
    rows = []
    for method, x, coords in [('RAMP_shared9', old_x, old_u), ('Cords_full41', full, u)]:
        rows.extend(neighbor_metrics(x, md, 'expression', method))
        rows.extend(neighbor_metrics(coords, md, 'UMAP', method))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(BASE / 'results/SEPARATION_METRICS.csv', index=False)

    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 9, 'svg.fonttype': 'none',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.labelcolor': '#43515a', 'axes.edgecolor': '#9ba8b0',
        'axes.linewidth': .7, 'savefig.facecolor': 'white',
    })
    fig, ax = plt.subplots(figsize=(5.4, 3.65))
    fig.subplots_adjust(left=.09, right=.78, top=.78, bottom=.14)
    draw(ax, u, md, '41 measured protein channels')
    fig.text(.065, .967, 'CAF-only UMAP · full panel', fontsize=14,
             fontweight='bold', color='#243745', va='top')
    fig.text(.065, .875, 'Cords IMC · 5,206 CAF', fontsize=9.5, color='#56646d')
    save_plot(fig, 'FULL41_CAF_UMAP')

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))
    fig.subplots_adjust(left=.045, right=.885, top=.80, bottom=.13, wspace=.60)
    for ax, coords, title in zip(axes, [old_u, u],
                                ['Previous shared 9 markers', 'Cords full 41 proteins']):
        draw(ax, coords, md, title)
    fig.suptitle('CAF separation on the same 5,206 cells', x=.04, y=.97,
                 ha='left', fontsize=15, fontweight='bold', color='#243745')
    save_plot(fig, 'SHARED9_VS_FULL41')

    source_paths = [BASE / 'provenance/EXPORT_REQUEST.json',
                    PRIOR / 'data/CAF_METADATA.csv.gz', PRIOR / 'data/CAF_STANDARDIZED.npy',
                    PRIOR / 'results/CAF_UMAP.npy', PRIOR / 'results/UMAP_CONFIG.json',
                    STORY / 'data_umap/data/RAW_SHARED9.npy']
    save_json(BASE / 'provenance/ANALYSIS_SOURCES.json', [
        {'path': str(path), 'sha256': sha(path)} for path in source_paths])
    report = {
        'status': 'PASS', 'n_cells': 5206, 'n_vCAF': 646, 'n_other_CAF': 4560,
        'n_patients': 28, 'n_proteins': 41, 'same_cells_and_order_as_previous_CAF_UMAP': True,
        'new_provider_assay': 'c_counts_asinh', 'author_asinh_reconstruction_max_error': transform_error,
        'new_counts_vs_original_raw_shared9_max_error': raw_error,
        'all_selected_cells_plotted': True, 'labels_not_used_to_fit_embedding': True,
        'full_panel_contains_original_annotation_markers': ['CD146', 'CD34'],
        'classification_definition_display_not_independent_validation': True,
        'vCAF_neighbor_results': metrics.loc[metrics.group.eq('vCAF')].to_dict(orient='records'),
        'outputs': ['FULL41_CAF_UMAP.png', 'FULL41_CAF_UMAP.svg',
                    'SHARED9_VS_FULL41.png', 'SHARED9_VS_FULL41.svg'],
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
