"""Read-only content comparison, separate from execution and structural validation."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
import pandas as pd
from common import STORY, EXP, MARKERS, sha, read, write, record, reserve
from specs import CODES, INPUTS, marker_dir
from prepare import verify as verify_lock
from study import verify_run


def array_comparison(a, b):
    result = {'shape': list(a.shape), 'dtype': str(a.dtype), 'same_shape_dtype': a.shape == b.shape and a.dtype == b.dtype}
    if not result['same_shape_dtype']:
        return result | {'equal': False}
    equal = np.array_equal(a, b, equal_nan=True) if a.dtype.kind in 'fc' else np.array_equal(a, b)
    result['equal'] = bool(equal)
    if a.dtype.kind in 'biufc':
        result['nonfinite_new'] = int(np.sum(~np.isfinite(a)))
        result['max_abs_difference'] = float(np.max(np.abs(a.astype(float) - b.astype(float)), initial=0))
    return result


def table_comparison(a, b):
    a, b = pd.read_csv(a, float_precision='round_trip'), pd.read_csv(b, float_precision='round_trip')
    same_schema = list(a.columns) == list(b.columns) and a.shape == b.shape
    out = {'rows': len(a), 'columns': list(a.columns), 'same_schema': same_schema}
    if not same_schema:
        return out | {'equal': False}
    out['equal'] = a.equals(b)
    out['different_columns'] = [c for c in a if not a[c].equals(b[c])]
    numeric = [c for c in a if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c])]
    out['max_abs_numeric_difference'] = max([0.] + [float(np.nanmax(np.abs(a[c].to_numpy(float) - b[c].to_numpy(float)), initial=0)) for c in numeric])
    return out


def tree_differences(a, b, path=''):
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                out.append({'field': path + '/' + key, 'new': a.get(key), 'old': b.get(key)})
            else:
                out.extend(tree_differences(a[key], b[key], path + '/' + key))
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in tree_differences(x, y, path + '/' + str(i))]
    return [] if a == b else [{'field': path, 'new': a, 'old': b}]


def json_compare(a, b, new_root, old_root, hash_pairs):
    def relocate(x):
        if isinstance(x, dict):
            return {relocate(k): relocate(v) for k, v in x.items()}
        if isinstance(x, list):
            return [relocate(v) for v in x]
        if isinstance(x, str):
            return x.replace(str(new_root), '$TREE').replace(str(old_root), '$TREE')
        return x
    new, old = relocate(read(a)), relocate(read(b))
    differences = tree_differences(new, old)
    allowed, substantive = [], []
    for item in differences:
        key = item['field'].rsplit('/', 1)[-1]
        if key in ('time_utc', 'seconds', 'elapsed_seconds'):
            allowed.append(item | {'reason': 'runtime time'})
        elif isinstance(item['new'], str) and isinstance(item['old'], str) and (item['new'], item['old']) in hash_pairs:
            allowed.append(item | {'reason': 'hashes of separately compared corresponding files'})
        else:
            substantive.append(item)
    return {'equal_after_declared_normalization': not substantive, 'provenance_differences': allowed, 'other_differences': substantive}


def required_payload():
    paths = [f'{STORY}/prediction/output/{name}' for name in ('codex_cd19_prediction.npz', 'masked_marker_model.pt', 'codex_prediction_metadata.csv.gz')]
    for marker in MARKERS:
        paths += [f'{marker_dir(marker)}/{name}' for name in (
            f'{marker}_prediction.npz', f'{marker}_permanent_holdout_nn.pt', 'codex_prediction_metadata.csv.gz', 'donor_metrics.csv')]
    for condition in ('without_cd19', 'with_wrong_observed_cd19', 'with_predicted_cd19'):
        paths += [f'{STORY}/alignment/output/{condition}/{name}' for name in (
            'kmeanslow_baseline.npy', 'ramp_aligned.npy', 'kmeanslow_baseline.csv.gz', 'ramp_aligned.csv.gz',
            'coupling_audit.csv', 'strength_audit.csv', 'evidence_audit.csv', 'reference_weight_audit.csv', 'profile_audit.csv', 'algorithm_audit.csv')]
    paths += [f'{STORY}/umap/data/{name}' for name in ('umap_coordinates.npz', 'shared_umap_initialization.npy', 'metric_row_sets.npz', 'metrics.csv')]
    paths += [f'{EXP}/{name}' for name in (
        'CD3_NN_UMAP_validation/ramp13_all.npy', 'CD3_NN_UMAP_validation/joint_umap_all.npy',
        'CD3_NN_UMAP_validation/codex_umap.npy', 'CD4_NN_UMAP_validation/ramp13_codex.npy',
        'CD4_NN_UMAP_validation/umap_coordinates.npy',
        'validation_subpanels/all_marker_validation/all_14_markers_by_donor.csv',
        'validation_subpanels/donor_mean_four_metrics/donor_mean_metrics.csv',
        'all_donor_marker_review0913/data/all_14_markers_scatter_values.npz',
        'real_spatial0913/data/H14_full_predictions.csv.gz', 'real_spatial0913/data/frozen_preprocessing.json')]
    paths += [f'{STORY}/figure/panel_B/data/{name}' for name in ('all_marker_qc_scores.csv', 'codex_marker_qc_summary.csv', 'codex_top3_marker_donor_scores.csv')]
    return set(paths)


def compare(lockdir, run, output):
    run = Path(run).resolve()
    structure = verify_run(lockdir, run)
    lock = verify_lock(lockdir, source=True)
    original = Path(lock['source_root']); tree = run / 'tree'
    missing = [rel for rel in required_payload() if not (tree / rel).is_file() or rel not in lock['files']]
    if missing:
        raise ValueError(f'Expected scientific payload missing or unbound: {missing}')
    candidates = {rel for rel, r in lock['files'].items() if r['role'] == 'reference' and (tree / rel).is_file()}
    hash_pairs = {(sha(tree / rel), lock['files'][rel]['sha256']) for rel in candidates}
    results = {}
    for rel in sorted(candidates):
        new, old = tree / rel, original / rel
        row = {'new_sha256': sha(new), 'old_sha256': sha(old), 'bytes_equal': sha(new) == sha(old)}
        if new.suffix == '.npy':
            row.update(kind='array', comparison=array_comparison(np.load(new, allow_pickle=False), np.load(old, allow_pickle=False)))
        elif new.suffix == '.npz':
            with np.load(new, allow_pickle=False) as a, np.load(old, allow_pickle=False) as b:
                if set(a.files) != set(b.files):
                    raise ValueError(f'Array keys differ: {rel}')
                row.update(kind='arrays', comparison={key: array_comparison(a[key], b[key]) for key in a.files})
        elif new.suffix == '.pt':
            import torch
            a = torch.load(new, weights_only=True, map_location='cpu'); b = torch.load(old, weights_only=True, map_location='cpu')
            if a.keys() != b.keys():
                raise ValueError(f'Model state keys differ: {rel}')
            row.update(kind='model', comparison={key: array_comparison(a[key].numpy(), b[key].numpy()) for key in a})
        elif new.name.endswith('.csv') or new.name.endswith('.csv.gz'):
            row.update(kind='table', comparison=table_comparison(new, old))
            if new.suffix == '.gz':
                with gzip.open(new, 'rb') as a, gzip.open(old, 'rb') as b:
                    row['decompressed_bytes_equal'] = a.read() == b.read()
        elif new.suffix == '.json':
            row.update(kind='json', comparison=json_compare(new, old, tree, original, hash_pairs))
        else:
            raise ValueError(f'Unhandled comparison: {rel}')
        results[rel] = row
    def equal(r):
        if r['kind'] in ('array', 'table'):
            return r['comparison']['equal']
        if r['kind'] in ('arrays', 'model'):
            return all(v['equal'] for v in r['comparison'].values())
        return r['comparison']['equal_after_declared_normalization']
    differing = [rel for rel, result in results.items() if not equal(result)]
    # Differences are reported, never tuned away. Coordinate/layout differences are
    # not automatically proof of a scientific error or silently treated as exact.
    report = {'status': 'exact_compared_content' if not differing else 'completed_with_differences',
              'structural': structure, 'files_compared': len(results),
              'byte_identical_files': sum(r['bytes_equal'] for r in results.values()),
              'differing_files': differing, 'results': results,
              'lock_sha256': sha(Path(lockdir) / 'LOCK.json'), 'run_manifest_sha256': sha(run / 'RUN_MANIFEST.json')}
    output = reserve(output, [run, original, Path(lockdir)])
    write(output / 'report.json', report)
    write(output / 'COMPLETE.json', {'status': 'complete', 'report_sha256': sha(output / 'report.json')})
    return {k: v for k, v in report.items() if k != 'results'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--lockdir', required=True); p.add_argument('--run', required=True); p.add_argument('--out', required=True)
    a = p.parse_args(); print(json.dumps(compare(a.lockdir, a.run, a.out), indent=2))
