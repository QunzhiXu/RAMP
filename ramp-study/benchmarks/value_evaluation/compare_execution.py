"""Compare fresh value-evaluation artifacts, with explicit provenance-only exceptions."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import execution as e
import inventory as i
import contract as c

EXCEPTIONS = {
    i.PERCENTILE + '/computation_manifest.json': {'runtime_seconds', 'input_sha256'},
    i.RECON + '/manifest.json': {'elapsed_seconds', 'all_input_sha256'},
    i.RECON + '/value_evaluation/summary/manifest.json': {'elapsed_seconds', 'input_sha256', 'protocol_source'},
    **{i.VALUE + '/results/' + case + '/COMPLETE.json': {'elapsed_seconds', 'input_hashes'} for case in i.CASES},
    i.VALUE + '/summary/VERIFICATION.json': set(),
}


def numerical_difference(a, b):
    i.require(a.shape == b.shape and a.dtype == b.dtype, 'Array dimensions/dtype differ')
    i.require(np.array_equal(np.isnan(a), np.isnan(b)) and np.array_equal(np.isinf(a), np.isinf(b)), 'Nonfinite pattern differs')
    good = np.isfinite(a) & np.isfinite(b)
    return {'array_equal': bool(np.array_equal(a, b, equal_nan=True)),
            'max_absolute_difference': float(np.max(abs(a[good] - b[good]))) if good.any() else 0.}


def json_check(name, current, historical, mirror, source_root):
    i.require(name in EXCEPTIONS, 'Unclassified result JSON: ' + name)
    fields = EXCEPTIONS[name]
    a = i.read(current); b = i.read(historical)
    i.require(set(a) == set(b), 'Result JSON field set changed')
    left = {k: v for k, v in a.items() if k not in fields}
    right = {k: v for k, v in b.items() if k not in fields}
    i.require(left == right, 'Non-provenance JSON field changed: ' + name)
    linked = 0
    for field in ('input_sha256', 'all_input_sha256', 'input_hashes'):
        if field not in fields: continue
        new_links = {}; old_links = {}
        for path, sha in a[field].items():
            p = Path(path).resolve(strict=True)
            i.require(p.is_relative_to(mirror) and i.record(p)['sha256'] == sha, 'New stage read outside mirror or stale hash')
            new_links[p.relative_to(mirror).as_posix()] = sha
        for path, sha in b[field].items():
            p = i.bind(source_root, path)
            i.require(i.record(p)['sha256'] == sha, 'Historical stage link changed')
            old_links[p.relative_to(source_root).as_posix()] = sha
        i.require(set(new_links) == set(old_links), 'Stage consumed a different file set')
        linked += len(new_links)
        for path in new_links:
            if new_links[path] != old_links[path]:
                # Only a relocated fastMNN manifest or a regenerated gzip table
                # may have a different byte hash; their semantics are checked
                # separately in verify_run and the exhaustive artifact loop.
                i.require(path.endswith('.csv.gz') or path in
                    {f'benchmark_results/{case}/fastmnn/alignment/data/manifest.json' for case in i.CASES},
                    'Unclassified changed input hash: ' + path)
    if 'protocol_source' in fields:
        i.require(Path(a['protocol_source']).resolve() == mirror / i.VALUE / 'PROTOCOL.md' and
                  i.bind(source_root, b['protocol_source']) == source_root / i.VALUE / 'PROTOCOL.md', 'Protocol path differs')
    return {'allowed_fields': sorted(fields), 'exact_other_fields': True, 'verified_input_links': linked}


def compare(run, bundle, snapshot, source_root, output):
    run = Path(run).resolve(strict=True); source_root = Path(source_root).resolve(strict=True)
    verified = e.verify_run(run, bundle, snapshot); s = c.Sources(snapshot, source_root)
    mirror = run / 'runtime/tree'; inputs = e.verify_bundle(bundle, snapshot)['files']
    all_files = e.tree(mirror); produced = {name: rec for name, rec in all_files.items() if name not in inputs}
    numerical = sorted(name for name in produced if name.endswith(('.csv', '.csv.gz', '.npy', '.txt', '.R', '.json')))
    i.require(all(name.startswith((i.PERCENTILE + '/', i.RECON + '/', i.VALUE + '/')) for name in numerical), 'Unexpected output directory')
    rows = []; json_reports = {}
    for name in numerical:
        current = mirror / name; old = s.path(name)
        row = {'path': name, 'new': i.record(current), 'historical': i.record(old), 'byte_equal': i.record(current) == i.record(old)}
        if name.endswith('.npy'):
            row.update(kind='array', **numerical_difference(np.load(current, allow_pickle=False), np.load(old, allow_pickle=False)))
            i.require(row['array_equal'], 'Recomputed array differs: ' + name)
        elif name.endswith(('.csv', '.csv.gz')):
            a = pd.read_csv(current, float_precision='round_trip', low_memory=False)
            b = pd.read_csv(old, float_precision='round_trip', low_memory=False)
            pd.testing.assert_frame_equal(a, b, check_exact=True)
            row.update(kind='table', rows=len(a), columns=len(a.columns), exact=True)
        elif name.endswith('.json'):
            json_reports[name] = json_check(name, current, old, mirror, source_root)
            row.update(kind='provenance_json', exact_scientific_fields=True)
        else:
            i.require(row['byte_equal'], 'Text result differs: ' + name); row['kind'] = 'text'
        rows.append(row)
    # Grid checks are mandatory and not inferred just from the number of files.
    expected_case_files = {'label_counts.csv', 'gate_eligibility.csv', 'source_gates.csv', 'gate_target_unit_scores.csv',
        'within_unit_auc_preservation.csv', 'distribution_eligibility.csv', 'marker_scales.csv',
        'emd_unit_pairs.csv.gz', 'within_type_dispersion.csv.gz', 'COMPLETE.json'}
    for case in i.CASES:
        prefix = i.VALUE + '/results/' + case + '/'
        i.require({name[len(prefix):] for name in numerical if name.startswith(prefix)} == expected_case_files, 'Incomplete gate/distribution output set')
        for name in ('reconstructed.npy', 'cosine_normalized.npy', 'pca_reconstructed.npy'):
            i.require(i.RECON + '/' + case + '/' + name in numerical, 'Missing reconstruction')
    report = {'status': 'passed', 'scope': 'fresh historical percentile, reconstruction, gate and distribution scripts; no alignment refit',
              'run_verification': verified, 'run_manifest': i.record(run / 'RUN_MANIFEST.json'),
              'input_manifest': i.record(Path(bundle) / 'MANIFEST.json'),
              'files': rows, 'json_comparisons': json_reports, 'artifact_count': len(rows),
              'byte_equal_count': sum(x['byte_equal'] for x in rows),
              'table_count': sum(x['kind'] == 'table' for x in rows),
              'array_count': sum(x['kind'] == 'array' for x in rows),
              'review_figures_not_claimed_as_publication_figures': sorted(set(produced) - set(numerical))}
    output = i.reserve(output, [run, bundle, snapshot, source_root]); i.write(output / 'report.json', report)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'bundle', 'snapshot', 'source-root', 'output'): p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args(); result = compare(args.run, args.bundle, args.snapshot, args.source_root, args.output)
    print(json.dumps({k: result[k] for k in ('status', 'artifact_count', 'byte_equal_count', 'table_count', 'array_count')}))
