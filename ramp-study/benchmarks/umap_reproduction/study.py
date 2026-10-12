"""Fresh common-protocol UMAP, followed by original metrics; never reuse UMAP cache."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

from bootstrap import HERE, check_support, r

INPUT_SHA = 'c80dcde5bfc3cd2f0556a6391f8b694124e6a768f261f568ebeb67a4df3c7904'
UMAP_PROTOCOL = {'n_neighbors': 30, 'min_dist': 0.15, 'metric': 'euclidean', 'init': 'spectral',
                 'random_state': 42, 'transform_seed': 42, 'n_jobs': 1, 'low_memory': True,
                 'preprocessing': 'pooled dimension-wise StandardScaler; no additional PCA'}
CODE = ('bootstrap.py', 'study.py', 'worker.py', 'support/LOCK.json',
        'support/replay.py', 'support/replay_worker.py', 'support/contract.py',
        'support/inventory.py', 'support/compare_replay.py')


def codes():
    check_support()
    return {name: r.record(HERE / name) for name in CODE}


def inputs(bundle):
    bundle = Path(bundle).resolve(strict=True)
    r.require(r.record(bundle / 'MANIFEST.json')['sha256'] == INPUT_SHA, 'Not the accepted replay input bundle')
    return r.verify_bundle(bundle)


def branches(manifest, scope):
    all_branches = [(c, v) for c in manifest['cases'] for v in c['variants']]
    r.require(scope in ('pilot', 'all'), 'Unknown batch scope')
    if scope == 'pilot':
        chosen = [(c, v) for c, v in all_branches if c['case'] in ('tnbc', 'hnscc_story') and v['method'] == 'RAMP']
        r.require(len(chosen) == 2, 'Pilot identity changed')
        return chosen
    r.require(len(all_branches) == 63 and len({(c['case'], v['id']) for c, v in all_branches}) == 63,
              'Expected complete 63-branch input grid')
    return all_branches


def grid(manifest, entries, scope):
    expected = {(c['case'], v['id']) for c, v in branches(manifest, scope)}
    r.require(isinstance(entries, list) and all(isinstance(x, dict) and
              isinstance(x.get('case'), str) and isinstance(x.get('variant'), str) for x in entries), 'Malformed branch grid')
    keys = [(x['case'], x['variant']) for x in entries]
    r.require(len(keys) == len(set(keys)) and set(keys) == expected, 'Branch grid differs')
    return dict(zip(keys, entries))


def finish_branch(run, code_records):
    r.require(codes() == code_records, 'Worker controller changed')
    r.write(run / 'RUN_MANIFEST.json', {'files': r.tree(run, exclude=('stdout.log', 'stderr.log'))})
    r.write(run / 'COMPLETE.json', {'status': 'complete', 'manifest': r.record(run / 'RUN_MANIFEST.json')})


def verify_branch(run, manifest, parent, bundle):
    request = r.read(run / 'REQUEST.json')
    r.require({k: v for k, v in request.items() if k not in ('case', 'variant')} ==
              {k: v for k, v in parent.items() if k != 'jobs'}, 'Worker request differs from parent')
    r.require(run.parent.name == request['case'] and run.name == request['variant'], 'Branch path identity differs')
    r.require(request['code_records'] == codes(), 'Unknown controller')
    case, variant = next((c, v) for c, v in branches(manifest, parent['scope'])
                         if (c['case'], v['id']) == (request['case'], request['variant']))
    done = r.read(run / 'COMPLETE.json'); r.require(done.get('status') == 'complete', 'Branch incomplete')
    r.check(run / 'RUN_MANIFEST.json', done['manifest'])
    data = r.read(run / 'RUN_MANIFEST.json')
    r.require(data['files'] == r.tree(run, exclude=('RUN_MANIFEST.json', 'COMPLETE.json', 'stdout.log', 'stderr.log')),
              'Branch payload changed')
    before = r.read(run / 'ENVIRONMENT_BEFORE.json')
    r.require(before == r.read(run / 'ENVIRONMENT_AFTER.json') and before['packages'] == r.PACKAGES,
              'Runtime mismatch')
    r.require(before['threads'] == {key: '1' for key in r.THREADS} and
              all(p['num_threads'] == 1 for p in before['threadpools']), 'Thread policy changed')
    rows = [r.read(run / 'replicates' / (str(seed) + '.json')) for seed in manifest['seeds']]
    r.require({p.name for p in (run / 'replicates').iterdir()} == {str(s) + '.json' for s in manifest['seeds']},
              'Resample file grid differs')
    r.require(all(x['benchmark'] == case['case'] and x['method'] == variant['method'] and
                  x['sampling_seed'] == seed for x, seed in zip(rows, manifest['seeds'])), 'Resample identity changed')
    fit = r.read(run / 'FIT.json')
    r.require(fit['fitted_from_empty_directory'] is True and fit['labels_used_for_fit'] is False and
              fit['protocol'] == UMAP_PROTOCOL, 'Fit boundary differs')
    r.check(Path(bundle) / 'evaluator.py', fit['evaluator'])
    import contract
    import numpy as np
    from worker import compare_coordinates
    helper = contract.pure_helpers(Path(bundle) / 'evaluator.py')
    old = r.read(Path(bundle) / variant['xy_manifest'])
    stem = helper['safe_name'](variant['method'])
    new = r.read(run / 'fit/umap/manifests' / (stem + '.json'))
    coordinates = np.load(run / 'fit/umap/coordinates' / (stem + '.npy'), allow_pickle=False)
    r.require(new['protocol'] == UMAP_PROTOCOL and new['method'] == variant['method'] and
              new['labels_used_for_fit'] is False and
              new['representation_sha256'] == old['representation_sha256'] == fit['representation_sha256'] and
              new['coordinate_sha256'] == helper['sha256_array'](coordinates) and
              new['n_cells'] == old['n_cells'] and new['n_dimensions'] == old['n_dimensions'], 'Fitted UMAP contract differs')
    expected_comparison = compare_coordinates(coordinates, np.load(Path(bundle) / variant['xy'], allow_pickle=False),
                                              helper['sha256_array'])
    r.require(expected_comparison == r.read(run / 'COORDINATE_COMPARISON.json'), 'Coordinate comparison differs')
    return {'case': case['case'], 'variant': variant['id'], 'method': variant['method'],
            'manifest': r.record(run / 'RUN_MANIFEST.json'),
            'coordinates': r.read(run / 'COORDINATE_COMPARISON.json')}


def verify_run(output, bundle):
    output = Path(output).resolve(strict=True)
    manifest = inputs(bundle)
    r.require((output / 'COMPLETE.json').is_file(), 'Batch incomplete')
    done = r.read(output / 'COMPLETE.json'); r.require(done.get('status') == 'complete', 'Batch incomplete')
    r.check(output / 'BATCH_MANIFEST.json', done['manifest'])
    parent = r.read(output / 'REQUEST.json'); r.require(parent['code_records'] == codes(), 'Unknown batch controller')
    r.check(Path(bundle) / 'MANIFEST.json', parent['bundle_manifest'])
    r.check(Path(parent['python']), parent['python_record'])
    for name, rec in parent['code_records'].items(): r.check(r.child(output / 'controller', name), rec)
    report = r.read(output / 'BATCH_MANIFEST.json')
    r.require(report['code_records'] == parent['code_records'] and report['scope'] == parent['scope'], 'Batch binding differs')
    entries = grid(manifest, report['branches'], parent['scope'])
    for (case, variant), entry in entries.items():
        run = r.child(output, case + '/' + variant)
        r.require(entry == verify_branch(run, manifest, parent, bundle), 'Branch record differs')
        r.require(r.read(run / 'COMMAND.json')['argv'] ==
                  [parent['python'], '-B', str(output / 'controller/worker.py'), '--run', str(run)], 'Worker command differs')
    return {'status': 'passed', 'branches': len(entries)}


def batch(bundle, output, python, jobs, scope):
    r.require(1 <= jobs <= 4, 'jobs must be 1..4')
    bundle = Path(bundle).resolve(strict=True); python = Path(python).absolute()
    manifest = inputs(bundle); selected = branches(manifest, scope)
    output = Path(output).absolute()
    r.require(output.resolve().is_relative_to((HERE / 'runs').resolve()) and
              output.resolve() != (HERE / 'runs').resolve(), 'Output outside this new module runs area')
    r.require(not output.resolve().is_relative_to(bundle) and not bundle.is_relative_to(output.resolve()), 'Output overlaps input')
    output = r.reserve(output, [bundle])
    records = codes()
    try:
        for name, rec in records.items(): r.copy(HERE / name, output / 'controller' / name, rec)
        parent = {'scope': scope, 'mode': 'fresh_umap_and_fresh_original_metrics', 'jobs': jobs,
                  'bundle': str(bundle), 'bundle_manifest': r.record(bundle / 'MANIFEST.json'),
                  'code_records': records, 'python': str(python), 'python_record': r.record(python)}
        r.write(output / 'REQUEST.json', parent)
        def launch(case, variant):
            run = r.reserve(output / case['case'] / variant['id'])
            r.write(run / 'REQUEST.json', {**{k: v for k, v in parent.items() if k != 'jobs'},
                                         'case': case['case'], 'variant': variant['id']})
            command = [str(python), '-B', str(output / 'controller/worker.py'), '--run', str(run)]
            r.write(run / 'COMMAND.json', {'argv': command})
            with (run / 'stdout.log').open('x') as stdout, (run / 'stderr.log').open('x') as stderr:
                rc = subprocess.run(command, env=r.environment(run), stdout=stdout, stderr=stderr).returncode
            r.require(rc == 0, 'Worker failed: ' + str(run))
            result = verify_branch(run, manifest, parent, bundle)
            print(case['case'], variant['method'], 'complete; coordinate exact=',
                  result['coordinates']['array_equal'], flush=True)
            return result
        results = []
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(launch, c, v) for c, v in selected]
            try:
                for f in as_completed(futures): results.append(f.result())
            except BaseException:
                for f in futures: f.cancel()
                raise
        r.require(codes() == records, 'Controller changed during batch')
        inputs(bundle); r.check(bundle / 'MANIFEST.json', parent['bundle_manifest']); r.check(python, parent['python_record'])
        grid(manifest, results, scope)
        r.write(output / 'BATCH_MANIFEST.json', {'scope': scope, 'code_records': records,
                  'branches': sorted(results, key=lambda x: (x['case'], x['variant']))})
        r.write(output / 'COMPLETE.json', {'status': 'complete', 'manifest': r.record(output / 'BATCH_MANIFEST.json')})
    except BaseException as error:
        r.write(output / 'FAILED.json', {'error': repr(error)}); raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['batch', 'verify-run', 'verify-input'])
    p.add_argument('--bundle', type=Path, required=True); p.add_argument('--output', type=Path)
    p.add_argument('--python', type=Path); p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--scope', choices=['pilot', 'all'], default='all')
    a = p.parse_args()
    if a.action == 'batch': batch(a.bundle, a.output, a.python, a.jobs, a.scope)
    elif a.action == 'verify-run': print(json.dumps(verify_run(a.output, a.bundle)))
    else: inputs(a.bundle); print('Accepted frozen input bundle verified')


if __name__ == '__main__': main()
