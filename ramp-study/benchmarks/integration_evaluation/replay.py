"""Frozen-coordinate metric replay. All output is create-only under this module."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
STUDY = HERE.parents[1]
SNAPSHOT = HERE / 'frozen_20261011_r2'
CATALOG_SHA = '965fdb13dcf0fe402c0435770f28d1816d1d0543e1420a78d5cd62d6f78ac961'
PHASE1_SHA = 'b722181e100e057d2a1c2cc3b9f7d2147cb839fbd452169f23aecd4ebb791fd4'
PHASE1 = HERE / 'validation/handoff_20261011/report.json'
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'LOKY_MAX_CPU_COUNT')
PACKAGES = {'numpy': '2.4.0', 'pandas': '2.3.3', 'scipy': '1.16.3', 'scikit-learn': '1.8.0',
            'umap-learn': '0.5.11', 'numba': '0.64.0', 'matplotlib': '3.10.8',
            'igraph': '1.0.0', 'leidenalg': '0.11.0', 'threadpoolctl': '3.6.0'}
CODE = ('replay.py', 'replay_worker.py', 'contract.py', 'inventory.py')
# Exact controller used by the externally audited r3 batch. Read-only
# verification may accept it; new executions always record current code.
ACCEPTED_R3_CODE = {
    'replay.py': {'bytes': 18513, 'sha256': '6e0d6147bed9e4caa72309ab35427a503c8adfbbc6284e91ed0bec3db297c726'},
    'replay_worker.py': {'bytes': 6595, 'sha256': '2c44e19b92c1f005a7c57676099d0d7b5e6d077906ca6dc7aad394e0eb8cf5ef'},
    'contract.py': {'bytes': 18624, 'sha256': '35e996fc9a0a52d92dc12fe1106b19378064c452eba450384c6d8eca917ce187'},
    'inventory.py': {'bytes': 5295, 'sha256': 'f71f1d36a3583bf4a9e49b0d195d38317a689f58d6dbcc4478092cd3b16b8b3d'},
}


def require(ok, message):
    if not ok: raise ValueError(message)


def record(path):
    with Path(path).open('rb') as handle: digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    return {'bytes': Path(path).stat().st_size, 'sha256': digest}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    with Path(path).open('x') as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write('\n')


def check(path, expected):
    require(record(path) == expected, 'Changed file: ' + str(path))


def reserve(path, protected=()):
    path = Path(path).absolute()
    if path.exists() or path.is_symlink(): raise FileExistsError(path)
    resolved = path.resolve()
    for source in protected:
        source = Path(source).resolve()
        require(not (resolved == source or resolved.is_relative_to(source) or source.is_relative_to(resolved)), 'Output overlaps input')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(exist_ok=False)
    return path.resolve()


def local_output(path, area):
    path = Path(path).absolute()
    root = HERE / area
    require(path.resolve().is_relative_to(root.resolve()) and path.resolve() != root.resolve(), 'Output outside permitted new-module area')
    return path


def child(root, name):
    rel = Path(name)
    require(rel.parts and not rel.is_absolute() and '..' not in rel.parts, 'Unsafe relative path')
    path = Path(root).resolve() / rel
    for p in (path, *path.parents):
        if p == Path(root).resolve(): break
        require(not p.is_symlink(), 'Unexpected symlink')
    return path


def tree(root, exclude=()):
    results = {}
    for p in sorted(Path(root).rglob('*')):
        rel = p.relative_to(root).as_posix()
        require(not p.is_symlink(), 'Unexpected symlink: ' + str(p))
        if p.is_file() and rel not in exclude and '.cache' not in p.relative_to(root).parts:
            results[rel] = record(p)
    return results


def copy(source, target, expected):
    target.parent.mkdir(parents=True, exist_ok=True)
    with Path(source).open('rb') as src, target.open('xb') as dst: shutil.copyfileobj(src, dst)
    check(target, expected)


def anchored(bundle=None):
    catalog = SNAPSHOT / 'catalog.json' if bundle is None else Path(bundle) / 'trust/catalog.json'
    report = PHASE1 if bundle is None else Path(bundle) / 'trust/phase1.json'
    require(record(catalog)['sha256'] == CATALOG_SHA, 'Evaluation catalog anchor differs')
    require(record(report)['sha256'] == PHASE1_SHA, 'Phase-1 report anchor differs')
    return read(catalog), read(report)


def source_binding(case, variant, catalog, prior):
    """Deterministic accepted-run selection; never pick an arbitrary hash match."""
    name = case['case']; method = variant['method']; original = variant['matrix']
    expected = catalog['sources'][original]
    if method.startswith('MaxFuse'):
        return {'root': 'original', 'path': original, **expected, 'route': 'audited_historical_MaxFuse_no_refit'}
    if method == 'Unaligned': prefix = f'benchmarks/integration_inputs/runs/frozen_inputs_20261010/{name}/'
    elif method == 'RAMP':
        prefix = ('benchmarks/bone_marrow/runs/rebuild_20261010/alignment/data/' if name == 'bone_marrow'
                  else f'benchmarks/ramp_alignment/runs/six_fit_20261011_r2/{name}/result/')
    elif method == 'fastMNN': prefix = f'methods/fastmnn/runs/seven_fit_20261010_r2/{name}/'
    elif method == 'CytoNorm': prefix = f'methods/cytonorm/runs/seven_fit_20261011/{name}/'
    elif method == 'cyCombine': prefix = f"methods/cycombine/runs/{'bm' if name == 'bone_marrow' else name}_correction_20261010/"
    elif method.startswith('Harmony-direct'):
        profile = 'fixed' if 'fixed_arcsinh' in method else 'source'
        stem = 'bm' if name == 'bone_marrow' and profile == 'source' else name
        prefix = f'methods/harmony/runs/{stem}_{profile}_fit_20261010/'
    else: raise ValueError('Unknown method: ' + method)
    matches = [p for p, rec in prior['outside_files'].items() if p.startswith(prefix) and rec == expected]
    require(len(matches) == 1, 'Ambiguous/missing accepted output: ' + name + '/' + method)
    return {'root': 'study', 'path': matches[0], **expected, 'route': 'accepted_frozen_raw' if method == 'Unaligned' else 'accepted_real_rerun'}


def plan(catalog, prior, registry):
    cases = []
    for case in catalog['cases']:
        spec = next(s for s in registry['cases'] if s['case'] == case['case'])
        branches = []
        for i, v in enumerate(case['variants']):
            binding = source_binding(case, v, catalog, prior)
            extension = '.npy' if v['matrix'].endswith('.npy') else '.csv.gz'
            branches.append({**v, 'id': f'v{i:02d}', 'binding': binding,
                             'input': f"{case['case']}/v{i:02d}/matrix{extension}",
                             'xy': f"{case['case']}/v{i:02d}/coordinates.npy",
                             'xy_manifest': f"{case['case']}/v{i:02d}/umap_manifest.json"})
        cases.append({**case, 'kind': spec['kind'], 'label_column': ('celltype_joint_evaluation' if spec['kind'] == 'technical' else 'celltype'),
                      'metadata_source': spec['files']['evaluation_metadata'], 'variants': branches})
    return cases


def preflight(output):
    import validate
    output = reserve(local_output(output, 'validation'))
    catalog, prior = anchored()
    require(validate.outside() == prior['outside_files'], 'Previously accepted files changed')
    old = {}
    for name, rec in prior['current_code'].items():
        check(HERE / name, rec); copy(HERE / name, output / 'phase1_code' / name, rec); old[name] = rec
    write(output / 'report.json', {'phase1_sha256': PHASE1_SHA, 'catalog_sha256': CATALOG_SHA,
                                  'old_code': old, 'outside_files': prior['outside_files'],
                                  'old_module_files': tree(HERE, exclude=()),
                                  'original_indexes': validate.originals(), 'status': 'passed'})


def prepare(original, output):
    import inventory
    catalog, prior = anchored()
    inventory.verify(SNAPSHOT, original)
    registry = read(SNAPSHOT / 'registry.json')
    cases = plan(catalog, prior, registry)
    output = reserve(local_output(output, 'runs'), [original, STUDY / 'methods', STUDY / 'benchmarks/ramp_alignment'])
    inputs = {}
    try:
        def put(source, name, expected):
            check(source, expected); copy(source, child(output, name), expected)
            inputs[name] = {'source': str(Path(source).resolve()), **expected}
        evaluator = 'benchmark_results/comparison_evaluation/scripts/evaluate_benchmark.py'
        put(SNAPSHOT / 'sources' / evaluator, 'evaluator.py', catalog['sources'][evaluator])
        for path, name in [(SNAPSHOT / 'catalog.json', 'catalog.json'), (PHASE1, 'phase1.json'),
                           (SNAPSHOT / 'registry.json', 'registry.json')]:
            put(path, 'trust/' + name, record(path))
        for case in cases:
            name = case['case']; md = case['metadata_source']
            staged_md = STUDY / 'benchmarks/integration_inputs/runs/frozen_inputs_20261010' / name / 'evaluation_metadata.csv.gz'
            put(staged_md, f'{name}/metadata.csv.gz', {k: md[k] for k in ('bytes', 'sha256')})
            for v in case['variants']:
                b = v['binding']; base = STUDY if b['root'] == 'study' else original
                put(base / b['path'], v['input'], {k: b[k] for k in ('bytes', 'sha256')})
                for key, dest in [('coordinates', v['xy']), ('umap_manifest', v['xy_manifest'])]:
                    put(original / v[key], dest, catalog['sources'][v[key]])
        for name, rec in inputs.items(): check(rec['source'], {k: rec[k] for k in ('bytes', 'sha256')})
        require(record(SNAPSHOT / 'catalog.json')['sha256'] == CATALOG_SHA, 'Catalog changed during prepare')
        manifest = {'catalog_sha256': CATALOG_SHA, 'phase1_sha256': PHASE1_SHA, 'cases': cases,
                    'seeds': catalog['seeds'], 'metric_registry': catalog['metric_registry'], 'packages': PACKAGES,
                    'files': {n: {k: r[k] for k in ('bytes', 'sha256')} for n, r in inputs.items()},
                    'sources': inputs, 'mode': 'frozen_coordinate_metric_replay_no_UMAP_fit'}
        write(output / 'MANIFEST.json', manifest)
        verify_bundle(output, require_complete=False)
        write(output / 'COMPLETE.json', {'status': 'complete', 'manifest': record(output / 'MANIFEST.json')})
    except BaseException as error:
        write(output / 'FAILED.json', {'error': repr(error)}); raise


def verify_bundle(bundle, require_complete=True):
    bundle = Path(bundle).resolve(strict=True)
    manifest = read(bundle / 'MANIFEST.json')
    catalog, prior = anchored(bundle)
    require(manifest['catalog_sha256'] == CATALOG_SHA and manifest['phase1_sha256'] == PHASE1_SHA, 'Input anchor differs')
    registry_path = bundle / 'trust/registry.json'
    require(record(registry_path)['sha256'] == catalog['registry_sha256'], 'Public registry differs')
    require(manifest['cases'] == plan(catalog, prior, read(registry_path)), 'Input routing contract differs')
    require(manifest['seeds'] == catalog['seeds'] and manifest['metric_registry'] == catalog['metric_registry'] and manifest['packages'] == PACKAGES, 'Evaluation protocol differs')
    expected = {'evaluator.py': catalog['sources']['benchmark_results/comparison_evaluation/scripts/evaluate_benchmark.py']}
    expected.update({'trust/' + n: record(bundle / 'trust' / n) for n in ('catalog.json', 'phase1.json', 'registry.json')})
    for case in manifest['cases']:
        expected[case['case'] + '/metadata.csv.gz'] = {k: case['metadata_source'][k] for k in ('bytes', 'sha256')}
        for v in case['variants']:
            expected[v['input']] = {k: v['binding'][k] for k in ('bytes', 'sha256')}
            expected[v['xy']] = catalog['sources'][v['coordinates']]
            expected[v['xy_manifest']] = catalog['sources'][v['umap_manifest']]
    require(manifest['files'] == expected, 'Bundle file contract differs')
    extras = {'MANIFEST.json'} | ({'COMPLETE.json'} if require_complete else set())
    require(set(tree(bundle)) == set(expected) | extras, 'Bundle inventory differs')
    for name, rec in expected.items(): check(child(bundle, name), rec)
    if require_complete:
        done = read(bundle / 'COMPLETE.json')
        require(done.get('status') == 'complete', 'Bundle incomplete'); check(bundle / 'MANIFEST.json', done['manifest'])
    return manifest


def code_records():
    return {name: record(HERE / name) for name in CODE}


def check_controller_records(records):
    require(records == code_records() or records == ACCEPTED_R3_CODE, 'Unknown controller version')


def branch_records(manifest, report):
    """Check the exact input-derived grid before accessing any branch path."""
    expected = {(case['case'], v['id']) for case in manifest['cases'] for v in case['variants']}
    entries = report.get('branches')
    require(isinstance(entries, list) and all(isinstance(x, dict) and
            isinstance(x.get('case'), str) and isinstance(x.get('variant'), str) for x in entries),
            'Batch branch grid differs: malformed entries')
    keys = [(x['case'], x['variant']) for x in entries]
    require(len(keys) == len(set(keys)), 'Batch branch grid differs: duplicate branches')
    require(set(keys) == expected, 'Batch branch grid differs: missing or unexpected branches')
    check_controller_records(report.get('code_records'))
    return dict(zip(keys, entries))


def environment(output):
    env = dict(os.environ)
    env.update({k: '1' for k in THREADS})
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0',
               MPLBACKEND='Agg', MPLCONFIGDIR=str(output / '.cache/matplotlib'),
               NUMBA_CACHE_DIR=str(output / '.cache/numba'), XDG_CACHE_HOME=str(output / '.cache/xdg'),
               RAMP_EVALUATION_CACHE=str(output / '.cache/evaluation'))
    env.pop('PYTHONPATH', None)
    # Historical environment supplies igraph/leidenalg from user site; keep it
    # visible and verify exact imported versions and native runtime instead.
    env.pop('PYTHONNOUSERSITE', None)
    return env


def batch(bundle, output, python, jobs=4):
    require(1 <= jobs <= 12, 'jobs must be 1..12')
    bundle = bundle.resolve(strict=True); python = python.absolute()
    manifest = verify_bundle(bundle)
    output = reserve(local_output(output, 'runs'), [bundle])
    codes = code_records(); records = []
    try:
        for name, rec in codes.items(): copy(HERE / name, output / 'controller' / name, rec)
        common = {'bundle': str(bundle), 'bundle_manifest': record(bundle / 'MANIFEST.json'),
                  'code_records': codes, 'python': str(python), 'python_record': record(python),
                  'mode': 'frozen_coordinate_metric_replay_no_UMAP_fit', 'n_seeds': 20}
        write(output / 'REQUEST.json', {**common, 'jobs': jobs})
        probe = reserve(output / 'probe')
        probe_code = ('import sys,json;from pathlib import Path;'
                      'sys.path.insert(0,sys.argv[1]);import replay_worker as w;'
                      'w.load_evaluator(Path(sys.argv[2]));print(json.dumps(w.runtime()))')
        probe_cmd = [str(python), '-B', '-c', probe_code, str(output / 'controller'), str(bundle)]
        write(probe / 'COMMAND.json', {'argv': probe_cmd})
        checked = subprocess.run(probe_cmd, env=environment(probe), text=True, capture_output=True)
        with (probe / 'stdout.log').open('x') as f: f.write(checked.stdout)
        with (probe / 'stderr.log').open('x') as f: f.write(checked.stderr)
        require(checked.returncode == 0, 'Runtime probe failed before any metrics')
        write(probe / 'ENVIRONMENT.json', json.loads(checked.stdout))
        def launch(case, v):
            dest = reserve(output / case['case'] / v['id'])
            request = {**common, 'case': case['case'], 'variant': v['id']}
            write(dest / 'REQUEST.json', request)
            cmd = [str(python), '-B', str(output / 'controller/replay_worker.py'), '--run', str(dest)]
            write(dest / 'COMMAND.json', {'argv': cmd})
            with (dest / 'stdout.log').open('x') as stdout, (dest / 'stderr.log').open('x') as stderr:
                status = subprocess.run(cmd, env=environment(dest), stdout=stdout, stderr=stderr).returncode
            require(status == 0, 'Worker failed: ' + str(dest))
            result = verify_branch(dest, bundle)
            print(case['case'], v['method'], '20/20 completed', flush=True)
            return {'case': case['case'], 'variant': v['id'], 'manifest': record(dest / 'RUN_MANIFEST.json'), 'result': result}
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(launch, case, v) for case in manifest['cases'] for v in case['variants']]
            try:
                for future in as_completed(futures): records.append(future.result())
            except BaseException:
                for future in futures: future.cancel()
                raise
        require(code_records() == codes, 'Controller changed during batch')
        check(python, common['python_record']); check(bundle / 'MANIFEST.json', common['bundle_manifest'])
        verify_bundle(bundle)
        branch_records(manifest, {'code_records': codes, 'branches': records})
        write(output / 'BATCH_MANIFEST.json', {'code_records': codes, 'branches': sorted(records, key=lambda r: (r['case'], r['variant']))})
        write(output / 'COMPLETE.json', {'status': 'complete', 'manifest': record(output / 'BATCH_MANIFEST.json')})
    except BaseException as error:
        write(output / 'FAILED.json', {'error': repr(error)}); raise


def verify_branch(run, bundle):
    manifest = verify_bundle(bundle)
    request = read(run / 'REQUEST.json')
    check(bundle / 'MANIFEST.json', request['bundle_manifest'])
    check_controller_records(request['code_records'])
    case = next(c for c in manifest['cases'] if c['case'] == request['case'])
    v = next(v for v in case['variants'] if v['id'] == request['variant'])
    done = read(run / 'COMPLETE.json'); require(done['status'] == 'complete', 'Worker incomplete')
    check(run / 'RUN_MANIFEST.json', done['manifest'])
    data = read(run / 'RUN_MANIFEST.json')
    for name, rec in data['files'].items(): check(child(run, name), rec)
    actual = tree(run, exclude=('RUN_MANIFEST.json', 'COMPLETE.json', 'stdout.log', 'stderr.log'))
    require(actual == data['files'], 'Worker file inventory differs')
    rows = [read(run / 'replicates' / (str(seed) + '.json')) for seed in manifest['seeds']]
    require(all(r['benchmark'] == case['case'] and r['method'] == v['method'] for r in rows), 'Result identity differs')
    require([r['sampling_seed'] for r in rows] == manifest['seeds'], 'Seed grid differs')
    return {'method': v['method'], 'resamples': len(rows), 'metric_values': 31 * len(rows)}


def verify_run(output, bundle):
    output = Path(output).resolve(strict=True); bundle = Path(bundle).resolve(strict=True)
    require((output / 'COMPLETE.json').is_file(), 'Batch incomplete: missing COMPLETE.json')
    done = read(output / 'COMPLETE.json')
    require(done.get('status') == 'complete', 'Batch incomplete')
    check(output / 'BATCH_MANIFEST.json', done['manifest'])
    manifest = verify_bundle(bundle)
    report = read(output / 'BATCH_MANIFEST.json')
    entries = branch_records(manifest, report)
    parent = read(output / 'REQUEST.json')
    require(parent['code_records'] == report['code_records'], 'Batch request controller differs')
    check(bundle / 'MANIFEST.json', parent['bundle_manifest'])
    for name, rec in parent['code_records'].items(): check(child(output / 'controller', name), rec)
    for (case, variant), entry in entries.items():
        run = child(output, case + '/' + variant)
        request = read(run / 'REQUEST.json')
        require((request.get('case'), request.get('variant')) == (case, variant), 'Branch identity differs')
        require({k: v for k, v in request.items() if k not in ('case', 'variant')} ==
                {k: v for k, v in parent.items() if k != 'jobs'}, 'Worker request differs from batch')
        check(run / 'RUN_MANIFEST.json', entry['manifest'])
        require(verify_branch(run, bundle) == entry['result'], 'Branch result summary differs')
    return {'branches': len(entries), 'status': 'passed'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['preflight', 'prepare', 'verify', 'batch', 'verify-run'])
    p.add_argument('--output', type=Path); p.add_argument('--bundle', type=Path)
    p.add_argument('--source-root', type=Path); p.add_argument('--python', type=Path)
    p.add_argument('--jobs', type=int, default=4)
    a = p.parse_args()
    if a.action == 'preflight': preflight(a.output)
    elif a.action == 'prepare': prepare(a.source_root.resolve(strict=True), a.output)
    elif a.action == 'verify': verify_bundle(a.bundle)
    elif a.action == 'batch': batch(a.bundle, a.output, a.python, a.jobs)
    elif a.action == 'verify-run': verify_run(a.output, a.bundle)


if __name__ == '__main__': main()
