"""Create-only execution of unchanged historical value-evaluation scripts in a mirror."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import inventory as i
import contract as c

HERE = Path(__file__).resolve().parent
STUDY = HERE.parents[1]
PREFLIGHT_SHA = '8360f5c47389ce2528d3e710dd8690fbf205fad344e3a2e305d63094c0f9b2f2'
CODE = ('execution.py', 'inventory.py', 'contract.py')
SCRIPTS = (i.PERCENTILE + '/compute_celltype_shifts.py',
           'Main Figure/pair_inversion_boxplot_20260914/compute_pair_inversions.py',
           i.RECON + '/run_audit.py', i.RECON + '/evaluate_reconstructed.py',
           i.VALUE + '/evaluate_values.py', i.VALUE + '/summarize_values.py')
EXTRA = (i.VALUE + '/PROTOCOL.md', 'algorithm/fastmnn/vendor/batchelor_1.18.1.tar.gz',
         'benchmark_results/exact_value_artifacts/config/canonical_expectations.csv',
         'Main Figure/redesign_full_icon_20260914/all_celltype_statistics.csv')
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'LOKY_MAX_CPU_COUNT')
PACKAGES = {'numpy': '2.4.0', 'pandas': '2.3.3', 'scipy': '1.16.3', 'scikit-learn': '1.8.0',
            'numba': '0.64.0', 'matplotlib': '3.10.8', 'threadpoolctl': '3.6.0'}


def codes(): return {name: i.record(HERE / name) for name in CODE}


def tree(root, exclude=()):
    files = {}
    for path in sorted(Path(root).rglob('*')):
        rel = path.relative_to(root)
        i.require(not path.is_symlink(), 'Unexpected symlink')
        if path.is_file() and rel.as_posix() not in exclude and not any(x in rel.parts for x in ('.cache', '__pycache__')):
            files[rel.as_posix()] = i.record(path)
    return files


def environment(out):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0', MPLBACKEND='Agg',
               MPLCONFIGDIR=str(out / '.cache/matplotlib'), NUMBA_CACHE_DIR=str(out / '.cache/numba'))
    env.update({key: '1' for key in THREADS}); env.pop('PYTHONPATH', None); env.pop('PYTHONNOUSERSITE', None)
    return env


def runtime():
    import threadpoolctl
    versions = {name: importlib.metadata.version(name) for name in PACKAGES}
    i.require(versions == PACKAGES, 'Dependency versions differ')
    threads = {name: os.environ.get(name) for name in THREADS}
    pools = threadpoolctl.threadpool_info()
    i.require(all(v == '1' for v in threads.values()) and all(p['num_threads'] == 1 for p in pools), 'Thread policy differs')
    return {'python': sys.version, 'packages': versions, 'threads': threads, 'threadpools': pools}


def probe(python, controller, output, name):
    result = subprocess.run([str(python), '-B', str(controller / 'execution.py'), 'runtime'],
                            env=environment(output), text=True, capture_output=True)
    with (output / (name + '.stdout')).open('x') as f: f.write(result.stdout)
    with (output / (name + '.stderr')).open('x') as f: f.write(result.stderr)
    i.require(result.returncode == 0, 'Actual child environment probe failed')
    return json.loads(result.stdout)


def prior():
    path = STUDY / 'benchmarks/umap_reproduction/validation/preflight_20261011/report.json'
    i.require(i.record(path)['sha256'] == PREFLIGHT_SHA, 'Prior accepted-run baseline changed')
    return i.read(path)['outside_files']


def rerun_prefix(case, method, profile=None):
    if method == 'ramp':
        return ('benchmarks/bone_marrow/runs/rebuild_20261010/alignment/data/' if case == 'bone_marrow' else
                f'benchmarks/ramp_alignment/runs/six_fit_20261011_r2/{case}/result/')
    if method == 'fastmnn': return f'methods/fastmnn/runs/seven_fit_20261010_r2/{case}/'
    if method == 'cytonorm': return f'methods/cytonorm/runs/seven_fit_20261011/{case}/'
    if method == 'cycombine': return f"methods/cycombine/runs/{'bm' if case == 'bone_marrow' else case}_correction_20261010/"
    if method == 'harmony':
        tag = 'source' if profile == 'source_scale' else 'fixed'
        stem = 'bm' if case == 'bone_marrow' and tag == 'source' else case
        return f'methods/harmony/runs/{stem}_{tag}_fit_20261010/'
    raise ValueError('Unknown method')


def binding(name, rec, accepted):
    parts = Path(name).parts
    # Only numerical method outputs are rebound. Historical manifests remain
    # provenance inputs; they are not mistaken for new-run manifests.
    if len(parts) > 4 and parts[0] == 'benchmark_results' and parts[1] in i.CASES:
        case, method = parts[1:3]
        numerical = Path(name).suffix in ('.npy', '.gz', '.csv')
        if 'alignment/data/' in name and numerical:
            prefix = rerun_prefix(case, method, parts[3] if method == 'harmony' else None)
            matches = [p for p, value in accepted.items() if p.startswith(prefix) and value == rec]
            same_name = [p for p in matches if Path(p).name == Path(name).name]
            if len(same_name) == 1: matches = same_name
            i.require(len(matches) == 1, 'Missing/ambiguous accepted rerun: ' + name)
            return {'root': 'study', 'path': matches[0], 'role': 'accepted_real_rerun', **rec}
        if method == 'ramp' and 'alignment/input/' in name:
            renamed = {'cell_metadata.csv.gz': 'evaluation_metadata.csv.gz'}.get(parts[-1], parts[-1])
            path = f'benchmarks/integration_inputs/runs/frozen_inputs_20261010/{case}/{renamed}'
            i.require(accepted.get(path) == rec, 'Frozen cohort binding differs')
            return {'root': 'study', 'path': path, 'role': 'accepted_frozen_cohort', **rec}
    return {'root': 'original', 'path': name, 'role': 'frozen_method_input_or_provenance', **rec}


def plan(snapshot):
    s = c.Sources(snapshot); accepted = prior()
    prefixes = tuple('benchmark_results/' + case + '/' for case in i.CASES)
    names = {name for name in s.catalog['sources'] if name.startswith(prefixes)} | set(SCRIPTS) | set(EXTRA)
    i.require(set(SCRIPTS).issubset(names) and all(name in s.catalog['sources'] for name in names), 'Missing input contract')
    return {name: binding(name, s.catalog['sources'][name], accepted) for name in sorted(names)}


def local_output(path):
    resolved = Path(path).resolve()
    i.require(resolved.is_relative_to(HERE / 'runs') and resolved != HERE / 'runs', 'Output outside this new module runs')
    return Path(path)


def prepare(snapshot, source_root, output):
    source_root = Path(source_root).resolve(strict=True); s = c.Sources(snapshot, source_root)
    inputs = plan(snapshot); output = i.reserve(local_output(output), [source_root, snapshot])
    try:
        for name, rec in inputs.items():
            source = (STUDY if rec['root'] == 'study' else source_root) / rec['path']
            i.require(i.record(source) == {k: rec[k] for k in ('bytes', 'sha256')}, 'Bound source changed')
            target = i.bind(output / 'tree', name); target.parent.mkdir(parents=True, exist_ok=True)
            with source.open('rb') as src, target.open('xb') as dst: shutil.copyfileobj(src, dst)
            i.require(i.record(target) == {k: rec[k] for k in ('bytes', 'sha256')}, 'Copy changed')
        for name, rec in inputs.items():
            source = (STUDY if rec['root'] == 'study' else source_root) / rec['path']
            i.require(i.record(source) == {k: rec[k] for k in ('bytes', 'sha256')}, 'Source changed during prepare')
        i.write(output / 'MANIFEST.json', {'catalog_sha256': c.CATALOG_SHA, 'files': inputs,
                                          'scope': 'method outputs and input/provenance; no historical gate or percentile results except quartile regression table'})
        verify_bundle(output, snapshot, require_complete=False)
        i.write(output / 'COMPLETE.json', {'status': 'complete', 'manifest': i.record(output / 'MANIFEST.json')})
    except BaseException as error:
        i.write(output / 'FAILED.json', {'error': repr(error)}); raise


def verify_bundle(bundle, snapshot, require_complete=True):
    bundle = Path(bundle).resolve(strict=True); manifest = i.read(bundle / 'MANIFEST.json')
    i.require(manifest['files'] == plan(snapshot) and manifest['catalog_sha256'] == c.CATALOG_SHA, 'Bundle plan differs')
    expected = {'tree/' + name: {k: row[k] for k in ('bytes', 'sha256')} for name, row in manifest['files'].items()}
    actual = tree(bundle, ('MANIFEST.json', 'COMPLETE.json'))
    i.require(expected == actual, 'Bundle payload inventory differs')
    if require_complete:
        done = i.read(bundle / 'COMPLETE.json')
        i.require(done.get('status') == 'complete' and done['manifest'] == i.record(bundle / 'MANIFEST.json'), 'Bundle incomplete')
    return manifest


def commands(mirror, python):
    prefix = [str(python), '-B']
    return [prefix + [str(mirror / i.PERCENTILE / 'compute_celltype_shifts.py')],
            prefix + [str(mirror / i.RECON / 'run_audit.py')],
            *[prefix + [str(mirror / i.VALUE / 'evaluate_values.py'), '--case', case] for case in i.CASES],
            prefix + [str(mirror / i.VALUE / 'summarize_values.py')],
            prefix + [str(mirror / i.RECON / 'evaluate_reconstructed.py')]]


def relocate(mirror, manifest):
    changes = []
    for case in i.CASES:
        name = f'benchmark_results/{case}/fastmnn/alignment/data/manifest.json'; path = mirror / name
        data = i.read(path); before = i.record(path); original = data['input']['path']
        relative = i.bind(mirror, original).relative_to(mirror).as_posix()
        i.require(relative in manifest['files'] and i.record(mirror / relative)['sha256'] == data['input']['sha256'], 'Unbound fastMNN input')
        data['input']['path'] = str(mirror / relative)
        # Only this newly copied runtime manifest is replaced. Never a source
        # file or shared inode: copytree produced independent regular files.
        path.write_text(json.dumps(data, indent=2) + '\n')
        changes.append({'path': name, 'before': before, 'after': i.record(path), 'field': ['input', 'path'],
                        'old_value': original, 'new_value': data['input']['path']})
    return changes


def execute(bundle, snapshot, output, python):
    bundle = Path(bundle).resolve(strict=True); snapshot = Path(snapshot).resolve(strict=True); python = Path(python).absolute()
    manifest = verify_bundle(bundle, snapshot); records = codes()
    output = i.reserve(local_output(output), [bundle, snapshot]); request = {'bundle_manifest': i.record(bundle / 'MANIFEST.json'),
        'bundle': str(bundle), 'code_records': records, 'python': str(python), 'python_record': i.record(python)}
    try:
        i.write(output / 'REQUEST.json', request)
        for name, rec in records.items():
            target = output / 'controller' / name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(HERE / name, target)
            i.require(i.record(target) == rec, 'Code changed during copy')
        before = probe(python, output / 'controller', output, 'environment_before')
        i.write(output / 'ENVIRONMENT_BEFORE.json', before)
        mirror = output / 'runtime/tree'; shutil.copytree(bundle / 'tree', mirror)
        changes = relocate(mirror, manifest); i.write(output / 'RELOCATION.json', changes)
        argv = commands(mirror, python); i.write(output / 'COMMANDS.json', argv)
        for index, command in enumerate(argv):
            tick = time.monotonic()
            with (output / f'stage_{index}.stdout').open('x') as stdout, (output / f'stage_{index}.stderr').open('x') as stderr:
                result = subprocess.run(command, cwd=mirror, env=environment(output), stdout=stdout, stderr=stderr)
            i.write(output / f'stage_{index}.json', {'argv': command, 'returncode': result.returncode, 'seconds': time.monotonic()-tick})
            i.require(result.returncode == 0, 'Historical stage failed: ' + str(index))
            i.require(codes() == records, 'Controller changed during execution')
            print('value stage', index, 'complete', round(time.monotonic()-tick, 1), flush=True)
        verify_bundle(bundle, snapshot)
        for name, rec in manifest['files'].items():
            expected = next((x['after'] for x in changes if x['path'] == name), {k: rec[k] for k in ('bytes', 'sha256')})
            i.require(i.record(mirror / name) == expected, 'Runtime input or script changed')
        after = probe(python, output / 'controller', output, 'environment_after')
        i.require(after == before, 'Runtime changed'); i.write(output / 'ENVIRONMENT_AFTER.json', after)
        i.require(codes() == records and i.record(python) == request['python_record'], 'Controller/runtime changed')
        i.write(output / 'RUN_MANIFEST.json', {'files': tree(output)})
        i.write(output / 'COMPLETE.json', {'status': 'complete', 'manifest': i.record(output / 'RUN_MANIFEST.json')})
    except BaseException as error:
        i.write(output / 'FAILED.json', {'error': repr(error)}); raise


def verify_run(output, bundle, snapshot):
    output = Path(output).resolve(strict=True); bundle = Path(bundle).resolve(strict=True)
    manifest = verify_bundle(bundle, snapshot); request = i.read(output / 'REQUEST.json'); done = i.read(output / 'COMPLETE.json')
    i.require(done.get('status') == 'complete' and done['manifest'] == i.record(output / 'RUN_MANIFEST.json'), 'Run incomplete')
    i.require(i.read(output / 'RUN_MANIFEST.json')['files'] == tree(output, ('RUN_MANIFEST.json', 'COMPLETE.json')), 'Run inventory differs')
    i.require(request['code_records'] == codes() and request['bundle_manifest'] == i.record(bundle / 'MANIFEST.json'), 'Unknown controller/input')
    i.require(i.record(Path(request['python'])) == request['python_record'], 'Python changed')
    mirror = output / 'runtime/tree'; changes = i.read(output / 'RELOCATION.json')
    expected_names = {f'benchmark_results/{case}/fastmnn/alignment/data/manifest.json' for case in i.CASES}
    i.require(len(changes) == 4 and {x['path'] for x in changes} == expected_names, 'Unexpected relocation set')
    for row in changes:
        original = i.read(bundle / 'tree' / row['path']); relocated = i.read(mirror / row['path'])
        expected = json.loads(json.dumps(original)); relative = i.bind(mirror, original['input']['path']).relative_to(mirror).as_posix()
        expected['input']['path'] = str(mirror / relative)
        i.require(relocated == expected and row['old_value'] == original['input']['path'] and
                  row['new_value'] == expected['input']['path'] and row['field'] == ['input', 'path'] and
                  row['before'] == i.record(bundle / 'tree' / row['path']) and row['after'] == i.record(mirror / row['path']), 'Relocation altered semantics')
    for name, rec in manifest['files'].items():
        expected = next((x['after'] for x in changes if x['path'] == name), {k: rec[k] for k in ('bytes', 'sha256')})
        i.require(i.record(mirror / name) == expected, 'Runtime input differs')
    for name, rec in request['code_records'].items(): i.require(i.record(output / 'controller' / name) == rec, 'Controller copy differs')
    argv = commands(mirror, Path(request['python'])); i.require(i.read(output / 'COMMANDS.json') == argv, 'Commands changed')
    for index, command in enumerate(argv):
        row = i.read(output / f'stage_{index}.json')
        i.require(row['argv'] == command and row['returncode'] == 0, 'Incomplete stage')
    before = i.read(output / 'ENVIRONMENT_BEFORE.json')
    i.require(before == i.read(output / 'ENVIRONMENT_AFTER.json') and before['packages'] == PACKAGES and
              before['threads'] == {key: '1' for key in THREADS} and all(x['num_threads'] == 1 for x in before['threadpools']), 'Runtime differs')
    return {'status': 'passed', 'stages': len(argv), 'inputs': len(manifest['files']), 'unchanged_historical_scripts': len(SCRIPTS)}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'run', 'verify', 'verify-run', 'runtime'])
    for name in ('snapshot', 'bundle', 'source-root', 'output', 'python'): p.add_argument('--' + name, type=Path)
    args = p.parse_args()
    if args.action == 'runtime': print(json.dumps(runtime()))
    elif args.action == 'prepare': prepare(args.snapshot, args.source_root, args.output)
    elif args.action == 'run': execute(args.bundle, args.snapshot, args.output, args.python)
    elif args.action == 'verify': print(json.dumps(verify_bundle(args.bundle, args.snapshot)))
    else: print(json.dumps(verify_run(args.output, args.bundle, args.snapshot)))
