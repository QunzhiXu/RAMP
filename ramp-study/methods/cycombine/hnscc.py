"""Safe HNSCC story correction using the byte-identical historical R dispatcher."""
import argparse
import os
from pathlib import Path
import tempfile

import inventory as inv
import study

HERE = Path(__file__).resolve().parent
CASE = 'hnscc_story'
BASE = Path('figure_HNSCC/Comparison')
RESULT = Path('runtime/tree') / BASE / 'methods/cycombine'


def plan(case=CASE):
    if case != CASE:
        raise ValueError('Only HNSCC story is enabled here')
    catalog = inv.verify_local()
    if inv.sha(inv.SNAPSHOT / 'catalog.json') != study.CATALOG_SHA:
        raise ValueError('Audited catalog anchor changed')
    spec = next(s for s in catalog['cases'] if s['case'] == case)
    records = {r['source']: r for r in catalog['files']}
    names = [spec['input'], spec['metadata'], spec['marker_file'], spec['config'],
             str(BASE / 'input/cell_ids.npy'), str(BASE / 'input/manifest.json')]
    files = [{'path': 'inputs/' + name, 'source': name, 'origin': 'source',
              **{k: records[name][k] for k in ('bytes', 'sha256')}} for name in names]
    for leaf in ('run_r.R', 'cycombine_repro.R', 'run_r_methods.py'):
        name = str(BASE / 'scripts' / leaf); row = records[name]
        files.append({'path': 'scripts/' + leaf, 'source': name, 'origin': 'snapshot',
                      'snapshot_path': row['copy'], **{k: row[k] for k in ('bytes', 'sha256')}})
    for row in catalog['files']:
        copy = row.get('copy', '')
        if copy.startswith('adapter/') or copy == 'environment/validated_direct_runtime_packages.csv':
            files.append({'path': copy, 'source': row['source'], 'origin': 'snapshot',
                          'snapshot_path': copy, **{k: row[k] for k in ('bytes', 'sha256')}})
    return spec, sorted(files, key=lambda r: r['path'])


def validate_inputs(bundle, spec):
    import numpy as np
    inp = inv.table(bundle / 'inputs' / spec['input'])
    meta = inv.table(bundle / 'inputs' / spec['metadata'])
    ids = np.load(bundle / 'inputs' / BASE / 'input/cell_ids.npy', allow_pickle=False)
    markers = (bundle / 'inputs' / spec['marker_file']).read_text().splitlines()
    cfg = inv.read_json(bundle / 'inputs' / spec['config'])
    mp = bundle / 'inputs' / BASE / 'input/manifest.json'
    if cfg['input_manifest_sha256'] != inv.sha(mp) or cfg['datasets'] != spec['datasets'] or cfg['markers'] != markers:
        raise ValueError('Historical configuration/input contract differs')
    if markers != spec['markers'] or list(inp.columns) != ['cell_id', 'dataset', 'unit_key', *markers]:
        raise ValueError('Wrong input columns or marker order')
    if list(meta.columns) != ['cell_id', 'dataset', 'unit_key'] or not inp.iloc[:, :3].equals(meta):
        raise ValueError('Fit metadata differs from input')
    if len(inp) != spec['cells'] or len(set(ids)) != len(ids) or not np.array_equal(inp.cell_id, ids):
        raise ValueError('Wrong cell count or identity')
    if list(dict.fromkeys(inp.dataset)) != spec['datasets'] or not np.isfinite(inp[markers].to_numpy(float)).all():
        raise ValueError('Wrong dataset order or non-finite values')
    counts = []
    for expected in spec['preprocessing']:
        part = inp.loc[inp.dataset == expected['dataset']]
        if len(part) != expected['cells'] or part.unit_key.nunique() != expected['units']:
            raise ValueError('Wrong platform cell/unit counts')
        counts.append({'dataset': expected['dataset'], 'cells': len(part), 'units': int(part.unit_key.nunique())})
    if (meta.groupby('unit_key').dataset.nunique() != 1).any():
        raise ValueError('Unit spans multiple datasets')
    return {'cells': len(inp), 'markers': len(markers), 'units': int(meta.unit_key.nunique()), 'platforms': counts}


def verify_bundle(bundle):
    bundle = Path(bundle).resolve(strict=True)
    done = inv.read_json(bundle / 'COMPLETE.json')
    manifest = inv.read_json(bundle / 'MANIFEST.json')
    if done.get('status') != 'complete' or done.get('manifest_sha256') != inv.sha(bundle / 'MANIFEST.json'):
        raise ValueError('Incomplete bundle')
    spec, files = plan(manifest['case'])
    if manifest['catalog_sha256'] != study.CATALOG_SHA or manifest['files'] != files:
        raise ValueError('Bundle does not match audited contract')
    study.payload(bundle, files, ('MANIFEST.json', 'COMPLETE.json'))
    validate_inputs(bundle, spec)
    return spec, files


def prepare(source, output, case=CASE):
    source = Path(source).resolve(strict=True)
    spec, files = plan(case)
    bindings = [(row, inv.within(source, row['source']) if row['origin'] == 'source'
                 else inv.within(inv.SNAPSHOT, row['snapshot_path'])) for row in files]
    for row, path in bindings: inv.check(path, row)
    output = inv.reserve(output, (source, inv.SNAPSHOT, HERE / 'hnscc.py'))
    try:
        for row, path in bindings: inv.copy_new(path, output / row['path'], row)
        counts = validate_inputs(output, spec)
        for row, path in bindings: inv.check(path, row)
        if plan(case)[1] != files: raise ValueError('Contract changed')
        inv.create_json(output / 'MANIFEST.json', {'schema': 1, 'case': case,
            'catalog_sha256': study.CATALOG_SHA, 'source_root_at_preparation': str(source),
            'files': files, 'counts': counts, 'scope': 'Frozen analysis-ready HNSCC correction inputs'})
        study.payload(output, files, ('MANIFEST.json',))
        inv.create_json(output / 'COMPLETE.json', {'status': 'complete', 'manifest_sha256': inv.sha(output / 'MANIFEST.json')})
    except BaseException as error:
        inv.create_json(output / 'FAILED.json', {'error': repr(error)})
        raise
    return output


def runtime_path(row):
    if row['path'].startswith('inputs/'):
        return Path('runtime/tree') / row['source']
    if row['path'].startswith('scripts/'):
        return Path('runtime/tree') / BASE / row['path']
    if row['path'].startswith('adapter/'):
        return Path('runtime/tree/algorithm/cycombine') / Path(row['path']).relative_to('adapter')
    return None  # The environment CSV is read directly from the input bundle.


def code_records():
    return {name: inv.record(HERE / name) for name in
            ('hnscc.py', 'inventory.py', 'study.py', 'check_environment.R', 'check_hnscc_runtime.R')}


def check_r_output(result, bundle, spec):
    import numpy as np
    runtime = inv.read_json(result / 'R_COMPLETE.json')
    expected = {'cells': spec['cells'], 'dimensions': len(spec['markers']),
                'row_identity_verified': True, 'output_type': 'marker_values'}
    if (runtime['method'] != 'cycombine' or runtime['parameters'] != spec['parameters'] or runtime['result'] != expected
            or runtime['input_sha256'] != inv.sha(bundle / 'inputs' / spec['input'])
            or runtime['helper_sha256'] != inv.sha(bundle / 'scripts/cycombine_repro.R')
            or runtime['output_sha256'] != inv.sha(result / 'corrected_expression.csv.gz')):
        raise ValueError('R correction contract differs')
    out = inv.table(result / 'corrected_expression.csv.gz')
    ids = np.load(bundle / 'inputs' / BASE / 'input/cell_ids.npy', allow_pickle=False)
    if list(out.columns) != ['cell_id', *spec['markers']] or not np.array_equal(out.cell_id, ids):
        raise ValueError('Wrong output cell/marker order')
    if not np.isfinite(out[spec['markers']].to_numpy(float)).all(): raise ValueError('Non-finite output')
    return runtime


def finish_values(result, bundle, spec):
    """Historical pandas-default CSV→float64 NPY, with exclusive output creation."""
    import numpy as np
    import pandas as pd
    runtime = check_r_output(result, bundle, spec)
    parsed = pd.read_csv(result / 'corrected_expression.csv.gz', low_memory=False)
    ids_path = bundle / 'inputs' / BASE / 'input/cell_ids.npy'
    ids = np.load(ids_path, allow_pickle=False)
    if not np.array_equal(parsed.cell_id.astype(str), ids): raise ValueError('CSV conversion changed IDs')
    values = parsed.iloc[:, 1:].to_numpy(float)
    if not np.isfinite(values).all(): raise ValueError('Non-finite NPY')
    with (result / 'values.npy').open('xb') as stream: np.save(stream, values)
    inv.create_json(result / 'COMPLETE.json', {'method': 'cycombine', 'cells': len(values), 'dimensions': values.shape[1],
        'values_sha256': inv.sha(result / 'values.npy'), 'cell_ids_sha256': inv.sha(ids_path),
        'input_manifest_sha256': inv.sha(bundle / 'inputs' / BASE / 'input/manifest.json'), 'runtime': runtime})


def fit(bundle, library, rscript, output, scratch_parent):
    bundle = Path(bundle).resolve(strict=True); library = Path(library).resolve(strict=True)
    rscript = Path(rscript).absolute(); scratch_parent = Path(scratch_parent).resolve(strict=True)
    if any(c.isspace() for c in str(scratch_parent)): raise ValueError('R scratch must be space-free')
    spec, files = verify_bundle(bundle)
    codes = code_records(); executable = inv.record(rscript); bm = inv.record(bundle / 'MANIFEST.json')
    source = inv.read_json(bundle / 'MANIFEST.json')['source_root_at_preparation']
    output = inv.reserve(output, (bundle, library, rscript, inv.SNAPSHOT, source, *(HERE / name for name in codes)))
    try:
        inv.create_json(output / 'REQUEST.json', {'case': CASE, 'bundle': str(bundle), 'bundle_manifest': bm,
            'library': str(library), 'rscript': str(rscript), 'rscript_record': executable,
            'code_records': codes, 'parameters': spec['parameters']})
        for row in files:
            path = runtime_path(row)
            if path is not None: inv.copy_new(bundle / row['path'], output / path, row)
        adapter = output / 'runtime/tree/algorithm/cycombine'
        (adapter / '.Rlib').symlink_to(library, target_is_directory=True)
        here = output / 'runtime/tree' / BASE
        commands = {
            'environment': [rscript, '--vanilla', HERE / 'check_environment.R', adapter,
                            bundle / 'environment/validated_direct_runtime_packages.csv'],
            'story_runtime': [rscript, '--vanilla', HERE / 'check_hnscc_runtime.R', adapter, here / 'scripts/cycombine_repro.R'],
            'correction': [rscript, '--vanilla', here / 'scripts/run_r.R', here, 'cycombine'],
        }
        inv.create_json(output / 'COMMANDS.json', {key: list(map(str, argv)) for key, argv in commands.items()})
        env = os.environ.copy(); env.pop('R_LIBS', None)
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'): env[key] = '1'
        with tempfile.TemporaryDirectory(prefix='ramp_hnscc_cycombine_', dir=scratch_parent) as temp:
            env['TMPDIR'] = str(Path(temp).resolve())
            for key in ('environment', 'story_runtime'): study.run_command(commands[key], output, key + '_before', env)
            study.run_command(commands['correction'], output, 'correction', env)
            for key in ('environment', 'story_runtime'): study.run_command(commands[key], output, key + '_after', env)
        for key in ('environment', 'story_runtime'):
            if (output / (key + '_before.stdout.log')).read_bytes() != (output / (key + '_after.stdout.log')).read_bytes():
                raise ValueError('Runtime changed during fitting')
        finish_values(output / RESULT, bundle, spec)
        verify_bundle(bundle); inv.check(bundle / 'MANIFEST.json', bm); inv.check(rscript, executable)
        for name, record in codes.items(): inv.check(HERE / name, record)
        for row in files:
            path = runtime_path(row)
            if path is not None: inv.check(output / path, row)
        records = {p.relative_to(output).as_posix(): inv.record(p) for p in sorted(output.rglob('*'))
                   if p.is_file() and not p.is_symlink()}
        inv.create_json(output / 'RUN_MANIFEST.json', {'case': CASE, 'files': records,
            'history_used_by_fit': False, 'scope': 'Actual HNSCC correction only; no clinical analysis'})
        inv.create_json(output / 'COMPLETE.json', {'status': 'complete', 'manifest_sha256': inv.sha(output / 'RUN_MANIFEST.json')})
    except BaseException as error:
        inv.create_json(output / 'FAILED.json', {'error': repr(error)})
        raise
    return output


def verify_run(run, bundle):
    import numpy as np
    import pandas as pd
    run = Path(run).resolve(strict=True); bundle = Path(bundle).resolve(strict=True)
    spec, files = verify_bundle(bundle)
    done = inv.read_json(run / 'COMPLETE.json')
    if done.get('status') != 'complete' or done.get('manifest_sha256') != inv.sha(run / 'RUN_MANIFEST.json'):
        raise ValueError('Run incomplete')
    manifest = inv.read_json(run / 'RUN_MANIFEST.json'); request = inv.read_json(run / 'REQUEST.json')
    if manifest['case'] != CASE or request['case'] != CASE: raise ValueError('Wrong case')
    inv.check(bundle / 'MANIFEST.json', request['bundle_manifest'])
    for name, record in manifest['files'].items(): inv.check(inv.within(run, name), record)
    actual = {p.relative_to(run).as_posix() for p in run.rglob('*') if p.is_file() and not p.is_symlink()}
    if actual != set(manifest['files']) | {'RUN_MANIFEST.json', 'COMPLETE.json'}: raise ValueError('Run inventory differs')
    for row in files:
        path = runtime_path(row)
        if path is not None: inv.check(run / path, row)
    result = run / RESULT; runtime = check_r_output(result, bundle, spec)
    complete = inv.read_json(result / 'COMPLETE.json')
    for key, expected in (('method', 'cycombine'), ('cells', spec['cells']), ('dimensions', len(spec['markers'])), ('runtime', runtime),
                          ('values_sha256', inv.sha(result / 'values.npy')),
                          ('cell_ids_sha256', inv.sha(bundle / 'inputs' / BASE / 'input/cell_ids.npy')),
                          ('input_manifest_sha256', inv.sha(bundle / 'inputs' / BASE / 'input/manifest.json'))):
        if complete[key] != expected: raise ValueError('Nested conversion completion differs: ' + key)
    parsed = pd.read_csv(result / 'corrected_expression.csv.gz', low_memory=False).iloc[:, 1:].to_numpy(float)
    values = np.load(result / 'values.npy', allow_pickle=False)
    if values.dtype != np.float64 or not np.array_equal(values, parsed): raise ValueError('NPY differs from historical parsing convention')
    return validate_inputs(bundle, spec)


def compare(run, bundle, source):
    import numpy as np
    run, bundle, source = (Path(p).resolve(strict=True) for p in (run, bundle, source))
    counts = verify_run(run, bundle)
    spec, _ = plan(); catalog = inv.verify_local(); records = {r['source']: r for r in catalog['files']}
    old = Path(spec['output']).parent; result = run / RESULT; checks = {}
    for name in ('corrected_expression.csv.gz', 'fit.rds', 'marker_batch_statistics.csv.gz',
                 'node_batch_counts.csv.gz', 'node_correction_modes.csv.gz', 'values.npy'):
        original = inv.within(source, old / name); inv.check(original, records[str(old / name)])
        current = result / name
        if inv.sha(original) != inv.sha(current): raise ValueError('Historical file differs: ' + name)
        checks[name] = {'bytes_equal': True, 'sha256': inv.sha(current)}
    a = np.load(source / spec['values_npy'], allow_pickle=False)
    b = np.load(result / 'values.npy', allow_pickle=False)
    if a.dtype != b.dtype or not np.array_equal(a, b): raise ValueError('Historical NPY differs')
    path = inv.within(source, old / 'R_COMPLETE.json'); inv.check(path, records[str(old / 'R_COMPLETE.json')])
    prior, current = inv.read_json(path), inv.read_json(result / 'R_COMPLETE.json')
    for key in set(prior) | set(current):
        if key != 'elapsed_seconds' and prior.get(key) != current.get(key): raise ValueError('Historical R completion differs: ' + key)
    md_path = inv.within(source, spec['canonical']['evaluation_metadata']); inv.check(md_path, records[spec['canonical']['evaluation_metadata']])
    meta = inv.table(md_path); inp = inv.table(bundle / 'inputs' / spec['input'])
    if not meta[['cell_id', 'dataset', 'unit_key']].equals(inp[['cell_id', 'dataset', 'unit_key']]):
        raise ValueError('Evaluation identity differs')
    return {**counts, 'comparisons': checks, 'NPY_max_abs_difference': float(np.max(np.abs(a - b))),
            'R_completion_equal_except_elapsed_seconds': True, 'evaluation_identity_matches': True}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); subs = p.add_subparsers(dest='command', required=True)
    prep = subs.add_parser('prepare'); prep.add_argument('--case', choices=[CASE], default=CASE)
    prep.add_argument('--source-root', type=Path, required=True); prep.add_argument('--output', type=Path, required=True)
    for name in ('fit', 'verify', 'verify-run', 'compare'):
        sub = subs.add_parser(name); sub.add_argument('--bundle', type=Path, required=True)
        if name == 'fit':
            for flag in ('library', 'rscript', 'scratch-parent', 'output'): sub.add_argument('--' + flag, type=Path, required=True)
        if name in ('verify-run', 'compare'): sub.add_argument('--run', type=Path, required=True)
        if name == 'compare': sub.add_argument('--source-root', type=Path, required=True)
    a = p.parse_args(argv)
    if a.command == 'prepare': print(prepare(a.source_root, a.output, a.case))
    elif a.command == 'fit': print(fit(a.bundle, a.library, a.rscript, a.output, a.scratch_parent))
    elif a.command == 'verify': print({'case': verify_bundle(a.bundle)[0]['case'], 'status': 'passed'})
    elif a.command == 'verify-run': print(verify_run(a.run, a.bundle))
    else:
        import json
        print(json.dumps(compare(a.run, a.bundle, a.source_root), indent=2))


if __name__ == '__main__': main()
