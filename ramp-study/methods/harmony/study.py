"""Create-only Harmony-direct execution from frozen method-ready inputs.

Safety flow adapted from cyCombine/study.py and lung.py; scientific R code is
copied unchanged. No preprocessing, selection, UMAP, metrics or survival here.
"""
import argparse
import copy
import os
from pathlib import Path
import subprocess
import tempfile

import harmony_io as io
import inventory as inv

HERE = Path(__file__).resolve().parent
R_LOCALE_ENV = {'LC_ALL': 'C.UTF-8', 'LANG': 'C.UTF-8'}


def r_environment(parent=None):
    """Lock the historical R collation at process startup, not after loading code."""
    env = dict(os.environ if parent is None else parent)
    env.pop('R_LIBS', None)
    env.update(R_LOCALE_ENV)
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        env[key] = '1'
    return env


def plan(case, profile='source_scale'):
    catalog = inv.verify_local()
    matches = [s for s in catalog['branches'] if (s['case'], s['profile']) == (case, profile)]
    if len(matches) != 1: raise ValueError('Unknown case/profile combination')
    spec = matches[0]; by_source = {r['source']: r for r in catalog['files']}
    needed = [spec[k] for k in ('input', 'marker_file', 'config', 'input_manifest')]
    needed += [spec['record']] if spec['kind'] == 'technical' else [spec['metadata'], spec['ids']]
    files = []
    for name in needed:
        row = by_source[name]
        files.append({'path': 'inputs/' + name, 'source': name, 'origin': 'source',
                      **{k: row[k] for k in ('bytes', 'sha256')}})
    for row in catalog['files']:
        if row.get('copy', '').startswith('adapter/'):
            files.append({'path': row['copy'], 'source': row['source'], 'origin': 'snapshot',
                          'snapshot_path': row['copy'], **{k: row[k] for k in ('bytes', 'sha256')}})
    if spec['kind'] == 'story':
        row = by_source[spec['base'] + '/scripts/run_r.R']
        files.append({'path': 'scripts/run_r.R', 'source': row['source'], 'origin': 'snapshot',
                      'snapshot_path': row['copy'], **{k: row[k] for k in ('bytes', 'sha256')}})
    return spec, sorted(files, key=lambda r: r['path'])


def validate_inputs(bundle, spec):
    import numpy as np
    root = Path(bundle) / 'inputs'
    inp = io.table(root / spec['input']); markers = (root / spec['marker_file']).read_text().splitlines()
    if (markers != spec['markers'] or list(inp.columns) != ['cell_id', 'dataset', 'unit_key', *markers]
            or len(inp) != spec['cells'] or inp.cell_id.duplicated().any() or (inp.iloc[:, :3] == '').any().any()):
        raise ValueError('Wrong input cells/columns/markers')
    if (list(dict.fromkeys(inp.dataset)) != spec['datasets'] or inp.unit_key.nunique() != spec['units'] or
            (inp.groupby('unit_key').dataset.nunique() != 1).any() or not np.isfinite(inp[markers].to_numpy(float)).all()):
        raise ValueError('Wrong input datasets/units or non-finite values')
    cfg = io.read_json(root / spec['config'])
    if spec['kind'] == 'technical':
        if cfg['algorithm_parameters'] != inv.PARAMETERS or cfg['batch_levels'] != spec['datasets']:
            raise ValueError('Technical parameters/datasets differ')
        scale = cfg['input_scale']
        for key, path in [('input_sha256', spec['input']), ('marker_file_sha256', spec['marker_file']),
                          ('preprocessing_record_sha256', spec['record'])]:
            if scale[key] != io.sha(root / path): raise ValueError('Technical input hash differs: ' + key)
        if scale['status'] != 'analysis_ready' or scale['algorithm_applies_transformation'] is not False:
            raise ValueError('Unexpected technical transform')
    else:
        ids = np.load(root / spec['ids'], allow_pickle=False)
        if not np.array_equal(ids, inp.cell_id) or not io.table(root / spec['metadata']).equals(inp.iloc[:, :3]):
            raise ValueError('Story metadata/IDs differ')
        if (cfg['datasets'] != spec['datasets'] or cfg['markers'] != markers or cfg['seed'] != 42 or
                cfg['input_manifest_sha256'] != io.sha(root / spec['input_manifest'])):
            raise ValueError('Story input configuration differs')
        manifest = io.read_json(root / spec['input_manifest'])
        for key in ('input', 'marker_file', 'metadata', 'ids'):
            if manifest['output_sha256'][Path(spec[key]).name] != io.sha(root / spec[key]):
                raise ValueError('Story input hash differs: ' + key)
    return {'cells': len(inp), 'markers': len(markers), 'units': int(inp.unit_key.nunique())}


def verify_bundle(bundle):
    bundle = Path(bundle).resolve(strict=True)
    manifest = io.complete(bundle, 'MANIFEST.json')
    if manifest.get('status') != 'complete': raise ValueError('Bundle manifest not complete')
    spec, files = plan(manifest['case'], manifest['profile'])
    if manifest['catalog_sha256'] != inv.CATALOG_SHA or manifest['files'] != files:
        raise ValueError('Bundle differs from frozen contract')
    io.payload(bundle, {r['path']: r for r in files}, ('MANIFEST.json', 'COMPLETE.json'))
    validate_inputs(bundle, spec)
    return spec, files


def prepare(source, output, case, profile='source_scale'):
    source = Path(source).resolve(strict=True)
    spec, files = plan(case, profile)
    bindings = [(r, io.within(source, r['source']) if r['origin'] == 'source'
                 else io.within(inv.SNAPSHOT, r['snapshot_path'])) for r in files]
    for row, path in bindings: io.check(path, row)
    output = io.reserve(output, (source, inv.SNAPSHOT, *(HERE / n for n in code_records())))
    try:
        for row, path in bindings: io.copy_new(path, output / row['path'], row)
        counts = validate_inputs(output, spec)
        for row, path in bindings: io.check(path, row)
        if plan(case, profile) != (spec, files): raise ValueError('Contract changed during preparation')
        io.create_json(output / 'MANIFEST.json', {'schema': 1, 'status': 'complete', 'case': case, 'profile': profile,
            'catalog_sha256': inv.CATALOG_SHA, 'files': files, 'counts': counts,
            'source_root_at_preparation': str(source), 'scope': 'Frozen method-ready inputs only'})
        io.payload(output, {r['path']: r for r in files}, ('MANIFEST.json',))
        io.create_json(output / 'COMPLETE.json', {'status': 'complete', 'manifest_sha256': io.sha(output / 'MANIFEST.json')})
    except BaseException as error:
        io.failure(output, error); raise
    return output


def code_records():
    return {name: io.record(HERE / name) for name in ('harmony_io.py', 'inventory.py', 'study.py', 'check_environment.R')}


def runtime_path(row, spec):
    if row['path'].startswith('adapter/'):
        return Path('runtime/tree/algorithm/harmony') / Path(row['path']).relative_to('adapter')
    if spec['kind'] == 'story':
        if row['path'].startswith('inputs/'): return Path('runtime/tree') / row['source']
        if row['path'] == 'scripts/run_r.R': return Path('runtime/tree') / spec['base'] / row['path']
    return None


def result_path(spec):
    return Path('result') if spec['kind'] == 'technical' else Path('runtime/tree') / spec['base'] / 'methods/harmony'


def relocated_config(bundle, output, spec):
    cfg = copy.deepcopy(io.read_json(bundle / 'inputs' / spec['config']))
    cfg['input_scale']['preprocessing_record_file'] = str(bundle / 'inputs' / spec['record'])
    return cfg


def run_command(argv, output, name, env):
    with (output / (name + '.stdout.log')).open('x') as out, (output / (name + '.stderr.log')).open('x') as err:
        proc = subprocess.run(list(map(str, argv)), cwd=output, env=env, stdout=out, stderr=err)
    if proc.returncode: raise RuntimeError(f'{name} exited {proc.returncode}; see logs')


def check_output(run, bundle, spec, conversion=False):
    import numpy as np
    import pandas as pd
    result = Path(run) / result_path(spec)
    inp = io.table(bundle / 'inputs' / spec['input'])
    outname = 'harmony_adjusted_marker_coordinates.csv.gz' if spec['kind'] == 'technical' else 'corrected_expression.csv.gz'
    out = io.table(result / outname)
    if list(out.columns) != ['cell_id', *spec['markers']] or not np.array_equal(out.cell_id, inp.cell_id):
        raise ValueError('Output cell/marker order differs')
    if not np.isfinite(out[spec['markers']].to_numpy(float)).all(): raise ValueError('Non-finite corrected values')
    meta = io.table(result / ('cell_metadata_label_free.csv.gz' if spec['kind'] == 'technical' else 'cell_metadata.csv.gz'))
    if (list(meta.columns) != ['cell_id', 'batch', 'sample'] or not np.array_equal(meta.cell_id, inp.cell_id)
            or not np.array_equal(meta.batch, inp.dataset) or not np.array_equal(meta['sample'], inp.unit_key)):
        raise ValueError('Output unit/dataset provenance differs')
    if spec['kind'] == 'technical':
        m = io.complete(result, 'manifest.json')
        if (m.get('status') != 'complete' or m['algorithm']['resolved_parameters'] != spec['parameters'] or
                m['algorithm']['pca_applied'] is not False or m['input']['markers'] != spec['markers'] or
                m['input']['n_cells'] != spec['cells'] or m['input']['algorithm_applies_transformation'] is not False):
            raise ValueError('Technical scientific output contract differs')
        for key, target in [('sha256', bundle / 'inputs' / spec['input']), ('marker_file_sha256', bundle / 'inputs' / spec['marker_file']),
                            ('preprocessing_record_sha256', bundle / 'inputs' / spec['record'])]:
            if m['input'][key] != io.sha(target): raise ValueError('R input hash differs: ' + key)
        for key, name in [('wrapper_file', 'harmony_direct_repro.R'), ('runner_file', 'run_harmony_direct.R'),
                          ('upstream_lock', 'upstream_lock.json'), ('runtime_lock', 'runtime_lock.csv')]:
            if m['implementation'][key]['sha256'] != io.sha(bundle / 'adapter' / name): raise ValueError('R implementation differs')
        if m['config_file']['sha256'] != io.sha(run / 'runtime_config.json'): raise ValueError('R runtime config differs')
        expected = {r['file'] for r in m['output_files']}
        required = {'cell_metadata_label_free.csv.gz', outname, 'harmony_adjusted_marker_coordinates.rds',
                    'marker_rank_audit.csv', 'marker_rank_audit_by_sample.csv', 'marker_value_shift_audit.csv',
                    'harmony_runtime_messages.txt', 'input_column_audit.json', 'method_semantics.json', 'resolved_config.json', 'sessionInfo.txt'}
        if expected != required or len(m['output_files']) != len(required): raise ValueError('R output set differs')
        for row in m['output_files']:
            if io.sha(io.within(result, row['file'])) != row['sha256']: raise ValueError('R output hash differs')
        if {p.name for p in result.iterdir()} != required | {'manifest.json', 'COMPLETE.json'}: raise ValueError('Extra/missing R output')
    else:
        rc = io.read_json(result / 'R_COMPLETE.json')
        if (rc['method'] != 'harmony' or rc['parameters'] != spec['parameters'] or
            rc['result'] != {'cells': spec['cells'], 'dimensions': len(spec['markers']), 'row_identity_verified': True, 'output_type': 'marker_values'} or
            rc['input_sha256'] != io.sha(bundle / 'inputs' / spec['input']) or
            rc['helper_sha256'] != io.sha(bundle / 'adapter/harmony_direct_repro.R') or rc['output_sha256'] != io.sha(result / outname)):
            raise ValueError('Story R contract differs')
        required = {outname, 'fit.rds', 'cell_metadata.csv.gz', 'marker_rank_audit.csv.gz',
                    'marker_rank_audit_by_sample.csv.gz', 'marker_value_shift_audit.csv.gz', 'sessionInfo.txt', 'R_COMPLETE.json'}
        if conversion: required |= {'values.npy', 'COMPLETE.json'}
        if {p.name for p in result.iterdir()} != required: raise ValueError('Story output set differs')
        if conversion:
            done = io.read_json(result / 'COMPLETE.json')
            for key, value in [('method', 'harmony'), ('cells', spec['cells']), ('dimensions', len(spec['markers'])), ('runtime', rc),
                               ('values_sha256', io.sha(result / 'values.npy')), ('cell_ids_sha256', io.sha(bundle / 'inputs' / spec['ids'])),
                               ('input_manifest_sha256', io.sha(bundle / 'inputs' / spec['input_manifest']))]:
                if done[key] != value: raise ValueError('Story completion differs: ' + key)
            values = np.load(result / 'values.npy', allow_pickle=False)
            parsed = pd.read_csv(result / outname, low_memory=False).iloc[:, 1:].to_numpy(float)
            if values.dtype != np.float64 or not np.array_equal(values, parsed): raise ValueError('NPY conversion differs')
    return {'cells': len(inp), 'markers': len(spec['markers']), 'units': int(inp.unit_key.nunique()), 'ordered_identity_matches': True}


def finish_story(run, bundle, spec):
    import numpy as np
    import pandas as pd
    check_output(run, bundle, spec)
    result = run / result_path(spec)
    parsed = pd.read_csv(result / 'corrected_expression.csv.gz', low_memory=False)
    ids = np.load(bundle / 'inputs' / spec['ids'], allow_pickle=False)
    if not np.array_equal(parsed.cell_id.astype(str), ids): raise ValueError('Default CSV parsing changed IDs')
    values = parsed.iloc[:, 1:].to_numpy(float)
    if not np.isfinite(values).all(): raise ValueError('Non-finite NPY')
    with (result / 'values.npy').open('xb') as stream: np.save(stream, values)
    io.create_json(result / 'COMPLETE.json', {'method': 'harmony', 'cells': len(values), 'dimensions': values.shape[1],
        'runtime': io.read_json(result / 'R_COMPLETE.json'), 'values_sha256': io.sha(result / 'values.npy'),
        'cell_ids_sha256': io.sha(bundle / 'inputs' / spec['ids']),
        'input_manifest_sha256': io.sha(bundle / 'inputs' / spec['input_manifest'])})


def fit(bundle, library, rscript, output, scratch_parent):
    bundle = Path(bundle).resolve(strict=True); library = Path(library).resolve(strict=True)
    rscript = Path(rscript).absolute(); scratch_parent = Path(scratch_parent).resolve(strict=True)
    if not library.is_dir() or not scratch_parent.is_dir(): raise ValueError('Library/scratch must be directories')
    if any(c.isspace() for c in str(scratch_parent)): raise ValueError('R scratch parent must be space-free')
    spec, files = verify_bundle(bundle)
    codes = code_records(); executable = io.record(rscript); bm = io.record(bundle / 'MANIFEST.json')
    source = io.read_json(bundle / 'MANIFEST.json')['source_root_at_preparation']
    output = io.reserve(output, (bundle, library, rscript, inv.SNAPSHOT, source, *(HERE / n for n in codes)))
    try:
        io.create_json(output / 'REQUEST.json', {'case': spec['case'], 'profile': spec['profile'], 'bundle': str(bundle),
            'bundle_manifest': bm, 'library': str(library), 'rscript': str(rscript), 'rscript_record': executable,
            'code_records': codes, 'catalog_sha256': inv.CATALOG_SHA, 'r_locale_environment': dict(R_LOCALE_ENV)})
        for row in files:
            target = runtime_path(row, spec)
            if target is not None: io.copy_new(bundle / row['path'], output / target, row)
        adapter = output / 'runtime/tree/algorithm/harmony'
        (adapter / '.Rlib').symlink_to(library, target_is_directory=True)
        commands = {'environment': [rscript, '--vanilla', HERE / 'check_environment.R', adapter, spec['kind']]}
        if spec['kind'] == 'technical':
            io.create_json(output / 'runtime_config.json', relocated_config(bundle, output, spec))
            io.create_json(output / 'CONFIG_RELOCATION.json', {'changed_field': 'input_scale.preprocessing_record_file',
                'original_sha256': io.sha(bundle / 'inputs' / spec['config']), 'derived_sha256': io.sha(output / 'runtime_config.json')})
            commands['correction'] = [rscript, '--vanilla', adapter / 'run_harmony_direct.R',
                '--input', bundle / 'inputs' / spec['input'], '--markers', bundle / 'inputs' / spec['marker_file'],
                '--config', output / 'runtime_config.json', '--output', output / result_path(spec)]
        else:
            here = output / 'runtime/tree' / spec['base']
            commands['correction'] = [rscript, '--vanilla', here / 'scripts/run_r.R', here, 'harmony']
        io.create_json(output / 'COMMANDS.json', {key: list(map(str, value)) for key, value in commands.items()})
        env = r_environment()
        with tempfile.TemporaryDirectory(prefix='ramp_harmony_', dir=scratch_parent) as temp:
            env['TMPDIR'] = str(Path(temp).resolve())
            run_command(commands['environment'], output, 'environment_before', env)
            run_command(commands['correction'], output, 'correction', env)
            run_command(commands['environment'], output, 'environment_after', env)
        if (output / 'environment_before.stdout.log').read_bytes() != (output / 'environment_after.stdout.log').read_bytes():
            raise ValueError('Environment changed during fit')
        if spec['kind'] == 'story': finish_story(output, bundle, spec)
        check_output(output, bundle, spec, conversion=spec['kind'] == 'story')
        verify_bundle(bundle); io.check(bundle / 'MANIFEST.json', bm); io.check(rscript, executable)
        for name, row in codes.items(): io.check(HERE / name, row)
        for row in files:
            target = runtime_path(row, spec)
            if target is not None: io.check(output / target, row)
        if spec['kind'] == 'technical' and io.read_json(output / 'runtime_config.json') != relocated_config(bundle, output, spec):
            raise ValueError('Runtime config changed')
        records = {p.relative_to(output).as_posix(): io.record(p) for p in sorted(output.rglob('*')) if p.is_file() and not p.is_symlink()}
        links = {'runtime/tree/algorithm/harmony/.Rlib': str(library)}
        io.payload(output, records, allowed_links=links)
        io.create_json(output / 'RUN_MANIFEST.json', {'schema': 1, 'status': 'complete', 'case': spec['case'], 'profile': spec['profile'],
            'files': records, 'links': links, 'history_used_by_fit': False, 'scope': 'Harmony correction only'})
        io.create_json(output / 'COMPLETE.json', {'status': 'complete', 'manifest_sha256': io.sha(output / 'RUN_MANIFEST.json')})
    except BaseException as error:
        io.failure(output, error); raise
    return output


def verify_run(run, bundle):
    run, bundle = Path(run).resolve(strict=True), Path(bundle).resolve(strict=True)
    spec, files = verify_bundle(bundle); m = io.complete(run, 'RUN_MANIFEST.json'); req = io.read_json(run / 'REQUEST.json')
    if (m.get('status') != 'complete' or (m['case'], m['profile']) != (spec['case'], spec['profile']) or
            (req['case'], req['profile']) != (spec['case'], spec['profile']) or req['catalog_sha256'] != inv.CATALOG_SHA or
            req.get('r_locale_environment') != R_LOCALE_ENV):
        raise ValueError('Run contract differs')
    io.check(bundle / 'MANIFEST.json', req['bundle_manifest'])
    links = {'runtime/tree/algorithm/harmony/.Rlib': req['library']}
    if m['links'] != links: raise ValueError('Runtime library binding differs')
    io.payload(run, m['files'], ('RUN_MANIFEST.json', 'COMPLETE.json'), links)
    for row in files:
        target = runtime_path(row, spec)
        if target is not None: io.check(run / target, row)
    if spec['kind'] == 'technical':
        # The stored bundle path is provenance; moved bundles remain verifiable.
        original_bundle = Path(req['bundle'])
        cfg = copy.deepcopy(io.read_json(bundle / 'inputs' / spec['config']))
        cfg['input_scale']['preprocessing_record_file'] = str(original_bundle / 'inputs' / spec['record'])
        if io.read_json(run / 'runtime_config.json') != cfg: raise ValueError('Unexpected runtime config edit')
        relocation = io.read_json(run / 'CONFIG_RELOCATION.json')
        if relocation != {'changed_field': 'input_scale.preprocessing_record_file',
                          'original_sha256': io.sha(bundle / 'inputs' / spec['config']), 'derived_sha256': io.sha(run / 'runtime_config.json')}:
            raise ValueError('Config relocation record differs')
    return check_output(run, bundle, spec, conversion=spec['kind'] == 'story')


def compare(run, bundle, source):
    """Optional read-only historical comparison. Never called by fit()."""
    import numpy as np
    run, bundle, source = (Path(p).resolve(strict=True) for p in (run, bundle, source))
    summary = verify_run(run, bundle); spec, _ = verify_bundle(bundle)
    records = {r['source']: r for r in inv.verify_local()['files']}
    names = (['harmony_adjusted_marker_coordinates.csv.gz', 'harmony_adjusted_marker_coordinates.rds',
              'cell_metadata_label_free.csv.gz', 'marker_rank_audit.csv', 'marker_rank_audit_by_sample.csv', 'marker_value_shift_audit.csv']
             if spec['kind'] == 'technical' else ['corrected_expression.csv.gz', 'fit.rds', 'cell_metadata.csv.gz',
              'marker_rank_audit.csv.gz', 'marker_rank_audit_by_sample.csv.gz', 'marker_value_shift_audit.csv.gz', 'values.npy'])
    results = {}
    for name in names:
        oldname = str(Path(spec['output']).parent / name); old = io.within(source, oldname); io.check(old, records[oldname])
        new = run / result_path(spec) / name; exact = io.sha(old) == io.sha(new)
        row = {'bytes_equal': exact, 'old_sha256': io.sha(old), 'new_sha256': io.sha(new)}
        if name.endswith(('.csv', '.csv.gz')):
            a, b = io.table(old), io.table(new); row['decoded_equal'] = a.equals(b)
            if not row['decoded_equal']: raise ValueError('Historical table differs: ' + name)
            if name == Path(spec['output']).name:
                row['maximum_absolute_difference'] = float(np.max(np.abs(a[spec['markers']].to_numpy(float) - b[spec['markers']].to_numpy(float))))
        elif not exact: raise ValueError('Historical binary differs: ' + name)
        results[name] = row
    return {**summary, 'comparisons': results}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); subs = p.add_subparsers(dest='command', required=True)
    sub = subs.add_parser('prepare'); sub.add_argument('--case', choices=(*inv.TECHNICAL, *inv.STORIES), required=True)
    sub.add_argument('--profile', choices=inv.PROFILES, default='source_scale')
    sub.add_argument('--source-root', type=Path, required=True); sub.add_argument('--output', type=Path, required=True)
    for name in ('fit', 'verify', 'verify-run', 'compare'):
        sub = subs.add_parser(name); sub.add_argument('--bundle', type=Path, required=True)
        if name == 'fit':
            for flag in ('library', 'rscript', 'output', 'scratch-parent'): sub.add_argument('--' + flag, type=Path, required=True)
        if name in ('verify-run', 'compare'): sub.add_argument('--run', type=Path, required=True)
        if name == 'compare': sub.add_argument('--source-root', type=Path, required=True)
    a = p.parse_args(argv)
    try:
        if a.command == 'prepare': print(prepare(a.source_root, a.output, a.case, a.profile))
        elif a.command == 'fit': print(fit(a.bundle, a.library, a.rscript, a.output, a.scratch_parent))
        elif a.command == 'verify': print({'case': verify_bundle(a.bundle)[0]['case'], 'status': 'passed'})
        elif a.command == 'verify-run': print(verify_run(a.run, a.bundle))
        else: print(compare(a.run, a.bundle, a.source_root))
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        p.exit(1, f'error: {error}\n')


if __name__ == '__main__': main()
