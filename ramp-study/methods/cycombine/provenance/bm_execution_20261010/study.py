"""Create-only BM correction from frozen method-ready inputs; no preprocessing."""
import argparse
import csv
import os
from pathlib import Path
import posixpath
import subprocess
import tempfile

import inventory as inv

CATALOG_SHA = '2a1cff9635875a8930f64e4e94935bb3d0a8612f3e8d90edd5ccffc9b1bd19eb'
HERE = Path(__file__).resolve().parent


def relative_reference(value, base):
    # BM's historical config/manifest are already relative. Preserve their bytes
    # by mirroring the necessary source subtree, not editing provenance fields.
    if Path(value).is_absolute():
        raise ValueError('This stage only supports relative historical references')
    result = posixpath.normpath(str(Path(base) / value))
    if result == '..' or result.startswith('../'):
        raise ValueError('Reference escapes input root')
    return result


def plan(case='bone_marrow'):
    if case != 'bone_marrow':
        raise ValueError('Only bone_marrow execution is enabled in this audit round')
    catalog = inv.verify_local()
    if inv.sha(inv.SNAPSHOT / 'catalog.json') != CATALOG_SHA:
        raise ValueError('Audited catalog anchor changed')
    spec = next(s for s in catalog['cases'] if s['case'] == case)
    records = {r['source']: r for r in catalog['files']}
    config = inv.read_json(inv.SNAPSHOT / records[spec['config']]['copy'])
    pp = relative_reference(config['input_scale']['provenance_file'], Path(spec['config']).parent)
    manifest = inv.read_json(inv.SNAPSHOT / records[pp]['copy'])
    needed = {spec['config'], pp, spec['input'], spec['marker_file']}
    for key in ('correction_marker_file', 'marker_harmonization_manifest_file', 'cell_selection_manifest_file'):
        needed.add(relative_reference(manifest[key], Path(pp).parent))
    for row in manifest['datasets']:
        for key in ('input_matrix_file', 'processing_code_file'):
            needed.add(relative_reference(row[key], Path(pp).parent))
    files = [{'path': 'inputs/' + name, 'source': name, 'origin': 'source',
              **{k: records[name][k] for k in ('bytes', 'sha256')}} for name in sorted(needed)]
    for row in catalog['files']:
        copy = row.get('copy', '')
        if copy.startswith('adapter/') or copy == 'environment/validated_direct_runtime_packages.csv':
            files.append({'path': copy, 'source': row['source'], 'origin': 'snapshot',
                          **{k: row[k] for k in ('bytes', 'sha256')}})
    return spec, sorted(files, key=lambda r: r['path'])


def payload(bundle, files, extras):
    expected = {r['path'] for r in files} | set(extras)
    actual = set()
    for p in bundle.rglob('*'):
        if p.is_symlink():
            raise ValueError('Bundle cannot contain symlinks')
        if p.is_file():
            actual.add(p.relative_to(bundle).as_posix())
    if actual != expected:
        raise ValueError('Bundle inventory differs')
    for row in files:
        inv.check(inv.within(bundle, row['path']), row)


def verify_bundle(bundle):
    bundle = Path(bundle).resolve(strict=True)
    complete = inv.read_json(bundle / 'COMPLETE.json')
    manifest = inv.read_json(bundle / 'MANIFEST.json')
    if complete.get('status') != 'complete' or complete.get('manifest_sha256') != inv.sha(bundle / 'MANIFEST.json'):
        raise ValueError('Incomplete or changed bundle')
    spec, files = plan(manifest['case'])
    if manifest['catalog_sha256'] != CATALOG_SHA or manifest['files'] != files:
        raise ValueError('Bundle does not match audited contract')
    payload(bundle, files, ('MANIFEST.json', 'COMPLETE.json'))
    return spec, files


def prepare(source, output, case='bone_marrow'):
    source = Path(source).resolve(strict=True)
    spec, files = plan(case)
    bindings = [(r, inv.within(source, r['source']) if r['origin'] == 'source'
                 else inv.within(inv.SNAPSHOT, r['path'])) for r in files]
    for row, path in bindings:
        inv.check(path, row)
    output = inv.reserve(output, (source, inv.SNAPSHOT, HERE / 'study.py', HERE / 'inventory.py'))
    try:
        for row, path in bindings:
            inv.copy_new(path, output / row['path'], row)
        for row, path in bindings:
            inv.check(path, row)
        if plan(case)[1] != files:
            raise ValueError('Contract changed during preparation')
        inv.create_json(output / 'MANIFEST.json', {
            'schema': 1, 'case': case, 'catalog_sha256': CATALOG_SHA,
            'source_root_at_preparation': str(source), 'files': files,
            'scope': 'Frozen analysis-ready correction inputs; not provider extraction',
        })
        payload(output, files, ('MANIFEST.json',))
        inv.create_json(output / 'COMPLETE.json', {
            'status': 'complete', 'manifest_sha256': inv.sha(output / 'MANIFEST.json'),
        })
    except BaseException as error:
        inv.create_json(output / 'FAILED.json', {'error': repr(error)})
        raise
    return output


def code_records():
    return {name: inv.record(HERE / name) for name in
            ('study.py', 'inventory.py', 'check_environment.R')}


def run_command(argv, output, name, env):
    # Called only inside an exclusively reserved run. Preserve diagnostic logs.
    with (output / (name + '.stdout.log')).open('x') as stdout, (output / (name + '.stderr.log')).open('x') as stderr:
        p = subprocess.run(list(map(str, argv)), cwd=output, env=env, stdout=stdout, stderr=stderr)
    if p.returncode:
        raise RuntimeError(f'{name} exited {p.returncode}; see its logs')


def check_output(result, bundle, spec):
    import numpy as np
    done = inv.read_json(result / 'COMPLETE.json')
    if done.get('status') != 'complete' or done.get('manifest_sha256') != inv.sha(result / 'manifest.json'):
        raise ValueError('R output is incomplete')
    m = inv.read_json(result / 'manifest.json')
    if (m['algorithm_parameters'] != spec['parameters'] or m['markers'] != spec['markers']
            or m['batch_levels'] != spec['datasets'] or not m['exact_dataset_binding_verified']):
        raise ValueError('R output scientific contract differs')
    for row in m['outputs'].values():
        if inv.sha(inv.within(result, row['path'])) != row['sha256']:
            raise ValueError('R output hash differs')
    inp = inv.table(bundle / 'inputs' / spec['input'])
    out = inv.table(result / 'corrected_expression.csv.gz')
    meta = inv.table(result / 'cell_metadata_label_free.csv.gz')
    som = inv.table(result / 'correction_som_assignments.csv.gz')
    if len(inp) != spec['cells'] or list(out.columns) != ['cell_id', *spec['markers']]:
        raise ValueError('Wrong matrix shape/markers')
    for table in (out, meta, som):
        if not np.array_equal(table.cell_id, inp.cell_id):
            raise ValueError('Wrong output cell order')
    if not np.array_equal(meta.batch, inp.dataset) or not np.array_equal(meta['sample'], inp.unit_key):
        raise ValueError('Wrong dataset/unit assignment')
    if not np.isfinite(out[spec['markers']].to_numpy(float)).all():
        raise ValueError('Non-finite corrected matrix')
    return {'cells': len(out), 'markers': len(spec['markers']), 'units': int(inp.unit_key.nunique()),
            'finite': True, 'ordered_ids_and_units_match': True}


def fit(bundle, library, rscript, output, scratch_parent):
    bundle = Path(bundle).resolve(strict=True)
    library = Path(library).resolve(strict=True)
    rscript = Path(rscript).absolute()  # Preserve executable entry point.
    scratch_parent = Path(scratch_parent).resolve(strict=True)
    if any(c.isspace() for c in str(scratch_parent)):
        raise ValueError('R scratch parent must not contain whitespace')
    spec, files = verify_bundle(bundle)
    codes = code_records()
    executable = inv.record(rscript)
    bundle_record = inv.record(bundle / 'MANIFEST.json')
    source = inv.read_json(bundle / 'MANIFEST.json')['source_root_at_preparation']
    output = inv.reserve(output, (bundle, library, rscript, inv.SNAPSHOT, source,
                                  *(HERE / name for name in codes)))
    try:
        inv.create_json(output / 'REQUEST.json', {
            'case': spec['case'], 'bundle': str(bundle), 'bundle_manifest': bundle_record,
            'rscript': str(rscript), 'rscript_record': executable, 'library': str(library),
            'code_records': codes, 'parameters': spec['parameters'],
        })
        for row in files:
            if row['path'].startswith('adapter/'):
                inv.copy_new(bundle / row['path'], output / 'runtime' / row['path'], row)
        adapter = output / 'runtime/adapter'
        (adapter / '.Rlib').symlink_to(library, target_is_directory=True)
        env = os.environ.copy()
        env.pop('R_LIBS', None)
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
            env[name] = '1'
        # The caller chooses a space-free parent; only this owned temp directory
        # is cleaned. No hardcoded /private/tmp assumption in the entry point.
        with tempfile.TemporaryDirectory(prefix='ramp_cycombine_', dir=scratch_parent) as temp:
            env['TMPDIR'] = str(Path(temp).resolve())
            environment_command = [rscript, '--vanilla', HERE / 'check_environment.R', adapter,
                                   bundle / 'environment/validated_direct_runtime_packages.csv']
            correction_command = [rscript, '--vanilla', adapter / 'run_cycombine.R',
                '--input', bundle / 'inputs' / spec['input'],
                '--markers', bundle / 'inputs' / spec['marker_file'],
                '--config', bundle / 'inputs' / spec['config'], '--output', output / 'result']
            inv.create_json(output / 'COMMANDS.json', {'environment': list(map(str, environment_command)),
                                                      'correction': list(map(str, correction_command))})
            run_command(environment_command, output, 'environment_before', env)
            run_command(correction_command, output, 'correction', env)
            run_command(environment_command, output, 'environment_after', env)
        if (output / 'environment_before.stdout.log').read_bytes() != (output / 'environment_after.stdout.log').read_bytes():
            raise ValueError('R environment changed during fitting')
        check_output(output / 'result', bundle, spec)
        verify_bundle(bundle)
        inv.check(bundle / 'MANIFEST.json', bundle_record)
        inv.check(rscript, executable)
        for name, record in codes.items():
            inv.check(HERE / name, record)
        for row in files:
            if row['path'].startswith('adapter/'):
                inv.check(output / 'runtime' / row['path'], row)
        records = {p.relative_to(output).as_posix(): inv.record(p) for p in sorted(output.rglob('*'))
                   if p.is_file() and not p.is_symlink()}
        inv.create_json(output / 'RUN_MANIFEST.json', {'case': spec['case'], 'files': records,
            'scope': 'Real correction from unchanged analysis-ready input', 'history_used_by_fit': False})
        # Last operation that can affect completion: failures above preserve the
        # partial run, even if the nested R result has its own COMPLETE marker.
        inv.create_json(output / 'COMPLETE.json', {'status': 'complete',
                        'manifest_sha256': inv.sha(output / 'RUN_MANIFEST.json')})
    except BaseException as error:
        inv.create_json(output / 'FAILED.json', {'error': repr(error)})
        raise
    return output


def verify_run(run, bundle):
    run = Path(run).resolve(strict=True)
    bundle = Path(bundle).resolve(strict=True)
    spec, _ = verify_bundle(bundle)
    done = inv.read_json(run / 'COMPLETE.json')
    if done.get('status') != 'complete' or done.get('manifest_sha256') != inv.sha(run / 'RUN_MANIFEST.json'):
        raise ValueError('Run is not complete')
    manifest = inv.read_json(run / 'RUN_MANIFEST.json')
    if manifest['case'] != spec['case']:
        raise ValueError('Wrong case')
    request = inv.read_json(run / 'REQUEST.json')
    inv.check(bundle / 'MANIFEST.json', request['bundle_manifest'])
    for name, record in manifest['files'].items():
        inv.check(inv.within(run, name), record)
    expected = set(manifest['files']) | {'RUN_MANIFEST.json', 'COMPLETE.json'}
    actual = {p.relative_to(run).as_posix() for p in run.rglob('*') if p.is_file() and not p.is_symlink()}
    if actual != expected:
        raise ValueError('Run inventory differs')
    return check_output(run / 'result', bundle, spec)


def compare(run, bundle, source):
    """Read-only historical comparison, never called by fit()."""
    import numpy as np
    run, bundle, source = (Path(p).resolve(strict=True) for p in (run, bundle, source))
    summary = verify_run(run, bundle)
    spec, _ = verify_bundle(bundle)
    catalog = inv.verify_local()
    records = {r['source']: r for r in catalog['files']}
    old = Path(spec['output']).parent
    comparisons = {}
    for name in ('corrected_expression.csv.gz', 'corrected_expression.rds',
                 'cell_metadata_label_free.csv.gz', 'correction_som_assignments.csv.gz',
                 'correction_som_node_batch_counts.csv', 'correction_som_node_modes.csv',
                 'input_marker_batch_statistics.csv', 'cycombine_messages.log'):
        original = inv.within(source, old / name)
        inv.check(original, records[str(old / name)])
        new = run / 'result' / name
        exact = inv.sha(original) == inv.sha(new)
        row = {'bytes_equal': exact, 'original_sha256': inv.sha(original), 'new_sha256': inv.sha(new)}
        if name.endswith(('.csv', '.csv.gz')):
            a, b = inv.table(original), inv.table(new)
            row['decoded_table_equal'] = a.equals(b)
            if not row['decoded_table_equal']:
                raise ValueError('Historical table differs: ' + name)
            if name == 'corrected_expression.csv.gz':
                row['maximum_absolute_difference'] = float(np.max(np.abs(a[spec['markers']].to_numpy(float) - b[spec['markers']].to_numpy(float))))
        elif not exact:
            raise ValueError('Historical binary/log differs: ' + name)
        comparisons[name] = row
    original_manifest = inv.within(source, old / 'manifest.json')
    inv.check(original_manifest, records[str(old / 'manifest.json')])
    a, b = inv.read_json(original_manifest), inv.read_json(run / 'result/manifest.json')
    scientific = ('algorithm_parameters', 'markers', 'batch_levels', 'dimensions', 'fitting_input_columns',
                  'provenance_only_input_columns', 'ignored_input_columns', 'rng', 'som_audit',
                  'runtime_diagnostics', 'interpretation_limits', 'local_source_sha256')
    for key in scientific:
        if a[key] != b[key]:
            raise ValueError('Historical scientific manifest field differs: ' + key)
    return {**summary, 'comparisons': comparisons, 'scientific_manifest_fields_equal': list(scientific)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest='command', required=True)
    prep = subs.add_parser('prepare')
    prep.add_argument('--case', choices=['bone_marrow'], default='bone_marrow')
    prep.add_argument('--source-root', type=Path, required=True)
    prep.add_argument('--output', type=Path, required=True)
    for name in ('fit', 'verify', 'verify-run', 'compare'):
        parser = subs.add_parser(name)
        parser.add_argument('--bundle', type=Path, required=True)
        if name == 'fit':
            for flag in ('library', 'rscript', 'output', 'scratch-parent'):
                parser.add_argument('--' + flag, type=Path, required=True)
        if name in ('verify-run', 'compare'):
            parser.add_argument('--run', type=Path, required=True)
        if name == 'compare':
            parser.add_argument('--source-root', type=Path, required=True)
    a = p.parse_args(argv)
    if a.command == 'prepare': print(prepare(a.source_root, a.output, a.case))
    elif a.command == 'fit': print(fit(a.bundle, a.library, a.rscript, a.output, a.scratch_parent))
    elif a.command == 'verify': print({'case': verify_bundle(a.bundle)[0]['case'], 'status': 'passed'})
    elif a.command == 'verify-run': print(verify_run(a.run, a.bundle))
    else:
        import json
        print(json.dumps(compare(a.run, a.bundle, a.source_root), indent=2))


if __name__ == '__main__':
    main()
