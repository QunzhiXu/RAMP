"""Record preflight / verify four saved Harmony fixed-arcsinh runs; never fit R."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import harmony_io as io
import inventory as inv
import study
from validate import BASELINE_SCRIPT, compare_baselines
from validate_bm_execution import scientific_comparison
from validate_locale import assert_locale_log
from validate_technical_execution import accepted_bm, changed_payload_hashes, existing_files, check_existing

TECHNICAL_REPORT_SHA = '5a82fbf124b734d4a329c343e942c0906fbf4756b12b165434655685b8747c75'
PROFILE = 'adtnorm_fixed_arcsinh'
CASES = {'bone_marrow': (88031, 14, 17), 'tnbc': (62701, 11, 11),
         'melanoma': (60000, 12, 46), 'lung_cancer': (60000, 8, 30)}
NEW_FILES = ('validate_sensitivity_execution.py', 'tests/test_sensitivity_execution.py',
             'STUDY_SENSITIVITY.md', 'AUDIT_SENSITIVITY_EXECUTION.md')


def branch(case):
    if case not in CASES: raise ValueError('Only four technical fixed-arcsinh branches belong to this round')
    spec, files = study.plan(case, PROFILE)
    if spec['kind'] != 'technical' or (spec['cells'], len(spec['markers']), spec['units']) != CASES[case]:
        raise ValueError('Frozen technical cohort differs')
    if spec['formula'] != 'ADTnorm official fixed transform: asinh(1 + x/5), applied element-wise.':
        raise ValueError('Unexpected historical input transform')
    return spec, files


def audit_transform(primary, fixed, markers):
    """Check the frozen formula in memory only; never produce fitting inputs."""
    import numpy as np
    columns = ['cell_id', 'dataset', 'unit_key', *markers]
    if list(primary.columns) != columns or list(fixed.columns) != columns or not primary.iloc[:, :3].equals(fixed.iloc[:, :3]):
        raise ValueError('Primary/fixed cell, unit or marker order differs')
    original, transformed = (frame[markers].to_numpy(float) for frame in (primary, fixed))
    if not np.isfinite(original).all() or not np.isfinite(transformed).all():
        raise ValueError('Non-finite frozen input')
    expected = np.arcsinh(1 + original / 5)
    if not np.array_equal(expected, transformed):
        raise ValueError('Frozen input is not exactly the single historical transform')
    return {'formula': 'asinh(1 + x/5)', 'maximum_absolute_difference': 0.0,
            'row_identity_equal': True, 'input_rewritten': False, 'algorithm_applies_transformation': False}


def accepted_primary(source, runs):
    bm = accepted_bm(source, runs)
    path = inv.HERE / 'validation/technical_execution_20261010/report.json'
    if io.sha(path) != TECHNICAL_REPORT_SHA: raise ValueError('Accepted technical report changed')
    report = io.read_json(path)
    for name, row in report['current_files'].items(): io.check(inv.HERE / name, row)
    checks = {}
    for case, old in report['cases'].items():
        bundle, run = runs / (case + '_source_inputs_20261010'), runs / (case + '_source_fit_20261010')
        if (io.sha(bundle / 'MANIFEST.json') != old['bundle']['manifest_sha256'] or
                io.sha(run / 'RUN_MANIFEST.json') != old['run']['manifest_sha256']):
            raise ValueError('Accepted primary bundle/run changed')
        check = study.compare(run, bundle, source)
        if check != old['comparison']: raise ValueError('Accepted primary result changed')
        checks[case] = check
    return {'bone_marrow': bm, 'technical_report_sha256': TECHNICAL_REPORT_SHA, 'technical': checks}


def check_case(case, source, runs):
    bundle, run = runs / (case + '_fixed_inputs_20261010'), runs / (case + '_fixed_fit_20261010')
    expected, files = branch(case); spec, actual = study.verify_bundle(bundle)
    if spec != expected or actual != files: raise ValueError('Wrong bundle branch')
    result = {'comparison': study.compare(run, bundle, source),
              'all_payload_comparison': scientific_comparison(source, bundle, run, spec)}
    result['normalized_payload_hash_pairs'] = changed_payload_hashes(result['all_payload_comparison'])
    request = io.read_json(run / 'REQUEST.json')
    if request['code_records'] != study.code_records(): raise ValueError('Fit entry code changed')
    if (request['case'], request['profile']) != (case, PROFILE): raise ValueError('Wrong fit case/profile')
    commands = io.read_json(run / 'COMMANDS.json')
    expected_command = [request['rscript'], '--vanilla', str(run / 'runtime/tree/algorithm/harmony/run_harmony_direct.R'),
        '--input', str(bundle / 'inputs' / spec['input']), '--markers', str(bundle / 'inputs' / spec['marker_file']),
        '--config', str(run / 'runtime_config.json'), '--output', str(run / 'result')]
    if commands['correction'] != expected_command: raise ValueError('Unexpected correction command')
    before, after = (run / ('environment_' + name + '.stdout.log') for name in ('before', 'after'))
    result['effective_locale'] = assert_locale_log(before); assert_locale_log(after)
    if before.read_bytes() != after.read_bytes(): raise ValueError('Environment changed during fit')
    primary_spec, _ = study.plan(case, 'source_scale')
    primary_bundle = runs / ('bm_source_inputs_20261010' if case == 'bone_marrow' else case + '_source_inputs_20261010')
    actual_primary, _ = study.verify_bundle(primary_bundle)
    if actual_primary != primary_spec: raise ValueError('Wrong primary reference bundle')
    fixed = io.table(bundle / 'inputs' / spec['input'])
    primary = io.table(primary_bundle / 'inputs' / primary_spec['input'])
    result['frozen_transform_check'] = audit_transform(primary, fixed, spec['markers'])
    result['frozen_transform_check']['primary_input'] = {
        'path': str(primary_bundle / 'inputs' / primary_spec['input']),
        **io.record(primary_bundle / 'inputs' / primary_spec['input'])}
    # Read evaluation identities only after fitting; labels never enter the runner.
    canonical = io.table(source / spec['canonical']['evaluation_metadata'])
    if not fixed.iloc[:, :3].equals(canonical[['cell_id', 'dataset', 'unit_key']]):
        raise ValueError('Canonical cell/unit identity differs')
    result['dataset_counts'] = fixed.groupby('dataset', sort=False).agg(
        cells=('cell_id', 'size'), units=('unit_key', 'nunique')).to_dict()
    result['request'], result['commands'] = request, commands
    result['runtime_copies'] = []
    for row in files:
        target = study.runtime_path(row, spec)
        if target is not None:
            io.check(run / target, row); result['runtime_copies'].append({'runtime_path': str(target), **row})
    result['bundle'] = {'path': str(bundle), 'manifest_sha256': io.sha(bundle / 'MANIFEST.json'),
                        'payload_files': len(files), 'payload_bytes': sum(r['bytes'] for r in files)}
    result['run'] = {'path': str(run), 'manifest_sha256': io.sha(run / 'RUN_MANIFEST.json'),
                     'completion_sha256': io.sha(run / 'COMPLETE.json')}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('preflight', 'verify'))
    for name in ('source-root', 'study-root', 'runs-root', 'python', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--preflight', type=Path); p.add_argument('--preflight-sha256')
    a = p.parse_args()
    if a.mode == 'verify' and (not a.preflight or not a.preflight_sha256):
        p.error('verify requires --preflight and its previously recorded --preflight-sha256')
    source, study_root, runs = (x.resolve(strict=True) for x in (a.source_root, a.study_root, a.runs_root))
    if not a.output.resolve().is_relative_to(inv.HERE / 'validation'):
        p.error('Use a new directory under this module/validation')
    old_files = existing_files() if a.mode == 'preflight' else None
    protected = (source, runs, inv.SNAPSHOT, *(inv.HERE / n for n in study.code_records()))
    if a.preflight: protected += (a.preflight,)
    output = io.reserve(a.output, protected)
    report = {'utc': datetime.now(timezone.utc).isoformat(), 'mode': a.mode,
              'scope': 'Four technical adtnorm_fixed_arcsinh correction branches only',
              'real_fit_performed_by_validator': False, 'provider_extraction': False,
              'preprocessing_recomputed_for_fit': False, 'UMAP_metrics_survival': False, 'clean_install': False}
    env = os.environ.copy(); env.pop('PYTHONPATH', None); env['PYTHONDONTWRITEBYTECODE'] = '1'

    def command(argv, key, cwd=output):
        argv = list(map(str, argv))
        with (output / (key + '.stdout.log')).open('x') as stdout, (output / (key + '.stderr.log')).open('x') as stderr:
            process = subprocess.run(argv, cwd=cwd, env=env, stdout=stdout, stderr=stderr)
        if process.returncode: raise RuntimeError(key + ' failed; see logs')
        print('PASS ' + key, flush=True)
        return {'command': argv, 'returncode': process.returncode,
                **{k: {'path': key + '.' + k + '.log', **io.record(output / (key + '.' + k + '.log'))}
                   for k in ('stdout', 'stderr')}}

    try:
        report['accepted_primary'] = accepted_primary(source, runs)
        report['code_records'] = study.code_records()
        if a.mode == 'preflight':
            report['existing_module_files'] = old_files
            for case in CASES:
                branch(case)
                for suffix in ('inputs', 'fit'):
                    target = runs / (case + '_fixed_' + suffix + '_20261010')
                    if target.exists() or target.is_symlink(): raise FileExistsError(target)
            report['disk_free_bytes'] = shutil.disk_usage(runs).free
        else:
            preflight = a.preflight.resolve(strict=True)
            if io.sha(preflight / 'report.json') != a.preflight_sha256: raise ValueError('Preflight report changed')
            before = io.read_json(preflight / 'report.json')
            if (before['status'], before['mode'], before['code_records']) != ('passed', 'preflight', study.code_records()):
                raise ValueError('Invalid preflight')
            old_files = before['existing_module_files']; check_existing(old_files)
            for channel in ('stdout', 'stderr'):
                row = before['prior_modules'][channel]; io.check(preflight / row['path'], row)
            report['preflight'] = {'path': str(preflight / 'report.json'), 'sha256': a.preflight_sha256}
            report['cases'] = {case: check_case(case, source, runs) for case in CASES}
            with tempfile.TemporaryDirectory(prefix='harmony_fixed_check_') as name:
                temp = Path(name).resolve(); module = temp / 'module'
                shutil.copytree(inv.HERE, module, ignore=shutil.ignore_patterns('runs', 'validation', '__pycache__', '.DS_Store'))
                report['tests'] = command([a.python.absolute(), '-B', '-m', 'unittest', 'discover', '-s', module / 'tests', '-v'], 'tests', temp)
                report['python_tests_passed'] = sum(map(int, re.findall(r'Ran (\d+) tests? in', (output / 'tests.stderr.log').read_text())))
                report['moved_bundle_checks'] = {}
                for case, item in report['cases'].items():
                    moved = temp / (case + ' moved bundle'); shutil.copytree(item['bundle']['path'], moved)
                    report['moved_bundle_checks'][case] = command([a.python.absolute(), '-B', module / 'study.py', 'verify-run',
                        '--bundle', moved, '--run', item['run']['path']], case + '_moved', temp)
        report['common_inputs'] = command([a.python.absolute(), '-B', study_root / 'benchmarks/integration_inputs/inputs.py',
            'verify-staged', '--input-root', study_root / 'benchmarks/integration_inputs/runs/frozen_inputs_20261010'], 'common_inputs')
        report['prior_modules'] = command([a.python.absolute(), '-B', '-c', BASELINE_SCRIPT, source, study_root / 'methods/cycombine'], 'prior_modules')
        if a.mode == 'verify':
            compare_baselines(io.read_json(preflight / before['prior_modules']['stdout']['path']), io.read_json(output / 'prior_modules.stdout.log'))
            report['prior_comparison_ignored_field'] = 'BM_RAMP.checked_at'
        report['source_verification'] = inv.verify_source(source)
        check_existing(old_files); accepted_primary(source, runs)
        report['unchanged_existing_files'] = len(old_files)
        report['current_files'] = {name: io.record(inv.HERE / name) for name in NEW_FILES if (inv.HERE / name).is_file()}
        report['status'] = 'passed'; io.create_json(output / 'report.json', report)
    except BaseException as error:
        io.create_json(output / 'FAILED.json', {**report, 'error': repr(error)})
        raise
    print(output / 'report.json')


if __name__ == '__main__': main()
