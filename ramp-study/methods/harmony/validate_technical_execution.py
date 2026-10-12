"""Record preflight / validate three existing Harmony source-scale runs; never fit."""
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
from validate_bm_execution import accepted_code, scientific_comparison
from validate_locale import assert_locale_log

BM_REPORT_SHA = 'b409770427d0ee613f7b09ea776f15dcd0514d1418966cf5c90e09c5533e27f4'
CASES = {'tnbc': (62701, 11, 11), 'melanoma': (60000, 12, 46), 'lung_cancer': (60000, 8, 30)}
NEW_FILES = ('validate_technical_execution.py', 'tests/test_technical_execution.py',
             'STUDY_TECHNICAL.md', 'AUDIT_TECHNICAL_EXECUTION.md')


def branch(case):
    if case not in CASES:
        raise ValueError('Only the three technical source_scale branches belong to this round')
    spec, files = study.plan(case, 'source_scale')
    if spec['kind'] != 'technical' or (spec['cells'], len(spec['markers']), spec['units']) != CASES[case]:
        raise ValueError('Frozen technical cohort differs')
    return spec, files


def changed_payload_hashes(comparison):
    """List only the explicit, independently checked provenance-file exceptions."""
    rows = {name: {'historical_sha256': row['historical']['sha256'],
                   'current_sha256': row['current']['sha256']}
            for name, row in comparison['payloads'].items() if not row['bytes_equal']}
    if not set(rows) <= {'resolved_config.json', 'sessionInfo.txt'}:
        raise ValueError('Unexpected differing R payload')
    if any(r['historical_sha256'] == r['current_sha256'] for r in rows.values()):
        raise ValueError('Inconsistent payload difference record')
    return rows


def existing_files():
    return {p.relative_to(inv.HERE).as_posix(): io.record(p) for p in sorted(inv.HERE.rglob('*'))
            if p.is_file() and not p.is_symlink() and p.name != '.DS_Store'
            and '__pycache__' not in p.relative_to(inv.HERE).parts}


def check_existing(records):
    for name, expected in records.items():
        io.check(io.within(inv.HERE, name), expected)


def accepted_bm(source, runs):
    accepted_code()
    path = inv.HERE / 'validation/bm_execution_20261010/report.json'
    if io.sha(path) != BM_REPORT_SHA:
        raise ValueError('Accepted BM report changed')
    report = io.read_json(path)
    for name, expected in report['current_files'].items(): io.check(inv.HERE / name, expected)
    bundle, run = runs / 'bm_source_inputs_20261010', runs / 'bm_source_fit_20261010'
    if (io.sha(bundle / 'MANIFEST.json') != report['bundle']['manifest_sha256'] or
            io.sha(run / 'RUN_MANIFEST.json') != report['run']['manifest_sha256']):
        raise ValueError('Accepted BM bundle/run changed')
    result = study.compare(run, bundle, source)
    if result != report['comparison']: raise ValueError('Accepted BM comparison changed')
    return {'report_sha256': BM_REPORT_SHA, 'comparison': result}


def check_case(case, source, bundle, run):
    expected, files = branch(case)
    spec, actual = study.verify_bundle(bundle)
    if spec != expected or files != actual: raise ValueError('Wrong bundle branch')
    result = {'comparison': study.compare(run, bundle, source),
              'all_payload_comparison': scientific_comparison(source, bundle, run, spec)}
    result['normalized_payload_hash_pairs'] = changed_payload_hashes(result['all_payload_comparison'])
    request = io.read_json(run / 'REQUEST.json')
    if request['code_records'] != study.code_records(): raise ValueError('Fit entry code changed')
    if (request['case'], request['profile']) != (case, 'source_scale'): raise ValueError('Wrong fit case/profile')
    commands = io.read_json(run / 'COMMANDS.json')
    expected_command = [request['rscript'], '--vanilla', str(run / 'runtime/tree/algorithm/harmony/run_harmony_direct.R'),
                        '--input', str(bundle / 'inputs' / spec['input']), '--markers', str(bundle / 'inputs' / spec['marker_file']),
                        '--config', str(run / 'runtime_config.json'), '--output', str(run / 'result')]
    if commands['correction'] != expected_command: raise ValueError('Correction did not use expected bundle/runtime')
    before, after = (run / ('environment_' + name + '.stdout.log') for name in ('before', 'after'))
    result['effective_locale'] = assert_locale_log(before)
    assert_locale_log(after)
    if before.read_bytes() != after.read_bytes(): raise ValueError('Environment changed during fit')
    # Evaluation labels are read only here, after fitting, solely to check row identity.
    canonical = io.table(source / spec['canonical']['evaluation_metadata'])
    inp = io.table(bundle / 'inputs' / spec['input'])
    if not inp.iloc[:, :3].equals(canonical[['cell_id', 'dataset', 'unit_key']]):
        raise ValueError('Canonical cohort identity differs')
    result['dataset_counts'] = inp.groupby('dataset', sort=False).agg(
        cells=('cell_id', 'size'), units=('unit_key', 'nunique')).to_dict()
    result['request'], result['commands'] = request, commands
    result['runtime_copies'] = []
    for row in files:
        target = study.runtime_path(row, spec)
        if target is not None:
            io.check(run / target, row)
            result['runtime_copies'].append({'runtime_path': str(target), **row})
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
    p.add_argument('--preflight', type=Path)
    p.add_argument('--preflight-sha256')
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
              'scope': 'TNBC, melanoma and lung technical source_scale correction only',
              'real_fit_performed_by_validator': False, 'provider_extraction': False,
              'preprocessing_recomputed': False, 'UMAP_metrics_survival': False, 'clean_install': False}
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
        report['accepted_bm'] = accepted_bm(source, runs)
        report['code_records'] = study.code_records()
        if a.mode == 'preflight':
            report['existing_module_files'] = old_files
            for case in CASES:
                branch(case)
                for suffix in ('inputs', 'fit'):
                    target = runs / (case + '_source_' + suffix + '_20261010')
                    if target.exists() or target.is_symlink(): raise FileExistsError(target)
            report['disk_free_bytes'] = shutil.disk_usage(runs).free
        else:
            preflight = a.preflight.resolve(strict=True)
            if io.sha(preflight / 'report.json') != a.preflight_sha256:
                raise ValueError('Preflight report changed')
            before = io.read_json(preflight / 'report.json')
            if (before['status'], before['mode'], before['code_records']) != ('passed', 'preflight', study.code_records()):
                raise ValueError('Invalid preflight')
            old_files = before['existing_module_files']; check_existing(old_files)
            for channel in ('stdout', 'stderr'):
                row = before['prior_modules'][channel]; io.check(preflight / row['path'], row)
            report['preflight'] = {'path': str(preflight / 'report.json'), 'sha256': a.preflight_sha256}
            report['cases'] = {case: check_case(case, source, runs / (case + '_source_inputs_20261010'),
                                               runs / (case + '_source_fit_20261010')) for case in CASES}
            with tempfile.TemporaryDirectory(prefix='harmony_technical_check_') as name:
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
        check_existing(old_files)
        accepted_bm(source, runs)
        report['unchanged_existing_files'] = len(old_files)
        report['current_files'] = {name: io.record(inv.HERE / name) for name in NEW_FILES if (inv.HERE / name).is_file()}
        report['status'] = 'passed'
        io.create_json(output / 'report.json', report)
    except BaseException as error:
        io.create_json(output / 'FAILED.json', {**report, 'error': repr(error)})
        raise
    print(output / 'report.json')


if __name__ == '__main__': main()
