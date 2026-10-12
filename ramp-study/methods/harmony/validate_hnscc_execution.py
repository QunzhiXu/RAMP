"""Record HNSCC preflight / verify a saved Harmony story run; never fit R."""
import argparse
import copy
from datetime import datetime, timezone
import math
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
from validate_locale import assert_locale_log
from validate_sensitivity_execution import accepted_primary
from validate_technical_execution import existing_files, check_existing

SENSITIVITY_REPORT_SHA = '3e0c5ac59c07fbfa972d12af56d0232fb8b564a13699f478d8f89564382c16ac'
CASE = 'hnscc_story'
PROFILE = 'source_scale'
STEM = 'hnscc_story_source'
CORE = ('corrected_expression.csv.gz', 'fit.rds', 'cell_metadata.csv.gz',
        'marker_rank_audit.csv.gz', 'marker_rank_audit_by_sample.csv.gz',
        'marker_value_shift_audit.csv.gz', 'values.npy')
EXCEPTIONS = ('R_COMPLETE.json', 'COMPLETE.json', 'sessionInfo.txt')
NEW_FILES = ('validate_hnscc_execution.py', 'tests/test_hnscc_execution.py',
             'STUDY_HNSCC.md', 'AUDIT_HNSCC_EXECUTION.md')


def branch(case=CASE):
    if case != CASE: raise ValueError('Only HNSCC story belongs to this round')
    spec, files = study.plan(case, PROFILE)
    if (spec['kind'], spec['cells'], len(spec['markers']), spec['units']) != ('story', 94488, 22, 77):
        raise ValueError('Frozen HNSCC cohort differs')
    if spec['input'] != 'figure_HNSCC/Comparison/input/raw.csv.gz':
        raise ValueError('HNSCC Harmony must use frozen raw.csv.gz, not analysis_ready')
    return spec, files


def normalize_elapsed(value, nested=False):
    """Permit elapsed time alone, without mutating the comparison evidence."""
    result = copy.deepcopy(value)
    runtime = result['runtime'] if nested else result
    elapsed = runtime['elapsed_seconds']
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError('Invalid elapsed time')
    runtime['elapsed_seconds'] = '<recorded elapsed seconds>'
    return result


def compare_completion(old, new, nested=False):
    if normalize_elapsed(old, nested) != normalize_elapsed(new, nested):
        raise ValueError('Completion metadata differs beyond elapsed time')
    return {'historical_elapsed_seconds': (old['runtime'] if nested else old)['elapsed_seconds'],
            'current_elapsed_seconds': (new['runtime'] if nested else new)['elapsed_seconds'],
            'only_allowed_difference': 'runtime.elapsed_seconds' if nested else 'elapsed_seconds'}


def compare_sessions(old, new):
    normalized, os_lines = [], []
    for text in (old, new):
        lines = text.splitlines()
        indices = [i for i, line in enumerate(lines) if line.startswith('Running under: ')]
        if len(indices) != 1 or not lines[indices[0]].startswith('Running under: macOS '):
            raise ValueError('Unexpected sessionInfo operating system entry')
        os_lines.append(lines[indices[0]]); lines[indices[0]] = '<recorded macOS release>'
        normalized.append(lines)
    if normalized[0] != normalized[1]: raise ValueError('sessionInfo differs beyond macOS release')
    return {'historical_OS': os_lines[0], 'current_OS': os_lines[1], 'equal_except_Running_under_line': True}


def story_comparison(source, run, spec):
    import numpy as np
    records = {r['source']: r for r in inv.verify_local()['files']}
    old_directory = Path(spec['output']).parent
    result = run / study.result_path(spec)
    if {p.name for p in result.iterdir()} != set(CORE + EXCEPTIONS):
        raise ValueError('Unexpected story result inventory')

    def original(name):
        relative = (old_directory / name).as_posix(); path = io.within(source, relative)
        io.check(path, records[relative]); return path

    payloads = {name: {'historical': io.record(original(name)), 'current': io.record(result / name),
                       'bytes_equal': original(name).read_bytes() == (result / name).read_bytes()}
                for name in CORE + EXCEPTIONS}
    if any(not payloads[name]['bytes_equal'] for name in CORE):
        raise ValueError('Non-identical core HNSCC result')
    completions = {name: compare_completion(io.read_json(original(name)), io.read_json(result / name),
                                            nested=name == 'COMPLETE.json')
                   for name in ('R_COMPLETE.json', 'COMPLETE.json')}
    a, b = (np.load(path, allow_pickle=False) for path in (original('values.npy'), result / 'values.npy'))
    if a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b):
        raise ValueError('Historical values.npy differs')
    return {'payloads': payloads, 'byte_identical_payloads': sum(r['bytes_equal'] for r in payloads.values()),
            'total_payloads': len(payloads), 'core_byte_identical': len(CORE), 'completion_comparisons': completions,
            'npy_maximum_absolute_difference': float(np.max(np.abs(a - b))),
            'session_info': compare_sessions(original('sessionInfo.txt').read_text(), (result / 'sessionInfo.txt').read_text()),
            'differing_payload_hash_pairs': {name: {'historical_sha256': row['historical']['sha256'],
                'current_sha256': row['current']['sha256']} for name, row in payloads.items() if not row['bytes_equal']}}


def accepted_technical(source, runs):
    primary = accepted_primary(source, runs)
    path = inv.HERE / 'validation/sensitivity_execution_20261010/report.json'
    if io.sha(path) != SENSITIVITY_REPORT_SHA: raise ValueError('Accepted sensitivity report changed')
    report = io.read_json(path)
    for name, row in report['current_files'].items(): io.check(inv.HERE / name, row)
    checks = {}
    for case, old in report['cases'].items():
        bundle, run = runs / (case + '_fixed_inputs_20261010'), runs / (case + '_fixed_fit_20261010')
        if (io.sha(bundle / 'MANIFEST.json') != old['bundle']['manifest_sha256'] or
                io.sha(run / 'RUN_MANIFEST.json') != old['run']['manifest_sha256']):
            raise ValueError('Accepted sensitivity bundle/run changed')
        check = study.compare(run, bundle, source)
        if check != old['comparison']: raise ValueError('Accepted sensitivity result changed')
        checks[case] = check
    return {'primary': primary, 'sensitivity_report_sha256': SENSITIVITY_REPORT_SHA, 'sensitivity': checks}


def check_case(source, runs):
    bundle, run = (runs / (STEM + '_' + suffix + '_20261010') for suffix in ('inputs', 'fit'))
    expected, files = branch(); spec, actual = study.verify_bundle(bundle)
    if spec != expected or actual != files: raise ValueError('Wrong bundle branch')
    result = {'comparison': study.compare(run, bundle, source),
              'all_payload_comparison': story_comparison(source, run, spec)}
    request = io.read_json(run / 'REQUEST.json')
    if request['code_records'] != study.code_records(): raise ValueError('Fit entry code changed')
    if (request['case'], request['profile']) != (CASE, PROFILE): raise ValueError('Wrong fit case/profile')
    commands = io.read_json(run / 'COMMANDS.json')
    here = run / 'runtime/tree' / spec['base']
    if commands['correction'] != [request['rscript'], '--vanilla', str(here / 'scripts/run_r.R'), str(here), 'harmony']:
        raise ValueError('Unexpected correction command')
    before, after = (run / ('environment_' + name + '.stdout.log') for name in ('before', 'after'))
    result['effective_locale'] = assert_locale_log(before); assert_locale_log(after)
    if before.read_bytes() != after.read_bytes(): raise ValueError('Environment changed during fit')
    inp = io.table(bundle / 'inputs' / spec['input'])
    # Evaluation labels are read by this post-fit verifier only, never by fit().
    canonical = io.table(source / spec['canonical']['evaluation_metadata'])
    if not inp.iloc[:, :3].equals(canonical[['cell_id', 'dataset', 'unit_key']]):
        raise ValueError('Canonical cell/unit identity differs')
    result['dataset_counts'] = inp.groupby('dataset', sort=False).agg(
        cells=('cell_id', 'size'), units=('unit_key', 'nunique')).to_dict()
    result['input_ranges'] = {dataset: {'minimum': float(part[spec['markers']].min().min()),
                                       'maximum': float(part[spec['markers']].max().max())}
                              for dataset, part in inp.groupby('dataset', sort=False)}
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
              'scope': 'HNSCC story source_scale Harmony correction only',
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
        report['accepted_technical'] = accepted_technical(source, runs)
        report['code_records'] = study.code_records()
        if a.mode == 'preflight':
            report['existing_module_files'] = old_files; branch()
            for suffix in ('inputs', 'fit'):
                target = runs / (STEM + '_' + suffix + '_20261010')
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
            report['case'] = check_case(source, runs)
            with tempfile.TemporaryDirectory(prefix='harmony_hnscc_check_') as name:
                temp = Path(name).resolve(); module = temp / 'module'
                shutil.copytree(inv.HERE, module, ignore=shutil.ignore_patterns('runs', 'validation', '__pycache__', '.DS_Store'))
                report['tests'] = command([a.python.absolute(), '-B', '-m', 'unittest', 'discover', '-s', module / 'tests', '-v'], 'tests', temp)
                report['python_tests_passed'] = sum(map(int, re.findall(r'Ran (\d+) tests? in', (output / 'tests.stderr.log').read_text())))
                moved = temp / 'HNSCC moved bundle'; shutil.copytree(report['case']['bundle']['path'], moved)
                report['moved_bundle_check'] = command([a.python.absolute(), '-B', module / 'study.py', 'verify-run',
                    '--bundle', moved, '--run', report['case']['run']['path']], 'moved_bundle', temp)
        report['common_inputs'] = command([a.python.absolute(), '-B', study_root / 'benchmarks/integration_inputs/inputs.py',
            'verify-staged', '--input-root', study_root / 'benchmarks/integration_inputs/runs/frozen_inputs_20261010'], 'common_inputs')
        report['prior_modules'] = command([a.python.absolute(), '-B', '-c', BASELINE_SCRIPT, source, study_root / 'methods/cycombine'], 'prior_modules')
        if a.mode == 'verify':
            compare_baselines(io.read_json(preflight / before['prior_modules']['stdout']['path']), io.read_json(output / 'prior_modules.stdout.log'))
            report['prior_comparison_ignored_field'] = 'BM_RAMP.checked_at'
        report['source_verification'] = inv.verify_source(source)
        check_existing(old_files); accepted_technical(source, runs)
        report['unchanged_existing_files'] = len(old_files)
        report['current_files'] = {name: io.record(inv.HERE / name) for name in NEW_FILES if (inv.HERE / name).is_file()}
        report['status'] = 'passed'; io.create_json(output / 'report.json', report)
    except BaseException as error:
        io.create_json(output / 'FAILED.json', {**report, 'error': repr(error)})
        raise
    print(output / 'report.json')


if __name__ == '__main__': main()
