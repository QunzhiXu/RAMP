"""Record lung preflight / verify a saved Harmony story run; never fit R."""
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
from validate_locale import assert_locale_log
from validate_hnscc_execution import story_comparison as shared_story_comparison
from validate_melanoma_execution import accepted_previous as accepted_before_melanoma, check_case as check_accepted_melanoma
from validate_technical_execution import existing_files, check_existing

MELANOMA_REPORT_SHA = '8d0383fa9c24b556062e513e4ae908a0f45b81275f23988b57d3cf8136ee804f'
CASE = 'lung_story'
PROFILE = 'source_scale'
STEM = 'lung_story_source'
BASE = 'story_lung_HLA_ABC/Comparison_senosain140k_20260921'
NEW_FILES = ('validate_lung_execution.py', 'tests/test_lung_execution.py',
             'STUDY_LUNG.md', 'AUDIT_LUNG_EXECUTION.md')


def branch(case=CASE):
    if case != CASE: raise ValueError('Only lung story belongs to this round')
    spec, files = study.plan(case, PROFILE)
    if (spec['kind'], spec['cells'], len(spec['markers']), spec['units']) != ('story', 84000, 9, 84):
        raise ValueError('Frozen lung story cohort differs')
    if spec['base'] != BASE or spec['input'] != BASE + '/input/raw.csv.gz':
        raise ValueError('Lung story must use current frozen raw.csv.gz, not analysis_ready or earlier Comparison')
    if spec['datasets'] != ['Senosain CyTOF', 'Cords IMC', 'Zheng mIF']:
        raise ValueError('Wrong lung story platforms/order; Monkman belongs to the technical benchmark')
    return spec, files


def story_comparison(source, run, spec):
    """Retain the accepted comparator, correcting only its legacy error label."""
    try:
        return shared_story_comparison(source, run, spec)
    except ValueError as error:
        if str(error) == 'Non-identical core HNSCC result':
            raise ValueError('Non-identical core ' + spec['case'] + ' result') from error
        raise


def accepted_previous(source, runs):
    prior = accepted_before_melanoma(source, runs)
    path = inv.HERE / 'validation/melanoma_execution_20261010/report.json'
    if io.sha(path) != MELANOMA_REPORT_SHA: raise ValueError('Accepted melanoma report changed')
    report = io.read_json(path)
    for name, row in report['current_files'].items(): io.check(inv.HERE / name, row)
    check = check_accepted_melanoma(source, runs)
    if check != report['case']: raise ValueError('Accepted melanoma bundle/run/evidence changed')
    return {'previous': prior, 'melanoma_report_sha256': MELANOMA_REPORT_SHA, 'melanoma': check}


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
              'scope': 'Lung story source_scale Harmony correction only',
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
        report['accepted_previous'] = accepted_previous(source, runs)
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
            with tempfile.TemporaryDirectory(prefix='harmony_lung_check_') as name:
                temp = Path(name).resolve(); module = temp / 'module'
                shutil.copytree(inv.HERE, module, ignore=shutil.ignore_patterns('runs', 'validation', '__pycache__', '.DS_Store'))
                report['tests'] = command([a.python.absolute(), '-B', '-m', 'unittest', 'discover', '-s', module / 'tests', '-v'], 'tests', temp)
                report['python_tests_passed'] = sum(map(int, re.findall(r'Ran (\d+) tests? in', (output / 'tests.stderr.log').read_text())))
                moved = temp / 'Lung moved bundle'; shutil.copytree(report['case']['bundle']['path'], moved)
                report['moved_bundle_check'] = command([a.python.absolute(), '-B', module / 'study.py', 'verify-run',
                    '--bundle', moved, '--run', report['case']['run']['path']], 'moved_bundle', temp)
        report['common_inputs'] = command([a.python.absolute(), '-B', study_root / 'benchmarks/integration_inputs/inputs.py',
            'verify-staged', '--input-root', study_root / 'benchmarks/integration_inputs/runs/frozen_inputs_20261010'], 'common_inputs')
        report['prior_modules'] = command([a.python.absolute(), '-B', '-c', BASELINE_SCRIPT, source, study_root / 'methods/cycombine'], 'prior_modules')
        if a.mode == 'verify':
            compare_baselines(io.read_json(preflight / before['prior_modules']['stdout']['path']), io.read_json(output / 'prior_modules.stdout.log'))
            report['prior_comparison_ignored_field'] = 'BM_RAMP.checked_at'
        report['source_verification'] = inv.verify_source(source)
        check_existing(old_files); accepted_previous(source, runs)
        report['unchanged_existing_files'] = len(old_files)
        report['current_files'] = {name: io.record(inv.HERE / name) for name in NEW_FILES if (inv.HERE / name).is_file()}
        report['status'] = 'passed'; io.create_json(output / 'report.json', report)
    except BaseException as error:
        io.create_json(output / 'FAILED.json', {**report, 'error': repr(error)})
        raise
    print(output / 'report.json')


if __name__ == '__main__': main()


