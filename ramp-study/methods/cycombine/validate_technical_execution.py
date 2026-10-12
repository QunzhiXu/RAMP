"""Create-only evidence for three technical cyCombine reruns; no fitting here."""
import argparse
import ast
from datetime import datetime, timezone
import importlib.util
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import inventory as inv
import study
from validate_bm_execution import old_work

BM_REPORT_SHA = '575086c410a69209ab840bb358a515d12ffa792ddfff4de4f0d4d7a87450efe9'
CASES = ('tnbc', 'melanoma', 'lung_cancer')


def bm_baseline(source):
    path = inv.HERE / 'validation/bm_execution_20261010/report.json'
    if inv.sha(path) != BM_REPORT_SHA:
        raise ValueError('Accepted BM report changed')
    report = inv.read_json(path)
    saved = inv.HERE / 'provenance/bm_execution_20261010'
    for name, oldname in (('study.py', 'study.py'), ('tests/test_study.py', 'test_study.py')):
        if inv.sha(saved / oldname) != report['current_code_sha256'][name]:
            raise ValueError('Pre-extension source does not match accepted BM report')
    old = (saved / 'study.py').read_text()
    expected = old.replace(
        '"""Create-only BM correction from frozen method-ready inputs; no preprocessing."""',
        '"""Create-only technical benchmark correction from frozen method-ready inputs."""',
    ).replace("# BM's historical config/manifest are already relative.",
              '# Technical config/manifests are already relative.').replace(
        "if case != 'bone_marrow':\n        raise ValueError('Only bone_marrow execution is enabled in this audit round')",
        "if case not in inv.TECHNICAL:\n        raise ValueError('Only the four technical benchmarks are enabled in this audit round')",
    ).replace("choices=['bone_marrow'], default='bone_marrow'", "choices=inv.TECHNICAL, default='bone_marrow'")
    if (inv.HERE / 'study.py').read_text() != expected:
        raise ValueError('More than the declared routing/doc changes in study.py')
    expected_test = (saved / 'test_study.py').read_text().replace("study.plan('tnbc')", "study.plan('hnscc_story')")
    if (inv.HERE / 'tests/test_study.py').read_text() != expected_test:
        raise ValueError('Unexpected edit of previously accepted tests')
    unchanged = []
    old_functions = {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(old).body if isinstance(node, ast.FunctionDef)}
    new_functions = {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(expected).body if isinstance(node, ast.FunctionDef)}
    for name in old_functions:
        if name not in ('plan', 'main'):
            if old_functions[name] != new_functions[name]: raise ValueError('Non-routing function changed')
            unchanged.append(name)
    for name, digest in report['current_code_sha256'].items():
        if name not in ('study.py', 'tests/test_study.py') and inv.sha(inv.HERE / name) != digest:
            raise ValueError('Accepted BM code/doc changed: ' + name)
    bundle = inv.HERE / 'runs/bm_inputs_20261010'
    run = inv.HERE / 'runs/bm_correction_20261010'
    if (inv.sha(bundle / 'MANIFEST.json') != report['bundle']['manifest_sha256']
            or inv.sha(run / 'RUN_MANIFEST.json') != report['run_manifest_sha256']):
        raise ValueError('Accepted BM run changed')
    comparison = study.compare(run, bundle, source)
    return {'report_sha256': BM_REPORT_SHA, 'unchanged_function_AST': unchanged,
            'only_function_edits': ['plan: case guard', 'main: CLI choices'],
            'bundle_manifest_sha256': inv.sha(bundle / 'MANIFEST.json'),
            'run_manifest_sha256': inv.sha(run / 'RUN_MANIFEST.json'), 'old_BM_comparison': comparison}


def command(argv, cwd, env):
    p = subprocess.run(list(map(str, argv)), cwd=cwd, env=env, capture_output=True, text=True)
    row = {'command': list(map(str, argv)), 'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
    if p.returncode: raise RuntimeError(str(row))
    return row


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'python', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args(argv)
    source = a.source_root.resolve(strict=True)
    if not a.output.resolve().is_relative_to(inv.HERE / 'validation'):
        p.error('Use a new directory inside this module/validation')
    output = inv.reserve(a.output, (source, inv.SNAPSHOT, inv.HERE / 'runs', inv.HERE / 'provenance'))
    report = {'utc': datetime.now(timezone.utc).isoformat(), 'scope': 'Three technical correction reruns from frozen method-ready inputs',
              'R_fitting_by_validator': False, 'provider_extraction': False, 'clean_install': False, 'cases': []}
    env = os.environ.copy(); env.pop('PYTHONPATH', None); env['PYTHONDONTWRITEBYTECODE'] = '1'
    try:
        report['prior_work'] = old_work(source)
        report['BM_baseline'] = bm_baseline(source)
        for case in CASES:
            bundle = inv.HERE / 'runs' / (case + '_inputs_20261010')
            run = inv.HERE / 'runs' / (case + '_correction_20261010')
            spec, files = study.verify_bundle(bundle)
            comparison = study.compare(run, bundle, source)
            request = inv.read_json(run / 'REQUEST.json')
            for name, record in request['code_records'].items(): inv.check(inv.HERE / name, record)
            for row in files:
                if row['path'].startswith('adapter/'):
                    inv.check(run / 'runtime' / row['path'], row)
            row = {'case': case, 'bundle': str(bundle), 'run': str(run),
                   'payload_files': len(files), 'payload_bytes': sum(f['bytes'] for f in files),
                   'bundle_manifest_sha256': inv.sha(bundle / 'MANIFEST.json'),
                   'run_manifest_sha256': inv.sha(run / 'RUN_MANIFEST.json'),
                   'comparison': comparison, 'request': request,
                   'commands': inv.read_json(run / 'COMMANDS.json'),
                   'environment': inv.read_json(run / 'environment_before.stdout.log')}
            # Preprocessing is a copied historical contract, not recomputed here.
            row['preprocessing'] = spec['preprocessing']
            report['cases'].append(row)
            print('PASS historical comparison:', case, flush=True)
        with tempfile.TemporaryDirectory(prefix='cycombine_technical_audit_') as name:
            tmp = Path(name).resolve()
            copied = tmp / 'module'
            shutil.copytree(inv.HERE, copied, ignore=shutil.ignore_patterns('runs', 'validation', '__pycache__', '.DS_Store'))
            report['tests'] = command([a.python.absolute(), '-B', '-m', 'unittest', 'discover', '-s', copied / 'tests', '-v'], tmp, env)
            report['tests_passed'] = sum(map(int, re.findall(r'Ran (\d+) tests? in', report['tests']['stderr'])))
            report['moved_verifications'] = []
            for row in report['cases']:
                moved = tmp / row['case']; shutil.copytree(row['bundle'], moved)
                checked = command([a.python.absolute(), '-B', copied / 'study.py', 'verify-run',
                                   '--bundle', moved, '--run', row['run']], tmp, env)
                report['moved_verifications'].append({'case': row['case'], 'reran_R': False, **checked})
        report['common_inputs'] = command([a.python.absolute(), '-B', inv.STUDY / 'benchmarks/integration_inputs/inputs.py',
            'verify-staged', '--input-root', inv.STUDY / 'benchmarks/integration_inputs/runs/frozen_inputs_20261010'], output, env)
        module_file = inv.STUDY / 'benchmarks/bone_marrow/verify.py'
        module_spec = importlib.util.spec_from_file_location('bm_readonly_verify', module_file)
        module = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(module)
        report['BM_RAMP_result_checks'] = module.verify(inv.STUDY / 'benchmarks/bone_marrow/runs/rebuild_20261010')
        if not report['BM_RAMP_result_checks']['all_reference_checks_passed']: raise ValueError('BM RAMP results changed')
        report['postcheck'] = old_work(source)
        bm_baseline(source)
        for row in report['cases']: study.verify_run(row['run'], row['bundle'])
        report['current_code_sha256'] = {name: inv.sha(inv.HERE / name) for name in (
            'study.py', 'tests/test_study.py', 'tests/test_technical.py', 'validate_technical_execution.py',
            'STUDY_TECHNICAL.md', 'AUDIT_TECHNICAL_EXECUTION.md',
            'provenance/bm_execution_20261010/study.py', 'provenance/bm_execution_20261010/test_study.py')}
        report['status'] = 'passed'
        inv.create_json(output / 'report.json', report)
    except BaseException as error:
        inv.create_json(output / 'FAILED.json', {**report, 'error': repr(error)})
        raise
    print(output / 'report.json')


if __name__ == '__main__': main()
