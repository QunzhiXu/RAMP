"""Audit an existing real BM correction; does not fit or modify accepted results."""
import argparse
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
from validate import unchanged_prior_work

OLD_REPORTS = {
    'handoff_20261010': '56972dcf61c8f3632e7070557da9aca8c308a9b4d2667ae272e075db31a44af6',
    'snapshot_fix_20261010': '016c477932e276b7046c37e566829b2d091f8208ccf98a3b95afc1ca83161d12',
}


def old_work(source):
    for name, digest in OLD_REPORTS.items():
        if inv.sha(inv.HERE / 'validation' / name / 'report.json') != digest:
            raise ValueError('Prior cyCombine report changed')
    fix = inv.read_json(inv.HERE / 'validation/snapshot_fix_20261010/report.json')
    for name, digest in {**fix['current_files'], **fix['unchanged_old_files']}.items():
        if inv.sha(inv.HERE / name) != digest:
            raise ValueError('Accepted cyCombine file changed: ' + name)
    return {'old_cycombine_reports': OLD_REPORTS, 'accepted_files_unchanged': len(fix['current_files']) + len(fix['unchanged_old_files']),
            'prior_modules': unchanged_prior_work(source), 'old_correction_sources': inv.verify_sources(source)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'bundle', 'run', 'python', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    a = parser.parse_args(argv)
    source, bundle, run = (p.resolve(strict=True) for p in (a.source_root, a.bundle, a.run))
    if not a.output.resolve().is_relative_to(inv.HERE / 'validation'):
        parser.error('Choose a new directory under this module/validation')
    output = inv.reserve(a.output, (source, bundle, run, inv.SNAPSHOT))
    report = {'utc': datetime.now(timezone.utc).isoformat(), 'scope': 'BM correction execution and regression only',
              'cohort_fit_performed_by_validator': False, 'actual_fit_run': str(run),
              'provider_extraction_performed': False, 'clean_install_performed': False}
    try:
        report['prior_work'] = old_work(source)
        report['comparison'] = study.compare(run, bundle, source)
        spec, files = study.verify_bundle(bundle)
        report['bundle'] = {'path': str(bundle), 'payload_files': len(files),
                            'payload_bytes': sum(r['bytes'] for r in files), 'manifest_sha256': inv.sha(bundle / 'MANIFEST.json')}
        request = inv.read_json(run / 'REQUEST.json')
        for name, record in request['code_records'].items():
            inv.check(inv.HERE / name, record)
        report['run_manifest_sha256'] = inv.sha(run / 'RUN_MANIFEST.json')
        report['run_request'] = request
        report['actual_commands'] = inv.read_json(run / 'COMMANDS.json')
        report['environment'] = inv.read_json(run / 'environment_before.stdout.log')
        # The runner itself is unmodified. Check every runtime adapter file
        # against the frozen catalog, not a newly self-generated manifest.
        report['runtime_adapter_files'] = []
        for row in files:
            if row['path'].startswith('adapter/'):
                inv.check(run / 'runtime' / row['path'], row)
                report['runtime_adapter_files'].append(row)
        with tempfile.TemporaryDirectory(prefix='cycombine_bm_audit_') as name:
            tmp = Path(name).resolve()
            copied = tmp / 'module'
            shutil.copytree(inv.HERE, copied, ignore=shutil.ignore_patterns('runs', 'validation', '__pycache__', '.DS_Store'))
            env = os.environ.copy(); env.pop('PYTHONPATH', None); env['PYTHONDONTWRITEBYTECODE'] = '1'
            command = [str(a.python.absolute()), '-B', '-m', 'unittest', 'discover', '-s', str(copied / 'tests'), '-v']
            p = subprocess.run(command, cwd=tmp, env=env, capture_output=True, text=True)
            report['tests'] = {'command': command, 'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr,
                               'isolated_copy': True, 'default_TMPDIR_preserved': True}
            if p.returncode: raise RuntimeError('Tests failed')
            report['tests_passed'] = sum(map(int, re.findall(r'Ran (\d+) tests? in', p.stderr)))
            # Replay no correction: verify a moved byte-for-byte bundle and
            # existing result using the copied module, with no source-root arg.
            moved = tmp / 'moved_inputs'; shutil.copytree(bundle, moved)
            command = [str(a.python.absolute()), '-B', str(copied / 'study.py'), 'verify-run',
                       '--bundle', str(moved), '--run', str(run)]
            p = subprocess.run(command, cwd=tmp, env=env, capture_output=True, text=True)
            report['moved_input_verification'] = {'command': command, 'returncode': p.returncode,
                                                 'stdout': p.stdout, 'stderr': p.stderr, 'reran_R': False}
            if p.returncode: raise RuntimeError('Moved bundle verification failed')
        command = [str(a.python.absolute()), '-B', str(inv.STUDY / 'benchmarks/integration_inputs/inputs.py'),
                   'verify-staged', '--input-root', str(inv.STUDY / 'benchmarks/integration_inputs/runs/frozen_inputs_20261010')]
        p = subprocess.run(command, capture_output=True, text=True)
        report['common_inputs'] = {'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
        if p.returncode: raise RuntimeError('Accepted common inputs changed')
        module_file = inv.STUDY / 'benchmarks/bone_marrow/verify.py'
        module_spec = importlib.util.spec_from_file_location('bm_readonly_verify', module_file)
        module = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(module)
        report['bm_ramp_reference_checks'] = module.verify(inv.STUDY / 'benchmarks/bone_marrow/runs/rebuild_20261010')
        if not report['bm_ramp_reference_checks']['all_reference_checks_passed']:
            raise ValueError('Existing BM RAMP results changed')
        report['postcheck'] = old_work(source)
        study.verify_run(run, bundle)
        report['current_code_sha256'] = {name: inv.sha(inv.HERE / name) for name in
            ('study.py', 'check_environment.R', 'tests/test_study.py', 'validate_bm_execution.py',
             'STUDY_BM.md', 'AUDIT_BM_EXECUTION.md')}
        report['status'] = 'passed'
        inv.create_json(output / 'report.json', report)
    except BaseException as error:
        inv.create_json(output / 'FAILED.json', {**report, 'error': repr(error)})
        raise
    print(output / 'report.json')


if __name__ == '__main__':
    main()
