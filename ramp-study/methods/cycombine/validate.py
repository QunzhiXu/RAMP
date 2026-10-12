"""Create-only evidence for the cyCombine code/provenance migration, not cohort fitting."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import inventory as inv


def unchanged_prior_work(source):
    study = inv.STUDY
    reports = []
    for path, root in (
        (study / 'inventory_20261010/evidence/source_census.json', source.parent),
        (study / 'inventory_20261010/evidence/curated_sources.json', source.parent),
        (study / 'benchmarks/bone_marrow/validation/code_snapshot_output_safety_20261010.json', study),
    ):
        count = 0
        for item in inv.read_json(path)['files']:
            if item.get('sha256'):
                inv.check(inv.within(root, item['path']), item)
                count += 1
        reports.append({'file': str(path), 'sha256': inv.sha(path), 'checked': count})
    final = study / 'methods/maxfuse/providers/lung_story/validation/provider_handoff_20261010_r2/report.json'
    if inv.sha(final) != '33afef9b373160e2d5fc528b1c6f4cdcffb8900a94adf04af3f06948db6e47f8':
        raise ValueError('Accepted MaxFuse handoff changed')
    data = inv.read_json(final)
    for name, digest in data['current_code_sha256'].items():
        if inv.sha(study / name) != digest: raise ValueError('Prior MaxFuse/module document changed: ' + name)
    preserved = {'final_lung_report': inv.sha(final), 'current_files': len(data['current_code_sha256']), 'earlier_reports': []}
    for old in data['prior_providers_unchanged']:
        path = study / 'methods/maxfuse/providers' / old['case'] / 'validation/provider_handoff_20261010/report.json'
        if inv.sha(path) != old['report_sha256']: raise ValueError('Earlier report changed')
        preserved['earlier_reports'].append({'case': old['case'], 'sha256': inv.sha(path)})
    return {'indexed_original_and_bm_code': reports, 'latest_maxfuse_handoff': preserved}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('source-root', 'r-library', 'rscript', 'output'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True)
    args = parser.parse_args(argv)
    source = args.source_root.resolve(strict=True)
    library = args.r_library.resolve(strict=True)
    rscript = args.rscript.absolute()
    if not args.output.resolve().is_relative_to(inv.HERE / 'validation'):
        parser.error('Use a new directory inside this module/validation')
    output = inv.reserve(args.output, (source, library, inv.SNAPSHOT))
    report = {'utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'Frozen adapter and seven correction-stage source contracts; existing results and synthetic runtime tests only',
              'commands': [], 'real_cohort_fit_performed': False, 'clean_install_performed': False}
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        env[key] = '1'

    def command(argv, command_env=None):
        p = subprocess.run(list(map(str, argv)), env=command_env or env, cwd=output,
                           capture_output=True, text=True)
        row = {'command': list(map(str, argv)), 'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
        report['commands'].append(row)
        if p.returncode: raise RuntimeError('Validation command failed: ' + str(argv[0]))
        print('PASS', argv[-1], flush=True)
        return row

    try:
        report['snapshot'] = {'catalog_sha256': inv.sha(inv.SNAPSHOT / 'catalog.json'), **inv.verify_sources(source)}
        report['wrapper_equivalence'] = inv.verify_wrappers()
        report['historical_results'] = inv.verify_results(source)
        command([args.python.absolute(), '-B', '-m', 'unittest', 'discover', '-s', inv.HERE / 'tests', '-v'])
        command([args.python.absolute(), '-B', inv.STUDY / 'benchmarks/integration_inputs/inputs.py',
                 'verify-staged', '--input-root', inv.STUDY / 'benchmarks/integration_inputs/runs/frozen_inputs_20261010'])
        # Only disposable copies/synthetic fixtures go in this space-free temp dir.
        # No dependency installs; existing R library is read via an explicit symlink.
        with tempfile.TemporaryDirectory(prefix='ramp_cycombine_validate_', dir='/private/tmp') as name:
            temporary = Path(name).resolve()
            adapter = temporary / 'adapter'
            shutil.copytree(inv.SNAPSHOT / 'adapter', adapter)
            (adapter / '.Rlib').symlink_to(library, target_is_directory=True)
            renv = {**env, 'TMPDIR': str(temporary)}
            result = command([rscript, '--vanilla', inv.HERE / 'check_runtime.R', inv.SNAPSHOT, library], renv)
            lines = [s for s in result['stdout'].splitlines() if s.startswith('CYCOMBINE_AUDIT_JSON=')]
            if len(lines) != 1: raise ValueError('Missing runtime JSON')
            report['runtime'] = json.loads(lines[0].split('=', 1)[1])
            for leaf in ('smoke_test.R', 'cli_contract_test.R'):
                command([rscript, '--vanilla', adapter / 'tests' / leaf], renv)
            report['unchanged_original_smokes_run_in_temporary_copy'] = True
        report['prior_work'] = unchanged_prior_work(source)
        # Check original input/code identities again after all read-only checks.
        report['postcheck'] = inv.verify_sources(source)
        report['tests_passed'] = sum(int(n) for row in report['commands'] for n in re.findall(r'Ran (\d+) tests? in', row['stderr']))
        report['r_synthetic_scripts_passed'] = 3
        report['source_copy_bytes'] = sum(row['bytes'] for row in inv.verify_local()['files'] if 'copy' in row)
        report['current_code_sha256'] = {
            p.relative_to(inv.HERE).as_posix(): inv.sha(p)
            for p in sorted(inv.HERE.rglob('*'))
            if p.is_file() and not ({'frozen_20261010', 'validation', '__pycache__', 'runs'} & set(p.relative_to(inv.HERE).parts))
        }
        report['status'] = 'passed'
        inv.create_json(output / 'report.json', report)
    except BaseException as error:
        inv.create_json(output / 'FAILED.json', {**report, 'error': repr(error)})
        raise
    print(output / 'report.json')


if __name__ == '__main__': main()
