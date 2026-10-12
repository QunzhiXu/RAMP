"""Create new evidence for phase 1; prepares real inputs but fits synthetic data only.

By default reuses this round's saved R evidence. --rerun-synthetic reruns only
the small generated tests in temporary directories, never a real cohort.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import harmony_io as io
import inventory as inv
import study

# This child process keeps the accepted cyCombine module names separate from
# this module's inventory/study imports. All called old functions are read-only.
BASELINE_SCRIPT = '''
import sys,json,importlib.util
from pathlib import Path
source=Path(sys.argv[1]); code=Path(sys.argv[2]); sys.path.insert(0,str(code))
import inventory as inv
from validate_bm_execution import old_work
from validate_hnscc_execution import prior_technical
from validate_melanoma_execution import prior_hnscc
from validate_lung_execution import prior_melanoma
import lung
anchors={
 'inventory_20261010/evidence/source_census.json':'8e9dec85799755661edddc797e84bf4cb12929db7551d6055ce333c280d7f476',
 'inventory_20261010/evidence/curated_sources.json':'75903bf9182cb3219a431d449e86a543a5ea0d2feb30345c35cdd9b5238d05f6',
 'benchmarks/bone_marrow/validation/code_snapshot_output_safety_20261010.json':'b98a13a4a48dfe6bad0a68d1d42fbe20bc9a85532e298d0ff231d8b5f1a12525'}
for name,digest in anchors.items():
 if inv.sha(inv.STUDY/name)!=digest: raise ValueError('Prior baseline anchor changed: '+name)
report={'old_work':old_work(source),'technical':prior_technical(source),
        'HNSCC':prior_hnscc(source),'melanoma':prior_melanoma(source)}
p=code/'validation/lung_execution_20261010/report.json'
if inv.sha(p)!='6becb0ad0d45013e95f675ea79abbe7569f39e02f3014c41023422f84be50ce3': raise ValueError('Lung handoff changed')
m=inv.read_json(p)
for name,digest in m['current_code_sha256'].items():
 if inv.sha(code/name)!=digest: raise ValueError('Accepted code changed: '+name)
bundle,run=Path(m['bundle']['path']),Path(m['run']['path'])
if inv.sha(bundle/'MANIFEST.json')!=m['bundle']['manifest_sha256'] or inv.sha(run/'RUN_MANIFEST.json')!=m['run']['manifest_sha256']:raise ValueError('Accepted lung run changed')
report['lung']=lung.compare(run,bundle,source)
p=inv.STUDY/'benchmarks/bone_marrow/verify.py'; spec=importlib.util.spec_from_file_location('bm_readonly',p)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
report['BM_RAMP']=mod.verify(inv.STUDY/'benchmarks/bone_marrow/runs/rebuild_20261010')
if not report['BM_RAMP']['all_reference_checks_passed']:raise ValueError('BM RAMP changed')
print(json.dumps(report))
'''


def compare_baselines(before, after):
    """Ignore only the BM verifier's observation timestamp, not any evidence."""
    normalized = []
    for value in (before, after):
        bm = value['BM_RAMP']
        datetime.fromisoformat(bm['checked_at'])  # Missing/invalid timestamps still fail.
        normalized.append({**value, 'BM_RAMP': {k: v for k, v in bm.items() if k != 'checked_at'}})
    if normalized[0] != normalized[1]: raise ValueError('Earlier module verification changed')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'python', 'output'): p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--study-root', type=Path, default=inv.HERE.parents[1], help='Accepted ramp-study tree for read-only baseline checks')
    p.add_argument('--rerun-synthetic', action='store_true')
    p.add_argument('--rscript', type=Path); p.add_argument('--library', type=Path)
    p.add_argument('--scratch-parent', type=Path, default=Path('/private/tmp'))
    a = p.parse_args(); source = a.source_root.resolve(strict=True); python = a.python.absolute()
    if a.rerun_synthetic and (not a.rscript or not a.library): p.error('Synthetic rerun requires --rscript and --library')
    if not a.output.resolve().is_relative_to(inv.HERE / 'validation'): p.error('Choose a new directory under module/validation')
    output = io.reserve(a.output, (source, inv.SNAPSHOT))
    env = os.environ.copy(); env.pop('PYTHONPATH', None); env['PYTHONDONTWRITEBYTECODE'] = '1'
    report = {'utc': datetime.now(timezone.utc).isoformat(), 'real_cohort_fit': False, 'provider_extraction': False,
              'clean_install': False, 'scope': 'Harmony phase 1: code, provenance, input bundles, safety and synthetic execution',
              'r_locale_environment': dict(study.R_LOCALE_ENV)}

    def command(argv, name, cwd=output, command_env=env):
        argv = list(map(str, argv))
        with (output / (name+'.stdout.log')).open('x') as stdout, (output / (name+'.stderr.log')).open('x') as stderr:
            result = subprocess.run(argv, cwd=cwd, env=command_env, stdout=stdout, stderr=stderr)
        if result.returncode: raise RuntimeError(name+' failed; see logs')
        print('PASS '+name, flush=True)
        return {'command': argv, 'returncode': result.returncode,
                'stdout': name+'.stdout.log', 'stderr': name+'.stderr.log'}

    try:
        accepted_study = a.study_root.resolve(strict=True)
        old_module = accepted_study / 'methods/cycombine'
        report['prior_before'] = command([python, '-B', '-c', BASELINE_SCRIPT, source, old_module], 'prior_before')
        report['source_verification'] = inv.verify_source(source)
        print('PASS source verification', flush=True)
        report['common_inputs'] = command([python, '-B', accepted_study / 'benchmarks/integration_inputs/inputs.py',
            'verify-staged', '--input-root', accepted_study / 'benchmarks/integration_inputs/runs/frozen_inputs_20261010'], 'common_inputs')
        report['bundles'] = []
        with tempfile.TemporaryDirectory(prefix='harmony_validation_') as name:
            root = Path(name).resolve(); module = root / 'module'
            shutil.copytree(inv.HERE, module, ignore=shutil.ignore_patterns('validation', 'runs', '__pycache__', '.DS_Store'))
            report['tests'] = command([python, '-B', '-m', 'unittest', 'discover', '-s', module / 'tests', '-v'], 'python_tests', root)
            report['python_tests_passed'] = sum(map(int, re.findall(r'Ran (\d+) tests? in', (output/'python_tests.stderr.log').read_text())))
            for spec in inv.verify_local()['branches']:
                key = spec['case'] + '__' + spec['profile']
                bundle = study.prepare(source, root / key, spec['case'], spec['profile'])
                bm = io.read_json(bundle / 'MANIFEST.json'); _, files = study.verify_bundle(bundle)
                moved = root / (key + '_moved'); bundle.rename(moved)
                check = command([python, '-B', module / 'study.py', 'verify', '--bundle', moved], key, root)
                report['bundles'].append({'case': spec['case'], 'profile': spec['profile'], 'counts': bm['counts'],
                    'payload_files': len(files), 'bytes': sum(r['bytes'] for r in files), 'manifest': bm,
                    'moved_verification': check, 'fit_performed': False})
                shutil.rmtree(moved)  # Only our owned temporary input bundle.
        if a.rerun_synthetic:
            report['synthetic_command'] = command([python, '-B', inv.HERE / 'tests/run_synthetic.py',
                '--library', a.library.resolve(strict=True), '--rscript', a.rscript.absolute(), '--scratch-parent', a.scratch_parent,
                '--output', output / 'synthetic'], 'synthetic')
            synthetic = output / 'synthetic'
            with tempfile.TemporaryDirectory(prefix='harmony_R_tests_', dir=a.scratch_parent) as name:
                root = Path(name).resolve(); adapter = root / 'adapter'; shutil.copytree(inv.SNAPSHOT / 'adapter', adapter)
                (adapter / '.Rlib').symlink_to(a.library.resolve(strict=True), target_is_directory=True)
                report['adapter_tests'] = [command([a.rscript.absolute(), '--vanilla', adapter / 'tests' / test], test, root,
                    {**study.r_environment(env), 'TMPDIR': str(root)}) for test in ('smoke_test.R', 'integrity_contract_test.R', 'cli_contract_test.R')]
        else:
            synthetic = inv.HERE / 'validation/synthetic_20261010'
            adapter_evidence = inv.HERE / 'validation/adapter_tests_20261010'
            m = io.read_json(adapter_evidence / 'report.json')
            if m['status'] != 'passed' or len(m['tests']) != 3 or any(r['returncode'] != 0 for r in m['tests']):
                raise ValueError('Missing passing R adapter evidence')
            report['adapter_tests_saved'] = {'report': str(adapter_evidence / 'report.json'), **io.record(adapter_evidence / 'report.json')}
        synth = io.read_json(synthetic / 'report.json')
        if synth['status'] != 'passed' or synth['real_cohort_fit'] is not False or len(synth['results']) != 2:
            raise ValueError('Missing passing synthetic evidence')
        for result in synth['results']:
            if result['comparison']['maximum_absolute_difference'] != 0 or not result['comparison']['direct_wrapper_identical']:
                raise ValueError('Synthetic parity differs')
            if result['request']['code_records'] != study.code_records(): raise ValueError('Synthetic tests used different execution code')
        report['synthetic'] = {'report': str(synthetic / 'report.json'), **io.record(synthetic / 'report.json'),
            'comparisons': [r['comparison'] for r in synth['results']]}
        report['postcheck'] = inv.verify_source(source)
        report['prior_after'] = command([python, '-B', '-c', BASELINE_SCRIPT, source, old_module], 'prior_after')
        compare_baselines(io.read_json(output/'prior_before.stdout.log'), io.read_json(output/'prior_after.stdout.log'))
        report['prior_comparison_ignored_field'] = 'BM_RAMP.checked_at'
        report['catalog_sha256'] = inv.CATALOG_SHA
        report['current_code_sha256'] = {p.relative_to(inv.HERE).as_posix(): io.sha(p) for p in sorted(inv.HERE.rglob('*'))
            if p.is_file() and not {'validation', 'runs', '__pycache__', 'frozen_20261010'} & set(p.relative_to(inv.HERE).parts)}
        report['status'] = 'passed'
        io.create_json(output / 'report.json', report)
    except BaseException as error:
        io.create_json(output / 'FAILED.json', {**report, 'error': repr(error)}); raise
    print(output / 'report.json')


if __name__ == '__main__': main()
