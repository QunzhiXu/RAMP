"""Read an existing BM source_scale run and write new evidence; never fit R."""
import argparse
import copy
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
from validate import BASELINE_SCRIPT, compare_baselines
from validate_locale import assert_locale_log

LOCALE_REPORT_SHA = '9ff13159040c9d9ba5c6d9d8cd37ed6eff4d3304ca92310e72f57497d4008e8b'
PREFLIGHT_SHA = '82cbce3a95d26205b9f65a0df2383182468a990b6211b14b5ded4aaeb487d755'


def accepted_code():
    path = inv.HERE / 'validation/locale_fix_20261010/report.json'
    if io.sha(path) != LOCALE_REPORT_SHA: raise ValueError('Accepted locale report changed')
    report = io.read_json(path)
    for name, digest in report['current_code_sha256'].items():
        if io.sha(inv.HERE / name) != digest: raise ValueError('Accepted Harmony code/doc changed: ' + name)
    return report['current_code_sha256']


def scientific_comparison(source, bundle, run, spec):
    """Compare all payloads and strictly compare metadata except named provenance changes."""
    records = {r['source']: r for r in inv.verify_local()['files']}
    old_directory = Path(spec['output']).parent
    result = run / study.result_path(spec)

    def original(name):
        relative = str(old_directory / name); path = io.within(source, relative)
        io.check(path, records[relative]); return path

    old = io.read_json(original('manifest.json')); new = io.read_json(result / 'manifest.json')
    payloads = {}
    for row in old['output_files']:
        path = original(row['file']); actual = result / row['file']
        payloads[row['file']] = {'bytes_equal': io.sha(path) == io.sha(actual),
                               'historical': io.record(path), 'current': io.record(actual)}
    different = {name for name, row in payloads.items() if not row['bytes_equal']}
    if not different <= {'resolved_config.json', 'sessionInfo.txt'}:
        raise ValueError('Unexpected non-identical payloads: '+repr(sorted(different)))
    for name in ('input_column_audit.json', 'method_semantics.json'):
        if io.read_json(original(name)) != io.read_json(result / name): raise ValueError('Scientific JSON differs: '+name)
    config_old = io.read_json(original('resolved_config.json')); config_new = io.read_json(result/'resolved_config.json')
    config_expected = copy.deepcopy(config_old)
    config_expected['input_scale']['preprocessing_record_file'] = str(bundle/'inputs'/spec['record'])
    if config_expected != config_new: raise ValueError('Resolved config differs beyond the authorized path relocation')
    sessions = [original('sessionInfo.txt').read_text().splitlines(), (result/'sessionInfo.txt').read_text().splitlines()]
    os_lines = []
    for lines in sessions:
        indices = [i for i, line in enumerate(lines) if line.startswith('Running under: ')]
        if len(indices) != 1 or not lines[indices[0]].startswith('Running under: macOS '):
            raise ValueError('Unexpected sessionInfo operating system entry')
        os_lines.append(lines[indices[0]])
        lines[indices[0]] = '<recorded macOS release>'
    if sessions[0] != sessions[1]: raise ValueError('sessionInfo differs beyond the macOS release line')
    normalized = []
    for manifest in (old, new):
        value = copy.deepcopy(manifest)
        value['created_at'] = '<run timestamp>'
        for key in ('path', 'marker_file', 'preprocessing_record'): value['input'][key] = '<relocated path>'
        value['config_file'] = {'path': '<relocated config>', 'sha256': '<hash of relocated config>'}
        for row in value['implementation'].values(): row['path'] = '<relocated implementation path>'
        # These hashes were checked against their actual files by verify_run and
        # the frozen source catalog. Record byte comparisons separately above.
        for row in value['output_files']: row['sha256'] = '<independently checked payload hash>'
        normalized.append(value)
    if normalized[0] != normalized[1]: raise ValueError('Scientific manifest differs')
    return {'payloads': payloads, 'byte_identical_payloads': sum(r['bytes_equal'] for r in payloads.values()),
        'total_payloads': len(payloads), 'scientific_manifest_equal': True,
        'input_column_audit_equal': True, 'method_semantics_equal': True,
        'resolved_config_only_difference': 'input_scale.preprocessing_record_file',
        'session_info': {'historical_OS': os_lines[0], 'current_OS': os_lines[1],
                         'equal_except_Running_under_line': True},
        'manifest_provenance_exceptions': ['created_at', 'input.path', 'input.marker_file', 'input.preprocessing_record',
            'config_file.path', 'config_file.sha256', 'implementation.*.path', 'output_files[*].sha256']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'bundle', 'run', 'python', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--preflight', type=Path, default=inv.HERE/'validation/bm_preflight_20261010')
    p.add_argument('--study-root', type=Path, default=inv.HERE.parents[1])
    a = p.parse_args(); source, bundle, run = (x.resolve(strict=True) for x in (a.source_root,a.bundle,a.run))
    preflight = a.preflight.resolve(strict=True)
    if not a.output.resolve().is_relative_to(inv.HERE/'validation'): p.error('Use a new module/validation directory')
    output = io.reserve(a.output, (source, bundle, run, preflight, inv.SNAPSHOT))
    report = {'utc': datetime.now(timezone.utc).isoformat(), 'scope': 'BM source_scale Harmony correction only',
              'real_fit_performed_by_validator': False, 'real_fit_run': str(run), 'provider_extraction': False,
              'preprocessing_recomputed': False, 'UMAP_metrics_survival': False, 'clean_install': False}
    env = os.environ.copy(); env.pop('PYTHONPATH', None); env['PYTHONDONTWRITEBYTECODE'] = '1'

    def command(argv, key, cwd=output):
        argv = list(map(str,argv))
        with (output/(key+'.stdout.log')).open('x') as stdout, (output/(key+'.stderr.log')).open('x') as stderr:
            proc = subprocess.run(argv, cwd=cwd, env=env, stdout=stdout, stderr=stderr)
        if proc.returncode: raise RuntimeError(key+' failed; see logs')
        print('PASS '+key, flush=True)
        return {'command': argv, 'returncode': proc.returncode,
                'stdout': {'path': key+'.stdout.log', **io.record(output/(key+'.stdout.log'))},
                'stderr': {'path': key+'.stderr.log', **io.record(output/(key+'.stderr.log'))}}

    try:
        report['accepted_harmony_code'] = accepted_code()
        if io.sha(preflight/'report.json') != PREFLIGHT_SHA: raise ValueError('BM preflight record changed')
        before = io.read_json(preflight/'report.json')
        if before['status'] != 'passed' or before['code_records'] != study.code_records(): raise ValueError('Invalid preflight')
        io.check(preflight/'prior.stdout.log', before['prior_baseline']['stdout'])
        io.check(preflight/'prior.stderr.log', before['prior_baseline']['stderr'])
        report['preflight'] = {'path': str(preflight/'report.json'), 'sha256': PREFLIGHT_SHA}
        for name, expected in before['existing_module_files'].items(): io.check(io.within(inv.HERE,name), expected)
        report['unchanged_existing_harmony_files'] = len(before['existing_module_files'])
        spec, files = study.verify_bundle(bundle)
        if (spec['case'], spec['profile'], spec['cells'], len(spec['markers']), spec['units']) != ('bone_marrow','source_scale',88031,14,17):
            raise ValueError('Wrong execution branch')
        report['comparison'] = study.compare(run,bundle,source)
        report['all_payload_comparison'] = scientific_comparison(source,bundle,run,spec)
        report['request'] = io.read_json(run/'REQUEST.json')
        if report['request']['code_records'] != study.code_records(): raise ValueError('Fit used different entry code')
        report['commands'] = io.read_json(run/'COMMANDS.json')
        report['environment'] = assert_locale_log(run/'environment_before.stdout.log')
        assert_locale_log(run/'environment_after.stdout.log')
        if (run/'environment_before.stdout.log').read_bytes() != (run/'environment_after.stdout.log').read_bytes():
            raise ValueError('Environment changed during the real fit')
        report['runtime_copies'] = []
        for row in files:
            target = study.runtime_path(row,spec)
            if target is not None:
                io.check(run/target,row); report['runtime_copies'].append({'path':str(target),**row})
        report['bundle'] = {'path': str(bundle), 'payload_files':len(files), 'payload_bytes':sum(r['bytes'] for r in files),
                            'manifest_sha256':io.sha(bundle/'MANIFEST.json')}
        report['run'] = {'path':str(run),'manifest_sha256':io.sha(run/'RUN_MANIFEST.json'),
                        'completion_sha256':io.sha(run/'COMPLETE.json')}
        # New tests use an isolated module copy; moving the bundle exercises the
        # existing completed-run verifier without refitting or source arguments.
        with tempfile.TemporaryDirectory(prefix='harmony_bm_audit_') as name:
            root = Path(name).resolve(); module = root/'module'
            shutil.copytree(inv.HERE,module,ignore=shutil.ignore_patterns('validation','runs','__pycache__','.DS_Store'))
            report['tests'] = command([a.python.absolute(),'-B','-m','unittest','discover','-s',module/'tests','-v'], 'python_tests',root)
            report['python_tests_passed'] = sum(map(int,re.findall(r'Ran (\d+) tests? in',(output/'python_tests.stderr.log').read_text())))
            moved = root/'moved bundle'; shutil.copytree(bundle,moved)
            report['moved_bundle_verification'] = command([a.python.absolute(),'-B',module/'study.py','verify-run',
                '--bundle',moved,'--run',run], 'moved_bundle',root)
        accepted_study = a.study_root.resolve(strict=True)
        report['common_inputs'] = command([a.python.absolute(),'-B',accepted_study/'benchmarks/integration_inputs/inputs.py',
            'verify-staged','--input-root',accepted_study/'benchmarks/integration_inputs/runs/frozen_inputs_20261010'], 'common_inputs')
        report['prior_modules'] = command([a.python.absolute(),'-B','-c',BASELINE_SCRIPT,source,
                                          accepted_study/'methods/cycombine'], 'prior_modules')
        compare_baselines(io.read_json(preflight/'prior.stdout.log'),io.read_json(output/'prior_modules.stdout.log'))
        report['prior_comparison_ignored_field'] = 'BM_RAMP.checked_at'
        report['source_postcheck'] = inv.verify_source(source)
        for name, expected in before['existing_module_files'].items(): io.check(io.within(inv.HERE,name),expected)
        accepted_code(); study.verify_run(run,bundle)
        report['current_files'] = {name:io.record(inv.HERE/name) for name in
                                  ('validate_bm_execution.py','STUDY_BM.md','AUDIT_BM_EXECUTION.md')}
        report['status'] = 'passed'; io.create_json(output/'report.json',report)
    except BaseException as error:
        io.create_json(output/'FAILED.json',{**report,'error':repr(error)});raise
    print(output/'report.json')


if __name__=='__main__': main()
