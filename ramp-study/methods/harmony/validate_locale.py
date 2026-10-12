"""Locale regression evidence, synthetic fitting only; every output is create-only."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import harmony_io as io
import inventory as inv
import study

EXPECTED_LOCALE = {name: ('C' if name == 'LC_NUMERIC' else 'C.UTF-8') for name in
                   ('LC_COLLATE', 'LC_CTYPE', 'LC_MONETARY', 'LC_NUMERIC', 'LC_TIME', 'LC_MESSAGES')}
ALLOWED_CHANGES = {'study.py', 'check_environment.R', 'tests/run_synthetic.py', 'validate.py',
                   'tests/test_contract.py', 'README.md'}


def assert_locale_log(path):
    import json
    rows = [json.loads(line.split('=', 1)[1]) for line in Path(path).read_text().splitlines()
            if line.startswith('R_LOCALE_JSON=')]
    expected = {'environment': study.R_LOCALE_ENV, 'effective': EXPECTED_LOCALE}
    if rows != [expected]: raise ValueError('Missing or different effective R locale: ' + str(path))
    return rows[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'python', 'rscript', 'library', 'output', 'baseline-before'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--study-root', type=Path, default=inv.HERE.parents[1])
    p.add_argument('--scratch-parent', type=Path, default=Path('/private/tmp'))
    a = p.parse_args(); source = a.source_root.resolve(strict=True); library = a.library.resolve(strict=True)
    output = io.reserve(a.output, (source, library, inv.SNAPSHOT, a.baseline_before))
    env = os.environ.copy(); env.pop('PYTHONPATH', None); env['PYTHONDONTWRITEBYTECODE'] = '1'
    for name in list(env):
        if name in ('LANG', 'LANGUAGE') or name.startswith('LC_'): del env[name]
    scenarios = {'C': {**env, 'LC_ALL': 'C', 'LANG': 'C'}, 'unset': env}
    report = {'utc': datetime.now(timezone.utc).isoformat(), 'real_cohort_fit': False,
              'scope': 'R locale pin, effective-locale checks and synthetic regression', 'commands': []}

    def command(argv, key, child_env, success=True, cwd=output):
        argv = list(map(str, argv))
        with (output/(key+'.stdout.log')).open('x') as stdout, (output/(key+'.stderr.log')).open('x') as stderr:
            proc = subprocess.run(argv, cwd=cwd, env=child_env, stdout=stdout, stderr=stderr)
        row = {'name': key, 'command': argv, 'returncode': proc.returncode, 'expected_success': success,
               'environment': {k: child_env.get(k) for k in ('LC_ALL','LANG')},
               'stdout': key+'.stdout.log', 'stderr': key+'.stderr.log'}
        report['commands'].append(row)
        if (proc.returncode == 0) != success: raise RuntimeError('Unexpected command result: '+key)
        print('PASS '+key, flush=True)
        return row

    try:
        before = io.read_json(a.baseline_before)
        io.copy_new(a.baseline_before, output/'baseline_before.json', io.record(a.baseline_before))
        catalog = inv.verify_local(); historical = []
        for spec in catalog['branches']:
            name = str(Path(spec['output']).parent/'sessionInfo.txt'); path = source/name
            index = path.read_text().splitlines().index('locale:')
            value = path.read_text().splitlines()[index+1]
            if value != '[1] C.UTF-8/C.UTF-8/C.UTF-8/C/C.UTF-8/C.UTF-8': raise ValueError('Historical locale differs')
            historical.append({'case': spec['case'], 'profile': spec['profile'], 'path': name,
                               'locale': value, **io.record(path)})
        report['historical_locales'] = historical
        # This exercises validate.py's own original-R-test dispatch and fit(),
        # while the Python parent really starts with LC_ALL=C and LANG=C.
        full = output/'parent_C'
        command([a.python.absolute(), '-B', inv.HERE/'validate.py', '--source-root', source,
            '--study-root', a.study_root.resolve(strict=True), '--python', a.python.absolute(), '--output', full,
            '--rerun-synthetic', '--rscript', a.rscript.absolute(), '--library', library,
            '--scratch-parent', a.scratch_parent], 'parent_C_validation', scenarios['C'])
        report['full_validation'] = {'path': str(full/'report.json'), **io.record(full/'report.json')}
        other = output/'parent_unset_synthetic'
        command([a.python.absolute(), '-B', inv.HERE/'tests/run_synthetic.py', '--library', library,
            '--rscript', a.rscript.absolute(), '--scratch-parent', a.scratch_parent, '--output', other],
            'parent_unset_synthetic', scenarios['unset'])
        report['synthetic_scenarios'] = []
        for mode, folder in [('C', full/'synthetic'), ('unset', other)]:
            data = io.read_json(folder/'report.json')
            expected_parent = {key: scenarios[mode].get(key) for key in ('LC_ALL','LANG')}
            if data['parent_locale_environment'] != expected_parent: raise ValueError('Wrong test parent locale')
            for row in data['results']:
                if (row['request']['r_locale_environment'] != study.R_LOCALE_ENV or
                    row['request']['code_records'] != study.code_records() or
                    row['comparison']['maximum_absolute_difference'] != 0 or not row['comparison']['direct_wrapper_identical']):
                    raise ValueError('Synthetic locale/identity mismatch')
                for when in ('before','after'):
                    assert_locale_log(folder/(row['kind']+'.environment_'+when+'.stdout.log'))
            report['synthetic_scenarios'].append({'parent': mode, 'parent_environment': expected_parent,
                'report': str(folder/'report.json'), **io.record(folder/'report.json'),
                'comparisons': [r['comparison'] for r in data['results']]})
        with tempfile.TemporaryDirectory(prefix='harmony_locale_', dir=a.scratch_parent) as name:
            root = Path(name).resolve(); adapter = root/'adapter'; shutil.copytree(inv.SNAPSHOT/'adapter', adapter)
            (adapter/'.Rlib').symlink_to(library, target_is_directory=True)
            # Independently reproduce the original unmodified code's reported failure.
            for mode, parent in scenarios.items():
                negative = {**parent, 'TMPDIR': str(root)}
                command([a.rscript.absolute(), '--vanilla', adapter/'tests/smoke_test.R'], 'original_'+mode,
                        negative, success=False, cwd=root)
                if 'fingerprint manifest does not contain the exact required file set' not in (output/('original_'+mode+'.stderr.log')).read_text():
                    raise ValueError('Negative control failed for an unrelated reason')
                # Our runtime check must reject a wrong/fallback locale explicitly.
                command([a.rscript.absolute(), '--vanilla', inv.HERE/'check_environment.R', adapter, 'technical'],
                        'guard_'+mode, negative, success=False, cwd=root)
                if 'R locale must be C.UTF-8' not in (output/('guard_'+mode+'.stderr.log')).read_text():
                    raise ValueError('Locale guard did not fail clearly')
            pinned = {**study.r_environment(scenarios['unset']), 'TMPDIR': str(root)}
            for kind in ('technical', 'story'):
                key = 'environment_unset_'+kind
                command([a.rscript.absolute(), '--vanilla', inv.HERE/'check_environment.R', adapter, kind], key, pinned, cwd=root)
                assert_locale_log(output/(key+'.stdout.log'))
            for test in ('smoke_test.R', 'integrity_contract_test.R', 'cli_contract_test.R'):
                command([a.rscript.absolute(), '--vanilla', adapter/'tests'/test], 'adapter_unset_'+test, pinned, cwd=root)
        # Old records remain evidence of the old code. Never rewrite their hashes.
        report['unchanged_existing_files'] = []; report['changed_files'] = []
        for name, old_sha in before['files'].items():
            new_sha = io.sha(inv.HERE/name)
            if new_sha != old_sha:
                if name not in ALLOWED_CHANGES: raise ValueError('Unexpected existing file changed: '+name)
                report['changed_files'].append({'path': name, 'before_sha256': old_sha, 'after_sha256': new_sha})
            else: report['unchanged_existing_files'].append({'path': name, 'sha256': new_sha})
        report['postcheck'] = inv.verify_source(source)
        report['current_code_sha256'] = {p.relative_to(inv.HERE).as_posix(): io.sha(p) for p in sorted(inv.HERE.rglob('*'))
            if p.is_file() and not {'validation','runs','__pycache__','frozen_20261010'} & set(p.relative_to(inv.HERE).parts)}
        report['status'] = 'passed'; io.create_json(output/'report.json', report)
    except BaseException as error:
        io.create_json(output/'FAILED.json', {**report, 'error': repr(error)}); raise
    print(output/'report.json')


if __name__ == '__main__': main()
