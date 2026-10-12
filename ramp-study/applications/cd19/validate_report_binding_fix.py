"""Small, create-only evidence for the report-binding repair; never refit or copy results."""
import argparse
import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from common import HERE, STORY, read, write, sha, record, files, links, reserve


def capture(output):
    before = {'files': files(HERE), 'links': links(HERE)}
    output = reserve(output, [HERE / 'runs', HERE / 'frozen_20261011'])
    (output / 'controller').mkdir()
    shutil.copyfile(HERE / 'validate.py', output / 'controller/validate.py')
    write(output / 'report.json', before)
    write(output / 'COMPLETE.json', {'report_sha256': sha(output / 'report.json')})
    return {'files': len(before['files']), 'links': len(before['links']), 'report_sha256': sha(output / 'report.json')}


def sealed_report(directory, data):
    directory.mkdir()
    write(directory / 'report.json', data)
    write(directory / 'COMPLETE.json', {'status': 'complete', 'report_sha256': sha(directory / 'report.json')})


def timeline():
    source = Path(read(HERE / 'frozen_20261011/LOCK.json')['source_root']) / STORY
    names = ('alignment/output/manifest.json', 'sampled_data/relabel/relabeled_metadata.csv.gz',
             'sampled_data/relabel/manifest.json')
    observations = {}
    for name in names:
        path = source / name
        observations[name] = record(path) | {
            'mtime_ns': path.stat().st_mtime_ns,
            'mtime_utc': datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()}
        if path.suffix == '.json':
            data = read(path)
            observations[name]['recorded_time_fields'] = {k: v for k, v in data.items()
                                                        if 'time' in k.lower() or 'date' in k.lower()}
    return {'observations': observations,
            'interpretation': 'Later regeneration is consistent with the file timeline and current producer hash; filesystem timestamps alone do not prove historical contents or the exact regeneration operation.',
            'old_bytes_recovered': False, 'old_reports_rewritten': False}


def check(preflight, output):
    preflight = Path(preflight).resolve()
    if read(preflight / 'COMPLETE.json') != {'report_sha256': sha(preflight / 'report.json')}:
        raise ValueError('Preflight changed')
    before = read(preflight / 'report.json')
    if record(preflight / 'controller/validate.py') != before['files']['validate.py']:
        raise ValueError('Archived validator changed')
    output = reserve(output, [preflight, HERE / 'runs', HERE / 'frozen_20261011'])
    try:
        run = HERE / 'runs/rebuild_20261011_r2'
        lockdir = HERE / 'frozen_20261011'
        comparison = HERE / 'validation/comparison_20261011'
        figure = HERE / 'validation/figure_metrics_20261011'
        env = os.environ.copy(); env['PYTHONDONTWRITEBYTECODE'] = '1'
        commands = []

        def invoke(label, command, expected_success, cwd=HERE):
            result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
            (output / (label + '.stdout')).write_text(result.stdout)
            (output / (label + '.stderr')).write_text(result.stderr)
            commands.append({'label': label, 'command': command, 'returncode': result.returncode})
            if (result.returncode == 0) != expected_success:
                raise ValueError(f'Unexpected CLI result: {label}')
            return result

        def args(out, comp, fig):
            return ['--lockdir', str(lockdir), '--run', str(run), '--comparison', str(comp),
                    '--figure-metrics', str(fig), '--out', str(out)]

        with tempfile.TemporaryDirectory(prefix='cd19_binding_fix_') as temporary:
            tmp = Path(temporary).resolve()
            # Only code and small frozen scripts; no source data or run payloads.
            controller = tmp / 'moved controller'; controller.mkdir()
            for p in HERE.glob('*.py'):
                shutil.copyfile(p, controller / p.name)
            shutil.copytree(HERE / 'tests', controller / 'tests', ignore=shutil.ignore_patterns('__pycache__'))
            shutil.copytree(lockdir, controller / 'frozen_20261011')
            invoke('tests', [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests', '-v'], True, controller)
            moved_lock = tmp / 'moved lock with spaces'
            (controller / 'frozen_20261011').rename(moved_lock)
            invoke('moved_run', [sys.executable, '-B', str(controller / 'study.py'), 'verify-run',
                                '--lockdir', str(moved_lock), '--run', str(run)], True, controller)
            # The helper must likewise use explicit paths, not a default local lock.
            probe = ('import sys; sys.path.insert(0,sys.argv[1]); from validate import report_bindings; '
                     'print(report_bindings(*sys.argv[2:])[2])')
            invoke('moved_binding', [sys.executable, '-B', '-c', probe, str(controller), str(moved_lock),
                                     str(run), str(comparison), str(figure)], True, controller)
            case_specs = [('both_run_hashes', 'both', None, 'wrong')]
            for side, fields in [('comparison', ('run_manifest_sha256', 'lock_sha256')),
                                 ('figure', ('base_run_manifest_sha256', 'supplementary_lock_sha256'))]:
                for field in fields:
                    case_specs += [(side + '_' + field + '_' + mode, side, field, mode) for mode in ('wrong', 'missing')]
            cases = []
            for label, side, field, mode in case_specs:
                folder = tmp / label; folder.mkdir()
                comp = read(comparison / 'report.json'); fig = read(figure / 'report.json')
                if side == 'both':
                    comp['run_manifest_sha256'] = '0' * 64
                    fig['base_run_manifest_sha256'] = '0' * 64
                else:
                    target = comp if side == 'comparison' else fig
                    if mode == 'missing':
                        del target[field]
                    else:
                        target[field] = '0' * 64
                sealed_report(folder / 'comparison', comp); sealed_report(folder / 'figure', fig)
                payload_before = files(folder)
                if side == 'both':
                    # Run the exact old validator CLI against the unchanged real companion modules.
                    entry = ('import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); '
                             'p=sys.argv.pop(1); sys.argv[0]=p; runpy.run_path(p,run_name="__main__")')
                    old_out = output / 'old_false_acceptance'
                    invoke('old_false_acceptance', [sys.executable, '-B', '-c', entry, str(HERE),
                           str(preflight / 'controller/validate.py'), *args(old_out, folder / 'comparison', folder / 'figure')], True)
                    if not (old_out / 'COMPLETE.json').exists():
                        raise ValueError('Old false acceptance did not complete')
                rejected = folder / 'rejected_output'
                result = invoke(label, [sys.executable, '-B', str(HERE / 'validate.py'),
                                *args(rejected, folder / 'comparison', folder / 'figure')], False)
                if 'report binding differs' not in result.stderr or rejected.exists() or files(folder) != payload_before:
                    raise ValueError(f'Incorrect rejection or changed fixture: {label}')
                cases.append({'case': label, 'returncode': result.returncode, 'output_created': False,
                              'reports_unchanged': True})
            copied_bytes = sum(p.stat().st_size for p in tmp.rglob('*') if p.is_file())
        # Normal acceptance writes only a NEW handoff evidence directory, never rewrites old reports.
        accepted = output / 'current_handoff'
        invoke('current_handoff', [sys.executable, '-B', str(HERE / 'validate.py'),
                                 *args(accepted, comparison, figure)], True)
        old = ast.parse((preflight / 'controller/validate.py').read_text())
        new = ast.parse((HERE / 'validate.py').read_text())
        defs = lambda tree: {n.name: ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        a, b = defs(old), defs(new)
        changed = sorted(k for k in a if k not in b or a[k] != b[k])
        added = sorted(set(b) - set(a))
        if changed != ['validate'] or added != ['report_bindings']:
            raise ValueError('Unexpected validator change scope')
        checked = 0
        for rel, expected in before['files'].items():
            if rel != 'validate.py':
                if record(HERE / rel) != expected:
                    raise ValueError(f'Existing file changed: {rel}')
                checked += 1
        for rel, expected in before['links'].items():
            if not (HERE / rel).is_symlink() or str((HERE / rel).resolve()) != expected:
                raise ValueError('Existing input link changed')
        report = {'status': 'passed_pending_external_audit', 'preflight': record(preflight / 'report.json'),
                  'old_validator': record(preflight / 'controller/validate.py'), 'current_validator': record(HERE / 'validate.py'),
                  'code': record(Path(__file__)), 'test_code': record(HERE / 'tests/test_report_binding.py'),
                  'old_functions_changed': changed, 'new_functions': added, 'negative_cli': cases,
                  'current_handoff': record(accepted / 'report.json'), 'commands': commands,
                  'existing_files_unchanged': checked, 'existing_links_unchanged': len(before['links']),
                  'temporary_copied_bytes': copied_bytes, 'no_retraining_or_result_copy': True,
                  'label_timeline': timeline()}
        write(output / 'report.json', report)
        write(output / 'COMPLETE.json', {'status': 'complete', 'report_sha256': sha(output / 'report.json')})
        return {k: v for k, v in report.items() if k not in ('commands', 'label_timeline')}
    except BaseException as exc:
        write(output / 'FAILED.json', {'error': repr(exc)})
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['capture', 'check'])
    p.add_argument('--out', required=True); p.add_argument('--preflight')
    a = p.parse_args()
    print(json.dumps(capture(a.out) if a.mode == 'capture' else check(a.preflight, a.out), indent=2))
