"""Create-only evidence for the replay branch-grid verification fix; no fitting."""
import argparse
import ast
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import replay as r
import validate
import validate_replay

HANDOFF_SHA = '6d16502ae4bac5f042aacc6e8a1806cfa3a4d57c208784fa2af7d100d8f6c607'
ALLOWED_CHANGED = {'replay.py', 'compare_replay.py'}


def cli(script, batch, bundle, python):
    command = [str(python), '-B', str(script), 'verify-run', '--output', str(batch),
               '--bundle', str(bundle)]
    result = subprocess.run(command, capture_output=True, text=True)
    return {'command': command, 'returncode': result.returncode,
            'stdout': result.stdout, 'stderr': result.stderr}


def write_batch(path, report):
    path.mkdir()
    r.write(path / 'BATCH_MANIFEST.json', report)
    r.write(path / 'COMPLETE.json', {'status': 'complete',
                                    'manifest': r.record(path / 'BATCH_MANIFEST.json')})


def scope_check(preflight):
    result = {}
    allowed = {'replay.py': {'batch', 'verify_branch', 'main'},
               'compare_replay.py': {'branch_records', 'compare_batch'}}
    additions = {'replay.py': {'check_controller_records', 'branch_records', 'verify_run'},
                 'compare_replay.py': set()}
    for name in sorted(ALLOWED_CHANGED):
        old = ast.parse((preflight.parent / 'code' / name).read_text())
        new = ast.parse((r.HERE / name).read_text())
        def functions(module):
            return {n.name: ast.dump(n, include_attributes=False) for n in module.body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        before, after = functions(old), functions(new)
        changed = {n for n in before if n in after and before[n] != after[n]}
        r.require(not set(before) - set(after), 'Function removed: ' + name)
        r.require(changed <= allowed[name] and set(after) - set(before) == additions[name],
                  'Unexpected function changes: ' + name)
        old_top = [n for n in old.body if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        new_top = [n for n in new.body if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and not (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and
                       t.id == 'ACCEPTED_R3_CODE' for t in n.targets))]
        r.require(ast.dump(ast.Module(body=old_top, type_ignores=[])) ==
                  ast.dump(ast.Module(body=new_top, type_ignores=[])), 'Module-level changes: ' + name)
        result[name] = {'changed_functions': sorted(changed),
                        'added_functions': sorted(set(after) - set(before)),
                        'other_functions_and_module_statements_unchanged': True}
    return result


def real_cli_checks(preflight, batch, bundle, python):
    results = {}
    original = r.read(batch / 'BATCH_MANIFEST.json')
    with tempfile.TemporaryDirectory(prefix='replay-grid-cli-') as temp:
        root = Path(temp).resolve()
        duplicate = copy.deepcopy(original)
        duplicate['branches'] = [original['branches'][0]] * len(original['branches'])
        target = root / 'duplicate'; write_batch(target, duplicate)
        first = original['branches'][0]
        relative = Path(first['case']) / first['variant']
        shutil.copytree(batch / relative, target / relative)
        before = r.tree(target)
        results['old_duplicate'] = cli(preflight.parent / 'code/replay.py', target, bundle, python)
        r.require(results['old_duplicate']['returncode'] == 0, 'Old defect was not reproduced')
        results['duplicate_63'] = cli(r.HERE / 'replay.py', target, bundle, python)
        r.require(results['duplicate_63']['returncode'] != 0 and
                  'Batch branch grid differs' in results['duplicate_63']['stderr'], 'Duplicate CLI accepted')
        r.require(r.tree(target) == before, 'Verification wrote to temporary batch')
        for name in ('one_duplicate', 'missing', 'extra', 'foreign_case', 'foreign_variant', 'unsafe_path'):
            malformed = copy.deepcopy(original)
            if name == 'one_duplicate': malformed['branches'][-1] = malformed['branches'][0]
            elif name == 'missing': malformed['branches'].pop()
            elif name == 'extra': malformed['branches'].append(copy.deepcopy(malformed['branches'][0]))
            elif name == 'foreign_case': malformed['branches'][0]['case'] = 'unregistered'
            elif name == 'foreign_variant': malformed['branches'][0]['variant'] = 'v99'
            else: malformed['branches'][0]['case'] = '../../outside'
            target = root / name; write_batch(target, malformed)
            before = r.tree(target)
            results[name] = cli(r.HERE / 'replay.py', target, bundle, python)
            r.require(results[name]['returncode'] != 0 and 'Batch branch grid differs' in results[name]['stderr'],
                      'Malformed grid not rejected before branch access: ' + name)
            r.require(r.tree(target) == before, 'Negative CLI wrote files')
        target = root / 'incomplete'; target.mkdir()
        results['incomplete'] = cli(r.HERE / 'replay.py', target, bundle, python)
        r.require(results['incomplete']['returncode'] != 0 and 'Batch incomplete' in results['incomplete']['stderr'],
                  'Incomplete batch diagnostic missing')
        r.require(not list(target.iterdir()), 'Incomplete CLI wrote files')
        moved = root / 'input bundle with spaces'; shutil.copytree(bundle, moved)
        results['real_63_moved_bundle'] = cli(r.HERE / 'replay.py', batch, moved, python)
        r.require(results['real_63_moved_bundle']['returncode'] == 0, 'Relocated bundle verification failed')
    results['real_63'] = cli(r.HERE / 'replay.py', batch, bundle, python)
    r.require(results['real_63']['returncode'] == 0, 'Valid real batch verification failed')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['preflight', 'handoff'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight', type=Path)
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    parser.add_argument('--batch', type=Path, default=r.HERE / 'runs/seven_metric_replay_20261011_r3')
    parser.add_argument('--bundle', type=Path, default=r.HERE / 'runs/replay_inputs_20261011')
    args = parser.parse_args()
    output = r.reserve(r.local_output(args.output, 'validation'))
    report = {'mode': args.mode, 'scope': 'read-only verification safety; no metric or UMAP fitting'}
    try:
        handoff = r.HERE / 'validation/replay_handoff_20261011/report.json'
        r.require(r.record(handoff)['sha256'] == HANDOFF_SHA, 'Previous handoff changed')
        report['previous_handoff'] = r.record(handoff)
        if args.mode == 'preflight':
            # Exclude this newly reserved evidence directory from its own baseline.
            excluded = set(r.tree(output))
            report['module_files'] = {name: rec for name, rec in r.tree(r.HERE).items()
                                      if not name.startswith(output.relative_to(r.HERE).as_posix() + '/')}
            for name in (*r.CODE, 'compare_replay.py'):
                r.copy(r.HERE / name, output / 'code' / name, r.record(r.HERE / name))
            report['code_records'] = r.code_records()
            report['outside_files'] = validate.outside()
            r.require(report['outside_files'] == r.read(r.PHASE1)['outside_files'], 'Outside baseline changed')
            report['original_indexes'] = validate.originals()
        else:
            r.require(args.preflight is not None, 'Missing preflight')
            prior = r.read(args.preflight)
            report['preflight'] = r.record(args.preflight)
            r.require(prior['code_records'] == r.ACCEPTED_R3_CODE, 'Historical controller pin differs')
            for name, rec in prior['module_files'].items():
                r.check(args.preflight.parent / 'code' / name if name in ALLOWED_CHANGED else r.HERE / name, rec)
            report['preserved_module_files'] = len(prior['module_files']) - len(ALLOWED_CHANGED)
            report['archived_changed_files'] = sorted(ALLOWED_CHANGED)
            report['scope_check'] = scope_check(args.preflight)
            report['tests'] = validate_replay.test_copy(output, args.python.absolute())
            report['cli'] = real_cli_checks(args.preflight, args.batch, args.bundle, args.python.absolute())
            from compare_replay import compare_batch
            comparison = compare_batch(args.batch, args.bundle)
            old = r.read(r.HERE / 'validation/replay_numerical_20261011/report.json')
            r.require(comparison == old, 'Numerical comparison report changed')
            report['numerical_report_unchanged'] = True
            report['comparison'] = comparison
            for name, rec in prior['module_files'].items():
                r.check(args.preflight.parent / 'code' / name if name in ALLOWED_CHANGED else r.HERE / name, rec)
            r.require(validate.outside() == prior['outside_files'], 'Outside files changed')
            report['outside_unchanged'] = len(prior['outside_files'])
            report['original_indexes'] = validate.originals()
            report['current_files'] = {name: r.record(r.HERE / name) for name in
                ('replay.py', 'compare_replay.py', 'tests/test_branch_grid.py',
                 'validate_branch_grid_fix.py', 'BRANCH_GRID_FIX_20261011.md')}
        report['status'] = 'passed'; r.write(output / 'report.json', report)
    except BaseException as error:
        r.write(output / 'FAILED.json', {'error': repr(error)})
        raise


if __name__ == '__main__': main()
