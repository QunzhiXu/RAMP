"""Evidence for the helper fix and full cached-coordinate metric replay."""
import argparse
import ast
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

import contract as c
import inventory as inv
import replay as r
import validate


def test_copy(output, python):
    with tempfile.TemporaryDirectory(prefix='metric-replay-tests-') as temp:
        dest = Path(temp) / 'module'
        shutil.copytree(r.HERE, dest, ignore=shutil.ignore_patterns('runs', 'validation', '__pycache__', 'provenance', '.cache'))
        target = dest / 'validation/handoff_20261011/report.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(r.PHASE1, target)
        cmd = [str(python), '-B', '-m', 'unittest', 'discover', '-s', str(dest / 'tests'), '-v']
        run = subprocess.run(cmd, cwd=dest, text=True, capture_output=True)
        with (output / 'tests.log').open('x') as handle: handle.write(run.stdout + run.stderr)
        r.require(run.returncode == 0, 'Tests failed in temporary module copy')
        counts = re.findall(r'Ran (\d+) tests? in ', run.stdout + run.stderr)
        r.require(len(counts) == 1, 'Cannot establish executed test count')
        r.require(int(counts[0]) > 0, 'No tests executed')
        return {'command': cmd, 'returncode': run.returncode, 'tests_run': int(counts[0]),
                'working_copy': 'temporary; removed on exit'}


def old_phase(preflight):
    report = r.read(preflight)
    allowed = {'contract.py', 'validate.py', 'tests/test_inventory.py'}
    for name, expected in report['old_code'].items():
        if name in allowed: r.check(preflight.parent / 'phase1_code' / name, expected)
        else: r.check(r.HERE / name, expected)
    old = ast.parse((preflight.parent / 'phase1_code/contract.py').read_text())
    new = ast.parse((r.HERE / 'contract.py').read_text())
    # The only numerical-helper fix is the missing json binding.
    def stripped(tree):
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'dict':
                node.keywords = [kw for kw in node.keywords if not (kw.arg == 'json' and isinstance(kw.value, ast.Name) and kw.value.id == 'json')]
        return ast.dump(tree, include_attributes=False)
    r.require(stripped(old) == stripped(new), 'Contract changed beyond json namespace fix')
    return {'allowed_changes': sorted(allowed), 'other_phase1_code_unchanged': True,
            'preflight': r.record(preflight), 'old_versions_retained': True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['test', 'handoff'])
    p.add_argument('--python', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--preflight', type=Path, required=True)
    p.add_argument('--batch', type=Path); p.add_argument('--bundle', type=Path)
    p.add_argument('--numerical-evidence', type=Path)
    p.add_argument('--source-root', type=Path, required=True)
    args = p.parse_args()
    output = r.reserve(r.local_output(args.output, 'validation'), [args.source_root])
    report = {'mode': args.mode}
    try:
        report['phase1_preservation'] = old_phase(args.preflight)
        prior = r.read(args.preflight)
        r.require(validate.outside() == prior['outside_files'], 'Accepted modules changed')
        report['outside_unchanged'] = len(prior['outside_files'])
        report['original_indexes'] = validate.originals()
        report['snapshot'] = inv.verify(r.SNAPSHOT, args.source_root)
        # Preserve the venv launcher path; resolving its symlink selects system Python.
        report['tests'] = test_copy(output, args.python.absolute())
        if args.mode == 'handoff':
            from compare_replay import compare_batch
            report['comparison'] = compare_batch(args.batch.resolve(), args.bundle.resolve())
            r.require(args.numerical_evidence is not None, 'Missing numerical evidence directory')
            evidence = args.numerical_evidence.resolve(strict=True)
            r.require(r.read(evidence / 'report.json') == report['comparison'], 'Saved comparison report differs')
            files = r.tree(evidence)
            r.require(set(files) == {'report.json', 'metric_replicates.csv', 'all31_ranks.csv', 'main23_ranks.csv'},
                      'Unexpected numerical evidence files')
            report['numerical_evidence_files'] = files
            report['retained_unsuccessful_attempts'] = {}
            for name in ('seven_metric_replay_20261011', 'seven_metric_replay_20261011_r2'):
                path = r.HERE / 'runs' / name
                r.require((path / 'FAILED.json').is_file() and not (path / 'COMPLETE.json').exists(),
                          'Prior partial attempt must not be promoted')
                report['retained_unsuccessful_attempts'][name] = {
                    'failure_record': r.record(path / 'FAILED.json'),
                    'request': r.record(path / 'REQUEST.json'),
                    'partial_resamples': len(list(path.glob('*/*/replicates/*.json'))),
                    'not_used_in_accepted_batch': args.batch.resolve() != path.resolve()}
            manifest = r.verify_bundle(args.bundle)
            before = r.record(args.bundle / 'MANIFEST.json')
            with tempfile.TemporaryDirectory(prefix='moved-evaluation-') as temp:
                dest = Path(temp) / 'bundle with spaces'; shutil.copytree(args.bundle, dest)
                r.verify_bundle(dest)
                # Inputs and run verification use explicit relocated bundle paths.
                for case in manifest['cases']:
                    for v in case['variants']:
                        r.verify_branch(args.batch / case['case'] / v['id'], dest)
            report['relocated_bundle_and_63_runs_verified'] = True
            r.check(args.bundle / 'MANIFEST.json', before)
        report['current_code'] = {p.relative_to(r.HERE).as_posix(): r.record(p)
                                  for p in sorted(r.HERE.glob('*.py'))}
        report['test_code'] = {p.name: r.record(p) for p in sorted((r.HERE / 'tests').glob('*.py'))}
        report['documentation'] = {name: r.record(r.HERE / name)
                                    for name in ('STUDY_REPLAY.md', 'AUDIT_REPLAY.md')}
        r.require(validate.outside() == prior['outside_files'], 'Accepted modules changed during validation')
        report['status'] = 'passed'; r.write(output / 'report.json', report)
    except BaseException as error:
        r.write(output / 'FAILED.json', {'error': repr(error)}); raise


if __name__ == '__main__': main()
