"""Explicit evidence writer, not a fitting or metrics runner."""
import argparse
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import shutil

import contract as c
import inventory as inv

RAMP_REPORT = 'benchmarks/ramp_alignment/validation/six_execution_20261011/report.json'
RAMP_SHA = 'a3930de7fd4b24014faa9fd2b8d7bda1871e84d9760be2e9ec8bfa527f373cf8'


def outside():
    records = {}
    for folder, dirs, names in os.walk(c.STUDY):
        root = Path(folder)
        dirs[:] = sorted(d for d in dirs if not d.startswith('.') and d not in
                         ('__pycache__', 'cache', 'tmp', 'node_modules') and (root / d).resolve() != c.HERE)
        for name in sorted(names):
            if name.startswith('.') or name.endswith('.pyc'): continue
            p = root / name
            records[p.relative_to(c.STUDY).as_posix()] = {'symlink': str(p.resolve())} if p.is_symlink() else c.record(p)
        for name in dirs[:]:
            p = root / name
            if p.is_symlink():
                records[p.relative_to(c.STUDY).as_posix()] = {'symlink': str(p.resolve())}
                dirs.remove(name)
    return records


def originals():
    counts = {}
    for name in ('source_census.json', 'curated_sources.json'):
        rows = c.read(c.STUDY / 'inventory_20261010/evidence' / name)['files']
        for row in rows:
            if row.get('sha256'):
                inv.check(c.STUDY.parent / row['path'], {k: row[k] for k in ('bytes', 'sha256')})
        counts[name] = sum(bool(r.get('sha256')) for r in rows)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['preflight', 'handoff'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight', type=Path)
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--source-root', type=Path, required=True)
    args = parser.parse_args()
    c.require(args.output.resolve().is_relative_to(c.HERE / 'validation'), 'Evidence must stay in this module')
    output = inv.reserve(args.output, [args.source_root, c.REGISTRY])
    report = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'mode': args.mode}
    report['verification_environment'] = {'python': sys.version, 'executable': sys.executable,
                                          'packages': {p: importlib.metadata.version(p) for p in ('numpy', 'pandas', 'scikit-learn')}}
    try:
        c.require(c.record(c.STUDY / RAMP_REPORT)['sha256'] == RAMP_SHA, 'Accepted RAMP report changed')
        report['accepted_ramp_report_sha256'] = RAMP_SHA
        report['outside_files'] = outside()
        report['original_indexes'] = originals()
        if args.mode == 'handoff':
            c.require(args.preflight is not None and args.snapshot is not None, 'Missing preflight/snapshot')
            before = c.read(args.preflight)
            c.require(report['outside_files'] == before['outside_files'], 'Previously accepted files changed')
            report['preflight_sha256'] = c.record(args.preflight)['sha256']
            report['snapshot'] = inv.verify(args.snapshot, args.source_root)
            doc = c.read(args.snapshot / 'catalog.json')
            report['selection_check'] = doc['selection_check']
            with tempfile.TemporaryDirectory(prefix='evaluation-audit-') as temp:
                copied = Path(temp) / 'moved snapshot with spaces'
                shutil.copytree(args.snapshot, copied)
                report['moved_snapshot'] = inv.verify(copied)
                env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
                run = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', str(c.HERE / 'tests'), '-v'],
                                     cwd=c.HERE, env=env, text=True, capture_output=True)
                with (output / 'tests.log').open('x') as handle: handle.write(run.stdout + run.stderr)
                c.require(run.returncode == 0, 'Tests failed; see tests.log')
            report['tests_returncode'] = run.returncode
            report['catalog_sha256'] = c.record(args.snapshot / 'catalog.json')['sha256']
            report['current_code'] = {p.relative_to(c.HERE).as_posix(): c.record(p)
                                      for p in sorted(c.HERE.rglob('*')) if p.is_file() and
                                      p.suffix in ('.py', '.md') and not any(x in p.parts for x in ('validation', 'frozen_20261011', 'frozen_20261011_r2'))}
            c.require(outside() == report['outside_files'], 'Accepted files changed during validation')
        report['status'] = 'passed'
        inv.write(output / 'report.json', report)
    except BaseException as error:
        inv.write(output / 'FAILED.json', {'error': repr(error)})
        raise


if __name__ == '__main__':
    main()
