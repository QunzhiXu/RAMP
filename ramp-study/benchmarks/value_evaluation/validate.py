"""Create-only local handoff evidence. This is not an independent external audit."""
import argparse
import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import contract as c
import inventory as i

PREFLIGHT_SHA = '8360f5c47389ce2528d3e710dd8690fbf205fad344e3a2e305d63094c0f9b2f2'


def baselines():
    study = i.HERE.parents[1]
    preflight = study / 'benchmarks/umap_reproduction/validation/preflight_20261011/report.json'
    i.require(i.record(preflight)['sha256'] == PREFLIGHT_SHA, 'Overnight preflight changed')
    prior = i.read(preflight)
    for name, rec in prior['outside_files'].items():
        path = study / name
        if 'symlink' in rec: i.require(path.is_symlink() and str(path.resolve()) == rec['symlink'], 'Old symlink changed')
        else: i.require(i.record(path) == rec, 'Old study file changed: ' + name)
    for name, rec in prior['integration_module'].items():
        i.require(i.record(study / 'benchmarks/integration_evaluation' / name) == rec, 'Previous evaluation changed')
    indexes = {}
    for name in ('source_census.json', 'curated_sources.json'):
        entries = i.read(study / 'inventory_20261010/evidence' / name)['files']
        for row in entries:
            if row.get('sha256'): i.require(i.record(study.parent / row['path']) ==
                                           {key: row[key] for key in ('bytes', 'sha256')}, 'Original source changed')
        indexes[name] = sum(bool(row.get('sha256')) for row in entries)
    return {'preflight_sha256': PREFLIGHT_SHA, 'preserved_study_records': len(prior['outside_files']),
            'preserved_integration_records': len(prior['integration_module']), 'original_indexes': indexes}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); source = args.source_root.resolve(strict=True)
    i.require(args.output.resolve().is_relative_to(i.HERE / 'validation'), 'Evidence outside module')
    out = i.reserve(args.output, [source]); snapshot = i.HERE / 'frozen_20261011_r2'
    try:
        before = baselines()
        report = {'kind': 'local self-validation; external audit pending', 'before': before,
                  'snapshot': i.verify(snapshot, c.CATALOG_SHA, source), 'contract': c.inspect(snapshot, source),
                  'environment': {'python': sys.version, 'packages': {x: importlib.metadata.version(x)
                    for x in ('numpy', 'pandas', 'scipy', 'scikit-learn')}}}
        with tempfile.TemporaryDirectory(prefix='value-contract-') as temp:
            temp = Path(temp).resolve(); module = temp / 'module'
            shutil.copytree(i.HERE, module, ignore=shutil.ignore_patterns('runs', 'validation', '__pycache__', '.cache', 'frozen_20261011'))
            cmd = [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', str(module / 'tests'), '-v']
            result = subprocess.run(cmd, cwd=module, text=True, capture_output=True,
                                    env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
            with (out / 'tests.log').open('x') as f: f.write(result.stdout + result.stderr)
            i.require(result.returncode == 0, 'Temporary-copy tests failed')
            report['tests'] = {'command': cmd, 'returncode': result.returncode, 'log': i.record(out / 'tests.log')}
            moved = temp / 'moved snapshot with spaces'; shutil.copytree(snapshot, moved)
            report['moved_without_originals'] = c.inspect(moved)
            rebuilt = temp / 'rebuilt'; report['rebuild'] = i.freeze(source, rebuilt)
            expected = {x.relative_to(snapshot).as_posix(): i.record(x) for x in snapshot.rglob('*') if x.is_file()}
            actual = {x.relative_to(rebuilt).as_posix(): i.record(x) for x in rebuilt.rglob('*') if x.is_file()}
            i.require(expected == actual, 'Temporary rebuilt snapshot differs')
            report['byte_identical_rebuilt_files'] = len(actual)
        report['after'] = baselines(); i.require(report['before'] == report['after'], 'Previous work changed')
        report['current_code'] = {p.relative_to(i.HERE).as_posix(): i.record(p) for p in sorted(i.HERE.rglob('*'))
             if p.is_file() and p.suffix in ('.py', '.md') and not any(x in p.parts for x in
                ('frozen_20261011', 'frozen_20261011_r2', 'validation', '__pycache__', 'runs'))}
        report['status'] = 'passed'; i.write(out / 'report.json', report)
    except BaseException as error:
        i.write(out / 'FAILED.json', {'error': repr(error)}); raise


if __name__ == '__main__': main()
