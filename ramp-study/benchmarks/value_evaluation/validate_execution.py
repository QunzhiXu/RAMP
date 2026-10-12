"""Local handoff for real value evaluation; preserves all prior stage evidence."""
import argparse
from collections import Counter
import copy
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import execution as e
import inventory as i
import contract as c
import compare_execution as comp
import validate as phase1


def command(script, action, bundle, snapshot, run=None):
    argv = [sys.executable, '-B', str(script), action, '--bundle', str(bundle), '--snapshot', str(snapshot)]
    if run is not None: argv.extend(['--output', str(run)])
    r = subprocess.run(argv, text=True, capture_output=True)
    return {'argv': argv, 'returncode': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'bundle', 'snapshot', 'source-root', 'comparison', 'output'): p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args(); run = a.run.resolve(strict=True); bundle = a.bundle.resolve(strict=True); snapshot = a.snapshot.resolve(strict=True)
    i.require(a.output.resolve().is_relative_to(e.HERE / 'validation'), 'Evidence outside module')
    out = i.reserve(a.output, [run, bundle, snapshot, a.source_root])
    try:
        old_report = e.HERE / 'validation/handoff_20261011/report.json'; old = i.read(old_report)
        for name, rec in old['current_code'].items(): i.require(i.record(e.HERE / name) == rec, 'Source-stage code changed')
        report = {'kind': 'local self-validation, external audit pending', 'source_stage_report': i.record(old_report),
                  'before': phase1.baselines(), 'verification': e.verify_run(run, bundle, snapshot),
                  'comparison': i.read(a.comparison / 'report.json'),
                  'input_roles': dict(Counter(x['role'] for x in e.verify_bundle(bundle, snapshot)['files'].values()))}
        records_before = e.tree(run); bundle_before = e.tree(bundle)
        gzip_checks = {}
        for row in report['comparison']['files']:
            if row['path'].endswith('.csv.gz'):
                with gzip.open(run / 'runtime/tree' / row['path'], 'rb') as x, gzip.open(a.source_root / row['path'], 'rb') as y:
                    equal = x.read() == y.read()
                i.require(equal, 'Uncompressed CSV bytes differ'); gzip_checks[row['path']] = equal
        report['gzip_payload_equality'] = gzip_checks
        with tempfile.TemporaryDirectory(prefix='value-execution-audit-') as temp:
            temp = Path(temp).resolve(); module = temp / 'detached study/benchmarks/value_evaluation'
            shutil.copytree(e.HERE, module, ignore=shutil.ignore_patterns('runs', 'validation', '__pycache__', '.cache', 'frozen_20261011'))
            # The pinned preflight is an explicit trust dependency, not a matrix
            # source. Supply it in the detached study layout; no old arrays copied.
            trust = module.parents[1] / 'benchmarks/umap_reproduction/validation/preflight_20261011/report.json'
            trust.parent.mkdir(parents=True); shutil.copyfile(e.STUDY / 'benchmarks/umap_reproduction/validation/preflight_20261011/report.json', trust)
            moved = temp / 'input bundle with spaces'; shutil.copytree(bundle, moved)
            moved_snapshot = module / 'frozen_20261011_r2'
            argv = [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', str(module / 'tests'), '-v']
            tests = subprocess.run(argv, cwd=module, text=True, capture_output=True)
            with (out / 'tests.log').open('x') as f: f.write(tests.stdout + tests.stderr)
            i.require(tests.returncode == 0, 'Scratch-copy tests failed')
            report['tests'] = {'argv': argv, 'returncode': tests.returncode, 'log': i.record(out / 'tests.log')}
            report['moved_bundle_and_controller'] = command(module / 'execution.py', 'verify-run', moved, moved_snapshot, run)
            i.require(report['moved_bundle_and_controller']['returncode'] == 0, 'Detached verification failed')
            negative = {}
            for name in ('changed_binding', 'changed_role', 'changed_catalog'):
                fake = temp / name; fake.mkdir(); manifest = copy.deepcopy(i.read(bundle / 'MANIFEST.json'))
                key = next(iter(manifest['files']))
                if name == 'changed_binding': manifest['files'][key]['path'] = '../outside'
                elif name == 'changed_role': manifest['files'][key]['role'] = 'historical_output'
                else: manifest['catalog_sha256'] = '0' * 64
                i.write(fake / 'MANIFEST.json', manifest)
                i.write(fake / 'COMPLETE.json', {'status': 'complete', 'manifest': i.record(fake / 'MANIFEST.json')})
                negative[name] = command(module / 'execution.py', 'verify', fake, moved_snapshot)
                i.require(negative[name]['returncode'] != 0 and 'Bundle plan differs' in negative[name]['stderr'], 'Rehashed changed binding accepted')
            report['negative_cli'] = negative
            fake_run = temp / 'unknown controller'; shutil.copytree(run, fake_run)
            request = i.read(fake_run / 'REQUEST.json'); request['code_records']['execution.py']['sha256'] = '0' * 64
            (fake_run / 'REQUEST.json').unlink(); i.write(fake_run / 'REQUEST.json', request)
            (fake_run / 'COMPLETE.json').unlink(); (fake_run / 'RUN_MANIFEST.json').unlink()
            i.write(fake_run / 'RUN_MANIFEST.json', {'files': e.tree(fake_run)})
            i.write(fake_run / 'COMPLETE.json', {'status': 'complete', 'manifest': i.record(fake_run / 'RUN_MANIFEST.json')})
            report['unknown_controller_cli'] = command(module / 'execution.py', 'verify-run', moved, moved_snapshot, fake_run)
            i.require(report['unknown_controller_cli']['returncode'] != 0 and 'Unknown controller/input' in report['unknown_controller_cli']['stderr'], 'Unknown code accepted')
            replayed = comp.compare(run, bundle, snapshot, a.source_root, temp / 'comparison')
            i.require(replayed == report['comparison'], 'Numerical comparison not reproducible')
            report['current_child_runtime'] = e.probe(Path(sys.executable), module, out, 'current_runtime')
            i.require(report['current_child_runtime'] == i.read(run / 'ENVIRONMENT_BEFORE.json'), 'Current child runtime differs')
        i.require(e.tree(run) == records_before and e.tree(bundle) == bundle_before, 'Read-only validation changed run/bundle')
        report['after'] = phase1.baselines(); i.require(report['before'] == report['after'], 'Old baseline changed')
        report['current_files'] = {p.relative_to(e.HERE).as_posix(): i.record(p) for p in sorted(e.HERE.rglob('*')) if
            p.is_file() and p.suffix in ('.py', '.md') and not any(x in p.parts for x in
                ('runs', 'validation', '__pycache__', '.cache', 'frozen_20261011', 'frozen_20261011_r2'))}
        report['status'] = 'passed'; i.write(out / 'report.json', report)
    except BaseException as error:
        i.write(out / 'FAILED.json', {'error': repr(error)}); raise


if __name__ == '__main__': main()
