"""Local fresh-UMAP handoff evidence; never a claim of independent external audit."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import study as s
from bootstrap import r
import compare

PREFLIGHT_SHA = '8360f5c47389ce2528d3e710dd8690fbf205fad344e3a2e305d63094c0f9b2f2'


def baseline():
    study = s.HERE.parents[1]
    file = s.HERE / 'validation/preflight_20261011/report.json'
    r.require(r.record(file)['sha256'] == PREFLIGHT_SHA, 'Overnight preflight changed')
    prior = r.read(file)
    for name, rec in prior['outside_files'].items():
        path = study / name
        if 'symlink' in rec: r.require(path.is_symlink() and str(path.resolve()) == rec['symlink'], 'Prior link changed')
        else: r.check(path, rec)
    for name, rec in prior['integration_module'].items(): r.check(study / 'benchmarks/integration_evaluation' / name, rec)
    indexes = {}
    for name in ('source_census.json', 'curated_sources.json'):
        records = r.read(study / 'inventory_20261010/evidence' / name)['files']
        for row in records:
            if row.get('sha256'): r.check(study.parent / row['path'], {k: row[k] for k in ('bytes', 'sha256')})
        indexes[name] = sum(bool(row.get('sha256')) for row in records)
    return {'preflight_sha256': PREFLIGHT_SHA, 'prior_study_records': len(prior['outside_files']),
            'prior_integration_records': len(prior['integration_module']), 'original_indexes': indexes}


def invoke(script, batch, bundle, python):
    command = [str(python), '-B', str(script), 'verify-run', '--output', str(batch), '--bundle', str(bundle)]
    result = subprocess.run(command, text=True, capture_output=True)
    return {'command': command, 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}


def negative_cli(root, batch, bundle, python):
    parent = r.read(batch / 'REQUEST.json'); original = r.read(batch / 'BATCH_MANIFEST.json'); result = {}
    for name in ('duplicate_63', 'one_duplicate', 'missing', 'extra', 'foreign_case', 'foreign_variant', 'unsafe_path'):
        report = copy.deepcopy(original)
        if name == 'duplicate_63': report['branches'] = [report['branches'][0]] * 63
        elif name == 'one_duplicate': report['branches'][-1] = report['branches'][0]
        elif name == 'missing': report['branches'].pop()
        elif name == 'extra': report['branches'].append(copy.deepcopy(report['branches'][0]))
        elif name == 'foreign_case': report['branches'][0]['case'] = 'unregistered'
        elif name == 'foreign_variant': report['branches'][0]['variant'] = 'v99'
        else: report['branches'][0]['case'] = '../../outside'
        target = r.reserve(root / name)
        for code, rec in parent['code_records'].items(): r.copy(batch / 'controller' / code, target / 'controller' / code, rec)
        r.write(target / 'REQUEST.json', parent); r.write(target / 'BATCH_MANIFEST.json', report)
        r.write(target / 'COMPLETE.json', {'status': 'complete', 'manifest': r.record(target / 'BATCH_MANIFEST.json')})
        before = r.tree(target); result[name] = invoke(s.HERE / 'study.py', target, bundle, python)
        r.require(result[name]['returncode'] != 0 and 'Branch grid differs' in result[name]['stderr'], 'Malformed real CLI accepted')
        r.require(r.tree(target) == before, 'Read-only CLI wrote files')
    target = r.reserve(root / 'incomplete')
    result['incomplete'] = invoke(s.HERE / 'study.py', target, bundle, python)
    r.require(result['incomplete']['returncode'] != 0 and 'Batch incomplete' in result['incomplete']['stderr'], 'Incomplete batch accepted')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('batch', 'bundle', 'snapshot', 'comparison', 'output', 'python'): p.add_argument('--' + key, type=Path, required=True)
    args = p.parse_args(); python = args.python.absolute()
    batch = args.batch.resolve(strict=True); bundle = args.bundle.resolve(strict=True); snapshot = args.snapshot.resolve(strict=True)
    r.require(args.output.resolve().is_relative_to(s.HERE / 'validation'), 'Evidence outside new module')
    out = r.reserve(args.output, [batch, bundle, snapshot])
    try:
        report = {'kind': 'local self-validation; independent external audit pending', 'before': baseline()}
        report['verify_run'] = s.verify_run(batch, bundle)
        manifest = s.inputs(bundle); parent = r.read(batch / 'REQUEST.json')
        requests = [r.read(batch / c['case'] / v['id'] / 'REQUEST.json')['code_records'] for c, v in s.branches(manifest, 'all')]
        report['identical_63_request_code_records'] = all(x == parent['code_records'] == s.codes() for x in requests)
        r.require(report['identical_63_request_code_records'], 'Mixed worker versions')
        report['logs'] = {str((batch / c['case'] / v['id'] / name).relative_to(batch)):
                         r.record(batch / c['case'] / v['id'] / name)
                         for c, v in s.branches(manifest, 'all') for name in ('stdout.log', 'stderr.log')}
        import numpy as np
        import pandas as pd
        import contract
        helpers = contract.pure_helpers(bundle / 'evaluator.py')
        coordinate_checks = []
        for case, variant in s.branches(manifest, 'all'):
            run = batch / case['case'] / variant['id']
            stem = helpers['safe_name'](variant['method'])
            coord = np.load(run / 'fit/umap/coordinates' / (stem + '.npy'), allow_pickle=False)
            table = pd.read_csv(run / 'fit/umap/coordinates' / (stem + '.csv.gz'), float_precision='round_trip')
            r.require(list(table.columns) == ['row_id', 'UMAP1', 'UMAP2'] and
                      np.array_equal(table.row_id.to_numpy(), np.arange(len(coord))) and
                      np.array_equal(table[['UMAP1', 'UMAP2']].to_numpy(dtype=np.float32), coord),
                      'Fresh coordinate CSV does not round-trip to NPY')
            r.require(not any((run / '.cache/evaluation').rglob('*')), 'Unexpected metric cache payload')
            log = (run / 'stdout.log').read_text()
            r.require(log.count('common-protocol UMAP: ') == 1 and 'Fresh UMAP ' in log, 'Fresh fitting log missing')
            coordinate_checks.append({'case': case['case'], 'variant': variant['id'], 'rows': len(coord),
                                      'csv_float32_roundtrip_equal': True, 'empty_metric_cache': True})
        report['coordinate_payload_checks'] = coordinate_checks
        report['comparison'] = r.read(args.comparison / 'report.json')
        r.require(report['comparison']['batch_manifest'] == r.record(batch / 'BATCH_MANIFEST.json'), 'Comparison binding differs')
        with tempfile.TemporaryDirectory(prefix='umap-handoff-') as temp:
            temp = Path(temp).resolve(); module = temp / 'module'
            shutil.copytree(s.HERE, module, ignore=shutil.ignore_patterns('runs', 'validation', '__pycache__', '.cache'))
            command = [str(python), '-B', '-m', 'unittest', 'discover', '-s', str(module / 'tests'), '-v']
            tests = subprocess.run(command, cwd=module, text=True, capture_output=True)
            with (out / 'tests.log').open('x') as f: f.write(tests.stdout + tests.stderr)
            r.require(tests.returncode == 0, 'Temporary-copy tests failed')
            report['tests'] = {'command': command, 'returncode': tests.returncode, 'log': r.record(out / 'tests.log')}
            report['negative_cli'] = negative_cli(temp / 'negative', batch, bundle, python)
            moved = temp / 'input bundle with spaces'; shutil.copytree(bundle, moved)
            report['moved_input_cli'] = invoke(module / 'study.py', batch, moved, python)
            r.require(report['moved_input_cli']['returncode'] == 0, 'Moved input verification failed')
            recomputed = compare.compare(batch, bundle, snapshot, temp / 'recomputed_comparison')
            r.require(recomputed == report['comparison'], 'Comparison report not reproducible')
            for file in args.comparison.iterdir():
                if file.is_file(): r.require(r.record(file) == r.record(temp / 'recomputed_comparison' / file.name), 'Comparison table differs')
            report['comparison_reproduced_exactly'] = True
            probe = subprocess.run([str(python), '-B', str(module / 'probe.py'), '--bundle', str(bundle)],
                                   env=r.environment(temp / 'probe'), text=True, capture_output=True)
            with (out / 'probe.log').open('x') as f: f.write(probe.stdout + probe.stderr)
            r.require(probe.returncode == 0, 'Runtime probe failed')
            current = json.loads(probe.stdout)
            for case, variant in s.branches(manifest, 'all'):
                r.require(current == r.read(batch / case['case'] / variant['id'] / 'ENVIRONMENT_BEFORE.json'), 'Current runtime differs')
            report['current_runtime'] = current
        report['after'] = baseline(); r.require(report['after'] == report['before'], 'Previous work changed')
        report['current_files'] = {p.relative_to(s.HERE).as_posix(): r.record(p) for p in sorted(s.HERE.rglob('*'))
                                  if p.is_file() and p.suffix in ('.py', '.md') and not any(x in p.parts for x in
                                    ('runs', 'validation', '__pycache__', '.cache'))}
        report['status'] = 'passed'; r.write(out / 'report.json', report)
    except BaseException as error:
        r.write(out / 'FAILED.json', {'error': repr(error)}); raise


if __name__ == '__main__': main()
