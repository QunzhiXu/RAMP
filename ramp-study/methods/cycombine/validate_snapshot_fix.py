"""Narrow snapshot-safety regression evidence. No R or real cohort fitting."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from unittest.mock import patch

import inventory as inv
from validate import unchanged_prior_work

BASELINE_REPORT = '56972dcf61c8f3632e7070557da9aca8c308a9b4d2667ae272e075db31a44af6'
BASELINE_GITIGNORE = '3f2e3ee39c10a3d330a440ce973caa7004c016213f0331ff33992a6e8978ce31'
# Recorded immediately before this fix, after verifying the six-file old report.
UNCHANGED_DEFINITIONS = {
    'sha': 'c36f795b9a0ac4d69ce43afc8e0d8b77267d9ef093edfabd5e4bff7eb9162e84',
    'read_json': '365d3e66bd292d63054b32562a90bc1556144a6bc3a7db8170d62907dd6d4aea',
    'create_json': 'b15515d7bad65066c4f2dbfb9afdbef6931c8f1c1cf62a5a8b51bb90fe66f5be',
    'within': '73251c120761caa46acf151828206a0560b40c7b4effbcda65fba590112eda61',
    'reserve': '2a5b8f59b9182f5c624ae463b61b7f60da4f4bae6dc3197019098039dd8df55a',
    'record': '0d35e1f4bc02a93d33dbe1e1bcfb6bb68cbe2d4f1db5da92b30b3651319731d9',
    'check': '23fe8cfd2c972abf1d51efc96d62dd0a3d96159891aaff12ea600b921c178909',
    'copy_new': '57324de21b549bebf14fe255819e113039cded775fe4ce94123ca2fc30071c71',
    'field': '171e13718e0eea9d75fb97dd772deca032bc32fe5839ec72a5dc8a431bb063a0',
    'original_relative': '4d02a095363a61a37c2bdb28de8e4a190ca5924714f4485028310a4e4b8553a9',
    'CatalogBuilder': '569ec01ebe3354e2d3a8610e9c7663a58efebddf1ca68c3e4e913514932dd26e',
    'verify_sources': '4bec0260e17321c64653fcc75bebee0e76d8cd974661d2b17852fd7f0e20c4bc',
    'verify_wrappers': '9d00e4bfd35c79b107f1c486f0e58e84197bbd98b1e1b7cf7f161598b8cc5e44',
    'table': 'a152c92f1a19fc20d6637573751b65856064766e0d283c83dd6a1fd70b5d1a3b',
    'verify_results': '2a2ed37fca052a3921f9543e529b1b773653f0575c4848f29899aabb36107eb5',
    'main': '802e690659d5b2526483d37f32db2a336894e5c4bb89cf1635f7a38c9d5aa513',
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    source = args.source_root.resolve(strict=True)
    if not args.output.resolve().is_relative_to(inv.HERE / 'validation'):
        parser.error('Use a new directory inside this module/validation')
    output = inv.reserve(args.output, (source, inv.SNAPSHOT, inv.REGISTRY))
    report = {'utc': datetime.now(timezone.utc).isoformat(), 'scope': 'Snapshot completion safety only',
              'R_tests_rerun': False, 'cohort_fit_performed': False}
    try:
        old_path = inv.HERE / 'validation/handoff_20261010/report.json'
        if inv.sha(old_path) != BASELINE_REPORT: raise ValueError('Old handoff report changed')
        old = inv.read_json(old_path)
        report['baseline_report_sha256'] = BASELINE_REPORT
        report['unchanged_old_files'] = {}
        for name, digest in old['current_code_sha256'].items():
            if name not in ('inventory.py', 'tests/test_inventory.py', 'README.md'):
                if inv.sha(inv.HERE / name) != digest: raise ValueError('Unexpected old code edit: ' + name)
                report['unchanged_old_files'][name] = digest
        definitions = {node.name: hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
                       for node in ast.parse((inv.HERE / 'inventory.py').read_text()).body
                       if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        for name, digest in UNCHANGED_DEFINITIONS.items():
            if definitions[name] != digest: raise ValueError('Unexpected function change: ' + name)
        report['unchanged_definitions'] = UNCHANGED_DEFINITIONS
        ignore = (inv.STUDY / '.gitignore').read_text()
        if ignore.count('.DS_Store\n') != 1 or hashlib.sha256(ignore.replace('.DS_Store\n', '').encode()).hexdigest() != BASELINE_GITIGNORE:
            raise ValueError('More than the requested .DS_Store ignore change')
        report['gitignore'] = {'old_sha256': BASELINE_GITIGNORE, 'new_sha256': inv.sha(inv.STUDY / '.gitignore'), 'only_added': '.DS_Store'}
        report['original_checks'] = inv.verify_sources(source)
        report['result_checks'] = inv.verify_results(source)
        if inv.sha(inv.SNAPSHOT / 'catalog.json') != old['snapshot']['catalog_sha256']:
            raise ValueError('Existing catalog changed')
        report['snapshot_catalog_sha256'] = inv.sha(inv.SNAPSHOT / 'catalog.json')
        with tempfile.TemporaryDirectory(prefix='cycombine_snapshot_fix_') as name:
            tmp = Path(name).resolve()
            copied = tmp / 'module'
            shutil.copytree(inv.HERE, copied, ignore=shutil.ignore_patterns('validation', '__pycache__', '.DS_Store'))
            env = os.environ.copy(); env.pop('PYTHONPATH', None); env['PYTHONDONTWRITEBYTECODE'] = '1'
            result = subprocess.run([str(args.python.absolute()), '-B', '-m', 'unittest', 'discover', '-s', str(copied / 'tests'), '-v'],
                                    cwd=tmp, env=env, capture_output=True, text=True)
            report['tests'] = {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr,
                               'temporary_copy': True, 'default_macOS_TMPDIR_preserved': True}
            if result.returncode: raise RuntimeError('Regression tests failed')
            report['tests_passed'] = sum(map(int, re.findall(r'Ran (\d+) tests? in', result.stderr)))
            # Exercise the actual collection/copy/publication chain, but only text
            # snapshots in a disposable directory. No provider extraction or R fit.
            rebuilt = inv.freeze(source, tmp / 'rebuilt')
            inv.verify_local(rebuilt)
            if inv.sha(rebuilt / 'catalog.json') != report['snapshot_catalog_sha256']:
                raise ValueError('Rebuilt catalog changed')
            all_files = [p for p in inv.SNAPSHOT.rglob('*') if p.is_file()]
            for path in all_files:
                if inv.sha(path) != inv.sha(rebuilt / path.relative_to(inv.SNAPSHOT)):
                    raise ValueError('Rebuilt snapshot differs')
            report['temporary_real_freeze'] = {'files_byte_identical': len(all_files), 'completed': True}
            # Reproduce the audit's exact timing with a PRIVATE registry copy:
            # real collect succeeds, registry changes, then real publication fails.
            registry = tmp / 'registry.json'; shutil.copyfile(inv.REGISTRY, registry)
            original_collect = inv.collect
            def changed_after_collect(*a, **kw):
                catalog = original_collect(*a, **kw)
                registry.write_text('{}')
                return catalog
            failed = tmp / 'interrupted'
            with patch.object(inv, 'REGISTRY', registry), patch.object(inv, 'collect', side_effect=changed_after_collect):
                try:
                    inv.freeze(source, failed)
                except ValueError:
                    pass
                else:
                    raise ValueError('Mutated registry was accepted')
            if (failed / 'COMPLETE.json').exists() or not (failed / 'FAILED.json').exists():
                raise ValueError('Failed snapshot completion status is incorrect')
            report['audit_reproduction'] = {'real_collection_and_copy': True, 'only_private_registry_mutated': True,
                                            'complete_absent': True, 'failure': inv.read_json(failed / 'FAILED.json')}
        report['prior_work'] = unchanged_prior_work(source)
        report['postcheck'] = inv.verify_sources(source)
        if inv.sha(old_path) != BASELINE_REPORT: raise ValueError('Old report changed during validation')
        report['current_files'] = {name: inv.sha(inv.HERE / name) for name in
                                   ('inventory.py', 'tests/test_inventory.py', 'README.md',
                                    'validate_snapshot_fix.py', 'AUDIT_SNAPSHOT_FIX.md')}
        report['status'] = 'passed'
        inv.create_json(output / 'report.json', report)
    except BaseException as error:
        inv.create_json(output / 'FAILED.json', {**report, 'error': repr(error)})
        raise
    print(output / 'report.json')


if __name__ == '__main__': main()
