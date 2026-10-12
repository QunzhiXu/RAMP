"""Small-evidence regression harness; never copies scientific payloads or deletes project data."""
import argparse
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import prepare_manifest as p

HERE = p.HERE
OLD_FILES = ('prepare_manifest.py', 'test_prepare_manifest.py', 'validate_manifest.py')
MANIFEST = HERE / 'manifest_20261011_r2'
PREFLIGHT = HERE / 'p2_binding_preflight_20261011'


def command(cmd, directory, label, expected):
    start = time.monotonic()
    with (directory / (label + '.stdout')).open('xb') as out, (directory / (label + '.stderr')).open('xb') as err:
        rc = subprocess.run(cmd, cwd=HERE, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), stdout=out, stderr=err).returncode
    p.require((rc == 0) == expected, 'Unexpected CLI result: ' + label)
    return dict(command=cmd, returncode=rc, seconds=time.monotonic()-start)


def read_lines(path):
    with gzip.open(path, 'rt') as f:
        yield from map(json.loads, f)


def copy_small(src, dst, limit=8*1024*1024):
    p.require(p.regular(src)['bytes'] <= limit, 'Refuse large copy')
    with Path(src).open('rb') as a, Path(dst).open('xb') as b:
        shutil.copyfileobj(a, b)
    p.require(p.hash_file(src)[0] == p.hash_file(dst)[0], 'Small copy differs')


def preflight():
    PREFLIGHT.mkdir(exist_ok=False)
    (PREFLIGHT / 'controller').mkdir()
    before = {str(x.relative_to(HERE)): p.hash_file(x)[0] for x in sorted(HERE.rglob('*'))
              if x.is_file() and not x.is_relative_to(PREFLIGHT)}
    for name in OLD_FILES:
        copy_small(HERE/name, PREFLIGHT/'controller'/name)
    audit = Path('/private/tmp/independent_cleanup_manifest_thg4s2bc/audit_result.json')
    if audit.exists():
        copy_small(audit, PREFLIGHT/'external_audit1.json', 65536)
    fixture = Path(tempfile.mkdtemp(prefix='cleanup_scope_regression_', dir='/private/tmp')).resolve()
    replacement_rel = 'runs/full_20261011/group_0000/runtime/src/trainability/heads.py'
    target = p.MODULE/replacement_rel
    canonical = p.SOURCE/'cross_platform_trainability_v2_20260919/src/trainability/heads.py'
    target_rec, target_stat, _ = p.hash_file(target)
    source_rec, source_stat, _ = p.hash_file(canonical)
    p.require(target_rec == source_rec, 'Counterexample files not identical')
    original = replacement = None
    with p.compressed_lines(fixture/'candidates.jsonl.gz') as emit:
        for r in read_lines(MANIFEST/'candidates.jsonl.gz'):
            if original is None and r['category'] == 'B2':
                original = dict(r)
                r = r | dict(path=str(target), relative_to_module=replacement_rel, **target_rec, stat=target_stat,
                    canonical=dict(path=str(canonical), **source_rec, stat=source_stat), sealed_target=target_rec)
                replacement = r
            emit(r)
    swapped = 0
    with p.compressed_lines(fixture/'retained.jsonl.gz') as emit:
        for r in read_lines(MANIFEST/'retained.jsonl.gz'):
            if r['relative_to_module'] == replacement_rel:
                r = {k: original[k] for k in ('path','relative_to_module','bytes','sha256','stat')}
                swapped += 1
            emit(r)
    p.require(swapped == 1, 'Counterexample retained swap differs')
    for name in ('before_metadata.jsonl.gz','REQUEST.json','external_audit1_plan.json'):
        copy_small(MANIFEST/name, fixture/name)
    report, _ = p.read_json(MANIFEST/'report.json')
    report['logical_bytes']['B2'] += replacement['bytes']-original['bytes']
    report['allocated_bytes']['B2'] += replacement['stat']['allocated_bytes']-original['stat']['allocated_bytes']
    p.write_new(fixture/'report.json', report)
    p.write_new(fixture/'COMPLETE.json', dict(status='proposal_complete_not_deleted', files={x.name:p.hash_file(x)[0] for x in sorted(fixture.iterdir())}))
    cmd = [sys.executable,'-B',str(HERE/'validate_manifest.py'),'--manifest',str(fixture)]
    result = command(cmd, PREFLIGHT, 'old_cli_accepts_code_as_prediction', True)
    old_code = {n:p.hash_file(HERE/n)[0] for n in OLD_FILES}
    p.write_new(PREFLIGHT/'report.json', dict(status='vulnerability_reproduced', old_code=old_code, prior_files=before,
        fixture=str(fixture), fixture_files={x.name:p.hash_file(x)[0] for x in sorted(fixture.iterdir())},
        fixture_bytes=sum(x.stat().st_size for x in fixture.iterdir()), old_cli=result,
        original_candidate=original, replacement_candidate=replacement, scientific_files_copied=0, deleted_project_files=0))
    p.write_new(PREFLIGHT/'COMPLETE.json', dict(status='complete', report=p.hash_file(PREFLIGHT/'report.json')[0]))
    print('Old CLI accepted the resealed heads.py counterexample; preflight retained', flush=True)


def validate(output):
    output = p.check_components(output)
    p.require(output.parent == HERE and output.name.startswith('p2_binding_checks_'), 'Invalid evidence path')
    output.mkdir(exist_ok=False)
    try:
        prior, _ = p.read_json(PREFLIGHT/'report.json')
        fixture = Path(prior['fixture'])
        p.require(fixture.parent == Path('/private/tmp') and fixture.name.startswith('cleanup_scope_regression_'), 'Unknown temporary fixture')
        p.require({x.name:p.hash_file(x)[0] for x in fixture.iterdir()} == prior['fixture_files'], 'Negative fixture changed')
        cmd = [sys.executable,'-B',str(HERE/'validate_manifest.py'),'--manifest',str(fixture)]
        negative = command(cmd, output, 'new_cli_rejects_code_as_prediction', False)
        p.require('Candidate not allowed by fixed locks and routes' in (output/'new_cli_rejects_code_as_prediction.stderr').read_text(), 'Wrong rejection reason')
        positive = command([*cmd[:-1],str(MANIFEST)], output, 'real_manifest', True)
        tests = command([sys.executable,'-B','-m','unittest','-v','test_prepare_manifest.py','test_candidate_binding.py'], output, 'cleanup_tests', True)
        legacy = command([sys.executable,'-B','-m','unittest','discover','-s',str(p.MODULE/'tests'),'-v'], output, 'existing_tests', True)
        unchanged = 0
        for name, rec in prior['prior_files'].items():
            if name not in ('validate_manifest.py',):
                p.hash_file(HERE/name, rec); unchanged += 1
        p.require({x.name:p.hash_file(x)[0] for x in fixture.iterdir()} == prior['fixture_files'], 'CLI modified negative fixture')
        p.require(p.hash_file(HERE/'validate_manifest.py')[0] != prior['old_code']['validate_manifest.py'], 'No validator fix')
        # Only our exact, recorded temporary index fixture is removed; no payload data.
        fixture_bytes = sum(x.stat().st_size for x in fixture.iterdir())
        shutil.rmtree(fixture)
        files = ['validate_manifest.py','test_candidate_binding.py','validate_binding_fix.py']
        report = dict(status='passed', old_cli=prior['old_cli'], negative_cli=negative, real_cli=positive,
            tests=tests, existing_tests=legacy, old_files_unchanged=unchanged,
            current_code={n:p.hash_file(HERE/n)[0] for n in files},
            preflight=p.hash_file(PREFLIGHT/'report.json')[0],
            deleted_project_files=0, scientific_payload_copies=0,
            temporary_fixture=dict(path=str(fixture), removed=True, bytes=fixture_bytes),
            limits='No new large payload rehash; fixed provenance, actual CLI and metadata checks; no deletion authorization')
        p.write_new(output/'report.json', report)
        p.write_new(output/'COMPLETE.json', dict(status='complete', files={x.name:p.hash_file(x)[0] for x in sorted(output.iterdir())}))
        print(json.dumps(dict(status='passed', old_files_unchanged=unchanged, deleted_project_files=0)), flush=True)
    except BaseException as exc:
        p.write_new(output/'FAILED.json', dict(error=repr(exc), deleted_project_files=0))
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['preflight','validate'])
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.mode=='preflight': preflight()
    else: validate(args.output)
