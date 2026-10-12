"""Read-only proposal review, small tests, and existing descriptor check. No deletion."""
import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import prepare_manifest as p


def lines(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def verify(directory):
    directory = p.check_components(directory)
    complete, _ = p.read_json(directory / 'COMPLETE.json')
    p.require(complete['status'] == 'proposal_complete_not_deleted', 'Proposal incomplete')
    p.require({x.name for x in directory.iterdir()} == set(complete['files']) | {'COMPLETE.json'}, 'Unexpected proposal payload')
    for name, rec in complete['files'].items():
        p.hash_file(p.member(directory, name), rec)
    report, _ = p.read_json(directory / 'report.json')
    p.require(report['status'] == 'proposal_ready_for_independent_review' and report['deletion_authorized'] is False and report['deleted_files'] == 0, 'Not a review-only proposal')
    before = {r['relative_to_module']: r['stat'] for r in lines(directory / 'before_metadata.jsonl.gz')}
    p.require(len(before) == report['module_files'] and p.census(p.MODULE) == before, 'Module metadata changed since proposal')
    candidates, retained, counts, logical, allocated, modes = set(), set(), Counter(), Counter(), Counter(), Counter()
    for r in lines(directory / 'candidates.jsonl.gz'):
        target = p.member(p.MODULE, r['relative_to_module'])
        p.require(str(target) == r['path'] and r['relative_to_module'] not in candidates, 'Candidate path duplicate or different')
        p.require(p.regular(target) == r['stat'] == before[r['relative_to_module']], 'Candidate metadata changed')
        canonical = p.member(p.SOURCE, str(Path(r['canonical']['path']).relative_to(p.SOURCE)))
        p.require(p.regular(canonical) == r['canonical']['stat'], 'Canonical metadata changed')
        p.require(r['category'] == p.category(r['origin'], r['payload']), 'Category differs')
        p.require(r['currently_authorized_action'] == 'read_only_review', 'Unauthorized action')
        if r['category'].startswith('B') or r['payload'] in ('ledger.json', 'predictions.npz'):
            p.require(r['current_bytes_equal_canonical'] is True and all(r[k] == r['canonical'][k] for k in ('bytes', 'sha256')), 'Duplicate relation differs')
        p.key_for(r['task_key'])
        candidates.add(r['relative_to_module'])
        counts[r['category']] += 1
        logical[r['category']] += r['bytes']
        allocated[r['category']] += r['stat']['allocated_bytes']
        modes[r['recovery']] += 1
    for r in lines(directory / 'retained.jsonl.gz'):
        rel = r['relative_to_module']
        p.require(rel not in retained and rel not in candidates, 'Retained set overlap/duplicate')
        p.require(r['stat'] == before[rel] and r['path'] == str(p.member(p.MODULE, rel)), 'Retained record differs')
        retained.add(rel)
    p.check_counts(counts)
    p.require(candidates | retained == set(before), 'Disposition inventory incomplete')
    p.require(dict(counts) == report['counts'] and dict(logical) == report['logical_bytes'] and dict(allocated) == report['allocated_bytes'], 'Totals differ')
    p.require(len(candidates) == report['candidate_count'] and len(retained) == report['retained_files'], 'Record counts differ')
    for name, rec in report['tool_code'].items():
        p.hash_file(p.HERE / name, rec)
    return dict(status='proposal_consistency_verified_not_deletion_authority', candidates=len(candidates), retained=len(retained), recovery_modes=dict(modes),
                payload_content_rehashed_here=False, content_evidence='prepare_manifest live pair hashes; current validation rereads small sealed indices and file metadata only')


def checks(directory, output):
    output = p.check_components(output)
    p.require(output.parent == p.HERE and output.name.startswith('checks_'), 'New checks directory required')
    output.mkdir(exist_ok=False)
    try:
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        commands = [
            ('proposal_tests', [sys.executable, '-B', '-m', 'unittest', '-v', 'test_prepare_manifest.py'], p.HERE),
            ('existing_58_tests', [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests', '-v'], p.MODULE),
            ('check_binding', [sys.executable, '-B', str(p.MODULE / 'combined.py'), 'check-binding',
                '--lockdir', str(p.MODULE / 'locks/full_20261011'), '--source-root', str(p.SOURCE),
                '--original-run', str(p.MODULE / 'runs/full_20261011'), '--continuation', str(p.MODULE / 'runs/continuation_20261011'),
                '--run', str(p.MODULE / 'validation/combined_full_20261011')], p.HERE),
        ]
        runs = []
        for name, cmd, cwd in commands:
            with (output / (name + '.stdout')).open('xb') as out, (output / (name + '.stderr')).open('xb') as err:
                rc = subprocess.run(cmd, cwd=cwd, env=env, stdout=out, stderr=err).returncode
            runs.append(dict(name=name, command=cmd, cwd=str(cwd), rc=rc))
            p.require(rc == 0, 'Check failed: ' + name)
        result = verify(directory)
        p.write_new(output / 'report.json', dict(status='passed', checks=runs, manifest=str(directory), verification=result,
                                               deleted_files=0, large_files_copied=0, when=p.now()))
        p.write_new(output / 'COMPLETE.json', dict(status='complete', files={x.name: p.hash_file(x)[0] for x in sorted(output.iterdir())}))
        print(json.dumps(result), flush=True)
    except BaseException as exc:
        p.write_new(output / 'FAILED.json', dict(error=repr(exc), deleted_files=0))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--checks-output', type=Path)
    args = parser.parse_args()
    if args.checks_output:
        checks(args.manifest, args.checks_output)
    else:
        print(json.dumps(verify(args.manifest)))
