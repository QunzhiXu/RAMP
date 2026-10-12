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

LEGACY_R2_CODE = {
    'prepare_manifest.py': dict(bytes=18865, sha256='cb51ed0ae71580b947a663bb596dbdda690564f4c805a6c57c08923879c12517'),
    'test_prepare_manifest.py': dict(bytes=6902, sha256='229bcc95c016247cf48ee23469e190b97cfd8be4bbb80ec3daa05364c79b88c7'),
    'validate_manifest.py': dict(bytes=6178, sha256='146ff7996130e39c0c941b27b85714b10e65a9b64abe6a47daa9667612864e29'),
}
SMOKE_MANIFESTS = {
    'group_0000': dict(bytes=38777, sha256='4acaeed9d683fb57e5c7d44f18eac280c27da9711e51cbdbe7c53e580e4ec213'),
    'group_0858': dict(bytes=38759, sha256='4be4d9cd046ed4c49f7b05f2d1c993e55d00d0add1624773d8c09136abdfc993'),
}
BINDING_FIELDS = ('path','relative_to_module','category','origin','group','local_job','task_key','payload','group_lock','sealed_target')
CANDIDATE_FIELDS = set(BINDING_FIELDS) | {'affected_checks','bytes','canonical','current_bytes_equal_canonical',
    'currently_authorized_action','proposed_action','recovery','sha256','stat'}


def read_pinned(name, sha):
    value, rec = p.read_json(p.member(p.MODULE, name))
    p.require(rec['sha256'] == sha, 'Pinned anchor changed: ' + name)
    return value, rec


def check_tool_contract(directory, report):
    recorded = report['tool_code']
    if recorded == LEGACY_R2_CODE:
        for name, rec in recorded.items():
            path = p.HERE/'p2_binding_preflight_20261011/controller'/name if name == 'validate_manifest.py' else p.HERE/name
            p.hash_file(path, rec)
    else:
        current = {x.name:p.hash_file(x)[0] for x in sorted(p.HERE.glob('*.py'))}
        p.require(recorded == current, 'Unknown proposal tool code set')
    request, _ = p.read_json(directory/'REQUEST.json')
    p.require(request == dict(scope='disposal_proposal_only', code=recorded,
        started_utc=report['started_utc'], deletion_authorized=False), 'Proposal request differs')


def expected_candidates(report):
    """Rebuild scope from pinned provenance, never from candidate labels or totals."""
    anchors, values = {}, {}
    for name, sha in p.PINS.items():
        values[name], anchors[name] = read_pinned(name, sha)
    p.require(report['anchors'] == anchors, 'Reported anchors differ')
    plans = {}
    for name, sha in p.PLAN_PINS.items():
        rec = p.hash_file(p.STUDY/name)[0]
        p.require(rec['sha256'] == sha, 'Reviewed plan changed')
        plans[name] = rec
    p.require(report['reviewed_plans'] == plans and report['module'] == str(p.MODULE)
              and report['canonical_root'] == str(p.SOURCE), 'Reported source roots/plans differ')
    root = values['locks/full_20261011/LOCK.json']
    batch = values['validation/combined_full_20261011/BATCH_MANIFEST.json']
    prior = values['validation/interruption_20261011/report.json']
    continuation, _ = p.read_json(p.MODULE/'runs/continuation_20261011/BATCH_MANIFEST.json', batch['continuation_manifest'])
    p.require(len(root['groups']) == 860 and [r['id'] for r in batch['groups']] == list(root['groups']), 'Pinned group grid differs')
    p.require(len(prior['sealed_groups']) == 845 and set(prior['failed_groups']) == {f'group_{i:04d}' for i in range(845,854)}, 'Pinned interruption grid differs')
    expected, counts, task_keys = {}, Counter(), set()

    def add_group(ident, origin, relative_root, manifest_record=None):
        lockpath = p.member(p.MODULE, 'locks/full_20261011/groups/'+ident+'.json')
        group, lockrec = p.read_json(lockpath, root['groups'][ident]['lock'])
        p.require(group['id'] == ident and list(group['tasks']) == root['groups'][ident]['jobs'], 'Group lock task grid differs')
        gp = p.member(p.MODULE, relative_root)
        manifest = None
        if manifest_record is not None:
            manifest, _ = p.read_json(gp/'RUN_MANIFEST.json', manifest_record)
            complete, _ = p.read_json(gp/'COMPLETE.json')
            p.require(complete == dict(status='complete',manifest=manifest_record) and not (gp/'FAILED.json').exists(), 'Group completion differs')
        if origin == 'failed':
            p.hash_file(gp/'FAILED.json', prior['failed_groups'][ident]['evidence'])
            p.require(not (gp/'COMPLETE.json').exists(), 'Failed group relabelled complete')
            resultdir = p.check_components(gp/'runtime/results')
            jobs = sorted(x.name for x in resultdir.iterdir()) if resultdir.exists() else []
            p.require(set(jobs) <= set(group['tasks']) and len(jobs) == prior['failed_groups'][ident]['unsealed_prediction_files'], 'Failed partial job grid differs')
        else:
            jobs = list(group['tasks'])
        for jobid in jobs:
            task = group['tasks'][jobid]
            key = p.key_for(task['job'])
            if origin in ('original','continuation'):
                k = p.digest(key)
                p.require(k not in task_keys, 'Duplicate locked task identity')
                task_keys.add(k)
            names = sorted(p.RESULT_NAMES) if origin in ('failed','smoke') else ['ledger.json','predictions.npz']
            jobroot = p.member(gp, 'runtime/results/'+jobid)
            for name in names:
                target = jobroot/name
                rel = str(target.relative_to(p.MODULE))
                p.require(rel not in expected, 'Duplicate allowed candidate path')
                source_rel = str(Path(task['folder'])/name)
                canonical = p.member(p.SOURCE, source_rel)
                source_rec = (group['inputs'] if name == 'ledger.json' else group['references'])[source_rel]
                sealed = manifest['files'][str(target.relative_to(gp))] if manifest else None
                cat = p.category(origin,name)
                if name in ('ledger.json','predictions.npz') and sealed is not None:
                    p.require(sealed == source_rec, 'Locked duplicate relation differs')
                expected[rel] = dict(path=str(target),relative_to_module=rel,category=cat,origin=origin,
                    group=ident,local_job=jobid,task_key=key,payload=name,
                    group_lock=dict(path=str(lockpath),**lockrec),sealed_target=sealed,
                    canonical=dict(path=str(canonical),**source_rec))
                counts[cat] += 1
    for route in batch['groups']:
        ident = route['id']
        origin = 'original' if ident in prior['sealed_groups'] else 'continuation'
        manifest_record = prior['sealed_groups'][ident]['manifest'] if origin == 'original' else continuation['groups'][ident]
        p.require(route == dict(id=ident,origin=origin,manifest=manifest_record), 'Pinned route differs')
        base = 'full_20261011' if origin == 'original' else 'continuation_20261011'
        add_group(ident,origin,'runs/'+base+'/'+ident,manifest_record)
    for ident in prior['failed_groups']:
        add_group(ident,'failed','runs/full_20261011/'+ident)
    for ident, rec in SMOKE_MANIFESTS.items():
        add_group(ident,'smoke','validation/group_smoke_20261011/'+ident,rec)
    p.check_counts(counts)
    p.require(len(task_keys) == 32370, 'Allowed task count differs')
    return expected


def check_candidate_binding(row, expected):
    rel = row.get('relative_to_module')
    p.require(isinstance(rel,str) and rel in expected, 'Candidate not allowed by fixed locks and routes')
    binding = expected[rel]
    p.require(set(row) == CANDIDATE_FIELDS, 'Candidate schema differs')
    for field in BINDING_FIELDS:
        p.require(row[field] == binding[field], 'Candidate fixed binding differs: '+field)
    canonical = row['canonical']
    p.require(set(canonical) == {'path','bytes','sha256','stat'} and
        {k:canonical[k] for k in ('path','bytes','sha256')} == binding['canonical'], 'Candidate canonical binding differs')
    rec = {k:row[k] for k in ('bytes','sha256')}
    source = {k:canonical[k] for k in ('bytes','sha256')}
    if binding['sealed_target'] is not None:
        p.require(rec == binding['sealed_target'], 'Candidate sealed payload differs')
    if row['payload'] in ('ledger.json','predictions.npz'):
        p.require(rec == source, 'Candidate locked duplicate differs')
    equal = rec == source
    recovery = 'canonical_copy_if_hash_still_matches' if equal else 'not_byte_restorable_from_canonical_keep_disposition_hash_only'
    impact = ['full verify-run','compare_combined','historical file inventories'] if row['category'].startswith('B') else ['failed_or_smoke_payload_inventory','historical file inventories']
    p.require(row['current_bytes_equal_canonical'] is equal and row['recovery'] == recovery and row['affected_checks'] == impact, 'Candidate recovery/impact differs')
    p.require(row['stat']['bytes'] == row['bytes'] and canonical['stat']['bytes'] == canonical['bytes'], 'Candidate size metadata differs')
    p.require(row['currently_authorized_action'] == 'read_only_review' and
        row['proposed_action'] == 'unlink_only_after_separate_explicit_approval_and_fresh_checks', 'Candidate action differs')


def check_candidate_grid(actual, expected):
    p.require(len(actual) == len(set(actual)) and set(actual) == set(expected), 'Candidate grid differs from fixed locks and routes')


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
    check_tool_contract(directory, report)
    allowed = expected_candidates(report)
    before = {r['relative_to_module']: r['stat'] for r in lines(directory / 'before_metadata.jsonl.gz')}
    p.require(len(before) == report['module_files'], 'Module inventory count differs')
    candidates, retained, counts, logical, allocated, modes = set(), set(), Counter(), Counter(), Counter(), Counter()
    for r in lines(directory / 'candidates.jsonl.gz'):
        check_candidate_binding(r, allowed)
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
    check_candidate_grid(candidates, allowed)
    for r in lines(directory / 'retained.jsonl.gz'):
        rel = r['relative_to_module']
        p.require(rel not in retained and rel not in candidates, 'Retained set overlap/duplicate')
        p.require(r['stat'] == before[rel] and r['path'] == str(p.member(p.MODULE, rel)), 'Retained record differs')
        retained.add(rel)
    p.check_counts(counts)
    p.require(candidates | retained == set(before), 'Disposition inventory incomplete')
    p.require(dict(counts) == report['counts'] and dict(logical) == report['logical_bytes'] and dict(allocated) == report['allocated_bytes'], 'Totals differ')
    p.require(len(candidates) == report['candidate_count'] and len(retained) == report['retained_files'], 'Record counts differ')
    p.require(p.census(p.MODULE) == before, 'Module metadata changed since proposal')
    return dict(status='proposal_consistency_verified_not_deletion_authority', candidates=len(candidates), retained=len(retained), recovery_modes=dict(modes),
                candidate_scope_bound_to_fixed_provenance=True,
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
