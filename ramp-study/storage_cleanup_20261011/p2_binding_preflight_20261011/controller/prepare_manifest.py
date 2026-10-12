"""Prepare a sealed, read-only disposal proposal. No deletion/execution mode exists."""
import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
MODULE = STUDY / 'benchmarks/trainability/head_retraining_full'
SOURCE = STUDY.parent / 'final_xu'
PINS = {
    'locks/full_20261011/LOCK.json': '085b7c3e5bfc974e8b99e774b604e094d74793f51e9b41d244783fedbe50a685',
    'validation/combined_full_20261011/BATCH_MANIFEST.json': '59102da7b9a180e7477d57fefb33c4850c0d572df70eb8d0122447a6d7afef88',
    'validation/interruption_20261011/report.json': 'c21ee267d4f804869c0568183298a4b51ea3fec2d4e80c6cdb3fc1c26f3dc363',
    'validation/combined_comparison_20261011/report.json': 'c683d55d92514c965dd41a417e4f1e1a3dc9b624384142b01672e3f510992800',
    'validation/combined_aggregation_20261011/report.json': 'b6368bc63b5468924908b7de99278568623d52653c28365c16ce724378c7033a',
    'validation/lockdir_fix_20261011/report.json': '3c014553ee3e86517a67f1ee117e68bd9f48f1eeef3f09756b0a5f8ae924fda2',
}
PLAN_PINS = {
    'STORAGE_CLEANUP_PLAN_20261011.md': '8e62cec5626e79f29b36c2f114f22cc554925a50d6d171900df901c9d467ef29',
    'STORAGE_CLEANUP_INVENTORY_20261011.json': '08e6fc237efc8423c161dccdbafbc0ab95e45a02a67d8993705534f7e3fcf734',
}
RESULT_NAMES = {'ledger.json', 'predictions.npz', 'predictions.json', 'RESULT.json'}
EXPECTED_COUNTS = {'A_failed': 936, 'A_smoke': 432, 'B1': 32370, 'B2': 32370}
TASK_FIELDS = ('task_type', 'task', 'method', 'family', 'seed', 'source', 'target', 'marker', 'id')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def encode(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n').encode()


def digest(value):
    return hashlib.sha256(encode(value)).hexdigest()


def check_components(path):
    path = Path(os.path.abspath(path))
    for p in reversed((path, *path.parents)):
        require(not p.is_symlink(), 'Symlink component: ' + str(p))
    return path


def member(root, relative):
    root = check_components(root)
    rel = Path(relative)
    require(not rel.is_absolute() and bool(rel.parts) and '..' not in rel.parts, 'Unsafe relative path')
    path = check_components(root / rel)
    require(path != root and path.is_relative_to(root), 'Path escaped root')
    return path


def identity(st):
    return dict(device=st.st_dev, inode=st.st_ino, mode=st.st_mode, links=st.st_nlink,
                bytes=st.st_size, mtime_ns=st.st_mtime_ns, ctime_ns=st.st_ctime_ns,
                allocated_bytes=st.st_blocks * 512)


def regular(path):
    st = check_components(path).lstat()
    require(stat.S_ISREG(st.st_mode) and st.st_nlink == 1, 'Not a single-link regular file: ' + str(path))
    return identity(st)


def hash_file(path, expected=None, capture=False):
    """Read payload once; refuse observed changes or aliases during the read."""
    before = regular(path)
    h = hashlib.sha256()
    chunks = []
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        require(identity(os.fstat(stream.fileno())) == before, 'File replaced before read')
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
            if capture:
                chunks.append(block)
        require(identity(os.fstat(stream.fileno())) == before, 'File changed during read')
    require(regular(path) == before, 'File replaced after read')
    rec = dict(bytes=before['bytes'], sha256=h.hexdigest())
    if expected is not None:
        require(rec == expected, 'Content differs from sealed record: ' + str(path))
    return rec, before, b''.join(chunks) if capture else None


def read_json(path, expected=None):
    rec, _, data = hash_file(path, expected, capture=True)
    return json.loads(data), rec


def write_new(path, value):
    with Path(path).open('xb') as f:
        f.write(encode(value))


@contextmanager
def compressed_lines(path):
    with Path(path).open('xb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as stream:
            yield lambda value: stream.write(encode(value))


def census(root):
    records = {}
    for base, dirs, names in os.walk(check_components(root), followlinks=False):
        for name in dirs:
            p = check_components(Path(base) / name)
            require(p.is_dir(), 'Unexpected directory type')
        for name in names:
            p = Path(base) / name
            records[str(p.relative_to(root))] = regular(p)
    return records


def key_for(task):
    require(all(k in task for k in TASK_FIELDS if k != 'marker'), 'Incomplete composite task key')
    require(task['task_type'] in ('J1', 'J2'), 'Unknown task type')
    marker = task.get('marker')
    require((task['task_type'] == 'J1' and marker is None) or
            (task['task_type'] == 'J2' and isinstance(marker, str) and bool(marker)), 'Marker does not match task type')
    return {k: task.get(k) for k in TASK_FIELDS}


def category(origin, name):
    require(name in RESULT_NAMES, 'Unknown payload name')
    if origin in ('failed', 'smoke'):
        return 'A_' + origin
    require(origin in ('original', 'continuation'), 'Unknown route origin')
    require(name in ('ledger.json', 'predictions.npz'), 'Accepted result is retained')
    return 'B1' if name == 'ledger.json' else 'B2'


def check_counts(counts):
    require(dict(counts) == EXPECTED_COUNTS, 'Candidate grid differs')


def reserve_output(path):
    path = check_components(path)
    require(path.parent == HERE and path.name.startswith('manifest_'), 'Output must be a new manifest directory beside this tool')
    path.mkdir(exist_ok=False)
    return path


def prepare(output):
    out = reserve_output(output)
    start = now()
    try:
        code = {p.name: hash_file(p)[0] for p in sorted(HERE.glob('*.py'))}
        write_new(out / 'REQUEST.json', dict(scope='disposal_proposal_only', code=code, started_utc=start, deletion_authorized=False))
        plans = {}
        for name, sha in PLAN_PINS.items():
            rec = hash_file(STUDY / name)[0]
            require(rec['sha256'] == sha, 'Reviewed plan changed')
            plans[name] = rec
        before = census(MODULE)
        anchors, values = {}, {}
        for name, sha in PINS.items():
            value, rec = read_json(member(MODULE, name))
            require(rec['sha256'] == sha, 'Pinned anchor changed')
            anchors[name], values[name] = rec, value
        root = values['locks/full_20261011/LOCK.json']
        batch = values['validation/combined_full_20261011/BATCH_MANIFEST.json']
        prior = values['validation/interruption_20261011/report.json']
        routes = batch['groups']
        require(len(routes) == 860 and [r['id'] for r in routes] == list(root['groups']), 'Accepted group grid differs')
        require(len(prior['sealed_groups']) == 845 and len(prior['failed_groups']) == 9, 'Interrupted group grid differs')
        require(set(prior['failed_groups']) == {f'group_{i:04d}' for i in range(845, 854)}, 'Unknown failed group')
        targets, observed, counts, logical, allocated = {}, {}, Counter(), Counter(), Counter()
        accepted_keys, failed_jobs, smoke_jobs = set(), 0, 0
        copied_audits = {}
        audit_path = Path('/private/tmp/independent_cleanup_plan_ndez57z3/audit_result.json')
        if audit_path.exists():
            require(regular(audit_path)['bytes'] <= 65536, 'Audit attachment too large')
            rec, _, data = hash_file(audit_path, capture=True)
            with (out / 'external_audit1_plan.json').open('xb') as f:
                f.write(data)
            copied_audits['external_audit1_plan.json'] = dict(source=str(audit_path), **rec)
        require((HERE / 'USER_SUPPLIED_AUDITS.txt').is_file(), 'User supplied audit transcript missing')
        transcript = hash_file(HERE / 'USER_SUPPLIED_AUDITS.txt')[0]
        with compressed_lines(out / 'candidates.jsonl.gz') as emit:
            def group(ident, origin, group_root, expected_manifest=None):
                nonlocal failed_jobs, smoke_jobs
                lockpath = member(MODULE, 'locks/full_20261011/groups/' + ident + '.json')
                g, gr = read_json(lockpath, root['groups'][ident]['lock'])
                require(g['id'] == ident and list(g['tasks']) == root['groups'][ident]['jobs'], 'Task grid differs')
                manifest = None
                if expected_manifest is not None:
                    manifest, _ = read_json(group_root / 'RUN_MANIFEST.json', expected_manifest)
                    complete, _ = read_json(group_root / 'COMPLETE.json')
                    require(complete == dict(status='complete', manifest=expected_manifest), 'Group completion differs')
                    require(not (group_root / 'FAILED.json').exists(), 'Accepted group has FAILED')
                if origin == 'failed':
                    hash_file(group_root / 'FAILED.json', prior['failed_groups'][ident]['evidence'])
                    require(not (group_root / 'COMPLETE.json').exists(), 'Failed group relabelled complete')
                results = group_root / 'runtime/results'
                actual_jobs = sorted(p.name for p in results.iterdir()) if results.exists() else []
                if origin == 'failed':
                    require(set(actual_jobs) <= set(g['tasks']), 'Unknown partial job')
                    failed_jobs += len(actual_jobs)
                else:
                    require(set(actual_jobs) == set(g['tasks']), 'Missing or extra result job')
                if origin == 'smoke':
                    smoke_jobs += len(actual_jobs)
                for jobid in actual_jobs:
                    row = g['tasks'][jobid]
                    taskkey = key_for(row['job'])
                    if origin in ('original', 'continuation'):
                        k = digest(taskkey)
                        require(k not in accepted_keys, 'Duplicate accepted task key')
                        accepted_keys.add(k)
                    jobroot = member(results, jobid)
                    require({p.name for p in jobroot.iterdir()} == RESULT_NAMES, 'Result payload inventory differs')
                    names = sorted(RESULT_NAMES) if origin in ('failed', 'smoke') else ['ledger.json', 'predictions.npz']
                    for name in names:
                        target = member(jobroot, name)
                        rel = str(target.relative_to(MODULE))
                        require(rel not in targets, 'Duplicate candidate path')
                        cat = category(origin, name)
                        source_rel = str(Path(row['folder']) / name)
                        source = member(SOURCE, source_rel)
                        source_record = (g['inputs'] if name == 'ledger.json' else g['references'])[source_rel]
                        sealed = manifest['files'][str(target.relative_to(group_root))] if manifest else None
                        target_rec, target_stat, _ = hash_file(target, sealed)
                        source_rec, source_stat, _ = hash_file(source, source_record)
                        require((target_stat['device'], target_stat['inode']) != (source_stat['device'], source_stat['inode']), 'Source and target alias')
                        equal = target_rec == source_rec
                        if name in ('ledger.json', 'predictions.npz'):
                            require(equal, 'Duplicate payload differs from canonical')
                        targets[rel] = target_rec
                        for p, s in ((target, target_stat), (source, source_stat)):
                            require(str(p) not in observed or observed[str(p)] == s, 'File changed between reads')
                            observed[str(p)] = s
                        counts[cat] += 1
                        logical[cat] += target_rec['bytes']
                        allocated[cat] += target_stat['allocated_bytes']
                        emit(dict(path=str(target), relative_to_module=rel, category=cat, origin=origin,
                            group=ident, local_job=jobid, task_key=taskkey, payload=name,
                            bytes=target_rec['bytes'], sha256=target_rec['sha256'], stat=target_stat,
                            canonical=dict(path=str(source), **source_rec, stat=source_stat),
                            current_bytes_equal_canonical=equal,
                            group_lock=dict(path=str(lockpath), **gr), sealed_target=sealed,
                            proposed_action='unlink_only_after_separate_explicit_approval_and_fresh_checks',
                            currently_authorized_action='read_only_review',
                            recovery='canonical_copy_if_hash_still_matches' if equal else 'not_byte_restorable_from_canonical_keep_disposition_hash_only',
                            affected_checks=['full verify-run', 'compare_combined', 'historical file inventories'] if cat.startswith('B') else ['failed_or_smoke_payload_inventory', 'historical file inventories']))
            for i, route in enumerate(routes):
                ident = route['id']
                expected_origin = 'original' if ident in prior['sealed_groups'] else 'continuation'
                require(route['origin'] == expected_origin, 'Accepted origin differs')
                if expected_origin == 'original':
                    require(route['manifest'] == prior['sealed_groups'][ident]['manifest'], 'Original manifest route differs')
                group(ident, expected_origin, member(MODULE, f'runs/{"full_20261011" if expected_origin == "original" else "continuation_20261011"}/{ident}'), route['manifest'])
                if (i + 1) % 25 == 0:
                    print('HASHED accepted groups', i + 1, '/ 860', flush=True)
            for ident in prior['failed_groups']:
                group(ident, 'failed', member(MODULE, 'runs/full_20261011/' + ident))
            for ident in ('group_0000', 'group_0858'):
                gp = member(MODULE, 'validation/group_smoke_20261011/' + ident)
                _, manifest_rec = read_json(gp / 'RUN_MANIFEST.json')
                group(ident, 'smoke', gp, manifest_rec)
        check_counts(counts)
        require(len(accepted_keys) == 32370 and failed_jobs == 234 and smoke_jobs == 108, 'Task counts differ')
        require(set(targets) <= set(before), 'Candidate not in original inventory')
        print('Hashing retained files (no copies)', flush=True)
        retained_count = 0
        with compressed_lines(out / 'retained.jsonl.gz') as emit:
            for rel in sorted(before):
                if rel not in targets:
                    rec, st, _ = hash_file(member(MODULE, rel))
                    require(st == before[rel], 'Retained file changed since start')
                    emit(dict(path=str(MODULE / rel), relative_to_module=rel, **rec, stat=st))
                    retained_count += 1
        with compressed_lines(out / 'before_metadata.jsonl.gz') as emit:
            for rel in sorted(before):
                emit(dict(relative_to_module=rel, stat=before[rel]))
        require(census(MODULE) == before, 'Module file inventory changed during proposal')
        for p, st in observed.items():
            require(regular(p) == st, 'Candidate or canonical changed since hashing: ' + p)
        for name, rec in anchors.items():
            hash_file(member(MODULE, name), rec)
        for name, rec in plans.items():
            hash_file(STUDY / name, rec)
        require(code == {p.name: hash_file(p)[0] for p in sorted(HERE.glob('*.py'))}, 'Proposal tool changed during scan')
        require(hash_file(HERE / 'USER_SUPPLIED_AUDITS.txt')[0] == transcript, 'Audit transcript changed')
        report = dict(status='proposal_ready_for_independent_review', deletion_authorized=False,
            deleted_files=0, large_files_copied=0, scientific_code_changed=False,
            started_utc=start, finished_utc=now(), module=str(MODULE), canonical_root=str(SOURCE),
            counts=dict(counts), candidate_count=sum(counts.values()), logical_bytes=dict(logical), allocated_bytes=dict(allocated),
            retained_files=retained_count, module_files=len(before), accepted_jobs=len(accepted_keys),
            failed_jobs=failed_jobs, smoke_jobs=smoke_jobs, fresh_hash_pairs=sum(counts.values()),
            module_metadata_before_after_equal=True, canonical_observed_metadata_unchanged=True,
            anchors=anchors, reviewed_plans=plans, tool_code=code,
            external_audit_attachments=copied_audits, user_supplied_audits=transcript,
            options={'A': ['A_failed', 'A_smoke'], 'A+B1': ['A_failed', 'A_smoke', 'B1'], 'A+B1+B2': list(EXPECTED_COUNTS)},
            limitations=[
                'This manifest is not deletion permission; no executor or deletion mode is provided.',
                'Hash evidence is point-in-time. Rehash both files immediately before any approved deletion.',
                'APFS allocation is not guaranteed reclaim; no off-device backup verified.',
                'A RESULT/receipt files can contain unique timing/path metadata. Their exact bytes are NOT necessarily recoverable from historical counterparts.',
                'No active-reader absence certification; process inspection was unavailable in this environment. Recheck before deletion.',
                'Preservation here covers the full head module; not a new content audit of every other research module.',
                'Old pilot preflight is reportedly already failing due to later full-module additions; do not attribute that pre-existing check to future cleanup.',
                'Independent reviewers accepted the earlier plan, not this newly generated manifest.',
            ])
        write_new(out / 'report.json', report)
        files = {p.name: hash_file(p)[0] for p in sorted(out.iterdir())}
        write_new(out / 'COMPLETE.json', dict(status='proposal_complete_not_deleted', files=files))
        print(json.dumps({k: report[k] for k in ('status','candidate_count','counts','retained_files','deleted_files')}, ensure_ascii=False), flush=True)
        return report
    except BaseException as exc:
        write_new(out / 'FAILED.json', dict(status='proposal_failed_no_deletion', error=repr(exc), when=now()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.output)


if __name__ == '__main__':
    main()
