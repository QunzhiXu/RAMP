"""One-shot execution of the user's pinned A+B1+B2 approval; never deletes directories."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import prepare_manifest as p
import validate_manifest as v

MANIFEST = p.HERE / 'manifest_20261011_r2'
OUTPUT = p.HERE / 'execution_20261011'
MANIFEST_SHA = '9e773c00f8a4c21af65c70d29e15b2878020281f2a6d46a0aadeaa508fb5c251'
VERIFIER_SHA = '605eec7527f8e91ac179da84d9e95599c42e6265b024598c4a0f44996826096f'
GENERATOR_SHA = 'cb51ed0ae71580b947a663bb596dbdda690564f4c805a6c57c08923879c12517'
APPROVAL = '批准 A＋B1＋B2'


def rec(row):
    return {k: row[k] for k in ('bytes', 'sha256')}


def disk():
    s = shutil.disk_usage(p.STUDY)
    return dict(total=s.total, used=s.used, free=s.free)


def consumers():
    """Require working process and open-file visibility; persist only relevant records."""
    ps = subprocess.run(['/bin/ps', '-axo', 'pid=,ppid=,command='], capture_output=True, text=True)
    p.require(ps.returncode == 0 and not ps.stderr.strip(), 'Process visibility unavailable')
    processes = {}
    for line in ps.stdout.splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) == 3:
            processes[int(fields[0])] = (int(fields[1]), fields[2])
    ancestors, pid = set(), os.getpid()
    while pid and pid not in ancestors:
        ancestors.add(pid)
        pid = processes.get(pid, (0, ''))[0]
    active = [dict(pid=pid, command=cmd) for pid, (_, cmd) in processes.items()
              if pid not in ancestors and ('head_retraining_full' in cmd or
                 'full_worker.py' in cmd or 'resume_worker.py' in cmd)]
    opened = subprocess.run(['/usr/sbin/lsof', '-nP', '-F', 'pn'], capture_output=True, text=True)
    p.require(opened.returncode == 0 and not opened.stderr.strip(), 'Open-file visibility unavailable')
    hits, pid = [], None
    for line in opened.stdout.splitlines():
        if line.startswith('p') and line[1:].isdigit():
            pid = int(line[1:])
        elif line.startswith('n' + str(p.MODULE) + '/') and pid != os.getpid():
            hits.append(dict(pid=pid, path=line[1:]))
    p.require(not active and not hits, 'Active consumers: ' + json.dumps(dict(processes=active, files=hits)))
    return dict(checked_utc=p.now(), matching_processes=active, matching_open_files=hits,
                scope='Current visible processes and open files; cooperative-use safety, not an OS lock')


def live_pair(row, module=p.MODULE, source=p.SOURCE):
    target = p.member(module, row['relative_to_module'])
    p.require(str(target) == row['path'], 'Target path differs')
    p.require(target.name == row['payload'] and
              p.category(row['origin'], target.name) == row['category'], 'Target role differs')
    canonical = p.member(source, str(Path(row['canonical']['path']).relative_to(source)))
    p.require(p.regular(target) == row['stat'], 'Target metadata changed')
    p.require(p.regular(canonical) == row['canonical']['stat'], 'Canonical metadata changed')
    target_rec, target_stat, _ = p.hash_file(target, rec(row))
    source_rec, source_stat, _ = p.hash_file(canonical, rec(row['canonical']))
    p.require((target_stat['device'], target_stat['inode']) !=
              (source_stat['device'], source_stat['inode']), 'Source and target alias')
    p.require((target_rec == source_rec) is row['current_bytes_equal_canonical'], 'Duplicate relation differs')
    if row['category'].startswith('B') or row['payload'] in ('ledger.json', 'predictions.npz'):
        p.require(target_rec == source_rec, 'Required byte-identical pair differs')
    return target, canonical, target_stat, source_stat


def remove_one(row, ordinal, allowed, emit, module=p.MODULE, source=p.SOURCE):
    v.check_candidate_binding(row, allowed)
    target, canonical, target_stat, source_stat = live_pair(row, module, source)
    # Pin the containing directory and recheck both files after the durable intent.
    parent_stat = target.parent.stat()
    parent_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(parent_fd)
        p.require((opened.st_dev, opened.st_ino) == (parent_stat.st_dev, parent_stat.st_ino), 'Parent replaced')
        emit(dict(event='intent', ordinal=ordinal, row_sha256=p.digest(row), utc=p.now(),
                  target=rec(row), canonical=rec(row['canonical'])))
        p.require(p.regular(target) == target_stat and p.regular(canonical) == source_stat,
                  'File changed after live hashing')
        current_parent = target.parent.stat()
        p.require((current_parent.st_dev, current_parent.st_ino) == (opened.st_dev, opened.st_ino), 'Parent moved')
        p.require(p.identity(os.stat(target.name, dir_fd=parent_fd, follow_symlinks=False)) == target_stat,
                  'Directory-relative target differs')
        os.unlink(target.name, dir_fd=parent_fd)
        p.require(not target.exists() and not target.is_symlink(), 'Unlink did not remove target')
        emit(dict(event='deleted', ordinal=ordinal, utc=p.now()))
    finally:
        os.close(parent_fd)


def check_retained(rows):
    for i, row in enumerate(rows, 1):
        target = p.member(p.MODULE, row['relative_to_module'])
        _, current, _ = p.hash_file(target, rec(row))
        p.require(current == row['stat'], 'Retained metadata changed: ' + str(target))
        if i % 20000 == 0:
            print('Retained hashes:', i, flush=True)
    return len(rows)


def run():
    p.require(p.hash_file(MANIFEST / 'COMPLETE.json')[0]['sha256'] == MANIFEST_SHA, 'Unapproved manifest')
    p.require(p.hash_file(p.HERE / 'validate_manifest.py')[0]['sha256'] == VERIFIER_SHA, 'Audited verifier changed')
    p.require(p.hash_file(p.HERE / 'prepare_manifest.py')[0]['sha256'] == GENERATOR_SHA, 'Audited primitives changed')
    p.check_components(OUTPUT).mkdir(exist_ok=False)
    code = {name: p.hash_file(p.HERE / name)[0] for name in
            ('execute_approved.py', 'test_execute_approved.py', 'prepare_manifest.py', 'validate_manifest.py')}
    request = dict(approval_quote=APPROVAL, scope=list(p.EXPECTED_COUNTS), expected_counts=p.EXPECTED_COUNTS,
                   manifest=str(MANIFEST), manifest_complete_sha256=MANIFEST_SHA, code=code,
                   started_utc=p.now(), disk_before=disk(), deletion_authorized=True,
                   recovery='65424 canonical-byte copies; 684 A metadata files not byte-restorable',
                   no_directories_deleted=True, no_large_files_copied=True)
    p.write_new(OUTPUT / 'REQUEST.json', request)
    count, logical, allocated = Counter(), 0, 0
    try:
        audit_path = Path('/private/tmp/independent_candidate_fix_lgj96yj3/audit_result.json')
        if audit_path.exists():
            ar, _, data = p.hash_file(audit_path, capture=True)
            p.require(ar['bytes'] < 131072, 'Audit attachment too large')
            with (OUTPUT / 'external_candidate_fix_audit1.json').open('xb') as f:
                f.write(data)
            p.write_new(OUTPUT / 'external_audit_source.json', dict(path=str(audit_path), **ar))
        proof = v.verify(MANIFEST)
        p.write_new(OUTPUT / 'approved_manifest_verification.json', proof)
        report, _ = p.read_json(MANIFEST / 'report.json')
        allowed = v.expected_candidates(report)
        rows = list(v.lines(MANIFEST / 'candidates.jsonl.gz'))
        retained = list(v.lines(MANIFEST / 'retained.jsonl.gz'))
        v.check_candidate_grid([r['relative_to_module'] for r in rows], allowed)
        for row in rows:
            v.check_candidate_binding(row, allowed)
        print('Pinned proposal passed; checking all retained content before deletion', flush=True)
        check_retained(retained)
        p.write_new(OUTPUT / 'RETENTION_STARTED.json', dict(status='approved_cleanup_in_progress',
                    approval_quote=APPROVAL, manifest_complete_sha256=MANIFEST_SHA,
                    note='Historical COMPLETE records remain unchanged; consult execution receipts for current payload state'))
        with (OUTPUT / 'receipts.jsonl').open('xb', buffering=0) as journal, \
             (OUTPUT / 'consumer_checks.jsonl').open('xb', buffering=0) as consumer_log:
            def emit(value):
                journal.write(p.encode(value))
                os.fsync(journal.fileno())
            for i, row in enumerate(rows):
                if i % 1000 == 0:
                    for name, sealed in code.items():
                        p.hash_file(p.HERE / name, sealed)
                    consumer_log.write(p.encode(consumers()))
                    os.fsync(consumer_log.fileno())
                    print('Approved deletions:', i, '/', len(rows), flush=True)
                remove_one(row, i, allowed, emit)
                count[row['category']] += 1
                logical += row['bytes']
                allocated += row['stat']['allocated_bytes']
        p.check_counts(count)
        remaining = p.census(p.MODULE)
        p.require(remaining == {r['relative_to_module']: r['stat'] for r in retained},
                  'Post-delete inventory differs from exact retained set')
        print('Deletion finished; rehashing all 84019 retained files', flush=True)
        check_retained(retained)
        for row in rows:
            p.require(p.regular(Path(row['canonical']['path'])) == row['canonical']['stat'], 'Canonical changed')
        # Recheck the immutable proposal and execution code before recording success.
        complete, _ = p.read_json(MANIFEST / 'COMPLETE.json')
        p.require(p.hash_file(MANIFEST / 'COMPLETE.json')[0]['sha256'] == MANIFEST_SHA, 'Proposal changed')
        for name, record in complete['files'].items():
            p.hash_file(MANIFEST / name, record)
        for name, record in code.items():
            p.hash_file(p.HERE / name, record)
        result = dict(status='deletion_complete_postchecks_pending', counts=dict(count), deleted_files=sum(count.values()),
                      logical_bytes_removed=logical, allocated_bytes_removed=allocated, retained_files=len(retained),
                      retained_hashes_unchanged=True, canonical_pairs_live_hashed_before_each_deletion=len(rows),
                      canonical_final_metadata_unchanged=True, directories_deleted=0, disk_before=request['disk_before'],
                      disk_after=disk(), finished_utc=p.now(), large_payload_copies=0)
        p.write_new(OUTPUT / 'deletion_report.json', result)
        p.write_new(OUTPUT / 'DELETION_COMPLETE.json', dict(status='complete', files={
            x.name: p.hash_file(x)[0] for x in sorted(OUTPUT.iterdir()) if x.is_file()}))
        print(json.dumps(result), flush=True)
    except BaseException as exc:
        p.write_new(OUTPUT / 'FAILED.json', dict(error=repr(exc), acknowledged_deleted_counts=dict(count),
                    when=p.now(), recovery='Inspect durable intent/deleted pairs; do not blindly rerun'))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--approved-scope', required=True, choices=['A+B1+B2'])
    parser.parse_args()
    run()
