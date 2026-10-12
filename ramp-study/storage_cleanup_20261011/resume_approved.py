"""Exact-prefix recovery from the stopped cleanup; same approval and deletion guards."""
from collections import Counter
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import execute_approved as e
import prepare_manifest as p
import validate_manifest as v

PRIOR = e.OUTPUT
OUTPUT = p.HERE / 'execution_20261011_r2'
PREFIX = 10000
EXECUTOR_SHA = '76a5ee4749df55f9f6491e149f270ea5aa6e325db287642f88e5d319d06cdedb'


def verify_prefix(rows, receipt, count):
    """An interrupted intent or extra line is not silently treated as completed."""
    with receipt.open() as f:
        for i in range(count):
            row = rows[i]
            intent = json.loads(next(f)); done = json.loads(next(f))
            p.require(intent == dict(event='intent', ordinal=i, row_sha256=p.digest(row), utc=intent['utc'],
                                     target=e.rec(row), canonical=e.rec(row['canonical'])), 'Prior intent differs')
            p.require(done == dict(event='deleted', ordinal=i, utc=done['utc']), 'Prior deletion receipt differs')
        p.require(not f.read(), 'Unexpected prior journal tail')


def run():
    p.require(p.hash_file(e.MANIFEST / 'COMPLETE.json')[0]['sha256'] == e.MANIFEST_SHA, 'Proposal changed')
    p.require(p.hash_file(p.HERE / 'execute_approved.py')[0]['sha256'] == EXECUTOR_SHA, 'Original executor changed')
    p.require(p.hash_file(p.HERE / 'validate_manifest.py')[0]['sha256'] == e.VERIFIER_SHA, 'Verifier changed')
    p.require(p.hash_file(p.HERE / 'prepare_manifest.py')[0]['sha256'] == e.GENERATOR_SHA, 'Primitives changed')
    prior_request, _ = p.read_json(PRIOR / 'REQUEST.json')
    failure, _ = p.read_json(PRIOR / 'FAILED.json')
    p.require(prior_request['approval_quote'] == e.APPROVAL and prior_request['deletion_authorized'] is True,
              'Prior authorization differs')
    p.require(failure['acknowledged_deleted_counts'] == {'B1': 5000, 'B2': 5000} and
              failure['error'].startswith("ValueError('Active consumers:"), 'Unexpected interruption')
    p.require(not (PRIOR / 'DELETION_COMPLETE.json').exists(), 'Prior phase was not interrupted')
    prior_files = {x.name: p.hash_file(x)[0] for x in sorted(PRIOR.iterdir())}
    p.check_components(OUTPUT).mkdir(exist_ok=False)
    code = {name: p.hash_file(p.HERE / name)[0] for name in
            ('resume_approved.py', 'test_resume_approved.py', 'execute_approved.py',
             'test_execute_approved.py', 'prepare_manifest.py', 'validate_manifest.py')}
    request = dict(approval_quote=e.APPROVAL, deletion_authorized=True, continued_from=str(PRIOR),
        prefix_deleted=PREFIX, prior_files=prior_files, code=code, manifest_complete_sha256=e.MANIFEST_SHA,
        started_utc=p.now(), reason='Transient PID 77320 exited; same consumer guard retained without exemptions',
        original_disk_before=prior_request['disk_before'])
    p.write_new(OUTPUT / 'REQUEST.json', request)
    count = Counter(B1=5000, B2=5000)
    try:
        seal, _ = p.read_json(e.MANIFEST / 'COMPLETE.json')
        for name, record in seal['files'].items():
            p.hash_file(e.MANIFEST / name, record)
        proposal, _ = p.read_json(e.MANIFEST / 'report.json')
        v.check_tool_contract(e.MANIFEST, proposal)
        allowed = v.expected_candidates(proposal)
        rows = list(v.lines(e.MANIFEST / 'candidates.jsonl.gz'))
        retained = list(v.lines(e.MANIFEST / 'retained.jsonl.gz'))
        v.check_candidate_grid([r['relative_to_module'] for r in rows], allowed)
        for row in rows:
            v.check_candidate_binding(row, allowed)
        verify_prefix(rows, PRIOR / 'receipts.jsonl', PREFIX)
        p.require(Counter(r['category'] for r in rows[:PREFIX]) == count, 'Prefix categories differ')
        before = {r['relative_to_module']: r['stat'] for r in v.lines(e.MANIFEST / 'before_metadata.jsonl.gz')}
        removed = {r['relative_to_module'] for r in rows[:PREFIX]}
        p.require(p.census(p.MODULE) == {k: value for k, value in before.items() if k not in removed},
                  'Current inventory is not exactly the approved inventory minus completed prefix')
        for row in rows:
            p.require(p.regular(Path(row['canonical']['path'])) == row['canonical']['stat'], 'Canonical changed')
        p.write_new(OUTPUT / 'PREFIX_VERIFIED.json', dict(deleted_prefix=PREFIX, remaining=len(rows)-PREFIX,
                    current_inventory_exact=True, original_receipts=p.hash_file(PRIOR / 'receipts.jsonl')[0]))
        print('Exact interrupted prefix verified; resuming at', PREFIX, flush=True)
        with (OUTPUT / 'receipts.jsonl').open('xb', buffering=0) as journal, \
             (OUTPUT / 'consumer_checks.jsonl').open('xb', buffering=0) as consumer_log:
            def emit(value):
                journal.write(p.encode(value)); os.fsync(journal.fileno())
            for i in range(PREFIX, len(rows)):
                if i % 1000 == 0:
                    for name, record in code.items():
                        p.hash_file(p.HERE / name, record)
                    consumer_log.write(p.encode(e.consumers())); os.fsync(consumer_log.fileno())
                    print('Approved deletions:', i, '/', len(rows), flush=True)
                e.remove_one(rows[i], i, allowed, emit)
                count[rows[i]['category']] += 1
        p.check_counts(count)
        p.require(p.census(p.MODULE) == {r['relative_to_module']: r['stat'] for r in retained},
                  'Final inventory differs from approved retained set')
        e.check_retained(retained)
        for row in rows:
            p.require(p.regular(Path(row['canonical']['path'])) == row['canonical']['stat'], 'Canonical changed')
        for name, record in code.items():
            p.hash_file(p.HERE / name, record)
        for name, record in prior_files.items():
            p.hash_file(PRIOR / name, record)
        p.require({x.name for x in PRIOR.iterdir()} == set(prior_files), 'Prior evidence inventory changed')
        for name, record in seal['files'].items():
            p.hash_file(e.MANIFEST / name, record)
        report = dict(status='deletion_complete_postchecks_pending', counts=dict(count), deleted_files=sum(count.values()),
            receipt_parts=[dict(directory=str(PRIOR), start=0, stop=PREFIX, record=prior_files['receipts.jsonl']),
                           dict(directory=str(OUTPUT), start=PREFIX, stop=len(rows), record=p.hash_file(OUTPUT/'receipts.jsonl')[0])],
            logical_bytes_removed=sum(r['bytes'] for r in rows), allocated_bytes_removed=sum(r['stat']['allocated_bytes'] for r in rows),
            retained_files=len(retained), retained_hashes_unchanged=True, canonical_pairs_live_hashed_before_each_deletion=len(rows),
            canonical_final_metadata_unchanged=True, directories_deleted=0, disk_before=prior_request['disk_before'],
            disk_after=e.disk(), finished_utc=p.now(), large_payload_copies=0,
            interrupted_phase_preserved=str(PRIOR), consumer_guard_unchanged=True)
        p.write_new(OUTPUT / 'deletion_report.json', report)
        p.write_new(OUTPUT / 'DELETION_COMPLETE.json', dict(status='complete', files={
            x.name:p.hash_file(x)[0] for x in sorted(OUTPUT.iterdir()) if x.is_file()}))
        print(json.dumps(report), flush=True)
    except BaseException as exc:
        p.write_new(OUTPUT / 'FAILED.json', dict(error=repr(exc), acknowledged_deleted_counts=dict(count),
                    when=p.now(), recovery='Inspect both journals; do not blindly rerun'))
        raise


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--approved-scope', required=True, choices=['A+B1+B2'])
    parser.parse_args(); run()
