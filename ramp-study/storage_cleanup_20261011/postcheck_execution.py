"""Non-destructive post-cleanup checks; leave every old result/report unchanged."""
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
import prepare_manifest as p
import validate_manifest as v
import execute_approved as e

OUT = p.HERE / 'post_cleanup_checks_20261011'
AGG = p.MODULE / 'validation/post_cleanup_aggregation_20261011'
OLD_AGG = p.MODULE / 'validation/combined_aggregation_20261011'
RUN = p.HERE / 'execution_20261011_r2'


def invoke(name, args, cwd, expect=0):
    with (OUT / (name + '.stdout')).open('xb') as out, (OUT / (name + '.stderr')).open('xb') as err:
        result = subprocess.run(args, cwd=cwd, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                                stdout=out, stderr=err)
    if expect == 0:
        p.require(result.returncode == 0, 'Postcheck failed: ' + name)
    else:
        p.require(result.returncode != 0, 'Full verifier unexpectedly accepted removed payloads')
    return dict(name=name, command=args, cwd=str(cwd), returncode=result.returncode)


def main():
    p.check_components(OUT).mkdir(exist_ok=False)
    try:
        p.write_new(OUT / 'REQUEST.json', dict(started_utc=p.now(),
            code=p.hash_file(Path(__file__))[0], deletion_seal=p.hash_file(RUN / 'DELETION_COMPLETE.json')[0],
            user_approval_and_reviews=p.hash_file(p.HERE / 'USER_APPROVAL_AND_FINAL_REVIEWS.txt')[0],
            scope='Post-cleanup retention, tests, expected full-verifier rejection, and new score reaggregation'))
        deletion_seal, _ = p.read_json(RUN / 'DELETION_COMPLETE.json')
        p.require(deletion_seal['status'] == 'complete', 'Deletion incomplete')
        for name, record in deletion_seal['files'].items():
            p.hash_file(p.member(RUN, name), record)
        report, _ = p.read_json(RUN / 'deletion_report.json')
        request, _ = p.read_json(RUN / 'REQUEST.json')
        original_request, _ = p.read_json(e.OUTPUT / 'REQUEST.json')
        for name, record in request['prior_files'].items():
            p.hash_file(e.OUTPUT / name, record)
        for name, record in request['code'].items():
            p.hash_file(p.HERE / name, record)
        p.require(request['approval_quote'] == e.APPROVAL and request['deletion_authorized'] is True, 'Approval differs')
        p.require(report['status'] == 'deletion_complete_postchecks_pending' and report['deleted_files'] == 66108, 'Deletion report differs')
        p.check_counts(report['counts'])
        p.require(p.hash_file(e.MANIFEST / 'COMPLETE.json')[0]['sha256'] == e.MANIFEST_SHA, 'Proposal changed')
        rows = list(v.lines(e.MANIFEST / 'candidates.jsonl.gz'))
        expected_parts = [(e.OUTPUT, 0, 10000), (RUN, 10000, 66108)]
        p.require(len(report['receipt_parts']) == 2, 'Receipt parts differ')
        for part, (directory, start, stop) in zip(report['receipt_parts'], expected_parts):
            p.require(part == dict(directory=str(directory),start=start,stop=stop,record=part['record']), 'Receipt routing differs')
            p.hash_file(directory / 'receipts.jsonl', part['record'])
            with (directory / 'receipts.jsonl').open() as f:
                for i in range(start, stop):
                    row = rows[i]
                    intent = json.loads(next(f)); done = json.loads(next(f))
                    p.require(intent == dict(event='intent', ordinal=i, row_sha256=p.digest(row), utc=intent['utc'],
                                             target=e.rec(row), canonical=e.rec(row['canonical'])), 'Intent differs')
                    p.require(done == dict(event='deleted', ordinal=i, utc=done['utc']), 'Deletion receipt differs')
                    path = p.member(p.MODULE, row['relative_to_module'])
                    p.require(not path.exists() and not path.is_symlink(), 'Deleted candidate present')
                    p.require(p.regular(Path(row['canonical']['path'])) == row['canonical']['stat'], 'Canonical metadata changed')
                p.require(not f.read(), 'Extra receipts')
        retained = list(v.lines(e.MANIFEST / 'retained.jsonl.gz'))
        retained_stat = {r['relative_to_module']: r['stat'] for r in retained}
        p.require(p.census(p.MODULE) == retained_stat, 'Unexpected post-deletion file inventory')
        common = ['--lockdir', str(p.MODULE / 'locks/full_20261011'), '--source-root', str(p.SOURCE),
                  '--original-run', str(p.MODULE / 'runs/full_20261011'),
                  '--continuation', str(p.MODULE / 'runs/continuation_20261011'),
                  '--run', str(p.MODULE / 'validation/combined_full_20261011')]
        checks = []
        checks.append(invoke('cleanup_69_tests', [sys.executable, '-B', '-m', 'unittest', '-v',
                      'test_prepare_manifest.py', 'test_candidate_binding.py', 'test_execute_approved.py', 'test_resume_approved.py'], p.HERE))
        checks.append(invoke('existing_58_tests', [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests', '-v'], p.MODULE))
        checks.append(invoke('check_binding', [sys.executable, '-B', str(p.MODULE / 'combined.py'), 'check-binding', *common], p.HERE))
        checks.append(invoke('full_verify_expected_rejection', [sys.executable, '-B', str(p.MODULE / 'combined.py'), 'verify-run', *common], p.HERE, expect=1))
        # Require a payload inventory/missing-file reason, not an unrelated failure.
        rejection = (OUT / 'full_verify_expected_rejection.stderr').read_text()
        p.require(('Artifact files differ' in rejection or 'Inventory differs' in rejection or 'inventory differs' in rejection or
                   ('FileNotFoundError' in rejection and ('ledger.json' in rejection or 'predictions.npz' in rejection))),
                  'Full verifier refused for an unexpected reason')
        print('Tests and descriptor passed; full payload verifier refused as expected', flush=True)
        checks.append(invoke('reaggregate', [sys.executable, '-B', str(p.MODULE / 'aggregate_combined.py'),
            '--lockdir', str(p.MODULE / 'locks/full_20261011'),
            '--run', str(p.MODULE / 'validation/combined_full_20261011'),
            '--comparison', str(p.MODULE / 'validation/combined_comparison_20261011'),
            '--output', str(AGG), '--original-run', str(p.MODULE / 'runs/full_20261011'),
            '--continuation', str(p.MODULE / 'runs/continuation_20261011')], p.HERE))
        tables = {}
        for name in ('primary_means', 'PER_BENCHMARK', 'OVERALL', 'J1_PER_BENCHMARK',
                     'J1_OVERALL', 'J2_PER_BENCHMARK', 'J2_OVERALL', 'per_job_metrics'):
            record = p.hash_file(OLD_AGG / (name + '.csv'))[0]
            p.hash_file(AGG / (name + '.csv'), record)
            tables[name] = record
        old_report, _ = p.read_json(OLD_AGG / 'report.json')
        new_report, _ = p.read_json(AGG / 'report.json')
        p.require(new_report == old_report, 'Aggregation report differs')
        e.check_retained(retained)
        prefix = str(AGG.relative_to(p.MODULE)) + '/'
        after = p.census(p.MODULE)
        p.require({k: value for k, value in after.items() if not k.startswith(prefix)} == retained_stat,
                  'Files outside new aggregation changed')
        extras = {k[len(prefix):]: value for k, value in after.items() if k.startswith(prefix)}
        p.require(set(extras) == {name + '.csv' for name in tables} | {'report.json', 'RUN_MANIFEST.json', 'COMPLETE.json'},
                  'Unexpected new aggregation payload')
        measured = {}
        for label, path in [('study', p.STUDY), ('head_retraining_full', p.MODULE)]:
            q = subprocess.run(['/usr/bin/du', '-sk', str(path)], capture_output=True, text=True, check=True)
            measured[label] = int(q.stdout.split()[0]) * 1024
        state = dict(status='audited_reproduction_evidence_retained_payloads_pruned', when=p.now(),
            deletion_report=p.hash_file(RUN / 'deletion_report.json')[0], approved_scope='A+B1+B2',
            deleted_files=66108, unchanged_retained_files=84019, receipts_verified=66108,
            source_copies_recoverable_if_canonical_available=65424, non_byte_restorable_A_metadata_files=684,
            new_aggregation=str(AGG), byte_identical_csvs=tables, aggregation_report_identical=True,
            full_payload_verifier='expected rejection: payload inventory no longer complete',
            scientific_results_and_historical_reports='unchanged; old COMPLETE denotes past execution, not current payload completeness',
            tests=dict(cleanup=69, existing_module=58), checks=checks,
            disk_before=original_request['disk_before'], disk_after=e.disk(), measured_allocated_bytes=measured,
            measured_before_allocated_bytes=dict(study=67062296*1024, head_retraining_full=56547800*1024),
            memory_note='This frees disk storage, not RAM', no_training=True, large_payload_copies=0)
        p.write_new(OUT / 'RETENTION_STATE.json', state)
        p.write_new(OUT / 'COMPLETE.json', dict(status='complete', files={
            x.name: p.hash_file(x)[0] for x in sorted(OUT.iterdir()) if x.is_file()}))
        print(json.dumps(state), flush=True)
    except BaseException as exc:
        p.write_new(OUT / 'FAILED.json', dict(error=repr(exc), when=p.now(), deletion_not_rolled_back=True))
        raise


if __name__ == '__main__': main()
