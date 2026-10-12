"""Read-only run inventory and baseline verification after the disk-reserve stop."""
from collections import Counter
import sys
sys.dont_write_bytecode = True
from full_common import *
import setup_phase


def capture():
    lockdir = HERE / 'locks/full_20261011'
    run = HERE / 'runs/full_20261011'
    root = load_lock(lockdir)
    require((run / 'FAILED.json').is_file() and not (run / 'COMPLETE.json').exists(),
            'Expected a failed, incomplete full batch')
    require(not (run / 'BATCH_MANIFEST.json').exists(), 'Unexpected full batch manifest')
    require(read(run / 'BATCH_REQUEST.json')['code_records'] == codes(), 'Fitting code changed')
    archive = HERE / 'validation/execution_preflight_20261011/controller'
    # The launch archive preserves the same relative paths as the guarded code set.
    for name, expected in codes().items():
        require(record(archive / name) == expected, 'Archived fitting code differs')
    complete, failed, pending = {}, {}, []
    counts = Counter()
    unsealed_predictions = 0
    for ident in root['groups']:
        path = run / ident
        group = group_lock(lockdir, root, ident)
        if (path / 'REQUEST.json').exists():
            require(read(path / 'REQUEST.json')['code_records'] == codes(), 'Group controller differs')
        if (path / 'COMPLETE.json').exists():
            require(not (path / 'FAILED.json').exists(), 'Conflicting group completion')
            require(read(path / 'COMPLETE.json') == dict(status='complete', manifest=record(path / 'RUN_MANIFEST.json')),
                    'Group completion binding differs')
            manifest = read(path / 'RUN_MANIFEST.json')
            require(record(path / 'WORKER_DONE.json') == manifest['files']['WORKER_DONE.json'], 'Group summary changed')
            done = read(path / 'WORKER_DONE.json')
            check_grid(list(done['jobs']), group['tasks'])
            require(done['code_records'] == codes(), 'Completed group controller differs')
            complete[ident] = dict(jobs=len(done['jobs']), manifest=record(path / 'RUN_MANIFEST.json'))
            for row in group['tasks'].values():
                job = row['job']; counts[job['task_type'] + '__' + job['task']] += 1
        elif (path / 'FAILED.json').exists():
            error = read(path / 'FAILED.json')
            require(error == dict(error="ValueError('Disk reserve insufficient')"), 'Unexpected worker failure')
            partial = len(list((path / 'runtime/results').glob('*/predictions.npz')))
            failed[ident] = dict(error=error['error'], evidence=record(path / 'FAILED.json'),
                                 unsealed_prediction_files=partial)
            unsealed_predictions += partial
        else:
            require(not path.exists(), 'Unexpected unsealed group without a failure record')
            pending.append(ident)
    require(sum(x['jobs'] for x in complete.values()) == 31560, 'Sealed job count differs')
    require(len(complete) == 845 and len(failed) == 9 and len(pending) == 6, 'Interruption grid differs')
    baseline = setup_phase.capture()
    require(baseline == read(HERE / 'validation/preflight_20261011/report.json'), 'Previously accepted files changed')
    finalizer = HERE / 'validation/finalization_20261011'
    require(read(finalizer / 'FAILED.json') == dict(error="ValueError('Full batch failed; finalization not started')"),
            'Unexpected finalizer outcome')
    require(not (finalizer / 'COMPLETE.json').exists(), 'Incomplete batch was finalized')
    for name in ('comparison_20261011', 'aggregation_20261011', 'handoff_20261011'):
        require(not (HERE / 'validation' / name).exists(), 'Unexpected completion evidence')
    result = dict(status='blocked_disk_reserve_not_complete', target_jobs=32370, sealed_jobs=31560,
                  jobs_not_accepted=810, sealed_groups=complete, failed_groups=failed, not_started_groups=pending,
                  unsealed_prediction_files=unsealed_predictions, sealed_branch_counts=counts,
                  free_bytes=shutil.disk_usage(HERE).free, reserve_bytes=RESERVE_BYTES,
                  old_trainability_files=len(baseline['trainability_files']),
                  external_non_run_files=len(baseline['external']), prior=baseline['prior'],
                  batch_failure=record(run / 'FAILED.json'), finalizer_failure=record(finalizer / 'FAILED.json'),
                  code_records=codes(), inventory_code=record(Path(__file__)),
                  limits='Completion-marker/summary inventory, not a fresh full-payload hash or numerical comparison. '
                         'No full comparison or published-table aggregation ran. Partial unsealed outputs are not accepted.')
    out = reserve(HERE / 'validation/interruption_20261011', [run, lockdir, finalizer])
    write(out / 'report.json', result)
    seal(out, dict(scope='interruption_inventory_only', batch_status='failed'))
    print({k: result[k] for k in ('status', 'sealed_jobs', 'jobs_not_accepted', 'unsealed_prediction_files',
                                  'old_trainability_files', 'external_non_run_files')})


if __name__ == '__main__':
    capture()
