"""Wait for the existing full batch, then run comparisons and handoff validation.

Does not fit, resume, or modify any branch. Fails on FAILED or a bounded timeout.
All report/log destinations are exclusively created.
"""
from pathlib import Path
import subprocess
import sys
import time
from bootstrap import r
import study as s


def main():
    batch = s.HERE / 'runs/seven_fresh_20261011'
    previous = s.HERE.parent / 'integration_evaluation'
    bundle = previous / 'runs/replay_inputs_20261011'; snapshot = previous / 'frozen_20261011_r2'
    out = r.reserve(s.HERE / 'validation/finalization_20261011')
    comparison = s.HERE / 'validation/seven_comparison_20261011'
    handoff = s.HERE / 'validation/handoff_20261011'
    try:
        r.require(not comparison.exists() and not handoff.exists(), 'Do not overwrite existing evidence')
        deadline = time.monotonic() + 6 * 3600; prior = None
        while not (batch / 'COMPLETE.json').exists():
            r.require(not (batch / 'FAILED.json').exists(), 'Full batch failed; preserve partial output')
            r.require(time.monotonic() < deadline, 'Finalization wait timed out; batch not terminated')
            coordinates = list(batch.glob('*/*/COORDINATE_COMPARISON.json'))
            status = {'fresh_fits': len(coordinates), 'exact_coordinates': sum(r.read(p)['array_equal'] for p in coordinates),
                      'finished_metric_branches': len(list(batch.glob('*/*/COMPLETE.json')))}
            if status != prior: print(status, flush=True); prior = status
            time.sleep(20)
        commands = [[sys.executable, '-B', str(s.HERE / 'compare.py'), '--batch', str(batch), '--bundle', str(bundle),
                     '--snapshot', str(snapshot), '--output', str(comparison)],
                    [sys.executable, '-B', str(s.HERE / 'validate.py'), '--batch', str(batch), '--bundle', str(bundle),
                     '--snapshot', str(snapshot), '--comparison', str(comparison), '--output', str(handoff), '--python', sys.executable]]
        r.write(out / 'COMMANDS.json', commands)
        for index, command in enumerate(commands):
            print('Finalization stage', index, 'started', flush=True)
            with (out / f'stage_{index}.stdout').open('x') as stdout, (out / f'stage_{index}.stderr').open('x') as stderr:
                result = subprocess.run(command, env=r.environment(out), stdout=stdout, stderr=stderr)
            r.require(result.returncode == 0, 'Finalization stage failed: ' + str(index))
            print('Finalization stage', index, 'passed', flush=True)
        r.write(out / 'COMPLETE.json', {'status': 'complete', 'comparison': r.record(comparison / 'report.json'),
                                      'handoff': r.record(handoff / 'report.json')})
    except BaseException as error:
        r.write(out / 'FAILED.json', {'error': repr(error)}); raise


if __name__ == '__main__': main()
