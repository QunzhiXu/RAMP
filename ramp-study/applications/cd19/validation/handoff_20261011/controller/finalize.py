"""Wait for the single CD19 batch, then run reporting-only checks."""
from pathlib import Path
import subprocess
import sys
import time
from common import HERE, reserve, write, sha, read


def main():
    run = HERE / 'runs/rebuild_20261011_r2'
    evidence = reserve(HERE / 'validation/finalization_20261011', [run, HERE / 'frozen_20261011'])
    write(evidence / 'REQUEST.json', {'run': str(run), 'fitting_permitted': False, 'script_sha256': sha(__file__)})
    try:
        while not (run / 'COMPLETE.json').exists():
            if (run / 'FAILED.json').exists():
                raise ValueError(f'Fitting batch failed: {read(run / "FAILED.json")}')
            time.sleep(15)
        commands = [
            ['compare.py', '--lockdir', 'frozen_20261011', '--run', str(run), '--out', 'validation/comparison_20261011'],
            ['figure_metrics.py', 'run', '--lockdir', 'frozen_20261011', '--supplementary', 'figure_bindings_20261011', '--run', str(run), '--out', 'validation/figure_metrics_20261011'],
            ['validate.py', '--lockdir', 'frozen_20261011', '--run', str(run), '--comparison', 'validation/comparison_20261011', '--figure-metrics', 'validation/figure_metrics_20261011', '--out', 'validation/handoff_20261011'],
        ]
        for i, args in enumerate(commands):
            command = [sys.executable, '-B', *args]
            write(evidence / f'command_{i}.json', command)
            print(f'FINALIZE_START {args[0]}', flush=True)
            with (evidence / f'{i}.stdout').open('x') as stdout, (evidence / f'{i}.stderr').open('x') as stderr:
                code = subprocess.call(command, cwd=HERE, stdout=stdout, stderr=stderr)
            if code:
                raise RuntimeError(f'{args[0]} returned {code}; retained logs')
            print(f'FINALIZE_COMPLETE {args[0]}', flush=True)
        write(evidence / 'COMPLETE.json', {'status': 'complete', 'handoff_report_sha256': sha(HERE / 'validation/handoff_20261011/report.json')})
    except BaseException as exc:
        write(evidence / 'FAILED.json', {'error': repr(exc)})
        raise


if __name__ == '__main__':
    main()
