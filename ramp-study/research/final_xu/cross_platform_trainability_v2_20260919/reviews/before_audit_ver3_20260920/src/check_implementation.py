"""Run tiny synthetic checks and write a traceable report, never real-data fits."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import platform
import sys
import time
import unittest

sys.dont_write_bytecode = True
from trainability.core import ROOT, file_sha256, output_path, require, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='validation/checks_v0_1')
    args = parser.parse_args()
    out = output_path(args.out)
    require(not out.exists(), 'Check output exists; use a new version')
    out.mkdir(parents=True)
    start = time.perf_counter()
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    with (out/'unittest.log').open('x') as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    files = sorted((ROOT/'src').rglob('*.py')) + sorted((ROOT/'tests').glob('*.py'))
    files += sorted((ROOT/'config').glob('*.json')) + [ROOT/'MOTIVATION_AND_DESIGN.md']
    report = dict(status='passed' if result.wasSuccessful() else 'failed',
        tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        skipped=len(result.skipped), seconds=time.perf_counter()-start,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        scope='Synthetic protocol tests and tiny synthetic head fits only',
        real_alignment_run=False, real_model_training_run=False, formal_experiment_run=False,
        python=platform.python_version(), executable=sys.executable,
        versions={p:version(p) for p in ['numpy','pandas','scipy','scikit-learn','torch']},
        sha256={str(p.relative_to(ROOT)):file_sha256(p) for p in files})
    write_json(out/'REPORT.json',report)
    print(f"{report['status']}: {report['tests']} synthetic tests; {report['failures']} failures, {report['errors']} errors")
    print(f'Report: {out}/REPORT.json')
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
