"""Real synthetic fresh-fit smoke, in owned output only; no cohort fitting."""
import argparse
import os
from pathlib import Path
import study as s
from bootstrap import r, previous_worker as w
from worker import fresh_fit, compare_coordinates


def main():
    p = argparse.ArgumentParser(); p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    s.inputs(a.bundle)
    out = r.reserve(a.output, [a.bundle])
    os.environ.update({k: v for k, v in r.environment(out).items() if k in (*r.THREADS, 'MPLCONFIGDIR', 'NUMBA_CACHE_DIR',
        'XDG_CACHE_HOME', 'RAMP_EVALUATION_CACHE', 'MPLBACKEND')})
    e = w.load_evaluator(a.bundle)
    before = w.runtime()
    import numpy as np
    values = e.standardized(np.random.default_rng(20261011).normal(size=(600, 8)), 'Unaligned')
    a1 = fresh_fit(e, values, 'synthetic', r.reserve(out / 'first'))
    a2 = fresh_fit(e, values, 'synthetic', r.reserve(out / 'second'))
    report = compare_coordinates(a1, a2, e.sha256_array)
    r.require(report['array_equal'], 'Repeated synthetic fit differs')
    r.require(before == w.runtime(), 'Synthetic environment changed')
    r.write(out / 'report.json', {'status': 'passed', 'comparison': report, 'runtime': before,
                                 'code_records': s.codes(), 'protocol': e.UMAP_PROTOCOL})
    print(report)


if __name__ == '__main__': main()
