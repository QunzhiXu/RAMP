"""Leave-one-marker-out refit: every method, every marker, every task.

For each (task, held-out marker) the alignment is re-run from scratch on the
remaining panel, so no arm ever sees the marker it will later be asked to
predict.  Each run's corrected values are stored row-aligned to the task
metadata; the staging directory is deleted afterwards.
"""
import json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import numpy as np, pandas as pd
import resolve, prepare, prepare_story, ramp_run, paths, common

PROJECT = HERE.parents[1]
WORK = PROJECT / 'lomo_work'
OUT = WORK / 'out'
LOGS = PROJECT / 'results' / 'lomo_logs'
BENCH = set(resolve.BENCH)
METHODS = ['ramp', 'harmony', 'cycombine', 'cytonorm', 'fastmnn']
# file holding each method's corrected values, and whether it is an embedding
VALUES = {'fastmnn': ('fastmnn_corrected_embedding.csv.gz', True),
          'cycombine': ('corrected_expression.csv.gz', False),
          'cytonorm': ('corrected_expression.csv.gz', False),
          'harmony': ('harmony_adjusted_marker_coordinates.csv.gz', False)}


def markers_of(task):
    return ramp_run.spec(task)['markers']


def jobs():
    return [(t, m, meth) for t in resolve.TASKS for m in markers_of(t) for meth in METHODS]


def _reindex(values, ids, order):
    if ids is None:
        if len(values) != len(order):
            raise ValueError(f'{len(values)} rows for {len(order)} cells and no ids')
        return values
    ids = np.asarray(ids).astype(str)
    if set(ids) != set(order):
        raise ValueError('cell id sets differ')
    pos = pd.Series(np.arange(len(ids)), index=ids)
    return values[pos.reindex(order).to_numpy()]


def _read_values(path, keep, embedding):
    df = pd.read_csv(path, low_memory=False)
    ids = None
    for c in ('cell_id', 'cell_ids', 'CellID'):
        if c in df.columns:
            ids = df[c].astype(str).to_numpy(); df = df.drop(columns=[c]); break
    if not embedding:
        cols = [c for c in keep if c in df.columns]
        if len(cols) != len(keep):
            raise ValueError(f'expected {len(keep)} markers, matched {len(cols)}')
        df = df[cols]
    else:
        df = df.select_dtypes('number')
    return df.to_numpy(np.float64), ids


def run_one(task, drop, method):
    dest = OUT / task / drop
    dest.mkdir(parents=True, exist_ok=True)
    tag = f'{task}__{drop}__{method}'
    npy, meta_json = dest / f'{method}.npy', dest / f'{method}.json'
    if npy.exists() and meta_json.exists():
        return tag, 'cached', 0.0
    order = common.load_meta(task).cell_id.astype(str).to_numpy()
    t0 = time.time()
    work = WORK / 'runs' / tag
    try:
        if method == 'ramp':
            keep, aligned, _ = ramp_run.fit(task, drop)
            values, ids = aligned, None
        elif task in BENCH:
            p, r = prepare.run(task, method, drop, work)
            if r.returncode:
                raise RuntimeError((r.stderr or r.stdout or '')[-1500:])
            keep = Path(p['markers']).read_text().split()
            name, emb = VALUES[method]
            values, ids = _read_values(Path(p['out']) / name, keep, emb)
        else:
            p, r = prepare_story.run(task, method, drop, work)
            if r.returncode:
                raise RuntimeError((r.stderr or r.stdout or '')[-1500:])
            keep = p['markers']
            values, ids = _read_values(Path(p['out']) / 'corrected_expression.csv.gz',
                                       keep, method == 'fastmnn')
        values = _reindex(np.asarray(values, float), ids, order)
        if not np.isfinite(values).all():
            raise ValueError('non-finite corrected values')
        np.save(npy, values.astype(np.float32))
        meta_json.write_text(json.dumps(
            dict(task=task, held_out=drop, method=method, markers=keep,
                 shape=list(values.shape), seconds=round(time.time() - t0, 1),
                 embedding=(method == 'fastmnn')), indent=1) + '\n')
        return tag, 'ok', time.time() - t0
    except Exception as exc:
        (LOGS / f'FAIL_{tag}.txt').write_text(
            f'{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}')
        return tag, f'FAIL {type(exc).__name__}', time.time() - t0
    finally:
        shutil.rmtree(work, ignore_errors=True)
        if method == 'cytonorm':
            shutil.rmtree(WORK / 'algorithm_test' / f'lomo_{task}_{drop}', ignore_errors=True)


def _worker(args):
    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMBA_NUM_THREADS', 'LOKY_MAX_CPU_COUNT'):
        os.environ[var] = '1'
    return run_one(*args)


if __name__ == '__main__':
    import multiprocessing as mp
    LOGS.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    todo = jobs()
    if len(sys.argv) > 1 and sys.argv[1] != 'all':
        todo = [j for j in todo if j[0] in sys.argv[1:]]
    n = int(os.environ.get('LOMO_WORKERS', '6'))
    print(f'{len(todo)} runs on {n} workers', flush=True)
    t0 = time.time(); done = 0
    with mp.Pool(n, maxtasksperchild=1) as pool:
        for tag, status, secs in pool.imap_unordered(_worker, todo):
            done += 1
            print(f'[{done}/{len(todo)} {(time.time()-t0)/60:6.1f}m] {status:16s} '
                  f'{secs:6.1f}s  {tag}', flush=True)
    print('finished', flush=True)
