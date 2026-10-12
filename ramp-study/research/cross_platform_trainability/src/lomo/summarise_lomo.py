"""Compare Job 2 refitted without the target marker against the main grid.

The main-grid row is `job2_marker_transfer`, where every arm was fitted on the
full panel; the leave-one-out row is the same job after refitting each
alignment without the marker being predicted.  Arms are matched by name;
fastMNN appears as `fastMNN-reconstructed` in the main grid and
`fastMNN-embedding` here (see RECORD.md).
"""
import glob, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import numpy as np, pandas as pd, paths

ROOT = paths.ROOT
ORDER = ['Raw', 'Harmony-direct', 'CytoNorm', 'cyCombine', 'fastMNN-reconstructed',
         'fastMNN-embedding', 'VirTues-style', 'Spatium-style', 'RAMP']
PAIR = {'fastMNN-embedding': 'fastMNN-reconstructed'}


def _read(pattern):
    files = sorted(glob.glob(str(pattern)))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)


def tables():
    lomo = _read(ROOT / 'results/lomo/*.csv.gz')
    main = _read(ROOT / 'results/raw/*.csv.gz')
    main = main[main.job == 'job2_marker_transfer']
    return main, lomo


def pivot(d, cap='linear'):
    """Mean Spearman at ONE capacity.

    Taking the maximum would let each arm pick its best capacity, marker and
    transfer direction independently, which is optimistically biased and is the
    error already corrected once for M1 (amendment 7).
    """
    d = d[(d.train_set == 'A@N') & (d.capacity == cap)]
    return d.groupby(['task', 'arm']).spearman.mean().unstack()


def mean_rank(piv):
    return piv.rank(axis=1, ascending=False).mean().sort_values()


if __name__ == '__main__':
    main, lomo = tables()
    if lomo.empty:
        sys.exit('no leave-one-out results yet')
    print('=' * 78)
    print('JOB 2 — alignment refitted WITHOUT the marker being predicted')
    print(f'{lomo.task.nunique()} tasks, {lomo.marker.nunique()} markers, '
          f'{len(lomo):,} rows')
    CAP = 'linear'
    pl = pivot(lomo, CAP)
    cols = [a for a in ORDER if a in pl.columns]
    print(f'\nSpearman at capacity={CAP}, mean over markers, reps and directions:')
    print(pl[cols].round(3).to_string())
    rl = mean_rank(pl[cols])
    print('\nmean rank (1 = best):')
    print('  ' + '  '.join(f'{a} {v:.2f}' for a, v in rl.items()))

    if not main.empty:
        pm = pivot(main, CAP)
        shared = [t for t in pl.index if t in pm.index]
        print('\n' + '=' * 78)
        print('SAME JOB ON THE MAIN GRID (alignment fitted on the full panel)')
        cm = [a for a in ORDER if a in pm.columns]
        print(pm.loc[shared, cm].round(3).to_string())
        rm = mean_rank(pm.loc[shared, cm])
        print('\nmean rank (1 = best):')
        print('  ' + '  '.join(f'{a} {v:.2f}' for a, v in rm.items()))

        print('\n' + '=' * 78)
        print('CHANGE FROM REFITTING  (leave-one-out minus main grid)')
        rows = []
        for arm in pl.columns:
            m = PAIR.get(arm, arm)
            if m not in pm.columns:
                continue
            a, b = pl.loc[shared, arm], pm.loc[shared, m]
            rows.append(dict(arm=arm, main_grid=b.mean(), lomo=a.mean(),
                             delta=(a - b).mean(),
                             rank_main=rm.get(m, np.nan), rank_lomo=rl.get(arm, np.nan)))
        t = pd.DataFrame(rows).set_index('arm')
        print(t.reindex([a for a in ORDER if a in t.index]).round(3).to_string())

    print('\n' + '=' * 78)
    print('EVERY CAPACITY (leave-one-out, A@N, mean over tasks/markers/directions)')
    cs = lomo[lomo.train_set == 'A@N'].groupby(['capacity', 'arm']).spearman.mean().unstack()
    caps = ['knn', 'linear'] + [f'mlp{h}' for h in (4, 8, 16, 32, 64, 128, 256)]
    cs = cs.reindex([c for c in caps if c in cs.index])
    cols = [a for a in ORDER if a in cs.columns]
    print(cs[cols].round(3).to_string())
    print('\nwins per capacity (arm with the highest mean):')
    print('  ' + cs[cols].idxmax(axis=1).to_string().replace('\n', '\n  '))

    print('\n' + '=' * 78)
    print('PER-MARKER WIN COUNT (leave-one-out, A@N, linear)')
    d = lomo[(lomo.train_set == 'A@N') & (lomo.capacity == CAP)]
    w = d.groupby(['task', 'marker', 'arm']).spearman.mean().unstack()
    w = w[[a for a in ORDER if a in w.columns]]
    print(w.idxmax(axis=1).value_counts().to_string())
    print(f'  ({len(w)} marker-tasks)')
