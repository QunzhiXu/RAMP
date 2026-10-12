"""Read M1-M4 off the grid using the rules frozen in design.json."""
import sys, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd, common, paths

D = common.D
ORDER = ['Raw', 'Harmony-direct', 'CytoNorm', 'cyCombine', 'fastMNN-reconstructed',
         'MaxFuse', 'VirTues-style', 'Spatium-style', 'RAMP']
CAPS = D['sweeps']['S1_capacity']['levels']
SCORE = {'job1_celltype': 'macro_f1', 'job2_masked_marker': 'spearman',
         'job3_platform': 'balanced_accuracy', 'job4_rare': 'average_precision'}

df = pd.concat([pd.read_csv(f, low_memory=False) for f in
                sorted(glob.glob(str(paths.ROOT / 'results/raw/*.csv.gz')))], ignore_index=True)


def m1(job, cap=None):
    """Accuracy at a fixed capacity, averaged over reps and transfer directions."""
    d = df[df.job == job]
    if job != 'job3_platform':
        d = d[d.train_set == 'A@N'] if 'train_set' in d else d
    if cap: d = d[d.capacity == cap]
    return d.groupby(['task', 'arm'])[SCORE[job]].mean().unstack()


def m2(job):
    """Smallest model reaching 95% of the best any arm achieves on that task."""
    d = df[(df.job == job)]
    if 'train_set' in d: d = d[d.train_set == 'A@N']
    out = {}
    for task, g in d.groupby('task'):
        piv = g.groupby(['arm', 'capacity'])[SCORE[job]].mean().unstack()[CAPS]
        thr = 0.95 * piv.max().max()
        row = {}
        for arm in piv.index:
            hit = [c for c in CAPS if piv.loc[arm, c] >= thr]
            row[arm] = (g[g.capacity == hit[0]].params.iloc[0] if hit else np.nan)
        out[task] = row
    return pd.DataFrame(out).T


def m4(job):
    """Second-cohort gain: A+B@2N minus A@2N."""
    d = df[(df.job == job) & (df.capacity == 'linear')]
    if 'train_set' not in d: return None
    key = ['task', 'held_out', 'train_platform', 'arm', 'rep'] + \
          (['marker'] if job == 'job2_masked_marker' else [])
    w = d.pivot_table(index=key, columns='train_set', values=SCORE[job]).reset_index()
    if 'A+B@2N' not in w: return None
    w['gain'] = w['A+B@2N'] - w['A@2N']
    return w


def mean_rank(piv, lower_better=False):
    r = piv.rank(axis=1, ascending=lower_better)
    return r.mean().sort_values()


if __name__ == '__main__':
    print('=' * 78)
    print('M1  ACCURACY — best capacity per arm, mean over reps and directions')
    for job in ['job1_celltype', 'job2_masked_marker', 'job4_rare']:
        d = df[df.job == job]
        d = d[d.train_set == 'A@N'] if 'train_set' in d else d
        piv = d.groupby(['task', 'arm'])[SCORE[job]].max().unstack()
        cols = [a for a in ORDER if a in piv.columns]
        print(f'\n--- {job}  ({SCORE[job]}) ---')
        print(piv[cols].round(3).to_string())
        print('mean rank (1 = best):')
        print('  ' + '  '.join(f'{a} {v:.1f}' for a, v in mean_rank(piv[cols]).items()))

    print('\n' + '=' * 78)
    print('JOB 3  PLATFORM DISCRIMINATION — control, LOWER is better')
    p3 = df[df.job == 'job3_platform'].groupby(['task', 'arm']).balanced_accuracy.mean().unstack()
    cols = [a for a in ORDER if a in p3.columns]
    print(p3[cols].round(3).to_string())
    print('chance:', df[df.job == 'job3_platform'].groupby('task').chance.first().round(3).to_dict())

    print('\n' + '=' * 78)
    print('M2  PARAMETERS to reach 95% of the task best')
    print(m2('job1_celltype').reindex(columns=[a for a in ORDER]).to_string())

    print('\n' + '=' * 78)
    print('M4  SECOND-COHORT GAIN (A+B@2N - A@2N), linear')
    for job in ['job1_celltype', 'job2_masked_marker']:
        w = m4(job)
        if w is None: continue
        print(f'\n--- {job} ---')
        t = w.groupby('arm').gain.agg(['mean', 'median'])
        t['% helps'] = w.groupby('arm').gain.apply(lambda s: (s > 0).mean() * 100)
        print(t.reindex([a for a in ORDER if a in t.index]).round(3).to_string())
