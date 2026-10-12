"""M3 (data efficiency) and the power-law slope of each arm's learning curve."""
import glob, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import numpy as np, pandas as pd, paths

ORDER = ['Raw', 'Harmony-direct', 'CytoNorm', 'cyCombine', 'fastMNN-reconstructed',
         'VirTues-style', 'Spatium-style', 'RAMP']
SCORE = {'job1_celltype': 'macro_f1', 'job4_rare': 'average_precision'}


def load():
    return pd.concat([pd.read_csv(f, low_memory=False)
                      for f in sorted(glob.glob(str(paths.ROOT / 'results/scaling/*.csv.gz')))],
                     ignore_index=True)


def curves(df, job, balanced_only=True):
    """Mean score per (task, arm, cells-per-class), over reps and directions."""
    d = df[df.job == job]
    if balanced_only and 'balanced' in d.columns:
        d = d[d.balanced]
    return d.groupby(['task', 'arm', 'per_class'])[SCORE[job]].mean().reset_index()


def m3(df, job, frac=0.95, relative_to='task'):
    """Fewest cells PER CLASS reaching `frac` of a reference score.

    relative_to='task' is the rule design.json freezes: the reference is the
    best score any arm reaches on that task.  That rule hands the arm which
    SETS the reference a structural advantage — it only has to reach 95 % of
    itself — so relative_to='arm' is reported beside it, scoring each arm
    against its own plateau.  The first reads "who gets to a good model
    cheapest", the second "who converts cells into accuracy fastest".
    """
    c = curves(df, job)
    out = {}
    for task, g in c.groupby('task'):
        best = g[SCORE[job]].max()
        row = {}
        for arm, ga in g.groupby('arm'):
            ref = best if relative_to == 'task' else ga[SCORE[job]].max()
            hit = ga[ga[SCORE[job]] >= frac * ref].per_class
            row[arm] = float(hit.min()) if len(hit) else np.nan
        out[task] = row
    return pd.DataFrame(out).T


def slopes(df, job):
    """Power-law exponent b in score = a * n^b, fitted per (task, arm)."""
    c = curves(df, job)
    rows = []
    for (task, arm), g in c.groupby(['task', 'arm']):
        g = g[(g[SCORE[job]] > 0) & (g.per_class > 0)]
        if len(g) < 3:
            continue
        b, a = np.polyfit(np.log10(g.per_class), np.log10(g[SCORE[job]]), 1)
        rows.append(dict(task=task, arm=arm, slope=b, intercept=a,
                         span=g[SCORE[job]].max() - g[SCORE[job]].min(),
                         n_levels=len(g)))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    df = load()
    print(f'{len(df):,} rows | tasks {df.task.nunique()} | arms {df.arm.nunique()} '
          f'| levels {sorted(df.level.unique())}')
    for job in ['job1_celltype', 'job4_rare']:
        if job not in set(df.job):
            continue
        print('\n' + '=' * 76)
        for rel, label in (('task', "95% of the TASK best (design.json rule)"),
                           ('arm', "95% of the ARM's own plateau")):
            t = m3(df, job, relative_to=rel)
            cols = [a for a in ORDER if a in t.columns]
            print(f'M3 — cells PER CLASS to reach {label}  ({job})')
            print(t[cols].to_string())
            med = t[cols].median()
            print('median over tasks (lower is better), NaN = never reached:')
            print('  ' + '  '.join(f'{a} {v:,.0f}' for a, v in med.sort_values().items()))
            miss = {a: int(t[a].isna().sum()) for a in cols if t[a].isna().any()}
            if miss:
                print('  never reached (of 7 tasks):', miss)
            print()

        s = slopes(df, job)
        print(f'\nPOWER-LAW SLOPE  ({job}) — how fast cells convert to accuracy')
        p = s.pivot(index='task', columns='arm', values='slope')
        print(p[[a for a in cols if a in p.columns]].round(3).to_string())
        print('\nmean slope (higher = steeper):')
        ms = p[[a for a in cols if a in p.columns]].mean().sort_values(ascending=False)
        print('  ' + '  '.join(f'{a} {v:+.3f}' for a, v in ms.items()))
