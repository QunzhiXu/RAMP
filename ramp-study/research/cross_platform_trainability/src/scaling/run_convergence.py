"""Where does RAMP's data-efficiency advantage end?

`design.json`'s reporting_commitments require the convergence point to be
reported: "Presenting an efficiency gain while omitting where the advantage ends
would let a reader infer an advantage at all scales." The per-class sweep cannot
answer it — holding every class exactly balanced caps the draw at what the
rarest class can supply, roughly 10,000 cells, well below the 100k the design
contemplates.

So this sweep draws at the training platform's own **natural class
proportions**. The proportions are then constant across sizes, which removes the
confound that broke the first attempt (a draw that silently de-balances as it
grows), while letting the size run to every labelled cell the platform has.
Macro F1 is lower than on balanced draws for every arm; only the comparison
between arms is read.
"""
import sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, balanced_accuracy_score
import common, jobs, models, paths

D = common.D
LEVELS = [200, 500, 1000, 2000, 5000, 10000, 20000, 'max']
CAP = 'linear'
ARMS = [a for a in common.ARMS if a != 'MaxFuse']


def proportional_draw(pool, y, n, rng):
    """Sample n cells keeping the pool's own class proportions."""
    cls, cnt = np.unique(y[pool], return_counts=True)
    quota = np.maximum(1, np.floor(cnt / cnt.sum() * n).astype(int))
    take = []
    for c, q in zip(cls, quota):
        ic = pool[y[pool] == c]
        take.append(rng.choice(ic, min(q, len(ic)), replace=False))
    return np.concatenate(take)


def run(task, reps=3):
    meta = common.load_meta(task)
    _, _, arms = common.load_arms(task, arms=ARMS)
    ylab = meta.celltype.to_numpy(); ds = meta.dataset.to_numpy()
    tab = pd.crosstab(meta.celltype, meta.dataset)
    shared = [c for c in tab.index[(tab > 0).all(1)] if c not in common.AMBIGUOUS]
    ok = np.isin(ylab, shared)
    plats = sorted(meta.dataset.unique())
    rows, t0 = [], time.time()
    for held in plats:
        for train_p in [p for p in plats if p != held]:
            pool = np.flatnonzero(ok & (ds == train_p))
            for rep in range(reps):
                rng = np.random.default_rng(D['protocol']['seed_base'] + rep)
                te = jobs.balanced_draw(np.flatnonzero(ok & (ds == held)), ylab, 12000, rng)
                seen = set()
                for level in LEVELS:
                    n = len(pool) if level == 'max' else int(level)
                    if n > len(pool):
                        n = len(pool)
                    tr = proportional_draw(pool, ylab, n, rng)
                    if len(tr) in seen:
                        continue          # 'max' duplicating the largest level
                    seen.add(len(tr))
                    cls = sorted(set(ylab[tr]))
                    if len(cls) < 2:
                        continue
                    ymap = {c: i for i, c in enumerate(cls)}
                    keep = np.array([i for i, c in enumerate(ylab[te]) if c in ymap])
                    ytr = np.array([ymap[c] for c in ylab[tr]])
                    yte = np.array([ymap[c] for c in ylab[te][keep]])
                    for arm in ARMS:
                        M = np.asarray(arms[arm], float)
                        Xtr, Xte = jobs._scale(M[tr], M[te])
                        m = models.clf(CAP, jobs._seed(task, 'conv', arm, level,
                                                       held, train_p, rep))
                        m.fit(Xtr, ytr)
                        pred = m.predict(Xte[keep])
                        rows.append(dict(
                            task=task, arm=arm, rep=rep, held_out=held,
                            train_platform=train_p, level=str(level),
                            n_train=len(tr), sampling='natural_proportion',
                            job='job1_celltype',
                            macro_f1=f1_score(yte, pred, average='macro'),
                            balanced_accuracy=balanced_accuracy_score(yte, pred)))
            print(f'  {task} held={held} train={train_p} rows={len(rows)} '
                  f'{time.time()-t0:.0f}s', flush=True)
    df = pd.DataFrame(rows)
    out = paths.ROOT / 'results' / 'convergence' / f'{task}.csv.gz'
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f'{task}: {len(df)} rows -> {out}  ({time.time()-t0:.0f}s)', flush=True)
    return df


if __name__ == '__main__':
    run(sys.argv[1], reps=int(sys.argv[2]) if len(sys.argv) > 2 else 3)
