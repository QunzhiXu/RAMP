"""The S2 data sweep, which the main grid never ran.

`design.json` names M3 (data efficiency) a *co-primary* metric alongside M1 and
argues for it explicitly — "reaches the same accuracy with N times fewer cells
is a decision, where +0.05 macro F1 is a number". But the grid only ever trained
at N, 2N and N+N, so M3 has never been computed. This sweeps the declared
levels so M3 can be read, and fits a power law to the same curves so the
*slope* — how fast each arm converts cells into accuracy — can be compared as
well as the level.
"""
import sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import numpy as np, pandas as pd
from sklearn.metrics import (f1_score, balanced_accuracy_score,
                             average_precision_score, roc_auc_score)
import common, jobs, models, paths, run_task

D = common.D
# Amendment 10: the sweep is per CLASS, not per total.
#
# jobs.balanced_draw takes a total and splits it, which has two effects that
# corrupt a learning curve.  Its floor of 20 per class makes every request below
# ~140 cells return the same draw, so the region where these models are still
# learning is never probed.  And above ~4,000 the rarer classes run out, so the
# draw silently becomes imbalanced -- max/min class ratio 1.0 at 4,000 but 3.2
# at 16,000 and 4.5 at full size on tnbc.  A curve fitted across that range is
# measuring class balance as much as data volume.
#
# Drawing a fixed number PER CLASS keeps every point exactly balanced, probes
# two decades below the old floor, and is comparable across tasks with
# different class counts.  Points where a class cannot supply the quota are
# recorded as balanced=False and dropped from the fits.
LEVELS = [5, 10, 20, 50, 100, 200, 500, 1000, 2000]
REPS = D['protocol']['repetitions']
CAP = 'linear'                      # the capacity M1 is read at
ARMS = [a for a in common.ARMS if a != 'MaxFuse']


def run(task, reps=REPS):
    meta = common.load_meta(task)
    _, _, arms = common.load_arms(task, arms=ARMS)
    ylab = meta.celltype.to_numpy(); ds = meta.dataset.to_numpy()
    tab = pd.crosstab(meta.celltype, meta.dataset)
    shared = [c for c in tab.index[(tab > 0).all(1)] if c not in common.AMBIGUOUS]
    ok = np.isin(ylab, shared)
    rares = run_task.rare_targets(meta)
    plats = sorted(meta.dataset.unique())
    rows, t0 = [], time.time()

    for held in plats:
        for train_p in [p for p in plats if p != held]:
            pool = np.flatnonzero(ok & (ds == train_p))
            rt = [r for r in rares
                  if r['abundant_in'] == train_p and r['rare_in'] == held]
            for rep in range(reps):
                rng = np.random.default_rng(D['protocol']['seed_base'] + rep)
                te = jobs.balanced_draw(np.flatnonzero(ok & (ds == held)), ylab, 12000, rng)
                pool_cls = sorted(set(ylab[pool]))
                for level in LEVELS:
                    take, full = [], True
                    for c in pool_cls:
                        ic = pool[ylab[pool] == c]
                        if len(ic) < level:
                            full = False
                        take.append(rng.choice(ic, min(level, len(ic)), replace=False))
                    tr = np.concatenate(take)
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
                        base = dict(task=task, arm=arm, capacity=CAP, rep=rep,
                                    held_out=held, train_platform=train_p,
                                    level=int(level), per_class=int(level),
                                    balanced=full, n_train=len(tr),
                                    n_markers=M.shape[1])
                        m = models.clf(CAP, jobs._seed(task, 's2', arm, level,
                                                       held, train_p, rep))
                        m.fit(Xtr, ytr)
                        pred = m.predict(Xte[keep])
                        rows.append(base | dict(
                            job='job1_celltype', n_classes=len(cls),
                            macro_f1=f1_score(yte, pred, average='macro'),
                            balanced_accuracy=balanced_accuracy_score(yte, pred)))
                        for r in rt:
                            yb = (ylab == r['label']).astype(int)
                            if yb[tr].sum() < 2 or yb[te].sum() < 1:
                                continue
                            mb = models.clf(CAP, jobs._seed(task, 's2r', arm, level,
                                                            held, train_p, rep))
                            mb.fit(Xtr, yb[tr])
                            s = mb.predict_proba(Xte)[:, 1]
                            rows.append(base | dict(
                                job='job4_rare', rare_label=r['label'],
                                rare_frac_test=float(yb[te].mean()),
                                average_precision=average_precision_score(yb[te], s),
                                auroc=roc_auc_score(yb[te], s)))
            print(f'  {task} held={held} train={train_p} rows={len(rows)} '
                  f'{time.time()-t0:.0f}s', flush=True)

    df = pd.DataFrame(rows)
    out = paths.ROOT / 'results' / 'scaling' / f'{task}.csv.gz'
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f'{task}: {len(df)} rows -> {out}  ({time.time()-t0:.0f}s)', flush=True)
    return df


if __name__ == '__main__':
    run(sys.argv[1], reps=int(sys.argv[2]) if len(sys.argv) > 2 else REPS)
