"""Pretrain on one corpus, freeze, probe on a platform held out of everything.

Per amendment 9.  The corpora are the main grid's own M4 draws, so the gain
measured here (A+B@2N minus A@2N) is directly comparable to the supervised gain
already recorded.  Job 2 is deliberately absent: it is the pretraining
objective.
"""
import json, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (f1_score, balanced_accuracy_score,
                             average_precision_score, roc_auc_score)
import common, jobs, paths, run_task
import mae

torch.set_num_threads(1)
D = common.D
N = D['sweeps']['S3_composition']['N']
REPS = D['protocol']['repetitions']
ARMS = [a for a in common.ARMS if a != 'MaxFuse']


def _std(fit, X):
    mu, sd = fit.mean(0), fit.std(0)
    return (X - mu) / np.where(sd > 0, sd, 1.0)


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
        srcs = [p for p in plats if p != held]
        for train_p in srcs:
            other = [p for p in srcs if p != train_p]
            for rep in range(reps):
                rng = np.random.default_rng(D['protocol']['seed_base'] + rep)
                te = jobs.balanced_draw(np.flatnonzero(ok & (ds == held)), ylab, 12000, rng)
                iA = np.flatnonzero(ok & (ds == train_p))
                sets = {'A@N': jobs.balanced_draw(iA, ylab, N, rng),
                        'A@2N': jobs.balanced_draw(iA, ylab, 2 * N, rng)}
                if other:
                    iB = np.flatnonzero(ok & (ds == other[0]))
                    sets['A+B@2N'] = np.concatenate(
                        [jobs.balanced_draw(iA, ylab, N, rng),
                         jobs.balanced_draw(iB, ylab, N, rng)])
                rt = [r for r in rares
                      if r['abundant_in'] == train_p and r['rare_in'] == held]

                for arm in ARMS:
                    M = np.asarray(arms[arm], float)
                    for tag, tr in sets.items():
                        Xc = _std(M[tr], M[tr]).astype(np.float32)
                        Xt = _std(M[tr], M[te]).astype(np.float32)
                        seed = jobs._seed(task, 'pretrain', arm, tag, held, train_p, rep)
                        enc, vloss, eps = mae.pretrain(Xc, seed % (2 ** 31))
                        # the probe sees the latent CONCATENATED with the input.
                        # Feeding it the latent alone discards the raw features
                        # and made pretraining look harmful for every arm; see
                        # the amendment-9 revision in RECORD.md.
                        Ztr = np.hstack([mae.embed(enc, Xc), Xc])
                        Zte = np.hstack([mae.embed(enc, Xt), Xt])

                        base = dict(task=task, arm=arm, train_set=tag, rep=rep,
                                    held_out=held, train_platform=train_p,
                                    latent=mae.LATENT, pretrain_val_loss=vloss,
                                    pretrain_epochs=eps, n_train=len(tr),
                                    n_markers=M.shape[1])
                        cls = sorted(set(ylab[tr]))
                        ymap = {c: i for i, c in enumerate(cls)}
                        keep = np.array([i for i, c in enumerate(ylab[te]) if c in ymap])
                        m = LogisticRegression(max_iter=3000, C=1.0).fit(
                            Ztr, [ymap[c] for c in ylab[tr]])
                        pred = m.predict(Zte[keep])
                        yte = [ymap[c] for c in ylab[te][keep]]
                        rows.append(base | dict(
                            job='job1_celltype', n_classes=len(cls),
                            macro_f1=f1_score(yte, pred, average='macro'),
                            balanced_accuracy=balanced_accuracy_score(yte, pred)))

                        for r in rt:
                            yb = (ylab == r['label']).astype(int)
                            if yb[tr].sum() < 2 or yb[te].sum() < 1:
                                continue
                            mb = LogisticRegression(max_iter=3000, C=1.0).fit(Ztr, yb[tr])
                            s = mb.predict_proba(Zte)[:, 1]
                            rows.append(base | dict(
                                job='job4_rare', rare_label=r['label'],
                                rare_frac_test=float(yb[te].mean()),
                                average_precision=average_precision_score(yb[te], s),
                                auroc=roc_auc_score(yb[te], s),
                                f1_rare=f1_score(yb[te], mb.predict(Zte), pos_label=1)))
            print(f'  {task} held={held} train={train_p} rows={len(rows)} '
                  f'{time.time()-t0:.0f}s', flush=True)

    df = pd.DataFrame(rows)
    out = paths.ROOT / 'results' / 'pretrain' / f'{task}.csv.gz'
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f'{task}: {len(df)} rows -> {out}  ({time.time()-t0:.0f}s)', flush=True)
    return df


if __name__ == '__main__':
    run(sys.argv[1], reps=int(sys.argv[2]) if len(sys.argv) > 2 else REPS)
