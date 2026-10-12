"""M5 — label efficiency on the target platform.

design.json: fit on the source, then adapt with k labelled cells per class from
the target platform. k = 0, 1, 5, 20, 100.

Three protocols per (arm, k), so that what the source contributes is visible:
  source_only    train on the source alone                      (k = 0 case)
  source_plus_k  train on the source plus k target cells/class   (the design protocol)
  target_only_k  train on the k target cells alone               (control)

Evaluated on target cells held out of the k used for adaptation.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, balanced_accuracy_score
import common, jobs, models, paths

D = common.D
# amendment 7: the frozen k = [0,1,5,20,100] are all computed; intermediate
# values are added to locate the crossover — how many target labels make the
# source unnecessary — which is the interpretable form of M5.
K_VALUES = [0, 1, 2, 3, 5, 10, 20, 50, 100]
K_FROZEN = D['metrics']['M5_labels']['k']
N_SOURCE = D['sweeps']['S3_composition']['N']
REPS = D['protocol']['repetitions']
ARMS = [a for a in common.ARMS if a != 'MaxFuse']
CAPS = ['linear', 'knn']


def run(task, reps=None):
    reps = reps or REPS
    meta0 = common.load_meta(task)
    plats = sorted(meta0.dataset.unique())
    ylab = meta0.celltype.to_numpy(); ds = meta0.dataset.to_numpy()
    tab = pd.crosstab(meta0.celltype, meta0.dataset)
    shared = [c for c in tab.index[(tab > 0).all(1)] if c not in common.AMBIGUOUS]
    ok = np.isin(ylab, shared)
    rows, t0 = [], time.time()

    for held in plats:
        for train_p in [p for p in plats if p != held]:
            meta, _, arms = common.load_arms(task, train_platform=train_p, arms=ARMS)
            for rep in range(reps):
                rng = np.random.default_rng(D['protocol']['seed_base'] + 991 * rep)
                src = jobs.balanced_draw(np.flatnonzero(ok & (ds == train_p)),
                                         ylab, N_SOURCE, rng)
                tgt_pool = np.flatnonzero(ok & (ds == held))
                classes = sorted(set(ylab[src]) & set(ylab[tgt_pool]))
                if len(classes) < 2:
                    continue
                ymap = {c: i for i, c in enumerate(classes)}
                # adaptation cells, drawn once per rep; the rest is the test set
                by_class = {c: rng.permutation(tgt_pool[ylab[tgt_pool] == c]) for c in classes}
                kmax = max(K_VALUES)
                held_out = np.concatenate([v[kmax:] for v in by_class.values()])
                te = jobs.balanced_draw(held_out, ylab, 12000, rng)
                te = np.array([i for i in te if ylab[i] in ymap])
                yte = np.array([ymap[c] for c in ylab[te]])

                for k in K_VALUES:
                    adapt = (np.concatenate([v[:k] for v in by_class.values()])
                             if k else np.array([], dtype=int))
                    protocols = {'source_only': src}
                    if k:
                        protocols['source_plus_k'] = np.concatenate([src, adapt])
                        if k >= 2:
                            protocols['target_only_k'] = adapt
                            # class-weighted so k target cells are not swamped
                            # by the 6,000 source cells
                            protocols['source_plus_k_weighted'] = np.concatenate([src, adapt])
                    for proto, tr in protocols.items():
                        if len(set(ylab[tr])) < 2:
                            continue
                        ytr = np.array([ymap[c] for c in ylab[tr] if c in ymap])
                        tr2 = np.array([i for i in tr if ylab[i] in ymap])
                        for cap in CAPS:
                            if cap == 'knn' and len(tr2) < models.K_NEIGHBORS:
                                continue
                            for arm, M in arms.items():
                                Xtr, Xte = jobs._scale(M[tr2], M[te])
                                m = models.clf(cap, jobs._seed(task, 'm5', arm, cap, k, proto, rep))
                                if proto == 'source_plus_k_weighted' and cap == 'linear':
                                    w = np.where(np.isin(tr2, adapt), len(src) / max(len(adapt), 1), 1.0)
                                    m.fit(Xtr, ytr, sample_weight=w)
                                else:
                                    m.fit(Xtr, ytr)
                                pred = m.predict(Xte)
                                rows.append(dict(task=task, arm=arm, capacity=cap, k=k,
                                    protocol=proto, rep=rep, held_out=held,
                                    train_platform=train_p, n_train=len(tr2), n_test=len(te),
                                    n_classes=len(classes),
                                    macro_f1=f1_score(yte, pred, average='macro'),
                                    balanced_accuracy=balanced_accuracy_score(yte, pred)))
            print(f'  {task} held={held} train={train_p} rows={len(rows)} '
                  f'{time.time()-t0:.0f}s', flush=True)

    df = pd.DataFrame(rows)
    out = paths.ROOT / 'results/raw_m5' / f'{task}.csv.gz'
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f'{task}: {len(df)} rows ({time.time()-t0:.0f}s)')
    return df


if __name__ == '__main__':
    run(sys.argv[1], reps=(int(sys.argv[2]) if len(sys.argv) > 2 else None))
