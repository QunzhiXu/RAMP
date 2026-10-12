"""Cross-platform cell-type annotation: accuracy, model capacity, and whether
adding a second training cohort helps.

For every benchmark and every choice of held-out platform P3:
    train on P1 alone, on P2 alone, and on P1+P2 pooled
    evaluate on P3, which the model has never seen
under four input representations and six model capacities.
"""
import numpy as np, pandas as pd, warnings, itertools, sys
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
import common

warnings.filterwarnings('ignore')
RNG = np.random.default_rng(20260917)
N_TRAIN, N_TEST = 12000, 12000
CAPACITY = [None, 4, 8, 16, 32, 64]          # None = multinomial logistic


def n_params(h, d, k):
    return d * k + k if h is None else d * h + h + h * k + k


def fit_predict(Xtr, ytr, Xte, h):
    if h is None:
        m = LogisticRegression(max_iter=2000, C=1.0)
    else:
        m = MLPClassifier(hidden_layer_sizes=(h,), max_iter=400, random_state=0,
                          early_stopping=True, n_iter_no_change=12)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    m.fit((Xtr - mu) / sd, ytr)
    return m.predict((Xte - mu) / sd)


def subsample(idx, y, n):
    """Class-balanced subsample, capped by the rarest class."""
    cls = np.unique(y[idx])
    per = max(20, n // len(cls))
    out = []
    for c in cls:
        pool = idx[y[idx] == c]
        out.append(RNG.choice(pool, min(per, len(pool)), replace=False))
    return np.concatenate(out)


rows = []
for b in common.BENCH:
    meta, markers, raw, ramp = common.load(b)
    units, datasets = meta.unit_key.to_numpy(), meta.dataset.to_numpy()
    y = LabelEncoder().fit_transform(meta.celltype.to_numpy())
    ylab = meta.celltype.to_numpy()
    tab = pd.crosstab(meta.celltype, meta.dataset)
    shared = [c for c in tab.index[(tab > 0).all(1)] if c not in common.AMBIGUOUS]
    ok = np.isin(ylab, shared)
    reps = common.arms(raw, ramp, units, datasets)
    plats = sorted(set(datasets))
    print(f'{b}: {len(shared)} classes, platforms {plats}', flush=True)

    for held in plats:
        src = [p for p in plats if p != held]
        te = subsample(np.flatnonzero(ok & (datasets == held)), y, N_TEST)
        configs = {src[0]: [src[0]], src[1]: [src[1]], 'pooled': src}
        for cname, cplats in configs.items():
            tr_pool = np.flatnonzero(ok & np.isin(datasets, cplats))
            tr = subsample(tr_pool, y, N_TRAIN)
            for arm, M in reps.items():
                for h in CAPACITY:
                    pred = fit_predict(M[tr], y[tr], M[te], h)
                    rows.append(dict(benchmark=b, held_out=held, train=cname,
                        n_train_platforms=len(cplats), arm=arm,
                        capacity='linear' if h is None else f'mlp{h}',
                        params=n_params(h, M.shape[1], len(shared)),
                        macro_f1=f1_score(y[te], pred, average='macro'),
                        bal_acc=balanced_accuracy_score(y[te], pred),
                        n_train=len(tr), n_test=len(te), n_classes=len(shared)))
        print(f'   held out {held}: done', flush=True)

res = pd.DataFrame(rows)
res.to_csv('celltype_transfer_results.csv', index=False)
print(f'\n{len(res)} model fits written')
