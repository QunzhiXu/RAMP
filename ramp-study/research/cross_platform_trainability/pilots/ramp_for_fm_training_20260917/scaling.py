"""Does adding a second cohort help, beyond simply adding more cells?

Three training sets per comparison, evaluated on a third, never-seen platform:
    A@N      N cells from platform A
    A@2N     2N cells from platform A      (more data, same platform)
    A+B@2N   N from A plus N from B        (more data, one more platform)

A+B@2N vs A@2N isolates what the second cohort contributes beyond its cell
count. A@2N vs A@N is the ordinary data-scaling curve.
"""
import numpy as np, pandas as pd, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import common

warnings.filterwarnings('ignore')
RNG = np.random.default_rng(20260917)
N, N_TEST, REPS = 6000, 12000, 3


def bal(idx, y, n):
    cls = np.unique(y[idx]); per = max(20, n // len(cls)); out = []
    for c in cls:
        pool = idx[y[idx] == c]
        out.append(RNG.choice(pool, min(per, len(pool)), replace=False))
    return np.concatenate(out)


def fit_score(Xtr, ytr, Xte, yte):
    mu, sd = Xtr.mean(0), Xtr.std(0); sd = np.where(sd > 0, sd, 1.)
    m = LogisticRegression(max_iter=3000, C=1.0).fit((Xtr - mu) / sd, ytr)
    return f1_score(yte, m.predict((Xte - mu) / sd), average='macro')


rows = []
for b in common.BENCH:
    meta, markers, raw, ramp = common.load(b)
    units, datasets = meta.unit_key.to_numpy(), meta.dataset.to_numpy()
    ylab = meta.celltype.to_numpy(); y = LabelEncoder().fit_transform(ylab)
    tab = pd.crosstab(meta.celltype, meta.dataset)
    shared = [c for c in tab.index[(tab > 0).all(1)] if c not in common.AMBIGUOUS]
    ok = np.isin(ylab, shared)
    reps = common.arms(raw, ramp, units, datasets)
    plats = sorted(set(datasets))
    print(b, flush=True)
    for held in plats:
        A, B = [p for p in plats if p != held]
        te = bal(np.flatnonzero(ok & (datasets == held)), y, N_TEST)
        for first, second in ((A, B), (B, A)):
            iA = np.flatnonzero(ok & (datasets == first))
            iB = np.flatnonzero(ok & (datasets == second))
            for rep in range(REPS):
                sets = {'A@N':    bal(iA, y, N),
                        'A@2N':   bal(iA, y, 2 * N),
                        'A+B@2N': np.concatenate([bal(iA, y, N), bal(iB, y, N)])}
                for arm, M in reps.items():
                    for name, tr in sets.items():
                        rows.append(dict(benchmark=b, held_out=held, source=first,
                            second=second, rep=rep, arm=arm, train_set=name,
                            n_train=len(tr), macro_f1=fit_score(M[tr], y[tr], M[te], y[te])))
        print('  ', held, 'done', flush=True)

pd.DataFrame(rows).to_csv('scaling_results.csv', index=False)
print('written', len(rows))
