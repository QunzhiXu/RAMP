"""Job 2 (masked-marker reconstruction), questions 2 and 3.

Q2 capacity sweep: six capacities, trained on one platform, evaluated on a
   never-seen third platform.
Q3 scaling: A@N vs A@2N vs A+B@2N, linear model, same evaluation.
"""
import numpy as np, pandas as pd, warnings
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr
import common

warnings.filterwarnings('ignore')
RNG = np.random.default_rng(20260917)
N, N_TEST = 6000, 8000
CAPS = [None, 4, 8, 16, 32, 64]


def model(h):
    return (Ridge(alpha=1.0) if h is None else
            MLPRegressor(hidden_layer_sizes=(h,), max_iter=200, random_state=0,
                         early_stopping=True, n_iter_no_change=8))


def score(obs, pred, u_te):
    rs = []
    for u in np.unique(u_te):
        m = u_te == u
        if m.sum() >= 30 and np.std(obs[m]) > 0:
            r = spearmanr(obs[m], pred[m]).statistic
            if np.isfinite(r):
                rs.append(r)
    return float(np.mean(rs)) if rs else np.nan


cap_rows, sc_rows = [], []
for b in common.BENCH:
    meta, markers, raw, ramp = common.load(b)
    units, datasets = meta.unit_key.to_numpy(), meta.dataset.to_numpy()
    reps = common.arms(raw, ramp, units, datasets)
    plats = sorted(set(datasets))
    print(b, flush=True)
    for held in plats:
        A, B = [p for p in plats if p != held]
        te_pool = np.flatnonzero(datasets == held)
        te = RNG.choice(te_pool, min(N_TEST, len(te_pool)), replace=False)
        iA = np.flatnonzero(datasets == A); iB = np.flatnonzero(datasets == B)
        trA  = RNG.choice(iA, min(N, len(iA)), replace=False)
        trA2 = RNG.choice(iA, min(2 * N, len(iA)), replace=False)
        trAB = np.concatenate([RNG.choice(iA, min(N, len(iA)), replace=False),
                               RNG.choice(iB, min(N, len(iB)), replace=False)])
        for arm, M in reps.items():
            for tag, tr, caps in (('A@N', trA, CAPS), ('A@2N', trA2, [None]),
                                  ('A+B@2N', trAB, [None])):
                mu, sd = M[tr].mean(0), M[tr].std(0); sd = np.where(sd > 0, sd, 1.)
                Z = (M - mu) / sd
                for j, mk in enumerate(markers):
                    keep = [k for k in range(M.shape[1]) if k != j]
                    for h in caps:
                        mdl = model(h).fit(Z[tr][:, keep], Z[tr][:, j])
                        s = score(Z[te][:, j], mdl.predict(Z[te][:, keep]), units[te])
                        if not np.isfinite(s):
                            continue
                        d = len(keep)
                        p = d + 1 if h is None else d * h + h + h + 1
                        rec = dict(benchmark=b, held_out=held, arm=arm, marker=mk,
                                   train_set=tag, spearman=s,
                                   capacity='linear' if h is None else f'mlp{h}', params=p)
                        (cap_rows if tag == 'A@N' else sc_rows).append(rec)
                        if tag == 'A@N' and h is None:
                            sc_rows.append(rec)
        print('  ', held, 'done', flush=True)

pd.DataFrame(cap_rows).to_csv('mm_capacity.csv', index=False)
pd.DataFrame(sc_rows).to_csv('mm_scaling.csv', index=False)
print('written', len(cap_rows), len(sc_rows))
