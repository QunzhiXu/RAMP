"""Masked-marker reconstruction across platforms — the objective VirTues and
Spatium pretrain on, run at small scale.

For every benchmark and every held-out platform P3: hold out one marker at a
time, predict it from the remaining markers, train on P1, on P2, and on P1+P2
pooled, and evaluate on P3. Scored by Spearman correlation between prediction
and the held-out measurement, computed within each target sample unit so the
score cannot be inflated by between-sample scale.
"""
import numpy as np, pandas as pd, warnings
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr
import common

warnings.filterwarnings('ignore')
RNG = np.random.default_rng(20260917)
N_TRAIN, N_TEST = 12000, 12000
CAPACITY = [None, 8, 32]                      # None = ridge (linear)

rows = []
for b in common.BENCH:
    meta, markers, raw, ramp = common.load(b)
    units, datasets = meta.unit_key.to_numpy(), meta.dataset.to_numpy()
    reps = common.arms(raw, ramp, units, datasets)
    plats = sorted(set(datasets))
    print(f'{b}: {len(markers)} markers, {len(plats)} platforms', flush=True)

    for held in plats:
        src = [p for p in plats if p != held]
        te_all = np.flatnonzero(datasets == held)
        te = RNG.choice(te_all, min(N_TEST, len(te_all)), replace=False)
        configs = {'single_A': [src[0]], 'single_B': [src[1]], 'pooled': src}
        for cname, cplats in configs.items():
            pool = np.flatnonzero(np.isin(datasets, cplats))
            tr = RNG.choice(pool, min(N_TRAIN, len(pool)), replace=False)
            for arm, M in reps.items():
                mu, sd = M[tr].mean(0), M[tr].std(0)
                sd = np.where(sd > 0, sd, 1.0)
                Z = (M - mu) / sd
                for j, mk in enumerate(markers):
                    keep = [k for k in range(M.shape[1]) if k != j]
                    for h in CAPACITY:
                        model = (Ridge(alpha=1.0) if h is None else
                                 MLPRegressor(hidden_layer_sizes=(h,), max_iter=300,
                                              random_state=0, early_stopping=True,
                                              n_iter_no_change=10))
                        model.fit(Z[tr][:, keep], Z[tr][:, j])
                        pred = model.predict(Z[te][:, keep])
                        obs = Z[te][:, j]
                        # within-unit Spearman, then average over units
                        rs = []
                        for u in np.unique(units[te]):
                            m = units[te] == u
                            if m.sum() >= 30 and np.std(obs[m]) > 0:
                                r = spearmanr(obs[m], pred[m]).statistic
                                if np.isfinite(r):
                                    rs.append(r)
                        if not rs:
                            continue
                        rows.append(dict(benchmark=b, held_out=held, train=cname,
                            n_train_platforms=len(cplats), arm=arm, marker=mk,
                            capacity='linear' if h is None else f'mlp{h}',
                            spearman=float(np.mean(rs)), n_units=len(rs)))
        print(f'   held out {held}: done', flush=True)

pd.DataFrame(rows).to_csv('masked_marker_results.csv', index=False)
print(f'\n{len(rows)} rows written')
