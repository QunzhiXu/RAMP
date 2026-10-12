"""The four jobs. Every model trains on one or two platforms and is evaluated on
a platform held out of training. Specs follow design.json; nothing is chosen here.
"""
import numpy as np, pandas as pd, warnings
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (f1_score, balanced_accuracy_score, average_precision_score,
                             roc_auc_score, brier_score_loss)
from scipy.stats import spearmanr
import common

warnings.filterwarnings('ignore')
D = common.D
CAPS = D['sweeps']['S1_capacity']['levels']
AMB = common.AMBIGUOUS


def _seed(*parts):
    return (D['protocol']['seed_base'] + abs(hash(parts))) % 1_000_000


def _clf(cap, seed):
    if cap == 'linear':
        return LogisticRegression(max_iter=3000, C=1.0)
    h = int(cap[3:])
    return MLPClassifier(hidden_layer_sizes=(h,), max_iter=400, early_stopping=True,
                         n_iter_no_change=12, random_state=seed)


def _reg(cap, seed):
    if cap == 'linear':
        return Ridge(alpha=1.0)
    h = int(cap[3:])
    return MLPRegressor(hidden_layer_sizes=(h,), max_iter=300, early_stopping=True,
                        n_iter_no_change=10, random_state=seed)


def n_params(cap, d, k):
    if cap == 'linear':
        return d * k + k
    h = int(cap[3:])
    return d * h + h + h * k + k


def _fit_scale(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    return (Xtr - mu) / sd, (Xte - mu) / sd


def balanced_draw(idx, y, n, rng):
    cls = np.unique(y[idx])
    per = max(20, n // len(cls))
    return np.concatenate([rng.choice(idx[y[idx] == c], min(per, (y[idx] == c).sum()),
                                      replace=False) for c in cls])


def ece(prob, true, bins=10):
    conf, pred = prob.max(1), prob.argmax(1)
    acc = (pred == true).astype(float)
    e, n = 0.0, len(true)
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            e += m.sum() / n * abs(acc[m].mean() - conf[m].mean())
    return float(e)


# ------------------------------------------------------------------- job 1
def job1_celltype(task, arms, meta, train_sets, test_idx, cap, rep):
    y_all = meta.celltype.to_numpy()
    rows = []
    for arm, M in arms.items():
        for tag, tr in train_sets.items():
            seed = _seed(task, 'job1', arm, cap, tag, rep)
            classes = sorted(set(y_all[tr]))
            ymap = {c: i for i, c in enumerate(classes)}
            ytr = np.array([ymap[c] for c in y_all[tr]])
            keep = np.array([i for i in test_idx if y_all[i] in ymap])
            yte = np.array([ymap[c] for c in y_all[keep]])
            Xtr, Xte = _fit_scale(M[tr], M[keep])
            m = _clf(cap, seed).fit(Xtr, ytr)
            pred = m.predict(Xte)
            r = dict(task=task, job='job1_celltype', arm=arm, capacity=cap,
                     params=n_params(cap, M.shape[1], len(classes)), train_set=tag,
                     rep=rep, n_train=len(tr), n_test=len(keep), n_classes=len(classes),
                     macro_f1=f1_score(yte, pred, average='macro'),
                     balanced_accuracy=balanced_accuracy_score(yte, pred))
            if hasattr(m, 'predict_proba'):
                p = m.predict_proba(Xte)
                r['ece'] = ece(p, yte)
            rows.append(r)
    return rows


# ------------------------------------------------------------------- job 2
def job2_masked_marker(task, arms, meta, markers, train_sets, test_idx, cap, rep):
    units = meta.unit_key.astype(str).to_numpy()
    rows = []
    for arm, M in arms.items():
        if arm in D['arms']['MaxFuse'].get('excluded_from', []) and arm == 'MaxFuse':
            continue
        if M.shape[1] != len(markers):
            continue                      # embedding arms have no marker axis
        for tag, tr in train_sets.items():
            mu, sd = M[tr].mean(0), M[tr].std(0); sd = np.where(sd > 0, sd, 1.)
            Z = (M - mu) / sd
            for j, mk in enumerate(markers):
                keep = [k for k in range(M.shape[1]) if k != j]
                seed = _seed(task, 'job2', arm, cap, tag, mk, rep)
                mod = _reg(cap, seed).fit(Z[tr][:, keep], Z[tr][:, j])
                pred, obs = mod.predict(Z[test_idx][:, keep]), Z[test_idx][:, j]
                rs = []
                for u in np.unique(units[test_idx]):
                    m_ = units[test_idx] == u
                    if m_.sum() >= D['jobs']['job2_masked_marker']['min_cells_per_unit'] \
                       and np.std(obs[m_]) > 0:
                        v = spearmanr(obs[m_], pred[m_]).statistic
                        if np.isfinite(v): rs.append(v)
                if rs:
                    rows.append(dict(task=task, job='job2_masked_marker', arm=arm,
                                     capacity=cap, params=n_params(cap, M.shape[1]-1, 1),
                                     train_set=tag, rep=rep, marker=mk,
                                     spearman=float(np.mean(rs)), n_units=len(rs)))
    return rows


# ------------------------------------------------------------------- job 3
def job3_platform(task, arms, meta, idx, rep):
    """Control. Lower is better. Largest capacity only."""
    cap = CAPS[-1]
    ds = meta.dataset.to_numpy()
    rows = []
    rng = np.random.default_rng(_seed(task, 'job3', rep))
    for arm, M in arms.items():
        classes = sorted(set(ds[idx]))
        ymap = {c: i for i, c in enumerate(classes)}
        y = np.array([ymap[c] for c in ds[idx]])
        cut = rng.permutation(len(idx))
        half = len(cut) // 2
        tr, te = idx[cut[:half]], idx[cut[half:]]
        ytr, yte = y[cut[:half]], y[cut[half:]]
        Xtr, Xte = _fit_scale(M[tr], M[te])
        m = _clf(cap, _seed(task, 'job3', arm, rep)).fit(Xtr, ytr)
        rows.append(dict(task=task, job='job3_platform', arm=arm, capacity=cap,
                         rep=rep, n_platforms=len(classes),
                         balanced_accuracy=balanced_accuracy_score(yte, m.predict(Xte)),
                         chance=1 / len(classes)))
    return rows


# ------------------------------------------------------------------- job 4
def job4_rare(task, arms, meta, rare_label, train_idx, test_idx, cap, rep):
    y = (meta.celltype.to_numpy() == rare_label).astype(int)
    rows = []
    for arm, M in arms.items():
        seed = _seed(task, 'job4', arm, cap, rare_label, rep)
        Xtr, Xte = _fit_scale(M[train_idx], M[test_idx])
        m = _clf(cap, seed).fit(Xtr, y[train_idx])
        s = (m.predict_proba(Xte)[:, 1] if hasattr(m, 'predict_proba')
             else m.decision_function(Xte))
        rows.append(dict(task=task, job='job4_rare', arm=arm, capacity=cap,
                         params=n_params(cap, M.shape[1], 2), rep=rep,
                         rare_label=rare_label, n_train=len(train_idx),
                         n_test=len(test_idx), rare_frac_test=float(y[test_idx].mean()),
                         average_precision=average_precision_score(y[test_idx], s),
                         auroc=roc_auc_score(y[test_idx], s),
                         f1_rare=f1_score(y[test_idx], m.predict(Xte), pos_label=1)))
    return rows
