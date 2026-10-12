"""The four jobs. Every model trains on one or two platforms and is evaluated on
a platform held out of training.

Job 2 is the redesigned version (amendment 5): marker X is transferred from the
platform that has it and checked against the held-out platform's MEASURED value,
not against the arm's own transformed value.
"""
import numpy as np, pandas as pd, warnings
from sklearn.metrics import (f1_score, balanced_accuracy_score,
                             average_precision_score, roc_auc_score)
from scipy.stats import spearmanr, rankdata
import common, models

warnings.filterwarnings('ignore')
D = common.D
CAPS = models.CAPS


def _seed(*p):
    return (D['protocol']['seed_base'] + abs(hash(p))) % 1_000_000


def _scale(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    return (Xtr - mu) / sd, (Xte - mu) / sd


def balanced_draw(idx, y, n, rng):
    cls = np.unique(y[idx])
    per = max(20, n // len(cls))
    return np.concatenate([rng.choice(idx[y[idx] == c], min(per, int((y[idx] == c).sum())),
                                      replace=False) for c in cls])


def within_unit_rank(values, idx, units):
    """Scale-free target: rank within each sample unit, mapped to (0, 1]."""
    out = np.empty(len(idx), dtype=float)
    u = units[idx]
    for uu in np.unique(u):
        m = u == uu
        out[m] = rankdata(values[idx][m]) / m.sum()
    return out


def ece(prob, true, bins=10):
    conf, pred = prob.max(1), prob.argmax(1)
    acc = (pred == true).astype(float)
    e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            e += m.sum() / len(true) * abs(acc[m].mean() - conf[m].mean())
    return float(e)


# ------------------------------------------------------------------- job 1
def job1_celltype(task, arms, meta, train_sets, test_idx, cap, rep):
    y_all = meta.celltype.to_numpy()
    rows = []
    for arm, M in arms.items():
        for tag, tr in train_sets.items():
            classes = sorted(set(y_all[tr]))
            ymap = {c: i for i, c in enumerate(classes)}
            ytr = np.array([ymap[c] for c in y_all[tr]])
            keep = np.array([i for i in test_idx if y_all[i] in ymap])
            yte = np.array([ymap[c] for c in y_all[keep]])
            Xtr, Xte = _scale(M[tr], M[keep])
            m = models.clf(cap, _seed(task, 'job1', arm, cap, tag, rep)).fit(Xtr, ytr)
            pred = m.predict(Xte)
            r = dict(task=task, job='job1_celltype', arm=arm, capacity=cap,
                     family=models.family(cap),
                     params=models.n_params(cap, M.shape[1], len(classes)),
                     train_set=tag, rep=rep, n_train=len(tr), n_test=len(keep),
                     n_classes=len(classes),
                     macro_f1=f1_score(yte, pred, average='macro'),
                     balanced_accuracy=balanced_accuracy_score(yte, pred))
            if hasattr(m, 'predict_proba'):
                r['ece'] = ece(m.predict_proba(Xte), yte)
            rows.append(r)
    return rows


# ------------------------------------------------------------------- job 2
def job2_marker_transfer(task, arms, meta, markers, raw, train_sets, test_idx,
                         cap, rep, units):
    """Marker X is unavailable on the target platform; transfer it from the
    source and score against the target's MEASURED X (within-unit rank)."""
    rows = []
    for arm, M in arms.items():
        if M.shape[1] != len(markers):
            continue                      # no marker axis
        for tag, tr in train_sets.items():
            for j, mk in enumerate(markers):
                keep = [k for k in range(M.shape[1]) if k != j]
                Xtr, Xte = _scale(M[tr][:, keep], M[test_idx][:, keep])
                ytr = within_unit_rank(raw[:, j], tr, units)
                yte = within_unit_rank(raw[:, j], test_idx, units)
                m = models.reg(cap, _seed(task, 'job2', arm, cap, tag, mk, rep))
                m.fit(Xtr, ytr)
                pred = m.predict(Xte)
                rs = []
                u = units[test_idx]
                for uu in np.unique(u):
                    s = u == uu
                    if s.sum() >= D['jobs']['job2_masked_marker']['min_cells_per_unit'] \
                       and np.std(yte[s]) > 0:
                        v = spearmanr(yte[s], pred[s]).statistic
                        if np.isfinite(v):
                            rs.append(v)
                if rs:
                    rows.append(dict(task=task, job='job2_marker_transfer', arm=arm,
                                     capacity=cap, family=models.family(cap),
                                     params=models.n_params(cap, M.shape[1] - 1, 1),
                                     train_set=tag, rep=rep, marker=mk,
                                     spearman=float(np.mean(rs)), n_units=len(rs)))
    return rows


# ------------------------------------------------------------------- job 3
def job3_platform(task, arms, meta, idx, rep, cap=None):
    """Control, lower is better. Test cells drawn at random — label-free."""
    cap = cap or models.PARAMETRIC[-1]
    ds = meta.dataset.to_numpy()
    rng = np.random.default_rng(_seed(task, 'job3', rep))
    rows = []
    for arm, M in arms.items():
        classes = sorted(set(ds[idx]))
        ymap = {c: i for i, c in enumerate(classes)}
        y = np.array([ymap[c] for c in ds[idx]])
        cut = rng.permutation(len(idx)); half = len(cut) // 2
        tr, te = idx[cut[:half]], idx[cut[half:]]
        Xtr, Xte = _scale(M[tr], M[te])
        m = models.clf(cap, _seed(task, 'job3', arm, rep)).fit(Xtr, y[cut[:half]])
        rows.append(dict(task=task, job='job3_platform', arm=arm, capacity=cap,
                         family=models.family(cap), rep=rep,
                         n_platforms=len(classes),
                         balanced_accuracy=balanced_accuracy_score(y[cut[half:]],
                                                                  m.predict(Xte)),
                         chance=1 / len(classes)))
    return rows


# ------------------------------------------------------------------- job 4
def job4_rare(task, arms, meta, rare_label, train_idx, test_idx, cap, rep):
    y = (meta.celltype.to_numpy() == rare_label).astype(int)
    rows = []
    for arm, M in arms.items():
        Xtr, Xte = _scale(M[train_idx], M[test_idx])
        m = models.clf(cap, _seed(task, 'job4', arm, cap, rare_label, rep))
        m.fit(Xtr, y[train_idx])
        s = (m.predict_proba(Xte)[:, 1] if hasattr(m, 'predict_proba')
             else m.decision_function(Xte))
        rows.append(dict(task=task, job='job4_rare', arm=arm, capacity=cap,
                         family=models.family(cap),
                         params=models.n_params(cap, M.shape[1], 2), rep=rep,
                         rare_label=rare_label, n_train=len(train_idx),
                         n_test=len(test_idx),
                         rare_frac_test=float(y[test_idx].mean()),
                         average_precision=average_precision_score(y[test_idx], s),
                         auroc=roc_auc_score(y[test_idx], s),
                         f1_rare=f1_score(y[test_idx], m.predict(Xte), pos_label=1)))
    return rows
