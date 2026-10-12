"""Source-only ranking head; no target data, labels or scores enter training."""
from __future__ import annotations
import numpy as np
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.neural_network._stochastic_optimizers import AdamOptimizer

LAMBDAS = (0.0, 0.01, 0.1)
TEMPERATURE = 0.1


def pairs(measured, units):
    """All strict within-unit pairs, with units equally weighted."""
    ii, jj, ss, ww = [], [], [], []
    for u in np.unique(units):
        ix = np.flatnonzero(units == u)
        a, b = np.triu_indices(len(ix), 1)
        signs = np.sign(measured[ix[a]] - measured[ix[b]])
        keep = signs != 0
        if keep.any():
            ii.append(ix[a[keep]]); jj.append(ix[b[keep]])
            ss.append(signs[keep]); ww.append(np.full(keep.sum(), 1.0 / keep.sum()))
    if not ii:
        return (np.array([], int), np.array([], int), np.array([], float), np.array([], float))
    return np.concatenate(ii), np.concatenate(jj), np.concatenate(ss), np.concatenate(ww) / len(ii)


def ranking_loss_gradient(prediction, pair_data):
    i, j, sign, weight = pair_data
    gradient = np.zeros(len(prediction), dtype=float)
    if not len(i):
        return 0.0, gradient
    margin = sign * (prediction[i] - prediction[j]) / TEMPERATURE
    loss = np.dot(weight, np.logaddexp(0, -margin))
    derivative = -weight * sign * expit(-margin) / TEMPERATURE
    gradient += np.bincount(i, weights=derivative, minlength=len(prediction))
    gradient -= np.bincount(j, weights=derivative, minlength=len(prediction))
    return float(loss), gradient


class SourceRankScore:
    def __init__(self, measured, units, patients, minimum_cells=30):
        self.groups = []
        for unit in np.unique(units):
            ix = np.flatnonzero(units == unit)
            pp = np.unique(patients[ix])
            assert len(pp) == 1
            if len(ix) < minimum_cells or np.ptp(measured[ix]) == 0:
                continue
            self.groups.append((ix, measured[ix].copy(), pp[0]))
        assert self.groups, 'No source validation units are scoreable'

    def __call__(self, prediction):
        assert np.isfinite(prediction).all()
        by_patient = {}
        for ix, truth, patient in self.groups:
            pred = prediction[ix]
            # Use the exact frozen scorer's arithmetic, including ties. A
            # mathematically equivalent dot product can shift an early-stop
            # tie by 1e-17 and select a different set of model weights.
            rho = float(spearmanr(truth, pred).statistic) if np.ptp(pred) > 0 else 0.0
            by_patient.setdefault(patient, []).append(rho)
        return float(np.mean([np.mean(v) for v in by_patient.values()]))


def initialize(n_features, width, seed):
    rng = np.random.RandomState(seed)
    coefs, intercepts = [], []
    for ni, no in [(n_features, width), (width, 1)]:
        bound = np.sqrt(6.0 / (ni + no))
        coefs.append(rng.uniform(-bound, bound, (ni, no)))
        intercepts.append(rng.uniform(-bound, bound, no))
    return coefs + intercepts, rng


def predict(params, x):
    w1, w2, b1, b2 = params
    return (np.maximum(x @ w1 + b1, 0) @ w2 + b2).ravel()


def train(x, y, measured, units, validation_x, source_score, candidate, seed, lam):
    assert len(x) == len(y) == len(measured) == len(units)
    assert x.shape[1] == validation_x.shape[1]
    params, first_rng = initialize(x.shape[1], candidate['width'], seed)
    opt = AdamOptimizer(params, learning_rate_init=candidate['learning_rate'])
    # sklearn partial_fit reinitializes the integer-seeded RNG each call.
    orders = [first_rng.permutation(len(x)), np.random.RandomState(seed).permutation(len(x))]
    batches = []
    for order in orders:
        b = []
        for start in range(0, len(x), candidate['batch_size']):
            ix = order[start:start + candidate['batch_size']]
            b.append((ix, pairs(measured[ix], units[ix])))
        batches.append(b)
    best_score = -np.inf; best_params = None; stale = 0; trace = []
    for epoch in range(candidate['max_epochs']):
        for ix, pair_data in batches[min(epoch, 1)]:
            xx = x[ix]; yy = y[ix, None]; n = len(ix)
            w1, w2, b1, b2 = params
            hidden = np.maximum(xx @ w1 + b1, 0)
            out = hidden @ w2 + b2
            delta = out - yy
            if lam:
                _, rg = ranking_loss_gradient(out.ravel(), pair_data)
                delta += lam * n * rg[:, None]
            g2 = (hidden.T @ delta + candidate['alpha'] * w2) / n
            gb2 = delta.mean(axis=0)
            dh = delta @ w2.T
            dh[hidden == 0] = 0
            g1 = (xx.T @ dh + candidate['alpha'] * w1) / n
            gb1 = dh.mean(axis=0)
            opt.update_params(params, [g1, g2, gb1, gb2])
        score = source_score(predict(params, validation_x))
        trace.append(score)
        if score > best_score:
            best_score = score; best_params = [p.copy() for p in params]
            selected_epoch = epoch + 1; stale = 0
        else:
            stale += 1
        if stale >= candidate['patience']:
            break
    assert best_params is not None and all(np.isfinite(p).all() for p in best_params)
    return best_params, dict(lambda_rank=lam, source_validation_score=best_score,
        selected_epoch=selected_epoch, epochs_executed=epoch + 1,
        gradient_updates_executed=opt.t, validation_trace=trace,
        minibatch_pairs_first_epoch=sum(len(p[0]) for _, p in batches[0]),
        minibatch_pairs_later_epoch=sum(len(p[0]) for _, p in batches[1]))
