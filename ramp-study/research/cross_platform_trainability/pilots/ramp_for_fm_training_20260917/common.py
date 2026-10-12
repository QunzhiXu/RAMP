"""Shared loading and the four input representations."""
import numpy as np, pandas as pd
from pathlib import Path

AMBIGUOUS = {'Non-joint', 'Mix', 'Unclassified / technical', 'Unannotated',
             'nan', 'Other', 'Others'}
BENCH = ['bone_marrow', 'tnbc', 'melanoma', 'lung_cancer']


def load(b):
    d = Path('work') / b
    meta = pd.read_csv(d / 'cell_metadata.csv.gz', low_memory=False)
    meta['celltype'] = meta['celltype'].fillna('Unannotated').astype(str)
    return (meta,
            open(d / 'markers.txt').read().split(),
            np.asarray(np.load(d / 'raw.npy'), dtype=np.float64),
            np.asarray(np.load(d / 'ramp_aligned.npy'), dtype=np.float64))


def virtues_style(X, units, datasets):
    """Per-unit 99th-percentile clip, shifted log, per-dataset channel z-score."""
    out = np.empty_like(X)
    for u in np.unique(units):
        m = units == u
        out[m] = np.minimum(X[m], np.percentile(X[m], 99, axis=0))
    out = np.log1p(np.clip(out, 0, None))
    for s in np.unique(datasets):
        m = datasets == s
        mu, sd = out[m].mean(0), out[m].std(0, ddof=0)
        out[m] = (out[m] - mu) / np.where(sd > 0, sd, 1.0)
    return out


def spatium_style(X, units, datasets, K=50):
    """Within-unit quantile normalization, K equal-count bins."""
    out = np.empty_like(X)
    for u in np.unique(units):
        m = np.flatnonzero(units == u)
        for j in range(X.shape[1]):
            r = pd.Series(X[m, j]).rank(method='average').to_numpy()
            out[m, j] = np.clip(((r - .5) / len(m) * K).astype(int), 0, K - 1)
    return out


def arms(raw, ramp, units, datasets):
    return {'Raw': raw, 'RAMP': ramp,
            'VirTues-style': virtues_style(raw, units, datasets),
            'Spatium-style': spatium_style(raw, units, datasets)}
