"""Does a fixed marker threshold survive the normalization schemes used by
protein foundation models?

Same frozen bone-marrow cells, same gating protocol as Figure 2E of the
manuscript. Only the normalization varies:

  Raw           the frozen source-scale matrix (published baseline)
  RAMP          the frozen RAMP output           (published)
  VirTues-style per-dataset channel-wise standardization, following the
                published VirTues recipe: clip channel-wise at the 99th
                percentile within each unit, shifted-log with size factor 1,
                then z-score with mean and s.d. computed over each dataset
  Spatium-style quantile normalization within each unit, discretized into
                K = 50 bins per protein across the cells of that unit

The script first reproduces the published Raw and RAMP balanced accuracies and
exits if it cannot.
"""
import numpy as np, pandas as pd
from sklearn.metrics import roc_curve

AMBIGUOUS = {'Non-joint', 'Mix', 'Unclassified / technical', 'Unannotated', 'nan'}
T_LABELS  = {'CD4 T', 'CD8 T', 'CD8− T', 'Other T / CD8−', 'Other T / NKT'}
NEG_CD8   = {'CD4 T', 'CD8− T', 'Other T / CD8−'}
SOURCE, TARGET = 'Triana AbSeq', 'BM CODEX'
PUBLISHED = {'Raw': {'CD3': .684, 'CD4': .504, 'CD8': .550},
             'RAMP': {'CD3': .912, 'CD4': .910, 'CD8': .881}}

meta = pd.read_csv('work/cell_metadata.csv.gz', low_memory=False)
meta['eval_label'] = meta['celltype'].fillna('Unannotated').astype(str)
markers = open('work/markers.txt').read().split()
raw  = np.asarray(np.load('work/raw.npy'), dtype=np.float64)
ramp = np.asarray(np.load('work/ramp_aligned.npy'), dtype=np.float64)
units, datasets = meta.unit_key.to_numpy(), meta.dataset.to_numpy()


def tasks():
    lab = meta.eval_label.to_numpy()
    clear = ~np.isin(lab, list(AMBIGUOUS))
    return {'CD3': (clear, np.isin(lab, list(T_LABELS))),
            'CD4': (np.isin(lab, ['CD4 T', 'CD8 T']), lab == 'CD4 T'),
            'CD8': (np.isin(lab, ['CD8 T'] + list(NEG_CD8)), lab == 'CD8 T')}


def virtues_style(X):
    """Per-image 99th-percentile clip, shifted log, per-dataset channel z-score."""
    out = np.empty_like(X)
    for u in np.unique(units):                       # 'image' == sample unit
        m = units == u
        hi = np.percentile(X[m], 99, axis=0)
        out[m] = np.minimum(X[m], hi)
    out = np.log1p(np.clip(out, 0, None))            # shifted log, size factor 1
    for d in np.unique(datasets):                    # z-score PER DATASET
        m = datasets == d
        mu, sd = out[m].mean(0), out[m].std(0, ddof=0)
        out[m] = (out[m] - mu) / np.where(sd > 0, sd, 1.0)
    return out


def spatium_style(X, K=50):
    """Quantile normalization within each unit, then K equal-count bins."""
    out = np.empty_like(X)
    for u in np.unique(units):
        m = np.flatnonzero(units == u)
        for j in range(X.shape[1]):
            r = pd.Series(X[m, j]).rank(method='average').to_numpy()
            q = (r - 0.5) / len(m)                   # within-unit quantile
            out[m, j] = np.clip((q * K).astype(int), 0, K - 1)
    return out


def fit_gate(v, pos, unit):
    w = np.zeros(len(v))
    for label in (False, True):
        sel = pos == label
        lev = np.unique(unit[sel])
        for u in lev:
            k = sel & (unit == u)
            w[k] = .5 / (len(lev) * k.sum())
    fpr, tpr, thr = roc_curve(pos, v, sample_weight=w, drop_intermediate=False)
    ba = (tpr + 1 - fpr) / 2
    i = int(np.flatnonzero(np.isclose(ba, ba.max(), rtol=0, atol=1e-12))[0])
    return float(thr[i]), float(ba[i])


def transfer(matrix, col, included, positive):
    """Calibrate on SOURCE units, apply unchanged to each TARGET unit."""
    def elig(ds):
        out = []
        for u in sorted(np.unique(units[datasets == ds])):
            rows = np.flatnonzero(included & (datasets == ds) & (units == u))
            p = int(positive[rows].sum())
            if p >= 10 and len(rows) - p >= 10:
                out.append((u, rows))
        return out
    src = elig(SOURCE)
    train = np.concatenate([r for _, r in src])
    thr, src_ba = fit_gate(matrix[train, col], positive[train], units[train])
    bas = []
    for _, rows in elig(TARGET):
        pred, pos = matrix[rows, col] >= thr, positive[rows]
        sens = (pred & pos).sum() / pos.sum()
        spec = (~pred & ~pos).sum() / (~pos).sum()
        bas.append((sens + spec) / 2)
    return thr, src_ba, float(np.mean(bas)), len(bas)


arms = {'Raw': raw, 'RAMP': ramp,
        'VirTues-style': virtues_style(raw), 'Spatium-style': spatium_style(raw)}

rows = []
for name, M in arms.items():
    for marker, (inc, pos) in tasks().items():
        thr, sba, tba, n = transfer(M, markers.index(marker), inc, pos)
        rows.append(dict(arm=name, marker=marker, threshold=thr,
                         source_BA=sba, target_BA=tba, n_target_donors=n))
res = pd.DataFrame(rows)

print('VALIDATION against published Figure 2E values')
bad = 0
for arm in ('Raw', 'RAMP'):
    for marker, want in PUBLISHED[arm].items():
        got = float(res[(res.arm == arm) & (res.marker == marker)].target_BA.iloc[0])
        ok = abs(got - want) < 5e-4
        bad += not ok
        print(f'  {arm:<14}{marker}  computed {got:.4f}  published {want:.3f}  {"OK" if ok else "MISMATCH"}')
if bad:
    raise SystemExit(f'\n{bad} mismatches - not reporting the new arms')
print('VALIDATION PASSED\n')

res.to_csv('gate_transfer_results.csv', index=False)
print(f'Fixed-gate transfer {SOURCE} -> {TARGET}, donor-averaged balanced accuracy\n')
piv = res.pivot(index='arm', columns='marker', values='target_BA').loc[list(arms)]
piv['mean'] = piv.mean(1)
print(piv.round(3).to_string())
print('\nsource-platform BA (sanity: every arm should gate well on its own platform)')
print(res.pivot(index='arm', columns='marker', values='source_BA').loc[list(arms)].round(3).to_string())
