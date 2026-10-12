"""Does the gate survive a target cohort whose composition differs from the source?

Per-dataset z-scoring and per-section quantile normalization are computed from
the marginal distribution of whatever cells are present, so they move when the
composition moves. A fitted monotone map does not. This measures that.

Protocol: the gate is calibrated once on the full AbSeq source and frozen. The
target CODEX cohort is then subsampled to change the fraction of positive cells,
each normalization is recomputed on the subsample as it would be in practice,
and the frozen gate is applied. RAMP values are taken as fitted and are not
refitted, which is the property under test.
"""
import numpy as np, pandas as pd
from sklearn.metrics import roc_curve

AMBIGUOUS = {'Non-joint', 'Mix', 'Unclassified / technical', 'Unannotated', 'nan'}
T_LABELS  = {'CD4 T', 'CD8 T', 'CD8− T', 'Other T / CD8−', 'Other T / NKT'}
SOURCE, TARGET, RNG = 'Triana AbSeq', 'BM CODEX', np.random.default_rng(20260917)

meta = pd.read_csv('work/cell_metadata.csv.gz', low_memory=False)
meta['eval_label'] = meta['celltype'].fillna('Unannotated').astype(str)
markers = open('work/markers.txt').read().split()
raw  = np.asarray(np.load('work/raw.npy'), dtype=np.float64)
ramp = np.asarray(np.load('work/ramp_aligned.npy'), dtype=np.float64)
units, datasets = meta.unit_key.to_numpy(), meta.dataset.to_numpy()
lab = meta.eval_label.to_numpy()
NEG_CD8 = {'CD4 T', 'CD8\u2212 T', 'Other T / CD8\u2212'}
TASKS = {'CD3': (~np.isin(lab, list(AMBIGUOUS)), np.isin(lab, list(T_LABELS))),
         'CD4': (np.isin(lab, ['CD4 T', 'CD8 T']), lab == 'CD4 T'),
         'CD8': (np.isin(lab, ['CD8 T'] + list(NEG_CD8)), lab == 'CD8 T')}


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
    return float(thr[int(np.flatnonzero(np.isclose(ba, ba.max(), rtol=0, atol=1e-12))[0])])


def norm_virtues(X):
    hi = np.percentile(X, 99, axis=0)
    Z = np.log1p(np.clip(np.minimum(X, hi), 0, None))
    return (Z - Z.mean(0)) / np.where(Z.std(0, ddof=0) > 0, Z.std(0, ddof=0), 1.0)


def norm_spatium(X, K=50):
    out = np.empty_like(X)
    for j in range(X.shape[1]):
        r = pd.Series(X[:, j]).rank(method='average').to_numpy()
        out[:, j] = np.clip(((r - .5) / len(X) * K).astype(int), 0, K - 1)
    return out


ALL = []
for MARKER, (included, positive) in TASKS.items():
  col = markers.index(MARKER)
  # --- source gates, frozen once on the full source cohort -------------------
  src = np.flatnonzero(included & (datasets == SOURCE))
  gates = {'Raw':  fit_gate(raw[src, col], positive[src], units[src]),
           'RAMP': fit_gate(ramp[src, col], positive[src], units[src]),
           'VirTues-style': fit_gate(norm_virtues(raw[src])[:, col], positive[src], units[src]),
           'Spatium-style': fit_gate(norm_spatium(raw[src])[:, col], positive[src], units[src])}

  # --- target, subsampled to a range of positive fractions -------------------
  tgt_units = sorted(np.unique(units[datasets == TARGET]))
  fractions = [0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80]
  rows = []
  for frac in fractions:
      for rep in range(5):
          per_unit = {}
          for u in tgt_units:
              rows_u = np.flatnonzero(included & (units == u))
              p, n = rows_u[positive[rows_u]], rows_u[~positive[rows_u]]
              if len(p) < 15 or len(n) < 15:
                  continue
              # subsample both classes to hit the requested fraction, keeping
              # the larger class as the binding constraint
              cap = min(len(p) / frac, len(n) / (1 - frac))
              n_pos, n_neg = int(round(frac * cap)), int(round((1 - frac) * cap))
              if n_pos < 10 or n_neg < 10:
                  continue
              sel = np.concatenate([RNG.choice(p, n_pos, replace=False),
                                    RNG.choice(n, n_neg, replace=False)])
              per_unit[u] = sel
          if not per_unit:
              continue
          allsel = np.concatenate(list(per_unit.values()))
          mats = {'Raw': raw[allsel], 'RAMP': ramp[allsel],
                  'VirTues-style': norm_virtues(raw[allsel]),
                  'Spatium-style': norm_spatium(raw[allsel])}
          idx = {u: np.flatnonzero(np.isin(allsel, s)) for u, s in per_unit.items()}
          realised = positive[allsel].mean()
          for arm, M in mats.items():
              bas = []
              for u, ii in idx.items():
                  pred, pos = M[ii, col] >= gates[arm], positive[allsel][ii]
                  if pos.sum() < 5 or (~pos).sum() < 5:
                      continue
                  bas.append(((pred & pos).sum()/pos.sum() + (~pred & ~pos).sum()/(~pos).sum()) / 2)
              rows.append(dict(target_positive_fraction=round(realised, 4), nominal=frac,
                               rep=rep, arm=arm, BA=float(np.mean(bas))))

  ALL.append(pd.DataFrame(rows).assign(marker=MARKER))

res = pd.concat(ALL, ignore_index=True)
res.to_csv('composition_sensitivity_results.csv', index=False)
piv = res.groupby(['arm', 'nominal']).BA.mean().unstack()
order = ['Raw', 'RAMP', 'VirTues-style', 'Spatium-style']
print('CD3 gate calibrated on AbSeq, frozen, applied to CODEX subsampled to a')
print('range of T-cell fractions. Donor-averaged balanced accuracy, mean of 5 draws.\n')
print('  averaged over CD3, CD4 and CD8\n')
print(piv.loc[order].round(3).to_string())
print('\nspread across the composition range (max - min):')
print((piv.loc[order].max(1) - piv.loc[order].min(1)).round(3).to_string())
