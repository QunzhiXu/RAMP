import numpy as np
import pandas as pd
from scipy import stats
SEED=20260909

def bh(p):
    p = np.asarray(p, float)
    good = np.isfinite(p)
    out = np.full(len(p), np.nan)
    a = p[good]
    ix = np.argsort(a)
    q = np.empty(len(a))
    q[ix] = np.minimum(1, np.minimum.accumulate((a[ix] * len(a) / np.arange(1, len(a) + 1))[::-1])[::-1])
    out[good] = q
    return out

def ols_hc3(y, x, term=1):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    pinv = np.linalg.pinv(x)
    beta = pinv @ y
    res = y - x @ beta
    hat = np.sum(x * pinv.T, axis=1)
    weighted = pinv * (res / np.maximum(1 - hat, 1e-08))[None, :]
    cov = weighted @ weighted.T
    se = np.sqrt(max(cov[term, term], 0))
    df = len(y) - np.linalg.matrix_rank(x)
    critical = stats.t.ppf(0.975, df)
    p = float(2 * stats.t.sf(abs(beta[term] / se), df))
    return dict(coef=float(beta[term]), se=float(se), ci_low=float(beta[term] - critical * se), ci_high=float(beta[term] + critical * se), p=p, n=len(y), df=int(df), max_leverage=float(hat.max()))

def design_matrix(g, exposure='high_HLA', full=False):
    x = pd.DataFrame({'intercept': np.ones(len(g))}, index=g.index)
    e = g[exposure].astype(float)
    if exposure == 'all_mean':
        e = (e - e.mean()) / e.std(ddof=1)
    x['exposure'] = e
    x['age'] = (g.age - g.age.mean()) / g.age.std(ddof=1)
    x = pd.concat([x, pd.get_dummies(g[['TMA', 'sex', 'Stage']].astype(str), drop_first=True, dtype=float)], axis=1)
    if full:
        for col in ['caf_fraction', 'old_global_CAF_score', 'vcaf_fraction']:
            x[col] = (g[col] - g[col].mean()) / g[col].std(ddof=1)
    return x

def compare(g, col, bootstrap=3000):
    a = g.loc[g.group.eq('Lower'), col].to_numpy()
    b = g.loc[g.group.eq('Higher'), col].to_numpy()
    test = stats.mannwhitneyu(b, a, alternative='two-sided', method='asymptotic')
    rng = np.random.default_rng(SEED)
    differences = b[rng.integers(len(b), size=(bootstrap, len(b)))].mean(1) - a[rng.integers(len(a), size=(bootstrap, len(a)))].mean(1)
    lo, hi = np.quantile(differences, [0.025, 0.975])
    return dict(n_lower=len(a), n_higher=len(b), lower_mean=float(a.mean()), higher_mean=float(b.mean()), lower_median=float(np.median(a)), higher_median=float(np.median(b)), difference=float(b.mean() - a.mean()), ci_low=float(lo), ci_high=float(hi), p=float(test.pvalue), rank_biserial=float(2 * test.statistic / (len(a) * len(b)) - 1))
