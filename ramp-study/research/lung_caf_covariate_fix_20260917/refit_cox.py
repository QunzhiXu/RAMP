"""Refit the Figure 5F Cox model. Validates against the published COX_TERMS.csv
using the original (stale) global CAF covariate, then emits the corrected model
that uses the present three-platform global CAF median."""
import csv, math, sys
import numpy as np

D = 'work/story_lung_HLA_ABC/summary_steps3_6/06_DFS'
TERMS = ['age','sex2','caf_fraction_z','global_raw_z','stage_code2','stage_code3',
         'stage_code4','abundance_z','state_z']

rows = [r for r in csv.DictReader(open(f'{D}/PATIENT_INPUTS.csv')) if r['eligible']=='True']

def z(v):
    v = np.array(v, float); return (v - v.mean())/v.std(ddof=1)

def design(global_col):
    t = np.array([float(r['time_years']) for r in rows])
    e = np.array([int(float(r['event'])) for r in rows])
    stage = np.array([float(r['Stage']) for r in rows])
    X = np.column_stack([
        [float(r['age']) for r in rows],
        [1.0 if float(r['sex'])==2 else 0.0 for r in rows],
        z([float(r['caf_fraction']) for r in rows]),
        z([float(r[global_col]) for r in rows]),
        (stage==2).astype(float), (stage==3).astype(float), (stage==4).astype(float),
        z([float(r['vcaf_fraction']) for r in rows]),
        z([float(r['all_mean']) for r in rows]),
    ])
    return t, e, X

def cox(t, e, X, iters=100):
    """Cox partial likelihood with Efron handling of ties, Newton-Raphson."""
    n, p = X.shape
    b = np.zeros(p)
    times = np.unique(t[e == 1])
    for _ in range(iters):
        w = np.exp(X @ b)
        g = np.zeros(p); H = np.zeros((p, p))
        for ti in times:
            risk = t >= ti
            dset = (t == ti) & (e == 1)
            d = int(dset.sum())
            wr, Xr = w[risk], X[risk]
            wd, Xd = w[dset], X[dset]
            S0r = wr.sum(); S1r = wr @ Xr; S2r = np.einsum('i,ij,ik->jk', wr, Xr, Xr)
            S0d = wd.sum(); S1d = wd @ Xd; S2d = np.einsum('i,ij,ik->jk', wd, Xd, Xd)
            g += Xd.sum(0)
            for l in range(d):
                f = l / d
                S0 = S0r - f*S0d; S1 = S1r - f*S1d; S2 = S2r - f*S2d
                m = S1 / S0
                g -= m
                H += S2/S0 - np.outer(m, m)
        step = np.linalg.solve(H, g); b += step
        if np.max(np.abs(step)) < 1e-12: break
    return b, np.sqrt(np.diag(np.linalg.inv(H)))


def fit(global_col):
    t, e, X = design(global_col)
    b, se = cox(t, e, X)
    out = []
    for i, term in enumerate(TERMS):
        p = math.erfc(abs(b[i]/se[i])/math.sqrt(2))
        out.append(dict(id='RAMP_k30__mean__all280__matched_full', config='RAMP_k30',
                        state='mean', cohort='all280', model='matched_full', term=term,
                        coef=b[i], se=se[i], hr=math.exp(b[i]),
                        ci_low=math.exp(b[i]-1.96*se[i]), ci_high=math.exp(b[i]+1.96*se[i]), p=p))
    return out

print(f'patients={len(rows)}  events={sum(int(float(r["event"])) for r in rows)}\n')

# --- validation against published values ---
pub = {r['term']: r for r in csv.DictReader(open(f'{D}/COX_TERMS.csv'))}
old = {d['term']: d for d in fit('old_global_CAF_score')}
worst = 0.0
print('validation vs published COX_TERMS.csv (original global CAF covariate):')
for term in TERMS:
    d = abs(old[term]['coef'] - float(pub[term]['coef']))
    worst = max(worst, d)
    print(f'  {term:<16} coef {old[term]["coef"]:+.8f}  published {float(pub[term]["coef"]):+.8f}  |diff| {d:.2e}')
print(f'\nmax |coef difference| = {worst:.3e}')
if worst > 1e-6:
    sys.exit('VALIDATION FAILED - not writing corrected model')
print('VALIDATION PASSED\n')

# --- corrected model ---
new = fit('current_global_CAF_median')
with open('COX_TERMS_corrected.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['id','config','state','cohort','model','term',
                                      'coef','se','hr','ci_low','ci_high','p'])
    w.writeheader()
    for d in new: w.writerow(d)

print('corrected model (current three-platform global CAF median):')
for term in TERMS:
    d = [x for x in new if x['term']==term][0]
    print(f'  {term:<16} HR {d["hr"]:.3f}  95% CI {d["ci_low"]:.2f}-{d["ci_high"]:.2f}  p={d["p"]:.4f}')
s = [x for x in new if x['term']=='state_z'][0]
print(f'\nstate_z (vCAF score): HR {s["hr"]:.4f}  CI {s["ci_low"]:.4f}-{s["ci_high"]:.4f}  p={s["p"]:.6f}')
print(f'  rounded for manuscript: HR {s["hr"]:.2f} (95% CI {s["ci_low"]:.2f}-{s["ci_high"]:.2f}), P={s["p"]:.4f}')
