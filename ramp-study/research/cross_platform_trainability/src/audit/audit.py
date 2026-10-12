"""Recompute every recorded number from the CSVs on disk.

Each check states the value as written in a results document and the value
recomputed from the raw rows.  A check that cannot be reproduced is reported,
not adjusted.
"""
import glob, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import numpy as np, pandas as pd, paths

ROOT = paths.ROOT
CHECKS = []


def check(doc, claim, recorded, computed, tol=0.0015):
    ok = (computed is not None and recorded is not None
          and abs(float(recorded) - float(computed)) <= tol)
    CHECKS.append(dict(doc=doc, claim=claim, recorded=recorded,
                       computed=None if computed is None else round(float(computed), 4),
                       status='PASS' if ok else 'FAIL'))


def check_exact(doc, claim, recorded, computed):
    ok = recorded == computed
    CHECKS.append(dict(doc=doc, claim=claim, recorded=recorded, computed=computed,
                       status='PASS' if ok else 'FAIL'))


def load(sub, dtypes=None):
    fs = sorted(glob.glob(str(ROOT / 'results' / sub / '*.csv.gz')))
    if not fs:
        fs = sorted(glob.glob(str(ROOT / 'results' / sub / '*.csv')))
    return pd.concat([pd.read_csv(f, low_memory=False, dtype=dtypes or {})
                      for f in fs], ignore_index=True)


def mean_rank(piv, ascending=False):
    return piv.rank(axis=1, ascending=ascending).mean()


# ------------------------------------------------------------------ main grid
def audit_main():
    D = 'RESULTS.md'
    df = load('raw')
    check_exact(D, 'total rows = 212,050', 212050, len(df))

    j1 = df[(df.job == 'job1_celltype') & (df.train_set == 'A@N')
            & (df.capacity == 'linear')]
    piv = j1.groupby(['task', 'arm']).macro_f1.mean().unstack()
    TAB = {  # as written in the Job 1 table
        'bone_marrow': dict(Raw=.608, CytoNorm=.558, cyCombine=.828, RAMP=.891),
        'tnbc': dict(Raw=.319, CytoNorm=.255, cyCombine=.515, RAMP=.546),
        'melanoma': dict(Raw=.401, CytoNorm=.457, cyCombine=.662, RAMP=.645),
        'lung_cancer': dict(Raw=.356, CytoNorm=.569, cyCombine=.559, RAMP=.617),
        'lung_story': dict(Raw=.536, CytoNorm=.540, cyCombine=.587, RAMP=.610),
        'melanoma_story': dict(Raw=.355, CytoNorm=.387, cyCombine=.463, RAMP=.610),
        'hnscc_story': dict(Raw=.271, CytoNorm=.423, cyCombine=.535, RAMP=.537),
    }
    for task, row in TAB.items():
        for arm, v in row.items():
            check(D, f'Job1 linear {task}/{arm}', v,
                  piv.loc[task, arm] if task in piv.index and arm in piv.columns else None)

    r = mean_rank(piv)
    for arm, v in dict(RAMP=1.14, cyCombine=3.00, **{'VirTues-style': 3.14},
                       **{'fastMNN-reconstructed': 3.71, 'Spatium-style': 4.43,
                          'CytoNorm': 6.29, 'Raw': 7.14, 'Harmony-direct': 7.14}).items():
        check(D, f'Job1 linear mean rank {arm}', v, r.get(arm), tol=0.02)
    check_exact(D, 'Job1 linear: RAMP first in 6 of 7',
                6, int((piv.idxmax(axis=1) == 'RAMP').sum()))
    check(D, 'Job1 melanoma: cyCombine beats RAMP by 0.017', 0.017,
          piv.loc['melanoma', 'cyCombine'] - piv.loc['melanoma', 'RAMP'])

    k = df[(df.job == 'job1_celltype') & (df.train_set == 'A@N') & (df.capacity == 'knn')]
    pk = k.groupby(['task', 'arm']).macro_f1.mean().unstack()
    rk = mean_rank(pk)
    check(D, 'Job1 kNN mean rank RAMP = 1.00', 1.00, rk.get('RAMP'), tol=0.02)
    check_exact(D, 'Job1 kNN: RAMP first in all 7',
                7, int((pk.idxmax(axis=1) == 'RAMP').sum()))
    check(D, 'Job1 kNN mean rank Raw = 7.71', 7.71, rk.get('Raw'), tol=0.02)

    j3 = df[(df.job == 'job3_platform') & (df.capacity == 'knn')]
    p3 = j3.groupby(['task', 'arm']).balanced_accuracy.mean().unstack()
    r3 = mean_rank(p3, ascending=True)
    for arm, v in {'RAMP': 2.6, 'cyCombine': 1.6, 'Spatium-style': 1.9,
                   'VirTues-style': 5.0, 'CytoNorm': 5.4,
                   'fastMNN-reconstructed': 5.6, 'Harmony-direct': 6.0,
                   'Raw': 8.0}.items():
        check(D, f'Job3 kNN mean rank {arm}', v, r3.get(arm), tol=0.1)

    # M4: both columns
    d4 = df[(df.job == 'job1_celltype') & (df.capacity == 'linear')]
    key = ['task', 'held_out', 'train_platform', 'arm', 'rep']
    w = d4.pivot_table(index=key, columns='train_set', values='macro_f1').reset_index()
    w = w.dropna(subset=['A+B@2N', 'A@2N', 'A@N'])
    w['same'] = w['A@2N'] - w['A@N']
    w['second'] = w['A+B@2N'] - w['A@2N']
    g = w.groupby('arm')
    M4 = {'Raw': (-0.007, -0.012, 58.7), 'Harmony-direct': (-0.004, 0.047, 66.0),
          'CytoNorm': (-0.008, 0.115, 80.7), 'cyCombine': (-0.011, 0.047, 88.7),
          'fastMNN-reconstructed': (-0.008, 0.056, 80.0),
          'VirTues-style': (-0.013, 0.046, 78.0), 'Spatium-style': (-0.011, 0.047, 87.3),
          'RAMP': (-0.012, 0.047, 88.0)}
    for arm, (s, sec, pct) in M4.items():
        check(D, f'M4 same-platform {arm}', s, g.same.mean().get(arm))
        check(D, f'M4 second-platform {arm}', sec, g.second.mean().get(arm))
        check(D, f'M4 %helps {arm}', pct,
              g.second.apply(lambda x: (x > 0).mean() * 100).get(arm), tol=0.1)


if __name__ == '__main__':
    audit_main()
    t = pd.DataFrame(CHECKS)
    pd.set_option('display.width', 200); pd.set_option('display.max_rows', 300)
    bad = t[t.status == 'FAIL']
    print(f'{len(t)} checks, {len(bad)} FAIL\n')
    if len(bad):
        print(bad.to_string(index=False))
    t.to_csv(ROOT / 'results' / 'AUDIT_checks.csv', index=False)


# ------------------------------------------------- main grid, remaining claims
def audit_main_rest():
    D = 'RESULTS.md'
    df = load('raw')
    CAPS = ['linear'] + [f'mlp{h}' for h in (4, 8, 16, 32, 64, 128, 256)]

    # M2: smallest model reaching 95% of the best any arm reaches on that task
    d = df[(df.job == 'job1_celltype') & (df.train_set == 'A@N')]
    reached, params = {}, {}
    for task, g in d.groupby('task'):
        piv = g.groupby(['arm', 'capacity']).macro_f1.mean().unstack()
        piv = piv[[c for c in CAPS if c in piv.columns]]
        thr = 0.95 * piv.max().max()
        for arm in piv.index:
            hit = [c for c in piv.columns if piv.loc[arm, c] >= thr]
            if hit:
                reached[arm] = reached.get(arm, 0) + 1
                p = g[(g.arm == arm) & (g.capacity == hit[0])].params
                params[(task, arm)] = float(p.iloc[0]) if len(p) else np.nan
    for arm, v in {'RAMP': 7, 'cyCombine': 4, 'VirTues-style': 4,
                   'fastMNN-reconstructed': 2, 'Raw': 0, 'Harmony-direct': 0,
                   'CytoNorm': 0, 'Spatium-style': 0}.items():
        check_exact(D, f'M2 tasks reached {arm}', v, reached.get(arm, 0))
    check(D, 'M2 RAMP params on lung_cancer = 54', 54,
          params.get(('lung_cancer', 'RAMP')), tol=0.5)
    check(D, 'M2 RAMP params on bone_marrow = 60', 60,
          params.get(('bone_marrow', 'RAMP')), tol=0.5)
    pr = d[d.capacity.isin(CAPS)].params
    # "Raw never reaches ... at any capacity from 60 to 3,846 parameters" reads as
    # one range; check whether any single task's Raw sweep actually spans it
    rw = d[d.arm == 'Raw'].groupby('task').params.agg(['min', 'max'])
    span = ((rw['min'] == 60) & (rw['max'] == 3846)).any()
    check_exact(D, 'M2 "60 to 3,846" is one task\'s Raw parameter range', True, bool(span))
    check_exact(D, 'M2 global Raw parameter range is 60..3,846', True,
                bool(pr.min() == 60 and pr.max() == 3846))

    # Job 2, fastMNN excluded
    j2 = df[(df.job == 'job2_marker_transfer') & (df.train_set == 'A@N')]
    for cap, tab in (('linear', {'Spatium-style': 2.86, 'RAMP': 3.00,
                                 'VirTues-style': 4.57, 'CytoNorm': 5.00,
                                 'cyCombine': 5.71, 'Raw': 6.14,
                                 'Harmony-direct': 7.71}),
                     ('knn', {'cyCombine': 2.29, 'RAMP': 2.57, 'Spatium-style': 2.86})):
        pin = j2[j2.capacity == cap].groupby(['task', 'arm']).spearman.mean().unstack()
        pex = pin.drop(columns=['fastMNN-reconstructed'])
        for arm, v in tab.items():
            check(D, f'Job2 {cap} rank {arm} — as written, i.e. fastMNN EXCLUDED',
                  v, mean_rank(pex).get(arm), tol=0.02)
            check(D, f'Job2 {cap} rank {arm} — fastMNN INCLUDED',
                  v, mean_rank(pin).get(arm), tol=0.02)

    # the archived first grid
    v1 = load('raw_v1_archived')
    o = v1[(v1.job == 'job2_masked_marker') & (v1.capacity == 'linear')]
    if 'train_set' in o.columns:
        o = o[o.train_set == 'A@N']
    om = o.groupby('arm').spearman.mean()
    check(D, 'Job2 original design: Harmony-direct = 0.696', 0.696,
          om.get('Harmony-direct'), tol=0.003)
    check(D, 'Job2 original design: RAMP = 0.271', 0.271, om.get('RAMP'), tol=0.003)
    check(D, 'Job2 original design: cyCombine = 0.318', 0.318,
          om.get('cyCombine'), tol=0.003)
    nm = df[(df.job == 'job2_marker_transfer') & (df.capacity == 'linear')
            & (df.train_set == 'A@N')].groupby('arm').spearman.mean()
    check(D, 'Job2 corrected: Harmony-direct = 0.219', 0.219,
          nm.get('Harmony-direct'), tol=0.003)

    # Job 4
    j4 = df[(df.job == 'job4_rare') & (df.capacity == 'linear')]
    p4 = j4.groupby(['task', 'arm']).average_precision.mean().unstack()
    check_exact(D, 'Job4 covers 6 tasks', 6, int(p4.shape[0]))
    r4 = mean_rank(p4)
    for arm, v in {'RAMP': 3.7, 'fastMNN-reconstructed': 3.8, 'cyCombine': 4.0,
                   'VirTues-style': 4.0, 'Raw': 4.2, 'CytoNorm': 4.2,
                   'Spatium-style': 6.0, 'Harmony-direct': 6.2}.items():
        check(D, f'Job4 mean rank {arm}', v, r4.get(arm), tol=0.06)
    rf = j4.groupby('task').rare_frac_test.mean() * 100
    check(D, 'Job4 rare fraction low = 3.5%', 3.5, rf.min(), tol=0.1)
    check(D, 'Job4 rare fraction high = 10.6%', 10.6, rf.max(), tol=0.1)


def audit_m5():
    D = 'RESULTS.md (M5)'
    m5_all = load('raw_m5')
    check_exact(D, 'M5 total rows = 80,800', 80800, len(m5_all))
    m5 = m5_all[m5_all.capacity == 'linear']   # the document reports the linear probe
    z = m5[m5.protocol == 'source_only']
    pz = z.groupby(['task', 'arm']).macro_f1.mean().unstack()
    rz = mean_rank(pz)
    for arm, v in {'RAMP': 1.1, 'cyCombine': 3.0, 'VirTues-style': 3.0,
                   'fastMNN-reconstructed': 3.7, 'Spatium-style': 4.6,
                   'CytoNorm': 6.3, 'Harmony-direct': 7.0, 'Raw': 7.3}.items():
        check(D, f'M5 zero-shot mean rank {arm}', v, rz.get(arm), tol=0.06)
    for task, v in {'bone_marrow': .893, 'tnbc': .524, 'melanoma': .638,
                    'lung_cancer': .606, 'lung_story': .607,
                    'melanoma_story': .595, 'hnscc_story': .524}.items():
        check(D, f'M5 RAMP zero-shot {task}', v, pz.loc[task, 'RAMP'])
    t100 = m5[(m5.protocol == 'target_only_k') & (m5.k == 100)].groupby('arm').macro_f1.mean()
    check(D, 'M5 k=100 target-only Raw = 0.791', 0.791, t100.get('Raw'))
    check(D, 'M5 k=100 target-only RAMP = 0.777', 0.777, t100.get('RAMP'))
    check(D, 'M5 k=100 target-only CytoNorm = 0.796', 0.796, t100.get('CytoNorm'))
    check(D, 'M5 k=100 target-only VirTues-style = 0.795', 0.795, t100.get('VirTues-style'))
    # k=0 exists only under source_only; source_plus_k starts at k=1
    so = m5[(m5.protocol == 'source_only') & (m5.arm == 'RAMP')].groupby('k').macro_f1.mean()
    check(D, 'M5 RAMP at k=0 = 0.643', 0.643, so.get(0))
    sp = m5[(m5.protocol == 'source_plus_k') & (m5.arm == 'RAMP')].groupby('k').macro_f1.mean()
    check(D, 'M5 pooled unweighted RAMP k=100 = 0.665', 0.665, sp.get(100))
    spw = m5[(m5.protocol == 'source_plus_k_weighted') & (m5.arm == 'RAMP')] \
        .groupby('k').macro_f1.mean()
    check(D, 'M5 weighted RAMP k=100 = 0.750', 0.750, spw.get(100))


# ------------------------------------------------- the three documents I wrote
def audit_new():
    L = 'RESULTS_LOMO.md'
    lomo = pd.concat([pd.read_csv(f, low_memory=False) for f in
                      sorted(glob.glob(str(ROOT / 'results/lomo/*.csv.gz')))
                      if 'control' not in Path(f).name], ignore_index=True)
    check_exact(L, 'leave-one-out rows = 172,151', 172151, len(lomo))
    check_exact(L, 'marker-tasks = 91', 91,
                int(lomo.groupby(['task', 'marker']).ngroups))
    pl = lomo[(lomo.train_set == 'A@N') & (lomo.capacity == 'linear')] \
        .groupby(['task', 'arm']).spearman.mean().unstack()
    assert 'fastMNN-embedding-fullpanel' not in pl.columns
    main = load('raw')
    pm = main[(main.job == 'job2_marker_transfer') & (main.train_set == 'A@N')
              & (main.capacity == 'linear')].groupby(['task', 'arm']).spearman.mean().unstack()
    for arm, lo, mn in [('RAMP', .386, .387), ('Spatium-style', .390, .389),
                        ('CytoNorm', .365, .360), ('cyCombine', .288, .319),
                        ('Harmony-direct', .215, .221), ('Raw', .328, .329),
                        ('VirTues-style', .364, .363)]:
        check(L, f'LOMO {arm}', lo, pl[arm].mean())
        check(L, f'main grid {arm}', mn, pm[arm].mean())
    check(L, 'LOMO fastMNN-embedding = 0.246', .246, pl['fastMNN-embedding'].mean())
    check(L, 'main grid fastMNN = 0.598', .598, pm['fastMNN-reconstructed'].mean())
    rl, rm = mean_rank(pl), mean_rank(pm)
    check(L, 'LOMO mean rank RAMP = 1.86', 1.86, rl.get('RAMP'), tol=0.02)
    check(L, 'LOMO mean rank Spatium-style = 1.71', 1.71, rl.get('Spatium-style'), tol=0.02)
    check(L, 'main grid mean rank RAMP = 3.00', 3.00, rm.get('RAMP'), tol=0.02)
    ctrl = pd.concat([pd.read_csv(f) for f in sorted(
        glob.glob(str(ROOT / 'results/lomo/control_fastmnn_*.csv.gz')))])
    check(L, 'control: full-panel embedding = 0.599', .599,
          ctrl[ctrl.capacity == 'linear'].groupby('task').spearman.mean().mean())

    P = 'RESULTS_PRETRAIN.md'
    pre = pd.concat([pd.read_csv(f, low_memory=False) for f in
                     sorted(glob.glob(str(ROOT / 'results/pretrain/[a-z]*.csv.gz')))],
                    ignore_index=True)
    check_exact(P, 'pretrain rows = 5,160', 5160, len(pre))
    FIVE = ['bone_marrow', 'tnbc', 'melanoma', 'lung_cancer', 'lung_story']
    w = pre[(pre.job == 'job1_celltype') & pre.task.isin(FIVE)].pivot_table(
        index=['task', 'held_out', 'train_platform', 'arm', 'rep'],
        columns='train_set', values='macro_f1').reset_index().dropna(
        subset=['A+B@2N', 'A@2N'])
    w['gain'] = w['A+B@2N'] - w['A@2N']
    g = w.groupby('arm')
    for arm, base, end, gain in [('CytoNorm', .462, .557, .095),
                                 ('fastMNN-reconstructed', .585, .652, .067),
                                 ('RAMP', .647, .690, .044),
                                 ('Raw', .425, .452, .027)]:
        check(P, f'pretrained gain {arm}', gain, g.gain.mean().get(arm))
        check(P, f'pretrained endpoint {arm}', end, g['A+B@2N'].mean().get(arm))
        check(P, f'pretrained baseline {arm}', base, g['A@2N'].mean().get(arm))

    S = 'RESULTS_SCALING_INTEGRATION.md'
    sc = pd.concat([pd.read_csv(f, low_memory=False) for f in
                    sorted(glob.glob(str(ROOT / 'results/scaling/*.csv.gz')))],
                   ignore_index=True)
    check_exact(S, 'scaling rows = 16,560', 16560, len(sc))
    # the document averages within task first, then across tasks, so that a task
    # with more markers or directions does not carry more weight
    cur = sc[(sc.job == 'job1_celltype') & sc.balanced] \
        .groupby(['task', 'arm', 'per_class']).macro_f1.mean().reset_index() \
        .groupby(['arm', 'per_class']).macro_f1.mean().unstack()
    check(S, 'RAMP at 5 cells/class = 0.563', .563, cur.loc['RAMP', 5])
    check(S, 'RAMP at 2000 cells/class = 0.687', .687, cur.loc['RAMP', 2000])
    check(S, 'Raw max over all levels = 0.418', .418, cur.loc['Raw'].max())
    check(S, 'CytoNorm max = 0.465', .465, cur.loc['CytoNorm'].max())
    check(S, 'Harmony-direct max = 0.433', .433, cur.loc['Harmony-direct'].max())

    cv = pd.concat([pd.read_csv(f, dtype={'level': str}) for f in
                    sorted(glob.glob(str(ROOT / 'results/convergence/*.csv.gz')))],
                   ignore_index=True)
    cv = cv[cv.level != 'max']
    k = ['task', 'held_out', 'train_platform', 'rep', 'level']
    cw = cv.pivot_table(index=k, values='macro_f1', columns='arm').reset_index()
    for other, small, large in [('Raw', .203, .191), ('Spatium-style', .101, .079),
                                ('cyCombine', .035, .038)]:
        cw['g'] = cw['RAMP'] - cw[other]
        a = cw[cw.level == '200'].set_index(k[:4]).g
        b = cw[cw.level == '20000'].set_index(k[:4]).g
        j = pd.concat([a.rename('s'), b.rename('l')], axis=1).dropna()
        check(S, f'convergence RAMP-{other} at ~200', small, j.s.mean())
        check(S, f'convergence RAMP-{other} at ~20,000', large, j.l.mean())

    it = pd.concat([pd.read_csv(f) for f in
                    sorted(glob.glob(str(ROOT / 'results/integration/*.csv')))])
    t = it.groupby('arm')[['iLISI_norm', 'kBET_reject', 'cLISI_norm', 'ARI']].mean()
    for arm, il, kb, cl, ari in [('RAMP', .407, .889, .872, .413),
                                 ('Spatium-style', .467, .836, .791, .212),
                                 ('fastMNN-reconstructed', .161, .980, .900, .393),
                                 ('Raw', .035, .999, .864, .187)]:
        check(S, f'integration iLISI {arm}', il, t.loc[arm, 'iLISI_norm'])
        check(S, f'integration kBET {arm}', kb, t.loc[arm, 'kBET_reject'])
        check(S, f'integration cLISI {arm}', cl, t.loc[arm, 'cLISI_norm'])
        check(S, f'integration ARI {arm}', ari, t.loc[arm, 'ARI'])
