"""Run one task end to end. Written to results/raw/<task>.csv.gz"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
import common, jobs, models, paths

D = common.D
N = D['sweeps']['S3_composition']['N']
REPS = D['protocol']['repetitions']
# amendment 4: value-producing arms only; MaxFuse returns an embedding
ARMS = [a for a in common.ARMS if a != 'MaxFuse']


def rare_targets(meta):
    spec = D['jobs']['job4_rare']
    t = pd.crosstab(meta.celltype, meta.dataset)
    shared = [c for c in t.index[(t > 0).all(1)] if c not in common.AMBIGUOUS]
    if not shared:
        return []
    frac = t.loc[shared] / t.loc[shared].sum(0) * 100
    out = []
    for lab in shared:
        lo, hi = frac.loc[lab].idxmin(), frac.loc[lab].idxmax()
        if (frac.loc[lab, lo] < spec['rare_threshold_pct']
                and frac.loc[lab, hi] / max(frac.loc[lab, lo], 1e-9) >= spec['abundance_ratio_min']
                and t.loc[lab, lo] >= spec['min_cells_rare_side']):
            out.append({'label': lab, 'abundant_in': hi, 'rare_in': lo})
    return out


def run(task, caps=None, reps=None):
    caps = caps or models.CAPS
    reps = reps or REPS
    meta0 = common.load_meta(task)
    markers = common.load_markers(task)
    plats = sorted(meta0.dataset.unique())
    ylab = meta0.celltype.to_numpy(); ds = meta0.dataset.to_numpy()
    units = meta0.unit_key.astype(str).to_numpy()
    tab = pd.crosstab(meta0.celltype, meta0.dataset)
    shared = [c for c in tab.index[(tab > 0).all(1)] if c not in common.AMBIGUOUS]
    ok = np.isin(ylab, shared)                      # labelled subset, jobs 1 and 4
    rares = rare_targets(meta0)
    rows, t0 = [], time.time()

    for held in plats:
        srcs = [p for p in plats if p != held]
        for train_p in srcs:
            meta, _, arms = common.load_arms(task, train_platform=train_p, arms=ARMS)
            raw = arms['Raw']
            other = [p for p in srcs if p != train_p]
            for rep in range(reps):
                rng = np.random.default_rng(D['protocol']['seed_base'] + rep)
                # labelled draws for jobs 1 and 4
                te_lab = jobs.balanced_draw(np.flatnonzero(ok & (ds == held)), ylab, 12000, rng)
                iA = np.flatnonzero(ok & (ds == train_p))
                sets = {'A@N': jobs.balanced_draw(iA, ylab, N, rng),
                        'A@2N': jobs.balanced_draw(iA, ylab, 2 * N, rng)}
                if other:
                    iB = np.flatnonzero(ok & (ds == other[0]))
                    sets['A+B@2N'] = np.concatenate(
                        [jobs.balanced_draw(iA, ylab, N, rng),
                         jobs.balanced_draw(iB, ylab, N, rng)])
                # amendment: jobs 2 and 3 are label-free, so their draws use no labels
                te_free = rng.choice(np.flatnonzero(ds == held), 12000, replace=False)
                fA = np.flatnonzero(ds == train_p)
                sets_free = {'A@N': rng.choice(fA, min(N, len(fA)), replace=False),
                             'A@2N': rng.choice(fA, min(2 * N, len(fA)), replace=False)}
                if other:
                    fB = np.flatnonzero(ds == other[0])
                    sets_free['A+B@2N'] = np.concatenate(
                        [rng.choice(fA, min(N, len(fA)), replace=False),
                         rng.choice(fB, min(N, len(fB)), replace=False)])

                for cap in caps:
                    base = {'held_out': held, 'train_platform': train_p}
                    rows += [r | base for r in
                             jobs.job1_celltype(task, arms, meta, sets, te_lab, cap, rep)]
                    # amendment 3: capacity sweep at A@N, composition sweep at linear
                    s2 = sets_free if cap == 'linear' else {'A@N': sets_free['A@N']}
                    rows += [r | base for r in
                             jobs.job2_marker_transfer(task, arms, meta, markers, raw,
                                                       s2, te_free, cap, rep, units)]
                    for rt in rares:
                        if rt['abundant_in'] != train_p or rt['rare_in'] != held:
                            continue
                        tr = jobs.balanced_draw(np.flatnonzero(ok & (ds == train_p)),
                                                ylab, N, rng)
                        rows += [r | base | {'rare_in': rt['rare_in'],
                                             'abundant_in': rt['abundant_in']}
                                 for r in jobs.job4_rare(task, arms, meta, rt['label'],
                                                         tr, te_lab, cap, rep)]
                if rep == 0 and train_p == srcs[0]:
                    idx = rng.choice(len(meta0), 12000, replace=False)   # label-free
                    for cap in ('knn', models.PARAMETRIC[-1]):
                        rows += jobs.job3_platform(task, arms, meta, idx, rep, cap)
            print(f'  {task} held={held} train={train_p} rows={len(rows)} '
                  f'{time.time()-t0:.0f}s', flush=True)

    df = pd.DataFrame(rows)
    out = paths.ROOT / 'results/raw' / f'{task}.csv.gz'
    df.to_csv(out, index=False)
    print(f'{task}: {len(df)} rows -> {out.name}  ({time.time()-t0:.0f}s)')
    return df


if __name__ == '__main__':
    run(sys.argv[1],
        caps=(sys.argv[2].split(',') if len(sys.argv) > 2 else None),
        reps=(int(sys.argv[3]) if len(sys.argv) > 3 else None))
