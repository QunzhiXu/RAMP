"""Job 2 under leave-one-marker-out refitting.

Same job, same target, same models, same draws as the main grid; the only
difference is where the features come from.  In the main grid every arm was
fitted on the full panel, so the arm's own transformed copy of marker X sat in
the training features and had to be dropped column-wise, which still left the
alignment itself informed by X.  Here the alignment was refitted without X, so
no feature the model sees was ever influenced by the marker it has to predict.

Arms: the five refitted methods, plus Raw and the two foundation-model
normalization recipes, which are deterministic functions of Raw and so need no
refit.  fastMNN contributes its corrected embedding rather than the marker-space
reconstruction, because the reconstruction is an orthonormal map of the same
information into a space that no longer contains the held-out marker.
"""
import json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import common, jobs, models, paths, ramp_run

D = common.D
OUT = HERE.parents[1] / 'lomo_work' / 'out'
N = D['sweeps']['S3_composition']['N']
REPS = D['protocol']['repetitions']
MIN_CELLS = D['jobs']['job2_masked_marker']['min_cells_per_unit']
METHOD_ARM = {'ramp': 'RAMP', 'harmony': 'Harmony-direct', 'cycombine': 'cyCombine',
              'cytonorm': 'CytoNorm', 'fastmnn': 'fastMNN-embedding'}


def load_lomo(task, marker):
    """Every arm's feature matrix for this held-out marker, or None if incomplete."""
    d = OUT / task / marker
    got = {}
    for meth, arm in METHOD_ARM.items():
        f = d / f'{meth}.npy'
        if not f.exists():
            return None
        got[arm] = np.load(f).astype(np.float64)
    return got


def run(task, reps=REPS, caps=None):
    caps = caps or models.CAPS
    meta = common.load_meta(task)
    markers = ramp_run.spec(task)['markers']
    raw = np.asarray(common.load_arms(task, arms=['Raw'])[2]['Raw'], float)
    units = meta.unit_key.astype(str).to_numpy()
    ds = meta.dataset.to_numpy()
    plats = sorted(meta.dataset.unique())
    rows, t0 = [], time.time()

    for j, mk in enumerate(markers):
        arms = load_lomo(task, mk)
        if arms is None:
            print(f'  {task} {mk}: incomplete, skipped', flush=True)
            continue
        keep = [k for k in range(raw.shape[1]) if k != j]
        arms['Raw'] = raw[:, keep]
        arms['VirTues-style'] = common.virtues_style(raw[:, keep], units, ds)
        arms['Spatium-style'] = common.spatium_style(raw[:, keep], units)
        y_all = raw[:, j]

        for held in plats:
            for train_p in [p for p in plats if p != held]:
                other = [p for p in plats if p not in (held, train_p)]
                for rep in range(reps):
                    rng = np.random.default_rng(D['protocol']['seed_base'] + rep)
                    te = rng.choice(np.flatnonzero(ds == held), 12000, replace=False)
                    fA = np.flatnonzero(ds == train_p)
                    sets = {'A@N': rng.choice(fA, min(N, len(fA)), replace=False),
                            'A@2N': rng.choice(fA, min(2 * N, len(fA)), replace=False)}
                    if other:
                        fB = np.flatnonzero(ds == other[0])
                        sets['A+B@2N'] = np.concatenate(
                            [rng.choice(fA, min(N, len(fA)), replace=False),
                             rng.choice(fB, min(N, len(fB)), replace=False)])
                    yte = jobs.within_unit_rank(y_all, te, units)
                    u = units[te]
                    for cap in caps:
                        use = sets if cap == 'linear' else {'A@N': sets['A@N']}
                        for tag, tr in use.items():
                            ytr = jobs.within_unit_rank(y_all, tr, units)
                            for arm, M in arms.items():
                                Xtr, Xte = jobs._scale(M[tr], M[te])
                                m = models.reg(cap, jobs._seed(task, 'job2lomo', arm,
                                                               cap, tag, mk, rep))
                                m.fit(Xtr, ytr)
                                pred = m.predict(Xte)
                                rs = []
                                for uu in np.unique(u):
                                    s = u == uu
                                    if s.sum() >= MIN_CELLS and np.std(yte[s]) > 0:
                                        v = spearmanr(yte[s], pred[s]).statistic
                                        if np.isfinite(v):
                                            rs.append(v)
                                if rs:
                                    rows.append(dict(
                                        task=task, job='job2_lomo', arm=arm, capacity=cap,
                                        family=models.family(cap),
                                        params=models.n_params(cap, M.shape[1], 1),
                                        train_set=tag, rep=rep, marker=mk,
                                        held_out=held, train_platform=train_p,
                                        spearman=float(np.mean(rs)), n_units=len(rs)))
        print(f'  {task} {mk}: {len(rows)} rows  {time.time()-t0:.0f}s', flush=True)

    df = pd.DataFrame(rows)
    out = paths.ROOT / 'results' / 'lomo' / f'{task}.csv.gz'
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f'{task}: {len(df)} rows -> {out}  ({time.time()-t0:.0f}s)', flush=True)
    return df


if __name__ == '__main__':
    run(sys.argv[1], reps=int(sys.argv[2]) if len(sys.argv) > 2 else REPS)
