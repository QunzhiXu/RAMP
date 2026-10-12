"""Control: separate 'refitted without the marker' from 'embedding vs reconstruction'.

fastMNN's Job 2 score falls by 0.351 under leave-one-out, but two things changed
at once: the alignment was refitted without the target marker, AND the arm
contributes its embedding here where the main grid used the marker-space
reconstruction.  This fits the same job on the FULL-PANEL embedding — fitted
with the target marker present, exactly as the main grid was.

If the full-panel embedding still scores near the main grid's 0.598, the fall is
caused by refitting.  If it scores near the leave-one-out 0.246, the fall is
caused by the representation change and the leave-one-out comparison is unfair
to fastMNN.
"""
import sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import common, jobs, models, paths, ramp_run

D = common.D
N = D['sweeps']['S3_composition']['N']
MIN_CELLS = D['jobs']['job2_masked_marker']['min_cells_per_unit']


def full_panel_embedding(task):
    """The main grid's fastMNN embedding, fitted on the full panel."""
    files = paths.arm_files(task, 'fastMNN-reconstructed')
    meta = common.load_meta(task)
    order = meta.cell_id.astype(str).to_numpy()
    if paths.LAYOUT[task] == 'benchmark':
        df = pd.read_csv(files[0], low_memory=False)
        ids = df['cell_id'].astype(str).to_numpy()
        v = df.drop(columns=['cell_id']).select_dtypes('number').to_numpy(float)
        pos = pd.Series(np.arange(len(ids)), index=ids)
        return v[pos.reindex(order).to_numpy()]
    return np.load(files[0]).astype(float)              # values.npy, already in order


def run(task, reps=D['protocol']['repetitions'], caps=None):
    caps = caps or models.CAPS
    meta = common.load_meta(task)
    markers = ramp_run.spec(task)['markers']
    raw = np.asarray(common.load_arms(task, arms=['Raw'])[2]['Raw'], float)
    M = full_panel_embedding(task)
    units = meta.unit_key.astype(str).to_numpy(); ds = meta.dataset.to_numpy()
    plats = sorted(meta.dataset.unique())
    rows, t0 = [], time.time()
    for j, mk in enumerate(markers):
        y_all = raw[:, j]
        for held in plats:
            for train_p in [p for p in plats if p != held]:
                for rep in range(reps):
                    rng = np.random.default_rng(D['protocol']['seed_base'] + rep)
                    te = rng.choice(np.flatnonzero(ds == held), 12000, replace=False)
                    tr = rng.choice(np.flatnonzero(ds == train_p),
                                    min(N, int((ds == train_p).sum())), replace=False)
                    ytr = jobs.within_unit_rank(y_all, tr, units)
                    yte = jobs.within_unit_rank(y_all, te, units)
                    u = units[te]
                    for cap in caps:
                        Xtr, Xte = jobs._scale(M[tr], M[te])
                        m = models.reg(cap, jobs._seed(task, 'job2ctrl',
                                                       'fastMNN', cap, 'A@N', mk, rep))
                        m.fit(Xtr, ytr)
                        pred = m.predict(Xte)
                        rs = [spearmanr(yte[u == uu], pred[u == uu]).statistic
                              for uu in np.unique(u)
                              if (u == uu).sum() >= MIN_CELLS and np.std(yte[u == uu]) > 0]
                        rs = [v for v in rs if np.isfinite(v)]
                        if rs:
                            rows.append(dict(task=task, arm='fastMNN-embedding-fullpanel',
                                             capacity=cap, family=models.family(cap),
                                             train_set='A@N', rep=rep, marker=mk,
                                             held_out=held, train_platform=train_p,
                                             spearman=float(np.mean(rs)), n_units=len(rs)))
        print(f'  {task} {mk}: {len(rows)} rows {time.time()-t0:.0f}s', flush=True)
    df = pd.DataFrame(rows)
    out = paths.ROOT / 'results' / 'lomo' / f'control_fastmnn_{task}.csv.gz'
    df.to_csv(out, index=False)
    print(f'{task}: {len(df)} rows -> {out.name}', flush=True)
    return df


if __name__ == '__main__':
    run(sys.argv[1])
