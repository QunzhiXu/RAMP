"""Sanity check: is 'pretraining hurts' a property of the setting or of my hyperparameters?

Pretraining came out negative for every arm on both downstream tasks. Before
reporting that, sweep the three choices that could produce it artefactually —
latent width, mask rate, and whether the probe sees the latent alone or the
latent concatenated with the input — and check whether ANY configuration turns
the delta positive.
"""
import sys, time, itertools
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import common, jobs, mae

torch.set_num_threads(1)
D = common.D
N = D['sweeps']['S3_composition']['N']
LATENTS = [8, 16, 32, 64, 128]
MASKS = [0.15, 0.30, 0.50]
ARMS = ['Raw', 'RAMP', 'CytoNorm', 'Spatium-style']


def _std(fit, X):
    mu, sd = fit.mean(0), fit.std(0)
    return (X - mu) / np.where(sd > 0, sd, 1.0)


def run(task, reps=2):
    meta = common.load_meta(task)
    _, _, arms = common.load_arms(task, arms=ARMS)
    ylab = meta.celltype.to_numpy(); ds = meta.dataset.to_numpy()
    tab = pd.crosstab(meta.celltype, meta.dataset)
    shared = [c for c in tab.index[(tab > 0).all(1)] if c not in common.AMBIGUOUS]
    ok = np.isin(ylab, shared)
    plats = sorted(meta.dataset.unique())
    rows = []
    for held in plats[:2]:
        train_p = [p for p in plats if p != held][0]
        for rep in range(reps):
            rng = np.random.default_rng(D['protocol']['seed_base'] + rep)
            te = jobs.balanced_draw(np.flatnonzero(ok & (ds == held)), ylab, 12000, rng)
            tr = jobs.balanced_draw(np.flatnonzero(ok & (ds == train_p)), ylab, 2 * N, rng)
            cls = sorted(set(ylab[tr])); ymap = {c: i for i, c in enumerate(cls)}
            keep = np.array([i for i, c in enumerate(ylab[te]) if c in ymap])
            ytr = [ymap[c] for c in ylab[tr]]
            yte = [ymap[c] for c in ylab[te][keep]]
            for arm in ARMS:
                M = np.asarray(arms[arm], float)
                Xc = _std(M[tr], M[tr]).astype(np.float32)
                Xt = _std(M[tr], M[te]).astype(np.float32)
                base = f1_score(yte, LogisticRegression(max_iter=3000)
                                .fit(Xc, ytr).predict(Xt[keep]), average='macro')
                rows.append(dict(task=task, arm=arm, rep=rep, held_out=held,
                                 config='no_pretrain', latent=np.nan, mask=np.nan,
                                 macro_f1=base))
                for lat, mk in itertools.product(LATENTS, MASKS):
                    mae.LATENT, mae.MASK_RATE = lat, mk
                    seed = abs(hash((task, arm, held, rep, lat, mk))) % (2 ** 31)
                    enc, _, _ = mae.pretrain(Xc, seed)
                    Ztr, Zte = mae.embed(enc, Xc), mae.embed(enc, Xt)
                    for probe, A, B in (('latent', Ztr, Zte),
                                        ('latent+input', np.hstack([Ztr, Xc]),
                                         np.hstack([Zte, Xt]))):
                        s = f1_score(yte, LogisticRegression(max_iter=3000)
                                     .fit(A, ytr).predict(B[keep]), average='macro')
                        rows.append(dict(task=task, arm=arm, rep=rep, held_out=held,
                                         config=probe, latent=lat, mask=mk, macro_f1=s))
        print(f'  {task} held={held} rows={len(rows)}', flush=True)
    mae.LATENT, mae.MASK_RATE = 32, 0.30
    return pd.DataFrame(rows)


if __name__ == '__main__':
    out = []
    for t in sys.argv[1:] or ['lung_cancer']:
        out.append(run(t))
    df = pd.concat(out)
    df.to_csv(HERE.parents[1] / 'results' / 'pretrain' / 'hparam_sweep.csv', index=False)
    base = df[df.config == 'no_pretrain'].groupby(['task', 'arm', 'rep', 'held_out']).macro_f1.mean()
    d = df[df.config != 'no_pretrain'].merge(
        base.rename('base').reset_index(), on=['task', 'arm', 'rep', 'held_out'])
    d['delta'] = d.macro_f1 - d.base
    print('\ndelta vs no-pretraining, mean over arms/reps/directions:')
    print(d.pivot_table(index=['config', 'latent'], columns='mask', values='delta')
          .round(4).to_string())
    print('\nbest configuration per arm:')
    g = d.groupby(['arm', 'config', 'latent', 'mask']).delta.mean().reset_index()
    print(g.loc[g.groupby('arm').delta.idxmax()].round(4).to_string(index=False))
