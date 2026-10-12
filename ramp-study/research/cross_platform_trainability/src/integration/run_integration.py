"""Score every arm on both axes, for every task."""
import sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import numpy as np, pandas as pd
import common, paths, metrics

ARMS = [a for a in common.ARMS if a != 'MaxFuse']


def run(task, seed=0):
    meta = common.load_meta(task)
    _, _, arms = common.load_arms(task, arms=ARMS)
    tab = pd.crosstab(meta.celltype, meta.dataset)
    shared = [c for c in tab.index[(tab > 0).all(1)] if c not in common.AMBIGUOUS]
    ok = np.isin(meta.celltype.to_numpy(), shared)   # ambiguous labels excluded
    ct = meta.celltype.to_numpy()[ok]
    ds = meta.dataset.to_numpy()[ok]
    rows, t0 = [], time.time()
    for arm in ARMS:
        X = np.asarray(arms[arm], float)[ok]
        rows.append(dict(task=task, arm=arm) | metrics.evaluate(X, ct, ds, seed))
        print(f'  {task} {arm} {time.time()-t0:.0f}s', flush=True)
    df = pd.DataFrame(rows)
    out = paths.ROOT / 'results' / 'integration' / f'{task}.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f'{task}: -> {out}', flush=True)
    return df


if __name__ == '__main__':
    run(sys.argv[1])
