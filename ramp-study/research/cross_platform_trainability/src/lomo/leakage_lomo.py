"""Did refitting actually remove the leak?

The main grid measured leakage as the R-squared of raw marker j regressed on
the arm's remaining columns.  The same quantity is recomputed here on the
refitted features.  Raw's value is the floor: it is the genuine
marker-to-marker correlation in the data, which no alignment can remove and
which a predictive model is entitled to use.
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
import common, ramp_run, job2_lomo


def run(tasks=None):
    rows = []
    for task in (tasks or ['bone_marrow', 'tnbc', 'melanoma', 'lung_cancer',
                           'lung_story', 'melanoma_story', 'hnscc_story']):
        markers = ramp_run.spec(task)['markers']
        meta = common.load_meta(task)
        raw = np.asarray(common.load_arms(task, arms=['Raw'])[2]['Raw'], float)
        units = meta.unit_key.astype(str).to_numpy()
        ds = meta.dataset.to_numpy()
        for j, mk in enumerate(markers):
            arms = job2_lomo.load_lomo(task, mk)
            if arms is None:
                continue
            keep = [k for k in range(raw.shape[1]) if k != j]
            arms = dict(arms)
            arms['Raw'] = raw[:, keep]
            arms['VirTues-style'] = common.virtues_style(raw[:, keep], units, ds)
            arms['Spatium-style'] = common.spatium_style(raw[:, keep], units)
            y = raw[:, j]
            for arm, M in arms.items():
                r2 = LinearRegression().fit(M, y).score(M, y)
                rows.append(dict(task=task, marker=mk, arm=arm, r2=float(r2)))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    df = run(sys.argv[1:] or None)
    out = HERE.parents[1] / 'results' / 'lomo' / 'leakage.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print('R-squared of the held-out marker on each arm\'s refitted features')
    print(df.groupby('arm').r2.agg(['mean', 'median', 'max']).round(4).to_string())
