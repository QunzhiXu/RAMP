"""The same leakage measure, applied to the main grid's full-panel alignments.

`leakage_lomo.py` measures the R-squared of the held-out marker on the refitted
features.  The figure quoted in amendment 8 (Raw 0.134, fastMNN 0.251) came
from an earlier ad-hoc calculation whose estimator was not recorded, so it is
recomputed here with exactly the estimator used on the refits: one global
ordinary least squares fit per (task, marker, arm) over all cells, features
being the arm's columns with column j removed.
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
import common, ramp_run

SKIP = {'MaxFuse'}


def run(tasks=None):
    rows = []
    for task in (tasks or ['bone_marrow', 'tnbc', 'melanoma', 'lung_cancer',
                           'lung_story', 'melanoma_story', 'hnscc_story']):
        markers = ramp_run.spec(task)['markers']
        arms_all = [a for a in common.ARMS if a not in SKIP]
        _, _, arms = common.load_arms(task, arms=arms_all)
        raw = np.asarray(arms['Raw'], float)
        for j, mk in enumerate(markers):
            for arm, M in arms.items():
                M = np.asarray(M, float)
                if M.shape[1] == len(markers):
                    X = M[:, [k for k in range(M.shape[1]) if k != j]]
                else:
                    X = M                      # no marker axis (embedding)
                rows.append(dict(task=task, marker=mk, arm=arm,
                                 r2=float(LinearRegression().fit(X, raw[:, j])
                                          .score(X, raw[:, j]))))
        print(f'  {task} done', flush=True)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    df = run(sys.argv[1:] or None)
    out = HERE.parents[1] / 'results' / 'lomo' / 'leakage_main_grid.csv'
    df.to_csv(out, index=False)
    print("R-squared of marker j on the FULL-PANEL arm's remaining columns")
    print(df.groupby('arm').r2.agg(['mean', 'median', 'max']).round(4).to_string())
