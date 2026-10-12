"""Read-only audit of cross_platform_trainability; writes only beside this script.

No aligner, predictor, or original summarization script is executed.
Run: PYTHONDONTWRITEBYTECODE=1 python3 recompute_audit.py
"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

SRC = Path('/Users/qxu6/Documents/New project/cross_platform_trainability')
OUT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(SRC / 'src'))
import common
import paths
import run_task

TASKS = [t['name'] for t in common.D['tasks']]


def load(folder):
    return pd.concat([pd.read_csv(SRC / 'results' / folder / f'{t}.csv.gz',
                                  low_memory=False) for t in TASKS], ignore_index=True)


def save(frame, name):
    frame.to_csv(OUT / name, index=False)


manifest = json.loads((SRC / 'data/MANIFEST.json').read_text())
root = Path(manifest['source_root'])
checks = {}
for t in manifest['tasks'].values():
    for k in ('metadata', 'markers'):
        f = t[k]
        checks[str(root / f['path'])] = f['sha256']
    for a in t['arms'].values():
        for f in a['files']:
            checks[str(root / f['path'])] = f['sha256']
checks[str(SRC / 'design.json')] = manifest['design_sha256']
hashed = []
for filename, expected in checks.items():
    h = hashlib.sha256()
    with open(filename, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    hashed.append(dict(path=filename, expected=expected, actual=h.hexdigest(),
                       matches=h.hexdigest() == expected))
save(pd.DataFrame(hashed), 'input_hash_checks.csv')

main = load('raw')
key = ['task', 'held_out', 'train_platform', 'rep']
classification = main[(main.job == 'job1_celltype') & (main.train_set == 'A@N')]
m1 = classification.groupby(['task', 'arm', 'capacity']).macro_f1.mean().reset_index()
save(m1, 'classification_task_means.csv')

m4 = main[(main.job == 'job1_celltype') & (main.capacity == 'linear')
          & (main.arm == 'RAMP')].pivot_table(index=key, columns='train_set', values='n_train')
m4 = m4.dropna(subset=['A@2N', 'A+B@2N'])
m4['extra_cells_in_pooled'] = m4['A+B@2N'] - m4['A@2N']
save(m4.reset_index(), 'm4_actual_training_budgets.csv')

rare, recipe, fewshot = [], [], []
r4 = main[(main.job == 'job4_rare') & (main.capacity == 'linear') & (main.arm == 'RAMP')]
for task in TASKS:
    meta = common.load_meta(task)
    assert meta.cell_id.is_unique
    for rt in run_task.rare_targets(meta):
        target = meta[meta.dataset == rt['rare_in']]
        v = r4[(r4.task == task) & (r4.rare_label == rt['label'])]
        rare.append(dict(task=task, label=rt['label'], target=rt['rare_in'],
                         natural_fraction=float((target.celltype == rt['label']).mean()),
                         test_fraction=float(v.rare_frac_test.mean())))
    tab = pd.crosstab(meta.celltype, meta.dataset)
    shared = [c for c in tab.index[(tab > 0).all(1)] if c not in common.AMBIGUOUS]
    for ds in tab.columns:
        for c in shared:
            count = int(tab.loc[c, ds])
            if count <= 100:
                fewshot.append(dict(task=task, target=ds, label=c, cells=count,
                                    test_cells_after_m5_reservation=max(count - 100, 0)))
    if paths.LAYOUT[task] == 'application':
        x = np.load(paths.arm_files(task, 'Raw')[0])
        raw = np.load(paths.root(task) / 'input/raw.npy')
        recipe.append(dict(task=task, input_path=str(paths.arm_files(task, 'Raw')[0]),
                           min_value=float(x.min()), negative_fraction=float((x < 0).mean()),
                           identical_to_source_scale=bool(np.array_equal(raw, x))))
save(pd.DataFrame(rare), 'rare_population_prevalence.csv')
save(pd.DataFrame(recipe), 'recipe_input_audit.csv')
save(pd.DataFrame(fewshot), 'fewshot_lost_test_classes.csv')

s = load('scaling')
s = s[(s.job == 'job1_celltype') & s.balanced]
support = s[s.arm == 'RAMP'].groupby('per_class').agg(
    tasks=('task', 'nunique'), direction_repetitions=('rep', 'size')).reset_index()
save(support, 'scaling_population_support.csv')
w = s.pivot_table(index=key, columns=['arm', 'per_class'], values='macro_f1')
pair = w[[('RAMP', 5), ('Raw', 2000)]].dropna()
pair.columns = ['RAMP_5_per_class', 'Raw_2000_per_class']
save(pair.reset_index(), 'scaling_matched_endpoint_comparison.csv')
paired_means = pair.groupby('task').mean()
save(paired_means.reset_index(), 'scaling_matched_endpoint_task_means.csv')
complete = s.pivot_table(index=key + ['arm'], columns='per_class', values='macro_f1').dropna()
save(complete.reset_index(), 'scaling_complete_directions_all_levels.csv')

lomo = load('lomo')
lo = lomo[(lomo.capacity == 'linear') & (lomo.train_set == 'A@N')]
lo_means = lo.groupby(['task', 'arm']).spearman.mean().unstack()
save(lo_means.reset_index(), 'lomo_linear_task_means.csv')
units = lo.pivot_table(index=key + ['marker'], columns='arm', values='n_units')
records = [json.loads(f.read_text()) for f in (SRC / 'lomo_work/out').glob('*/*/*.json')]

pre = load('pretrain')
pk = key + ['arm', 'train_set']
pre = pre[pre.job == 'job1_celltype'].merge(
    main[(main.job == 'job1_celltype') & (main.capacity == 'linear')],
    on=pk, suffixes=('_pre', '_no'))
pre['delta'] = pre.macro_f1_pre - pre.macro_f1_no
pre_summary = pd.DataFrame({
    'row_weighted_delta': pre.groupby('arm').delta.mean(),
    'equal_task_delta': pre.groupby(['task', 'arm']).delta.mean().groupby('arm').mean()})
save(pre_summary.reset_index(), 'pretraining_deltas.csv')

seed_expr = 'print((20260918+abs(hash(("bone_marrow","job1","RAMP","mlp32","A@N",0))))%1000000)'
seed_demo = [subprocess.check_output([sys.executable, '-c', seed_expr], text=True).strip()
             for _ in range(3)]
summary = {
    'source_directory_modified': False,
    'main_result_rows': len(main),
    'hash_checks': len(hashed), 'hash_failures': sum(not x['matches'] for x in hashed),
    'm4_pairs': len(m4), 'm4_unequal_budgets': int((m4.extra_cells_in_pooled != 0).sum()),
    'm4_largest_extra_cells': float(m4.extra_cells_in_pooled.max()),
    'm1_linear_equal_task_means': m1[m1.capacity == 'linear'].groupby('arm').macro_f1.mean().to_dict(),
    'm1_ramp_task_wins': {cap: int((m1[m1.capacity == cap].pivot(
        index='task', columns='arm', values='macro_f1').idxmax(axis=1) == 'RAMP').sum())
        for cap in ('linear', 'knn', 'mlp32')},
    'lomo_records': len(records),
    'lomo_target_in_feature_names': sum(r['held_out'] in r['markers'] for r in records),
    'lomo_linear_equal_task_means': lo_means.mean().to_dict(),
    'lomo_linear_unequal_unit_counts_across_arms': int((units.max(axis=1) != units.min(axis=1)).sum()),
    'scaling_paired_endpoint_task_count': len(paired_means),
    'scaling_paired_endpoint_equal_task_means': paired_means.mean().to_dict(),
    'PYTHONHASHSEED': os.environ.get('PYTHONHASHSEED'),
    'fresh_process_seed_examples': seed_demo,
    'saved_main_seed_columns': [c for c in main.columns if 'seed' in c],
}
(OUT / 'audit_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
