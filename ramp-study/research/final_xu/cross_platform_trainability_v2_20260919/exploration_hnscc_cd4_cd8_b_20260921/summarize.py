"""Verify every prediction/ledger, score cell and patient weighting, report all arms."""
import os
import sys
sys.dont_write_bytecode = True
from pathlib import Path
OUT = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(OUT / 'matplotlib')
from run import (ROOT, CAP, TASK, CLASSES, METHODS, restricted_task, config,
                 primary_jobs, job_key, save)
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix
from trainability.core import digest, file_sha256
from trainability.scoring import classification_score


def table(df, columns):
    lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    for _, row in df.iterrows():
        lines.append('| ' + ' | '.join(f'{row[c]:.4f}' if isinstance(row[c], (float, np.floating))
                                       else str(row[c]) for c in columns) + ' |')
    return '\n'.join(lines)


def main():
    manifest = json.loads((OUT / 'MANIFEST.json').read_text())
    complete = json.loads((OUT / 'RUN_COMPLETE.json').read_text())
    assert complete['all_passed']
    for p, h in {**manifest['protected_files'], **manifest['experiment_hashes']}.items():
        assert file_sha256(p) == h, p
    tasks = {c: restricted_task(c) for c in ['full', 'cap20']}
    records, per_class, per_patient, baselines, comparison_hashes = [], [], [], [], {}
    for arm in manifest['arms']:
        task = tasks[arm['cohort']]
        md = task.metadata.set_index('cell_id')
        old_jobs = {job_key(j): j for j in primary_jobs(arm['method'])}
        for j in manifest['jobs'][arm['cohort']]:
            folder = OUT / 'jobs' / arm['id'] / j['id']
            proof = json.loads((folder / 'EXPERIMENT_COMPLETE.json').read_text())
            assert proof['manifest_sha256'] == file_sha256(OUT / 'MANIFEST.json')
            for p, h in proof['files'].items(): assert file_sha256(p) == h
            ledger = json.loads((folder / 'ledger.json').read_text())
            plan = json.loads((OUT / 'plans' / arm['cohort'] / (j['id'] + '.json')).read_text())
            for key in ['fit', 'validation', 'calibration']: assert ledger[key] == plan[key]
            assert ledger['target_cell_ids'] == plan['target_ids']
            assert ledger['candidates'] == j['candidates']
            assert ledger['formal_benchmark'] is False
            with np.load(folder / 'predictions.npz', allow_pickle=False) as f:
                ids, pred = f['cell_ids'], f['predictions']
            assert ids.tolist() == plan['target_ids']
            assert set(pred) <= set(CLASSES)
            y = md.loc[ids, 'celltype'].to_numpy()
            pats = md.loc[ids, 'patient_id'].to_numpy()
            result = json.loads((folder / 'RESULT.json').read_text())
            rescored = classification_score(y, pred, pats, CLASSES)
            for metric in ['accuracy', 'macro_f1', 'balanced_accuracy']:
                assert np.isclose(rescored[metric], result['score'][metric], atol=1e-12)
            assert not result['selection'].get('constant_classifier_fallback', False)
            row = dict(arm=arm['id'], cohort=arm['cohort'], method=arm['method'],
                source=j['source'], target=j['target'], family=j['family'], seed=j['seed'],
                cells=len(y), patients=len(set(pats)),
                cell_accuracy=accuracy_score(y, pred),
                cell_macro_f1=f1_score(y, pred, labels=CLASSES, average='macro', zero_division=0),
                cell_balanced_accuracy=balanced_accuracy_score(y, pred),
                cell_weighted_f1=f1_score(y, pred, labels=CLASSES, average='weighted', zero_division=0),
                patient_accuracy=result['score']['accuracy'],
                patient_macro_f1=result['score']['macro_f1'],
                patient_balanced_accuracy=result['score']['balanced_accuracy'],
                supervision_sha256=j['supervision_sha256'],
                target_ids_sha256=digest(ids.tolist()))
            records.append(row)
            cm = confusion_matrix(y, pred, labels=CLASSES)
            fs = f1_score(y, pred, labels=CLASSES, average=None, zero_division=0)
            for k, ct in enumerate(CLASSES):
                per_class.append(dict(arm=arm['id'], source=j['source'], target=j['target'],
                    family=j['family'], seed=j['seed'], celltype=ct, f1=fs[k],
                    recall=cm[k,k]/cm[k].sum(), precision=cm[k,k]/max(1, cm[:,k].sum()),
                    true_cells=int(cm[k].sum()), predicted_cells=int(cm[:,k].sum())))
            for pat in sorted(set(pats)):
                m = pats == pat
                per_patient.append(dict(arm=arm['id'], source=j['source'], target=j['target'],
                    family=j['family'], seed=j['seed'], patient=pat, cells=int(m.sum()),
                    accuracy=accuracy_score(y[m], pred[m]),
                    macro_f1=f1_score(y[m], pred[m], labels=CLASSES, average='macro', zero_division=0)))
            # Descriptive old-head comparison on EXACTLY the same true B/CD4/CD8 targets.
            # Wrong predictions of identities outside B/CD4/CD8 remain errors; never filter on prediction.
            if arm['cohort'] == 'full':
                oldj = old_jobs[job_key(j)]
                oldfolder = Path(config(arm['method'])['output']) / 'jobs' / oldj['id']
            else:
                variant = 'refit_cap20' if arm['id'].endswith('refit') else 'old_alignment_same_subset'
                oldfolder = CAP / 'jobs' / TASK / variant / j['original_job_id']
            oldfile = oldfolder / 'predictions.npz'
            comparison_hashes[str(oldfile)] = file_sha256(oldfile)
            oldresult = json.loads((oldfolder / 'RESULT.json').read_text())
            assert comparison_hashes[str(oldfile)] == oldresult['prediction_receipt']['sha256']
            with np.load(oldfile, allow_pickle=False) as f:
                lookup = dict(zip(f['cell_ids'], f['predictions']))
            assert set(ids) <= set(lookup)
            oldpred = np.asarray([lookup[i] for i in ids])
            baselines.append(dict(arm=arm['id'], source=j['source'], target=j['target'],
                family=j['family'], seed=j['seed'],
                nine_class_accuracy_on_same_B_T_cells=accuracy_score(y, oldpred),
                three_class_accuracy_on_same_B_T_cells=row['cell_accuracy'],
                nine_class_predictions_outside_B_T=float((~np.isin(oldpred, CLASSES)).mean()),
                original_nine_class_patient_macro_f1=oldresult['score']['macro_f1'],
                original_nine_class_patient_accuracy=oldresult['score']['accuracy']))
    df = pd.DataFrame(records)
    assert len(df) == 216
    group = df[df.cohort.eq('full')].groupby(['source', 'target', 'seed', 'family'])
    assert group.size().eq(10).all()
    assert group.supervision_sha256.nunique().eq(1).all()
    assert group.target_ids_sha256.nunique().eq(1).all()
    capgroup = df[df.cohort.eq('cap20')].groupby(['source', 'target', 'seed', 'family'])
    assert capgroup.supervision_sha256.nunique().eq(1).all()
    assert capgroup.target_ids_sha256.nunique().eq(1).all()
    metrics = ['cell_accuracy', 'cell_macro_f1', 'cell_balanced_accuracy',
               'cell_weighted_f1', 'patient_accuracy', 'patient_macro_f1', 'patient_balanced_accuracy']
    df.to_csv(OUT / 'SCORES_PER_JOB.csv', index=False)
    pd.DataFrame(per_class).to_csv(OUT / 'PER_CLASS_PER_JOB.csv', index=False)
    pd.DataFrame(per_patient).to_csv(OUT / 'PER_PATIENT_PER_JOB.csv', index=False)
    old = pd.DataFrame(baselines)
    old.to_csv(OUT / 'OLD_NINE_CLASS_HEAD_ON_B_T_CELLS.csv', index=False)
    full = df[df.cohort.eq('full')]
    summary = full.groupby('method')[metrics].mean().reset_index()
    for metric in metrics:
        summary[metric + '_rank'] = summary[metric].rank(ascending=False, method='min').astype(int)
    summary = summary.sort_values('cell_accuracy', ascending=False)
    summary.to_csv(OUT / 'FULL_COHORT_SUMMARY.csv', index=False)
    full.groupby(['method', 'family'])[metrics].mean().to_csv(OUT / 'FULL_COHORT_BY_MODEL.csv')
    directions = full.groupby(['method', 'source', 'target'])[metrics].mean().reset_index()
    directions.to_csv(OUT / 'FULL_COHORT_BY_DIRECTION.csv', index=False)
    caps = df[df.cohort.eq('cap20')].groupby('arm')[metrics].mean().reset_index()
    caps.to_csv(OUT / 'CAP20_RAMP_ONLY_SUMMARY.csv', index=False)
    pd.DataFrame(per_class).groupby(['arm', 'celltype'])[['f1', 'recall', 'precision']].mean().to_csv(OUT / 'PER_CLASS_MEANS.csv')
    save(OUT / 'CHECKS.json', dict(jobs=216, ten_methods_same_full_U=True,
        paired_supervision_and_target_ids=True, patient_disjoint_source_validation=True,
        all_three_classes_in_every_fit_and_validation=True, predictions_verified_and_rescored=True,
        original_files_unchanged=True, cap20_not_in_method_ranking=True,
        original_predictions_sha256=comparison_hashes))
    oldmean = old.groupby('arm').mean(numeric_only=True).reset_index()
    oldmean.to_csv(OUT / 'OLD_VS_NEW_HEAD_SUMMARY.csv', index=False)
    cols = ['method', 'cell_accuracy', 'cell_macro_f1', 'patient_accuracy', 'patient_macro_f1']
    previous = pd.read_csv(ROOT / 'exploration_hnscc_cd4_cd8_20260921' / 'FULL_COHORT_SUMMARY.csv')
    scope_comparison = previous[cols].merge(summary[cols], on='method', validate='one_to_one',
                                            suffixes=('_two_class', '_three_class'))
    for metric in cols[1:]:
        scope_comparison[metric + '_change'] = (scope_comparison[metric + '_three_class']
                                               - scope_comparison[metric + '_two_class'])
    scope_comparison.to_csv(OUT / 'TWO_VS_THREE_CLASS_SUMMARY.csv', index=False)
    sampling_rows = []
    all_md = tasks['full'].metadata.set_index('cell_id')
    for cohort in ['full', 'cap20']:
        previous_plans = {}
        for p in (ROOT / 'exploration_hnscc_cd4_cd8_20260921' / 'plans' / cohort).glob('*.json'):
            plan = json.loads(p.read_text())
            previous_plans[job_key(plan['job'])] = plan
        for p in (OUT / 'plans' / cohort).glob('*.json'):
            plan = json.loads(p.read_text())
            job = plan['job']
            oldplan = previous_plans[job_key(job)]
            for role in ['fit', 'validation']:
                same = set(all_md.loc[plan[role], 'patient_id']) == set(all_md.loc[oldplan[role], 'patient_id'])
                sampling_rows.append(dict(cohort=cohort, source=job['source'], seed=job['seed'],
                    family=job['family'], role=role, same_sampled_patient_set=same,
                    three_class_B_cells=int(all_md.loc[plan[role], 'celltype'].eq('B').sum())))
    pd.DataFrame(sampling_rows).to_csv(OUT / 'TWO_VS_THREE_CLASS_SAMPLING_CHECK.csv', index=False)
    scope_cols = ['method', 'cell_accuracy_two_class', 'cell_accuracy_three_class',
                  'cell_macro_f1_two_class', 'cell_macro_f1_three_class']
    report = f'''# HNSCC CD4/CD8/B 实验结果

全部 216 个作业完成。主比较为 10 种方法 × 3 种模型 × 3 个 seed × 2 个方向，共 180 个；cap20 的 RAMP 配对检查为另外 36 个。

## 全队列十方法比较

{table(summary, cols)}

`cell_*` 为每个方向内所有目标细胞等权；`patient_*` 为原实现的患者等权混淆矩阵评分。上表对两个方向、三个模型、三个 seed 等权汇总。源超参数选择仍使用原来的患者等权 macro-F1，没有根据目标表现换候选。每次 500 个训练细胞、512 个验证细胞。所有方法的细胞和候选网格完全匹配。

## 加入 B 前后的任务结果

{table(scope_comparison, scope_cols)}

两列分别为 CD4/CD8 二分类和 B/CD4/CD8 三分类。总训练预算均为 500，但三分类将部分预算用于 B，目标细胞范围也包含 B，因此这是不同任务范围下的描述性比较。B 保持原独立标签，Plasma 不并入 B。

## RAMP 方向结果

{table(directions[directions.method.eq('ramp')], ['source', 'target', 'cell_accuracy', 'cell_macro_f1', 'patient_accuracy', 'patient_macro_f1'])}

## 固定 cap20 子集：仅 RAMP 配对

{table(caps, ['arm', 'cell_accuracy', 'cell_macro_f1', 'patient_accuracy', 'patient_macro_f1'])}

这是相同 cap20 目标和监督细胞上的旧表示/先前重拟合表示对比，不与完整队列的其他方法混排。两者均在本次重新训练三分类头。

## 原九类分类器在相同 B/CD4/CD8 细胞上的表现

{table(oldmean[oldmean.arm.isin(['full__ramp', 'cap20__ramp_original', 'cap20__ramp_refit'])], ['arm', 'nine_class_accuracy_on_same_B_T_cells', 'three_class_accuracy_on_same_B_T_cells', 'nine_class_predictions_outside_B_T'])}

这里保留九类分类器把 B/CD4/CD8 细胞预测为其他类的错误，没有按预测标签筛选细胞。三类分类器在全部 500 个名额中只训练 B/CD4/CD8；九类分类器的 500 个名额分配给九类，监督内容不同，因此该差值不能单独归因于对齐改善。

## 解释范围

- 对齐仍使用全部细胞、完整共同 panel；本次没有修改或重跑对齐。
- 任务是在原标签 B、CD4 T、CD8 T 这三个类别之间注释；Plasma 不并入 B，不能宣称识别所有 HNSCC 细胞类型。
- CD3/CD4/CD8/CD20 marker 仍是输入；这不是缺失 marker 预测，也不是独立 marker 验证。
- 该探索是在看到原九类及 CD4/CD8 二分类结果后提出的，不是预注册筛选。
- 源 ADT 只有三个患者，三个随机种子不是三个独立队列。方向和模型分项均保留。
- 固定抽样中，每次训练含至少 37 个 B；ADT 有一个 seed 的源验证抽样仅含 4 个 B，未因此重新选 donor 或随机种子。源模型选择对这一类的稳定性有限。
- 原结果、原算法、正式图与稿件均未改动。`CHECKS.json` 与 `MANIFEST.json` 保存核验记录。
'''
    (OUT / 'RESULTS_ZH.md').write_text(report)
    plot(full, summary)
    print(summary[cols].to_string(index=False))
    print(caps[['arm', 'cell_accuracy', 'cell_macro_f1']].to_string(index=False))


def plot(full, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = {'ramp': 'RAMP', 'unaligned': 'Unaligned', 'clipped_zscore': 'Clipped z-score',
             'minmax': 'Min–max', 'percentile': 'Percentile', 'harmony_direct': 'Harmony Direct',
             'cycombine': 'cyCombine', 'cytonorm': 'CytoNorm', 'fastmnn': 'fastMNN', 'maxfuse': 'MaxFuse'}
    order = summary.method.tolist()
    plt.rcParams.update({'font.size': 10, 'axes.labelweight': 'bold', 'axes.titleweight': 'bold',
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, metric, title in zip(axes, ['cell_accuracy', 'cell_macro_f1'], ['Accuracy', 'Macro-F1']):
        for i, method in enumerate(order):
            values = full[full.method.eq(method)].sort_values(['source', 'family', 'seed'])[metric].to_numpy()
            color = '#087e8b' if method == 'ramp' else '#738194'
            ax.scatter(values, i + np.linspace(-.16, .16, len(values)), s=14, alpha=.6, color=color)
            ax.scatter(values.mean(), i, s=60, marker='D', edgecolor='white', linewidth=.8, color=color, zorder=4)
        ax.set(xlim=(0, 1), xlabel=title, title=title, yticks=range(len(order)),
               yticklabels=[names[m] for m in order])
        ax.invert_yaxis() if not ax.yaxis_inverted() else None
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='x', color='#e5e5e5', linewidth=.6)
        ax.set_axisbelow(True)
    fig.suptitle('HNSCC: cross-platform CD4/CD8/B annotation', fontweight='bold')
    fig.text(.5, .01, '18 model/seed/direction evaluations per method; diamonds show means', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .045, 1, .95))
    fig.savefig(OUT / 'METHOD_COMPARISON.png', dpi=220)
    fig.savefig(OUT / 'METHOD_COMPARISON.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
