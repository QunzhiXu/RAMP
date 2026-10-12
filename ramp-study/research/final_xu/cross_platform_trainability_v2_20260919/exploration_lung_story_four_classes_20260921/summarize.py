"""Rescore frozen Lung four-class predictions and audit paired scope changes."""
import os
import sys
sys.dont_write_bytecode = True
from pathlib import Path
OUT = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(OUT / 'matplotlib')
from run import (ROOT, TASK, CLASSES, config, primary_jobs, job_key, fixed_plan,
                 load_task, restricted_task, save)
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix
from trainability.core import digest, file_sha256
from trainability.scoring import classification_score

METRICS = ['cell_accuracy', 'cell_macro_f1', 'cell_balanced_accuracy',
           'patient_accuracy', 'patient_macro_f1', 'patient_balanced_accuracy']


def table(df, columns):
    lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    for _, row in df.iterrows():
        lines.append('| ' + ' | '.join(f'{row[c]:.4f}' if isinstance(row[c], (float, np.floating))
                                       else str(row[c]) for c in columns) + ' |')
    return '\n'.join(lines)


def per_class(cm):
    cm = np.asarray(cm, dtype=float)
    tp = np.diag(cm)
    recall = np.divide(tp, cm.sum(1), out=np.zeros(len(tp)), where=cm.sum(1) > 0)
    precision = np.divide(tp, cm.sum(0), out=np.zeros(len(tp)), where=cm.sum(0) > 0)
    denominator = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2*tp, denominator, out=np.zeros(len(tp)), where=denominator > 0)
    return f1, recall, precision


def coarse_labels(labels):
    return np.asarray(['T cell' if x in {'CD4 T', 'CD8 T'} else x for x in labels])


def main():
    manifest = json.loads((OUT / 'MANIFEST.json').read_text())
    assert json.loads((OUT / 'RUN_COMPLETE.json').read_text())['all_passed']
    for p, h in {**manifest['protected_files'], **manifest['experiment_hashes']}.items():
        assert file_sha256(p) == h, p
    full = load_task(TASK)
    task = restricted_task(full)
    md = task.metadata.set_index('cell_id')
    all_old_classes = full.receipt['classes']
    coarse = ['T cell', 'Myeloid / APC', 'Tumor / epithelial']
    coarse_indices = [all_old_classes.index(c) for c in coarse]
    rebuilt_plans = {}
    for j in primary_jobs('ramp'):
        plan, parents, _ = fixed_plan(full, task, j)
        rebuilt_plans[job_key(j)] = (plan, parents)
    records, class_rows, confusion_rows, patients, baseline_rows = [], [], [], [], []
    for arm in manifest['arms']:
        method = arm['method']
        old_jobs = {job_key(j): j for j in primary_jobs(method)}
        for j in manifest['jobs']:
            folder = OUT / 'jobs' / method / j['id']
            proof = json.loads((folder / 'EXPERIMENT_COMPLETE.json').read_text())
            assert proof['manifest_sha256'] == file_sha256(OUT / 'MANIFEST.json')
            for p, h in proof['files'].items(): assert file_sha256(p) == h
            ledger = json.loads((folder / 'ledger.json').read_text())
            plan = json.loads((OUT / 'plans' / (j['id'] + '.json')).read_text())
            for key in ['fit', 'validation', 'calibration']: assert ledger[key] == plan[key]
            assert ledger['target_cell_ids'] == plan['target_ids']
            assert ledger['candidates'] == j['candidates']
            assert ledger['formal_benchmark'] is False
            rebuilt, parents = rebuilt_plans[job_key(j)]
            assert parents == plan['parent_source_patients']
            assert list(rebuilt.fit) == plan['fit'] and list(rebuilt.validation) == plan['validation']
            with np.load(folder / 'predictions.npz', allow_pickle=False) as f:
                ids, pred = f['cell_ids'], f['predictions']
            assert ids.tolist() == plan['target_ids'] and set(pred) <= set(CLASSES)
            y = md.loc[ids, 'celltype'].to_numpy()
            pats = md.loc[ids, 'patient_id'].to_numpy()
            assert set(y) == set(CLASSES)
            result = json.loads((folder / 'RESULT.json').read_text())
            assert result['prediction_receipt']['sha256'] == file_sha256(folder / 'predictions.npz')
            rescored = classification_score(y, pred, pats, CLASSES)
            for metric in ['accuracy', 'macro_f1', 'balanced_accuracy']:
                assert np.isclose(rescored[metric], result['score'][metric], atol=1e-12)
            assert not result['selection'].get('constant_classifier_fallback', False)
            key = dict(method=method, source=j['source'], target=j['target'], family=j['family'], seed=j['seed'])
            row = dict(**key, cells=len(y), patients=len(set(pats)),
                cell_accuracy=accuracy_score(y, pred),
                cell_macro_f1=f1_score(y, pred, labels=CLASSES, average='macro', zero_division=0),
                cell_balanced_accuracy=balanced_accuracy_score(y, pred),
                patient_accuracy=rescored['accuracy'], patient_macro_f1=rescored['macro_f1'],
                patient_balanced_accuracy=rescored['balanced_accuracy'],
                supervision_sha256=j['supervision_sha256'], target_ids_sha256=digest(ids.tolist()))
            records.append(row)
            for weighting, cm in [('cell', confusion_matrix(y, pred, labels=CLASSES)),
                                  ('patient', np.asarray(rescored['confusion_matrix']))]:
                fs, recall, precision = per_class(cm)
                for k, ct in enumerate(CLASSES):
                    class_rows.append(dict(**key, weighting=weighting, celltype=ct,
                        f1=fs[k], recall=recall[k], precision=precision[k]))
                    for p, pc in enumerate(CLASSES):
                        confusion_rows.append(dict(**key, weighting=weighting, truth=ct, predicted=pc,
                            mass=float(cm[k,p]), conditional_fraction=float(cm[k,p]/cm[k].sum())))
            for pat in sorted(set(pats)):
                m = pats == pat
                patients.append(dict(**key, patient=pat, cells=int(m.sum()),
                    accuracy=accuracy_score(y[m], pred[m]),
                    macro_f1=f1_score(y[m], pred[m], labels=CLASSES, average='macro', zero_division=0)))
            # Old head cannot predict CD4 versus CD8. Compare only a shared
            # three-class ontology on exactly the same target cells.
            oldj = old_jobs[job_key(j)]
            oldfolder = Path(config(method)['output']) / 'jobs' / oldj['id']
            oldresult = json.loads((oldfolder / 'RESULT.json').read_text())
            assert file_sha256(oldfolder / 'predictions.npz') == oldresult['prediction_receipt']['sha256']
            with np.load(oldfolder / 'predictions.npz', allow_pickle=False) as f:
                lookup = dict(zip(f['cell_ids'], f['predictions']))
            assert set(ids) <= set(lookup)
            oldpred = np.asarray([lookup[i] for i in ids])
            cy, cp = coarse_labels(y), coarse_labels(pred)
            old_patient = classification_score(cy, oldpred, pats, all_old_classes)
            new_patient = classification_score(cy, cp, pats, coarse)
            baseline_rows.append(dict(**key,
                old_cell_accuracy_same_targets_coarse=accuracy_score(cy, oldpred),
                new_cell_accuracy_same_targets_coarse=accuracy_score(cy, cp),
                old_cell_macro_f1_same_targets_coarse=f1_score(cy, oldpred, labels=coarse, average='macro', zero_division=0),
                new_cell_macro_f1_same_targets_coarse=f1_score(cy, cp, labels=coarse, average='macro', zero_division=0),
                old_patient_accuracy_same_targets_coarse=old_patient['accuracy'],
                new_patient_accuracy_same_targets_coarse=new_patient['accuracy'],
                old_patient_macro_f1_same_targets_coarse=per_class(old_patient['confusion_matrix'])[0][coarse_indices].mean(),
                new_patient_macro_f1_same_targets_coarse=new_patient['macro_f1'],
                old_predictions_outside_three=float((~np.isin(oldpred, coarse)).mean()),
                original_five_class_patient_macro_f1=oldresult['score']['macro_f1']))
    df = pd.DataFrame(records)
    assert len(df) == 540
    matched = df.groupby(['source', 'target', 'seed', 'family'])
    assert matched.size().eq(10).all() and df.groupby('method').size().eq(54).all()
    assert matched.supervision_sha256.nunique().eq(1).all()
    assert matched.target_ids_sha256.nunique().eq(1).all()
    df.to_csv(OUT / 'SCORES_PER_JOB.csv', index=False)
    classes = pd.DataFrame(class_rows)
    classes.to_csv(OUT / 'PER_CLASS_PER_JOB.csv', index=False)
    pd.DataFrame(patients).to_csv(OUT / 'PER_PATIENT_PER_JOB.csv', index=False)
    pd.DataFrame(confusion_rows).to_csv(OUT / 'CONFUSIONS_PER_JOB.csv', index=False)
    summary = df.groupby('method')[METRICS].mean().reset_index()
    for metric in METRICS:
        summary[metric + '_rank'] = summary[metric].rank(ascending=False, method='min').astype(int)
    summary = summary.sort_values('cell_accuracy', ascending=False)
    summary.to_csv(OUT / 'FULL_COHORT_SUMMARY.csv', index=False)
    models = df.groupby(['method', 'family'])[METRICS].mean().reset_index()
    models.to_csv(OUT / 'BY_MODEL.csv', index=False)
    directions = df.groupby(['method', 'source', 'target'])[METRICS].mean().reset_index()
    directions.to_csv(OUT / 'BY_DIRECTION.csv', index=False)
    cm = classes.groupby(['method', 'weighting', 'celltype'])[['f1', 'recall', 'precision']].mean().reset_index()
    cm.to_csv(OUT / 'PER_CLASS_MEANS.csv', index=False)
    classes.groupby(['method', 'weighting', 'source', 'target', 'celltype'])[['f1', 'recall', 'precision']].mean().to_csv(OUT / 'PER_CLASS_BY_DIRECTION.csv')
    pd.DataFrame(confusion_rows).groupby(['method', 'weighting', 'source', 'target', 'truth', 'predicted']).conditional_fraction.mean().to_csv(OUT / 'CONFUSION_MEANS.csv')
    baseline = pd.DataFrame(baseline_rows)
    baseline.to_csv(OUT / 'OLD_VS_NEW_COARSE_HEAD_PER_JOB.csv', index=False)
    bmetrics = [c for c in baseline.columns if c.startswith(('old_', 'new_', 'original_'))]
    oldmean = baseline.groupby('method')[bmetrics].mean().reset_index()
    oldmean.to_csv(OUT / 'OLD_VS_NEW_COARSE_HEAD_SUMMARY.csv', index=False)
    old_audit = pd.read_csv(ROOT / 'reports/lung_story_scope_audit_20260921/METHOD_SUMMARY.csv').set_index('method')
    for _, r in oldmean.iterrows():
        assert np.isclose(r.original_five_class_patient_macro_f1, old_audit.loc[r.method, 'patient_macro_f1'], atol=1e-12)
    for p, h in {**manifest['protected_files'], **manifest['experiment_hashes']}.items():
        assert file_sha256(p) == h, p
    save(OUT / 'CHECKS.json', dict(jobs=540, ten_methods_same_full_U=True,
        paired_supervision_and_target_ids=True, original_source_patient_splits_reproduced=True,
        patient_disjoint_source_validation=True, all_four_classes_ge10_in_every_fit_and_validation=True,
        predictions_verified_and_rescored=True, original_files_unchanged=True,
        original_five_class_audit_reproduced=True, only_T_labels_refined=True,
        script_sha256=file_sha256(__file__)))
    cols = ['method', 'cell_accuracy', 'cell_macro_f1', 'patient_accuracy', 'patient_macro_f1']
    report = f'''# Lung story 四分类重训结果

540/540 作业完成并独立重算评分。10 种方法 × 3 模型 × 3 seed × 6 方向；原 28 Senosain + 28 Cords + 28 Zheng 患者、84,000 个全类型对齐细胞及九个共同 marker 均冻结不变。

## 总体结果

{table(summary, cols)}

每个方向先计算目标细胞等权（cell）或患者等权混淆矩阵（patient）指标，再对六个方向、三个模型和三个 seed 等权平均。患者指标不是逐患者 F1 的简单平均；随机 seed 不是独立生物学队列。源模型选择仍使用原患者等权 macro-F1 和原候选参数。

## RAMP 逐类、方向、模型

{table(cm[cm.method.eq('ramp')], ['weighting', 'celltype', 'f1', 'recall', 'precision'])}

{table(directions[directions.method.eq('ramp')], ['source', 'target'] + METRICS)}

{table(models[models.method.eq('ramp')], ['family'] + METRICS)}

## 原五分类与新四分类不能直接作同任务前后比较

原五类包含合并 T、Endothelial、Stroma、Myeloid/APC、Tumor；本次是 CD4、CD8、Myeloid/APC、Tumor。原 RAMP 患者等权 F1 0.5022 并不是本次四分类的对照基线。模型输出类别、训练预算分配及目标范围同时变化，不代表对齐算法得到改进。

为作同一目标细胞的描述性核对，将新预测的 CD4/CD8 合回 T，与原头按共同 T/Myeloid/Tumor 三类比较。原预测为 Endothelial/Stroma 的错误全部保留；不按预测是否属于选定类过滤细胞。这张表是共同三类分辨率，不能当四类主成绩：

{table(oldmean, ['method', 'old_cell_accuracy_same_targets_coarse', 'new_cell_accuracy_same_targets_coarse', 'old_cell_macro_f1_same_targets_coarse', 'new_cell_macro_f1_same_targets_coarse'])}

## 设计与解释范围

- 对齐仍基于全 84k 细胞，使用原冻结表示；仅重新训练监督头。全部九 marker（含 CD3/CD4/CD8）都是输入，这是注释而非缺失 marker 预测。
- 四类源训练、源验证和目标评分范围一致，不设 Remaining。Zheng Th+Treg 为 CD4，Tc 为 CD8，587 个 DNT 不进入此四类任务。
- Senosain 的 2,096 个 Cycling tumor 仍沿用原 Tumor 大类，没有根据预测分数改标或删除；其谱系不确定性保留。Myeloid/APC 是亚类组成不同的宽泛髓系宏类。
- 先重建原五类的 source 患者拆分，再在各角色内筛选四类，均匀抽取 500 训练/512 验证。所有方法使用同一批细胞；不额外做类别平衡。各类实际训练和验证均 >=10，Cords CD4 最小训练数恰为 10。
- 排除 Stroma 的依据是 Senosain 历史标签语义不明确；内皮在包含它的候选五类中最少 6 个训练细胞，属于此预算下支持不足，不等于其标签不可比。
- 本方案是在查看旧结果并审查标签后提出的探索，不是预注册最终七组类别筛选规则。原完整结果保留；无正式图稿替换。

输入与原结果哈希均未改变。`CHECKS.json` 记录预测重评分与配对检查；`MANIFEST.json`、`LABEL_MAPPING.csv.gz`、`plans/` 保存映射和拆分。
'''
    (OUT / 'RESULTS_ZH.md').write_text(report)
    plot(df, summary)
    print(summary[cols].to_string(index=False), flush=True)
    print(cm[cm.method.eq('ramp')].to_string(index=False), flush=True)
    print(directions[directions.method.eq('ramp')].to_string(index=False), flush=True)


def plot(df, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = {'ramp': 'RAMP', 'unaligned': 'Unaligned', 'clipped_zscore': 'Clipped z-score',
        'minmax': 'Min-max', 'percentile': 'Percentile', 'harmony_direct': 'Harmony Direct',
        'cycombine': 'cyCombine', 'cytonorm': 'CytoNorm', 'fastmnn': 'fastMNN', 'maxfuse': 'MaxFuse'}
    order = summary.method.tolist()
    plt.rcParams.update({'font.size': 10, 'axes.labelweight': 'bold', 'axes.titleweight': 'bold', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.3), sharey=True)
    for ax, metric, title in zip(axes, ['cell_accuracy', 'cell_macro_f1'], ['Accuracy', 'Macro-F1']):
        for i, method in enumerate(order):
            vals = df[df.method.eq(method)].sort_values(['source', 'target', 'family', 'seed'])[metric].to_numpy()
            color = '#087e8b' if method == 'ramp' else '#738194'
            ax.scatter(vals, i + np.linspace(-.18, .18, len(vals)), s=11, alpha=.5, color=color)
            ax.scatter(vals.mean(), i, s=65, marker='D', edgecolor='white', linewidth=.8, color=color, zorder=4)
        ax.set(xlim=(0, 1), xlabel=title, title=title, yticks=range(len(order)), yticklabels=[names[m] for m in order])
        if not ax.yaxis_inverted(): ax.invert_yaxis()
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='x', color='#e5e5e5', linewidth=.6)
        ax.set_axisbelow(True)
    fig.suptitle('Lung story: CD4 / CD8 / Myeloid / Tumor annotation', fontweight='bold')
    fig.text(.5, .015, '54 model/seed/direction evaluations per method; diamonds show means', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .95))
    fig.savefig(OUT / 'METHOD_COMPARISON.png', dpi=220)
    fig.savefig(OUT / 'METHOD_COMPARISON.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
