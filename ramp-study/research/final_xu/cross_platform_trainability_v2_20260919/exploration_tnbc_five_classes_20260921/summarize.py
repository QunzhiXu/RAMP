"""Verify all frozen jobs and compare cell- and patient-weighted scores."""
import os
import sys
sys.dont_write_bytecode = True
from pathlib import Path
OUT = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(OUT / 'matplotlib')
from run import (ROOT, TASK, CLASSES, DISPLAY_LABELS, METHODS, AUDIT, config,
                 primary_jobs, job_key, fixed_plan, load_task, save)
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


def per_class(cm):
    cm = np.asarray(cm, dtype=float)
    tp = np.diag(cm)
    recall = np.divide(tp, cm.sum(1), out=np.zeros(len(tp)), where=cm.sum(1) > 0)
    precision = np.divide(tp, cm.sum(0), out=np.zeros(len(tp)), where=cm.sum(0) > 0)
    denominator = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2*tp, denominator, out=np.zeros(len(tp)), where=denominator > 0)
    return f1, recall, precision


def main():
    manifest = json.loads((OUT / 'MANIFEST.json').read_text())
    assert json.loads((OUT / 'RUN_COMPLETE.json').read_text())['all_passed']
    for p, h in {**manifest['protected_files'], **manifest['experiment_hashes']}.items():
        assert file_sha256(p) == h, p
    full = load_task(TASK)
    md = full.metadata.set_index('cell_id')
    all_classes = full.receipt['classes']
    selected_indices = [all_classes.index(ct) for ct in CLASSES]
    records, class_rows, confusion_rows, patients, baseline_rows, old_class_rows = [], [], [], [], [], []
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
            assert ledger['split_mode'] == j['split_mode']
            rebuilt, parent_patients, _ = fixed_plan(full, old_jobs[job_key(j)])
            assert parent_patients == plan['parent_source_patients']
            assert list(rebuilt.fit) == plan['fit'] and list(rebuilt.validation) == plan['validation']
            with np.load(folder / 'predictions.npz', allow_pickle=False) as f:
                ids, pred = f['cell_ids'], f['predictions']
            assert ids.tolist() == plan['target_ids'] and set(pred) <= set(CLASSES)
            y = md.loc[ids, 'celltype'].to_numpy()
            pats = md.loc[ids, 'patient_id'].to_numpy()
            result = json.loads((folder / 'RESULT.json').read_text())
            assert result['prediction_receipt']['sha256'] == file_sha256(folder / 'predictions.npz')
            rescored = classification_score(y, pred, pats, CLASSES)
            for metric in ['accuracy', 'macro_f1', 'balanced_accuracy']:
                assert np.isclose(rescored[metric], result['score'][metric], atol=1e-12)
            assert not result['selection'].get('constant_classifier_fallback', False)
            key = dict(method=method, source=j['source'], target=j['target'], family=j['family'], seed=j['seed'])
            row = dict(**key, cells=len(y), patients=len(set(pats)), split_mode=j['split_mode'],
                source_validation_f1=result['selection']['source_validation_score'],
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
                        display_label=DISPLAY_LABELS[ct], f1=fs[k], recall=recall[k], precision=precision[k]))
                    for p, pc in enumerate(CLASSES):
                        confusion_rows.append(dict(**key, weighting=weighting, truth=ct, predicted=pc,
                            mass=float(cm[k,p]), conditional_fraction=float(cm[k,p]/cm[k].sum())))
            for pat in sorted(set(pats)):
                m = pats == pat
                patients.append(dict(**key, patient=pat, cells=int(m.sum()),
                    accuracy=accuracy_score(y[m], pred[m]),
                    macro_f1=f1_score(y[m], pred[m], labels=CLASSES, average='macro', zero_division=0)))
            oldj = old_jobs[job_key(j)]
            oldfolder = Path(config(method)['output']) / 'jobs' / oldj['id']
            oldresult = json.loads((oldfolder / 'RESULT.json').read_text())
            assert file_sha256(oldfolder / 'predictions.npz') == oldresult['prediction_receipt']['sha256']
            with np.load(oldfolder / 'predictions.npz', allow_pickle=False) as f:
                lookup = dict(zip(f['cell_ids'], f['predictions']))
            assert set(ids) <= set(lookup)
            oldpred = np.asarray([lookup[i] for i in ids])
            # Old predictions outside the five target classes remain errors.
            # Include all seven output labels in the CM, then average only the
            # five target-class F1s; never silently discard outside predictions.
            old_patient = classification_score(y, oldpred, pats, all_classes)
            old_fs, old_recalls, old_precisions = per_class(old_patient['confusion_matrix'])
            old_f1 = old_fs[selected_indices].mean()
            old_ba = old_recalls[selected_indices].mean()
            for k in selected_indices:
                old_class_rows.append(dict(**key, celltype=all_classes[k], f1=old_fs[k],
                    recall=old_recalls[k], precision=old_precisions[k]))
            original_selected_f1 = per_class(oldresult['score']['confusion_matrix'])[0][selected_indices].mean()
            baseline_rows.append(dict(**key,
                old_cell_accuracy_same_targets=accuracy_score(y, oldpred),
                new_cell_accuracy_same_targets=row['cell_accuracy'],
                old_cell_macro_f1_same_targets=f1_score(y, oldpred, labels=CLASSES, average='macro', zero_division=0),
                new_cell_macro_f1_same_targets=row['cell_macro_f1'],
                old_patient_accuracy_same_targets=old_patient['accuracy'],
                new_patient_accuracy_same_targets=row['patient_accuracy'],
                old_patient_macro_f1_same_targets=old_f1,
                new_patient_macro_f1_same_targets=row['patient_macro_f1'],
                old_patient_balanced_accuracy_same_targets=old_ba,
                new_patient_balanced_accuracy_same_targets=row['patient_balanced_accuracy'],
                old_predictions_outside_five=float((~np.isin(oldpred, CLASSES)).mean()),
                original_seven_class_patient_macro_f1=oldresult['score']['macro_f1'],
                original_selected_class_f1=original_selected_f1))
    df = pd.DataFrame(records)
    assert len(df) == 540
    matched = df.groupby(['source', 'target', 'seed', 'family'])
    assert matched.size().eq(10).all() and df.groupby('method').size().eq(54).all()
    assert matched.supervision_sha256.nunique().eq(1).all()
    assert matched.target_ids_sha256.nunique().eq(1).all()
    metrics = ['cell_accuracy', 'cell_macro_f1', 'cell_balanced_accuracy',
               'patient_accuracy', 'patient_macro_f1', 'patient_balanced_accuracy']
    df.to_csv(OUT / 'SCORES_PER_JOB.csv', index=False)
    classes = pd.DataFrame(class_rows)
    classes.to_csv(OUT / 'PER_CLASS_PER_JOB.csv', index=False)
    pd.DataFrame(patients).to_csv(OUT / 'PER_PATIENT_PER_JOB.csv', index=False)
    pd.DataFrame(confusion_rows).to_csv(OUT / 'CONFUSIONS_PER_JOB.csv', index=False)
    summary = df.groupby('method')[metrics].mean().reset_index()
    for metric in metrics:
        summary[metric + '_rank'] = summary[metric].rank(ascending=False, method='min').astype(int)
    summary = summary.sort_values('patient_macro_f1', ascending=False)
    summary.to_csv(OUT / 'FULL_COHORT_SUMMARY.csv', index=False)
    models = df.groupby(['method', 'family'])[metrics].mean().reset_index()
    for metric in metrics:
        models[metric + '_rank'] = models.groupby('family')[metric].rank(ascending=False, method='min').astype(int)
    models.to_csv(OUT / 'BY_MODEL.csv', index=False)
    directions = df.groupby(['method', 'source', 'target'])[metrics + ['source_validation_f1']].mean().reset_index()
    for metric in metrics:
        directions[metric + '_rank'] = directions.groupby(['source', 'target'])[metric].rank(ascending=False, method='min').astype(int)
    directions.to_csv(OUT / 'BY_DIRECTION.csv', index=False)
    classes.groupby(['method', 'weighting', 'display_label'])[['f1', 'recall', 'precision']].mean().to_csv(OUT / 'PER_CLASS_MEANS.csv')
    classes.groupby(['method', 'weighting', 'source', 'target', 'display_label'])[['f1', 'recall', 'precision']].mean().to_csv(OUT / 'PER_CLASS_BY_DIRECTION.csv')
    pd.DataFrame(confusion_rows).groupby(['method', 'weighting', 'source', 'target', 'truth', 'predicted']).conditional_fraction.mean().to_csv(OUT / 'CONFUSION_MEANS.csv')
    baseline = pd.DataFrame(baseline_rows)
    baseline.to_csv(OUT / 'OLD_VS_NEW_HEAD_PER_JOB.csv', index=False)
    comparison_metrics = [c for c in baseline.columns if c.startswith(('old_', 'new_', 'original_'))]
    oldmean = baseline.groupby('method')[comparison_metrics].mean().reset_index()
    oldmean.to_csv(OUT / 'OLD_VS_NEW_HEAD_SUMMARY.csv', index=False)
    oldclass = pd.DataFrame(old_class_rows)
    oldclass.to_csv(OUT / 'OLD_HEAD_SAME_TARGETS_PER_CLASS.csv', index=False)
    oldclass.groupby(['method', 'celltype'])[['f1', 'recall', 'precision']].mean().to_csv(OUT / 'OLD_HEAD_SAME_TARGETS_CLASS_MEANS.csv')
    old_audit = pd.read_csv(AUDIT / 'PER_BENCHMARK_SCORES.csv')
    old_audit = old_audit[old_audit.task.eq(TASK) & old_audit.rule.eq('support_ge10') & old_audit.metric.eq('macro_f1')].set_index('method')
    for _, r in oldmean.iterrows():
        assert np.isclose(r.original_selected_class_f1, old_audit.loc[r.method, 'score'], atol=1e-12)
    save(OUT / 'CHECKS.json', dict(jobs=540, ten_methods_same_full_U=True,
        paired_supervision_and_target_ids=True, original_source_splits_reproduced=True,
        patient_disjoint_source_validation_jobs=int(df.split_mode.eq('patient_disjoint_validation').sum()),
        tietscher_within_patient_validation_jobs=int(df.split_mode.eq('within_patient_cell_validation').sum()),
        all_five_classes_in_every_fit_and_validation=True,
        predictions_verified_and_rescored=True, original_files_unchanged=True,
        old_selected_class_audit_reproduced=True, script_sha256=file_sha256(__file__)))
    cols = ['method', 'patient_macro_f1', 'patient_macro_f1_rank',
            'patient_balanced_accuracy', 'patient_balanced_accuracy_rank', 'patient_accuracy']
    classmeans = classes[classes.weighting.eq('patient')].groupby(['method', 'display_label'])[['f1', 'recall', 'precision']].mean().reset_index()
    report = f'''# TNBC 五类重训结果

540/540 作业完成并独立重算评分。十种方法使用相同的原冻结对齐池（62,701 个细胞、11 个共同 marker），仅将监督和评价范围限定为 CD4 T、CD8 T、Myeloid、Stroma、Tumor / epithelial。

## 总体结果

{table(summary, cols)}

主指标来自患者等权混淆矩阵；不是逐患者 F1 的简单平均。每方法 6 个方向 × 3 个模型 × 3 个 seed，共 54 个任务，等权平均。源端 500 训练细胞、512 验证细胞，沿用候选网格和源验证 macro-F1 模型选择；所有方法完全配对。

## RAMP 方向、模型与逐类结果

{table(directions[directions.method.eq('ramp')], ['source', 'target', 'patient_macro_f1', 'patient_macro_f1_rank', 'patient_balanced_accuracy', 'patient_accuracy', 'source_validation_f1'])}

{table(models[models.method.eq('ramp')], ['family', 'patient_macro_f1', 'patient_macro_f1_rank', 'patient_balanced_accuracy', 'patient_accuracy'])}

{table(classmeans[classmeans.method.eq('ramp')], ['display_label', 'f1', 'recall', 'precision'])}

## 同一批五类目标细胞：旧七类分类器 vs 新五类分类器

{table(oldmean, ['method', 'old_patient_macro_f1_same_targets', 'new_patient_macro_f1_same_targets', 'old_patient_balanced_accuracy_same_targets', 'new_patient_balanced_accuracy_same_targets', 'old_predictions_outside_five'])}

旧分类器预测到 B lineage 或 Endothelial 的错误仍计入，不按预测标签过滤细胞。此比较同时改变训练名额分配和输出类别，不能单独归因为 RAMP 对齐质量改变。逐类配对见 OLD_HEAD_SAME_TARGETS_CLASS_MEANS.csv 和 PER_CLASS_MEANS.csv。

## 必须区分的三个旧分数口径

- 原七类 RAMP 患者等权 F1 = 0.487940：原完整七类目标细胞和七类宏平均。
- 原混淆矩阵仅平均五类 F1 = 0.582819：没有移除被排除真实类别产生的假阳性，也没有重训。
- 上表的 old_patient_macro_f1_same_targets：只保留五类真实目标细胞，仍使用旧七类分类器；这是与新分类器目标细胞相同的配对基线。

## 设计边界

- 原五位 Meyer、五位 Shiao、一位 Tietscher 患者全部保留；没有选择高分 donor。
- 原 source 划分先按七类重建，再筛选五类。Meyer/Shiao 共 360 个任务使用患者隔离源验证；Tietscher 共 180 个任务保持原患者内不重叠细胞验证。不能把后者描述为独立患者验证。
- B lineage 和 Endothelial 在原训练最少只有 4 和 1 个细胞；此次范围依据此前统一支持审查，不是在本次结果出来后删除低分类别。共享 CD20/CD31 存在，不声称这些类别本质上不可比。
- Stroma 仍保留。Myeloid 是广义髓系层级，跨平台 macrophage/monocyte/neutrophil 构成不同。标记区分度和旧错误诊断见 EXPECTED_MARKER_AUC.csv、OLD_CONFUSION_BY_DIRECTION.csv。
- 没有重新拟合对齐，没有修改 RAMP 算法、共享面板、原始标签或原冻结数据。这是五类范围的探索性重训，不代表所有细胞的注释性能。
- 原完整七类结果保留；原 Figure 3、稿件、J2、D/E/F 均未修改。

## 验证

CHECKS.json 记录 540 个任务预测重评分、监督/目标身份配对、原拆分复现、哈希和旧支持子集 F1 重现。MANIFEST.json 记录输入、原文件和训练方案；DESIGN_ZH.md 在训练前写入并锁定哈希。
'''
    (OUT / 'RESULTS_ZH.md').write_text(report)
    plot(df, summary)
    print(summary[cols].to_string(index=False))
    print(directions[directions.method.eq('ramp')][['source', 'target', 'patient_macro_f1', 'patient_balanced_accuracy']].to_string(index=False))


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
    for ax, metric, title in zip(axes, ['patient_macro_f1', 'patient_balanced_accuracy'], ['Macro-F1', 'Balanced accuracy']):
        for i, method in enumerate(order):
            vals = df[df.method.eq(method)].sort_values(['source', 'family', 'seed'])[metric].to_numpy()
            color = '#087e8b' if method == 'ramp' else '#738194'
            ax.scatter(vals, i + np.linspace(-.16, .16, len(vals)), s=14, alpha=.6, color=color)
            ax.scatter(vals.mean(), i, s=60, marker='D', edgecolor='white', linewidth=.8, color=color, zorder=4)
        ax.set(xlim=(0, 1), xlabel=title, title=title, yticks=range(len(order)), yticklabels=[names[m] for m in order])
        if not ax.yaxis_inverted(): ax.invert_yaxis()
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='x', color='#e5e5e5', linewidth=.6)
        ax.set_axisbelow(True)
    fig.suptitle('TNBC: five-class annotation on frozen alignment', fontweight='bold')
    fig.text(.5, .015, '54 model/seed/direction evaluations per method; diamonds show means', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .95))
    fig.savefig(OUT / 'METHOD_COMPARISON.png', dpi=220)
    fig.savefig(OUT / 'METHOD_COMPARISON.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
