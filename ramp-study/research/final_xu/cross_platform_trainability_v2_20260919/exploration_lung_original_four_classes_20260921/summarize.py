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
    rebuilt_plans = {}
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
            split_key = (j['source'], j['target'], j['seed'])
            if split_key not in rebuilt_plans:
                rebuilt_plans[split_key] = fixed_plan(full, old_jobs[job_key(j)])
            rebuilt, parent_identities, _ = rebuilt_plans[split_key]
            assert parent_identities == plan['parent_source_identities']
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
            # Old predictions outside the four target classes remain errors.
            # Include all six output labels in the CM, then average only the
            # four target-class F1s; never silently discard outside predictions.
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
                old_predictions_outside_four=float((~np.isin(oldpred, CLASSES)).mean()),
                original_six_class_patient_macro_f1=oldresult['score']['macro_f1'],
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
    assert df.split_mode.value_counts().to_dict() == {'patient_disjoint_validation': 360, 'unit_disjoint_descriptive_validation': 180}
    save(OUT / 'CHECKS.json', dict(jobs=540, ten_methods_same_full_U=True,
        paired_supervision_and_target_ids=True, original_source_splits_reproduced=True,
        patient_disjoint_source_validation_jobs=int(df.split_mode.eq('patient_disjoint_validation').sum()),
        unit_disjoint_source_validation_jobs=int(df.split_mode.eq('unit_disjoint_descriptive_validation').sum()),
        all_source_identity_splits_disjoint=True,
        all_jobs_patient_disjoint=bool(df.split_mode.eq('patient_disjoint_validation').all()),
        all_four_classes_in_every_fit_and_validation=True,
        predictions_verified_and_rescored=True, original_files_unchanged=True,
        old_selected_class_audit_reproduced=True, script_sha256=file_sha256(__file__)))
    cols = ['method', 'patient_macro_f1', 'patient_macro_f1_rank',
            'patient_balanced_accuracy', 'patient_balanced_accuracy_rank', 'patient_accuracy']
    classmeans = classes[classes.weighting.eq('patient')].groupby(['method', 'display_label'])[['f1', 'recall', 'precision']].mean().reset_index()
    report = f'''# 原 Lung benchmark：四类重训

540/540 任务完成并从冻结预测独立重算。Senosain CyTOF、Cords IMC、Monkman CODEX 各 20,000 个细胞，共 60,000 个、8 个共同通道全部保留。只将监督和评价从六类限定为 CD8 T、Endothelial、Myeloid、Tumor/Epithelial，不重新对齐。

## 十方法结果

{table(summary, cols)}

主指标为患者/注册单位等权混淆矩阵指标，不是逐患者 F1 的简单平均。每方法六方向 × 三模型 × 三 seed，共 54 个结果等权汇总。源训练 500、验证 512，所有方法配对使用同一批细胞及候选网格，源验证 macro-F1 选择模型。Monkman 是 descriptive unit，表格延续原代码 patient_* 字段名，不代表其患者映射已被验证。

## RAMP 方向、模型及逐类指标

{table(directions[directions.method.eq('ramp')], ['source', 'target', 'patient_macro_f1', 'patient_macro_f1_rank', 'patient_balanced_accuracy', 'patient_accuracy', 'source_validation_f1'])}

{table(models[models.method.eq('ramp')], ['family', 'patient_macro_f1', 'patient_macro_f1_rank', 'patient_balanced_accuracy', 'patient_accuracy'])}

{table(classmeans[classmeans.method.eq('ramp')], ['display_label', 'f1', 'recall', 'precision'])}

## 旧六类分类器与新四类分类器：完全相同四类目标细胞

{table(oldmean, ['method', 'old_patient_macro_f1_same_targets', 'new_patient_macro_f1_same_targets', 'old_patient_balanced_accuracy_same_targets', 'new_patient_balanced_accuracy_same_targets', 'old_predictions_outside_four'])}

旧模型预测为 B-like 或 CD4 T 仍计错误；没有按预测结果筛选目标细胞。此次同时改变源训练名额分配和输出类别，分数变化不能全部归因于对齐改善。

## 原评分口径

- 原完整六类 RAMP F1 约 0.5670、BA 约 0.6097。
- 在原完整混淆矩阵中，仅平均本四类 F1 约 0.7007、BA 约 0.7245，仍保留真实 B-like/CD4 T 细胞带来的假阳性；这不等于只对四类真实目标细胞重算的指标。
- 上表 old_*_same_targets 才是和新分类器目标完全一致的配对结果。

## 解释范围

- B-like 在 Senosain 为 provisional，缺少 CD19/CD20 直接支持。CD4 是共享类别且有对应通道，本次移出评分的依据是原 Cords 500-cell 抽样只有 5–10 个，而不是生物学不可比。
- 原六类先拆分源患者/unit，再筛选四类；360 个任务患者隔离、180 个 Monkman 源任务 unit 隔离。没有换 donor、seed 或按目标表现调参。
- Myeloid 保留 C02，与已采用 Lung story 的主分析一致；其标签不确定性仍在。本次没有删除对齐细胞或更改注释。
- 原 Senosain 20k 的宽类别、8 通道表达与当前 140k 对应部分完全一致，不需要换细胞池才能进行此次重训。
- 这不是 Senosain/Cords/Zheng 的 HLA 生物学 story。本次未修改 Figure 3、manuscript 或生物学 story。
- 这是查看旧结果后的探索性类别范围调整；完整六类结果保留，不改称预注册。J2、学习曲线、容量曲线、多来源任务未重跑。

## 核验与来源

CHECKS.json 核验 540 个任务、十方法配对监督/目标身份、原划分复现、冻结预测与旧评分重现、原文件哈希。DESIGN_ZH.md 在训练前写入并绑定哈希；MANIFEST.json、plans/、jobs/ 保存来源、监督方案和冻结预测。

原类别与 marker 审查位于 ../reports/lung_original_scope_audit_20260921/。
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
    fig.suptitle('Lung benchmark (3 platforms): four-class annotation', fontweight='bold')
    fig.text(.5, .015, '54 model/seed/direction evaluations per method; diamonds show means', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .95))
    fig.savefig(OUT / 'METHOD_COMPARISON.png', dpi=220)
    fig.savefig(OUT / 'METHOD_COMPARISON.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
