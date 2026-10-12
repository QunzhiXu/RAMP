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
            # Old predictions outside the three target classes remain errors.
            # Include all five output labels in the CM, then average only the
            # three target-class F1s; never silently discard outside predictions.
            old_patient = classification_score(y, oldpred, pats, all_classes)
            old_f1 = per_class(old_patient['confusion_matrix'])[0][selected_indices].mean()
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
                old_predictions_outside_three=float((~np.isin(oldpred, CLASSES)).mean()),
                original_five_class_patient_macro_f1=oldresult['score']['macro_f1'],
                original_selected_class_f1=original_selected_f1))
    df = pd.DataFrame(records)
    assert len(df) == 180
    matched = df.groupby(['source', 'target', 'seed', 'family'])
    assert matched.size().eq(10).all() and df.groupby('method').size().eq(18).all()
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
    summary = summary.sort_values('cell_accuracy', ascending=False)
    summary.to_csv(OUT / 'FULL_COHORT_SUMMARY.csv', index=False)
    models = df.groupby(['method', 'family'])[metrics].mean().reset_index()
    models.to_csv(OUT / 'BY_MODEL.csv', index=False)
    directions = df.groupby(['method', 'source', 'target'])[metrics].mean().reset_index()
    directions.to_csv(OUT / 'BY_DIRECTION.csv', index=False)
    classes.groupby(['method', 'weighting', 'display_label'])[['f1', 'recall', 'precision']].mean().to_csv(OUT / 'PER_CLASS_MEANS.csv')
    classes.groupby(['method', 'weighting', 'source', 'target', 'display_label'])[['f1', 'recall', 'precision']].mean().to_csv(OUT / 'PER_CLASS_BY_DIRECTION.csv')
    pd.DataFrame(confusion_rows).groupby(['method', 'weighting', 'source', 'target', 'truth', 'predicted']).conditional_fraction.mean().to_csv(OUT / 'CONFUSION_MEANS.csv')
    baseline = pd.DataFrame(baseline_rows)
    baseline.to_csv(OUT / 'OLD_VS_NEW_HEAD_PER_JOB.csv', index=False)
    comparison_metrics = [c for c in baseline.columns if c.startswith(('old_', 'new_', 'original_'))]
    oldmean = baseline.groupby('method')[comparison_metrics].mean().reset_index()
    oldmean.to_csv(OUT / 'OLD_VS_NEW_HEAD_SUMMARY.csv', index=False)
    old_audit = pd.read_csv(AUDIT / 'REPRODUCED_METHOD_SUMMARY.csv').set_index('method')
    for _, r in oldmean.iterrows():
        assert np.isclose(r.original_selected_class_f1, old_audit.loc[r.method, 'selected_class_f1'], atol=1e-12)
    save(OUT / 'CHECKS.json', dict(jobs=180, ten_methods_same_full_U=True,
        paired_supervision_and_target_ids=True, original_source_patient_splits_reproduced=True,
        patient_disjoint_source_validation=True, all_three_classes_in_every_fit_and_validation=True,
        predictions_verified_and_rescored=True, original_files_unchanged=True,
        old_selected_class_audit_reproduced=True, script_sha256=file_sha256(__file__)))
    cols = ['method', 'cell_accuracy', 'cell_macro_f1', 'patient_accuracy', 'patient_macro_f1']
    report = f'''# Melanoma story 三类重训结果

180/180 作业完成并独立重算评分。全部 10 方法使用相同的 30 IMC + 8 CITE-seq 患者冻结表示，监督与评价限定于 B lineage、CD4 T（含 Treg）、CD8 T。每次使用 500 源训练细胞、512 源验证细胞。

## 总体结果

{table(summary, cols)}

每个方向先计算目标细胞等权指标（cell）或患者等权混淆矩阵指标（patient），再对两个方向、三个模型和三个 seed 等权平均。表中的患者指标不是逐患者 F1 的简单平均。源模型选择沿用患者等权 macro-F1；没有根据目标结果改选候选。随机种子不代表独立生物学重复。

## RAMP 方向及模型

{table(directions[directions.method.eq('ramp')], ['source', 'target'] + metrics)}

{table(models[models.method.eq('ramp')], ['family'] + metrics)}

## 旧五分类头与新三分类头：完全相同的三类目标细胞

{table(oldmean, ['method', 'old_cell_accuracy_same_targets', 'new_cell_accuracy_same_targets', 'old_cell_macro_f1_same_targets', 'new_cell_macro_f1_same_targets'])}

旧头预测为 Myeloid/Endothelial 的错误全部保留；只按真实类别限制评价细胞，不按预测标签筛除细胞。该比较同时改变训练预算的类别分配与输出类别数，不能解释为对齐本身改善。

此前表中的 RAMP **0.327 → 0.490** 是旧五类任务患者等权 F1 → 仅平均旧混淆矩阵中三类 F1。0.490 并未重训，仍受被排除真实类别造成的假阳性影响。本次新三分类是不同任务范围，不应把两者当成完全相同指标的干预前后结果；`OLD_VS_NEW_HEAD_SUMMARY.csv` 另列完全相同目标细胞上的配对比较。

## 解释范围与保留问题

- 这是审查旧结果后开展的淋巴细胞限定探索；并非所有细胞注释性能。完整原结果保留。
- CD4 展示名称来自原作者更细分标签核对；内部仍沿用原类别编码。B lineage 包括 CITE-seq 的 234 个 Plasma cells，未更改其集合。
- 未纳入 Endothelial：CITE 仅 41 个，原 500 训练预算中两个 seed 为零。未纳入 Myeloid：CITE 主要为 DC/pDC，而 IMC 为 macrophage/monocyte，原训练支持也低。排除理由由此前独立审查记录。
- source 患者拆分先按原五类重建，再在每一侧筛选三类；避免减少类别后随机患者划分漂移。每次训练与验证均含三类，所有方法完全配对。
- 15 个 shared marker、165,686 个对齐细胞及全部对齐表示未改动。CD3/CD4/CD8/CD20 仍是输入，这是注释任务而非缺失 marker 预测。
- IMC 为肿瘤组织，CITE 为淋巴结；两队列生物学构成不同。IMC 的 98.18% B lineage 集中在三位患者，限制跨患者泛化。
- 原 RAMP 中 11/30 IMC donor 的表示完全等于 KMeans-low baseline，此次未修复；结果对应当前冻结表示。
- 原数据、算法、正式图和稿件未改动。`CHECKS.json` 核验完整性、配对关系、预测哈希与评分；`MANIFEST.json` 记录输入及实验哈希。
'''
    (OUT / 'RESULTS_ZH.md').write_text(report)
    plot(df, summary)
    print(summary[cols].to_string(index=False))
    print(directions[directions.method.eq('ramp')][['source', 'target'] + metrics].to_string(index=False))


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
            vals = df[df.method.eq(method)].sort_values(['source', 'family', 'seed'])[metric].to_numpy()
            color = '#087e8b' if method == 'ramp' else '#738194'
            ax.scatter(vals, i + np.linspace(-.16, .16, len(vals)), s=14, alpha=.6, color=color)
            ax.scatter(vals.mean(), i, s=60, marker='D', edgecolor='white', linewidth=.8, color=color, zorder=4)
        ax.set(xlim=(0, 1), xlabel=title, title=title, yticks=range(len(order)), yticklabels=[names[m] for m in order])
        if not ax.yaxis_inverted(): ax.invert_yaxis()
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='x', color='#e5e5e5', linewidth=.6)
        ax.set_axisbelow(True)
    fig.suptitle('Melanoma story: B lineage / CD4 / CD8 annotation', fontweight='bold')
    fig.text(.5, .015, '18 model/seed/direction evaluations per method; diamonds show means', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .95))
    fig.savefig(OUT / 'METHOD_COMPARISON.png', dpi=220)
    fig.savefig(OUT / 'METHOD_COMPARISON.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
