"""Independently score frozen predictions and separate C02 scoring/training effects."""
from run import (OUT, OLD, ARMS, CLASSES, save, task_for, verify_inputs,
                 file_sha256, digest)
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix
from trainability.scoring import classification_score

METRICS = ['patient_macro_f1', 'patient_balanced_accuracy', 'patient_accuracy',
           'cell_macro_f1', 'cell_balanced_accuracy', 'cell_accuracy']


def score(md, ids, pred):
    target = md.loc[list(ids)]
    y = target.celltype.to_numpy()
    s = classification_score(y, pred, target.patient_id.to_numpy(), CLASSES)
    values = {'patient_' + k: s[k] for k in ['macro_f1', 'balanced_accuracy', 'accuracy']}
    values.update(cell_macro_f1=f1_score(y, pred, labels=CLASSES, average='macro', zero_division=0),
                  cell_accuracy=accuracy_score(y, pred),
                  cell_balanced_accuracy=balanced_accuracy_score(y, pred))
    return values, s


def table(df, columns):
    lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---']*len(columns)) + ' |']
    for _, row in df.iterrows():
        lines.append('| ' + ' | '.join(f'{row[c]:.4f}' if isinstance(row[c], (float, np.floating))
                                      else str(row[c]) for c in columns) + ' |')
    return '\n'.join(lines)


def main():
    manifest = json.loads((OUT / 'MANIFEST.json').read_text())
    assert json.loads((OUT / 'RUN_COMPLETE.json').read_text())['all_passed']
    verify_inputs(manifest)
    metadata = {a: task_for(a).metadata.set_index('cell_id') for a in ARMS}
    records, class_rows, confusions, cached, c02rows = [], [], [], {}, []
    for j in manifest['jobs']:
        arm = j['arm']
        md = metadata[arm]
        dest = OUT / 'jobs' / arm / j['id']
        proof = json.loads((dest / 'EXPERIMENT_COMPLETE.json').read_text())
        assert proof['manifest_sha256'] == file_sha256(OUT / 'MANIFEST.json')
        for p, h in proof['files'].items():
            assert file_sha256(p) == h
        plan = json.loads((OUT / 'plans' / f"{j['id']}.json").read_text())
        ledger = json.loads((dest / 'ledger.json').read_text())
        result = json.loads((dest / 'RESULT.json').read_text())
        assert ledger['candidates'] == j['candidates'] and not ledger['formal_benchmark']
        assert ledger['fit'] == plan['fit'] and ledger['validation'] == plan['validation']
        assert not set(md.loc[plan['fit'], 'patient_id']) & set(md.loc[plan['validation'], 'patient_id'])
        with np.load(dest / 'predictions.npz', allow_pickle=False) as f:
            ids, pred = f['cell_ids'], f['predictions']
        assert ids.tolist() == plan['target_ids'] == ledger['target_cell_ids']
        assert set(pred) <= set(CLASSES)
        assert file_sha256(dest / 'predictions.npz') == result['prediction_receipt']['sha256']
        assert not result['selection'].get('constant_classifier_fallback', False)
        key = dict(arm=arm, source=j['source'], target=j['target'], seed=j['seed'], family=j['family'])
        scores, patient = score(md, ids, pred)
        for name in ['macro_f1', 'balanced_accuracy', 'accuracy']:
            assert np.isclose(patient[name], result['score'][name], atol=1e-12)
        records.append(dict(**key, **scores, cells=len(ids), patients=patient['patients'],
                            target_ids_sha256=digest(ids.tolist()), supervision_sha256=j['supervision_sha256']))
        cms = {'patient': np.asarray(patient['confusion_matrix']),
               'cell': confusion_matrix(md.loc[ids, 'celltype'], pred, labels=CLASSES)}
        for weighting, cm in cms.items():
            tp = np.diag(cm)
            fs = np.divide(2*tp, cm.sum(0)+cm.sum(1), out=np.zeros(4), where=cm.sum(0)+cm.sum(1)>0)
            recall = np.divide(tp, cm.sum(1), out=np.zeros(4), where=cm.sum(1)>0)
            precision = np.divide(tp, cm.sum(0), out=np.zeros(4), where=cm.sum(0)>0)
            for k, ct in enumerate(CLASSES):
                class_rows.append(dict(**key, weighting=weighting, celltype=ct,
                                       f1=fs[k], recall=recall[k], precision=precision[k]))
                for q, pc in enumerate(CLASSES):
                    confusions.append(dict(**key, weighting=weighting, truth=ct, predicted=pc,
                                           mass=float(cm[k,q])))
        cachekey = arm, j['source'], j['target'], j['seed'], j['family']
        cached[cachekey] = (ids, pred, ledger)
        if arm == 'keep_c02' and j['target'] == 'Senosain CyTOF':
            sub = md.loc[ids, 'is_c02'].to_numpy()
            for pc in CLASSES:
                c02rows.append(dict(**key, predicted=pc, cells=int(sub.sum()),
                                    prediction_fraction=float(np.mean(pred[sub] == pc))))
    df = pd.DataFrame(records)
    assert len(df) == 108 and df.groupby('arm').size().eq(54).all()
    df.to_csv(OUT / 'SCORES_PER_JOB.csv', index=False)
    summary = df.groupby('arm', sort=False)[METRICS].mean().reset_index()
    summary.to_csv(OUT / 'SUMMARY.csv', index=False)
    models = df.groupby(['arm', 'family'])[METRICS].mean().reset_index()
    models.to_csv(OUT / 'BY_MODEL.csv', index=False)
    directions = df.groupby(['arm', 'source', 'target'])[METRICS].mean().reset_index()
    directions.to_csv(OUT / 'BY_DIRECTION.csv', index=False)
    cr = pd.DataFrame(class_rows)
    cr.to_csv(OUT / 'PER_CLASS_PER_JOB.csv', index=False)
    cm = cr.groupby(['arm', 'weighting', 'celltype'])[['f1', 'recall', 'precision']].mean().reset_index()
    cm.to_csv(OUT / 'PER_CLASS_MEANS.csv', index=False)
    cr.groupby(['arm', 'weighting', 'source', 'target', 'celltype'])[['f1','recall','precision']].mean().to_csv(
        OUT / 'PER_CLASS_BY_DIRECTION.csv')
    pd.DataFrame(confusions).to_csv(OUT / 'CONFUSIONS_PER_JOB.csv', index=False)
    pd.DataFrame(c02rows).to_csv(OUT / 'C02_PREDICTIONS.csv', index=False)
    paired = []
    unaffected = 0
    for key, (ids, pred, ledger) in cached.items():
        arm, source, target, seed, family = key
        if arm != 'exclude_c02_supervision':
            continue
        keepids, keeppred, keepledger = cached[('keep_c02', source, target, seed, family)]
        lookup = dict(zip(keepids, keeppred))
        assert set(ids) <= set(keepids)
        kp = np.asarray([lookup[i] for i in ids])
        if source != 'Senosain CyTOF':
            assert ledger['fit'] == keepledger['fit'] and ledger['validation'] == keepledger['validation']
            # Prediction heads are identical when Senosain is not the source.
            assert np.array_equal(kp, pred)
        if source != 'Senosain CyTOF' and target != 'Senosain CyTOF':
            assert np.array_equal(ids, keepids) and np.array_equal(kp, pred)
            unaffected += 1
        for training, prediction in [('keep_c02', kp), ('exclude_c02_supervision', pred)]:
            sc, _ = score(metadata['exclude_c02_supervision'], ids, prediction)
            paired.append(dict(training_arm=training, source=source, target=target, seed=seed,
                               family=family, cells=len(ids), target_ids_sha256=digest(ids.tolist()), **sc))
    assert unaffected == 18
    paired = pd.DataFrame(paired)
    assert paired.groupby(['source','target','seed','family']).target_ids_sha256.nunique().eq(1).all()
    paired.to_csv(OUT / 'SAME_TARGET_SCORES_PER_JOB.csv', index=False)
    ps = paired.groupby('training_arm', sort=False)[METRICS].mean().reset_index()
    ps.to_csv(OUT / 'SAME_TARGET_SUMMARY.csv', index=False)
    paired.groupby(['training_arm','source','target'])[METRICS].mean().to_csv(OUT / 'SAME_TARGET_BY_DIRECTION.csv')
    old = pd.read_csv(OLD / 'FULL_COHORT_SUMMARY.csv')
    old = old[old.method.eq('ramp')][METRICS].copy()
    old.insert(0, 'version', 'old_reference_four_classes')
    new = summary.rename(columns={'arm':'version'})
    pd.concat([old, new], ignore_index=True).to_csv(OUT / 'OLD_NEW_DESCRIPTIVE.csv', index=False)
    verify_inputs(manifest)
    save(OUT / 'CHECKS.json', dict(jobs=108, predictions_independently_rescored=True,
        source_patient_disjoint_validation=True, identical_all84000_representation_in_both_arms=True,
        c02_excluded_only_from_supervision_and_scoring=True,
        same_target_control_verified=True, non_senosain_source_predictions_identical=True,
        cords_zheng_identical_jobs=unaffected, source_files_unchanged=True,
        no_formal_figure_or_manuscript_changes=True))
    cols = ['patient_macro_f1', 'patient_balanced_accuracy', 'patient_accuracy']
    report = f'''# 新版 Lung RAMP：四分类与 C02 敏感性结果

108/108 作业完成，预测已冻结并独立重算。三个平台各 28 位患者；六方向 × 三模型 × 三 seed × 两组。全部 84,000 个细胞参与的同一份 RAMP 表达矩阵保持不变。

## 总体结果

{table(summary, ['arm'] + cols)}

主表使用患者等权混淆矩阵计算指标，再对方向、模型、seed 等权平均。不是逐患者 F1 的简单平均。cell 等权结果另存 CSV。

## 同一目标细胞对照

以下两组都在完全相同的非 C02 目标细胞上评分，只比较源监督是否包含 C02；排除组保持相同总训练预算，会重新抽取 Senosain 源监督细胞。

{table(ps, ['training_arm'] + cols)}

当 Senosain 为目标时，两组源训练完全相同，评分差别只来自移除了目标 C02。Senosain 为源时，目标不变，差别来自重新抽取源训练和验证数据。Cords↔Zheng 的 18 对作业预测逐细胞完全一致。

## 分模型

{table(models, ['arm', 'family'] + cols)}

## 分方向

{table(directions, ['arm', 'source', 'target'] + cols)}

## 分类别（患者等权）

{table(cm[cm.weighting.eq('patient')], ['arm', 'celltype', 'f1', 'recall', 'precision'])}

## 旧版参照与解释范围

{table(pd.concat([old, new], ignore_index=True), ['version'] + cols)}

旧版和新版的 Senosain 细胞、注释及对齐拟合输入不同；该对比只描述整个数据更新后的成绩变化，不能解释为算法改进，也不能与旧输入上的其他方法成绩直接排名。

本次只跑 RAMP 的分类任务；没有运行 missing-marker prediction、预算/容量曲线或比较方法，没有更新正式 Figure 3 或 manuscript。C02 的谱系不确定性与 QC 分开处理，不能仅凭删除后分类成绩更高判定其原标签错误。设计和纳入规则见 [DESIGN_ZH.md](DESIGN_ZH.md)。
'''
    (OUT / 'README.md').write_text(report)
    print(summary.to_string(index=False))
    print('\nSAME TARGET\n' + ps.to_string(index=False))
    print('\nAll checks passed.')


if __name__ == '__main__':
    main()
