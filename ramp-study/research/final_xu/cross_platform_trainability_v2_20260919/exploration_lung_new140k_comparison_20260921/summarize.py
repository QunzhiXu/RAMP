"""Audit and summarize ten-method J1 comparisons; all target errors retained."""
from run import OUT, PILOT, METHODS, CLASSES, ARMS, save, verify_inputs, task_for, file_sha256, digest
import json
import numpy as np
import pandas as pd

METRICS=['patient_macro_f1','patient_balanced_accuracy','patient_accuracy',
         'cell_macro_f1','cell_balanced_accuracy','cell_accuracy']


def metrics(cm):
    tp=np.diag(cm);den=cm.sum(0)+cm.sum(1);support=cm.sum(1)
    f1=np.divide(2*tp,den,out=np.zeros(4),where=den>0)
    recall=np.divide(tp,support,out=np.zeros(4),where=support>0)
    precision=np.divide(tp,cm.sum(0),out=np.zeros(4),where=cm.sum(0)>0)
    return dict(macro_f1=float(f1.mean()),balanced_accuracy=float(recall.mean()),
                accuracy=float(tp.sum()/cm.sum())),f1,recall,precision


def target_template(md,ids):
    sub=md.loc[list(ids)]
    y=pd.Categorical(sub.celltype,categories=CLASSES).codes
    assert (y>=0).all()
    counts=sub.patient_id.value_counts()
    weights=1/sub.patient_id.map(counts).to_numpy(float)/len(counts)
    assert np.isclose(weights.sum(),1)
    return dict(y=y,weights=weights,patients=len(counts),cells=len(ids))


def score(template,pred):
    p=pd.Categorical(pred,categories=CLASSES).codes
    assert (p>=0).all() and len(p)==template['cells']
    ix=4*template['y']+p
    cms={'patient':np.bincount(ix,weights=template['weights'],minlength=16).reshape(4,4),
         'cell':np.bincount(ix,minlength=16).reshape(4,4).astype(float)}
    scores={};details={}
    for weighting,cm in cms.items():
        m,f,r,p=metrics(cm)
        scores.update({weighting+'_'+k:v for k,v in m.items()})
        details[weighting]=(cm,f,r,p)
    return scores,details


def table(df,cols):
    lines=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
    for _,r in df.iterrows():
        lines.append('| '+' | '.join(f'{r[c]:.4f}' if isinstance(r[c],(float,np.floating)) else str(r[c]) for c in cols)+' |')
    return '\n'.join(lines)


def main():
    m=json.loads((OUT/'MANIFEST.json').read_text())
    assert json.loads((OUT/'RUN_COMPLETE.json').read_text())['all_passed']
    verify_inputs(m)
    md={a:task_for(a).metadata.set_index('cell_id') for a in ARMS}
    pilot_hash=file_sha256(PILOT/'MANIFEST.json');our_hash=file_sha256(OUT/'MANIFEST.json')
    templates={};target_ids={}
    for j in m['jobs']:
        key=j['arm'],j['target']
        plan=json.loads((PILOT/'plans'/f"{j['id']}.json").read_text())
        if key not in templates:
            target_ids[key]=plan['target_ids']
            templates[key]=target_template(md[j['arm']],plan['target_ids'])
        else:assert target_ids[key]==plan['target_ids']
    records=[];classes=[];confusions=[];paired=[];unaffected=0
    for method in ['ramp']+METHODS:
        cached={}
        for j in m['jobs']:
            arm=j['arm'];key=arm,j['target']
            dest=(PILOT/'jobs'/arm/j['id']) if method=='ramp' else (OUT/'jobs'/method/arm/j['id'])
            proof=json.loads((dest/'EXPERIMENT_COMPLETE.json').read_text())
            assert proof['manifest_sha256']==(pilot_hash if method=='ramp' else our_hash)
            for path,h in proof['files'].items():assert file_sha256(path)==h,path
            plan=json.loads((PILOT/'plans'/f"{j['id']}.json").read_text())
            ledger=json.loads((dest/'ledger.json').read_text())
            result=json.loads((dest/'RESULT.json').read_text())
            assert ledger['fit']==plan['fit'] and ledger['validation']==plan['validation']
            assert ledger['candidates']==j['candidates'] and ledger['pool_sha256']==m['pool_sha256']
            assert not ledger['formal_benchmark']
            assert not set(md[arm].loc[ledger['fit'],'patient_id']) & set(md[arm].loc[ledger['validation'],'patient_id'])
            assert result['fit_cells']==500 and result['validation_cells']==512
            with np.load(dest/'predictions.npz',allow_pickle=False) as z:
                ids,pred=z['cell_ids'],z['predictions']
            assert ids.tolist()==plan['target_ids']==ledger['target_cell_ids']==target_ids[key]
            assert file_sha256(dest/'predictions.npz')==result['prediction_receipt']['sha256']
            sc,details=score(templates[key],pred)
            for metric in ['macro_f1','balanced_accuracy','accuracy']:
                assert np.isclose(sc['patient_'+metric],result['score'][metric],atol=1e-12)
            assert np.allclose(details['patient'][0],result['score']['confusion_matrix'],atol=1e-12)
            rec=dict(method=method,arm=arm,source=j['source'],target=j['target'],seed=j['seed'],family=j['family'])
            records.append(dict(**rec,**sc,cells=len(ids),patients=templates[key]['patients'],
                supervision_sha256=j['supervision_sha256'],target_ids_sha256=digest(ids.tolist())))
            for weighting,(cm,f,r,p) in details.items():
                for k,ct in enumerate(CLASSES):
                    classes.append(dict(**rec,weighting=weighting,celltype=ct,f1=f[k],recall=r[k],precision=p[k]))
                    for q,pc in enumerate(CLASSES):
                        confusions.append(dict(**rec,weighting=weighting,truth=ct,predicted=pc,mass=cm[k,q]))
            cached[(arm,j['source'],j['target'],j['seed'],j['family'])]=(ids,pred,ledger)
        for key,(ids,pred,ledger) in cached.items():
            arm,source,target,seed,family=key
            if arm!='exclude_c02_supervision':continue
            ki,kp,kl=cached[('keep_c02',source,target,seed,family)]
            lookup=dict(zip(ki,kp));assert set(ids)<=set(ki)
            keep_pred=np.asarray([lookup[i] for i in ids])
            if source!='Senosain CyTOF':
                assert ledger['fit']==kl['fit'] and ledger['validation']==kl['validation']
                assert np.array_equal(pred,keep_pred)
            if source!='Senosain CyTOF' and target!='Senosain CyTOF':
                assert np.array_equal(ids,ki);unaffected+=1
            for training,prediction in [('keep_c02',keep_pred),('exclude_c02_supervision',pred)]:
                sc,_=score(templates[('exclude_c02_supervision',target)],prediction)
                paired.append(dict(method=method,training_arm=training,source=source,target=target,
                    seed=seed,family=family,**sc,target_ids_sha256=digest(ids.tolist())))
        print('Audited',method,'108/108',flush=True)
    df=pd.DataFrame(records);assert len(df)==1080
    groups=df.groupby(['arm','source','target','seed','family'])
    assert groups.size().eq(10).all()
    assert groups.supervision_sha256.nunique().eq(1).all()
    assert groups.target_ids_sha256.nunique().eq(1).all()
    assert unaffected==180
    df.to_csv(OUT/'SCORES_PER_JOB.csv',index=False)
    summary=df.groupby(['arm','method'])[METRICS].mean().reset_index()
    for metric in METRICS:
        summary[metric+'_rank']=summary.groupby('arm')[metric].rank(ascending=False,method='min').astype(int)
    summary=summary.sort_values(['arm','patient_macro_f1'],ascending=[True,False])
    summary.to_csv(OUT/'SUMMARY.csv',index=False)
    primary=summary[summary.arm.eq('keep_c02')]
    primary.to_csv(OUT/'PRIMARY_SUMMARY.csv',index=False)
    sensitivity=summary[summary.arm.eq('exclude_c02_supervision')]
    sensitivity.to_csv(OUT/'SENSITIVITY_SUMMARY.csv',index=False)
    models=df.groupby(['arm','method','family'])[METRICS].mean().reset_index()
    models.to_csv(OUT/'BY_MODEL.csv',index=False)
    directions=df.groupby(['arm','method','source','target'])[METRICS].mean().reset_index()
    directions.to_csv(OUT/'BY_DIRECTION.csv',index=False)
    cr=pd.DataFrame(classes);cr.to_csv(OUT/'PER_CLASS_PER_JOB.csv',index=False)
    cr.groupby(['arm','method','weighting','celltype'])[['f1','recall','precision']].mean().to_csv(OUT/'PER_CLASS_MEANS.csv')
    cr.groupby(['arm','method','weighting','source','target','celltype'])[['f1','recall','precision']].mean().to_csv(OUT/'PER_CLASS_BY_DIRECTION.csv')
    pd.DataFrame(confusions).to_csv(OUT/'CONFUSIONS_PER_JOB.csv',index=False)
    ps=pd.DataFrame(paired);ps.to_csv(OUT/'SAME_TARGET_SCORES_PER_JOB.csv',index=False)
    ps.groupby(['method','training_arm'])[METRICS].mean().to_csv(OUT/'SAME_TARGET_SUMMARY.csv')
    old=pd.read_csv(PILOT/'SUMMARY.csv').set_index('arm')
    actual=summary[summary.method.eq('ramp')].set_index('arm')
    assert np.allclose(old.loc[ARMS,METRICS],actual.loc[ARMS,METRICS],atol=1e-12)
    verify_inputs(m)
    save(OUT/'CHECKS.json',dict(jobs=1080,new_jobs=972,methods=10,ramp_pilot_reproduced=True,
        same_training_validation_and_target_ids_across_methods=True,
        all_frozen_predictions_independently_rescored=True,source_patient_disjoint_validation=True,
        all_84000_cells_retained_in_all_representations=True,original_files_unchanged=True,
        cords_zheng_identical_arm_pairs=180,maxfuse_shared9_protocol=True,
        no_formal_figure_or_manuscript_changes=True))
    cols=['method','patient_macro_f1','patient_macro_f1_rank','patient_balanced_accuracy',
          'patient_balanced_accuracy_rank','patient_accuracy','patient_accuracy_rank']
    report=f'''# 新版 Lung 四分类：十方法比较完成

共 1,080/1,080 作业：复用 108 个冻结 RAMP 作业，新运行 972 个比较作业。每方法均为三模型、三个种子、六个有向迁移和两种 C02 处理。输入为三个平台各 28 位患者、各 28,000 个细胞、九个共同 marker。

## 主分析：保留 C02

{table(primary,cols)}

## 敏感性：仅在分类监督及目标评分排除 C02

{table(sensitivity,cols)}

指标均为每个方向的患者等权混淆矩阵评分，然后对三模型、三种子、六方向等权平均；不将种子作为独立生物学重复。Accuracy、balanced accuracy、macro-F1 分别排序，不临时添加总体指标。细胞等权结果保留在 CSV 中。

## 分模型：主分析

{table(models[models.arm.eq('keep_c02')],['method','family']+METRICS[:3])}

## RAMP 分方向：主分析

{table(directions[directions.arm.eq('keep_c02') & directions.method.eq('ramp')],['source','target']+METRICS[:3])}

## 运行与核验

- 所有方法逐作业具有相同的 500 个训练、512 个验证细胞以及目标细胞 IDs；源训练/验证患者不重叠。候选网格、选择标准、seed 完全沿用前版。
- 对齐始终使用全部 84,000 个细胞，不因 C02 监督处理而重新拟合或删行。
- Harmony Direct、cyCombine、CytoNorm、fastMNN 经过原始输入、预处理、单位、marker、row ID、算法 helper 和输出 SHA-256 验证后复用新版 Figure 2 的兼容结果。
- MaxFuse 沿用 Figure 3 原定共同九 marker 配置，重新拟合 shared=active=9 和固定 Senosain hub。Figure 2 的 full-active 蛋白面板结果没有混入。
- `SAME_TARGET_SUMMARY.csv` 在相同非 C02 目标细胞上比较两组；Cords↔Zheng 两方向、所有方法共 180 对作业预测逐细胞相同。
- 两组完整保留；主分析保留 C02 的定位在已经查看 RAMP 试跑后确定，不称预注册。分类性能不独立证明 C02 的细胞身份。
- 本次仅完成新版 Lung 的 J1 分类比较。J2 缺失 marker、标签效率/容量/多来源曲线以及正式 Figure 3、manuscript 尚未更新。

详细设计见 [DESIGN_ZH.md](DESIGN_ZH.md)，逐作业成绩见 `SCORES_PER_JOB.csv`，逐类与逐方向成绩见对应 CSV，核验见 `CHECKS.json`。
'''
    (OUT/'README.md').write_text(report)
    print('\nPRIMARY\n'+primary[cols].to_string(index=False))
    print('\nSENSITIVITY\n'+sensitivity[cols].to_string(index=False))
    print('All checks passed.',flush=True)


if __name__=='__main__':main()
