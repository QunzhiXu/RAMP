#!/usr/bin/env python3
"""Recalculate AUROC for all seven frozen program predictions."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE=Path(__file__).resolve().parent
OUT=HERE.parent/'output'
inputs=[OUT/'all_program_predictions.csv.gz',OUT/'program_summary.csv']
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
d=pd.read_csv(inputs[0],dtype={'patient':str})
s=pd.read_csv(inputs[1])
assert len(d)==38299 and d.patient.nunique()==74 and len(s)==7
roi=d[d.unique_region.eq('522_S2reg054')]
crop=roi[roi.x.between(1205,1805,inclusive='left')&roi.y.between(205,670,inclusive='left')]
assert len(roi)==650 and len(crop)==61
names={'pair':'原始 CD39/TIGIT','protein3':'仅三个 protein','rna3':'仅三个 RNA',
       'pair_protein3':'Pair + 三个 protein','pair_rna3':'Pair + 三个 RNA',
       'protein3_rna3':'三个 protein + 三个 RNA','pair_protein3_rna3':'全部八项'}
rows=[]
for r in s.itertuples():
    pred=r.program+'_predicted'
    values=np.array([roc_auc_score(b.spatial_y>0,b[pred]) for _,b in d.groupby('patient')])
    np.testing.assert_allclose(values.mean(),r.niche_auroc_mean,atol=1e-14,rtol=0)
    rows.append(dict(program=r.program,label=names[r.program],patients=len(values),
        mean_patient_auroc=float(values.mean()),median_patient_auroc=float(np.median(values)),
        mean_ci_low=r.niche_auroc_ci_low,mean_ci_high=r.niche_auroc_ci_high,
        delta_vs_pair=r.niche_auroc_delta_vs_pair,
        delta_ci_low=r.niche_auroc_delta_ci_low,delta_ci_high=r.niche_auroc_delta_ci_high,
        full_roi_auroc=float(roc_auc_score(roi.spatial_y>0,roi[pred])),
        displayed_crop_auroc=float(roc_auc_score(crop.spatial_y>0,crop[pred])),
        pooled_cell_auroc=float(roc_auc_score(d.spatial_y>0,d[pred]))))
result=pd.DataFrame(rows)
result.to_csv(HERE/'seven_program_auroc.csv',index=False)
lines=['# 七种 program 的 AUROC 专项比较','',
    '**结论：全队列最高为 Pair + 三个 RNA，AUROC 0.5306；原 Pair 为 0.5300，差值仅 +0.0006，没有明确改善。**','',
    '使用原来七个 program 的连续回投分数，observed niche 固定为 spatial_y >0；74 位患者、38,299 个 CODEX PT-CD8；对齐、k=30、三位来源 donor 聚合均相同。没有重新训练或更改阈值。主要指标为逐患者 AUROC 的等权平均。已直接从细胞级预测重算 74×7 个 AUROC，与此前汇总逐项一致。','',
    '| Program | 平均患者 AUROC | 完整展示 ROI（650 CD8） | 当前裁切（61 CD8） |','|---|---:|---:|---:|']
for r in result.itertuples():
    lines.append(f'| {r.label} | {r.mean_patient_auroc:.4f} | {r.full_roi_auroc:.4f} | {r.displayed_crop_auroc:.4f} |')
best=result.sort_values('mean_patient_auroc',ascending=False).iloc[0]
lines += ['',f'最高版本相对原 Pair 的差值为 {best.delta_vs_pair:+.6f}；配对患者 bootstrap 95% 区间 {best.delta_ci_low:+.6f} 至 {best.delta_ci_high:+.6f}，跨过 0。这是固定程序比较下的描述性区间，未作为新候选选择后的独立验证。','',
    'RNA-only 在当前完整 ROI 的 AUROC 从原 Pair 的 0.648 提高到 0.675，但全队列平均没有提高。当前裁切中，原 Pair 的 0.724 仍高于所有扩展组合。不能据一个 ROI 的改善替换全队列结论。','',
    '这里七个组合沿用等特征权重与原符号方向：LAG-3↑、CD127↓、CD62L↓；CTLA4↑、CXCL13↑、IL7R↓。详细构建方法和完整结果见上一级 PLAN_ZH.md 和 RESULTS_ZH.md。','',
    '结论仍是增加这几项没有解决整体逐细胞空间区分能力较弱的问题。正式图、预测文件和 manuscript 均保持不变。']
(HERE/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n')
assert hashes=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
(HERE/'AUDIT.json').write_text(json.dumps(dict(status='PASS',input_sha256=hashes,inputs_unchanged=True,
    patient_auroc_recalculations=518,all_original_auroc_means_reproduced=True,
    label='spatial_y >0',primary='equal-patient mean AUROC',models_refit=False),indent=2)+'\n')
print(result[['label','mean_patient_auroc','displayed_crop_auroc']].to_string(index=False))
