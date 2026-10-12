#!/usr/bin/env python3
"""Patient-balanced spatial enrichment of the frozen CD39/TIGIT Pair."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCE=ROOT/'spatial_return_survival/output/haist74_pt_cd8_tigit_cd39_return.csv.gz'


def main():
    before=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    d=pd.read_csv(SOURCE,dtype={'patient':str})
    assert len(d)==38299 and d.patient.nunique()==74
    rows=[]
    for contrast,hi_cut,lo_cut in [('top40_vs_other',.6,None),('Q5_vs_Q1',.8,.2)]:
        for patient,b in d.groupby('patient'):
            high=b.within_patient_rank>=hi_cut
            low=~high if lo_cut is None else b.within_patient_rank<lo_cut
            truth=b.spatial_y>0
            assert high.any() and low.any() and not (high&low).any()
            rows.append(dict(contrast=contrast,patient=patient,high_cells=int(high.sum()),low_cells=int(low.sum()),
                high_niche_cells=int(truth[high].sum()),low_niche_cells=int(truth[low].sum()),
                high_fraction=float(truth[high].mean()),low_fraction=float(truth[low].mean())))
    per_patient=pd.DataFrame(rows)
    per_patient.to_csv(HERE/'patient_enrichment.csv',index=False)
    summary=[]
    for contrast,b in per_patient.groupby('contrast',sort=False):
        v=b[['low_fraction','high_fraction']].to_numpy(float)
        low,high=v.mean(axis=0)
        delta=v[:,1]-v[:,0]
        ix=np.random.default_rng(20260913).integers(0,len(v),(10000,len(v)))
        boot=v[ix].mean(axis=1)
        rr=high/low; odds_ratio=(high/(1-high))/(low/(1-low))
        statistics={'difference':(high-low,boot[:,1]-boot[:,0]),
                    'risk_ratio':(rr,boot[:,1]/boot[:,0]),
                    'odds_ratio':(odds_ratio,(boot[:,1]/(1-boot[:,1]))/(boot[:,0]/(1-boot[:,0])))}
        r=dict(contrast=contrast,patients=len(b),low_fraction=low,high_fraction=high,
               positive_patients=int((delta>0).sum()),zero_difference_patients=int((delta==0).sum()),
               negative_patients=int((delta<0).sum()),wilcoxon_two_sided_p=float(wilcoxon(delta,alternative='two-sided').pvalue))
        for metric,(value,samples) in statistics.items():
            lo,hi=np.quantile(samples,[.025,.975])
            r.update({metric:float(value),metric+'_ci_low':float(lo),metric+'_ci_high':float(hi)})
        summary.append(r)
    summary=pd.DataFrame(summary)
    summary.to_csv(HERE/'enrichment_summary.csv',index=False)
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==before
    audit=dict(status='PASS',source=str(SOURCE),source_sha256=before,inputs_unchanged=True,
        comparisons=['Original patient-specific top 40% vs remaining 60%',
                     'Original highest quintile Q5 vs lowest quintile Q1'],
        label='Observed niche = spatial_y > 0',unit='74 patients, equal weight',
        effect_estimands='Risk difference, risk ratio and odds ratio computed from patient-equal group mean probabilities',
        ratio_note='Marginal patient-standardized ratios, not arithmetic averages of patient-specific ratios',
        bootstrap='10000 paired patient resamples; seed 20260913',
        test='Two-sided paired Wilcoxon, zero differences excluded by scipy default',
        no_new_threshold_selection=True,no_predictions_changed=True,
        classification_auroc_patient_mean=0.5300022635693257,
        interpretation='Spatial enrichment, not improved cell-wise niche prediction accuracy')
    (HERE/'AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    lines=['# 原始 Pair 的空间富集指标','',
        '可以用空间富集回答“预测高状态 CD8 是否更倾向位于 CAF/macrophage niche”，但它与逐细胞区分 niche 的 AUROC 是不同问题。全队列平均 AUROC 仍为 0.530，不能因为换指标就把预测准确性说得更强。','',
        '来源为原始 CD39/TIGIT 回投，74 位患者、38,299 个 CD8。Observed niche 固定为 spatial_y >0。没有更改阈值、ROI、裁切、预测或 cohort。','',
        '| 比较 | 低组/其余组 niche 比例 | 高组 niche 比例 | 绝对差（百分点，95% CI） | 相对风险 RR（95% CI） | 优势比 OR（95% CI） |',
        '|---|---:|---:|---|---|---|']
    for r in summary.itertuples():
        label='原 top40% vs 其余60%' if r.contrast=='top40_vs_other' else '原 Q5 vs Q1（最高/最低20%）'
        lines.append(f'| {label} | {100*r.low_fraction:.2f}% | {100*r.high_fraction:.2f}% | {100*r.difference:.2f} [{100*r.difference_ci_low:.2f}, {100*r.difference_ci_high:.2f}] | {r.risk_ratio:.3f} [{r.risk_ratio_ci_low:.3f}, {r.risk_ratio_ci_high:.3f}] | {r.odds_ratio:.3f} [{r.odds_ratio_ci_low:.3f}, {r.odds_ratio_ci_high:.3f}] |')
    q=summary[summary.contrast.eq('Q5_vs_Q1')].iloc[0]
    lines += ['',f'Q5−Q1：51/74 位患者为正向，18 位负向、5 位相同；双侧配对 Wilcoxon P={q.wilcoxon_two_sided_p:.6g}。旧空间图用预设单侧 greater 检验，所以旧 P 值约为这里的一半，不是数据改变。',
        '', '所有比例先在患者内计算，再对患者等权平均。RR 为两个等患者风险的比，OR 为对应 odds 的比；不是患者比值的算术平均。bootstrap 以患者为单位成对重抽样 10,000 次。Q1/Q5 的细胞数因秩和小样本会略有差异，完整分母见 patient_enrichment.csv。',
        '', '更适合 Panel F 的定量补充是 Q1–Q5 的 niche 比例趋势，或 Q1/Q5 的患者配对图，标注绝对增加 7.5 个百分点和 RR=1.17。这些数字支持温和的群体空间富集，不能称为强逐细胞预测或独立验证；较小的 P 值不等于较大的效应。',
        '', '本比较沿用之前已经存在的 top40 和五分位定义。把中间60%排除后的 Q5/Q1 是尾部状态对比，应明确写出，不能当成覆盖全部细胞的分类准确度。这些富集指标回答不同的生物学问题，并不改变整体分类能力较弱的结论。',
        '', '脚本、逐患者统计、汇总与哈希审计全部在本目录。正式图和 manuscript 未修改。']
    (HERE/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n')
    print(summary.to_string(index=False))


if __name__=='__main__':main()
