#!/usr/bin/env python3
"""Read-only audit of original Pair: AUROC by scale, and fixed threshold grid."""
from pathlib import Path
import os,hashlib,json
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(HERE/'.cache/matplotlib'))
os.environ.setdefault('MPLBACKEND','Agg')
import fitz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score,roc_curve,auc

SOURCE=ROOT/'spatial_return_survival/output/haist74_pt_cd8_tigit_cd39_return.csv.gz'
THRESHOLDS=np.arange(5,100,5)/100
INK='#182027';BLUE='#2775B6';ORANGE='#D95F02';TEAL='#238A8D'


def scores(b):
    truth=b.spatial_y.to_numpy()>0
    return dict(cells=len(b),positive_cells=int(truth.sum()),negative_cells=int((~truth).sum()),
        prevalence=float(truth.mean()),auroc=float(roc_auc_score(truth,b.predicted_score)) if np.unique(truth).size==2 else np.nan,
        rank_auroc=float(roc_auc_score(truth,b.within_patient_rank)) if np.unique(truth).size==2 else np.nan,
        spearman=float(spearmanr(b.spatial_y,b.predicted_score).statistic) if np.ptp(b.spatial_y)>0 else np.nan)


def threshold_rows(b,identity):
    truth=b.spatial_y.to_numpy()>0;r=b.within_patient_rank.to_numpy()
    rows=[]
    for t in THRESHOLDS:
        high=r>=t;tp=int((high&truth).sum());fp=int((high&~truth).sum())
        fn=int((~high&truth).sum());tn=int((~high&~truth).sum())
        recall=tp/(tp+fn);spec=tn/(tn+fp)
        precision=tp/(tp+fp) if tp+fp else 0.
        rows.append(dict(identity=identity,threshold=float(t),top_fraction=1-float(t),tp=tp,fp=fp,fn=fn,tn=tn,
                         precision=precision,recall=recall,specificity=spec,balanced_accuracy=(recall+spec)/2,
                         f1=2*tp/(2*tp+fp+fn),jaccard=tp/(tp+fp+fn),selected_fraction=float(high.mean())))
    return rows


def bootstrap(v):
    v=np.asarray(v,float);v=v[np.isfinite(v)]
    idx=np.random.default_rng(20260913).integers(0,len(v),(10000,len(v)))
    lo,hi=np.quantile(v[idx].mean(axis=1),[.025,.975])
    return dict(n=len(v),mean=float(v.mean()),median=float(np.median(v)),ci_low=float(lo),ci_high=float(hi))


def threshold_choice(table,metric):
    best=table[metric].max()
    b=table[np.isclose(table[metric],best,rtol=0,atol=1e-12)].copy()
    b['distance_to_original']=abs(b.threshold-.60)
    return b.sort_values(['distance_to_original','threshold']).iloc[0]


def make_plot(patient,scopes,mean_roc,summary,meta):
    fig=plt.figure(figsize=(12.2,4.3))
    fig.text(.02,.97,'Original CD39/TIGIT return: local example versus the full cohort',fontsize=14.5,weight='bold',va='top')
    axes=[fig.add_axes([.064,.19,.27,.60]),fig.add_axes([.413,.19,.16,.60]),fig.add_axes([.664,.19,.318,.60])]
    ax=axes[0]
    ax.plot(mean_roc.fpr,mean_roc.mean_tpr,color=BLUE,lw=2,label=f'Patient mean: {meta["patient_auroc"]["mean"]:.3f}')
    for name,color,title in [('full_roi',TEAL,'Full ROI'),('display_crop',ORANGE,'Displayed crop')]:
        b=scopes[name];fpr,tpr,_=roc_curve(b.spatial_y>0,b.predicted_score)
        ax.plot(fpr,tpr,color=color,lw=1.8,label=f'{title}: {roc_auc_score(b.spatial_y>0,b.predicted_score):.3f}')
    ax.plot([0,1],[0,1],color='#9AA3AB',ls='--',lw=.8)
    ax.set(xlim=(0,1),ylim=(0,1.025),xticks=[0,.5,1],yticks=[0,.5,1])
    ax.set_title('ROC comparison',fontsize=11.5,weight='bold',loc='left',pad=10)
    ax.set_xlabel('False-positive rate',fontsize=9.5);ax.set_ylabel('True-positive rate',fontsize=9.5)
    ax.legend(loc='lower right',fontsize=8.5,frameon=False)
    ax=axes[1];v=patient.auroc.to_numpy()
    ax.boxplot([v],positions=[0],widths=.40,patch_artist=True,showfliers=False,whis=(0,100),
        boxprops=dict(facecolor='#DFEAF3',edgecolor=BLUE),medianprops=dict(color=BLUE,lw=1.5),
        whiskerprops=dict(color=BLUE),capprops=dict(color=BLUE))
    ax.scatter(np.random.default_rng(20260913).uniform(-.17,.17,len(v)),v,s=13,color=BLUE,alpha=.65,edgecolor='white',linewidth=.25,zorder=3)
    ax.axhline(.5,color='#9AA3AB',ls='--',lw=.8)
    ax.set(xlim=(-.5,.5),ylim=(0,1.025),xticks=[],yticks=[0,.25,.5,.75,1])
    ax.set_title('74 individual patients',fontsize=11.3,weight='bold',loc='center',pad=10)
    ax.set_ylabel('AUROC',fontsize=9.5)
    ax.text(.5,-.16,f'Mean {v.mean():.3f}; median {np.median(v):.3f}',transform=ax.transAxes,ha='center',fontsize=8.8)
    ax=axes[2]
    for metric,color,label in [('precision',TEAL,'Precision'),('recall',BLUE,'Recall'),('balanced_accuracy',ORANGE,'Balanced accuracy')]:
        ax.plot(summary.threshold,summary[metric],color=color,label=label,lw=1.7)
    ax.axvline(.60,color='#7C858D',ls='--',lw=.9)
    ax.set(xlim=(.05,.95),ylim=(0,1.025),xticks=[.1,.3,.5,.7,.9],yticks=[0,.25,.5,.75,1])
    ax.set_title('Threshold trade-off across patients',fontsize=11.3,weight='bold',loc='left',pad=10)
    ax.set_xlabel('Within-patient percentile cutoff',fontsize=9.5)
    ax.set_ylabel('Patient-equal mean',fontsize=9.5)
    ax.legend(loc='upper right',fontsize=8.4,frameon=False)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False);ax.tick_params(labelsize=8.5,length=3)
    fig.text(.02,.035,'Observed niche: spatial score > 0. Dashed cutoff: original top 40%. AUROC does not change when this cutoff changes.',fontsize=9.5)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    items=list(fig.texts)
    for ax in axes:items+=list(ax.texts)+[ax.title,ax._left_title,ax.xaxis.label,ax.yaxis.label]+ax.get_xticklabels()+ax.get_yticklabels()
    for t in items:
        if t.get_visible() and t.get_text():
            b=t.get_window_extent(renderer)
            assert b.x0>=-1 and b.y0>=-1 and b.x1<=fig.bbox.width+1 and b.y1<=fig.bbox.height+1,t.get_text()
    for ext in ['png','pdf','svg']:fig.savefig(HERE/f'pair_auroc_and_thresholds.{ext}',dpi=240,facecolor='white')
    plt.close(fig)
    with fitz.open(HERE/'pair_auroc_and_thresholds.pdf') as doc:
        doc[0].get_pixmap(dpi=170,alpha=False).save(HERE/'pdf_preview.png')


def main():
    original_hash=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    d=pd.read_csv(SOURCE,dtype={'patient':str})
    assert len(d)==38299 and d.patient.nunique()==74
    assert np.array_equal(d.in_niche,d.spatial_y>0)
    scopes={'all_cells':d,'patient15219':d[d.patient.eq('15219')],'full_roi':d[d.unique_region.eq('522_S2reg054')]}
    b=scopes['full_roi'];scopes['display_crop']=b[b.x.between(1205,1805,inclusive='left')&b.y.between(205,670,inclusive='left')]
    scope_metrics=pd.DataFrame([dict(scope=name,**scores(b)) for name,b in scopes.items()])
    scope_metrics.to_csv(HERE/'metrics_by_scale.csv',index=False)
    patient=pd.DataFrame([dict(patient=k,**scores(b)) for k,b in d.groupby('patient')])
    patient.to_csv(HERE/'patient_auroc.csv',index=False)
    rois=pd.DataFrame([dict(patient=str(b.patient.iloc[0]),roi=k,**scores(b)) for k,b in d.groupby('unique_region')])
    rois['eligible_for_roi_ranking']=(rois.cells>=50)&(rois.positive_cells>=10)&(rois.negative_cells>=10)
    rois['auroc_rank_eligible']=rois.loc[rois.eligible_for_roi_ranking,'auroc'].rank(method='min',ascending=False)
    rois.sort_values('auroc',ascending=False).to_csv(HERE/'all_roi_auroc.csv',index=False)
    rois[rois.eligible_for_roi_ranking].sort_values('auroc',ascending=False).to_csv(HERE/'eligible_roi_ranking.csv',index=False)
    records=[]
    for key,b in d.groupby('patient'):records+=threshold_rows(b,str(key))
    patient_threshold=pd.DataFrame(records);patient_threshold.to_csv(HERE/'patient_threshold_metrics.csv',index=False)
    fields=['precision','recall','specificity','balanced_accuracy','f1','jaccard','selected_fraction']
    summary=patient_threshold.groupby('threshold',as_index=False)[fields].mean()
    summary['top_fraction']=1-summary.threshold
    summary.to_csv(HERE/'threshold_macro_summary.csv',index=False)
    crop_threshold=pd.DataFrame(threshold_rows(scopes['display_crop'],'display_crop'))
    crop_threshold.to_csv(HERE/'display_crop_thresholds.csv',index=False)
    roi_threshold=pd.DataFrame(threshold_rows(scopes['full_roi'],'full_roi'))
    roi_threshold.to_csv(HERE/'full_roi_thresholds.csv',index=False)
    best=threshold_choice(summary,'balanced_accuracy');old=summary[np.isclose(summary.threshold,.60)].iloc[0]
    crop_best=threshold_choice(crop_threshold,'f1');crop_old=crop_threshold[np.isclose(crop_threshold.threshold,.60)].iloc[0]
    heldout=[]
    for patient_id in patient.patient:
        train=patient_threshold[patient_threshold.identity.ne(patient_id)].groupby('threshold',as_index=False)[fields].mean()
        chosen=threshold_choice(train,'balanced_accuracy').threshold
        test=patient_threshold[patient_threshold.identity.eq(patient_id)]
        current=test[np.isclose(test.threshold,.60)].iloc[0]
        picked=test[np.isclose(test.threshold,chosen)].iloc[0]
        heldout.append(dict(patient=patient_id,chosen_threshold=float(chosen),original_balanced_accuracy=current.balanced_accuracy,
                            heldout_balanced_accuracy=picked.balanced_accuracy,delta_balanced_accuracy=picked.balanced_accuracy-current.balanced_accuracy))
    heldout=pd.DataFrame(heldout);heldout.to_csv(HERE/'leave_patient_out_threshold_selection.csv',index=False)
    fpr_grid=np.linspace(0,1,2001);curves=[]
    for _,b in d.groupby('patient'):
        fpr,tpr,_=roc_curve(b.spatial_y>0,b.predicted_score);curves.append(np.interp(fpr_grid,fpr,tpr))
    mean_roc=pd.DataFrame({'fpr':fpr_grid,'mean_tpr':np.mean(curves,axis=0)})
    mean_roc.to_csv(HERE/'mean_patient_roc.csv',index=False)
    assert abs(auc(mean_roc.fpr,mean_roc.mean_tpr)-patient.auroc.mean())<.002
    current_roi=rois[rois.roi.eq('522_S2reg054')].iloc[0]
    meta=dict(status='PASS',source=str(SOURCE),source_sha256=original_hash,inputs_unchanged=True,
        observed_definition='spatial_y > 0',patient_auroc=bootstrap(patient.auroc),patients_auroc_above_half=int((patient.auroc>.5).sum()),
        roi_total=len(rois),roi_eligible_count=int(rois.eligible_for_roi_ranking.sum()),
        current_roi_auroc_rank=int(current_roi.auroc_rank_eligible),
        roi_ranking_eligibility='at least 50 CD8 cells and at least 10 cells in each observed class',
        original_threshold=old.to_dict(),best_in_sample_balanced_accuracy=best.to_dict(),
        crop_original=crop_old.to_dict(),crop_best_in_sample_f1=crop_best.to_dict(),
        heldout_threshold_balanced_accuracy=bootstrap(heldout.heldout_balanced_accuracy),
        heldout_threshold_delta=bootstrap(heldout.delta_balanced_accuracy),
        threshold_grid=THRESHOLDS.tolist(),no_refit=True,no_roi_or_crop_changed=True,no_main_figure_replacement=True)
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==original_hash
    (HERE/'AUDIT.json').write_text(json.dumps(meta,indent=2)+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','text.color':INK})
    make_plot(patient,scopes,mean_roc,summary,meta)
    lines=['# Panel F 原始 Pair：AUROC 与阈值核查','',
      '**结论：当前展示局部 AUROC 0.724，但全队列患者等权平均 AUROC 0.530。它是空间对应的展示例，整体区分能力较弱，不能把局部表现当成整体预测验证。调整预测阈值不会改变 AUROC。**','',
      '## 不同尺度','', '| 范围 | CD8 数 | 连续回投分数 AUROC | 患者内 rank AUROC |','|---|---:|---:|---:|']
    for r in scope_metrics.itertuples():lines.append(f'| {r.scope} | {r.cells} | {r.auroc:.4f} | {r.rank_auroc:.4f} |')
    m=meta['patient_auroc']
    lines += ['',f'主要整体指标：74 位患者分别计算 AUROC 后等权平均 {m["mean"]:.4f}，中位数 {m["median"]:.4f}；患者 bootstrap 95% 区间 {m["ci_low"]:.4f}–{m["ci_high"]:.4f}。{meta["patients_auroc_above_half"]}/74 位患者 AUROC >0.5。',
      'all_cells 的 pooled AUROC 按细胞汇总，细胞多的患者占更大权重；raw score 与患者内 rank 的 pooled AUROC 不同，因为跨患者排序不同。单个患者内，两者 AUROC 一致。','',
      '## 目前是不是最好 ROI','',
      '原来的 ROI 选择以 top40 重叠细胞数 × 高状态相对其余 CD8 的正向 niche 比例差排序；当前 ROI 在该原始指标下排第一。这不等于 AUROC 第一。当前裁切也不是经过全域 AUROC 最大化搜索的裁切。',
      f'本次统计全部 {len(rois)} 个 ROI；为避免几枚细胞产生虚高的 ROI AUROC，另列出至少 50 个 CD8、正负类各至少 10 个细胞的 {meta["roi_eligible_count"]} 个 ROI。当前完整 ROI 的 AUROC 排第 {meta["current_roi_auroc_rank"]}。具体排名见 eligible_roi_ranking.csv。没有据此更换展示 ROI。','',
      '## 固定 observed niche 后的预测阈值扫描','',
      '固定真实标签 spatial_y >0。仅把预测患者内 percentile 阈值从 0.05 到 0.95，每步 0.05 扫描；阈值 0.60 就是原 top40%。没有更改 observed niche 定义。','',
      f'原阈值 0.60：平均患者 precision {old.precision:.4f}，recall {old.recall:.4f}，balanced accuracy {old.balanced_accuracy:.4f}，F1 {old.f1:.4f}。',
      f'同队列最大平均 balanced accuracy 的阈值为 {best.threshold:.2f}（top {100*(1-best.threshold):.0f}%），balanced accuracy {best.balanced_accuracy:.4f}，相比原阈值增加 {best.balanced_accuracy-old.balanced_accuracy:.4f}。',
      f'只对阈值做留一患者选择：在另外 73 位患者选阈值，再评价被留出的患者，平均 balanced accuracy {meta["heldout_threshold_balanced_accuracy"]["mean"]:.4f}，相对原阈值平均变化 {meta["heldout_threshold_delta"]["mean"]:.4f}。这是阈值选择诊断，并非重新发现、对齐和预测的完整独立验证。','',
      f'局部图原 0.60：precision {crop_old.precision:.4f}，recall {crop_old.recall:.4f}，F1 {crop_old.f1:.4f}。同一局部图上按 F1 选的阈值 {crop_best.threshold:.2f}：precision {crop_best.precision:.4f}，recall {crop_best.recall:.4f}，F1 {crop_best.f1:.4f}。这只是该展示图上的探索选择，不能提升全队列 AUROC。','',
      '## 对 Panel F 的建议','',
      '可以在空间图上明确标注本展示区域的 AUROC=0.724（n=61 CD8），同时提供全队列 74 位患者的 AUROC 分布或均值0.530。若将 F 描述为回投后的空间对应，使用全患者的 niche enrichment 也可以，但不能用它替代或掩盖较弱的 AUROC。','',
      '不建议靠更换 predicted-high cutoff 把图写成更强的预测验证，因为 AUROC 使用全连续分数、与这种 cutoff 无关。若改 observed niche 的 cutoff，等于更换被预测的生物学定义，需要另作有明确动机的敏感性分析，不能把挑出来的数值当作原任务的准确度提升。','',
      '图、表和脚本均独立保存在本目录；源数据、当前 Panel F 和 manuscript 均未修改。']
    (HERE/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(meta,indent=2))


if __name__=='__main__':main()
