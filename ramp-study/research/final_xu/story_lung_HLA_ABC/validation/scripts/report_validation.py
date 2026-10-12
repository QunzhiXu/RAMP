"""Evaluate the fixed all-source predictor and render Zheng validation artifacts."""
import package_runtime as rt
from package_runtime import BASE
import json, html, hashlib
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from matplotlib.colors import Normalize
from spatial_plot import pair_axes, add_colorbar, plt

OUT=BASE/'results'
FIG=OUT/'figures'
CONFIGS=['RAMP_k30','RAMP_k100','baseline_k30','baseline_k100']


def read_predictions():
    return pd.read_csv(OUT/'PREDICTIONS_AND_TRUTH.csv.gz',
                       dtype={'observed_HLA_A':np.float32},float_precision='round_trip')


def rho(a,b):
    if len(a)<3 or np.ptp(a)<1e-12 or np.ptp(b)<1e-12:return float('nan')
    return float(spearmanr(a,b).statistic)


def median_interval(values):
    a=np.asarray(values,float);a=a[np.isfinite(a)]
    if not len(a):return [float('nan')]*3
    rng=np.random.default_rng(20260909)
    samples=np.median(a[rng.integers(0,len(a),(3000,len(a)))],axis=1)
    return [float(np.median(a)),*map(float,np.quantile(samples,[.025,.975]))]


def verify_reproduction(data):
    with np.load(BASE/'reference/FULL_PREDICTIONS.npz') as old:
        assert np.array_equal(data.source_row_id,old['source_row_id'])
        errors={c:float(np.abs(data[c]-old['prediction'][:,i]).max()) for i,c in enumerate(CONFIGS)}
    assert max(errors.values())<1e-12,errors
    assert data.source_row_id.is_unique and len(data)==693693
    assert data.patient_id.nunique()==28 and data.core_id.nunique()==119
    for _,g in data.groupby('patient_id'):
        assert np.max(np.abs(rankdata(g.observed_HLA_A)/(len(g)+1)-g.observed_HLA_A_patient_rank))<1e-12
    oldmetrics=pd.read_csv(BASE/'reference/PREVIOUS_METRICS.csv').fillna({'core_id':''})
    metrics=pd.read_csv(OUT/'METRICS.csv').fillna({'core_id':''})
    keys=['level','patient_id','core_id','config']
    paired=metrics.merge(oldmetrics,on=keys,validate='one_to_one',suffixes=('','_old'))
    assert len(paired)==len(metrics)==588
    metric_error=float(np.abs(paired.cell_spearman-paired.cell_spearman_old).max())
    assert metric_error<1e-12
    result={'status':'PASS','time_utc':rt.now(),'n_cells':len(data),'n_patients':28,'n_cores':119,
            'full_prediction_max_error':errors,'cell_spearman_max_error':metric_error,
            'measured_patient_ranks_recomputed':True,'n_original_anchor_cells':int(data.original_reference_cell.sum()),
            'same_fixed_alignment':True,'historical_results_reproduced':True}
    rt.save_json(OUT/'REPRODUCTION.json',result)
    return result


def evaluate(data):
    metrics=pd.read_csv(OUT/'METRICS.csv')
    summary=[]
    for (level,config),g in metrics.groupby(['level','config'],sort=False):
        med,lo,hi=median_interval(g.cell_spearman)
        if level=='core':
            groups=[z.cell_spearman.to_numpy() for _,z in g.groupby('patient_id')]
            choices=np.random.default_rng(20260909).integers(0,len(groups),(3000,len(groups)))
            bootstrap=[np.median(np.concatenate([groups[i] for i in ids])) for ids in choices]
            lo,hi=map(float,np.quantile(bootstrap,[.025,.975]))
        summary.append({'level':level,'config':config,'n_units':len(g),'n_cells':int(g.n_cells.sum()),
                        'median_spearman':med,'median_ci_low':lo,'median_ci_high':hi,
                        'bootstrap_unit':'patient (all cores retained together)' if level=='core' else 'patient',
                        'min_spearman':float(g.cell_spearman.min()),'max_spearman':float(g.cell_spearman.max()),
                        'positive_units':int((g.cell_spearman>0).sum()),'mean_rank_MAE':float(g.rank_MAE.mean())})
    pd.DataFrame(summary).to_csv(OUT/'SUMMARY_METRICS.csv',index=False)
    records=[];means=[]
    for (pid,typ),g in data.groupby(['patient_id','cell_type'],sort=True):
        if len(g)<20:continue
        for config in CONFIGS:
            records.append({'patient_id':pid,'cell_type':typ,'config':config,'n_cells':len(g),
                            'spearman':rho(g.observed_HLA_A,g[config]),
                            'rank_MAE':float(np.abs(g.observed_HLA_A_patient_rank-g[config]).mean())})
            means.append({'patient_id':pid,'cell_type':typ,'config':config,'n_cells':len(g),
                          'predicted_mean':float(g[config].mean()),'observed_rank_mean':float(g.observed_HLA_A_patient_rank.mean()),
                          'observed_raw_mean':float(g.observed_HLA_A.astype(float).mean())})
    per=pd.DataFrame(records);per.to_csv(OUT/'PATIENT_CELLTYPE_METRICS.csv',index=False)
    means=pd.DataFrame(means);means.to_csv(OUT/'PATIENT_CELLTYPE_MEANS.csv',index=False)
    grouped=[];mean_corr=[]
    for (typ,config),g in per.groupby(['cell_type','config']):
        med,lo,hi=median_interval(g.spearman)
        grouped.append({'cell_type':typ,'config':config,'n_patients':len(g),'n_cells':int(g.n_cells.sum()),
                        'n_valid_spearman':int(g.spearman.notna().sum()),'median_spearman':med,'ci_low':lo,'ci_high':hi})
    for (typ,config),g in means.groupby(['cell_type','config']):
        for truth in ['observed_rank_mean','observed_raw_mean']:
            mean_corr.append({'cell_type':typ,'config':config,'truth_summary':truth,'n_patients':len(g),
                              'spearman':rho(g.predicted_mean,g[truth])})
    types=pd.DataFrame(grouped);types.to_csv(OUT/'CELLTYPE_SUMMARY.csv',index=False)
    pd.DataFrame(mean_corr).to_csv(OUT/'PATIENT_MEAN_CORRELATIONS.csv',index=False)
    # Keep the original UMAP sample separately; its measured ranks use 1,000 cells/patient.
    md=pd.read_csv(BASE.parent/'data_umap/data/CELL_METADATA.csv.gz',low_memory=False)
    zm=md[md.dataset.eq('Zheng mIF')].copy()
    sample=data.set_index('source_row_id').loc[zm.source_row_id].reset_index()
    assert np.array_equal(sample.patient_id,zm.patient_id)
    sample['cell_key']=zm.cell_key.to_numpy()
    sample['observed_HLA_A_sample_patient_rank']=sample.groupby('patient_id').observed_HLA_A.transform(
        lambda a:rankdata(a)/(len(a)+1))
    sample.to_csv(OUT/'UMAP_SAMPLE_PREDICTIONS.csv.gz',index=False,compression={'method':'gzip','compresslevel':3,'mtime':0})
    sample_metrics=[]
    for pid,g in sample.groupby('patient_id'):
        for config in CONFIGS:
            sample_metrics.append({'patient_id':pid,'config':config,'n_cells':len(g),'spearman':rho(g.observed_HLA_A,g[config])})
    pd.DataFrame(sample_metrics).to_csv(OUT/'UMAP_SAMPLE_PATIENT_METRICS.csv',index=False)
    return metrics,pd.DataFrame(summary),types,sample,md


def draw_summary(metrics,types):
    fig,axes=plt.subplots(1,2,figsize=(12.5,5.4),layout='constrained')
    cases=[('baseline_k30','Baseline\nk = 30','#9099a3'),('baseline_k100','Baseline\nk = 100','#b4bbc2'),
           ('RAMP_k30','RAMP\nk = 30','#167d8d'),('RAMP_k100','RAMP\nk = 100','#71b2ba')]
    for ax,level,title in zip(axes,['patient','core'],['28 patients','119 tissue cores']):
        for i,(config,label,color) in enumerate(cases):
            g=metrics[metrics.level.eq(level)&metrics.config.eq(config)]
            x=np.random.default_rng(42).uniform(i-.20,i+.20,len(g))
            ax.scatter(x,g.cell_spearman,s=15 if level=='core' else 26,color=color,alpha=.6,linewidths=0)
            med=float(g.cell_spearman.median());ax.plot([i-.25,i+.25],[med,med],color='#202b36',lw=2)
            ax.text(i,.975,f'{med:.3f}',ha='center',va='top',fontsize=11)
        ax.set(xticks=np.arange(4),xticklabels=[c[1] for c in cases],ylim=(0,1),
               ylabel='Cell-wise Spearman correlation',title=f'{title} | all cell types')
        ax.grid(axis='y',alpha=.15);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Zheng HLA-A validation | 693,693 cells\nAll Senosain source cells → all Zheng target cells; numbers show medians',fontsize=14)
    fig.savefig(FIG/'HLA_A_validation_summary.png',dpi=220);plt.close(fig)

    types=types[types.config.isin(['RAMP_k30','baseline_k30'])]
    labels=sorted(types.cell_type.unique())
    fig,ax=plt.subplots(figsize=(9.5,max(4.8,len(labels)*.48+1.5)),layout='constrained')
    for i,label in enumerate(labels):
        for config,offset,color,legend in [('baseline_k30',-.12,'#939da8','Baseline k=30'),('RAMP_k30',.12,'#167d8d','RAMP k=30')]:
            r=types[types.cell_type.eq(label)&types.config.eq(config)].iloc[0]
            if not np.isfinite(r.median_spearman):continue
            ax.errorbar(r.median_spearman,i+offset,xerr=[[r.median_spearman-r.ci_low],[r.ci_high-r.median_spearman]],
                        fmt='o',color=color,capsize=3,label=legend if i==0 else None)
    counts=types[types.config.eq('RAMP_k30')].set_index('cell_type').n_patients
    ax.set(yticks=np.arange(len(labels)),yticklabels=[f'{t} (n={counts[t]} patients)' for t in labels],
           xlabel='Median within-patient Spearman (95% patient-bootstrap interval)',xlim=(-.2,1),
           title='Zheng author cell types | all-source transfer\nAt least 20 cells per patient and cell type')
    ax.invert_yaxis();ax.axvline(0,color='#bcc3ca',lw=.8);ax.grid(axis='x',alpha=.15)
    ax.legend(loc='lower right',frameon=False);ax.spines[['top','right']].set_visible(False)
    fig.savefig(FIG/'HLA_A_celltype_validation.png',dpi=220);plt.close(fig)


def draw_umap(sample,md):
    u=np.load(BASE.parent/'umap/results/RAMP_UMAP.npy')
    zi=np.flatnonzero(md.dataset.eq('Zheng mIF'))
    assert np.array_equal(md.iloc[zi].cell_key,sample.cell_key)
    order=np.random.default_rng(42).permutation(len(zi));xy=u[zi][order]
    fig,axes=plt.subplots(1,2,figsize=(12.5,6.3),layout='constrained')
    for ax,col,title in zip(axes,['observed_HLA_A_sample_patient_rank','RAMP_k30'],['Measured HLA-A','Predicted HLA-A proxy']):
        ax.scatter(u[:,0],u[:,1],s=.7,c='#cbd0d5',alpha=.14,linewidths=0,rasterized=True)
        artist=ax.scatter(xy[:,0],xy[:,1],c=sample[col].to_numpy()[order],cmap='viridis',vmin=0,vmax=1,
                          s=2.5,linewidths=0,alpha=1,rasterized=True)
        ax.set(title=title,xlabel='UMAP 1',ylabel='UMAP 2',xticks=[],yticks=[])
        ax.set_aspect('equal',adjustable='box');ax.spines[['top','right']].set_visible(False)
        assert np.array_equal(artist.get_offsets(),xy)
    fig.colorbar(artist,ax=axes,shrink=.72,label='Measured patient-relative rank / predicted rank')
    fig.suptitle('Zheng validation on the fixed three-cohort UMAP\n28,000 Zheng cells in color; all three cohorts in grey | all-source k=30',fontsize=14)
    fig.savefig(FIG/'HLA_A_measured_vs_predicted_UMAP.png',dpi=220);plt.close(fig)


def draw_spatial(data,metrics):
    (OUT/'cores').mkdir(exist_ok=True)
    cm=metrics[metrics.level.eq('core')&metrics.config.eq('RAMP_k30')].set_index('core_id')
    audit=[];cards=[]
    for no,(cid,g) in enumerate(data.groupby('core_id',sort=True),1):
        r=float(cm.loc[cid,'cell_spearman'])
        fig,axes=plt.subplots(1,2,figsize=(12.2,6.5))
        fig.subplots_adjust(left=.025,right=.975,bottom=.22,top=.82,wspace=.025)
        artist,rec=pair_axes(axes,g,r)
        fig.suptitle(f'Zheng | {g.patient_id.iloc[0]} | {cid} | {len(g):,} cells',fontsize=17,y=.973)
        fig.text(.5,.912,f'Same tissue coordinates · all cell types · k = 30 · Spearman ρ = {r:.3f}',ha='center',fontsize=11)
        add_colorbar(fig,artist,[.35,.12,.30,.025])
        fig.text(.5,.025,'Measured: within-patient HLA-A rank. Predicted: transferred Senosain HLA-ABC rank; no spatial smoothing.',ha='center',fontsize=9)
        filename=f'{g.patient_id.iloc[0]}__{cid}__HLA_A_spatial.png'
        fig.savefig(OUT/'cores'/filename,dpi=150);plt.close(fig)
        rec['file']='cores/'+filename;audit.append(rec)
        cards.append(f'<article><h2>{html.escape(str(g.patient_id.iloc[0]))} · {html.escape(cid)}</h2><p>{len(g):,} cells · ρ={r:.3f}</p><a href="cores/{filename}"><img loading="lazy" src="cores/{filename}" alt="Measured versus predicted HLA at cell coordinates"></a></article>')
        if no%20==0:print(f'Spatial figures {no}/119',flush=True)
    # Retain the previously selected presentation core and color range, without a new search.
    g=data[data.patient_id.eq('P_26')&data.core_id.eq('C_30')]
    assert len(g)==10719
    g.to_csv(OUT/'C30_CELLS.csv.gz',index=False,compression={'method':'gzip','compresslevel':3,'mtime':0})
    r=float(cm.loc['C_30','cell_spearman']);lo,hi=.25,.60
    fig,axes=plt.subplots(1,2,figsize=(12.8,7.2))
    fig.subplots_adjust(left=.03,right=.97,bottom=.225,top=.81,wspace=.025)
    artist,rec=pair_axes(axes,g,r,point_size=4.6)
    for ax in axes:
        ax.collections[0].set_norm(Normalize(lo,hi,clip=True));ax.collections[0].set_rasterized(False)
        ax.title.set_fontsize(17)
    fig.suptitle('Spatial agreement of measured and predicted HLA signal',fontsize=20,y=.976)
    fig.text(.5,.914,f'Zheng P_26 · C_30 · {len(g):,} cells · Spearman ρ = {r:.3f}',ha='center',fontsize=13,color='#344253')
    cb=fig.colorbar(artist,cax=fig.add_axes([.35,.135,.30,.025]),orientation='horizontal',ticks=[.25,.3,.4,.5,.6])
    cb.ax.set_xticklabels(['≤0.25','0.3','0.4','0.5','≥0.6']);cb.set_label('Relative expression rank / predicted rank')
    fig.text(.5,.058,'Same cells and coordinates · k = 30 · shared color range 0.25–0.6; endpoint colors saturate.',ha='center',fontsize=10,color='#4B5868')
    fig.text(.5,.023,'Measured: within-patient HLA-A rank. Prediction: transferred Senosain HLA-ABC rank (HLA-A proxy).',ha='center',fontsize=10,color='#4B5868')
    fig.savefig(FIG/'C30_HLA_A_spatial.png',dpi=300);fig.savefig(FIG/'C30_HLA_A_spatial.svg');plt.close(fig)
    rec.update({'color_vmin':lo,'color_vmax':hi,'numeric_values_changed':False,'display_scale_changed':True,
                'predicted_color_rescaled':True,'endpoint_saturation_labelled':True,'file':'figures/C30_HLA_A_spatial.png',
                'selection':'Previously explored and user-selected core; existing display limits retained'})
    for name,col in [('measured','observed_HLA_A_patient_rank'),('predicted','RAMP_k30')]:
        rec[name+'_low_saturation']=float((g[col]<=lo).mean());rec[name+'_high_saturation']=float((g[col]>=hi).mean())
    rt.save_json(OUT/'C30_DISPLAY.json',rec)
    rt.save_json(OUT/'PLOT_AUDIT.json',{'time_utc':rt.now(),'n_core_pairs':119,'all_cores_same_color_range':[0,1],
                                    'plots':audit,'presentation':rec})
    (OUT/'GALLERY.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Zheng HLA-A validation</title><style>body{font:16px/1.5 system-ui;margin:25px;color:#233746}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:20px}article{border:1px solid #dce2e6;padding:15px}h2,p{margin:5px}img{width:100%}</style><h1>Zheng HLA-A：全部 119 个 core</h1><p>28 位患者、693,693 个细胞；全部源细胞 → 全部目标细胞，k=30。以下所有 core 统一使用 0–1 色标。</p><p><a href="figures/C30_HLA_A_spatial.png">C30 展示图（已选定的 0.25–0.6 色标）</a> · <a href="figures/HLA_A_measured_vs_predicted_UMAP.png">UMAP 实测／预测对照</a> · <a href="REPORT_ZH.md">结果说明</a></p><main>'+''.join(cards)+'</main></html>')


def write_report(data,summary):
    def row(level,config):return summary[summary.level.eq(level)&summary.config.eq(config)].iloc[0]
    p=row('patient','RAMP_k30');c=row('core','RAMP_k30')
    c30=json.loads((OUT/'C30_DISPLAY.json').read_text())
    lines=['# Zheng HLA-A 验证','',
           '固定方案重新运行完成：28,000 个 Senosain 参考细胞 → 同一批 28 位 Zheng 患者的全部 693,693 个细胞、119 个 core。主版本 RAMP k=30，保留 k=100 敏感性和相同预测器的 KMEANSLOW baseline。','',
           '## 预测的含义','',
           '输入为 CD3、CD4、CD8、CD31、KI67、VIMENTIN、EPI、CD45、HLA_DR 九标记。源端应变量是 Senosain 患者内 HLA-ABC 的平均秩/(1000+1)。每种方法分别用全部源细胞拟合 StandardScaler，在九维表达空间做欧氏近邻，以 exp(-(d/d_k)^2) 归一化加权平均。HLA-A 和 HLA-ABC 不进入对齐特征；Zheng 实测 HLA-A 不进入预测器。','',
           '预测步骤与 Cords HLA-ABC 状态转移相同。Zheng 的 HLA-A 是相关的 MHC-I 验证读数，所以图中称为 Predicted HLA-A proxy；并非另行训练的原始 HLA-A 荧光回归。Cords 的高状态细胞比例是预测后的阈值汇总，本目录不重新选择该阈值。','',
           '原来每位 Zheng 患者的 1,000 个参考细胞用于恢复该患者的 KMEANSLOW 参数、经验百分位和 RAMP 单调 profile，然后应用于同一患者的其余细胞。没有重新对齐，没有用细胞标签、空间坐标或临床结局训练近邻预测器。','',
           '## 全队列结果','',
           '| 方法 | 患者内 Spearman 中位数（28 人） | Core 内 Spearman 中位数（119 个） |',
           '| --- | ---: | ---: |']
    for conf in CONFIGS:lines.append(f'| {conf} | {row("patient",conf).median_spearman:.3f} | {row("core",conf).median_spearman:.3f} |')
    lines += ['',f'RAMP k=30：119 个 core 中 {int(c.positive_units)} 个相关性为正，范围 {c.min_spearman:.3f}–{c.max_spearman:.3f}。这些指标使用未作显示截断、未作空间平滑的单细胞数值。Baseline 也有较强相关性，结果不支持将信号归因于 RAMP 独有的优势。','',
              '按作者细胞类型分层的患者内结果见 CELLTYPE_SUMMARY.csv；每个患者每类至少 20 个细胞，所有源细胞继续参与预测。PATIENT_MEAN_CORRELATIONS.csv 同时保留各细胞类型患者均值对实测相对秩均值、原始荧光均值的对应关系。细胞内排序恢复不等价于患者绝对蛋白水平恢复。置信区间采用 3,000 次患者 bootstrap；core 中位数的区间按患者整体重采样，保留同一患者内的全部 core。','',
              '## UMAP 与真实空间展示','',
              '- figures/HLA_A_measured_vs_predicted_UMAP.png：使用 ../umap/results/RAMP_UMAP.npy 的同一三队列坐标。28,000 个 Zheng 参考细胞着色，三队列细胞作为灰色背景；左侧实测秩按原每人 1,000 个抽样细胞计算，与早期 UMAP 验证定义相同。','',
              f'- figures/C30_HLA_A_spatial.png / .svg：P_26 / C_30 全部 {c30["n_plotted_each_panel"]:,} 个细胞，ρ={c30["cell_spearman"]:.3f}。沿用用户选定 core 和 0.25–0.6 色标，端点颜色饱和并有标注。它是经过探索挑选的展示例子，不代表未选择的总体表现。','',
              '- GALLERY.html 和 cores/：全部 119 个 core，统一 0–1 色标。左右使用完全相同细胞、原始质心坐标、绘制顺序、坐标范围和点大小；100 µm 比例尺。空间图左侧的秩按患者完整细胞计算。','',
              '## 复现范围','',
              '这次是已经看过的验证数据上，对固定算法与已选展示的整理复现；没有新一轮模型、core 或色标搜索。全部 693,693 个细胞四组预测与既有结果逐项核对，原 28,000 个参考细胞预测亦核对一致。详见 REPRODUCTION.json 与 PREPARATION_VERIFICATION.json。','',
              'Zheng 作者的一般 Fibroblast 不是 Cords 的 vCAF 标签；本验证支持可转移的相对 HLA 信号，并不直接验证 Cords vCAF 中真实 HLA-ABC，也不复制其 DFS 关联。','']
    (OUT/'REPORT_ZH.md').write_text('\n'.join(lines))
    rt.save_json(OUT/'SUMMARY.json',{'n_source_cells':28000,'n_target_cells':len(data),'n_patients':28,'n_cores':119,
                                  'primary_config':'RAMP_k30','median_patient_rho':float(p.median_spearman),
                                  'median_core_rho':float(c.median_spearman),'positive_cores':int(c.positive_units),
                                  'C30_rho':c30['cell_spearman'],'C30_n_cells':c30['n_plotted_each_panel']})


def main():
    FIG.mkdir(exist_ok=True)
    data=read_predictions();verify_reproduction(data)
    metrics,summary,types,sample,md=evaluate(data)
    draw_summary(metrics,types);draw_umap(sample,md)
    draw_spatial(data,metrics);write_report(data,summary)
    prep=json.loads((OUT/'PREPARATION_VERIFICATION.json').read_text())
    assert prep['status']=='PASS'
    design=json.loads((OUT/'DESIGN.json').read_text())
    unchanged={p:rt.sha(p)==h for p,h in design['input_sha256'].items()}
    assert all(unchanged.values())
    files=[p for p in OUT.rglob('*') if p.is_file() and p.name not in {'RUN_MANIFEST.json','RUN_STATUS.json','RUNNING.lock','FIGURE_QA.json'}]
    files += [p for p in (BASE/'data').glob('*') if p.is_file()]
    rt.save_json(OUT/'RUN_MANIFEST.json',{'time_utc':rt.now(),'source_inputs':design['input_sha256'],
                                      'script_sha256':{str(p.relative_to(BASE)):rt.sha(p) for p in (BASE/'scripts').glob('*.py')},
                                      'output_sha256':{str(p.relative_to(BASE)):rt.sha(p) for p in files},
                                      'reference_sha256':{str(p.relative_to(BASE)):rt.sha(p) for p in (BASE/'reference').glob('*') if p.is_file()}})
    rt.save_json(OUT/'RUN_STATUS.json',{'phase':'complete','updated_utc':rt.now(),'n_core_pairs':119,'visual_review':'pending'})
    print(json.dumps(json.loads((OUT/'SUMMARY.json').read_text()),indent=2),flush=True)


if __name__=='__main__':main()
