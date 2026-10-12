from runtime import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
ORDER=['vCAF','SMA_CAF','mCAF','iCAF','Collagen_CAF','tpCAF','dCAF','hypoxic_CAF','hypoxic_tpCAF','PDPN_CAF','IDO_CAF','CD8','CD4','Bcell','Myeloid','Neutrophil','Blood','HEV','Lymphatic','normal','hypoxic','Other']
LABELS={'normal':'Tumour (non-hypoxic)','hypoxic':'Tumour (hypoxic)','Blood':'Blood vessel','HEV':'HEV','Lymphatic':'Lymphatic vessel','Other':'Other / unclassified','Bcell':'B cells','CD8':'CD8 T cells','CD4':'CD4 T cells'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
def name(x):return LABELS.get(x,x)
def output(fig,stem):
    fig.savefig(FIG/(stem+'.png'),dpi=210);fig.savefig(FIG/(stem+'.svg'));plt.close(fig)

def plots(mol,matched,immune):
    fig,axes=plt.subplots(1,2,figsize=(12.6,10.9));fig.subplots_adjust(left=.22,right=.975,bottom=.14,top=.85,wspace=.23)
    for ax,marker in zip(axes,MARKERS):
        for i,typ in enumerate(ORDER):
            row=mol[mol.population.eq('all_available')&mol.config.eq('RAMP_k30')&mol.metric.eq('rho')&mol.cell_type.eq(typ)&mol.marker.eq(marker)].iloc[0]
            color='#c34865' if typ=='vCAF' else ('#9ea5ad' if row.n_patients<20 else '#3b8993')
            ax.errorbar(row['mean'],i,xerr=[[row['mean']-row.ci_low],[row.ci_high-row['mean']]],fmt='o',ms=5.3 if typ=='vCAF' else 4.2,capsize=2.5,color=color,mfc='white' if row.n_patients<20 else color)
        ref=mol[mol.population.eq('all_available')&mol.config.eq('RAMP_k30')&mol.metric.eq('rho')&mol.cell_type.eq('vCAF')&mol.marker.eq(marker)].iloc[0]
        ax.axvline(ref['mean'],color='#c34865',ls=':',alpha=.65,lw=1)
        ax.axvline(0,color='#8d959c',ls='--',lw=1)
        labels=[name(t)+(' (n=2)' if t=='PDPN_CAF' else ' (n=9)' if t=='hypoxic_tpCAF' else '') for t in ORDER]
        ax.set(yticks=np.arange(22),yticklabels=labels if marker=='CD73' else [],title='Predicted HLA-ABC vs measured '+marker,xlabel='Mean patient correlation (Spearman ρ)',ylim=(21.6,-.7))
        ax.grid(axis='x',alpha=.13)
        for tick in ax.get_yticklabels():
            if tick.get_text()=='vCAF':tick.set_color('#c34865');tick.set_weight('bold')
    fig.suptitle('Are CD73 and IDO associations restricted to vCAF?',fontsize=16,y=.966)
    fig.text(.5,.916,'All 22 author cell types | Same patient, core and cell type for each molecular comparison',ha='center',fontsize=11,color='#4d6170')
    fig.text(.5,.070,'Each core has ≥15 cells of the indicated type; cores are averaged within patient, then patients equally weighted.',ha='center',fontsize=9.5,color='#4d6170')
    fig.text(.5,.043,'Bars: 95% patient-bootstrap confidence intervals. Red dot and dotted line: vCAF reference. No global cell pooling.',ha='center',fontsize=9.5,color='#4d6170')
    fig.text(.5,.018,'CD73 and IDO were not prediction inputs. Hollow grey points: fewer than 20 patients; limited population support.',ha='center',fontsize=9.5,color='#4d6170')
    output(fig,'molecular_all_celltypes')

    types=[t for t in ORDER if t!='vCAF'];fig,axes=plt.subplots(1,2,figsize=(12.6,10.5));fig.subplots_adjust(left=.22,right=.975,bottom=.13,top=.84,wspace=.25)
    for ax,marker in zip(axes,MARKERS):
        for i,typ in enumerate(types):
            r=matched[matched.population.eq('all_available')&matched.config.eq('RAMP_k30')&matched.matching.eq('same_patient_ROI')&matched.cell_type.eq(typ)&matched.marker.eq(marker)].iloc[0]
            color=('#247f8c' if r['mean']>0 else '#ac6b7a') if r.n_patients>=20 else '#9ea5ad'
            ax.errorbar(r['mean'],i,xerr=[[r['mean']-r.ci_low],[r.ci_high-r['mean']]],fmt='o',ms=4.2,capsize=2.5,color=color,mfc=color if r.n_patients>=20 else 'white')
        ax.set(yticks=np.arange(21),yticklabels=[name(t) for t in types] if marker=='CD73' else [],ylim=(20.6,-.7),title=marker,xlabel='Correlation difference: other type − vCAF')
        ax.axvline(0,c='#8d959c',ls='--');ax.grid(axis='x',alpha=.13)
    fig.suptitle('Comparison with vCAF within shared patients and tissue cores',fontsize=15,y=.959)
    fig.text(.5,.899,'Each paired core contains ≥15 vCAF and ≥15 cells of the other type',ha='center',fontsize=11,color='#4d6170')
    fig.text(.5,.064,'Positive values: stronger association in the other type. Negative values: stronger association in vCAF.',ha='center',fontsize=10,color='#4d6170')
    fig.text(.5,.026,'Patient-bootstrap intervals. Hollow grey points: <20 paired patients. Different rows can have different eligible patients.',ha='center',fontsize=9.2,color='#4d6170')
    output(fig,'molecular_matched_vcaf')

    main=immune[immune.config.eq('RAMP_k30')&immune.analysis.eq('main')]
    vals=main.pivot(index='cell_type',columns='endpoint',values='coef').reindex(index=ORDER,columns=IMMUNE_ENDPOINTS)
    qs=main.pivot(index='cell_type',columns='endpoint',values='q_bh').reindex(index=ORDER,columns=IMMUNE_ENDPOINTS)
    fig,ax=plt.subplots(figsize=(12.9,11));fig.subplots_adjust(left=.235,right=.89,bottom=.17,top=.87)
    limit=max(1.0,float(np.nanmax(np.abs(vals.to_numpy()))));im=ax.imshow(vals,aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit)
    for i in range(len(ORDER)):
        for j in range(len(IMMUNE_ENDPOINTS)):
            v=vals.iloc[i,j];q=qs.iloc[i,j]
            if np.isfinite(v):ax.text(j,i,f'{v:+.2f}'+('*' if q<.05 else ''),ha='center',va='center',fontsize=9,color='white' if abs(v)>.57*limit else '#223343')
            else:ax.text(j,i,'NA',ha='center',va='center',color='#88919a',fontsize=9)
    ax.set(yticks=np.arange(22),yticklabels=[name(t) for t in ORDER],xticks=np.arange(6),xticklabels=['All immune\n/ all cells','CD4\n/ immune','B cells\n/ immune','Neutrophils\n/ immune','CD8\nTCF1','CD4\nTCF1'])
    ax.tick_params(length=0,pad=8)
    for tick in ax.get_yticklabels():
        if tick.get_text()=='vCAF':tick.set_weight('bold');tick.set_color('#c34865')
    for spine in ax.spines.values():spine.set_visible(False)
    cax=fig.add_axes([.925,.31,.018,.40]);cb=fig.colorbar(im,cax=cax);cb.set_label('Adjusted β (outcome SD / HLA score SD)',fontsize=10)
    fig.suptitle('Which cell-type HLA scores track the patient immune environment?',fontsize=15,y=.969)
    fig.text(.5,.922,'Each row uses that cell type’s own continuous mean predicted HLA-ABC',ha='center',fontsize=11,color='#4d6170')
    fig.text(.5,.095,'Original 280-patient cohort; ≥15 cells of the predictor type per patient. TMA, age, sex and stage adjusted; HC3 SE.',ha='center',fontsize=9.5,color='#4d6170')
    fig.text(.5,.060,'* BH q < 0.05 across all tested cell-type/readout combinations. Counts and matched-patient comparisons are in the result tables.',ha='center',fontsize=9.2,color='#4d6170')
    fig.text(.5,.026,'Blue: higher predicted HLA accompanies a lower readout; red: a higher readout. NA: fewer than 40 eligible patients.',ha='center',fontsize=9.7,color='#4d6170')
    output(fig,'patient_immune_celltype_heatmap')

def report(mol,matched,immune):
    lines=['# 其他细胞的高预测 HLA-ABC：分子表型与患者免疫环境','']
    interpretation=BASE/'INTERPRETATION_ZH.md'
    if interpretation.exists():lines += [interpretation.read_text(),'']
    lines += ['## 固定分析口径','',
    '- 完整读取已有投射的 527 位患者、2,849,385 个细胞，覆盖 22 种作者 cell_type。模型与全部原有目录保持只读，未重做预测或生存分组。','- 分子层面：同一患者、同一 ROI、同一细胞类型内至少 15 个细胞，计算预测 HLA-ABC 与实测 CD73／IDO 的 Spearman 相关。ROI 在患者内等权，再对患者等权汇总。','- 高／低细胞状态：该类型在该 ROI 内预测值的上／下三分位；每组至少 5 个细胞。比较实测标记在同类细胞内的百分位秩。与患者生存高／低分组不同。','- vCAF 特异性比较：只使用候选细胞和 vCAF 都足够的共同患者，另以共同 ROI 做配对比较。全部结果同时重复原 DFS 280 人子集及 k100。','- 患者层面：原来的 280 人范围内，对每种细胞使用其自身的患者平均预测 HLA，至少 15 个细胞；连续变量分析，不优化高低阈值。六个固定免疫读数来自上一轮。','- 调整 TMA、年龄、性别、分类分期，HC3 标准误；预测变量和结局分别按患者间标准差标准化。最低 50 个细胞、增加该细胞类型比例、增加 vCAF HLA 作为敏感性；也在相同候选患者中重算 vCAF 关联。','- 分子检验对 22×2 项进行 BH 校正，配对差异对 21×2 项校正，患者模型对所有有足够样本的类型×六个结局校正；不同模型和配置分别处理。','- 原作者 Tumour/normal 是非缺氧肿瘤细胞标签，不是正常肺组织。Other 是混合／未分类细胞，不能赋予统一的生物学身份。','',
    '## 各类型分子关联（k30，全部可用患者）','',
    '| 类型 | CD73：患者数；平均ρ；95% CI | IDO：患者数；平均ρ；95% CI |','|---|---|---|']
    main=mol[mol.population.eq('all_available')&mol.config.eq('RAMP_k30')&mol.metric.eq('rho')]
    for typ in ORDER:
        cells=[]
        for marker in MARKERS:
            r=main[main.cell_type.eq(typ)&main.marker.eq(marker)].iloc[0];cells.append(f'{r.n_patients}；{r["mean"]:+.3f}；[{r.ci_low:+.3f}, {r.ci_high:+.3f}]')
        lines.append('| '+name(typ)+' | '+' | '.join(cells)+' |')
    lines+=['','全部 p、q、方向一致的患者比例和高低三分位对比见 MOLECULAR_SUMMARY.csv；当前表按效应与区间呈现，显著不等于关联强。','',
    '## 与 vCAF 在共同 ROI 内的配对比较','',
    '| 其他类型 | CD73：其他−vCAF 的ρ差；患者数；q | IDO：其他−vCAF 的ρ差；患者数；q |','|---|---|---|']
    paired=matched[matched.population.eq('all_available')&matched.config.eq('RAMP_k30')&matched.matching.eq('same_patient_ROI')]
    for typ in ORDER:
        if typ=='vCAF':continue
        cells=[]
        for marker in MARKERS:
            r=paired[paired.cell_type.eq(typ)&paired.marker.eq(marker)].iloc[0];cells.append(f'{r["mean"]:+.3f}；{r.n_patients}；{r.q_bh:.3g}')
        lines.append('| '+name(typ)+' | '+' | '.join(cells)+' |')
    lines+=['','## 患者免疫环境关联','',
    '每个值为该细胞自身平均预测 HLA 每升高 1 SD，免疫读数的调整变化（SD）；星号为完整检验家族 BH q<0.05。每行可能覆盖不同患者，配对患者和额外调整的全部模型均另存。','',
    '| 类型 | 免疫/全部 | CD4/免疫 | B/免疫 | 中性粒/免疫 | CD8 TCF1 | CD4 TCF1 |','|---|---:|---:|---:|---:|---:|---:|']
    main=immune[immune.config.eq('RAMP_k30')&immune.analysis.eq('main')]
    for typ in ORDER:
        cells=[]
        for endpoint in IMMUNE_ENDPOINTS:
            r=main[main.cell_type.eq(typ)&main.endpoint.eq(endpoint)]
            cells.append('NA' if not len(r) else f'{r.iloc[0].coef:+.3f}'+('*' if r.iloc[0].q_bh<.05 else ''))
        lines.append('| '+name(typ)+' | '+' | '.join(cells)+' |')
    lines+=['','## 文件','',
    '- data/CELL_VALUES.csv.gz：全部细胞 ID、类型、固定 HLA 预测与实测 CD73/IDO。','- results/MOLECULAR_SUMMARY.csv：全部类型、k30/k100、全队列/DFS 子集的分子关联及高低状态对比。','- results/MATCHED_VCAF_COMPARISONS.csv：共同患者、共同 ROI 的配对比较。','- results/PATIENT_CELLTYPE_SCORES.csv：患者自身每类细胞的 HLA 与 CD73/IDO 均值。','- results/PATIENT_IMMUNE_ASSOCIATIONS.csv：连续 HLA 与六项患者免疫读数，以及全部敏感性模型。','- results/VCAF_ON_MATCHED_PATIENTS.csv：候选细胞相同患者范围内的 vCAF 比较。','- results/BETWEEN_PATIENT_MARKER_ASSOCIATIONS.csv：患者间该类细胞平均 HLA 与平均 CD73/IDO 的伴随相关，与细胞内分析层次不同。','- results/CELLTYPE_SCORE_OVERVIEW.csv：每类预测 HLA 得分描述。high_fraction 使用已有 CAF 阈值，仅作描述，不能作为新细胞类型的已验证分类器。','- results/figures：三组 PNG/SVG 图。','- results/*VERIFICATION.json：原始读数、索引、原 vCAF 结果复现和模型核对。','',
    '复现：`zsh run.sh`。已有原始读数导出时自动跳过提取步骤。','']
    (BASE/'README_ZH.md').write_text('\n'.join(lines))

def manifest():
    files=[p for p in BASE.rglob('*') if p.is_file() and not any(part in {'cache','tmp','logs','__pycache__'} for part in p.relative_to(BASE).parts) and p.name!='MANIFEST.json']
    save(OUT/'MANIFEST.json',{'time_utc':now(),'sha256':{str(p.relative_to(BASE)):sha(p) for p in files}})

if __name__=='__main__':
    mol=pd.read_csv(OUT/'MOLECULAR_SUMMARY.csv');matched=pd.read_csv(OUT/'MATCHED_VCAF_COMPARISONS.csv');immune=pd.read_csv(OUT/'PATIENT_IMMUNE_ASSOCIATIONS.csv')
    plots(mol,matched,immune);report(mol,matched,immune)
    figures={}
    for p in FIG.glob('*.png'):
        with Image.open(p) as im:assert np.asarray(im.convert('RGB')).std()>10;figures[p.name]={'size':list(im.size),'nonblank':True}
    save(OUT/'FIGURE_QA.json',{'time_utc':now(),'figures':figures,'visual_review':'pending'})
    manifest();print('Figures and report ready.',flush=True)
