"""Curate fixed step 3–6 results; no alignment, prediction or model fitting."""
from runtime import BASE,STORY
from pathlib import Path
import json,hashlib,datetime,shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

SOURCES={}
COLORS={'Lower':'#588299','Higher':'#cb506a'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def register(rel):
    p=STORY/rel
    assert p.is_file(),p
    if rel not in SOURCES:SOURCES[rel]=sha(p)
    return p
def read(rel):return pd.read_csv(register(rel),float_precision='round_trip')
def save_csv(d,rel):d.to_csv(BASE/rel,index=False)
def copy(src,dest):shutil.copyfile(register(src),BASE/dest)
def save_json(rel,data):(BASE/rel).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
def link(path,label):return f'[{label}](<{path}>)'
def local(rel,label):return link(BASE/rel,label)
def source(rel,label):return link(STORY/rel,label)
def write(rel,text):(BASE/rel).write_text(text.strip()+'\n')
def output(fig,rel):
    fig.savefig(BASE/(rel+'.png'),dpi=220);fig.savefig(BASE/(rel+'.svg'));plt.close(fig)

def curate():
    cov=read('data_cords/results/PATIENT_COVERAGE.csv')
    scores=read('data_cords/results/ALL_PATIENT_SCORES.csv');scores=scores[scores.config.eq('RAMP_k30')]
    assert len(cov)==len(scores)==527 and cov.n_cells.sum()==2849385 and cov.n_vCAF.sum()==52328
    save_csv(cov,'03_projection/PATIENT_COVERAGE.csv');save_csv(scores,'03_projection/PATIENT_VCAF_SCORES_k30.csv')
    copy('data_cords/results/REPRODUCTION.json','03_projection/ORIGINAL_REPRODUCTION.json')
    mol=read('mechanism_support/celltype_specificity/results/MOLECULAR_SUMMARY.csv')
    mol=mol[mol.cell_type.eq('vCAF')&mol.config.eq('RAMP_k30')]
    save_csv(mol,'04_vcaf_markers/MOLECULAR_SUMMARY.csv')
    paired=read('mechanism_support/results/PATIENT_ASSOCIATIONS.csv')
    paired=paired[paired.config.eq('RAMP_k30')&paired.radius.eq(0)&paired.endpoint.isin(['vCAF_CD73','vCAF_IDO'])]
    save_csv(paired,'04_vcaf_markers/PATIENT_PAIRED_MARKERS.csv')
    b=read('mechanism_support/celltype_specificity/results/BETWEEN_PATIENT_MARKER_ASSOCIATIONS.csv')
    save_csv(b[b.cell_type.eq('vCAF')&b.config.eq('RAMP_k30')],'04_vcaf_markers/BETWEEN_PATIENT_MARKER_ASSOCIATIONS.csv')
    immune=read('mechanism_support/patient_immune/results/PATIENT_IMMUNE_COMPOSITION.csv')
    columns=['patient_id','group','high_HLA','all_mean','n_vcaf','all_cells','immune_cells','CD4','CD8','Bcell','Myeloid','Neutrophil','immune_of_all','CD4_of_immune','Bcell_of_immune','Neutrophil_of_immune','CD8_of_immune','Myeloid_of_immune','Treg_of_CD4','TMA','age','sex','Stage']
    im=immune[columns].copy()
    expr=read('mechanism_support/patient_immune/results/PATIENT_LINEAGE_MARKERS.csv')
    expr=expr[expr.lineage.isin(['CD8','CD4'])&expr.marker.eq('TCF1')]
    save_csv(expr[['patient_id','lineage','marker','n_cells','n_rois','mean','median','roi_equal','group']],'05_immune_environment/PATIENT_TCF1.csv')
    for lineage in ['CD8','CD4']:
        a=expr[expr.lineage.eq(lineage)][['patient_id','mean']].rename(columns={'mean':lineage+'_TCF1'})
        im=im.merge(a,on='patient_id',how='left',validate='one_to_one')
    save_csv(im,'05_immune_environment/PATIENT_SUMMARY.csv')
    comp=read('mechanism_support/patient_immune/results/COMPOSITION_COMPARISONS.csv')
    comp_adj=read('mechanism_support/patient_immune/results/COMPOSITION_ADJUSTED.csv')
    save_csv(comp,'05_immune_environment/COMPOSITION_COMPARISONS.csv');save_csv(comp_adj,'05_immune_environment/COMPOSITION_ADJUSTED.csv')
    adj=read('mechanism_support/patient_immune/results/ADJUSTED_COMPARISONS.csv')
    adj=adj[adj.family.eq('primary')&adj.lineage.isin(['CD8','CD4'])&adj.marker.eq('TCF1')]
    save_csv(adj,'05_immune_environment/TCF1_ADJUSTED_AND_SENSITIVITY.csv')
    direct=read('mechanism_support/celltype_specificity/results/PATIENT_IMMUNE_ASSOCIATIONS.csv')
    save_csv(direct[direct.cell_type.eq('vCAF')],'05_immune_environment/CONTINUOUS_VCAF_HLA_SENSITIVITY.csv')
    for ext in ['png','svg']:
        copy('mechanism_support/patient_immune/results/figures/immune_composition.'+ext,'05_immune_environment/immune_composition.'+ext)
    ident='RAMP_k30__mean__all280'
    km=read('survival/results/KM_SUMMARY.csv');km=km[km.id.eq(ident)]
    five=read('survival/results/FIVE_YEAR_DFS.csv');five=five[five.id.eq(ident)]
    cox=read('survival/results/COX_TERMS.csv');cox=cox[cox.config.eq('RAMP_k30')&cox.state.eq('mean')&cox.cohort.eq('all280')&cox.model.eq('matched_full')]
    flow=read('survival/results/COHORT_FLOW.csv');flow=flow[flow.config.eq('RAMP_k30')]
    p=read('survival/results/PATIENT_INPUTS.csv');p=p[p.config.eq('RAMP_k30')]
    for d,rel in [(km,'KM_SUMMARY'),(five,'FIVE_YEAR_DFS'),(cox,'COX_TERMS'),(flow,'COHORT_FLOW'),(p,'PATIENT_INPUTS')]:save_csv(d,'06_DFS/'+rel+'.csv')
    copy(f'survival/results/km/{ident}_curve.csv','06_DFS/KM_CURVE.csv')
    copy(f'survival/results/km/{ident}_patients.csv','06_DFS/KM_PATIENTS.csv')
    copy('survival/results/STATISTICAL_REPRODUCTION.json','06_DFS/ORIGINAL_STATISTICAL_REPRODUCTION.json')
    for src,dest in [('mechanism_support/celltype_specificity/INTERPRETATION_ZH.md','context/CELLTYPE_COMPARISON_ZH.md'),
                     ('mechanism_support/celltype_specificity/results/MOLECULAR_SUMMARY.csv','context/ALL_CELLTYPE_MOLECULAR_SUMMARY.csv'),
                     ('mechanism_support/patient_immune/results/ADJUSTED_COMPARISONS.csv','context/ALL_IMMUNE_EXPRESSION_RESULTS.csv')]:copy(src,dest)
    return cov,scores,mol,paired,im,expr,adj,comp,comp_adj,km.iloc[0],five,cox

def marker_plot(mol,paired):
    fig,axes=plt.subplots(1,2,figsize=(10.8,6.5));fig.subplots_adjust(left=.09,right=.98,bottom=.21,top=.72,wspace=.29)
    for ax,marker in zip(axes,['CD73','IDO']):
        g=paired[paired.endpoint.eq('vCAF_'+marker)].dropna(subset=['low','high'])
        r=mol[mol.population.eq('all_available')&mol.marker.eq(marker)&mol.metric.eq('rho')].iloc[0]
        for a,b in g[['low','high']].to_numpy():ax.plot([0,1],[a,b],c='#82939d',alpha=.12,lw=.65,zorder=1)
        for i,col in enumerate(['low','high']):
            v=g[col].to_numpy();color=COLORS['Lower' if i==0 else 'Higher']
            jitter=np.random.default_rng(20260909).uniform(-.075,.075,len(v))
            ax.scatter(i+jitter,v,s=8,c=color,alpha=.37,lw=0,zorder=2)
            ax.plot([i-.16,i+.16],[v.mean(),v.mean()],color='#273846',lw=2.7,zorder=3)
        ax.set(xticks=[0,1],xticklabels=['Lower predicted HLA\nvCAF third','Higher predicted HLA\nvCAF third'],xlim=(-.3,1.3),ylim=(0,1),ylabel=f'Measured {marker}: within-core rank')
        ax.text(.5,1.18,marker,ha='center',transform=ax.transAxes,fontsize=14)
        ax.text(.5,1.045,f'Mean within-patient ρ = {r["mean"]:.3f}\n95% CI {r.ci_low:.3f}–{r.ci_high:.3f}; {r.n_patients} patients',ha='center',transform=ax.transAxes,fontsize=10)
        ax.grid(axis='y',alpha=.12)
    fig.suptitle('Molecular context of predicted HLA-ABC in vCAF',fontsize=15,y=.967)
    fig.text(.5,.881,'Measured CD73 and IDO | Comparisons within the same patient, core and author-annotated vCAF type',ha='center',fontsize=10,color='#4d6170')
    fig.text(.5,.091,'Each line pairs one patient’s lower / higher vCAF thirds; cores are averaged equally within patient.',ha='center',fontsize=9.5,color='#4d6170')
    fig.text(.5,.048,'Bars: patient-equal means. Correlations use continuous HLA predictions and all eligible vCAF cells.',ha='center',fontsize=9.5,color='#4d6170')
    output(fig,'04_vcaf_markers/vCAF_CD73_IDO')

def tcf_plot(expr,adj):
    fig,axes=plt.subplots(1,2,figsize=(10.8,6.1));fig.subplots_adjust(left=.09,right=.98,bottom=.22,top=.71,wspace=.30)
    for ax,lineage in zip(axes,['CD8','CD4']):
        g=expr[expr.lineage.eq(lineage)]
        r=adj[adj.lineage.eq(lineage)&adj.analysis.eq('all_available_mean')&~adj.full_CAF_covariates].iloc[0]
        for i,group in enumerate(['Lower','Higher']):
            v=g[g.group.eq(group)]['mean'].to_numpy();j=np.random.default_rng(20260909).uniform(-.18,.18,len(v))
            ax.scatter(i+j,v,s=17,c=COLORS[group],alpha=.62,lw=0)
            lo,med,hi=np.quantile(v,[.25,.5,.75]);ax.plot([i-.23,i+.23],[med,med],lw=2.4,c='#273846');ax.plot([i,i],[lo,hi],lw=1.5,c='#273846')
        ns=g.group.value_counts()
        ax.set(xticks=[0,1],xticklabels=[f'Low vCAF HLA\nn={ns["Lower"]}',f'High vCAF HLA\nn={ns["Higher"]}'],xlim=(-.4,1.4),ylim=(0,g['mean'].max()*1.08),ylabel='Patient mean measured TCF1\nasinh(provider intensity)')
        ax.text(.5,1.18,lineage+' T cells',ha='center',transform=ax.transAxes,fontsize=13)
        ax.text(.5,1.045,f'Adjusted high − low: {r.coef:+.2f} SD\nBH q = {r.q_bh:.2g}',ha='center',transform=ax.transAxes,fontsize=10)
        ax.grid(axis='y',alpha=.12)
    fig.suptitle('T-cell TCF1 in the fixed vCAF HLA patient groups',fontsize=15,y=.968)
    fig.text(.5,.878,'All available cells of each T-cell lineage | One dot per patient | Original group labels retained',ha='center',fontsize=10,color='#4d6170')
    fig.text(.5,.091,'Bars: median and interquartile range across patients. Adjusted for TMA, age, sex and stage; HC3 standard errors.',ha='center',fontsize=9.2,color='#4d6170')
    fig.text(.5,.045,'BH q values retain the original family of 10 expression readouts. No distance or nearest-vCAF restriction.',ha='center',fontsize=9.2,color='#4d6170')
    output(fig,'05_immune_environment/CD8_CD4_TCF1')

def km_plot(km,five,cox):
    curve=pd.read_csv(BASE/'06_DFS/KM_CURVE.csv');p=pd.read_csv(BASE/'06_DFS/KM_PATIENTS.csv')
    fig,ax=plt.subplots(figsize=(8.7,7.5));fig.subplots_adjust(left=.15,right=.96,top=.78,bottom=.34)
    for group,color in COLORS.items():
        c=curve[curve.group.eq(group)];x=np.r_[0,c.time];y=np.r_[1,c.survival]
        ax.step(x,y,where='post',c=color,lw=2.2,label=f'{group} vCAF HLA (n=140)')
        ax.fill_between(x,np.r_[1,c.lower],np.r_[1,c.upper],step='post',color=color,alpha=.13,lw=0)
        censored=c[c.n_censor.gt(0)];ax.scatter(censored.time,censored.survival,marker='+',s=19,c=color,linewidths=.75)
    ax.set(xlim=(0,8),ylim=(0,1.04),xticks=[0,2,4,6,8],xlabel='Years',ylabel='Disease-free survival')
    ax.legend(loc='lower left',frameon=False,fontsize=10)
    fs=five.set_index('group');c=cox[cox.term.eq('state_z')].iloc[0]
    ax.text(.04,.21,f'5-year DFS: {fs.loc["Lower","five_year_survival"]:.1%} / {fs.loc["Higher","five_year_survival"]:.1%}\nLog-rank p = {km.logrank_p:.3g}',transform=ax.transAxes,fontsize=11,bbox={'facecolor':'white','edgecolor':'none','alpha':.86})
    ax.text(0,-.23,'Number at risk',transform=ax.transAxes,fontsize=10)
    risk=[]
    for i,(group,color) in enumerate(COLORS.items()):
        ax.text(-.045,-.34-i*.105,group,transform=ax.transAxes,ha='right',color=color,fontsize=10)
        for t in [0,2,4,6,8]:
            n=int((p.group.eq(group)&p.time_years.ge(t)).sum());ax.text(t/8,-.34-i*.105,str(n),transform=ax.transAxes,ha='center',color=color,fontsize=10);risk.append(dict(group=group,time_years=t,n_at_risk=n))
    save_csv(pd.DataFrame(risk),'06_DFS/NUMBER_AT_RISK.csv')
    fig.suptitle('Mean predicted vCAF HLA-ABC and DFS',fontsize=17,y=.965)
    fig.text(.5,.905,'280 patients · 96 DFS events · fixed patient median split',ha='center',fontsize=11,color='#4d6170')
    fig.text(.5,.853,f'Adjusted HR per 1 SD: {c.hr:.2f} (95% CI {c.ci_low:.2f}–{c.ci_high:.2f}); p = {c.p:.3g}',ha='center',fontsize=11,color='#4d6170')
    fig.text(.5,.076,'KM curves are unadjusted. Shading: 95% CI. Plus signs: censoring.',ha='center',fontsize=9.5,color='#4d6170')
    fig.text(.5,.039,'Cox: age, sex, stage, CAF fraction, global CAF score and vCAF/CAF fraction.',ha='center',fontsize=9.2,color='#4d6170')
    output(fig,'06_DFS/vCAF_mean_HLA_DFS')

def documents(mol,im,adj,comp,comp_adj,km,five,cox):
    hr=cox[cox.term.eq('state_z')].iloc[0];fs=five.set_index('group')
    entries=[]
    for marker in ['CD73','IDO']:
        r=mol[mol.population.eq('all_available')&mol.marker.eq(marker)&mol.metric.eq('rho')].iloc[0]
        entries.append(dict(step=4,endpoint='vCAF_'+marker,statistic='patient_equal_mean_within_ROI_Spearman',estimate=r['mean'],ci_low=r.ci_low,ci_high=r.ci_high,q=r.q_bh,n=r.n_patients))
    for endpoint in ['immune_of_all','CD4_of_immune','Bcell_of_immune','Neutrophil_of_immune']:
        a=comp[comp.endpoint.eq(endpoint)].iloc[0];b=comp_adj[comp_adj.endpoint.eq(endpoint)&~comp_adj.full_CAF_covariates].iloc[0]
        entries.append(dict(step=5,endpoint=endpoint,statistic='adjusted_high_minus_low_SD',estimate=b.coef,ci_low=b.ci_low,ci_high=b.ci_high,q=b.q_bh,n=b.n,lower_mean=a.lower_mean,higher_mean=a.higher_mean))
    for lineage in ['CD8','CD4']:
        a=adj[adj.lineage.eq(lineage)&adj.analysis.eq('all_available_mean')&~adj.full_CAF_covariates].iloc[0]
        entries.append(dict(step=5,endpoint=lineage+'_TCF1',statistic='adjusted_high_minus_low_SD',estimate=a.coef,ci_low=a.ci_low,ci_high=a.ci_high,q=a.q_bh,n=a.n))
    entries.append(dict(step=6,endpoint='DFS',statistic='adjusted_HR_per_1SD_mean_vCAF_HLA',estimate=hr.hr,ci_low=hr.ci_low,ci_high=hr.ci_high,p=hr.p,n=280,events=96,lower_DFS5=fs.loc['Lower','five_year_survival'],higher_DFS5=fs.loc['Higher','five_year_survival'],logrank_p=km.logrank_p))
    save_csv(pd.DataFrame(entries),'KEY_RESULTS.csv')
    lines=['# 肺癌 HLA-ABC 故事：第 3–6 步总结','',
    '**主线：将未测量的 HLA-ABC 相对状态迁移到 Cords，聚焦原作者标注的 vCAF，描述其分子表型、患者免疫环境与 DFS 的关联。**','',
    '本文件夹是已有结果的整理与展示，不重做对齐、预测、阈值搜索或统计模型。第 1–2 步（三数据集对齐与 Zheng 验证）仍在 '+source('data_umap','data_umap')+'、'+source('umap','umap')+' 和 '+source('validation','validation')+'。','',
    '## 3. 将固定模型推广到全部 Cords，并定义患者 vCAF 状态','',
    '覆盖 527 位患者、2,849,385 个细胞；其中 52,328 个是作者注释的 vCAF。全部细胞先获得预测，再按作者 vCAF 标签汇总。主指标是每位患者全部 vCAF 的平均预测 HLA-ABC 相对秩，k=30。','',
    '原 28 位 Cords 参考患者使用原单调 profile；其余 499 人各以固定的 1,000 个细胞拟合基础变换，再将固定平台映射应用到该患者全部细胞。新增患者不进入三队列联合拟合。预测源为全部 28,000 个 Senosain 参考细胞。','',
    '完整矩阵和逐细胞预测通过 '+local('03_projection/DATA_INDEX.md','数据索引')+' 访问；本包保存 527 人覆盖表和患者 vCAF 分数。','',
    '## 4. vCAF 内部的分子背景','',
    '在同一患者、同一 core、同一种 vCAF 内，较高预测 HLA-ABC 伴随较高实测 CD73、IDO。每个 core 至少 15 个 vCAF，先对 ROI 等权、再对患者等权；473 位患者可用于主分子分析。','',
    '- HLA–CD73：平均相关 ρ=0.256，95% CI 0.240–0.271。','- HLA–IDO：平均相关 ρ=0.174，95% CI 0.159–0.189。','',
    '下图的高／低是同一 core 内 vCAF 预测值的上／下三分位；线条连接同一患者。它与后面的患者生存分组是两个不同层次。','',
    f'![vCAF CD73 IDO](<{BASE/"04_vcaf_markers/vCAF_CD73_IDO.png"}>)','',
    '## 5. 与患者整体免疫环境相联系','',
    f'沿用原 280 位 DFS 患者及固定 vCAF 均值分组：阈值 {km.median_cut:.15f}，低／高各 140 位。每位患者使用其全部可用组织 core 的相应免疫细胞；统计以患者为单位。','',
    '| 患者平均比例 | 低 HLA 组 | 高 HLA 组 | 调整后 BH q |','|---|---:|---:|---:|']
    labels={'immune_of_all':'免疫细胞／全部细胞','CD4_of_immune':'CD4／免疫细胞','Bcell_of_immune':'B 细胞／免疫细胞','Neutrophil_of_immune':'中性粒细胞／免疫细胞'}
    for endpoint,label in labels.items():
        a=comp[comp.endpoint.eq(endpoint)].iloc[0];b=comp_adj[comp_adj.endpoint.eq(endpoint)&~comp_adj.full_CAF_covariates].iloc[0]
        lines.append(f'| {label} | {a.lower_mean:.2%} | {a.higher_mean:.2%} | {b.q_bh:.3g} |')
    lines += ['',
    '中性粒细胞是免疫细胞内部的相对占比上升；其占全部细胞的比例没有明确调整后差异。','',
    '高 vCAF HLA 组的 CD8 与 CD4 实测 TCF1 也较低，调整差异分别为 −0.698、−0.743 SD；相应 q=9.50e−8、4.07e−8。CD8 可用 280 人，CD4 可用 275 人；另外 5 人没有作者 CD4 细胞，不插补表达值。','',
    f'![immune composition](<{BASE/"05_immune_environment/immune_composition.png"}>)','',
    f'![CD8 CD4 TCF1](<{BASE/"05_immune_environment/CD8_CD4_TCF1.png"}>)','',
    '免疫模型调整 TMA、年龄、性别、分期，采用 HC3 标准误；比例 q 保留原 12 项检验家族，表达 q 保留原 10 项检验家族。最低细胞数、ROI 等权、CAF 协变量和连续 vCAF HLA 的敏感性表均已保留。','',
    '## 6. 与 DFS 联系','',
    '从 527 人按既定条件纳入：作者分期组 I–II、至少 15 个 vCAF、至少 50 个 CAF、有效 DFS 时间与事件，以及所需临床协变量完整。逐步人数见 06_DFS/COHORT_FLOW.csv。','',
    f'同一 280 位患者共有 96 个 DFS 事件。低／高 HLA 组的五年 DFS 为 **{fs.loc["Lower","five_year_survival"]:.1%}／{fs.loc["Higher","five_year_survival"]:.1%}**，log-rank p={km.logrank_p:.3g}。','',
    f'连续 vCAF 均值每增加 1 SD，调整 HR={hr.hr:.3f}（95% CI {hr.ci_low:.3f}–{hr.ci_high:.3f}），p={hr.p:.4g}。Cox 调整年龄、性别、作者 Stage 分类、原 CAF 比例、原 global-CAF 分数以及 vCAF/全部 CAF 比例。','',
    '本页只展示最终采用的“患者 vCAF 预测均值”指标。曲线、区间、删失点、风险人数和统计数字均读取已有固定结果。','',
    f'![vCAF DFS](<{BASE/"06_DFS/vCAF_mean_HLA_DFS.png"}>)','',
    '## 讲述时保留的层次','',
    '第 4 步是同一患者内部的细胞共变；第 5、6 步是患者之间的关联。高 HLA 患者的 vCAF 平均 CD73/IDO 并没有相应显著升高，不能将细胞内相关改写成患者间均值差异。','',
    'CD73/IDO 的共变及部分免疫环境关联也存在于其他细胞类型；vCAF 在 CAF 亚型中的关联图景较完整，可继续作为当前故事的观察对象。当前证据不支持 vCAF 独有机制或已证实的 CD73/IDO→免疫变化→复发因果链。PD-1 没有出现预期升高，也不将 TCF1 较低直接命名为已证明的耗竭。完整比较留在 '+local('context/CELLTYPE_COMPARISON_ZH.md','背景核查')+'。','',
    '## 文件入口','',
    '- '+local('KEY_RESULTS.csv','KEY_RESULTS.csv')+'：九项主要数值及统计口径。',
    '- '+local('03_projection/DATA_INDEX.md','第 3 步数据索引')+'：全部预测、矩阵和行对应。',
    '- '+local('SOURCE_SCRIPTS.md','原始脚本索引')+'：各分析的完整实现位置。',
    '- '+local('FIGURE_GUIDE_ZH.md','图与图注说明')+'：四张展示图及阅读方式。',
    '- '+local('provenance/SOURCE_INDEX.csv','源文件索引')+'、'+local('provenance/VERIFICATION.json','核对记录')+'：内容来源与原结果一致性。',
    '- '+local('scripts/build_summary.py','build_summary.py')+'：重建本总结包的脚本，仅写入本文件夹。','',
    '运行：`/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -B scripts/build_summary.py`。重绘使用现有统计结果，不重新拟合。','']
    write('README_ZH.md','\n'.join(lines))
    large=[('data_cords/data/CELL_METADATA.csv.gz','全部细胞 ID、作者注释、患者、ROI、真实坐标与 matrix_row_index'),('data_cords/data/RAW_SHARED9.npy','原始九标记矩阵'),('data_cords/data/RAMP_SHARED9.npy','固定变换后的完整九标记矩阵'),('data_cords/data/MARKERS.csv','矩阵的九标记列顺序'),('data_cords/results/ALL_CORDS_PREDICTIONS.csv.gz','全部细胞的固定预测'),('data_cords/results/patients','逐患者计算检查点'),('mechanism_support/results/VCAF_MEASURED_AND_PREDICTED.csv.gz','全部 vCAF 的实测分子与固定预测')]
    for rel,_ in large:assert (STORY/rel).exists()
    write('03_projection/DATA_INDEX.md','# 第 3 步完整数据索引\n\n本总结包不重复复制表达矩阵和全量逐细胞表。以下均为只读来源。\n\n'+'\n'.join('- '+source(rel,rel)+'：'+purpose for rel,purpose in large)+'\n\n矩阵按 matrix_row_index 连接；不同表用 source_row_id 精确连接，不能假设物理行顺序一致。source_row_id 对应原 R 对象的 1 基索引。\n\n患者主指标是全部作者 vCAF 的预测相对秩均值。没有 T 邻域或 SMA-CAF 纳入条件。')
    script_refs=[('data_cords/scripts','第 3 步：固定参考、全患者投射和预测'),('mechanism_support/scripts','第 4 步：原始分子导出、vCAF 内部关联'),('mechanism_support/patient_immune/scripts','第 5 步：免疫组成与整患者表达'),('mechanism_support/celltype_specificity/scripts','连续 vCAF HLA、敏感性和其他类型核查'),('survival/scripts','第 6 步：队列、固定 KM/Cox 和报告')]
    write('SOURCE_SCRIPTS.md','# 原始实现索引\n\n下列目录保持原状。本总结的构建脚本不运行或修改这些原始脚本。\n\n'+'\n'.join('- '+source(rel,rel)+'：'+purpose for rel,purpose in script_refs)+'\n\n总结重建脚本：'+local('scripts/build_summary.py','build_summary.py')+'。')
    write('FIGURE_GUIDE_ZH.md',f'''# 第 3–6 步展示图

第 3 步提供预测与患者分数，是下面各图的共同输入。本包保留四张独立展示图，每张都有 PNG 和可编辑 SVG。

1. {local('04_vcaf_markers/vCAF_CD73_IDO.png','vCAF CD73 / IDO')}：同一患者内较低／较高预测 HLA 的 vCAF 三分位，纵轴为实测标记在同类细胞内的秩。每条线一位患者，473 人；标题相关系数使用连续预测值。
2. {local('05_immune_environment/immune_composition.png','患者免疫组成')}：原 vCAF 平均预测 HLA 的 140／140 患者分组。左侧为患者等权比例均值，右侧为调整效应；分母逐行注明。
3. {local('05_immune_environment/CD8_CD4_TCF1.png','CD8 / CD4 TCF1')}：沿用同一患者分组，全部可用 CD8 或 CD4 的平均实测 TCF1；一人一个点。CD4 缺失的 5 位患者不插补。
4. {local('06_DFS/vCAF_mean_HLA_DFS.png','平均预测 vCAF HLA 与 DFS')}：固定 280 人、96 事件，按患者均值中位数分组，含 95% CI、删失标记和风险人数；图顶端注明连续指标的调整 HR。

数值、选择和模型均未改变。第 1 张图的细胞高低三分位，与第 2–4 张图的患者高低分组分开解释。免疫图为患者层面汇总，不是空间邻域图。
''')

def verify(im,km,five):
    patients=pd.read_csv(BASE/'06_DFS/KM_PATIENTS.csv');p=im.merge(patients,on='patient_id',validate='one_to_one',suffixes=('_immune','_DFS'))
    assert len(p)==280 and (p.group_immune==p.group_DFS).all() and patients.event.sum()==96
    assert np.array_equal(im.group.eq('Higher'),im.all_mean.ge(km.median_cut))
    curve=pd.read_csv(BASE/'06_DFS/KM_CURVE.csv')
    for _,r in five.iterrows():
        values=curve[curve.group.eq(r.group)&curve.time.le(5)]
        assert abs(values.iloc[-1].survival-r.five_year_survival)<1e-12
    assert all(sha(STORY/rel)==digest for rel,digest in SOURCES.items())
    figures={}
    for path in BASE.rglob('*.png'):
        with Image.open(path) as img:
            assert np.asarray(img.convert('RGB')).std()>10;figures[str(path.relative_to(BASE))]={'size':list(img.size),'nonblank':True}
    rows=[dict(source_relative=rel,source_absolute=str(STORY/rel),sha256=digest) for rel,digest in SOURCES.items()]
    save_csv(pd.DataFrame(rows),'provenance/SOURCE_INDEX.csv')
    save_json('provenance/VERIFICATION.json',{'status':'PASS','time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'same_original_280_patients_and_140_140_groups':True,'same_96_DFS_events':True,'five_year_probabilities_equal_saved_curve':True,'source_hashes_unchanged':True,'models_refit':False,'predictions_refit':False,'figures':figures,'visual_review':'pending'})
    manifest()

def manifest():
    files=[p for p in BASE.rglob('*') if p.is_file() and not any(part in {'cache','tmp','logs','__pycache__'} for part in p.relative_to(BASE).parts) and p.name!='MANIFEST.json']
    save_json('provenance/MANIFEST.json',{'sha256':{str(p.relative_to(BASE)):sha(p) for p in files}})

if __name__=='__main__':
    cov,scores,mol,paired,im,expr,adj,comp,comp_adj,km,five,cox=curate()
    marker_plot(mol,paired);tcf_plot(expr,adj);km_plot(km,five,cox)
    documents(mol,im,adj,comp,comp_adj,km,five,cox);verify(im,km,five)
    print('Summary package created and numerical consistency verified.',flush=True)
