from runtime import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
COLORS={'Lower':'#588299','Higher':'#cb506a'}
LABELS={'CD8':'CD8 T','CD4':'CD4 T','Bcell':'B cells','Myeloid':'Myeloid','Neutrophil':'Neutrophils'}
M_LABELS={'PD1':'PD-1','TCF1':'TCF1','FOXP3':'FOXP3','IDO':'IDO','CD73':'CD73'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})

def output(fig,name):
    fig.savefig(FIG/(name+'.png'),dpi=220);fig.savefig(FIG/(name+'.svg'));plt.close(fig)

def plots(summary,adj,comp,compadj):
    fig,ax=plt.subplots(figsize=(12.5,7.8));fig.subplots_adjust(left=.23,right=.69,bottom=.19,top=.82)
    labels=[]
    for i,(lineage,marker) in enumerate(PRIMARY):
        g=adj[(adj.family=='primary')&adj.lineage.eq(lineage)&adj.marker.eq(marker)&~adj.full_CAF_covariates]
        main=g[g.analysis.eq('all_available_mean')].iloc[0]
        sen=g[g.analysis.eq('min20_mean')]
        ax.errorbar(main.coef,i-.10,xerr=[[main.coef-main.ci_low],[main.ci_high-main.coef]],fmt='o',color='#187f8b',capsize=3,ms=6)
        if len(sen):
            r=sen.iloc[0];ax.errorbar(r.coef,i+.13,xerr=[[r.coef-r.ci_low],[r.ci_high-r.coef]],fmt='s',color='#989eaa',capsize=2,ms=4)
        labels.append(LABELS[lineage]+'  /  '+M_LABELS[marker])
        ax.text(1.04,i,f'{int(main.n_lower)} / {int(main.n_higher)}',transform=ax.get_yaxis_transform(),va='center',fontsize=9.5)
        ax.text(1.38,i,f'{main.q_bh:.2g}',transform=ax.get_yaxis_transform(),va='center',fontsize=9.5)
    ax.set(yticks=np.arange(len(PRIMARY)),yticklabels=labels,xlabel='Adjusted high − low expression difference (SD)')
    ax.set_ylim(len(PRIMARY)-.5,-.9);ax.axvline(0,c='#a3a6ab',ls='--',lw=1);ax.grid(axis='x',alpha=.13)
    ax.text(1.04,-.8,'Patients\nlow / high',transform=ax.get_yaxis_transform(),va='bottom',fontsize=9.5)
    ax.text(1.38,-.8,'Adjusted\nBH q',transform=ax.get_yaxis_transform(),va='bottom',fontsize=9.5)
    handles=[Line2D([0],[0],marker='o',color='#187f8b',label='All patients with ≥1 cell in lineage'),Line2D([0],[0],marker='s',color='#989eaa',label='Sensitivity: ≥20 cells in lineage')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.082),ncol=2,frameon=False)
    fig.suptitle('Immune expression across the fixed vCAF HLA survival groups',y=.972,fontsize=15)
    fig.text(.5,.907,'Measured proteins | Original 280 patients | Separate summaries within each immune lineage',ha='center',fontsize=11,color='#4e6070')
    fig.text(.5,.049,'Models: TMA, age, sex and stage; HC3 95% confidence intervals. BH correction across all 10 primary readouts.',ha='center',fontsize=9.3,color='#4e6070')
    fig.text(.5,.022,'Negative values indicate lower expression in patients with higher predicted vCAF HLA-ABC.',ha='center',fontsize=10,color='#4e6070')
    output(fig,'immune_expression_overview')

    endpoints=[(lineage+'_of_immune',LABELS[lineage]+' / immune') for lineage in LINEAGES]
    endpoints += [('immune_of_all','All immune / all cells'),('Treg_of_CD4','Author Treg / CD4')]
    fig,(left,right)=plt.subplots(1,2,figsize=(12.2,6.6));fig.subplots_adjust(left=.215,right=.92,bottom=.27,top=.80,wspace=.54)
    for i,(key,label) in enumerate(endpoints):
        a=comp[comp.endpoint.eq(key)].iloc[0];b=compadj[compadj.endpoint.eq(key)&~compadj.full_CAF_covariates].iloc[0]
        left.barh(i-.14,a.lower_mean*100,height=.26,color=COLORS['Lower'])
        left.barh(i+.14,a.higher_mean*100,height=.26,color=COLORS['Higher'])
        right.errorbar(b.coef,i,xerr=[[b.coef-b.ci_low],[b.ci_high-b.coef]],fmt='o',color='#187f8b',capsize=3)
        right.text(1.05,i,f'q={b.q_bh:.2g}',transform=right.get_yaxis_transform(),va='center',fontsize=9)
    left.set(yticks=np.arange(7),yticklabels=[x[1] for x in endpoints],xlabel='Mean patient fraction (%)',title='Cell composition')
    right.set(yticks=np.arange(7),yticklabels=[],xlabel='Adjusted high − low difference (SD)',title='Patient group association')
    for ax in (left,right):ax.invert_yaxis();ax.grid(axis='x',alpha=.12)
    right.axvline(0,c='#a3a6ab',ls='--',lw=1)
    fig.suptitle('Immune composition in the original 140 low / 140 high patient groups',y=.955,fontsize=14)
    fig.legend(handles=[Line2D([0],[0],color=COLORS[group],lw=8,label=group+' vCAF HLA') for group in ['Lower','Higher']],loc='lower center',bbox_to_anchor=(.5,.11),ncol=2,frameon=False)
    fig.text(.5,.052,'Fractions are calculated per patient; denominators are named explicitly. Absent lineages contribute zero.',ha='center',fontsize=9.6,color='#4e6070')
    fig.text(.5,.022,'Adjustment: TMA, age, sex and stage. BH includes all 12 composition tests (including five all-cell fractions).',ha='center',fontsize=9.2,color='#4e6070')
    output(fig,'immune_composition')

    fig,axes=plt.subplots(1,3,figsize=(14.5,5.7));fig.subplots_adjust(left=.065,right=.98,bottom=.23,top=.71,wspace=.34)
    for ax,marker in zip(axes,['PD1','TCF1','FOXP3']):
        g=summary[summary.lineage.eq('CD4')&summary.marker.eq(marker)]
        r=adj[(adj.family=='primary')&adj.lineage.eq('CD4')&adj.marker.eq(marker)&adj.analysis.eq('all_available_mean')&~adj.full_CAF_covariates].iloc[0]
        for i,group in enumerate(['Lower','Higher']):
            v=g[g.group.eq(group)]['mean'].to_numpy()
            jitter=np.random.default_rng(SEED).uniform(-.18,.18,len(v))
            ax.scatter(i+jitter,v,s=16,c=COLORS[group],alpha=.6,lw=0)
            lo,med,hi=np.quantile(v,[.25,.5,.75]);ax.plot([i-.23,i+.23],[med,med],lw=2.3,color='#283c4c');ax.plot([i,i],[lo,hi],lw=1.6,color='#283c4c')
        ax.set(xticks=[0,1],xticklabels=[f'Low vCAF HLA\nn={r.n_lower}',f'High vCAF HLA\nn={r.n_higher}'],xlim=(-.4,1.4),ylim=(0,g['mean'].max()*1.08),ylabel='Patient mean asinh(intensity)')
        ax.grid(axis='y',alpha=.13)
        ax.text(.5,1.18,f'Measured CD4 {M_LABELS[marker]}',ha='center',transform=ax.transAxes,fontsize=12)
        ax.text(.5,1.045,f'Adjusted difference: {r.coef:+.2f} SD\nBH q = {r.q_bh:.2g}',ha='center',transform=ax.transAxes,fontsize=10)
    fig.suptitle('Whole-patient CD4 T-cell expression',fontsize=16,y=.965)
    fig.text(.5,.875,'All author-annotated CD4 cells, including Treg | Same frozen patient groups',ha='center',fontsize=11,color='#4e6070')
    fig.text(.5,.095,'One dot per patient. Bars: median and interquartile range across patients. No spatial proximity restriction.',ha='center',fontsize=10,color='#4e6070')
    fig.text(.5,.048,'TMA, age, sex and stage adjusted; BH across 10 primary readouts. Conventional CD4 and Treg are also analyzed separately.',ha='center',fontsize=9.3,color='#4e6070')
    output(fig,'whole_patient_CD4_groups')

def report(summary,adj,comp,compadj):
    raw=pd.read_csv(OUT/'GROUP_COMPARISONS.csv')
    coverage=pd.read_csv(OUT/'LINEAGE_COVERAGE.csv')
    lines=['# 固定 vCAF HLA 生存分组下的其他免疫细胞','',
    '本轮比较原来 280 位患者（高／低各 140 位）的全部可用免疫细胞，区分细胞组成与谱系内实测表达。不使用最近 vCAF 归属，不改变预测模型、患者筛选或分组阈值。','',
    '## 本轮结论','',
    '**PD-1 未升高并不只是因为此前仅看 CD8。** CD4 的 PD-1 也较低（调整差异 −0.339 SD，q=0.0305），作者 Treg 内 PD-1 未见明确差异（q=0.604）。两类 T 细胞的 PD-1 差异对最低细胞数要求较敏感，不适合作为主要机制证据。','',
    '更清楚的表达结果是 **CD4 和 CD8 的 TCF1 都较低**：CD4 −0.743 SD（q=4.07e−8），CD8 −0.698 SD（q=9.50e−8）。CD4 在至少 20／50 个细胞、ROI 等权和进一步调整 CAF 协变量后均保持同方向且 q<0.05；移除作者 Treg 后，常规 CD4 的差异仍为 −0.728 SD（次要终点 q=3.37e−8）。因此这个结果不只是 Treg 与常规 CD4 混合比例造成。','',
    '细胞组成提供了另一条较清楚的线索：高 HLA 组中，所有免疫细胞占全部细胞的患者平均比例从 21.75% 降至 13.89%（调整 q=2.60e−5）；CD4 占免疫细胞的比例 11.11%→6.66%，B 细胞 10.25%→6.54%。中性粒细胞占免疫细胞的比例 8.27%→18.37%（调整 q=1.28e−4），但其占全部细胞的比例并无明确调整后差异（q=0.187）。所以应称“免疫组成中相对偏向中性粒细胞”，不能称已证明中性粒细胞绝对浸润增加。','',
    'CD4 的 FOXP3 及作者 Treg/CD4 比例均无明确调整后差异；髓系 IDO 和中性粒细胞 IDO/CD73 未见明确升高。髓系与 B 细胞 CD73 的主分析方向为降低，且并非所有敏感性分析都显著。这些结果不支持把故事写成广泛的抑制性标记上调。','',
    '适合继续发展的关联叙事：**预测 vCAF HLA-ABC 高的患者，同时呈现较低的整体免疫细胞比例、较低的 CD4/CD8 TCF1 状态，以及中性粒细胞相对占比更高的免疫组成，并且具有较差 DFS。** 这里的“同时呈现”不表示已经确定因果先后，也没有证明免疫细胞被排斥在肿瘤边界外。','',
    '## 分析口径','',
    '- 使用 Cords 原作者细胞注释，共 283,289 个免疫细胞；这些是患者可用组织芯片 core 中的细胞，不是整块肿瘤所有细胞。','- 表达：逐细胞 asinh(provider counts)，再对患者同一谱系全部细胞取均值。统计以患者为单位；无该谱系细胞的患者不插补表达值。','- Treg 采用原作者 CD4_Treg 注释；常规 CD4 是其余作者 CD4 细胞。Treg 是 CD4 子集，不能与全 CD4 作为互斥类别相加。','- 主调整模型包含 TMA、年龄、性别与分类分期，采用 HC3 稳健标准误。效应为高组减低组，以患者间标准差 SD 为单位。','- 10 个主要表达终点整体做 BH；10 个次要终点单独做 BH；12 个组成终点单独做 BH。CD8 效应与上一轮相同，q 因校正的终点数从 2 增至 10 而变化。','- 保留至少 20／50 个该谱系细胞、ROI 等权、进一步调整原 CAF 协变量及连续 HLA 分数作为敏感性／伴随分析。','',
    '## 主要表达结果','',
    '| 细胞 | 标记 | 低／高患者数 | 低／高均值（asinh） | 调整差异 SD（95% CI） | BH q |','|---|---|---|---|---|---:|']
    main=adj[adj.family.eq('primary')&adj.analysis.eq('all_available_mean')&~adj.full_CAF_covariates]
    for lineage,marker in PRIMARY:
        a=main[main.lineage.eq(lineage)&main.marker.eq(marker)].iloc[0]
        b=raw[raw.family.eq('primary')&raw.analysis.eq('all_available_mean')&raw.lineage.eq(lineage)&raw.marker.eq(marker)].iloc[0]
        lines.append(f'| {lineage} | {marker} | {a.n_lower}/{a.n_higher} | {b.lower_mean:.4f}/{b.higher_mean:.4f} | {a.coef:+.3f} ({a.ci_low:+.3f}, {a.ci_high:+.3f}) | {a.q_bh:.3g} |')
    lines += ['','## 细胞组成','',
    '下表的百分比是先在每位患者计算比例，再对患者等权取平均；不是合并所有细胞后的比例。表达缺失和细胞数量为零是不同情形：比例分析保留零。','',
    '| 终点 | 低组平均比例 | 高组平均比例 | 调整差异 SD（95% CI） | BH q |','|---|---:|---:|---|---:|']
    for _,a in compadj[~compadj.full_CAF_covariates].iterrows():
        b=comp[comp.endpoint.eq(a.endpoint)].iloc[0]
        lines.append(f'| {a.endpoint} | {100*b.lower_mean:.2f}% | {100*b.higher_mean:.2f}% | {a.coef:+.3f} ({a.ci_low:+.3f}, {a.ci_high:+.3f}) | {a.q_bh:.3g} |')
    lines += ['','## CD4 亚群与其他次要读数','',
    'HLA-DR 是原 HLA 预测输入，只作描述，不能作为独立验证。MPO／MMP9 是中性粒细胞的附加实测读数。下表使用单独的次要终点 BH 校正。','',
    '| 细胞 | 标记 | 低／高患者数 | 调整差异 SD（95% CI） | BH q |','|---|---|---|---|---:|']
    for _,a in adj[adj.family.eq('secondary')&adj.analysis.eq('all_available_mean')&~adj.full_CAF_covariates].iterrows():
        lines.append(f'| {a.lineage} | {a.marker} | {a.n_lower}/{a.n_higher} | {a.coef:+.3f} ({a.ci_low:+.3f}, {a.ci_high:+.3f}) | {a.q_bh:.3g} |')
    lines += ['','## 细胞覆盖','',
    '| 谱系 | 分组 | 患者数 | 细胞数 | 每人细胞中位数 | ≥20 细胞患者数 | ≥50 细胞患者数 |','|---|---|---:|---:|---:|---:|---:|']
    for _,a in coverage.iterrows():lines.append(f'| {a.lineage} | {a.group} | {a.n_patients} | {a.n_cells} | {a.median_cells:.0f} | {a.n_at_least20} | {a.n_at_least50} |')
    lines += ['','## 解释','',
    '本轮针对“只看 CD8 是否遗漏其他免疫状态”的探索性扩展；方向由数据决定。PD-1 升高并不是所有免疫细胞或所有不良预后的共同要求。PD-1／TCF1 按 T 细胞谱系解释，不能把其在不同谱系中的强度混在一起当作统一耗竭指标。CD73／IDO 表达差异可以作为候选免疫调节状态，但不能由此证明功能性抑制。','',
    '高组中某类细胞在免疫细胞内占比上升，不等于该类细胞在所有细胞内也增加；两种分母均已保留。CD68 不足以定义 M2 巨噬细胞，这里保留原作者 Myeloid 名称。现有作者注释没有单独 NK 类，不能凭这些标签补出 NK 结论。','',
    '所有当前结果来自与预测 vCAF HLA 生存分析相同的患者，因此是关联层面的补充，不能代替独立数据或局部功能实验。','',
    '相关原始研究：[Cords 等人的肺癌 CAF 原文](https://pmc.ncbi.nlm.nih.gov/articles/10929690/)研究了 CAF 与多类免疫细胞的关系；[TRACERx 多重成像研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC11145179/)区分了免疫细胞较少、淋巴细胞稀少而中性粒细胞较多等微环境。这些工作支持同时审视多个免疫谱系，但不直接验证我们的 vCAF HLA 关联。','',
    '## 文件与复现','',
    '- results/PATIENT_LINEAGE_MARKERS.csv：患者 × 谱系 × 标记汇总与固定临床分组。','- results/PATIENT_IMMUNE_COMPOSITION.csv：每患者细胞数及各分母比例。','- results/ADJUSTED_COMPARISONS.csv：全部主要／次要表达模型与敏感性。','- results/COMPOSITION_ADJUSTED.csv：全部 12 项组成比较及 CAF 协变量扩展。','- results/CONTINUOUS_ASSOCIATIONS.csv：连续预测 HLA 的伴随关联。','- results/PREVIOUS_CD8_REPRODUCTION.csv：此前 CD8 模型数值复现。','- results/figures/immune_expression_overview.png / .svg：各免疫谱系实测表达与最低细胞数敏感性。','- results/figures/immune_composition.png / .svg：患者免疫组成。','- results/figures/whole_patient_CD4_groups.png / .svg：CD4 的 PD-1、TCF1、FOXP3 散点图。','- results/DESIGN.json / EXPORT_VERIFICATION.json / VERIFICATION.json：方案、源数据索引与数值核对。','',
    '复现命令见 run.sh；数据提取已完成时仅重新执行 analyze.py、report.py 即可。所有新文件都限制在本 patient_immune 子目录，原始数据、模型、生存分析和此前 CD8／邻域分析保持只读。','']
    (BASE/'README_ZH.md').write_text('\n'.join(lines))

def manifest():
    files=[p for p in BASE.rglob('*') if p.is_file() and not any(part in {'cache','tmp','logs','__pycache__'} for part in p.relative_to(BASE).parts) and p.name!='MANIFEST.json']
    save(OUT/'MANIFEST.json',{'time_utc':now(),'sha256':{str(p.relative_to(BASE)):sha(p) for p in files}})

if __name__=='__main__':
    summary=pd.read_csv(OUT/'PATIENT_LINEAGE_MARKERS.csv');adj=pd.read_csv(OUT/'ADJUSTED_COMPARISONS.csv');comp=pd.read_csv(OUT/'COMPOSITION_COMPARISONS.csv');compadj=pd.read_csv(OUT/'COMPOSITION_ADJUSTED.csv')
    plots(summary,adj,comp,compadj);report(summary,adj,comp,compadj)
    figures={}
    for p in FIG.glob('*.png'):
        with Image.open(p) as im:
            assert np.asarray(im.convert('RGB')).std()>10
            figures[p.name]={'size':list(im.size),'nonblank':True}
    save(OUT/'FIGURE_QA.json',{'time_utc':now(),'figures':figures,'visual_review':'pending'})
    manifest();print('Figures and Chinese report generated.',flush=True)
