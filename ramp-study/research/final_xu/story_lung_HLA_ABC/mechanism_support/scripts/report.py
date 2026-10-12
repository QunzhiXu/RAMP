"""Report molecular support and the weak/absent neighboring T-cell findings."""
import runtime as rt
from runtime import BASE
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import rankdata, spearmanr
from matplotlib.lines import Line2D

OUT=BASE/'results';FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
LOW='#588299';HIGH='#cb506a';FULL='#167d8d';DFS='#bd7551'
PRIMARY=['vCAF_CD73','vCAF_IDO','nearby_CD8_PD1','nearby_CD8_TCF1']
LABELS={'vCAF_CD73':'vCAF: measured CD73','vCAF_IDO':'vCAF: measured IDO','nearby_CD8_PD1':'Nearby CD8: measured PD-1','nearby_CD8_TCF1':'Nearby CD8: measured TCF1'}

def savefig(fig,name):
    fig.savefig(FIG/(name+'.png'),dpi=230,facecolor='white')
    fig.savefig(FIG/(name+'.svg'),facecolor='white');plt.close(fig)

def getrow(s,endpoint,metric='rho',pop='all_available',config='RAMP_k30',radius=None):
    if radius is None:radius=0 if endpoint.startswith('vCAF') else 40
    g=s[(s.endpoint==endpoint)&(s.metric==metric)&(s.population==pop)&(s.config==config)&(s.radius==radius)]
    assert len(g)==1;return g.iloc[0]

def overview(s,p):
    fig,axs=plt.subplots(1,3,figsize=(16.5,6.3),gridspec_kw={'width_ratios':[1,1,1.7]})
    fig.subplots_adjust(left=.045,right=.98,bottom=.20,top=.77,wspace=.38)
    for ax,marker in zip(axs[:2],['CD73','IDO']):
        ep='vCAF_'+marker;r=getrow(s,ep,'difference')
        g=p[(p.config=='RAMP_k30')&(p.endpoint==ep)].dropna(subset=['difference'])
        order=np.argsort(g.difference.to_numpy())
        for _,row in g.iloc[order].iterrows():
            ax.plot([0,1],[row.low,row.high],c='#a8b5bf',alpha=.17,lw=.55,zorder=1)
        ax.scatter(np.zeros(len(g)),g.low,c=LOW,s=8,alpha=.22,lw=0)
        ax.scatter(np.ones(len(g)),g.high,c=HIGH,s=8,alpha=.22,lw=0)
        ax.plot([0,1],[r.low,r.high],'-o',color='#253746',lw=2.5,ms=8,zorder=4)
        ax.set(xticks=[0,1],xticklabels=['Low HLA\nthird','High HLA\nthird'],ylabel=f'Measured {marker}: within-ROI vCAF rank',ylim=(0,1),xlim=(-.22,1.22),title=f'{marker} | {int(r.n_patients)} patients',yticks=[0,.2,.4,.6,.8,1])
        ax.grid(axis='y',alpha=.13)
        ax.text(.5,.99,f'Mean difference = +{r["mean"]:.3f}\n95% CI {r.ci_low:.3f} to {r.ci_high:.3f}',transform=ax.transAxes,ha='center',va='top',fontsize=10)
    ax=axs[2]
    for j,ep in enumerate(PRIMARY):
        for pop,offset,color,marker in [('all_available',-.12,FULL,'o'),('DFS_280_members',.12,DFS,'s')]:
            r=getrow(s,ep,pop=pop);y=j+offset
            ax.errorbar(r['mean'],y,xerr=[[r['mean']-r.ci_low],[r.ci_high-r['mean']]],fmt=marker,color=color,capsize=3,ms=5)
    ax.set(yticks=np.arange(4),yticklabels=['vCAF: CD73','vCAF: IDO','CD8: PD-1','CD8: TCF1'],xlim=(-.04,.32),xlabel='Mean within-ROI Spearman\n(equal weight per patient)',title='Molecular and neighboring T-cell evidence')
    ax.invert_yaxis();ax.axvline(0,color='#a8b1bb',ls='--',lw=1);ax.grid(axis='x',alpha=.12)
    ax.legend(handles=[Line2D([],[],color=FULL,marker='o',ls='',label='All eligible patients'),Line2D([],[],color=DFS,marker='s',ls='',label='Fixed DFS cohort members')],loc='lower right',frameon=False,fontsize=9)
    for i,ax in enumerate(axs):ax.text(-.08,1.15,'ABC'[i],transform=ax.transAxes,fontsize=15,weight='bold')
    fig.suptitle('Measured CD73 and IDO accompany higher predicted HLA-ABC within vCAF',fontsize=17,y=.965)
    fig.text(.5,.875,'Original Cords vCAF annotations | Same patient and ROI comparisons | Frozen all-source k = 30',ha='center',fontsize=11,color='#4d5c69')
    fig.text(.5,.068,'A–B: one line per patient; mean ROI contrasts between the lower and upper HLA-score thirds. Thick line: cohort mean.',ha='center',fontsize=10,color='#4d5c69')
    fig.text(.5,.032,'C: CD8 cells assigned once to their nearest vCAF within 40 provider coordinate units. Error bars: 95% patient-bootstrap intervals.',ha='center',fontsize=10,color='#4d5c69')
    savefig(fig,'vCAF_measured_marker_support')

def spatial_sensitivity(s):
    fig,axes=plt.subplots(1,2,figsize=(12.2,5.7))
    fig.subplots_adjust(left=.075,right=.975,top=.79,bottom=.22,wspace=.24)
    for ax,ep in zip(axes,['nearby_CD8_PD1','nearby_CD8_TCF1']):
        for config,offset,marker in [('RAMP_k30',-.10,'o'),('RAMP_k100',.10,'s')]:
            for pop,color,style in [('all_available',FULL,'-'),('DFS_280_members',DFS,'--')]:
                rows=[getrow(s,ep,pop=pop,config=config,radius=r) for r in [20,40,80]]
                y=np.array([r['mean'] for r in rows]);lo=np.array([r.ci_low for r in rows]);hi=np.array([r.ci_high for r in rows])
                x=np.arange(3)+offset
                ax.errorbar(x,y,yerr=np.array([y-lo,hi-y]),color=color,fmt=marker,linestyle=style,lw=1,ms=4,capsize=3,alpha=.8)
        ax.axhline(0,color='#7d8995',ls=':',lw=1)
        ax.set(xticks=[0,1,2],xticklabels=['20','40','80'],xlabel='Maximum nearest-vCAF distance\n(provider coordinate units)',ylabel='Mean within-ROI Spearman',title=LABELS[ep],ylim=(-.065,.085))
        ax.grid(axis='y',alpha=.12)
    fig.suptitle('Nearby CD8 marker associations across fixed distances and k values',fontsize=15,y=.965)
    handles=[Line2D([],[],color=FULL,label='All eligible patients'),Line2D([],[],color=DFS,ls='--',label='DFS cohort members'),Line2D([],[],color='#465563',marker='o',ls='',label='k = 30'),Line2D([],[],color='#465563',marker='s',ls='',label='k = 100')]
    fig.legend(handles=handles,ncol=4,loc='lower center',bbox_to_anchor=(.5,.018),frameon=False)
    savefig(fig,'CD8_neighborhood_sensitivity')

def spatial_example():
    v=pd.read_csv(OUT/'VCAF_MEASURED_AND_PREDICTED.csv.gz',float_precision='round_trip')
    # Selection uses the number of vCAF only, not HLA values or marker correlations.
    counts=v.groupby(['patient_id','roi_id']).size().sort_values(ascending=False)
    pid,roi=counts.index[0];g=v[v.patient_id.eq(pid)&v.roi_id.eq(roi)].copy()
    md=pd.read_csv(rt.INPUTS['metadata'],usecols=['patient_id','roi_id','x','y'],float_precision='round_trip')
    bg=md[md.patient_id.eq(pid)&md.roi_id.eq(roi)]
    order=np.random.default_rng(42).permutation(len(g));xy=g[['x','y']].to_numpy()[order]
    fig,axes=plt.subplots(1,3,figsize=(16.5,6.4))
    fig.subplots_adjust(left=.025,right=.90,top=.78,bottom=.16,wspace=.05)
    lo=bg[['x','y']].min().to_numpy();hi=bg[['x','y']].max().to_numpy();pad=(hi-lo)*.025
    correlations={}
    for ax,col,title in zip(axes,['RAMP_k30','CD73','IDO'],['Predicted HLA-ABC','Measured CD73','Measured IDO']):
        values=rankdata(g[col])/(len(g)+1)
        ax.scatter(bg.x,bg.y,c='#dce0e3',alpha=.5,s=1.2,lw=0,rasterized=True)
        artist=ax.scatter(xy[:,0],xy[:,1],c=values[order],vmin=0,vmax=1,cmap='viridis',s=13,lw=.2,edgecolors='white',rasterized=True)
        ax.set(xlim=(lo[0]-pad[0],hi[0]+pad[0]),ylim=(hi[1]+pad[1],lo[1]-pad[1]),xticks=[],yticks=[],title=title)
        ax.set_aspect('equal');ax.spines[:].set_visible(False)
        assert np.array_equal(artist.get_offsets(),xy)
        if col!='RAMP_k30':
            correlations[col]=float(spearmanr(g.RAMP_k30,g[col]).statistic)
            ax.text(.5,-.06,f'Within-vCAF Spearman = {correlations[col]:.3f}',transform=ax.transAxes,ha='center',fontsize=10)
    cb=fig.colorbar(artist,cax=fig.add_axes([.925,.23,.012,.44]),ticks=[0,.25,.5,.75,1]);cb.set_label('Within-ROI vCAF rank')
    fig.suptitle(f'Predicted and measured states in the same {len(g):,} vCAF',fontsize=17,y=.965)
    fig.text(.5,.882,f'Cords {pid} | {roi} | Real tissue cell coordinates',ha='center',fontsize=12,color='#4d5c69')
    fig.text(.5,.07,'Example chosen solely by the largest number of vCAF; all three panels use the same cells, coordinates and 0–1 rank scale.',ha='center',fontsize=10,color='#4d5c69')
    fig.text(.5,.028,'Grey: other cells in the ROI. Color represents relative rank among vCAF in this ROI; no spatial smoothing.',ha='center',fontsize=10,color='#4d5c69')
    savefig(fig,'vCAF_measured_marker_spatial_example')
    g.to_csv(OUT/'SPATIAL_EXAMPLE_CELLS.csv',index=False)
    rt.save(OUT/'SPATIAL_EXAMPLE_SELECTION.json',{'patient_id':pid,'roi_id':roi,'n_vcaf':len(g),'selection':'Largest vCAF count among all available ROIs; no expression or outcome used in selection','within_roi_rho':correlations,'coordinates':'original provider x/y; y axis downward','color':'rank/(n_vCAF+1) within this ROI for all three panels','smoothing':False})

def write_report(s,p):
    lines=['# Cords vCAF：实测分子标记与邻近 T 细胞的机制支持','',
      '本轮在新的 mechanism_support 目录内完成。原始数据、三队列对齐、全量预测、既有生存结果保持只读。分析方案在读取新标记数值前冻结于 results/DESIGN.json。','',
      '## 主要结论','',
      '高预测 HLA-ABC 的 vCAF 在同患者、同 ROI 内同时具有较高的实测 CD73 和 IDO，相对高低关系在多数患者中方向一致。邻近 CD8 的 PD-1 关联很小、在固定 DFS 队列中不成立；TCF1 未显示预期的一致降低。因此，这轮得到的是 vCAF 分子表型上的支持，没有证明 vCAF 导致 CD8 耗竭或死亡。','',
      '## 预先指定的四个主要终点','',
      '相关系数先在每个 ROI 内计算，再将 ROI 等权汇总到患者，最后将患者等权汇总。置信区间为 3,000 次患者 bootstrap；p 为患者层面的双侧 Wilcoxon；q 对四个主要终点进行 BH 校正。','',
      '| 终点 | 患者数 | 患者平均 ROI 内 Spearman | 95% CI | 正向患者比例 | q |',
      '|---|---:|---:|---|---:|---:|']
    for ep in PRIMARY:
        r=getrow(s,ep)
        lines.append(f'| {LABELS[ep]} | {int(r.n_patients)} | {r["mean"]:.4f} | {r.ci_low:.4f}–{r.ci_high:.4f} | {r.fraction_positive:.1%} | {r.q_primary:.3g} |')
    lines+=['','## vCAF 高低状态差异','',
      '在各 ROI 内将 vCAF 按既有 HLA 预测分数的下／上三分位分组，比较原作者实测 CD73／IDO 的 ROI 内 vCAF 百分位秩。每位患者先平均其有效 ROI 的高低组差异。分组用于展示连续关联，不是生存分析的患者中位数分组。','',
      '| 标记 | 低分 vCAF 的平均秩 | 高分 vCAF 的平均秩 | 差值及 95% CI |','|---|---:|---:|---|']
    for ep in PRIMARY[:2]:
        r=getrow(s,ep,'difference')
        lines.append(f'| {ep.removeprefix("vCAF_")} | {r.low:.3f} | {r.high:.3f} | +{r["mean"]:.3f} ({r.ci_low:.3f}–{r.ci_high:.3f}) |')
    lines+=['','这里的差值是相对秩差，不是原始蛋白强度的倍数，也不是阳性细胞比例。','',
      '## 固定 DFS 队列与敏感性','',
      '固定 280 人 DFS 队列中，274 人有符合条件的单个 ROI（至少 15 个 vCAF）。CD8 在主距离范围内的有效相关分析覆盖 221 人。没有为了空间分析重新定义原生存队列。','',
      '| 终点 | 有效患者 | 平均相关及 95% CI | q |','|---|---:|---|---:|']
    for ep in PRIMARY:
        r=getrow(s,ep,pop='DFS_280_members')
        lines.append(f'| {LABELS[ep]} | {int(r.n_patients)} | {r["mean"]:.4f} ({r.ci_low:.4f}–{r.ci_high:.4f}) | {r.q_primary:.3g} |')
    lines+=['','k=100、距离 20／80 的完整结果保留在 ASSOCIATION_SUMMARY.csv。下列摘要保留全部预指定敏感性结果：','',
      '| 终点 | k | 距离 | 全部有效患者 | 平均相关 | 95% CI |','|---|---:|---:|---:|---:|---|']
    a=s[(s.population=='all_available')&(s.metric=='rho')&s.endpoint.isin(PRIMARY)].sort_values(['endpoint','config','radius'])
    for _,r in a.iterrows():lines.append(f'| {LABELS[r.endpoint]} | {r.config.split("k")[1]} | {int(r.radius)} | {int(r.n_patients)} | {r["mean"]:.4f} | {r.ci_low:.4f}–{r.ci_high:.4f} |')
    between=pd.read_csv(OUT/'BETWEEN_PATIENT_ASSOCIATIONS.csv');cox=pd.read_csv(OUT/'MEASURED_MARKER_DFS_COX.csv')
    lines+=['','## 与患者预后之间的联系','',
      '细胞／ROI 内的关联与患者间的关联分开解释。患者间实测蛋白均值采用 counts 的 asinh(x)，对每位患者全部 vCAF 求均值。患者自身 ROI 内的秩均值趋近 0.5，不能用于患者间表达量比较。','',
      '| 标记 | DFS280：与患者平均 HLA 分数的 Spearman | p | 调整临床变量及 TMA 后的 DFS HR / SD | p |','|---|---:|---:|---|---:|']
    for m in ['CD73','IDO']:
        r=between[(between.population=='DFS_280_members')&(between.marker==m)].iloc[0]
        c=cox[(cox.marker==m)&cox.adjust_TMA&~cox.with_HLA&(cox.term=='marker_z')].iloc[0]
        lines.append(f'| {m} | {r.rho:.3f} | {r.p:.3f} | {c.HR:.3f} ({c.CI_low:.3f}–{c.CI_high:.3f}) | {c.p:.3f} |')
    lines+=['','CD73／IDO 的患者均值没有显示明确的 HLA 或 DFS 关联；调整 TMA 的患者间秩相关也接近零。原 HLA 生存系数复现误差小于 1e-10。加入各实测标记及 TMA 后，预测 HLA 的 DFS 关联仍存在，但这不构成中介机制证据。','',
      '## 读数与空间定义','',
      '- 同一批既有 527 位 Cords 患者，原作者 vCAF 共 52,328 个；提取 vCAF、CD8、CD4 共 209,301 个细胞的 counts 实测值。原始 source_row_id 是 R SCE 的 1-based 列号。','- vCAF 内分析：同患者同 ROI 至少 15 个 vCAF；低／高组三分位均至少 5 个细胞。','- CD8／CD4：每个 T 细胞分配给同患者同 ROI 最近的一个 vCAF，仅一次分配；主距离 40、伴随距离 20 和 80，均沿用原提供方坐标单位。每 ROI 至少 15 个有效 T 细胞，且最近 vCAF 至少 5 个不同细胞。','- CD8 的 PD-1、TCF1 和 CD4 的 FOXP3 秩在该 ROI 的全部同类 T 细胞内计算，不按距离重新计算秩。','- CD4 的作者 Treg 标签（CD4_Treg）和 PD1-rank 减 TCF1-rank 的连续联合指标仅为预指定次要分析；完整结果保留。未从强度自选阳性阈值。','- CD73、IDO、PD1、TCF1、FOXP3 均未参与当前九标记预测。HLA-DR、Ki-67 等共同标记仅用于逐细胞数值核对。','',
      '## 适合用于故事中的表达','',
      '> 在原作者标注的 vCAF 内，较高的预测 HLA-ABC 得分与实测 CD73 和 IDO 的升高相伴，提示该预测状态具有免疫调节相关的分子表型。','',
      '尚不支持“这些 vCAF 已导致 CD8 耗竭／死亡”或“CD73／IDO 已解释 HLA 与 DFS 的联系”。Cords 缺少实测 HLA-ABC、PD-L1、PD-L2、FASL、IFN-γ 与功能干预；标记共变不是酶活性或因果证据。','',
      '生物学背景：[人肺癌 CAF 中 MHC-I 与 CD73 等共同上调的实验](https://pubmed.ncbi.nlm.nih.gov/34290905/)；[肺癌 CD8 T 细胞 PD-1／TCF1 表型与功能研究](https://pubmed.ncbi.nlm.nih.gov/32332901/)。这些实验提供背景，并不是本数据中 vCAF 机制的直接证明。','',
      '## 图与复现','',
      '- figures/vCAF_measured_marker_support.png / .svg：CD73／IDO 患者配对图及四个主要终点。','- figures/CD8_neighborhood_sensitivity.png / .svg：全部预定距离与 k 值的 CD8 结果。','- figures/vCAF_measured_marker_spatial_example.png / .svg：按 vCAF 数量最多选出的 ROI，真实坐标下同一批 vCAF 的预测 HLA 与实测 CD73、IDO；三者都使用该 ROI 内 vCAF 的 0–1 相对秩，无平滑。','- ROI_ASSOCIATIONS.csv / PATIENT_ASSOCIATIONS.csv / ASSOCIATION_SUMMARY.csv：分层结果与所有方向。','- MEASURED_MARKER_DFS_COX.csv：预后伴随分析及加入 HLA 的联合模型。','- DESIGN.json / EXPORT_VERIFICATION.json / JOIN_VERIFICATION.json / NEIGHBOR_VERIFICATION.json / VERIFICATION.json：冻结方案、源数据与逐细胞连接核对。','',
      '运行：`bash run.sh`。首次提取需要读取约 9 GB 的原始 SCE；后续复现使用已提取的原作者实测值并核对哈希。所有输出和缓存均限定于本目录。','']
    (BASE/'README_ZH.md').write_text('\n'.join(lines))
    rt.save(OUT/'SUMMARY.json',{'primary_all':s[(s.population=='all_available')&(s.config=='RAMP_k30')&s.primary_endpoint].to_dict('records'),'conclusion':'Within-vCAF molecular co-variation supported; nearby CD8 impairment and a marker-mediated DFS chain are not established.'})

def main():
    s=pd.read_csv(OUT/'ASSOCIATION_SUMMARY.csv');p=pd.read_csv(OUT/'PATIENT_ASSOCIATIONS.csv')
    overview(s,p);spatial_sensitivity(s);spatial_example();write_report(s,p)
    print('Three figure pairs and Chinese report saved.',flush=True)

if __name__=='__main__':main()
