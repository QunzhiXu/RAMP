"""Figures and an honest bridge between reverse protein association and the DFS story."""
import runtime as rt
from runtime import BASE
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import NullFormatter

OUT=BASE/'results';FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'svg.fonttype':'none','savefig.facecolor':'white','axes.titleweight':'bold'})
BLUE='#357b9a';RED='#b64744';GRAY='#c8ccce';SEED=20260909
STATES=['r40__CAF_fraction','r40__Immune_fraction','r40__Tumour_fraction','r40__Vessel_fraction','r40__Vascular_proximity','r40__Mixing_entropy']+[f'niche__N{i}' for i in range(1,7)]
LABELS=['Non-CAF-neighbor score (1 - CAF fraction)','Immune-neighbor fraction','Tumour-neighbor fraction','Vessel-neighbor fraction','Vascular proximity','Neighborhood mixing',
        'N1: CAF-containing tumour niche','N2: CAF-dominant niche','N3: Immune-dominant niche','N4: Mixed tumour niche','N5: Tumour-dominant niche','N6: Mixed non-tumour niche']

def save(fig,name):
    fig.savefig(FIG/(name+'.png'),dpi=190,bbox_inches='tight');fig.savefig(FIG/(name+'.svg'),bbox_inches='tight');plt.close(fig)

def main():
    sm=pd.read_csv(OUT/'SENOSAIN_REVERSE_PREDICTIONS.csv.gz',dtype={'patient_id':str},low_memory=False)
    y=pd.read_csv(OUT/'SENOSAIN_MEASURED_HELDOUT_PROTEINS.csv.gz',dtype={'patient_id':str})
    assert np.array_equal(sm.source_row_id,y.source_row_id)
    summ=pd.read_csv(OUT/'SENOSAIN_PROTEIN_CONTRASTS.csv');paired=pd.read_csv(OUT/'SENOSAIN_DONOR_PROTEIN_CONTRASTS.csv.gz',dtype={'patient_id':str})
    xy=np.load(BASE.parent/'umap/results/RAMP_UMAP.npy')[sm.reference_index.to_numpy(int)]
    gate=sm.paper_mesenchymal_gate.to_numpy(bool);order=np.random.default_rng(SEED).permutation(np.flatnonzero(gate))
    score=1-sm.k30__r40__CAF_fraction.to_numpy();lo,hi=np.quantile(score[gate],[.05,.95])
    # Display limits are a declared 5th/95th percentile rule, not a fit to HLA or survival.
    rt.save(OUT/'PLOT_DESIGN.json',dict(spatial_score='1 minus predicted CAF-neighbor fraction at radius 40',orientation='Algebraic complement chosen for readable high-state orientation after the full screen; does not change any association.',
        umap='Existing frozen joint UMAP positions; only Senosain shown. Grey = other Senosain cells; colored = all 1180 inherited mesenchymal-gate cells.',
        spatial_color_limits=[float(lo),float(hi)],spatial_color_rule='5th/95th percentiles among highlighted cells; clipping shown on colorbar.',protein_color_limits=[0,1],
        protein_values='Measured within-donor percentile among original 1000 cells, no spatial smoothing.',
        plotted_cell_order='Fixed identical random order across all paired UMAP panels',survival='All 12 primary spatial states retained; no KM cut optimization.'))
    fig,axes=plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
    vals=[score,y.rank__HLA_ABC.to_numpy(),y.rank__PD_L1.to_numpy()]
    titles=['Inferred non-CAF-neighbor score','Measured HLA-ABC','Measured PD-L1']
    for j,(ax,val,title) in enumerate(zip(axes,vals,titles)):
        ax.scatter(xy[:,0],xy[:,1],s=1,c='#e5e7e8',alpha=.45,rasterized=True,linewidths=0)
        sc=ax.scatter(xy[order,0],xy[order,1],s=7,c=val[order],cmap='viridis',norm=Normalize(lo,hi) if j==0 else Normalize(0,1),rasterized=True,linewidths=0)
        ax.set_title(title);ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([]);ax.set_xlabel('Frozen UMAP; Senosain cells only')
        cb=fig.colorbar(sc,ax=ax,shrink=.65,pad=.02,extend='both' if j==0 else 'neither')
        cb.set_label('Inferred neighbor fraction' if j==0 else 'Within-donor percentile')
        ax.text(0,1.03,chr(65+j),transform=ax.transAxes,weight='bold',fontsize=14)
    fig.suptitle('Reverse spatial transfer and held-out measured proteins\n1,180 cells in the inherited Senosain mesenchymal gate; all 28 donors',fontsize=13)
    save(fig,'reverse_spatial_state_measured_proteins_UMAP')
    # Unselected all-cell version is provided separately.
    fig,axes=plt.subplots(1,3,figsize=(15,4.6),layout='constrained');allorder=np.random.default_rng(SEED).permutation(len(sm))
    for j,(ax,val,title) in enumerate(zip(axes,vals,titles)):
        norm=Normalize(0,1)
        sc=ax.scatter(xy[allorder,0],xy[allorder,1],s=2,c=val[allorder],cmap='viridis',norm=norm,rasterized=True,linewidths=0)
        ax.set_title(title);ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])
        cb=fig.colorbar(sc,ax=ax,shrink=.65,pad=.02);cb.set_label('Inferred neighbor fraction' if j==0 else 'Within-donor percentile')
    fig.suptitle('All-to-all reverse transfer: all 28,000 Senosain cells',fontsize=13);save(fig,'reverse_spatial_state_all_Senosain_UMAP')
    # Paired donor plots, with the complement orientation applied exactly.
    fig,axes=plt.subplots(1,3,figsize=(14,4.8),layout='constrained',gridspec_kw={'width_ratios':[1,1,1.45]})
    for ax,marker,label in zip(axes[:2],['HLA_ABC','PD_L1'],['HLA-ABC','PD-L1']):
        d=paired.query("population=='paper_mesenchymal' and k==30 and state=='r40__CAF_fraction' and marker==@marker").sort_values('patient_id')
        a=summ.query("population=='paper_mesenchymal' and k==30 and state=='r40__CAF_fraction' and marker==@marker").iloc[0]
        for _,r in d.iterrows():ax.plot([0,1],[r.high,r.low],color=GRAY,lw=.8,zorder=1)
        ax.scatter(np.zeros(len(d)),d.high,c=BLUE,s=20,zorder=2);ax.scatter(np.ones(len(d)),d.low,c=RED,s=20,zorder=2)
        ax.plot([0,1],[d.high.mean(),d.low.mean()],color='black',marker='D',lw=2,markersize=6,zorder=3)
        ax.set_xticks([0,1],['Lower score','Higher score']);ax.set_xlim(-.25,1.25);ax.set_ylim(0,.85);ax.set_ylabel('Measured protein: mean within-donor percentile')
        ax.set_title(f'{label}\nMean increase = {-a["mean"]:.3f}; {int((d.difference<0).sum())}/{len(d)} donors',fontsize=11)
        ax.text(.5,.02,f'Paired Wilcoxon p = {a.wilcoxon_p:.2g}',transform=ax.transAxes,ha='center',fontsize=9)
        ax.set_xlabel('Predicted non-CAF-neighbor score\nDonor-specific lower / upper thirds')
    ax=axes[2];z=summ.query("population=='paper_mesenchymal' and marker=='HLA_ABC'")
    labels=[];pos=0
    for radius in [20,40,80]:
        for k in [30,100]:
            a=z[(z.state==f'r{radius}__CAF_fraction')&(z.k==k)].iloc[0]
            point=-a['mean'];lower=-a.ci_high;upper=-a.ci_low
            ax.errorbar(point,pos,xerr=np.array([[point-lower],[upper-point]]),fmt='o',color=BLUE if k==30 else RED,capsize=3)
            labels.append(f'Radius {radius}, k = {k}');pos+=1
    ax.axvline(0,color=GRAY,ls='--');ax.set_yticks(range(6),labels);ax.invert_yaxis();ax.set_xlabel('HLA-ABC difference: higher - lower score\nMean and 95% patient bootstrap interval')
    ax.set_title('Direction persists across radius and k',fontsize=11);ax.grid(axis='x',color='#eeeeee')
    fig.suptitle('CAF-poor neighborhood-like phenotype has higher measured HLA-ABC and PD-L1\n26 donors with at least 15 cells in the inherited gate; donor-weighted comparisons',fontsize=12)
    save(fig,'measured_HLA_PDL1_donor_comparisons')
    # Three linked questions, showing all primary states instead of selecting favorable ones.
    corr=pd.read_csv(OUT/'CORDS_PATIENT_HLA_ASSOCIATIONS.csv');cox=pd.read_csv(OUT/'SPATIAL_DFS_COX.csv');bridge=[]
    for state,label in zip(STATES,LABELS):
        a=summ.query("population=='paper_mesenchymal' and k==30 and marker=='HLA_ABC' and state==@state").iloc[0]
        b=corr.query("population=='DFS_280' and k==30 and state==@state").iloc[0]
        c=cox.query("model=='spatial_only' and term=='spatial_z' and state==@state").iloc[0]
        sign=-1 if state=='r40__CAF_fraction' else 1
        bridge.append(dict(state=state,label=label,orientation=sign,protein_delta=sign*a['mean'],protein_low=a.ci_low if sign==1 else -a.ci_high,
            protein_high=a.ci_high if sign==1 else -a.ci_low,protein_q=a.q_HLA_states,cords_HLA_rho=sign*b.rho,cords_HLA_p=b.p,
            DFS_HR=c.HR if sign==1 else 1/c.HR,DFS_low=c.CI_low if sign==1 else 1/c.CI_high,DFS_high=c.CI_high if sign==1 else 1/c.CI_low,DFS_p=c.p,DFS_q=c.q_bh))
    bridge=pd.DataFrame(bridge);bridge.to_csv(OUT/'STORY_BRIDGE_SUMMARY.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(15,6.8),sharey=True,layout='constrained',gridspec_kw={'width_ratios':[1.2,1,1.1]})
    yy=np.arange(len(bridge));ax=axes[0]
    ax.errorbar(bridge.protein_delta,yy,xerr=np.stack([bridge.protein_delta-bridge.protein_low,bridge.protein_high-bridge.protein_delta]),fmt='o',color=BLUE,capsize=2)
    ax.axvline(0,color=GRAY,ls='--');ax.set_yticks(yy,LABELS);ax.invert_yaxis();ax.set_title('Senosain: measured HLA-ABC',fontsize=11)
    ax.set_xlabel('Higher - lower spatial score\nMean protein percentile difference, 26 donors')
    axes[1].barh(yy,bridge.cords_HLA_rho,color=[RED if x>0 else BLUE for x in bridge.cords_HLA_rho],height=.55)
    axes[1].axvline(0,color=GRAY,lw=1);axes[1].set_title('Cords: predicted vCAF HLA-ABC',fontsize=11);axes[1].set_xlabel('Patient-level Spearman correlation\nMean spatial score versus mean vCAF HLA, n = 280')
    axes[2].errorbar(bridge.DFS_HR,yy,xerr=np.stack([bridge.DFS_HR-bridge.DFS_low,bridge.DFS_high-bridge.DFS_HR]),fmt='o',color='#505b63',capsize=2)
    axes[2].axvline(1,color=GRAY,ls='--');axes[2].set_xscale('log');axes[2].set_xlim(.45,1.85);axes[2].set_xticks([.5,.75,1,1.5],['0.5','0.75','1.0','1.5'])
    axes[2].xaxis.set_minor_formatter(NullFormatter())
    axes[2].set_title('Cords: spatial state versus DFS',fontsize=11);axes[2].set_xlabel('Adjusted HR per SD and 95% CI\n280 patients, 96 events')
    for ax in axes:ax.grid(axis='y',color='#f0f0f0');ax.set_axisbelow(True)
    fig.suptitle('Spatial states across measured HLA-ABC, predicted vCAF HLA-ABC, and DFS',fontsize=12)
    save(fig,'spatial_HLA_survival_bridge_all_states')
    # Niche composition and annotation sensitivity.
    niches=pd.read_csv(OUT/'NICHE_DEFINITIONS.csv')
    fig,axes=plt.subplots(1,2,figsize=(13.5,6.1),layout='constrained',gridspec_kw={'width_ratios':[1,1.1]})
    cats=['CAF','Tumour','Vessel','T_cell','Other_immune','Other'];ax=axes[0]
    im=ax.imshow(niches[cats],vmin=0,vmax=1,cmap='Blues',aspect='auto')
    for i in range(6):
        for j in range(6):ax.text(j,i,f'{niches[cats].iloc[i,j]:.0%}',ha='center',va='center',color='white' if niches[cats].iloc[i,j]>.5 else '#222222',fontsize=9)
    ax.set_xticks(range(6),['CAF','Tumour','Vessel','T cell','Other\nimmune','Other'],rotation=45,ha='right')
    ax.set_yticks(range(6),['N1','N2','N3','N4','N5','N6']);ax.set_title('Spatial niches: actual neighbor composition');fig.colorbar(im,ax=ax,shrink=.6)
    groups=['paper_mesenchymal','broad_stromal','all_cells'];mat=np.zeros((12,3))
    for i,state in enumerate(STATES):
        for j,pop in enumerate(groups):
            a=summ.query("population==@pop and k==30 and state==@state and marker=='HLA_ABC'").iloc[0]
            mat[i,j]=a['mean']*(-1 if i==0 else 1)
    ax=axes[1];im=ax.imshow(mat,cmap='RdBu_r',vmin=-.15,vmax=.15,aspect='auto')
    for i in range(12):
        for j in range(3):ax.text(j,i,f'{mat[i,j]:+.3f}',ha='center',va='center',fontsize=9)
    ax.set_yticks(range(12),['Non-CAF score','Immune fraction','Tumour fraction','Vessel fraction','Vascular proximity','Mixing']+[f'N{i}' for i in range(1,7)])
    ax.set_xticks(range(3),['Inherited gate\n26 donors','Broad stromal\n28 donors','All cells\n28 donors']);ax.set_title('Measured HLA association depends on the cell group')
    fig.colorbar(im,ax=ax,shrink=.6,label='HLA-ABC percentile difference');save(fig,'niches_and_cell_group_sensitivity')
    # Narrative report with exact quantitative limits.
    h=summ.query("population=='paper_mesenchymal' and k==30 and state=='r40__CAF_fraction' and marker=='HLA_ABC'").iloc[0]
    p=summ.query("population=='paper_mesenchymal' and k==30 and state=='r40__CAF_fraction' and marker=='PD_L1'").iloc[0]
    residual=pd.read_csv(OUT/'BROAD_LABEL_ADJUSTED_RESIDUAL_DIFFERENCE.csv')
    rh=residual.query("population=='paper_mesenchymal' and k==30 and state=='r40__CAF_fraction' and marker=='HLA_ABC'").iloc[0]
    broad=summ.query("population=='broad_stromal' and k==30 and state=='r40__CAF_fraction' and marker=='HLA_ABC'").iloc[0]
    cv=pd.read_csv(OUT/'CORDS_LODO_METRICS.csv');cv.groupby(['population','state']).agg(n_donors=('rho','count'),median_rho=('rho','median'),mean_rho=('rho','mean'),median_R2=('R2','median')).to_csv(OUT/'CORDS_LODO_SUMMARY.csv')
    report=f'''# 反向空间转移：Cords → Senosain

本轮已完成。反向转移出现实测 HLA-ABC / PD-L1 关联，但尚未形成“空间状态 → HLA-ABC → 较差 DFS”的完整证据链。

## 数据与设计

- 所有新增脚本、缓存、数据表、图仅位于本文件夹。现有对齐、原始数据、验证、生存结果均只读。
- Cords 28 位原参考患者、每人 1,000 个锚点 → Senosain 28 位患者、每人 1,000 个细胞；全部细胞 → 全部细胞，无门控转移。
- 计算空间标签时使用 Cords 全 ROI 真实坐标和全部邻居，排除查询细胞自身。覆盖 28,000 个参考细胞及全队列 52,328 个作者注释 vCAF；全数据 527 位患者，其中 524 位有 vCAF。
- 主半径 40，伴随半径 20/80；单位沿用提供者坐标，不擅自解释为微米。六类邻居比例、血管接近程度、混合度；另在均衡 Cords 锚点上固定 K=6 无监督空间生态位。
- 11 个参考细胞半径 40 内没有其他细胞；其邻域标签设为缺失，在转移权重中按该特征重新归一化。所有 28,000 个源细胞仍参加表达空间邻居搜索，没有按表型删选。
- 空间定义和反向预测在读取新响应之前写入 PREDICTION_FREEZE.json。HLA、DFS 从未用于空间聚类、标签、距离或转移权重。
- 用冻结三平台 RAMP 共享 9 标记空间进行反向 kNN；StandardScaler 拟合 Cords 源端。k=30 主分析，k=100 伴随分析；UMAP 仅用于展示。
- 26 个未参与对齐的实测蛋白全部保留。主分析沿用历史 paper_mesenchymal_gate，至少 15 个细胞/患者；28 位中 26 位符合条件。另完整保存全部细胞、宽泛 stromal 标签分析。
- 表达量为各患者全部 1,000 个细胞内的实测蛋白百分位；高低比较为各患者分析细胞内空间得分的上/下三分位。表中 0.109 是 10.9 个百分位点，不是原始表达增加 10.9%。
- 空间得分的主展示方向为 1 − 预测 CAF 邻居比例，称“预测非 CAF 邻居得分”。这是对原始 CAF 比例的代数翻转，不是另外训练的高风险标签，也不专指肿瘤或免疫环境。

## 得到了什么

在历史间质门内，预测非 CAF 邻居得分更高的细胞：

| 实测蛋白 | 低得分 | 高得分 | 患者平均差 | 95% 患者 bootstrap CI | 方向一致 |
|---|---:|---:|---:|---|---:|
| HLA-ABC | {h.high:.3f} | {h.low:.3f} | {-h['mean']:.3f} | {-h.ci_high:.3f}–{-h.ci_low:.3f} | 24/26 |
| PD-L1 | {p.high:.3f} | {p.low:.3f} | {-p['mean']:.3f} | {-p.ci_high:.3f}–{-p.ci_low:.3f} | 23/26 |

HLA 在 12 个主空间状态比较内 BH q={h.q_HLA_states:.3g}；全部 26 蛋白 ×12 状态的 BH q={h.q_all_heldout:.3g}。20/40/80 半径、k30/k100 的 HLA 差值均为正，范围约 0.100–0.122。

扣除患者内既有 broad_type 组间均值差异后，HLA 的残差表达差仍为 {-rh['mean']:.3f}，q={rh.q_bh:.3g}。这说明并非完全由已有宽泛标签比例解释。

但是两个既有细胞分组不等价：历史间质门 1,180 个细胞、broad stromal 1,836 个细胞，仅重叠 193 个。核对原始 gate.npy 后身份完全一致，不是本轮错位。不能把历史门直接写成作者确认的 CAF/vCAF。改用 broad stromal，非 CAF 邻居得分对应 HLA 差值为 {-broad['mean']:.3f}，CI {-broad.ci_high:.3f}–{-broad.ci_low:.3f}，q={broad.q_HLA_states:.3g}，明显减弱。

## 与原 HLA / DFS 故事的连接

- Cords 280 位 DFS 患者，非 CAF 邻居得分与平均 vCAF 预测 HLA-ABC 的患者间相关仅 rho=+0.101，p=0.091。不能由此声称同一空间状态在两端得到强一致复现。
- 免疫邻域得分在 Senosain 历史门内与实测 HLA 正相关；但 Cords 患者间平均免疫邻域比例与平均 vCAF HLA 负相关。ROI 内细胞比较与患者间比较不是同一个问题，不能拼接成同方向链条。
- 12 个主空间状态分别进入原协变量调整的 Cox 模型，全部 p≥0.162，没有明确 DFS 关联。没有进行生存阈值优化。
- 加入任一空间状态后，原 vCAF HLA 状态仍与较差 DFS 相关，HR 约 1.36–1.40。它支持原 HLA 结果未被这些邻域组成替代，但不构成空间状态导致 HLA 或生存变化的证据。
- 留一 Cords 患者验证中，所有细胞的空间状态有部分可预测性；vCAF 内部的空间关联较弱，多数 R²为负。反向得分适合称“空间环境倾向/表型相似性”，不能称恢复真实微环境或位置。

## 建议如何使用

目前可用的表述是：**反向空间注释揭示了一个伴随较高实测 HLA-ABC 和 PD-L1 的表型倾向。** 这是探索性结果，并依赖细胞分组。暂不建议将它作为原 vCAF 生存故事的核心机制解释。原 DFS 主结果保持不变；新结果可作为补充探索。

## 图和复现

- results/figures/reverse_spatial_state_measured_proteins_UMAP：相同 Senosain UMAP，历史门上显示空间得分、实测 HLA-ABC、实测 PD-L1。
- results/figures/reverse_spatial_state_all_Senosain_UMAP：不筛选细胞的全部 28,000 个 Senosain 细胞，三列原样提供。
- results/figures/measured_HLA_PDL1_donor_comparisons：逐患者配对差异与半径/k敏感性。
- results/figures/spatial_HLA_survival_bridge_all_states：全部 12 个主空间状态的蛋白、原 HLA、DFS 三端结果。
- results/figures/niches_and_cell_group_sensitivity：生态位组成和不同细胞分组的完整对比。

运行 `./run.sh`；已冻结的空间预测不会重复覆盖。单独重画用 `./run.sh report`。所有脚本受本目录写入边界约束；R 临时文件也位于本目录。
'''
    (BASE/'README_ZH.md').write_text(report)
    rt.save(OUT/'SUMMARY.json',dict(primary_signal='Inverse CAF-neighbor fraction in inherited Senosain mesenchymal gate',n_donors=26,
        HLA_delta=float(-h['mean']),HLA_ci=[float(-h.ci_high),float(-h.ci_low)],HLA_q=float(h.q_HLA_states),PDL1_delta=float(-p['mean']),
        complete_spatial_HLA_DFS_chain_supported=False,minimum_primary_spatial_DFS_p=float(cox.query("primary_state and model=='spatial_only'").p.min()),
        broad_stromal_HLA_delta=float(-broad['mean']),broad_stromal_HLA_q=float(broad.q_HLA_states),figures=[p.name for p in sorted(FIG.glob('*.png'))]))

if __name__=='__main__':main()
