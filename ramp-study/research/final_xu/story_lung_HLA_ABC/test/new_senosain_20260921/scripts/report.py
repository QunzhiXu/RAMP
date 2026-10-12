import runtime as rt
from runtime import BASE,STORY
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=BASE/'analysis';FIG=BASE/'figures'
plt.rcParams.update({'font.family':'Arial','font.size':10,'axes.titlesize':12,'axes.titleweight':'bold','axes.labelweight':'bold','axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
def fmtp(p):return f'{p:.3g}'

def main():
    z=pd.read_csv(OUT/'ZHENG_SUMMARY.csv');zm=pd.read_csv(OUT/'ZHENG_METRICS.csv')
    stab=pd.read_csv(OUT/'VCAF_SCORE_STABILITY.csv');pairs=pd.read_csv(OUT/'VCAF_PATIENT_SCORE_PAIRS.csv')
    cox=pd.read_csv(OUT/'COX_TERMS.csv');km=pd.read_csv(OUT/'KM_CURVES.csv');five=pd.read_csv(OUT/'FIVE_YEAR_DFS.csv');kms=pd.read_csv(OUT/'KM_SUMMARY.csv')
    protein=pd.read_csv(OUT/'VCAF_MARKER_SUMMARY.csv');immune=pd.read_csv(OUT/'IMMUNE_COMPOSITION.csv');expr=pd.read_csv(OUT/'IMMUNE_EXPRESSION.csv')
    primary=cox[(cox.config=='RAMP_k30')&(cox.state=='mean')&(cox.cohort=='all280')&(cox.term=='state_z')].set_index('version')
    fig,axs=plt.subplots(2,3,figsize=(14,8.1),gridspec_kw={'hspace':.38,'wspace':.3})
    ax=axs[0,0];d=zm[(zm.level=='patient')&(zm.config=='RAMP_k30')].pivot(index='patient_id',columns='version',values='cell_spearman')
    ax.boxplot([d.old,d.new],positions=[0,1],widths=.45,showfliers=False,patch_artist=True,boxprops={'facecolor':'#E3E9EE'})
    for _,r in d.iterrows():ax.plot([0,1],[r.old,r.new],'-o',lw=.7,ms=3,alpha=.65,c='#7994A7')
    ax.set_xticks([0,1],['Old','New']);ax.set_ylim(0,1);ax.set_ylabel('Patient Spearman');ax.set_title('A  Zheng validation (28 patients)',loc='left')
    ax.text(.03,.96,f'Median: {d.old.median():.3f} → {d.new.median():.3f}',transform=ax.transAxes,va='top',fontweight='bold')
    ax=axs[0,1];p=pairs[(pairs.config=='RAMP_k30')&(pairs.state=='mean')];ax.scatter(p.old_score,p.new_score,s=14,c='#356F7E',alpha=.65)
    lim=[min(p.old_score.min(),p.new_score.min())-.02,max(p.old_score.max(),p.new_score.max())+.02];ax.plot(lim,lim,'--',c='gray',lw=.8);ax.set_xlim(lim);ax.set_ylim(lim)
    ax.set_xlabel('Old patient vCAF mean');ax.set_ylabel('New patient vCAF mean');ax.set_title('B  Patient-score stability',loc='left')
    s=stab[(stab.config=='RAMP_k30')&(stab.state=='mean')].iloc[0];ax.text(.03,.96,f'ρ = {s.old_new_spearman:.3f}\nGroup changed: {int(s.changed_group)}/280',transform=ax.transAxes,va='top',fontweight='bold')
    ax=axs[0,2]
    for v,offset,color in [('old',-.13,'#869BAA'),('new',.13,'#22796C')]:
        q=protein[(protein.config=='RAMP_k30')&(protein.metric=='difference')&(protein.version==v)].set_index('endpoint').loc[['vCAF_CD73','vCAF_IDO']]
        ax.errorbar(np.arange(2)+offset,q['mean'],yerr=np.array([q['mean']-q.ci_low,q.ci_high-q['mean']]),fmt='o',capsize=4,label=v.title(),c=color,ms=6)
    ax.axhline(0,c='gray',lw=.8);ax.set_xticks([0,1],['CD73','IDO']);ax.set_ylabel('High − low marker percentile');ax.set_title('C  Within-ROI vCAF phenotype',loc='left');ax.legend(frameon=False)
    for col,v in enumerate(['old','new']):
        ax=axs[1,col];q=km[(km.version==v)&(km.config=='RAMP_k30')&(km.state=='mean')&(km.cohort=='all280')]
        for group,color in [('Lower','#3379A7'),('Higher','#C76547')]:
            curve=q[q.group.eq(group)];ax.step(np.r_[0,curve.time],np.r_[1,curve.survival],where='post',lw=1.8,c=color,label=group)
        ax.set_xlim(0,10);ax.set_ylim(0,1.02);ax.set_xlabel('Years');ax.set_ylabel('Disease-free survival');ax.set_title(f'{"D" if col==0 else "E"}  {v.title()} reference: DFS',loc='left');ax.legend(frameon=False,loc='lower left',fontsize=9)
        r=primary.loc[v];ax.text(.97,.96,f'Adjusted HR / SD: {r.hr:.2f}\n95% CI {r.ci_low:.2f}–{r.ci_high:.2f}; Cox P = {fmtp(r.p)}',transform=ax.transAxes,ha='right',va='top',fontweight='bold',fontsize=9)
    ax=axs[1,2];endpoints=['immune_of_all','CD4_of_immune','CD8_of_immune','Bcell_of_immune','Myeloid_of_immune','Neutrophil_of_immune']
    q=immune[immune.version.eq('new')].set_index('endpoint').loc[endpoints];x=np.arange(len(q))
    ax.barh(x-.17,q.lower_mean*100,.32,label='Lower',color='#3379A7');ax.barh(x+.17,q.higher_mean*100,.32,label='Higher',color='#C76547')
    ax.set_yticks(x,['Immune / all','CD4 / immune','CD8 / immune','B / immune','Myeloid / immune','Neutrophil / immune']);ax.invert_yaxis();ax.set_xlabel('Mean patient fraction (%)');ax.set_title('F  New patient-group composition',loc='left');ax.legend(frameon=False,fontsize=9)
    fig.savefig(FIG/'STORY_RECHECK_OVERVIEW.png',dpi=230,bbox_inches='tight');plt.close(fig)
    # Existing C30 is retained; raw cell centroids only, no generated tissue image.
    spatial=pd.read_csv(OUT/'C30_SPATIAL.csv.gz');fig,axs=plt.subplots(1,3,figsize=(13,4))
    for ax,col,title in zip(axs,['measured_relative_score','old_prediction','new_prediction'],['Measured HLA-A','Old prediction','New prediction']):
        sc=ax.scatter(spatial.x_um,spatial.y_um,c=spatial[col],cmap='viridis',vmin=.25,vmax=.6,s=1.2,linewidths=0,rasterized=True)
        ax.set_aspect('equal');ax.invert_yaxis();ax.set_title(title);ax.set_xticks([]);ax.set_yticks([])
        for sp in ax.spines.values():sp.set_visible(False)
        if col!='measured_relative_score':ax.text(.5,-.02,f'Spearman ρ = {spearmanr(spatial.measured_relative_score,spatial[col]).statistic:.3f}',ha='center',transform=ax.transAxes)
    fig.subplots_adjust(wspace=.03,right=.93);cb=fig.colorbar(sc,ax=axs,fraction=.02,pad=.025);cb.set_ticks([.25,.4,.6],labels=['≤0.25','0.40','≥0.60']);cb.set_label('Relative score')
    fig.suptitle('Zheng C30: same cells and real coordinates',fontweight='bold');fig.savefig(FIG/'C30_OLD_NEW_SPATIAL.png',dpi=230,bbox_inches='tight');plt.close(fig)
    lines=['# 新版 Senosain 输入：肺癌 HLA-ABC 故事测试结果','','完整规则见 [PLAN_ZH.md](PLAN_ZH.md)。所有输出均在本 test 下；未更新正式图或 manuscript。','',
    '## 本轮判断','','核心关联方向保留，生存效应略弱。Zheng HLA-A 验证基本稳定；新旧患者 vCAF 分数高度相关；高分 vCAF 的 CD73/IDO 相关表型仍在；高分患者的整体免疫比例较低、CD4/B 在免疫细胞中占比较低、中性粒细胞相对占比较高的组成模式仍在。新版主分析以及 k100、排除参考患者敏感性中，vCAF 分数与较差 DFS 的关联均保持。','',
    '这支持继续使用该故事，但替换输入版本时需要同步更新数值。Baseline 仍有类似的验证和生存信号；本测试不说明这些生物关联为 RAMP 独有。','',
    '## 变更范围','','原 28+28+28 患者和各 1000 细胞配额、九个输入 marker、RAMP 2.3.0 与 kNN 规则固定。Senosain 换为新版注释池中 QC=keep 的冻结顺序前 1000 cells/患者，C02 保留；Cords/Zheng 参考细胞不变。新版源细胞仅 849 个与旧版重合，重合细胞表达值完全一致。这是输入版本敏感性，不是仅改标签。','',
    '## Zheng 全量验证','','28 患者、119 cores、693,693 细胞。预测 HLA-ABC 相对信号并用实测 HLA-A 辅助验证；不是直接验证 Cords vCAF 的 HLA-ABC。','',
    '| 汇总（RAMP k30） | 旧版 | 新版 |','|---|---:|---:|']
    for level,label in [('patient','患者 Spearman 中位数'),('core','Core Spearman 中位数')]:
        q=z[(z.config=='RAMP_k30')&(z.level==level)].set_index('version');lines.append(f'| {label} | {q.loc["old","median_spearman"]:.4f} | {q.loc["new","median_spearman"]:.4f} |')
    q=z[(z.config=='RAMP_k30')&(z.level=='patient')].set_index('version');lines.append(f'| 患者等权 relative-score MAE | {q.loc["old","mean_relative_score_MAE"]:.4f} | {q.loc["new","mean_relative_score_MAE"]:.4f} |')
    lines+=['','## vCAF 分数与 DFS','',f'同一 280 位患者、96 个事件。新旧患者 vCAF 均值 Spearman = **{s.old_new_spearman:.4f}**；各自患者中位数分组后 **{int(s.changed_group)}/280** 人改变组别。',
    '', '两版 Cox 均调整该版本当前三平台 global-CAF 分数，以及年龄、性别、分类分期、CAF 比例、vCAF/CAF 比例。旧版基准为修正后的 HR，而非过时的 1.40。',
    '', '| RAMP k30 主指标：vCAF 均值 | 旧版 | 新版 |','|---|---:|---:|',
    f'| 调整 HR / SD | {primary.loc["old","hr"]:.3f} | {primary.loc["new","hr"]:.3f} |',
    f'| 95% CI | {primary.loc["old","ci_low"]:.3f}–{primary.loc["old","ci_high"]:.3f} | {primary.loc["new","ci_low"]:.3f}–{primary.loc["new","ci_high"]:.3f} |',
    f'| Cox P | {fmtp(primary.loc["old","p"])} | {fmtp(primary.loc["new","p"])} |']
    for group in ['Lower','Higher']:
        q=five[(five.config=='RAMP_k30')&(five.state=='mean')&(five.cohort=='all280')&(five.group==group)].set_index('version')
        lines.append(f'| {group} 组 5 年 DFS | {q.loc["old","five_year_survival"]:.1%} | {q.loc["new","five_year_survival"]:.1%} |')
    lines+=['','### 固定敏感性结果','', '| 版本 | 配置 | 状态 | 人群 | HR (95% CI) | P |','|---|---|---|---|---|---:|']
    for _,r in cox[cox.term.eq('state_z')].iterrows():lines.append(f'| {r.version} | {r.config} | {r.state} | {r.cohort} | {r.hr:.3f} ({r.ci_low:.3f}–{r.ci_high:.3f}) | {fmtp(r.p)} |')
    lines+=['','## vCAF 分子表型','','按原规则：各 ROI 预测上下三分位，CD73/IDO 用 ROI 内表达 percentile 比较，先 ROI 后患者汇总。下表是患者平均 High−Low 差值；它不是绝对表达差异。','', '| Marker | 旧版差值 | 新版差值 | 新版 95% CI |','|---|---:|---:|---|']
    for m in ['vCAF_CD73','vCAF_IDO']:
        q=protein[(protein.config=='RAMP_k30')&(protein.metric=='difference')&(protein.endpoint==m)].set_index('version');a=q.loc['new']
        lines.append(f'| {m} | {q.loc["old","mean"]:.3f} | {a["mean"]:.3f} | {a.ci_low:.3f}–{a.ci_high:.3f} |')
    lines+=['','## 患者免疫组成与表达','','组成/表达按原 TMA、年龄、性别、分类分期模型调整，HC3 标准误。组成 12 终点整体 BH；主要表达 10 终点 BH，次要表达另列。比例不等于绝对细胞浸润。','', '| 新版组成 | Lower 均值 | Higher 均值 | 调整差异 SD | q |','|---|---:|---:|---:|---:|']
    for _,r in immune[immune.version.eq('new')].iterrows():lines.append(f'| {r.endpoint} | {r.lower_mean:.2%} | {r.higher_mean:.2%} | {r.coef:+.3f} | {fmtp(r.q_bh)} |')
    lines+=['','| 新版主要表达 | Lower 均值 | Higher 均值 | 调整差异 SD | q |','|---|---:|---:|---:|---:|']
    for _,r in expr[(expr.version=='new')&(expr.family=='primary')&(expr.analysis=='all_available_mean')].iterrows():lines.append(f'| {r.lineage} {r.marker} | {r.lower_mean:.3f} | {r.higher_mean:.3f} | {r.coef:+.3f} | {fmtp(r.q_bh)} |')
    lines+=['','## 文件','','- [结果总览图](figures/STORY_RECHECK_OVERVIEW.png)','- [新旧 UMAP](figures/OLD_NEW_UMAP.png)：分别拟合，仅作展示，不能通过坐标位置变化解释生物变化。','- [C30 真实坐标对照](figures/C30_OLD_NEW_SPATIAL.png)：保留原示例，色标 0.25–0.60 仅作显示，不参与指标。','- `inputs/`：新参考细胞 ID、表达、响应和抽样核对。','- `alignment/`：重拟合矩阵、profile、平台映射、全部算法审计及 UMAP。','- `validation/`、`cords/`：全部目标细胞预测、变换矩阵、分患者检查点与输入哈希。','- `analysis/`：完整统计、患者分组、PH 诊断、敏感性和新旧比较。','- `provenance/FINAL_CHECKS.json`：原始输入不变、重合细胞表达一致、旧投影复现、旧已修正 Cox 复现。','',
    '## 解释边界','','本轮没有优化模型、marker、donor、ROI、阈值或生存入组规则。旧版结果已经探索过，当前测试属于同一数据体系中的稳健性检查，不是外部验证。C02 的谱系不确定性仍保留，标签没有进入 HLA 预测。若仅替换标签且保持所有输入细胞不变，预测本应不变。']
    (BASE/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n')
    rt.save(BASE/'REPORT_COMPLETE.json',dict(time_utc=rt.now(),figures=[p.name for p in FIG.glob('*.png')],stats_from_csv=True))
    print('Report and figures complete',flush=True)
if __name__=='__main__':main()
