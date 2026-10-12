"""Figures linking inferred vCAF state to abundance and prognosis."""
import runtime as rt
from runtime import BASE
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image
from xml.etree import ElementTree

OUT=BASE/'results';FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
LOW='#217992';HIGH='#d36645'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})

def save(fig,name):
    fig.savefig(FIG/f'{name}.png',dpi=220,facecolor='white');fig.savefig(FIG/f'{name}.svg',facecolor='white');plt.close(fig)

def state_abundance():
    d=pd.read_csv(OUT/'RAMP_k30_STATE_ABUNDANCE.csv');km=pd.read_csv(OUT/'KM_5YEAR.csv').query("id=='RAMP_k30_joint'").set_index('group')
    cuts=pd.read_csv(OUT/'ABUNDANCE_STRATIFIED_COX.csv').query("config=='RAMP_k30'").iloc[0]
    fig,axes=plt.subplots(1,2,figsize=(12.4,5.5),gridspec_kw={'width_ratios':[1.2,1]})
    for state,col in [('Low HLA',LOW),('High HLA',HIGH)]:
        a=d[d.state_group.eq(state)];axes[0].scatter(a.all_mean,a.vcaf_fraction*100,s=21,c=col,alpha=.72,edgecolors='white',linewidths=.25,label=state)
    axes[0].axvline(cuts.state_cut,ls='--',color='.45',lw=1);axes[0].axhline(cuts.abundance_cut*100,ls='--',color='.45',lw=1)
    axes[0].set(xlabel='Mean predicted HLA-ABC score in vCAFs',ylabel='vCAFs / all CAFs (%)',title='HLA state and vCAF abundance')
    axes[0].text(.04,.96,'Spearman ρ = −0.008\n280 patients',ha='left',va='top',transform=axes[0].transAxes,fontsize=10)
    axes[0].legend(frameon=False,loc='upper left',bbox_to_anchor=(.02,.78))
    matrix=np.zeros((2,2))
    for i,ab in enumerate(['Low abundance','High abundance']):
        for j,st in enumerate(['Low HLA','High HLA']):matrix[i,j]=km.loc[f'{ab} | {st}','DFS5']
    axes[1].imshow(matrix,cmap='YlGnBu',vmin=0,vmax=1,aspect='auto')
    for i,ab in enumerate(['Low abundance','High abundance']):
        for j,st in enumerate(['Low HLA','High HLA']):
            row=km.loc[f'{ab} | {st}'];axes[1].text(j,i,f'{100*row.DFS5:.1f}%\nn = {int(row.n)}',ha='center',va='center',fontsize=17,color='white')
    axes[1].set_xticks([0,1],['Low HLA state','High HLA state']);axes[1].set_yticks([0,1],['Lower vCAF\nabundance','Higher vCAF\nabundance'])
    axes[1].set_title('Five-year disease-free survival',pad=16)
    axes[1].set_xticks([-.5,.5,1.5],minor=True);axes[1].set_yticks([-.5,.5,1.5],minor=True);axes[1].grid(which='minor',color='white',lw=3);axes[1].tick_params(which='both',length=0)
    for ax in axes:ax.text(-.12,1.04,'a' if ax==axes[0] else 'b',transform=ax.transAxes,weight='bold',fontsize=15)
    fig.suptitle('Predicted vCAF HLA state adds information beyond cell abundance',fontsize=15,y=.97)
    fig.subplots_adjust(left=.08,right=.98,top=.8,bottom=.23,wspace=.37)
    fig.text(.5,.055,'Current three-dataset RAMP k30. Both patient scores split at their overall cohort medians; each quadrant contains 70 patients.\nAbundance = author-annotated vCAFs / all CAFs. Matrix shows unadjusted Kaplan–Meier estimates; color range 0–100%.',ha='center',fontsize=9,color='.35')
    save(fig,'vCAF_state_abundance_DFS')

def dose():
    c=pd.read_csv(OUT/'CONTINUOUS_HR.csv').query("config=='RAMP_k30'")
    q=pd.read_csv(OUT/'QUARTILE_COX.csv').query("config=='RAMP_k30'")
    d=pd.read_csv(OUT/'PATIENT_INPUTS.csv').query("config=='RAMP_k30'")
    fig,axes=plt.subplots(1,2,figsize=(12.5,5.4))
    axes[0].fill_between(c.score,c.CI_low,c.CI_high,color=HIGH,alpha=.17);axes[0].plot(c.score,c.HR,color=HIGH,lw=2)
    axes[0].axhline(1,ls='--',color='.4',lw=.8);axes[0].axvline(c.reference.iloc[0],ls=':',color='.5',lw=.8)
    axes[0].set(xlabel='Mean predicted HLA-ABC score',ylabel='Adjusted hazard ratio vs median score',title='Continuous association')
    axes[0].plot(d.all_mean,np.repeat(.03,len(d)),'|',color='.6',alpha=.4,transform=axes[0].get_xaxis_transform(),ms=6)
    axes[0].set_xlim(c.score.min(),c.score.max());axes[0].set_ylim(0,max(2.5,c.CI_high.max()*1.06))
    x=np.arange(4)
    axes[1].errorbar(x,q.HR,yerr=[q.HR-q.CI_low,q.CI_high-q.HR],fmt='o',color=HIGH,ms=7,lw=1.8,capsize=4)
    axes[1].axhline(1,ls='--',color='.4',lw=.8);axes[1].set_xticks(x,[f'{r.quartile}\n{r.events}/{r.n} events' for r in q.itertuples()])
    axes[1].set(xlabel='Patient HLA-score quartile',ylabel='Adjusted hazard ratio vs Q1',title='Fixed quartiles; 70 patients each')
    axes[1].set_xlim(-.5,3.5);axes[1].set_ylim(0,q.CI_high.max()*1.15)
    for j,r in enumerate(q.itertuples()):axes[1].text(j,r.HR+.18,f'{r.HR:.2f}',ha='center',fontsize=9)
    for i,ax in enumerate(axes):ax.text(-.12,1.04,'ab'[i],transform=ax.transAxes,weight='bold',fontsize=15)
    fig.suptitle('Higher vCAF HLA states are associated with greater DFS risk',fontsize=15,y=.97)
    fig.subplots_adjust(left=.08,right=.98,top=.82,bottom=.24,wspace=.3)
    fig.text(.5,.035,'Same adjustment: age, sex, categorical stage, CAF fraction, original global CAF score and vCAF abundance.\nLeft: fixed natural cubic spline; displayed 5th–95th score percentiles. Right: categorical quartiles, Q1 reference. Shading/bars: 95% CI.\nQuartiles do not show a strictly increasing sequence; Q3 and Q4 have similar elevated risks.',ha='center',fontsize=8.7,color='.35')
    save(fig,'vCAF_continuous_and_quartile_risk')

def compartments():
    s=pd.read_csv(OUT/'COMPARTMENT_COX.csv').query("config=='RAMP_k30'").set_index('compartment')
    j=pd.read_csv(OUT/'JOINT_COMPARTMENT_COX.csv').query("config=='RAMP_k30' and term=='state_z'").set_index('compartment')
    order=['vCAF','Other CAF','All CAF','Tumour','Immune','Vessel','All non-vCAF cells']
    fig,axes=plt.subplots(1,2,figsize=(13.8,5.6))
    for y,t in enumerate(order[::-1]):
        r=s.loc[t];col=HIGH if t=='vCAF' else '#6993a3'
        axes[0].errorbar(r.HR,y,xerr=[[r.HR-r.CI_low],[r.CI_high-r.HR]],fmt='o',color=col,capsize=3,ms=6,lw=1.6)
    axes[0].set_yticks(range(7),[f'{t} (n={int(s.loc[t,"n"])})' for t in order[::-1]])
    axes[0].set_title('Each compartment’s HLA score',fontsize=12)
    for y,t in enumerate(order[:0:-1]):
        r=j.loc[t];axes[1].errorbar(r.HR,y,xerr=[[r.HR-r.CI_low],[r.CI_high-r.HR]],fmt='o',color=HIGH,capsize=3,ms=6,lw=1.6)
    axes[1].set_yticks(range(6),[f'+ {t} HLA' for t in order[:0:-1]])
    axes[1].set_title('vCAF HLA after including another compartment',fontsize=12)
    for ax in axes:
        ax.axvline(1,color='.4',ls='--',lw=1);ax.set_xlim(.6,2.3);ax.set_xlabel('Adjusted DFS hazard ratio per SD');ax.grid(axis='x',alpha=.15)
    fig.suptitle('Compartment-level prognostic associations',fontsize=15,y=.97)
    fig.subplots_adjust(left=.17,right=.99,top=.81,bottom=.2,wspace=.64)
    fig.text(.5,.045,'Current RAMP k30; same clinical/CAF adjustment. Require ≥15 cells in the evaluated compartment.\nJoint models compare on the same patients. These comparisons do not establish that the association is exclusive to vCAFs.',ha='center',fontsize=9,color='.35')
    save(fig,'HLA_compartment_associations')

def verify_and_report():
    design=json.loads((OUT/'DESIGN.json').read_text())
    for k,path in design['input_paths'].items():assert rt.sha(path)==design['input_sha256'][k],k
    d=pd.read_csv(OUT/'RAMP_k30_STATE_ABUNDANCE.csv');km=pd.read_csv(OUT/'KM_5YEAR.csv').query("id=='RAMP_k30_joint'").set_index('group')
    km_error=0.
    for name,g in d.groupby('joint_group'):
        surv=1.
        for t in np.sort(g.loc[g.event.eq(1)&g.time_years.le(5),'time_years'].unique()):
            surv*=1-((g.time_years==t)&g.event.eq(1)).sum()/(g.time_years>=t).sum()
        km_error=max(km_error,abs(surv-km.loc[name,'DFS5']))
    assert km_error<1e-12
    assert np.all(d.groupby('joint_group').size()==70)
    repro=pd.read_csv(OUT/'REPRODUCTION.csv');assert repro.coef_error.max()<1e-10
    for p in FIG.glob('*.png'):
        with Image.open(p) as im:im.load()
    for p in FIG.glob('*.svg'):ElementTree.parse(p)
    report='''# 生存部分的扩展：预测状态补充细胞数量信息

## 最值得加入故事的结果

**vCAF 的预测 HLA-ABC 状态与 vCAF 数量比例几乎不相关，但在不同数量水平的患者中，都能看到 HLA 状态对应的 DFS 分层。** 因此这部分回答的是“分子状态给作者的细胞注释和数量信息增加了什么”。

保持原来的 280 位患者、96 个 DFS 事件和冻结三数据集预测。主指标仍为每位患者全部作者 vCAF 的预测 HLA 均值，k30 为主。vCAF 丰度定义为作者 vCAF 数 / 全部作者 CAF 数，不能混同于所有组织细胞中的 vCAF 比例。

## 数量与状态

两者 Spearman ρ=−0.008，几乎没有单调相关。分别按全队列中位数划分，数量阈值为 0.0982783，预测 HLA 均值阈值为 0.6669387。正好四组各 70 人：

| vCAF 丰度 | 低预测 HLA 状态 | 高预测 HLA 状态 | 5 年 DFS 差 |
| --- | ---: | ---: | ---: |
| 较低 | 80.0% | 57.0% | 23.0 个百分点 |
| 较高 | 75.2% | 50.2% | 25.0 个百分点 |

每个低 HLA 组 17 个事件，每个高 HLA 组 31 个事件。两行内部的 log-rank p 分别为 0.00579、0.00366。这里没有为各子组重新选择 HLA 阈值，也没有优化丰度阈值。

全体患者的既定调整模型中，HLA 均值每 SD 的 HR=1.395（95% CI 1.112–1.751，p=0.00407），vCAF 丰度每 SD 的 HR=0.979（0.772–1.242，p=0.864）。加入 HLA 状态的似然比检验 p=0.00402。状态与丰度的连续交互检验 p=0.739，未观察到效应随丰度明显变化的证据。

分开在两个 140 人子组重新拟合完整调整模型后，状态 HR 分别为 1.405（p=0.044）和 1.219（p=0.261）。因此不能把两条未调整分层曲线写成“两个子组各自均有独立显著关联”；独立关联的主要依据是预先固定的全体调整模型。

## 连续得分和四分位数

固定四分位，每组 70 人。以 Q1 为参照，调整后 HR：Q2=1.62，Q3=2.89，Q4=2.63；相应事件数为 13、21、34、28。较高两个分位风险更高，但不是严格逐分位递增。

使用固定 10/50/90 百分位结点的自然三次样条作连续展示，非线性检验 p=0.158。此图用于展示风险如何随分数变化，未基于曲线重新挑选阈值。图的横轴显示 5%–95% 分位范围，保留 95% CI。

## 与其他细胞组分的比较

相同调整条件下，k30 的每 SD HR：

| 评分细胞组分 | N | HR | 95% CI | P |
| --- | ---: | ---: | ---: | ---: |
| vCAF | 280 | 1.395 | 1.112–1.751 | 0.0041 |
| 其他 CAF | 280 | 1.233 | 0.879–1.730 | 0.226 |
| 全部 CAF | 280 | 1.444 | 0.988–2.112 | 0.058 |
| 肿瘤 | 279 | 1.149 | 0.912–1.446 | 0.238 |
| 免疫细胞（含 T） | 280 | 1.036 | 0.838–1.280 | 0.746 |
| 血管细胞 | 268 | 1.310 | 1.009–1.699 | 0.042 |
| 全部非 vCAF 细胞 | 280 | 1.056 | 0.856–1.304 | 0.611 |

进一步同时纳入其他组分的 HLA 均值时，vCAF 状态仍保留关联：控制其他 CAF 得分后 HR=1.377，p=0.00848；控制全部非 vCAF 得分后 HR=1.540，p=0.000813。所有联合模型、系数差异检验和 k100 结果均保留。

这些结果支持 vCAF 状态提供额外信息，尚不能宣称只有 vCAF 存在生存关联。单组显著、另一组不显著本身不等于两者效应显著不同；对其他 CAF、肿瘤、血管的联合系数差异检验也没有证明这种排他性。k30 七组分别检验的 BH q 已保存在 COMPARTMENT_COX.csv。

## 主图建议

保留原 KM 作为总体临床结果，增加一个“状态与丰度”panel：左侧患者 HLA 得分与 vCAF 丰度散点，右侧固定 2×2 分层的 5 年 DFS。它让读者看到原有细胞数量信息相近时，预测状态还能提供什么。

连续风险和其他细胞组分对照可用于补充图。这样主线从“转移 → 验证 → 生存相关”扩展为“转移 → 验证 → 提取细胞状态 → 补充数量信息的风险分层”。这属于临床关联的深化，不等价于已经确定 HLA 导致生存差异的生物学机制。

## 文件与复现

- figures/vCAF_state_abundance_DFS.png / .svg：主要新增 panel。
- figures/vCAF_continuous_and_quartile_risk.png / .svg：连续曲线和固定四分位。
- figures/HLA_compartment_associations.png / .svg：组分和联合模型。
- 所有细胞得分来自只读 data_cords，队列来自只读 survival；原空间、生存、预测均未修改。
- `bash run.sh` 可重新生成扩展分析；模型运行在独立 R 临时目录，Python 写入保护限制在 survival_extension。
'''
    (BASE/'README_ZH.md').write_text(report)
    rt.save(OUT/'VERIFICATION.json',dict(status='pass',finished_at_utc=rt.now(),input_hashes_unchanged=True,n_patients=len(d),quadrant_sizes=[70,70,70,70],independent_KM_max_error=km_error,original_Cox_max_error=float(repro.coef_error.max()),pngs=3,svgs=3,manual_visual_review='pending'))
    print('Three figures and Chinese report complete; original Cox and four-group KM verified.')

def main():state_abundance();dose();compartments();verify_and_report()
if __name__=='__main__':main()
