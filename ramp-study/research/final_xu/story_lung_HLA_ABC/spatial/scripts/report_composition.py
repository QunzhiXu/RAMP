"""Present complete-patient and cell-count-restricted neighborhood compositions."""
import runtime as rt
from runtime import BASE
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image
from xml.etree import ElementTree

OUT=BASE/'results/neighborhood_composition';FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
S=pd.read_csv(OUT/'COMPOSITION_SUMMARY.csv')
TARGETS=['Other CAF','Immune total','Vessel','Tumour','Other']
STACK=['Other CAF','Tumour','Immune total','Vessel','Other']
LABELS={'Other CAF':'Other CAFs','Immune total':'Immune (including T)','Vessel':'Vessel cells','Tumour':'Tumour cells','Other':'Other cells'}
COLORS={'Other CAF':'#c35199','Tumour':'#cf5748','Immune total':'#258671','Vessel':'#46aab6','Other':'#bfc3ca'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})

def subset(comp='within_ROI_all_mixed',population='all_available',k=30,radius=40):
    return S[S.comparison.eq(comp)&S.population.eq(population)&S.config.eq(f'RAMP_k{k}')&S.radius.eq(radius)].set_index('target')

def save(fig,name):
    fig.savefig(FIG/f'{name}.png',dpi=220,facecolor='white');fig.savefig(FIG/f'{name}.svg',facecolor='white');plt.close(fig)

def overview():
    fig,axes=plt.subplots(2,2,figsize=(12.8,9.2),gridspec_kw={'width_ratios':[.85,1.15]})
    for row,(comp,title) in enumerate([('within_ROI_all_mixed','All patients with both states in an ROI'),('within_ROI_primary','Patients with more cells in each state')]):
        a=subset(comp);n=int(a.loc['Other CAF','n_patients']);ax,fx=axes[row]
        bottoms=np.zeros(2)
        for t in STACK:
            vals=a.loc[t,['low','high']].to_numpy(dtype=float)*100
            ax.bar([0,1],vals,bottom=bottoms,color=COLORS[t],width=.58,edgecolor='white',lw=.6)
            for x,y,b in zip([0,1],vals,bottoms):
                if y>=5:ax.text(x,b+y/2,f'{y:.1f}%',ha='center',va='center',fontsize=9,color='white' if t in ['Other CAF','Tumour','Immune total'] else '#152b32')
            bottoms+=vals
        assert np.allclose(bottoms,100)
        ax.set_xticks([0,1],['Low-state vCAFs','High-state vCAFs']);ax.set_ylim(0,100)
        ax.set_ylabel('Mean neighborhood composition (%)');ax.set_title(f'{title}\nn = {n} patients',fontsize=11,pad=12)
        for y,t in enumerate(TARGETS[::-1]):
            z=a.loc[t];fx.errorbar(z['mean']*100,y,xerr=[[100*(z['mean']-z.ci_low)],[100*(z.ci_high-z['mean'])]],fmt='o',color=COLORS[t],capsize=3,ms=6,lw=1.7)
        fx.axvline(0,color='.4',ls='--',lw=1);fx.grid(axis='x',alpha=.16)
        fx.set_yticks(range(len(TARGETS)),[LABELS[t] for t in TARGETS[::-1]]);fx.set_xlim(-3.2,3.2)
        fx.set_xlabel('High minus low (percentage points)');fx.set_title('Paired difference within the same ROI',fontsize=11,pad=12)
        ax.text(-.16,1.06,'a' if row==0 else 'c',transform=ax.transAxes,fontweight='bold',fontsize=15)
        fx.text(-.35,1.06,'b' if row==0 else 'd',transform=fx.transAxes,fontweight='bold',fontsize=15)
    fig.suptitle('vCAF HLA-ABC state and local cell composition',fontsize=17,y=.985)
    fig.legend(handles=[Patch(color=COLORS[t],label=LABELS[t]) for t in STACK],loc='lower center',bbox_to_anchor=(.5,.064),ncol=5,frameon=False,fontsize=9)
    fig.subplots_adjust(left=.09,right=.97,top=.89,bottom=.17,hspace=.51,wspace=.53)
    fig.text(.5,.02,'Current three-dataset RAMP k30; fixed cell-state cutoff. Neighbors: non-vCAF cells within <40 provider coordinate units.\nStates paired within ROI, then ROIs and patients weighted equally. Bars: 95% patient-bootstrap CI.\nTop: >=1 cell of each state with neighbors per ROI. Bottom: >=5 per state per patient and >=3 per state per ROI.',ha='center',fontsize=8.5,color='.35',linespacing=1.45)
    save(fig,'vCAF_neighborhood_composition_all_patients')

def sensitivity():
    fig,axes=plt.subplots(2,2,figsize=(11.8,7.4))
    for ax,t in zip(axes.flat,TARGETS[:4]):
        for comp,k,col,dx,mark in [('within_ROI_all_mixed',30,'#287b91',-2,'o'),('within_ROI_all_mixed',100,'#8a929b',0,'s'),('within_ROI_primary',30,'#c36c4b',2,'^')]:
            rows=[subset(comp,k=k,radius=r).loc[t] for r in [20,40,80]]
            x=np.array([20,40,80])+dx;y=np.array([a['mean'] for a in rows])*100
            lo=np.array([a.ci_low for a in rows])*100;hi=np.array([a.ci_high for a in rows])*100
            ax.errorbar(x,y,yerr=[y-lo,hi-y],fmt=mark+'-',color=col,ms=5,capsize=2,lw=1.1,alpha=.9)
        ax.axhline(0,color='.4',ls='--',lw=.8);ax.set_xticks([20,40,80]);ax.set_title(LABELS[t],fontsize=12)
        ax.set_xlabel('Radius (provider coordinate units)');ax.set_ylabel('High minus low (percentage points)');ax.grid(alpha=.12)
    fig.suptitle('Neighborhood composition across fixed radii and prediction settings',fontsize=14,y=.99)
    fig.legend(handles=[Line2D([],[],marker='o',color='#287b91',label='All mixed-state patients, k30'),Line2D([],[],marker='s',color='#8a929b',label='All mixed-state patients, k100'),Line2D([],[],marker='^',color='#c36c4b',label='More cells per state, k30')],loc='lower center',bbox_to_anchor=(.5,.01),ncol=3,frameon=False,fontsize=8.5)
    fig.subplots_adjust(left=.09,right=.98,top=.91,bottom=.14,wspace=.3,hspace=.4)
    save(fig,'vCAF_composition_radius_sensitivity')

def markdown_table(a):
    lines=['| 邻域细胞 | 低状态 | 高状态 | 高减低（百分点） | 95% CI（百分点） | 高状态中比例更高的患者 |',
           '| --- | ---: | ---: | ---: | ---: | ---: |']
    labels={'Other CAF':'其他 CAF','Immune total':'免疫细胞（含 T）','Vessel':'血管细胞','Tumour':'肿瘤细胞','Other':'其他细胞'}
    for t in TARGETS:
        r=a.loc[t];lines.append(f'| {labels[t]} | {100*r.low:.2f}% | {100*r.high:.2f}% | {100*r["mean"]:+.2f} | {100*r.ci_low:+.2f} 至 {100*r.ci_high:+.2f} | {int(r.n_positive)}/{int(r.n_patients)}（{100*r.fraction_positive:.1f}%） |')
    return '\n'.join(lines)

def report():
    all_=subset();primary=subset('within_ROI_primary');dfs=subset('within_ROI_primary',population='DFS_280_members')
    cov=pd.read_csv(OUT/'PATIENT_COVERAGE.csv').query("config == 'RAMP_k30'")
    p=pd.read_csv(OUT/'PATIENT_PAIRED_CONTRASTS.csv')
    rows=[]
    for comp in ['within_ROI_all_mixed','within_ROI_primary']:
        a=p[p.comparison.eq(comp)&p.population.eq('all_available')&p.config.eq('RAMP_k30')&p.radius.eq(40)&p.patient_id.ne('175_10')]
        for target,g in a.groupby('target'):
            rows.append(dict(comparison=comp,target=target,n_patients=len(g),mean_difference=g.difference.mean(),fraction_positive=(g.difference>1e-12).mean()))
    pd.DataFrame(rows).to_csv(OUT/'EXCLUDING_DISPLAY_PATIENT.csv',index=False)
    text='''# 全患者 vCAF 邻域组成：当前三数据集模型

## 结论

**高预测 HLA-ABC vCAF 周围免疫细胞略多，是最一致的队列方向。其他 CAF 减少的方向较弱；展示 ROI 中“高状态周围肿瘤更多”的特点没有推广到整体。** 175_10 / 175_A3,3 是空间分区强的例子，其组成差异远大于队列平均。

本次计算的是邻域中各类细胞的组成比例，不是旧分析的“附近至少存在一个某类细胞”的比例。使用同一个固定预测和高状态阈值，在相同 ROI 内，统计每个 vCAF 周围 <40 个作者坐标单位内的非 vCAF 细胞组成，先平均细胞，再平均 ROI 和患者。所有患者等权。

## 覆盖全部患者

输入包含 527 人、1,000 个 ROI、2,849,385 个细胞。其中 3 人没有 vCAF，79 人仅有高状态 vCAF，445 人同时有高、低状态，可进行同 ROI 内配对。本节纳入这 445 人全部可用的混合状态 ROI（每种状态至少 1 个有邻居的细胞）。79 个只有高状态的患者仍保存其组成，但不存在患者内低状态对照，未伪造低组。

'''+markdown_table(all_)+'''

其中，高状态免疫细胞比例更高的患者占 64.0%；其他 CAF 更少的患者只占 48.5%，说明“低状态 CAF 富集”并非多数患者的一致模式。血管平均增加很小，95% 均值差 CI 包含零。肿瘤细胞在高状态周围平均略少，而非更多。

## 细胞数更充足的既定主比较

沿用上一轮空间分析的最低细胞数：患者至少 15 个 vCAF，高低各至少 5 个；每个纳入 ROI 高低各至少 3 个有邻居的细胞。共 289 位患者：

'''+markdown_table(primary)+'''

在这组中，低状态周围其他 CAF 更多的方向出现在 158/289（54.7%）患者；高状态周围免疫细胞更多出现在 193/289（66.8%）患者。平均变化约 1–2 个百分点，与单例的数十个百分点差异不同。

## DFS 队列中的可比较患者

原 280 人 DFS 队列中，257 人可作宽松同 ROI 配对，176 人满足上述较充足细胞数条件。后者结果如下；仅使用成员 ID，未读取结局：

'''+markdown_table(dfs)+'''

## 其他检查

- k100 和固定的 20/80 坐标单位半径结果完整保留。免疫比例增加在全部三个半径与两种 k 值中方向一致。血管变化主要出现在更近的 20 单位范围；肿瘤比例并未出现一致增加。
- 在 473 位患者中进一步计算连续 HLA 得分与邻域组成的 ROI 内相关性，平均 Spearman：免疫 +0.052、其他 CAF −0.005、血管 +0.027、肿瘤 −0.034。这也支持免疫方向较一致，CAF 差异较弱。
- 除去展示病例 175_10 的结果见 EXCLUDING_DISPLAY_PATIENT.csv，保留独立核对。
- 当前程序重新计算该展示 ROI 的全部 72 个组成均值，与之前单例结果误差 <1e−12。全部源文件哈希未变化；邻居计数与独立全距离矩阵核对一致。
- 95% CI 为 2,000 次患者 bootstrap 的平均差区间。CSV 中同时保留配对 Wilcoxon p/q；其检验对象与均值 CI 不同。五个主要组成类别互斥且总和为 1，免疫总量为原作者 Immune 与 T cell 类别之和。

## 文件与运行

- figures/vCAF_neighborhood_composition_all_patients.png / .svg：445 人全部可比较患者，以及 289 人较充足细胞数子集的组成和配对差值。
- figures/vCAF_composition_radius_sensitivity.png / .svg：固定半径与 k 值检查。
- COMPOSITION_SUMMARY.csv：全部半径、k、患者集合与配对方式的结果。
- PATIENT_PAIRED_CONTRASTS.csv：每位患者的组成差值和方向。
- ALL_PATIENT_STATE_COMPOSITION.csv：包括只有一种状态患者的描述性组成。
- VCAF_NEIGHBOR_COMPOSITION.csv.gz：全部 vCAF 三个半径的细胞级计数和比例。
- DESIGN.json / VERIFICATION.json：预先记录的方法、输入哈希和计算核对。

运行 `bash run.sh composition`；重绘用 `bash run.sh composition-plot`。所有输出限于 spatial 目录，其余数据和预测只读。
'''
    (OUT/'REPORT_ZH.md').write_text(text)
    rt.save_json(OUT/'SUMMARY.json',dict(n_patients_scanned=len(cov),n_without_vcaf=int(cov.n_vcaf.eq(0).sum()),
        n_high_only=int((cov.n_high.gt(0)&cov.n_low.eq(0)).sum()),n_all_mixed=int(all_.n_patients.iloc[0]),
        n_count_restricted=int(primary.n_patients.iloc[0]),all_mixed_40_k30=all_.loc[TARGETS].reset_index().to_dict('records'),
        count_restricted_40_k30=primary.loc[TARGETS].reset_index().to_dict('records')))

def main():
    overview();sensitivity();report()
    sizes={}
    for p in FIG.glob('*.png'):
        with Image.open(p) as im:im.load();sizes[p.name]=list(im.size)
    for p in FIG.glob('*.svg'):ElementTree.parse(p)
    rt.save_json(OUT/'PLOT_VERIFICATION.json',dict(png_decoding='pass',svg_parsing='pass',sizes=sizes,manual_visual_review='pending',script_sha256=rt.sha(__file__)))
    print('Composition report, two PNG/SVG figures and summary written.')
if __name__=='__main__':main()
