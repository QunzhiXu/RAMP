"""Summarize fixed-scale spatial architecture exploration."""
import runtime as rt
from runtime import BASE
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
OUT=BASE/'results/spatial_architecture';FIG=OUT/'figures';FIG.mkdir(exist_ok=True)

def main():
    s=pd.read_csv(OUT/'DISTANCE_DECAY_SUMMARY.csv');r=pd.read_csv(OUT/'ROI_GEOMETRIC_STRUCTURE.csv')
    b=pd.read_csv(OUT/'PATIENT_BETWEEN_ROI_STRUCTURE.csv')
    fig,axes=plt.subplots(1,3,figsize=(14.4,4.9),gridspec_kw={'width_ratios':[1.25,1,1]})
    plt.rcParams.update({'svg.fonttype':'none','pdf.fonttype':42})
    for k,col,dx in [(30,'#297f93',-.05),(100,'#9199a2',.05)]:
        a=s[s.config.eq(f'RAMP_k{k}')].sort_values('lo')
        y=a.mean_reduction.to_numpy()*100;lo=a.ci_low.to_numpy()*100;hi=a.ci_high.to_numpy()*100
        axes[0].errorbar(np.arange(5)+dx,y,yerr=[y-lo,hi-y],fmt='o-',color=col,lw=1.6,ms=5,capsize=3,label=f'k={k}')
    axes[0].axhline(0,ls='--',color='.4',lw=.8);axes[0].set_xticks(range(5),['<20','20–40','40–80','80–160','≥160'])
    axes[0].set(xlabel='Pair distance (provider coordinate units)',ylabel='Reduction in score difference (%)',title='Short-range similarity decays with distance')
    axes[0].legend(frameon=False);axes[0].grid(alpha=.14)
    a=r[r.config.eq('RAMP_k30')&r.spatial_auc.notna()]
    axes[1].hist(a.spatial_auc,bins=np.linspace(0,1,21),color='#57a2a8',edgecolor='white',lw=.5)
    axes[1].axvline(.5,ls='--',color='.4',lw=.8)
    axes[1].set(xlabel='Spatial neighbor-state AUC',ylabel='ROIs',title=f'Strong segregation in a subset\n{len(a)} eligible ROIs')
    axes[1].text(.98,.95,f'{int(a.spatial_auc.ge(.8).sum())} ROIs with AUC ≥0.8',transform=axes[1].transAxes,ha='right',va='top',fontsize=9)
    a=b[b.config.eq('RAMP_k30')]
    axes[2].hist(a.mean_range,bins=np.linspace(0,.24,25),color='#c7836b',edgecolor='white',lw=.5)
    axes[2].set(xlabel='Range of ROI mean predicted HLA score',ylabel='Patients',title=f'Variation between sampled ROIs\n{len(a)} patients with ≥2 eligible ROIs')
    axes[2].axvline(a.mean_range.median(),color='#754b39',ls='--',lw=1)
    axes[2].text(.98,.95,f'Median {a.mean_range.median():.3f}',transform=axes[2].transAxes,ha='right',va='top',fontsize=9)
    for i,ax in enumerate(axes):
        ax.spines[['top','right']].set_visible(False);ax.text(-.15,1.05,'abc'[i],transform=ax.transAxes,weight='bold',fontsize=15)
    fig.suptitle('Spatial architecture of predicted HLA-ABC in vCAFs',fontsize=16,y=.99)
    fig.subplots_adjust(left=.07,right=.99,top=.79,bottom=.24,wspace=.35)
    fig.text(.5,.035,'Fixed three-dataset predictions. Left: compare each distance bin with all-pair differences inside the same ROI; patients equally weighted.\nMiddle: leave-self-out state agreement with six nearest vCAFs. Right: compare separate ROIs within patient; no distances across ROIs.',ha='center',fontsize=8.8,color='.35')
    for suffix in ['png','svg']:fig.savefig(FIG/f'spatial_architecture_overview.{suffix}',dpi=220,facecolor='white')
    plt.close(fig)
    report='''# vCAF 空间结构：后续探索

## 哪一种模式更一致？

相较于 1–2 个百分点的细胞邻域组成差异，较一致的空间现象是**相邻 vCAF 的预测 HLA 状态更相似，而且相似性随距离增加而减弱**。该模式是温和的短距离空间连续性，并非全体患者都具有强烈高低区域分隔。

## 距离衰减

每个 ROI 至少 30 个 vCAF；统计同 ROI 内所有无序细胞对。在每个距离区间内至少 10 对细胞，比较其预测得分绝对差与该 ROI 全部细胞对的平均绝对差。后者是随机打乱细胞预测分数后的精确期望。先平均 ROI 比值，再平均患者，因此“平均减少百分比”不等于两个总体均值直接相除。

| 距离（作者坐标单位） | k30：得分差减少 | 95% 患者 bootstrap CI | 患者数 |
| --- | ---: | ---: | ---: |
'''
    for row in s[s.config.eq('RAMP_k30')].sort_values('lo').itertuples():
        report+=f'| {row.distance_bin} | {100*row.mean_reduction:.1f}% | {100*row.ci_low:.1f}% 至 {100*row.ci_high:.1f}% | {row.n_patients} |\n'
    report+='''
距离 <20 时，76.1% 患者的局部差异小于随机期望。k100 下平均减少 11.9%，与 k30 的 10.9% 方向一致。绝对得分差的患者均值为 0.127（近邻）和 0.145（同 ROI 随机细胞对），不能把 10.9% 解释成蛋白表达提高或降低 10.9%。80–160 单位时相似性接近消失。

## 高低状态是否形成可分辨区域？

对每个 vCAF，使用其真实坐标上的 6 个最近其他 vCAF 的高状态比例，描述高低状态能否按空间邻居分开。固定高低阈值不变；这是描述空间结构的 AUC，不是再次验证 HLA 转移准确度。每个 ROI 至少 30 个 vCAF、高低各至少 5 个，332 个 ROI（242 位患者）可评估。

- k30 的 ROI AUC 中位数为 0.549。
- 45/332 个 ROI 的 AUC ≥0.7；10/332 个 ROI ≥0.8。
- 499 次 ROI 内状态置换、BH 校正后，30 个 ROI q<0.05。
- 175_10 / 175_A3,3 的 AUC=0.920，属于最强例子，不能用它代替全队列水平。
- 另见 88_377 / 88_A7,5（0.884）、175_141 / 175_C1,2（0.860）、175_14 / 175_A3,4（0.828，230 个 vCAF）。完整名单保留于 ROI_GEOMETRIC_STRUCTURE.csv。

固定五折交叉验证的线性平面 score ~ x+y 检查大尺度单向梯度。在 582 个 ROI 中只有 4 个的留出 R² ≥0.25；175_10 / 175_A3,3 为 0.517。因此明显的上下梯度属于部分组织区域的特征。

## 同一患者不同 ROI 的状态变化

310 位患者有至少两个各含 15 个 vCAF 的 ROI。不同 ROI 平均预测得分的最大差，中位数为 0.036；随机分配细胞至保持原细胞数的 ROI 后，差值期望的中位数为 0.021。28/310 位患者的 ROI 均值差 ≥0.1。最大例子 86_68 的两个 ROI 均值约为 0.441 和 0.668，差 0.226。

总体上，这也是部分患者具有明显区域差异，未出现全体患者都强烈分区的现象。没有把不同 ROI 的局部坐标拼在一起计算距离。

## 适合的表述

可描述为“预测 HLA 状态具有短距离空间连续性，并在部分组织区域呈现明显空间分区”。目前没有发现足够强、普遍的“高状态固定偏向某类邻居”的新模式。主图可保留强例子展示空间异质性，生存关联继续作为主要结果。

## 可复现文件

脚本为 ../../scripts/explore_architecture.py 和 ../../scripts/report_architecture.py；输入为前一轮冻结的全部 52,328 个作者 vCAF 预测和坐标。方法与输入哈希记录在 DESIGN.json，全部输出仅位于 spatial。运行：

```bash
bash run.sh architecture
```

各 ROI、各患者的完整距离、几何和区域结果均保存为 CSV。图为 figures/spatial_architecture_overview.png / .svg。所有结果为新增空间描述；未重新训练 HLA 预测器、变更阈值或读取临床结局。
'''
    (OUT/'REPORT_ZH.md').write_text(report)
    with Image.open(FIG/'spatial_architecture_overview.png') as im:im.load();size=list(im.size)
    rt.save_json(OUT/'PLOT_VERIFICATION.json',dict(png_decoding='pass',size=size,manual_visual_review='pending',script_sha256=rt.sha(__file__)))
    print('Spatial architecture report and figure complete.')
if __name__=='__main__':main()
