from pathlib import Path
import json
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/hnscc_spatial_score_review_mpl')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
import fitz
from analyze_spatial_scores import HERE, ROOT, sha

LOW,HIGH,INK='#3574A3','#C34D42','#18232C'
plt.rcParams.update({'font.family':'Arial','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none','axes.labelcolor':INK,'text.color':INK,'axes.spines.top':False,'axes.spines.right':False})

def header(fig,x,y,letter,title):
    fig.text(x,y,letter,fontsize=19,weight='bold',va='top')
    fig.text(x+.031,y-.005,title,fontsize=12,weight='bold',va='top')

def pairs(ax,data,labels,scale=1.):
    rng=np.random.default_rng(20260913)
    for x,(key,label) in enumerate(labels):
        block=data[data['label'].eq(key)]
        assert len(block)==74
        v=block[['low','high']].to_numpy()*scale
        jit=rng.uniform(-.035,.035,len(v))
        for j,row in enumerate(v):ax.plot([x-.16+jit[j],x+.16+jit[j]],row,color='#B9C2C8',lw=.45,alpha=.35,zorder=1)
        for i,(name,color,offset) in enumerate([('low',LOW,-.16),('high',HIGH,.16)]):
            ax.boxplot([v[:,i]],positions=[x+offset],widths=.23,patch_artist=True,showfliers=False,manage_ticks=False,
                       boxprops=dict(facecolor='white',edgecolor=color,linewidth=1.),medianprops=dict(color=color,lw=1.4),whiskerprops=dict(color=color,lw=.8),capprops=dict(color=color,lw=.8))
            ax.scatter(x+offset+jit,v[:,i],s=7,alpha=.4,color=color,zorder=3,edgecolors='none')
    ax.set_xticks(range(len(labels)),[v[1] for v in labels]);ax.tick_params(axis='x',length=0,pad=7)
    ax.grid(axis='y',color='#E8ECEF',lw=.55);ax.set_axisbelow(True)
    for s in ['bottom','left']:ax.spines[s].set_color('#83919C');ax.spines[s].set_linewidth(.6)

def main():
    d=pd.read_csv(HERE/'codex_measured_protein_cells.csv.gz',dtype={'patient':str})
    roi=d[d.unique_region.eq('522_S2reg054')].copy();assert len(roi)==650
    dapi_path=ROOT/'figure_HNSCC/spatial_return_survival/input/planes/DAPI.raw'
    hash_before=sha(dapi_path)
    dapi=np.fromfile(dapi_path,dtype='>u2').reshape(1440,1920).astype(float)
    positive=dapi[dapi>0];im=np.clip(dapi/np.percentile(positive,99.8),0,1)**.72
    # Real DAPI plane; marker values appear only as cell-centroid overlays.
    fig=plt.figure(figsize=(11.,7.3))
    for left,letter,title,column in [(.055,'A','Predicted CD39/TIGIT state','within_patient_rank'),(.555,'B','Measured LAG3','LAG3_patient_rank')]:
        header(fig,left-.025,.98,letter,title)
        ax=fig.add_axes([left,.53,.40,.385]);ax.imshow(im*.45,cmap='gray',vmin=0,vmax=1,extent=(0,1920,1440,0))
        sc=ax.scatter(roi.x,roi.y,c=roi[column],s=8.5,cmap='viridis',vmin=0,vmax=1,lw=.1,edgecolors='#D0D0D0')
        ax.set(xlim=(0,1920),ylim=(1440,0));ax.set_axis_off()
    cax=fig.add_axes([.465,.585,.009,.22]);cb=fig.colorbar(sc,cax=cax,ticks=[0,.5,1]);cb.ax.tick_params(labelsize=8,length=2,pad=2)
    cb.set_label('Within-patient percentile',fontsize=8,labelpad=5)
    rho=spearmanr(roi.predicted_score,roi.LAG3).statistic
    fig.text(.055,.51,'Same complete ROI | 650 CD8 cells | Cell overlays on measured DAPI',fontsize=9.)
    fig.text(.555,.51,f'Example Spearman ρ = {rho:.2f}; cohort mean ρ = 0.52',fontsize=9.)
    header(fig,.03,.452,'C','Measured CODEX phenotype')
    header(fig,.53,.452,'D','Spatial neighborhood')
    ax=fig.add_axes([.075,.105,.39,.27]);d1=pd.read_csv(HERE/'codex_protein_patient_contrasts.csv');d1=d1[d1.aggregation.eq('patient')].rename(columns={'marker':'label'})
    pairs(ax,d1,[('LAG3','LAG3'),('IL10','IL10'),('GranzymeB','Granzyme B'),('Ki67','Ki67')])
    ax.set(ylim=(0,1),yticks=[0,.25,.5,.75,1],ylabel='Measured protein percentile')
    ax=fig.add_axes([.575,.105,.38,.27]);d2=pd.read_csv(HERE/'cell_neighbor_patient_contrasts.csv');d2=d2[(d2.k==15)&d2.metric.eq('raw_fraction')&d2.aggregation.eq('patient')].rename(columns={'endpoint':'label'})
    pairs(ax,d2,[('Tumor','Tumor'),('CAF','CAF'),('Macrophage','Macrophage')],100)
    ax.set(ylim=(0,70),yticks=[0,20,40,60],ylabel='Neighbor fraction (%)')
    fig.legend(handles=[Line2D([],[],marker='o',color=c,lw=0,label=t,markersize=4) for c,t in [(LOW,'Lower 40%'),(HIGH,'Higher 40%')]],loc='center',bbox_to_anchor=(.52,.41),ncol=2,frameon=False,fontsize=9)
    fig.text(.075,.027,'74 patients; one paired value per patient. The maps show an example, not ground truth for CD39/TIGIT.',fontsize=9.)
    for ext in ['png','pdf','svg']:fig.savefig(HERE/f'spatial_phenotype_candidate.{ext}',dpi=230,facecolor='white')
    plt.close(fig)
    pdf=fitz.open(HERE/'spatial_phenotype_candidate.pdf');pdf[0].get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(HERE/'pdf_preview.png');pdf.close()
    assert sha(dapi_path)==hash_before
    roi.to_csv(HERE/'fixed_example_cells.csv',index=False)
    summary=pd.read_csv(HERE/'cell_neighbor_summary.csv')
    tumor=summary[(summary.k==15)&summary.endpoint.eq('Tumor')&summary.metric.eq('raw_fraction')&summary.aggregation.eq('within_roi_equal')].iloc[0]
    report=f'''# HNSCC 空间结尾复核：有可用表型支持，但原 niche 分类仍弱

## 结论

推荐新增的是“**返回 CODEX 后的实测分子表型 + 肿瘤邻域偏好**”。本轮没有找到能把原 CAF/macrophage niche 预测改成强分类器的新 spatial score。固定区域聚合也未出现强相关。

较有力的新复核结果是：原 CD8 专用模型的 CD39/TIGIT 预测状态与 CODEX 实测 LAG3 具有一致关联。这个结果可以接在 CITE-seq 的分子发现之后，作为目标组织中的辅助表型印证。LAG3 不在 shared22 中，也没有加入返回状态的定义；但它与 shared 蛋白相关，因此不是直接测到了 CODEX 中缺失的 CD39/TIGIT。

## 同一个正式模型，在 CODEX 中实测什么

固定 38,299 个 CD8、74 位患者、患者内预测 top40% 与 bottom40%。下列均为每患者先汇总、再患者等权。

| CODEX 实测蛋白 | 低状态平均百分位 | 高状态平均百分位 | 高 > 低患者 | 同 ROI 比较的平均差 |
|---|---:|---:|---:|---:|
| LAG3 | 0.338 | 0.652 | 74/74 | +0.306，73 位可估计患者 |
| IL10 | 0.368 | 0.625 | 74/74 | +0.251，73 位 |
| Granzyme B | 0.394 | 0.601 | 73/74 | +0.199，73 位 |
| Ki67 | 0.416 | 0.576 | 71/74 | +0.164，73 位 |

这些数值是蛋白在患者内（同 ROI 分析则在 ROI 内）的百分位，不是蛋白强度或阳性比例。不能将 0.338→0.652 写成蛋白量增加 1.93 倍。缺少足够高低 CD8 的 ROI 未参与同 ROI 估计，资格表与脚本完整保留。

LAG3 与连续预测分数的患者内 Spearman：平均 **0.522**，95% 患者 bootstrap CI **0.496–0.550**，中位 **0.533**，74/74 正向。只保留 CD8≥40 的 70 位患者，平均仍为 **0.517**。

在上述 70 位患者中控制 ROI 和 shared22 对齐表达的平均强度后，LAG3 平均偏秩相关为 **0.220 [0.194, 0.248]**，67/70 正向。对应 IL10 为 0.185，Granzyme B 为 0.113，Ki67 为 0.078。调整后减弱说明相当一部分未调整相关与广泛蛋白共变有关；不能只引用 0.52 来声称独立预测精度。

旧全类型分析的蛋白表使用全细胞参考分组；本轮重新连接原始 CODEX 表并用当前正式 CD8 专用模型计算，因此不可将两版数值混用。原表与缓存蛋白值逐细胞核对一致。

## 空间位置补充

每个 CD8 周围原 15 个非自身最近邻，肿瘤定义为作者 tumor + tumor_CA9。

- 全部 74 位患者：低状态 CD8 的肿瘤邻居平均 **9.59%**，高状态 **18.10%**；差 **8.51 个百分点 [6.49, 10.60]**，65 位正向、5 位负向、4 位并列。两个人群平均比例之比为 1.89；它不是患者风险比或分类准确率。
- 在同 ROI 内比较、再 ROI 等权汇总至患者：**10.02%→17.00%**；差 **{tumor['mean']*100:.2f} 个百分点 [{tumor['ci_low']*100:.2f}, {tumor['ci_high']*100:.2f}]**，62/73 正向。说明此方向并非完全由不同 ROI 的组成差异产生。
- 同 ROI 的 CAF 差 **+1.29 个百分点**；macrophage 差 **−0.09 个百分点**，没有增加。因此不能将结尾写成高状态 CD8 特异聚集于 macrophage niche。
- 以上高低分组未改变，图中没有按空间效应挑患者。

## 区域 score 尝试：结果仍有限

原点 (0,0)、边长 200 px（100 μm）网格：每格至少 10 个 CD8 和 30 个非 CD8。每患者至少 5 个合格格子才计算相关。37 位患者可以估计，另 37 位信息不足；不能写成 74 人的区域验证。

- 区域平均预测状态 vs 区域 CD8 原连续 niche 均值：平均 Spearman **0.229**；同 ROI 中心化秩后 **0.234**，36 位可估计患者。
- 区域平均预测状态 vs 实测非 CD8 中 CAF + macrophage 比例：平均 **0.223**；同 ROI 调整后 **0.175**。
- 200/400 px 与半格平移的完整结果均保存。未按最佳尺度更改主结果。
- 100 px 网格没有任何格子同时满足上述细胞数要求，其结果是不可估计，非零效应。全部六个配置的逐患者资格表仍包括 74 位患者。

网格尺度的物理换算依据 Haist 原文第 21 页的 `100px = 50 μm`。这是探索性区域关联，与原逐细胞 AUROC=0.530 不是同一任务；原 AUROC 没有被提升。

## 讨论用图及建议

`spatial_phenotype_candidate.png/.pdf/.svg` 是四部分讨论图，未覆盖正式 Figure 或 manuscript。

- A/B：继续用之前的完整 ROI 522_S2reg054（患者 15219），650 个 CD8；未重新按本次 LAG3 表现选择 ROI。真实 DAPI 图像上叠加真实细胞中心，分别显示患者内预测 CD39/TIGIT 状态百分位与实测 LAG3 百分位。此 ROI 的 Spearman 为 **{rho:.3f}**；示例不是新的最佳 ROI。背景是真实图像，颜色圆点是细胞级数值覆盖层，并非原始 LAG3 荧光图。
- C：四个未共享 CODEX 蛋白的患者配对高低对比；图中为真实 74 位患者。
- D：同样 74 位患者的肿瘤、CAF、macrophage 邻居比例。完整同 ROI 敏感性结果见表。

如果采纳主图，建议 E 保留紧凑的 CITE-seq protein/RNA 表征，F 改为 **Spatial localization and measured phenotype**：优先 LAG3 的目标端印证，配合肿瘤邻接结果。可减少 RNA 在主图中的面积，把细节留在补充材料。不要使用“准确恢复 CAF/macrophage niche”的标题，也不据此推断 CD39/TIGIT 导致生存变差。

解释上应写“检查点相关状态，同时保留部分效应/增殖特征，并偏向肿瘤邻域”。LAG3/IL10/Granzyme B/Ki67 同向不支持简单称为完全失能的终末耗竭细胞；功能因果解释需要额外实验证据。

## 可复现文件

- `PLAN_ZH.md`：问题、固定规则、追加表型检验说明。
- `analyze_spatial_scores.py`：旧空间缓存核对、患者/ROI 邻域对比、全部网格尺度。
- `check_codex_phenotype.py`：原始实测蛋白核对与当前 CD8 模型的表型检验。
- `plot_and_report.py`：讨论图与本报告。
- `regional_summary.csv` / `all74_grid_eligibility.csv` / `all_grid_tiles.csv.gz`：所有区域结果。
- `cell_neighbor_summary.csv` / `cell_neighbor_patient_contrasts.csv`：所有邻域方向。
- `codex_protein_contrast_summary.csv` / `codex_protein_correlation_summary.csv`：全部四个蛋白结果。
- `AUDIT.json`、`PROTEIN_AUDIT.json`：输入文件 SHA256 前后一致，正式预测逐细胞复现。

本轮没有读取生存结局、重新对齐、改返回模型、扫高低切点或替换正式图。复核和探索均使用已有数据，尚未构成外部独立验证。
'''
    (HERE/'RESULTS_ZH.md').write_text(report)
    (HERE/'PLOT_AUDIT.json').write_text(json.dumps(dict(status='PASS',example_roi='522_S2reg054',example_cells=len(roi),example_spearman=rho,dapi_sha256=hash_before,dapi_unchanged=True,synthetic_image=False,cell_overlays=True,official_figure_modified=False),indent=2)+'\n')

if __name__=='__main__':main()
