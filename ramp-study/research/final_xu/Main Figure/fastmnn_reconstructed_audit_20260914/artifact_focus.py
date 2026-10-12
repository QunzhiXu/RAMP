"""Artifact-focused counts, examples and report, with explicit denominators."""
from pathlib import Path
import os,sys,json
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
os.environ['MPLCONFIGDIR']=str(HERE/'.cache/matplotlib')
os.environ['MPLBACKEND']='Agg'
import run_audit as a
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

NAMES=dict(zip(a.CASES,['Bone marrow','TNBC','Melanoma','Lung cancer']))
ORDER=['Harmony-direct[source_scale]','Harmony-direct[fixed_arcsinh]','cyCombine','CytoNorm','fastMNN reconstructed','RAMP']
LABELS=['Harmony-direct','Harmony-arcsinh','cyCombine','CytoNorm','fastMNN reconstructed','RAMP']

def counts(x):
    return dict(n_contrasts=len(x),n_sample_marker=len(x[['case','unit_key','marker']].drop_duplicates()),
                n_samples=len(x[['case','unit_key']].drop_duplicates()),n_markers=x.marker.nunique())

def save(fig,name):
    for ext in ['png','svg']:fig.savefig(HERE/f'{name}.{ext}',dpi=220,facecolor='white')
    plt.close(fig)

def markdown_table(d):
    header='| '+' | '.join(map(str,d.columns))+' |\n'
    divider='| '+' | '.join(['---']*len(d.columns))+' |\n'
    return header+divider+''.join('| '+' | '.join(map(str,row))+' |\n' for row in d.to_numpy())

def make_example(ax,case,unit,marker,cells,names,stages=True):
    meta=a.read_csv(HERE/case/'cell_metadata.csv.gz')
    markers=a.track(HERE/case/'markers.txt').read_text().splitlines()
    rows=np.flatnonzero(meta.unit_key.eq(unit));j=markers.index(marker)
    labels=meta.iloc[rows].celltype.to_numpy()
    base=ROOT/'benchmark_results'/case/'ramp/alignment'
    rp=base/'data/ramp_aligned_float64.npy'
    if not rp.exists():rp=base/'data/ramp_aligned.npy'
    arrays={'Raw':np.load(a.track(base/'input/raw.npy')),
        'PCA only':np.load(a.track(HERE/case/'pca_reconstructed.npy')),
        'fastMNN':np.load(a.track(HERE/case/'reconstructed.npy')),
        'RAMP':np.load(a.track(rp))}
    if not stages:arrays.pop('PCA only')
    colors=['#297BB0','#D879A0','#C4C9CC']
    result=[]
    for i,(stage,matrix) in enumerate(arrays.items()):
        v=matrix[rows,j];pr=a.ranks(v)
        groups=[pr[labels==c] for c in cells]
        width=.23 if len(cells)==3 else .28
        offset=np.linspace(-.27,.27,len(cells)) if len(cells)==3 else np.array([-.18,.18])
        b=ax.boxplot(groups,positions=i+offset,widths=width,showfliers=False,patch_artist=True,manage_ticks=False,
            medianprops={'color':'#29343B','lw':.8},boxprops={'lw':.65},whiskerprops={'lw':.65},capprops={'lw':.65})
        for box,color in zip(b['boxes'],colors):box.set_facecolor(color)
        auc=a.auc(v[labels==cells[0]],v[labels==cells[1]])
        ax.text(i,105,f'{auc:.2f}',ha='center',fontsize=9,color='#705395' if stage=='fastMNN' else '#34434A')
        for c,pp in zip(cells,groups):
            result.append(dict(case=case,unit_key=unit,marker=marker,stage=stage,celltype=c,n_cells=len(pp),
                q25=float(np.quantile(pp,.25)),median=float(np.median(pp)),q75=float(np.quantile(pp,.75)),auc_blue_vs_pink=auc))
    ax.set_xticks(range(len(arrays)),list(arrays),fontsize=9)
    ax.set_ylim(0,113);ax.set_yticks([0,50,100]);ax.set_ylabel(marker+' percentile')
    ax.spines[['top','right']].set_visible(False)
    ax.legend(handles=[Line2D([],[],ls='',marker='s',ms=6,color=c,label=n) for c,n in zip(colors,names)],
        loc='upper left',bbox_to_anchor=(0,1.16),ncol=3,frameon=False,fontsize=8,handletextpad=.35,columnspacing=.8)
    return result

def main():
    a.track(__file__)
    d=a.read_csv(HERE/'canonical_contrasts_all.csv')
    d['nontarget_median_up_pp']=d['fastMNN reconstructed_negative_median_percentile']-d['Raw_negative_median_percentile']
    d['target_median_down_pp']=d['Raw_positive_median_percentile']-d['fastMNN reconstructed_positive_median_percentile']
    qualified=d[d.Raw_auc.ge(.8)].copy()
    qualified['nontarget_increase_with_auc_loss']=qualified.nontarget_median_up_pp.ge(20)&qualified.auc_loss_raw_to_corrected.ge(.10)
    qualified['large_auc_loss']=qualified.auc_loss_raw_to_corrected.ge(.20)
    qualified['auc_direction_reversal']=qualified['fastMNN reconstructed_auc'].lt(.5)
    # These screens overlap and must never be summed as independent events.
    screens=['nontarget_increase_with_auc_loss','large_auc_loss','auc_direction_reversal']
    reports=[]
    for screen in screens:
        hits=qualified[qualified[screen]].copy()
        hits.to_csv(HERE/f'artifact_{screen}.csv',index=False)
        for case in ['all']+a.CASES:
            q=qualified if case=='all' else qualified[qualified.case.eq(case)]
            h=q[q[screen]]
            reports.append(dict(screen=screen,case=case,**counts(h),eligible_contrasts=len(q),
                                fraction_percent=100*len(h)/len(q) if len(q) else np.nan))
    pd.DataFrame(reports).to_csv(HERE/'artifact_counts.csv',index=False)
    hits=qualified[qualified.nontarget_increase_with_auc_loss].copy()
    replication=hits.groupby(['case','dataset','marker','positive_labels','negative_label']).agg(
        n_samples=('unit_key','nunique'),samples=('unit_key',lambda x:'; '.join(sorted(set(x)))),
        median_nontarget_up_pp=('nontarget_median_up_pp','median'),mean_auc_loss=('auc_loss_raw_to_corrected','mean')).reset_index()
    replication.sort_values('n_samples',ascending=False).to_csv(HERE/'artifact_repeated_signatures.csv',index=False)
    shifts=a.read_csv(HERE/'comparison_celltype_shifts.csv.gz')
    rows=[]
    for method in ORDER:
        f=shifts[shifts.method.eq(method)]
        for case in ['all']+a.CASES:
            q=f if case=='all' else f[f.case.eq(case)]
            for threshold in [10,20,30,40]:
                h=q[q.abs_delta_median_pp.gt(threshold)]
                rows.append(dict(case=case,method=method,threshold_pp=threshold,n_total=len(q),n_above=len(h),
                    percent_above=100*len(h)/len(q),median_abs_shift_pp=q.abs_delta_median_pp.median(),
                    mean_abs_shift_pp=q.abs_delta_median_pp.mean(),n_samples_evaluated=q.unit_key.nunique(),
                    n_samples_above=h.unit_key.nunique()))
    shift_counts=pd.DataFrame(rows);shift_counts.to_csv(HERE/'percentile_shift_counts.csv',index=False)
    fm=shifts[shifts.method.eq('fastMNN reconstructed')]
    fm.sort_values('abs_delta_median_pp',ascending=False).to_csv(HERE/'fastmnn_all_celltype_shifts_sorted.csv.gz',index=False)

    plt.rcParams.update({'font.family':'Arial','font.size':9,'svg.fonttype':'none','axes.linewidth':.6,
                         'text.color':'#26353D','axes.labelcolor':'#26353D'})
    fig=plt.figure(figsize=(10.8,4.2))
    ax=fig.add_axes([.185,.17,.34,.64]);right=fig.add_axes([.62,.17,.36,.64])
    fig.text(.015,.955,'Marker-value preservation',weight='bold',fontsize=14)
    fig.text(.185,.875,'All 4 benchmarks · 6,173 combinations',weight='bold',fontsize=10)
    for i,method in enumerate(ORDER):
        v=shifts[shifts.method.eq(method)].abs_delta_median_pp.to_numpy()
        color='#765BA6' if method.startswith('fastMNN') else '#008969' if method=='RAMP' else '#B6BEC2'
        yy=i+np.random.default_rng(20260914).uniform(-.16,.16,len(v));within=v<=60
        ax.scatter(v[within],yy[within],s=1,alpha=.13,color=color,lw=0,rasterized=True)
        ax.scatter(np.full((~within).sum(),60.),yy[~within],s=3,marker='>',alpha=.4,color=color,lw=0,clip_on=False)
        b=ax.boxplot(v,positions=[i],vert=False,widths=.4,showfliers=False,patch_artist=True,manage_ticks=False,
            boxprops={'lw':.7},medianprops={'color':'#26353D','lw':.9},whiskerprops={'lw':.7},capprops={'lw':.7})
        b['boxes'][0].set_facecolor(color)
    ax.set_yticks(range(6),LABELS);ax.invert_yaxis();ax.set_xlim(0,60);ax.set_xticks([0,20,40,60])
    ax.set_xlabel('Absolute median percentile shift (pp)');ax.spines[['top','right','left']].set_visible(False)
    ax.tick_params(axis='y',length=0)
    right.set_title('CD14 | Bone marrow\nAUROC: monocytes versus B cells',loc='left',pad=37,fontsize=10,weight='bold')
    example_stats=make_example(right,'bone_marrow','BM CyTOF|H2','CD14',
        ['Monocyte','Mature / CODEX B','CD4 T'],['Monocytes','B cells','CD4 T'])
    fig.text(.185,.04,'Shift >20 pp: fastMNN 23.75% · RAMP 0%',fontsize=10,weight='bold')
    fig.text(.62,.04,'B-cell median: 44.3 → 82.2 (+37.9 pp)',fontsize=10)
    save(fig,'artifact_focus')
    fig,axes=plt.subplots(1,3,figsize=(10.8,3.7),layout='constrained')
    for ax,unit in zip(axes,['Triana AbSeq|BM1','Triana AbSeq|BM2','Triana AbSeq|BM3']):
        example_stats.extend(make_example(ax,'bone_marrow',unit,'CD14',['Monocyte','Pre / immature B'],['Monocytes','Immature B'],False))
        ax.set_title(unit.split('|')[-1],loc='left',pad=34,weight='bold',fontsize=10)
    fig.suptitle('Repeated CD14 changes in all three AbSeq samples\nNumbers above boxes: monocyte-versus-immature-B AUROC',fontsize=12,weight='bold')
    save(fig,'cd14_abseq_three_samples')
    pd.DataFrame(example_stats).to_csv(HERE/'artifact_focus_plot_statistics.csv',index=False)

    summary=shift_counts[shift_counts.case.eq('all')&shift_counts.threshold_pp.eq(20)].copy()
    summary['method']=LABELS
    display=summary[['method','median_abs_shift_pp','mean_abs_shift_pp','n_above','percent_above']].copy()
    display.columns=['方法','中位绝对偏移（pp）','平均绝对偏移（pp）','偏移 >20 pp 的组合数','比例（%）']
    for c in ['中位绝对偏移（pp）','平均绝对偏移（pp）','比例（%）']:display[c]=display[c].map(lambda x:f'{x:.2f}')
    stat=a.read_csv(HERE/'reconstruction_checks.csv')
    text='''# fastMNN reconstructed：marker artifact 与 percentile 变化

本次重点是检查重建 marker 值是否改变已知的细胞类型表达特征。所有旧文件、benchmark 数据、原图和 manuscript 均未修改；新文件只写在本目录。

## 主要结果

覆盖原四组 benchmark，共 270,732 个细胞、104 个样本单元。四组标准 fastMNN 的 corrected embedding 与 rotation 均通过冻结 manifest 的 SHA-256 校验；未重新拟合 alignment。

Figure 1 当前的 6,173 个合格“样本 × cell type × marker”组合中，fastMNN reconstructed 的中位绝对 percentile 偏移为 **9.21 pp**，平均为 **13.09 pp**。**1,466/6,173（23.75%）偏移超过 20 pp**；RAMP 为 **0/6,173**。这些合格组合来自 103 个样本；全部 104 个样本都被读取，另一个样本没有满足 n≥50 的有效细胞类型组。

'''+markdown_table(display)+'''
RAMP 在少数 tied-value 情况下有极小变化：平均 0.000045 pp、最大 0.0821 pp。因此显示为 0.00 不代表全部数值严格等于零。

![聚焦图](artifact_focus.png)

## 典型 marker 异常计数

沿用已有 canonical_expectations.csv 的生物学对比规则，逐样本计算，阳性和阴性组均须至少 50 个细胞。共有 1,542 个可评价对比，其中 **1,293 个原始 AUROC ≥0.80**。以下是本次探索性筛查的操作性定义，不能当成所有 artifact 的通用发生率：

| 判据（均要求原始 AUROC ≥0.80） | 对比数 | 去重样本×marker | 去重样本 |
| --- | ---: | ---: | ---: |
| 非目标细胞中位 percentile 上升 ≥20 pp，且 AUROC 下降 ≥0.10 | 26 | 24 | 17 |
| AUROC 下降 ≥0.20 | 18 | 13 | 10 |
| 最终 AUROC <0.50，原本的区分方向反转 | 7 | 5 | 4 |

三行互相重叠，不可相加。一个样本×marker 可以有多个阴性细胞类型对比，不能把对比数误称为独立患者数。

26 个非目标细胞异常上升对比在四组中的分布为：bone marrow 9、TNBC 9、melanoma 5、lung cancer 3。相应完整列表、分母和筛查条件均保存于 artifact_*.csv；没有删除未命中的对比。

## 最清楚的例子：BM CyTOF H2 / CD14

- 单核细胞 n=2,600；成熟 B 细胞 n=2,185；图中另加入 CD4 T 作为对照。
- 单核细胞 vs B 细胞 AUROC：Raw **0.8114** → cosine **0.8109** → PCA-only **0.8141** → fastMNN reconstructed **0.1493**。
- B 细胞中位 percentile：**44.34 → 82.21（+37.87 pp）**。
- 单核细胞中位 percentile：**86.62 → 57.22（−29.40 pp）**。
- PCA-only 保持原有方向，而 MNN 后重建值反转，因此此例不能仅归因于 cosine normalization 或 PCA 投影。
- RAMP 的这两类细胞保持原始 percentile，AUROC 仍为 **0.8114**。

另一个 CD3/TNBC 强反例为原始 AUROC 0.9765 → 最终 0.1275，但其变化在 PCA-only 时已经出现，故不作为最推荐的 MNN 校正阶段例子。

## 跨样本重复

AbSeq 的三个样本中，未成熟 B 细胞的 CD14 中位 percentile 均异常升高：BM1 **+50.09 pp**、BM2 **+40.34 pp**、BM3 **+35.54 pp**。单核细胞 vs 未成熟 B 细胞的 AUROC 分别为 **0.9531→0.7730、0.9176→0.4788、0.9729→0.5916**。这说明发现不局限于 H2 单个样本。CD11B 的同一细胞类型对比也在这三个样本中重复；melanoma 的 CD68/CD8 T 对比在两个样本中重复。

![AbSeq 三个样本](cd14_abseq_three_samples.png)

## 计算口径和解释范围

每个样本、每个 marker 内，先用该样本全部细胞计算 `100 × average_rank / (n+1)`；再对每个 cell type 取中位数，比较 reconstructed 与原始值。与 Figure 1 完全相同的组集合、n≥50 规则和未解析标签排除；被排除组的细胞仍在 percentile 参照中。20 pp 表示在样本内排序中的位置变化，不是表达强度增加 20%。分布偏移本身不自动等同于生物学 artifact，因此另用已知 marker–cell type 关系和 AUROC 下降检查具体例子。

输出严格按 batchelor 1.18.1 的 `convertPCsToSCE` 计算：cell×marker = corrected PCs × rotation.T；不加回中心、不逆转 cosine normalization。这是官方 reconstructed assay 对应的重建值，不能称为恢复原始量纲的 protein abundance。官方建议其用于可视化而非定量表达推断，因此这里报告的是重建值的性质和诊断，不声称 fastMNN 的标准 embedding 整合因此无效。

## UMAP、ridge 和其他指标的保存位置

四组 `ridge_umap_review.png` 都包含 Raw、fastMNN、RAMP。UMAP 沿用已有共同评价协议的冻结坐标，不重新选种子或拟合。fastMNN 的 UMAP 用 corrected PCs，ridge 用 reconstructed markers。主 review 的 Raw ridge 用 arcsinh(raw) 仅改善显示，同时保留 `ridge_umap_native_raw.png`。`all_markers_ridge.png` 包含每组全部 shared markers，统一 160 bins、sigma=6、0.5%–99.5% 显示范围；所有截出尾部的细胞数保存于 ridge_display_audit.csv。

此前的 23 项整合指标已包含 fastMNN，清单保存于 existing_fastmnn_integration_metrics.csv。额外计算的 gating、normalized EMD 和 AUROC 数值诊断保存在 value_evaluation/，本次 artifact 聚焦不改动或隐藏这些结果，也尚未将任何新结果加入主图或 manuscript。

## 复现

按顺序执行本目录下 `run_audit.py`、`evaluate_reconstructed.py`、`plot_review.py`、`artifact_focus.py`。第一个脚本恢复矩阵并计算分布与 canonical 对比；第二个复用原 value-evaluation 函数补算指标；第三个出 UMAP/ridge 和综合诊断图；第四个生成本报告与 artifact 聚焦图。各阶段 manifest 保存输入 SHA-256，旧数据保持只读。

参考：[batchelor 官方方法说明](https://bioconductor.org/packages/release/bioc/vignettes/batchelor/inst/doc/correction.html)、[fastMNN 官方输出说明](https://github.com/LTLA/batchelor/blob/master/R/fastMNN.R)。
'''
    (HERE/'README.md').write_text(text)
    assert all(a.sha(p)==h for p,h in a.READS.items())
    (HERE/'artifact_focus_manifest.json').write_text(json.dumps(dict(
        focus='Cell-type marker percentile shifts and biologically interpretable reconstruction artifacts',
        screening='Exploratory canonical-rule screen; same >=50 cells per group and raw AUROC >=0.80 for every reported contrast',
        screens_overlap=True,counts_unit='sample x marker x positive-label-set x negative-label contrast',
        percentile_unit='sample x celltype x marker; identical 6173 eligible combinations to the current Figure 1',
        source_data_unchanged=True,main_figure_or_manuscript_modified=False,input_sha256=a.READS),indent=2)+'\n')
    print(pd.DataFrame(reports).query('case == "all"').to_string(index=False),flush=True)
    print(display.to_string(index=False),flush=True)

if __name__=='__main__':main()
