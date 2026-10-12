#!/usr/bin/env python3
"""Display all predeclared variants on unchanged real CODEX microscopy."""
from pathlib import Path
import json,os
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OUT=HERE/'output';FIG=HERE/'figures';FIG.mkdir(exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR',str(HERE/'.cache/matplotlib'))
os.environ.setdefault('MPLBACKEND','Agg')
import fitz
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

KEYS=['pair','protein3','rna3','pair_protein3','pair_rna3','protein3_rna3','pair_protein3_rna3']
NAMES={'pair':'CD39/TIGIT (original)','protein3':'3 proteins','rna3':'3 RNA',
       'pair_protein3':'Pair + 3 proteins','pair_rna3':'Pair + 3 RNA',
       'protein3_rna3':'3 proteins + 3 RNA','pair_protein3_rna3':'Pair + 3 proteins + 3 RNA'}
INK='#182027';BLUE='#2775B6';ORANGE='#D95F02';TEAL='#238A8D'
CROP=(1205,205,1805,670)


def save(fig,stem):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    texts=list(fig.texts)
    for ax in fig.axes:
        texts += list(ax.texts)+[ax.title,ax._left_title,ax.xaxis.label,ax.yaxis.label]+ax.get_xticklabels()+ax.get_yticklabels()
    for legend in fig.legends:texts+=legend.get_texts()
    for text in texts:
        if text.get_visible() and text.get_text():
            b=text.get_window_extent(renderer)
            assert b.x0>=-1 and b.y0>=-1 and b.x1<=fig.bbox.width+1 and b.y1<=fig.bbox.height+1,(stem,text.get_text())
    for ext in ['png','pdf','svg']:fig.savefig(FIG/f'{stem}.{ext}',dpi=240,facecolor='white')
    plt.close(fig)


def compare_plot(summary):
    fig=plt.figure(figsize=(11.3,4.4))
    fig.text(.025,.97,'Does a larger program improve spatial correspondence?',fontsize=15,weight='bold',va='top')
    fig.text(.025,.88,'74 patients | 38,299 CODEX CD8 cells | fixed alignment, k = 30 and top 40%',fontsize=10.4)
    axes=[fig.add_axes([.255,.18,.305,.60]),fig.add_axes([.64,.18,.33,.60])]
    for ax,metric,factor,title,xlabel in [(axes[0],'spearman',1.,'Continuous spatial score','Mean within-patient Spearman'),
                                        (axes[1],'high_minus_other',100.,'High-state niche enrichment','High minus other CD8 (percentage points)')]:
        for y,key in zip(range(6,-1,-1),KEYS):
            row=summary.loc[key];v=row[metric+'_mean']*factor
            lo,hi=row[metric+'_ci_low']*factor,row[metric+'_ci_high']*factor
            color=ORANGE if key=='pair' else TEAL if key=='rna3' else BLUE
            ax.errorbar(v,y,xerr=[[v-lo],[hi-v]],fmt='o',ms=5.5,color=color,capsize=3,lw=1.4)
        ax.axvline(summary.loc['pair',metric+'_mean']*factor,color=ORANGE,lw=.9,ls='--',alpha=.7)
        ax.set_title(title,fontsize=11.7,weight='bold',loc='left',pad=12)
        ax.set_ylim(-.5,6.5);ax.set_yticks(range(6,-1,-1),[NAMES[k] for k in KEYS] if ax==axes[0] else [])
        ax.tick_params(axis='y',length=0,pad=7,labelsize=10.)
        ax.tick_params(axis='x',length=3,labelsize=9)
        ax.set_xlabel(xlabel,fontsize=10,labelpad=6)
        if metric=='spearman':
            ax.set(xlim=(-.015,.13),xticks=[0,.04,.08,.12])
        else:
            ax.set(xlim=(-1.5,10),xticks=[0,3,6,9])
        ax.spines[['top','right','left']].set_visible(False)
        ax.xaxis.grid(True,color='#E7EBEE',lw=.6);ax.set_axisbelow(True)
    fig.text(.255,.035,'Dots: patient-equal means; bars: patient bootstrap 95% intervals.',fontsize=9.5)
    save(fig,'program_comparison_all74')


def background():
    root=ROOT/'spatial_return_survival/input/planes'
    rgb=np.zeros((1440,1920,3),dtype=np.float32)
    for marker,pct,gamma,color in [('DAPI',99.80,.72,[.20,.22,.34]),('CD8',99.55,.70,[0,.78,.92]),
                                  ('FAP',99.70,.74,[0,.76,0]),('CD68',99.55,.70,[.88,0,.70])]:
        a=np.fromfile(root/f'{marker}.raw',dtype='>u2').reshape(1440,1920).astype(np.float32)
        a=np.power(np.clip(a/max(float(np.percentile(a[a>0],pct)),1.),0,1),gamma)
        rgb+=a[...,None]*np.array(color,dtype=np.float32)
    rgb=np.clip(np.power(np.clip(rgb,0,1),.93),0,1)*.60
    x0,y0,x1,y1=CROP
    return rgb[y0:y1,x0:x1]


def spatial_plot(image,cells,architecture,keys,stem,ncols,title):
    nrows=(len(keys)+ncols-1)//ncols
    width=3.35*ncols+.15; height=3.15*nrows+.77
    fig=plt.figure(figsize=(width,height))
    fig.text(.12/width,1-.07/height,title,fontsize=15,weight='bold',va='top')
    fig.text(.12/width,1-.39/height,'Identical real CODEX region and coordinates; all returned states use the patient-specific top 40%.',fontsize=10.,va='top')
    legend=[Line2D([],[],marker='o',linestyle='',color=c,markersize=4.3,label=l)
            for l,c in [('Highlighted CD8','#D7AD00'),('Other CD8','#B8BEC7'),('CAF','#39C76B'),('Macrophage','#DF409D')]]
    fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,.01),ncol=4,fontsize=9,frameon=False,handlelength=.7)
    for i,key in enumerate(keys):
        col,row=i%ncols,i//ncols
        x=.075+3.35*col;y=height-.98-2.519-3.15*row
        ax=fig.add_axes([x/width,y/height,3.25/width,2.519/height])
        ax.imshow(image,extent=(CROP[0],CROP[2],CROP[3],CROP[1]),origin='upper',interpolation='nearest')
        for kind,color in [('CAF','#39EF72'),('macrophages','#FF43B5')]:
            b=architecture[architecture.cell_types_mediumres.eq(kind)]
            ax.scatter(b.x,b.y,s=3.7,c=color,alpha=.75,linewidths=0,zorder=3)
        ax.scatter(cells.x,cells.y,s=3.2,c='#B8BEC7',alpha=.52,linewidths=0,zorder=4)
        selected=cells.spatial_y.gt(0) if key=='observed' else cells[key+'_rank'].ge(.60)
        b=cells[selected]
        ax.scatter(b.x,b.y,s=20,c='#FFD400',edgecolor='#241A00',linewidth=.6,zorder=5)
        label='Observed spatial niche' if key=='observed' else NAMES[key]
        ax.set_title(label,loc='left',fontsize=10.7,weight='bold',pad=5)
        subtitle=f'{len(b)} in-niche CD8' if key=='observed' else f'{int((selected & cells.spatial_y.gt(0)).sum())}/{int(selected.sum())} highlighted CD8 are in-niche'
        ax.text(.025,.025,subtitle,transform=ax.transAxes,color='white',fontsize=8.7,weight='bold',va='bottom',
                bbox=dict(facecolor='black',edgecolor='none',alpha=.65,pad=2))
        ax.set(xlim=(CROP[0],CROP[2]),ylim=(CROP[3],CROP[1]),xticks=[],yticks=[])
        for s in ax.spines.values():s.set_visible(False)
    save(fig,stem)


def report(summary,crop,sensitivity,audit):
    labels={'pair':'原始 CD39/TIGIT','protein3':'仅 3 protein','rna3':'仅 3 RNA','pair_protein3':'Pair + 3 protein',
            'pair_rna3':'Pair + 3 RNA','protein3_rna3':'3 protein + 3 RNA','pair_protein3_rna3':'全部 8 项'}
    text=['# HNSCC 扩展 program 回投结果','',
          '**结论：这轮等权扩展没有带来明确的整体改善。建议主图继续保留原 CD39/TIGIT pair，额外蛋白与 RNA 用于描述其分子表型。**','',
          '七个预先指定的版本均已完成，使用同一来源 4,788 个肿瘤 CD8、三位 CITE-seq donor、同一目标 74 位患者的全部 38,299 个 CODEX PT-CD8。对齐、特征缩放、邻居、k=30 和 top-40% 阈值不变。未读生存结局，未重新寻找 ROI 或阈值。','',
          '## 全队列比较','',
          '| Program | 平均患者内 Spearman | 中位 Spearman | 高状态相对其余 CD8 的 in-niche 比例差 |',
          '|---|---:|---:|---:|']
    for key in KEYS:
        r=summary.loc[key];text.append(f'| {labels[key]} | {r.spearman_mean:.4f} | {r.spearman_median:.4f} | {100*r.high_minus_other_mean:.2f} 个百分点 |')
    r=summary.loc['rna3'];rp=summary.loc['pair_rna3']
    text += ['',
        '主要比较是先逐患者计算预测 program 与实测空间 niche score（spatial_y）的 Spearman，再对 74 位患者等权平均。所有版本的 74 位患者均可计算。相关性绝对值较小；没有任何新增 program 超过原 Pair 的平均值。完整中位数、AUROC、最高/最低五分位和区间见 output/program_summary.csv。','',
        f'RNA-only 的富集比例差由原 Pair 的 4.16 个百分点变为 {100*r.high_minus_other_mean:.2f} 个百分点，即增加 {100*r.high_minus_other_delta_vs_pair:.2f} 个百分点；配对患者 bootstrap 95% 区间为 {100*r.high_minus_other_delta_ci_low:.2f} 至 {100*r.high_minus_other_delta_ci_high:.2f} 个百分点，跨过 0。Pair + RNA 增加 {100*rp.high_minus_other_delta_vs_pair:.2f} 个百分点，区间 {100*rp.high_minus_other_delta_ci_low:.2f} 至 {100*rp.high_minus_other_delta_ci_high:.2f}，同样跨过 0。该细小收益不足以支持替换原 program。','',
        '这些 bootstrap 区间是固定发现链条下的描述性不确定性，不是校正了候选选择或重新发现全过程的独立验证区间。','',
        '## 同一区域的展示','',
        '仍使用 patient 15219、ROI 522_S2reg054、crop x=[1205,1805)、y=[205,670)。61 个 CD8 中 32 个属于原空间 niche。','',
        '| Program | 回投高状态 CD8 数 | 与 32 个原 in-niche CD8 重叠数 | Jaccard |',
        '|---|---:|---:|---:|']
    for key in KEYS:
        r=crop.loc[key];text.append(f'| {labels[key]} | {int(r.high_cells)} | {int(r.overlap)} | {r.jaccard:.3f} |')
    text += ['',
        '在原展示区域里，扩展 program 也没有更好的集合重叠。这里不把这个区域当独立测试集：它是以前依据空间对应、但不使用生存结局选出的展示 ROI。全队列均值才是本轮事前指定的主要比较。','',
        '## 来源 donor 敏感性','',
        '依次去掉三个来源 donor，每次用另外两位的预测中位数。Protein-only、Pair+Protein、RNA-only、Protein+RNA、全部八项在三个去 donor 版本中，平均患者内 Spearman 均低于对应的 Pair。Pair+RNA 仅在去掉 donor 126 时略高（+0.0013），另外两次仍低。没有出现一致的扩展收益。','',
        '## 状态如何构建','',
        '- 正向：TIGIT、CD39、LAG-3、CTLA4、CXCL13。反向：CD127、CD62L、IL7R。每位来源 donor 内先取 midrank；反向用 1−rank；对列出的特征等权平均。','- 原 Pair 保留原来的 no-LAG3 CLR；三项额外蛋白沿用 Panel E 的 pair-free CLR。RNA 用提供者 raw.X 连续表达取秩；Panel E 的 RNA 检出率是另一种分组后展示量，未直接拿检出百分比作细胞预测。','- CD127 和 IL7R 对应同一受体的蛋白与 RNA，全部八项包含这一重复生物信息；未在看完结果后再调权重。','- 原 Pair 的重算最大误差为 '+f'{audit["baseline_max_absolute_error"]:.3g}'+'；患者内排名复现。所有输入文件哈希保持不变。','',
        '## 对故事的含义','',
        '更大的 program 可以描述更丰富的分子特征，但这里没有增加 CAF/macrophage 空间对应的有效信息。一个与现有结果相容的解释是：新增指标主要描述了 CD39/TIGIT 高状态的伴随表型，未提供额外的空间区分能力。此解释是基于本次比较的推断。','',
        '若把 Panel E 六项也纳入 program 定义，它们便不能再作为该新状态的独立表型支持。当前既没有明确空间收益，又会减少可用于解释的留出指标，因此暂不替换主线更合适。','',
        '## 文件','',
        '- figures/program_comparison_all74.png/.pdf：全队列两个主要展示指标。','- figures/all_programs_same_roi.png/.pdf：原空间 niche 与七种回投的完整并排。','- figures/pair_rna_all8_same_roi.png/.pdf：原 Pair、RNA-only 与全部八项，便于近看。','- output/patient_metrics.csv：74×7 个患者层面条目。','- output/program_summary.csv：七项汇总和配对 bootstrap 区间。','- output/all_program_predictions.csv.gz：全部目标细胞、坐标与所有版本的预测/排名。','- output/source_features_and_programs.csv.gz：来源细胞全部组成特征和分数。','- output/source_donor_sensitivity.csv：三次去来源 donor 结果。','- output/RUN_AUDIT.json：输入哈希、基线复现与固定参数。','',
        '所有试验独立保存，正式图与 manuscript 的科学结果均未替换。']
    (HERE/'RESULTS_ZH.md').write_text('\n'.join(text)+'\n')


def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','text.color':INK})
    s=pd.read_csv(OUT/'program_summary.csv').set_index('program')
    compare_plot(s)
    all_cells=pd.read_csv(OUT/'all_program_predictions.csv.gz',dtype={'patient':str})
    cells=all_cells[all_cells.patient.eq('15219') & all_cells.unique_region.eq('522_S2reg054') &
                    all_cells.x.between(CROP[0],CROP[2],inclusive='left') & all_cells.y.between(CROP[1],CROP[3],inclusive='left')].copy()
    assert len(cells)==61 and cells.spatial_y.gt(0).sum()==32
    architecture=pd.read_csv(ROOT/'big_figure/polish_20260913/horizontal_E_spatial_left/panel_F/plotted_caf_macrophage_cells.csv')
    image=background()
    spatial_plot(image,cells,architecture,['observed']+KEYS,'all_programs_same_roi',4,'Observed niche and all seven molecular programs')
    spatial_plot(image,cells,architecture,['pair','rna3','pair_protein3_rna3'],'pair_rna_all8_same_roi',3,'Original pair, RNA-only and the eight-feature program')
    rows=[]
    for key in KEYS:
        high=cells[key+'_rank'].ge(.60);truth=cells.spatial_y.gt(0)
        rows.append(dict(program=key,high_cells=int(high.sum()),observed_niche_cells=int(truth.sum()),
                         overlap=int((high&truth).sum()),jaccard=float((high&truth).sum()/(high|truth).sum())))
    crop=pd.DataFrame(rows).set_index('program');crop.to_csv(OUT/'fixed_roi_comparison.csv')
    sensitivity=pd.read_csv(OUT/'source_donor_sensitivity.csv')
    audit=json.loads((OUT/'RUN_AUDIT.json').read_text())
    report(s,crop,sensitivity,audit)
    qa=HERE/'qa';qa.mkdir(exist_ok=True)
    for name in ['program_comparison_all74','all_programs_same_roi','pair_rna_all8_same_roi']:
        with fitz.open(FIG/f'{name}.pdf') as doc:
            doc[0].get_pixmap(dpi=150,alpha=False).save(qa/f'{name}.png')
    print('Three figures, ROI metrics and Chinese report complete.')


if __name__=='__main__':main()
