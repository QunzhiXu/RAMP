"""Display two selected full-ROI examples without cropping or color-cut searches."""
from pathlib import Path
import hashlib
import json
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/hnscc_roi_candidates_mpl')
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fitz

HERE=Path(__file__).resolve().parent
REVIEW=HERE.parent
plt.rcParams.update({'font.family':'Arial','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none','text.color':'#18232B'})

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def row(fig,base,d,bg,cohort,label):
    W,H=fig.get_size_inches()
    for x,title,col in [(.08,'Predicted CD39/TIGIT state','within_patient_rank'),(4.19,'Measured LAG3','LAG3_patient_rank')]:
        ax=fig.add_axes([x/W,(base+.52)/H,4/W,3/H])
        ax.imshow(bg,cmap='gray',vmin=0,vmax=1,extent=(0,1920,1440,0),interpolation='nearest')
        dots=ax.scatter(d.x,d.y,c=d[col],s=18,cmap='viridis',vmin=0,vmax=1,alpha=1,
                        edgecolors='#CFD7DC',linewidths=.16,rasterized=True)
        np.testing.assert_array_equal(dots.get_offsets(),d[['x','y']].to_numpy())
        np.testing.assert_array_equal(dots.get_array(),d[col].to_numpy())
        ax.set(xlim=(0,1920),ylim=(1440,0));ax.set_axis_off()
        fig.text((x+2)/W,(base+3.72)/H,title,ha='center',va='center',fontsize=12.7,weight='bold')
    cax=fig.add_axes([8.33/W,(base+1.00)/H,.105/W,2.00/H]);cb=fig.colorbar(dots,cax=cax,ticks=[0,.5,1])
    cb.ax.tick_params(labelsize=8.5,length=2.5,pad=3);cb.ax.set_title('Rank',fontsize=9,pad=7);cb.outline.set_linewidth(.5)
    rho=float(spearmanr(d.predicted_score,d.LAG3).statistic)
    fig.text(2.08/W,(base+.24)/H,f'{label}: {len(d)} CD8 cells  |  ρ = {rho:.2f}',ha='center',va='center',fontsize=10.0)
    fig.text(6.19/W,(base+.24)/H,f'74 patients: mean ρ = {cohort:.2f}',ha='center',va='center',fontsize=10.0)

def check_text(fig):
    fig.canvas.draw();r=fig.canvas.get_renderer()
    for t in list(fig.texts)+[v for a in fig.axes if a.axison for v in list(a.get_xticklabels())+list(a.get_yticklabels())+[a.title]]:
        if not t.get_visible() or not t.get_text():continue
        b=t.get_window_extent(r)
        assert b.x0>=-1 and b.y0>=-1 and b.x1<=fig.bbox.width+1 and b.y1<=fig.bbox.height+1,(t.get_text(),b.bounds)

def main():
    audit=json.loads((HERE/'RANKING_AUDIT.json').read_text())
    sources=[REVIEW/'codex_measured_protein_cells.csv.gz',REVIEW/'codex_protein_correlation_summary.csv']
    d=pd.read_csv(sources[0]);s=pd.read_csv(sources[1]);cohort=float(s[s.marker.eq('LAG3')&s.aggregation.eq('unadjusted')]['mean'].iloc[0])
    assert np.isclose(cohort,.5216258306549971)
    examples=[]
    for key,label in [('min100','Highest correlation'),('min200','Larger ROI')]:
        rec=audit['top'][key];roi=rec['roi'];data=d[d.unique_region.eq(roi)].copy()
        image_dir=HERE/'images'/roi;meta=json.loads((image_dir/'DOWNLOAD.json').read_text());path=image_dir/'DAPI.raw';sources.extend([path,image_dir/'DOWNLOAD.json'])
        assert sha(path)==meta['sha256']
        arr=np.fromfile(path,dtype=meta['endian']+'u2').reshape(meta['height'],meta['width']).astype(float)
        cap=float(np.percentile(arr[arr>0],99.8));bg=np.clip(arr/cap,0,1)**.72*.30
        assert len(data)==rec['n_cd8']
        np.testing.assert_allclose(spearmanr(data.predicted_score,data.LAG3).statistic,rec['rho'])
        examples.append((key,label,rec,data,bg,cap))
    before={str(p):sha(p) for p in sources}
    for key,label,rec,data,bg,_ in examples:
        fig=plt.figure(figsize=(8.8,3.95));row(fig,0,data,bg,cohort,label);check_text(fig)
        for ext in ['png','svg']:fig.savefig(HERE/f'spatial_{key}_{rec["roi"]}.{ext}',dpi=300,facecolor='white')
        plt.close(fig)
    fig=plt.figure(figsize=(8.8,7.90))
    for base,(_,label,rec,data,bg,_) in zip([3.95,0],examples):row(fig,base,data,bg,cohort,label)
    check_text(fig)
    for ext in ['png','pdf','svg']:fig.savefig(HERE/f'two_candidate_ROIs.{ext}',dpi=250,facecolor='white')
    plt.close(fig)
    doc=fitz.open(HERE/'two_candidate_ROIs.pdf');assert len(doc)==1
    doc[0].get_pixmap(matrix=fitz.Matrix(1.7,1.7)).save(HERE/'pdf_preview.png');doc.close()
    assert before=={str(p):sha(p) for p in sources}
    (HERE/'PLOT_AUDIT.json').write_text(json.dumps(dict(status='PASS',all_inputs_unchanged=before,complete_rois=True,
              identical_cells_in_two_maps=True,coordinate_units='source pixels',point_size=18,color_range=[0,1],colormap='viridis',
              display_ranks='within-patient percentiles',DAPI_scaling='each plane positive-pixel p99.8, gamma0.72, amplitude0.30',
              ROI_selection_uses_correlation=True,threshold_or_crop_search=False,cohort_mean_unchanged=cohort),indent=2)+'\n')
    r=pd.read_csv(HERE/'all_176_roi_ranking.csv');eligible=r[r.eligible_100]
    rank_old=int(np.flatnonzero(eligible.roi.eq('522_S2reg054').to_numpy())[0]+1)
    first,large=audit['top']['min100'],audit['top']['min200']
    (HERE/'README_ZH.md').write_text(f'''# HNSCC：选择更清楚的真实空间示例

扫描当前正式 CD8 返回模型的全部 **176 个 ROI、74 位患者、38,299 个 CD8**。完整排名保留，包括细胞不足或不可估计者。展示候选至少 100 个 CD8，共 93 个；另报告至少 200 个 CD8 的较大视野候选，共 61 个。

| 示例 | ROI | 患者 | CD8 数 | ROI Spearman |
|---|---|---:|---:|---:|
| ≥100 CD8 中最高相关 | {first['roi']} | {first['patient']} | {first['n_cd8']} | {first['rho']:.3f} |
| ≥200 CD8 中最高相关 | {large['roi']} | {large['patient']} | {large['n_cd8']} | {large['rho']:.3f} |
| 原示例 | 522_S2reg054 | 15219 | 650 | 0.384 |

原示例在 93 个 ≥100 CD8 候选中排第 {rank_old}。因此沿用原 niche 选择的示例并不适合最清楚地展示与 LAG3 的关联。

原全队列指标保持 **74 位患者各自 Spearman 的平均值 0.522，95% CI 0.496–0.550**。新示例的局部 rho 不是新的整体性能，也不是独立测试结果。ROI 选择使用了已观察到的相关，须在 caption 中说明。

## 图像和可读性

- `two_candidate_ROIs.png/.pdf/.svg`：上行为最高相关的 115-cell ROI，下行为细胞较多的 343-cell ROI。
- `spatial_min100_{first['roi']}.png/.svg` 与 `spatial_min200_{large['roi']}.png/.svg`：可单独排入主图的双图。
- 左图是预测 CD39/TIGIT 综合状态，右图是 CODEX 实测 LAG3；两边细胞和坐标完全相同。颜色保持患者内百分位 0–1，未扫描上下界。
- 使用完整 1920×1440 视野，没有裁剪或选择局部。彩点放大到 size=18，DAPI 显示亮度为原规范化值的 0.30，使覆盖层更清楚；这些都是固定显示参数，不改数据。
- 背景来自官方 BioImage Archive S-BIAD1638 的真实 DAPI。只下载两张图的原始 DAPI 层，各 5,529,600 bytes；TIFF 文件头和全部 HTTP Content-Range 已核对。彩点是细胞级定量覆盖层，不是原始 LAG3 荧光影像。

## Caption 可用措辞

“An illustrative high-correlation ROI is shown, selected for the highest within-ROI Spearman association between predicted CD39/TIGIT state and measured LAG3 among ROIs containing at least 100 CD8 T cells. Both maps show the same cells at their original CODEX coordinates over the measured DAPI image. Colors indicate within-patient percentile ranks. Cohort-level association is summarized separately across all 74 patients.”

这里明确称为选出的高相关示例，不称其为典型队列表现。若最终使用 343-cell 版本，将至少 100 改为至少 200，或明确其为全部 ≥100 候选中排名第二的较大 ROI。

## 文件与复现

`rank_rois.py` → `download_dapi.py` → `plot_candidates.py`。所有原文件只读，正式 Figure 和 manuscript 未替换。

- `all_176_roi_ranking.csv`：完整排名与真实图像清单匹配。
- `eligible_min100_ranking.csv`、`eligible_min200_ranking.csv`：两个固定细胞数条件下的全部候选。
- `RANKING_AUDIT.json`、`PLOT_AUDIT.json`：数据数量、选择规则、输入哈希及绘图参数。
- `images/<ROI>/DOWNLOAD.json`：官方 URL、TIFF 参数及图层 SHA256。
- LAG3 未进入当前对齐或状态构建，但历史探索曾查看该指标；图用于辅助表型展示，不称为完全独立的盲验证。
''')
    print('PASS: two complete-ROI image pairs, unchanged cohort metric, PNG/SVG and comparison PDF.')

if __name__=='__main__':main()
