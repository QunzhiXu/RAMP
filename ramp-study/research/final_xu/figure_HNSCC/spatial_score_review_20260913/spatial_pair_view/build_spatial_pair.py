"""A spatial-only view of frozen CD39/TIGIT state and measured CODEX LAG3."""
from pathlib import Path
import hashlib
import json
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/hnscc_spatial_pair_mpl')
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fitz

HERE=Path(__file__).resolve().parent
REVIEW=HERE.parent
FIGURE=REVIEW.parent
CELLS=REVIEW/'codex_measured_protein_cells.csv.gz'
STATS=REVIEW/'codex_protein_correlation_summary.csv'
PATIENT_STATS=REVIEW/'codex_protein_patient_correlations.csv'
DAPI=FIGURE/'spatial_return_survival/input/planes/DAPI.raw'
ROI='522_S2reg054'

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    paths=[CELLS,STATS,PATIENT_STATS,DAPI]
    hashes={str(p):sha(p) for p in paths}
    cells=pd.read_csv(CELLS,dtype={'patient':str,'cell_uid':str})
    d=cells[cells.unique_region.eq(ROI)].copy()
    assert len(d)==650 and d.patient.nunique()==1 and d.patient.iloc[0]=='15219'
    assert d.cell_uid.is_unique
    rho=float(spearmanr(d.predicted_score,d.LAG3).statistic)
    summary=pd.read_csv(STATS)
    cohort=summary[summary.marker.eq('LAG3')&summary.aggregation.eq('unadjusted')].iloc[0]
    p=pd.read_csv(PATIENT_STATS)
    p=p[p.marker.eq('LAG3')&p.aggregation.eq('unadjusted')]
    assert len(p)==74
    np.testing.assert_allclose(p.rho.mean(),cohort['mean'],rtol=1e-12)
    plane=np.fromfile(DAPI,dtype='>u2').reshape(1440,1920).astype(float)
    cap=float(np.percentile(plane[plane>0],99.8))
    background=np.clip(plane/cap,0,1)**.72*.45
    plt.rcParams.update({'font.family':'Arial','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none','text.color':'#17232B'})
    W,H=8.8,3.95
    fig=plt.figure(figsize=(W,H))
    specs=[(.08,'Predicted CD39/TIGIT state','within_patient_rank'),(4.19,'Measured LAG3','LAG3_patient_rank')]
    for left,title,column in specs:
        ax=fig.add_axes([left/W,.54/H,4./W,3./H])
        ax.imshow(background,cmap='gray',vmin=0,vmax=1,extent=(0,1920,1440,0),interpolation='nearest')
        points=ax.scatter(d.x,d.y,c=d[column],s=11,cmap='viridis',vmin=0,vmax=1,
                          edgecolors='#DFE5E8',linewidths=.12,alpha=1.,rasterized=True)
        np.testing.assert_array_equal(points.get_offsets(),d[['x','y']].to_numpy())
        np.testing.assert_array_equal(points.get_array(),d[column].to_numpy())
        ax.set(xlim=(0,1920),ylim=(1440,0));ax.set_axis_off()
        fig.text((left+2)/W,3.73/H,title,ha='center',va='center',fontsize=12.7,weight='bold')
    cax=fig.add_axes([8.33/W,1.00/H,.105/W,2.04/H])
    cb=fig.colorbar(points,cax=cax,ticks=[0,.5,1]);cb.ax.tick_params(labelsize=8.5,length=2.5,pad=3)
    cb.outline.set_linewidth(.5)
    cb.ax.set_title('Rank',fontsize=9,pad=8)
    fig.text(2.08/W,.25/H,f'Shown ROI: ρ = {rho:.2f}  |  650 CD8 cells',ha='center',va='center',fontsize=10.1)
    fig.text(6.19/W,.25/H,f'74 patients: mean ρ = {cohort["mean"]:.2f}',ha='center',va='center',fontsize=10.1)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for text in list(fig.texts)+list(cb.ax.get_yticklabels())+[cb.ax.title]:
        bbox=text.get_window_extent(renderer)
        assert bbox.x0>=0 and bbox.y0>=0 and bbox.x1<=fig.bbox.width+1 and bbox.y1<=fig.bbox.height+1,(text.get_text(),bbox.bounds)
    for ext in ['png','pdf','svg']:fig.savefig(HERE/f'predicted_state_vs_measured_LAG3_spatial.{ext}',dpi=300,facecolor='white')
    plt.close(fig)
    doc=fitz.open(HERE/'predicted_state_vs_measured_LAG3_spatial.pdf')
    assert len(doc)==1
    doc[0].get_pixmap(matrix=fitz.Matrix(2,2)).save(HERE/'pdf_preview.png');doc.close()
    d[['cell_uid','patient','unique_region','x','y','predicted_score','within_patient_rank','LAG3','LAG3_patient_rank']].to_csv(HERE/'plotted_cells.csv',index=False)
    pd.DataFrame([dict(scope='shown_complete_ROI',n_cells=650,n_patients=1,spearman=rho),dict(scope='equal_patient_mean',n_cells=len(cells),n_patients=74,spearman=cohort['mean'])]).to_csv(HERE/'displayed_metrics.csv',index=False)
    assert hashes=={str(p):sha(p) for p in paths}
    (HERE/'AUDIT.json').write_text(json.dumps(dict(status='PASS',roi=ROI,patient='15219',roi_cells=650,
        identical_cells_and_coordinates=True,whole_roi=True,roi_reselected=False,threshold_search=False,
        colormap='viridis',vmin=0,vmax=1,color_values='within-patient percentile ranks',
        background='measured CODEX DAPI',foreground='cell-centroid quantitative overlays, not marker fluorescence',
        roi_spearman=rho,cohort_mean_spearman=float(cohort['mean']),inputs_unchanged=hashes),indent=2)+'\n')
    (HERE/'README_ZH.md').write_text(f'''# CD39/TIGIT 状态与实测 LAG3：真实空间展示

本文件夹只生成独立的空间双图，未改动正式 Figure 或 manuscript。

- 左：CODEX CD8 的预测 CD39/TIGIT 综合状态。
- 右：同一批 CD8 的 CODEX 实测 LAG3。
- 两图使用相同细胞、相同原始坐标、相同完整 ROI；背景是已下载的真实 CODEX DAPI。彩色圆点是细胞中心上的数值覆盖层，不是生成或模拟的 LAG3 荧光。
- 两图颜色均表示各自数值在该患者 CD8 内的百分位，统一 0–1、viridis；未扫描或裁剪颜色阈值。颜色相同表示相近的相对排序，并不代表两种量的绝对表达相等。
- 沿用患者 15219、完整 ROI `{ROI}` 的全部 650 个 CD8，未按本次 LAG3 表现重新挑 ROI，也未沿用原 61-cell 小裁剪。
- 这张 ROI 的 Spearman **{rho:.3f}**；另一项标注 **{cohort['mean']:.3f}** 是 74 位患者各自 Spearman 的等权平均，不是这张图的相关。患者均值的 bootstrap 95% CI 为 {cohort['ci_low']:.3f}–{cohort['ci_high']:.3f}。
- LAG3 未进入本次 shared22 对齐或 CD39/TIGIT 状态构建，但在历史方法探索中已被查看，因此这里展示的是后续辅助表型检验，不是从未查看过的独立盲验证。

复现：`python3 build_spatial_pair.py`。`plotted_cells.csv` 为全部绘图数值，`displayed_metrics.csv` 区分示例与队列，`AUDIT.json` 记录文件哈希和绘图参数。
''')
    print(json.dumps(dict(roi_spearman=rho,cohort_mean=float(cohort['mean']),roi_cells=650,status='PASS')))

if __name__=='__main__':main()
