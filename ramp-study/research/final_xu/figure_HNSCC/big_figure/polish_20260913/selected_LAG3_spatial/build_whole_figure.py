"""Return the approved LAG3 spatial pair to the established two-row HNSCC layout."""
from pathlib import Path
import hashlib
import json
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/hnscc_selected_lag3_whole')
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fitz

HERE=Path(__file__).resolve().parent
TRIAL=HERE.parent
FIGURE=TRIAL.parent.parent
SELECTION=FIGURE/'spatial_score_review_20260913/roi_selection_20260913'
APPROVED=SELECTION/'selected_spatial_panel'
OLD=TRIAL/'program_return_trial/LAYOUT_AUDIT.json'
INK='#182027'

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def build_f():
    out=HERE/'panel_F';out.mkdir(exist_ok=True)
    cells=pd.read_csv(SELECTION/'selected_roi_cells.csv',dtype={'patient':str})
    assert len(cells)==115 and cells.unique_region.eq('522_S1reg067').all()
    plane_path=SELECTION/'images/522_S1reg067/DAPI.raw'
    meta=json.loads((plane_path.parent/'DOWNLOAD.json').read_text())
    assert sha(plane_path)==meta['sha256']
    plane=np.fromfile(plane_path,dtype=meta['endian']+'u2').reshape(1440,1920).astype(float)
    background=np.clip(plane/np.percentile(plane[plane>0],99.8),0,1)**.72*.30
    approved=json.loads((APPROVED/'AUDIT.json').read_text())
    rho=float(spearmanr(cells.predicted_score,cells.LAG3).statistic)
    np.testing.assert_allclose(rho,approved['selected_roi']['rho'])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none','text.color':INK})
    W,H=7.35,3.45
    fig=plt.figure(figsize=(W,H))
    fig.text(.08/W,.976,'F',fontsize=23,weight='bold',va='top')
    fig.text(.51/W,.947,'SPATIAL PHENOTYPE CORRESPONDENCE',fontsize=11.5,weight='bold',va='top')
    for x,title,column in [(.06,'Predicted CD39/TIGIT state','within_patient_rank'),(3.36,'Measured LAG3','LAG3_patient_rank')]:
        ax=fig.add_axes([x/W,.40/H,3.20/W,2.40/H])
        ax.imshow(background,cmap='gray',vmin=0,vmax=1,extent=(0,1920,1440,0),interpolation='nearest')
        dots=ax.scatter(cells.x,cells.y,c=cells[column],s=18*(3.20/4)**2,cmap='viridis',vmin=0,vmax=1,
            edgecolors='#CFD7DC',linewidths=.16*(3.20/4),alpha=1,rasterized=True)
        np.testing.assert_array_equal(dots.get_offsets(),cells[['x','y']].to_numpy())
        np.testing.assert_array_equal(dots.get_array(),cells[column].to_numpy())
        ax.set(xlim=(0,1920),ylim=(1440,0));ax.set_axis_off()
        fig.text((x+1.60)/W,2.88/H,title,ha='center',va='bottom',fontsize=10.2,weight='bold')
    cax=fig.add_axes([6.73/W,.79/H,.085/W,1.57/H])
    cb=fig.colorbar(dots,cax=cax,ticks=[0,.5,1]);cb.ax.tick_params(labelsize=7.8,length=2,pad=2)
    cb.ax.set_title('Rank',fontsize=8.2,pad=6);cb.outline.set_linewidth(.5)
    footer=approved['display_footer'];assert footer=='74 patients: mean Spearman ρ = 0.52'
    fig.text(.5,.13/H,footer,ha='center',va='center',fontsize=9.9)
    fig.canvas.draw();r=fig.canvas.get_renderer()
    for t in list(fig.texts)+list(cb.ax.get_yticklabels())+[cb.ax.title]:
        b=t.get_window_extent(r)
        assert b.x0>=-1 and b.y0>=-1 and b.x1<=fig.bbox.width+1 and b.y1<=fig.bbox.height+1,(t.get_text(),b.bounds)
    path=out/'panel_F_selected_LAG3.pdf'
    for ext in ['pdf','png','svg']:fig.savefig(path.with_suffix('.'+ext),dpi=300,facecolor='white')
    plt.close(fig)
    cells.to_csv(out/'plotted_cells.csv',index=False)
    (out/'CAPTION.md').write_text((APPROVED/'CAPTION.md').read_text())
    return path,dict(roi='522_S1reg067',patient='23956',cells=115,roi_spearman=rho,
                    footer=footer,cohort_mean=approved['cohort_mean'],color_limits=[0,1],full_roi=True,
                    real_DAPI_sha256=sha(plane_path),spatial_values_unchanged=True)

def main():
    HERE.mkdir(exist_ok=True)
    old=json.loads(OLD.read_text())
    source_paths=[Path(p['source']) for k,p in old['placements'].items() if k!='F']
    source_paths+=[SELECTION/'selected_roi_cells.csv',SELECTION/'images/522_S1reg067/DAPI.raw',APPROVED/'AUDIT.json',APPROVED/'CAPTION.md']
    before={str(p):sha(p) for p in source_paths}
    f,details=build_f()
    doc=fitz.open();page=doc.new_page(width=16.99*72,height=6.98*72);slots={}
    for letter,oldslot in old['placements'].items():
        source=f if letter=='F' else Path(oldslot['source'])
        rect=fitz.Rect(*[v*72 for v in oldslot['bounds_inches']])
        with fitz.open(source) as src:
            np.testing.assert_allclose([rect.width,rect.height],[src[0].rect.width,src[0].rect.height],atol=.01)
            page.show_pdf_page(rect,src,0)
        slots[letter]=dict(source=str(source),sha256=sha(source),bounds_inches=oldslot['bounds_inches'])
    title='HNSCC: selected LAG3 spatial phenotype in the two-row story figure'
    doc.set_metadata({'title':title})
    target=HERE/'hnscc_selected_LAG3_spatial.pdf'
    doc.save(target,garbage=4,deflate=True)
    page.get_pixmap(dpi=180,alpha=False).save(target.with_suffix('.png'))
    page.get_pixmap(dpi=220,alpha=False,clip=fitz.Rect(0,3.53*72,16.99*72,6.98*72)).save(HERE/'second_row_preview.png')
    doc.close()
    with fitz.open(target) as check:
        text=check[0].get_text()
        assert 'Highest correlation' not in text
        assert 'Observed spatial niche' not in text
        assert 'Measured LAG3' in text and 'mean Spearman' in text
        assert len(check)==1
    assert before=={p:sha(Path(p)) for p in before}
    (HERE/'LAYOUT_AUDIT.json').write_text(json.dumps(dict(status='PASS',layout='A B D / C F E',size_inches=[16.99,6.98],
        placements=slots,source_layout=str(OLD),changes='F replaced with selected full-ROI CD39/TIGIT state versus measured LAG3',
        panel_F=details,other_panels_reused_without_changes=True,inputs_unchanged=before,
        model_refit=False,cohort_statistics_changed=False,formal_figure_replaced=False,manuscript_replaced=False),indent=2)+'\n')
    (HERE/'README_ZH.md').write_text('''# HNSCC 整图：选定的 LAG3 空间示例

整图：`hnscc_selected_LAG3_spatial.png/.pdf`。本版本将用户选定的 115-cell 空间双图放回此前确认的两行布局，尺寸保持 16.99 × 6.98 英寸。

- 上排：A workflow（放大的贴框 RAMP 图标）、B 空间 niche 与生存、D 紧凑 marker 筛选。
- 下排：C 两张 UMAP、F 新空间双图、E 紧凑的 protein/RNA 高低状态表征。
- A–E 使用已保存的 PDF，未改数据或重新绘制。F 替换旧的 niche 返回示例，采用预测 CD39/TIGIT 状态与实测 LAG3 的对应。
- F 保留完整 ROI 522_S1reg067、同一批 115 个 CD8、真实 DAPI 底图及原来 0–1 的患者内百分位色阶。底部仅保留居中的 “74 patients: mean Spearman ρ = 0.52”。
- ROI 的展示选择及局部 rho=0.732 保留在 `panel_F/CAPTION.md`；图中文字 0.52 是 74 位患者各自相关的平均值，不是该 ROI 的值。

这里只更新试稿，原整图和 manuscript 保留。`build_whole_figure.py` 可复现，`LAYOUT_AUDIT.json` 记录所有来源、位置与哈希。
''')
    print('PASS: full two-row figure assembled with approved spatial ROI; other panels unchanged.')

if __name__=='__main__':main()
