"""Direct quantitative CD19 colors, with separate gray segmentation contours.

Acquired CD19 pixels are not blended with DAPI. Prediction values retain the
accepted masks and display limits. This is a spatial data rendering, not a
replacement for the acquired microscopy files.
"""
from pathlib import Path
import os
import json
import hashlib
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
os.environ['MPLCONFIGDIR'] = str(HERE/'mpl_cache')
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import maximum_filter, minimum_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.patches import Rectangle

REAL = BASE/'real_spatial0913'
OLD = BASE/'figure_layout_v30_polish'
STOPS = [(0.,'#242424'),(.25,'#5C2A83'),(.5,'#AD4777'),(.75,'#E67C52'),(1.,'#F5DA45')]
CMAP = LinearSegmentedColormap.from_list('CD19_dark_grey_purple_yellow',STOPS,N=4096)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def show(ax,image,title):
    ax.imshow(image,origin='upper',interpolation='nearest')
    ax.axis('off')
    ax.set_title(title,fontsize=12,pad=9)
    length = 50/record['pixel_size_um']
    x,y = 512-length-24,512-28
    ax.plot([x,x+length],[y,y],color='white',lw=2.1,solid_capstyle='butt')
    ax.text(x+length/2,y-10,'50 µm',ha='center',va='bottom',color='white',fontsize=10)


def save(fig,name):
    for ext in ['png','svg']:
        fig.savefig(HERE/f'{name}.{ext}',dpi=240,facecolor='white')
    plt.close(fig)


if __name__=='__main__':
    paths = [OLD/'data/H14_G_ROI_observed_CD19_raw.tif',
             OLD/'data/H14_G_ROI_observed_DAPI_raw.tif',
             OLD/'data/ROI_CORRESPONDENCE.json',
             REAL/'results/render_audit.json',
             REAL/'data/selected_roi_mask.npy',
             REAL/'data/H14_full_predictions.csv.gz',
             BASE/'panel_A_palette_trial_20260915/data/display_arrays.npz',
             OLD/'cd19_story_three_row_v30_polish.png',
             OLD/'cd19_story_three_row_v30_polish.svg',
             OLD/'cd19_story_three_row_v30_polish.pdf']
    hashes = {str(p):sha(p) for p in paths}
    raw = np.asarray(Image.open(paths[0]))
    dapi_raw = np.asarray(Image.open(paths[1]))
    record = json.loads(paths[2].read_text())
    display = json.loads(paths[3].read_text())['display']
    parent = np.load(paths[4])
    crop = np.s_[320:832,384:896]
    mask = parent[crop]
    cell_pixels = mask>0
    # One-pixel inner contours of the authors' segmentation, computed before crop.
    contour = (maximum_filter(parent,3)!=minimum_filter(parent,3))[crop] & cell_pixels
    fill = cell_pixels & ~contour
    cells = pd.read_csv(paths[5])
    length = int(max(cells.CellID.max(),parent.max()))+1
    lookup = np.full(length,np.nan)
    lookup[cells.CellID.to_numpy(int)] = cells.predicted_cd19.to_numpy()
    known = np.isfinite(lookup[mask]) & cell_pixels
    b_lookup = np.zeros(length,bool)
    b_lookup[cells.CellID.to_numpy(int)] = cells.annotated_b.to_numpy(bool)
    bmap = b_lookup[mask]
    dapi = np.clip(dapi_raw.astype(float)/display['dapi_hi'],0,1)**display['dapi_gamma']

    # DAPI informs only achromatic contour brightness; it never alters fill RGB.
    context = np.zeros((*raw.shape,3),float)
    contour_gray = .32+.28*dapi[contour]
    context[contour] = contour_gray[:,None]
    observed = context.copy()
    observed_rgb = CMAP(Normalize(0,255,clip=True)(raw))[...,:3]
    observed[fill] = observed_rgb[fill]
    predicted = context.copy()
    predicted_rgb = CMAP(Normalize(1.,1.5,clip=True)(np.nan_to_num(lookup[mask],nan=1.)))[...,:3]
    predicted[fill & known] = predicted_rgb[fill & known]
    # Missing predictions have a neutral gray fill rather than a numeric score.
    predicted[fill & ~known] = .24
    annotated = context.copy()
    annotated[fill] = .24
    annotated[fill & bmap] = CMAP(1.)[:3]

    assert raw.shape==(512,512) and raw.dtype==np.uint8
    assert np.array_equal(observed[fill],observed_rgb[fill])
    assert np.array_equal(predicted[fill & known],predicted_rgb[fill & known])
    assert np.array_equal(observed[contour],predicted[contour])
    assert np.array_equal(predicted[contour],annotated[contour])
    assert np.all(observed[~cell_pixels]==0)
    # Check luminance increases monotonically along the quantitative palette.
    rgb=CMAP(np.linspace(0,1,4096))[:,:3]
    linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
    luminance=linear@np.array([.2126,.7152,.0722])
    assert np.diff(luminance).min()>-1e-10

    plt.rcParams.update({'font.family':'Arial','svg.fonttype':'none','text.color':'#202020'})
    (HERE/'data').mkdir(exist_ok=True)
    np.savez_compressed(HERE/'data/display_arrays.npz',observed=observed,predicted=predicted,
                        annotated=annotated,context=context,quantitative_fill=fill,contour=contour)
    for name,array in [('observed',observed),('predicted',predicted),('annotated_B',annotated)]:
        plt.imsave(HERE/f'{name}_no_labels.png',array)

    fig,axes=plt.subplots(1,3,figsize=(10.8,4.25))
    for ax,array,title in zip(axes,[observed,predicted,annotated],
                             ['Observed CD19','Predicted CD19','Annotated B cells']):
        show(ax,array,title)
    fig.subplots_adjust(left=.012,right=.988,top=.89,bottom=.17,wspace=.045)
    for ax,bounds,ticks,labels,title in [
        (axes[0],(0,255),[0,32,64,128,255],['0','32','64','128','255'],'Acquired CD19 intensity'),
        (axes[1],(1.,1.5),[1.,1.25,1.5],['≤1.0','1.25','≥1.5'],'Predicted CD19 score')]:
        pos=ax.get_position()
        cax=fig.add_axes([pos.x0+.025,.12,pos.width-.05,.018])
        cb=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(*bounds),cmap=CMAP),
                       cax=cax,orientation='horizontal',ticks=ticks)
        cb.ax.set_xticklabels(labels)
        cb.ax.tick_params(labelsize=8,length=2,pad=2)
        cb.outline.set_visible(False)
        cb.set_label(title,fontsize=9,labelpad=3)
    pos=axes[2].get_position()
    fig.text((pos.x0+pos.x1)/2,.125,'Gray outlines: cell boundaries',
             ha='center',va='center',fontsize=9.5)
    save(fig,'A_G_direct_purple_yellow_trial')

    fig=plt.figure(figsize=(3.2,3.55))
    ax=fig.add_axes([.045,.025,.91,.82])
    show(ax,observed,'')
    fig.text(.045,.965,'A',va='top',fontsize=17,fontweight='bold')
    fig.text(.135,.962,'Failed CODEX CD19',va='top',fontsize=12.5,fontweight='bold')
    save(fig,'panel_A_preview')

    with np.load(paths[6]) as z:
        old_observed=z['observed']
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.85))
    show(axes[0],old_observed,'Previous: DAPI blending')
    show(axes[1],observed,'Trial: direct intensity colors')
    fig.subplots_adjust(left=.015,right=.985,top=.88,bottom=.02,wspace=.04)
    save(fig,'panel_A_before_after')

    fig,ax=plt.subplots(figsize=(8.4,1.35))
    ax.set(xlim=(0,6),ylim=(0,1));ax.axis('off')
    swatch_values=[0,17,32,64,128,255]
    for i,value in enumerate(swatch_values):
        ax.add_patch(Rectangle((i+.12,.36),.76,.40,facecolor=CMAP(value/255.),edgecolor='none'))
        ax.text(i+.5,.15,str(value),ha='center',va='center',fontsize=12)
    fig.subplots_adjust(left=.025,right=.975,bottom=.01,top=.92)
    save(fig,'observed_value_color_key')

    audit={'status':'PASS','inputs_sha256':hashes,'palette':STOPS,
           'observed_range':[0,255],'predicted_range':[1,1.5],
           'DAPI_blended_with_CD19_fill':False,
           'quantitative_fill_RGB_matches_colorbar_exactly':True,
           'palette_luminance_monotonically_increasing':True,
           'contours':'author Mesmer segmentation; gray intensity 0.32+0.28*displayed DAPI',
           'background_outside_cell_masks':'black',
           'observed_representation':'raw acquired CD19 pixels inside author cell masks, no cell averaging',
           'prediction_representation':'frozen per-cell scores on author cell masks, no DAPI texture blending',
           'annotation_representation':'author B-cell labels; gray other-cell fill',
           'missing_prediction_display':'neutral gray, not a numeric score',
           'contour_pixel_fraction_of_cell_area':float(contour.sum()/cell_pixels.sum()),
           'cell_pixel_CD19_median':float(np.median(raw[cell_pixels])),
           'cell_pixel_CD19_fraction_lt32':float((raw[cell_pixels]<32).mean()),
           'cell_pixels_including_boundary_objects':int(cell_pixels.sum()),
           'ROI':record['roi_xyxy'],'orientation_unchanged':True,
           'cell_type_labels_used_for_observed_image':False,
           'no_model_refit_or_intensity_change':True,'manuscript_replaced':False}
    assert all(sha(Path(p))==v for p,v in hashes.items())
    (HERE/'DISPLAY_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    (HERE/'README.md').write_text('''# CD19 定量配色试稿：深灰—深紫—亮黄

文件：`A_G_direct_purple_yellow_trial.png/svg` 三图对照；`panel_A_preview.png/svg`；
`panel_A_before_after.png/svg`；`observed_value_color_key.png/svg`。

CD19 填色直接使用与色条相同的 colormap 和 Normalize，不再混入 DAPI 亮度或纹理。
实测 A 在作者 Mesmer 分割轮廓内部显示原始 CD19 像素，并非每个细胞的平均值；
预测 G 仍在对应分割轮廓内显示冻结的每细胞预测值；注释图使用作者的 B-cell labels。
实测图不使用预测或 cell-type label 定色。三图同 ROI、同方向、同一套灰色轮廓。

灰色轮廓由作者分割生成，灰度为 0.32+0.28×原 G 显示参数处理后的 DAPI。
轮廓仅提供形态上下文，其亮度不表示 CD19；轮廓外背景为黑色。
因此这是基于真实像素与作者分割的定量空间显示，不是原始单通道显微图。
轮廓位置不显示定量填色；这项规则对三图一致。没有修改原始像素文件。

色板：0% 深灰 #242424、25% 深紫 #5C2A83、50% 紫红 #AD4777、
75% 橙色 #E67C52、100% 亮黄 #F5DA45；亮度随分数单调增加。
实测色条 0–255，预测色条 1.0–1.5；两者单位不同。
实测细胞区域像素中位数为 17，91.3% 低于 32；未做局部拉伸、百分位归一化或扣背景。
控制点仅定义颜色渐变，不是阳性阈值。

原正式图、此前试稿及 manuscript 均保留。参数、输入哈希和检查见 DISPLAY_AUDIT.json。
复现：`python3 render_trial.py`。
''')
    print(json.dumps({k:v for k,v in audit.items() if k!='inputs_sha256'},indent=2))
