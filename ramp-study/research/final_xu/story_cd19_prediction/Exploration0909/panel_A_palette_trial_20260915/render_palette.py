"""Replot acquired/predicted CD19 with a shared blue-orange display palette."""
from pathlib import Path
import os
import json
import hashlib
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
os.environ['MPLCONFIGDIR'] = str(BASE/'panel_A_dapi_trial_20260915/mpl_cache')
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import maximum_filter, minimum_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb

REAL = BASE/'real_spatial0913'
OLD = BASE/'figure_layout_v30_polish'
LOW = np.array(to_rgb('#3478DE'))
HIGH = np.array(to_rgb('#FF8A24'))
ANCHOR = .25


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def colors(level, grey, texture, low_rgb, high_rgb, use=None, boundary=None):
    low_color = texture*low_rgb
    high_color = texture*high_rgb
    if boundary is not None:
        low_color[boundary] *= .78
        high_color[boundary] *= .78
    if use is None:
        use = np.ones(level.shape, bool)
    image = grey.copy()
    low = use & (level <= ANCHOR)
    high = use & (level > ANCHOR)
    a = (level[low]/ANCHOR)[:,None]
    image[low] = (1-a)*grey[low]+a*low_color[low]
    a = ((level[high]-ANCHOR)/(1-ANCHOR))[:,None]
    image[high] = (1-a)*low_color[high]+a*high_color[high]
    return image, high_color


def show(ax, image, title):
    ax.imshow(image, interpolation='nearest', origin='upper')
    ax.axis('off')
    ax.set_title(title, fontsize=12, pad=9)
    length = 50/record['pixel_size_um']
    x,y = 512-length-24,512-28
    ax.plot([x,x+length],[y,y],color='white',lw=2.1,solid_capstyle='butt')
    ax.text(x+length/2,y-10,'50 µm',ha='center',va='bottom',color='white',fontsize=10)


def save(fig,name):
    for ext in ['png','svg']:
        fig.savefig(HERE/f'{name}.{ext}',dpi=240,facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    paths = [OLD/'data/H14_G_ROI_observed_CD19_raw.tif',
             OLD/'data/H14_G_ROI_observed_DAPI_raw.tif',
             OLD/'data/ROI_CORRESPONDENCE.json',
             REAL/'results/render_audit.json',
             REAL/'data/selected_roi_mask.npy',
             REAL/'data/H14_full_predictions.csv.gz',
             BASE/'figure_layout_v16/panels/G_predicted_CD19_no_labels.png',
             BASE/'figure_layout_v15/panels/G_annotated_B_no_labels.png',
             BASE/'panel_A_dapi_trial_20260915/data/display_arrays.npz',
             OLD/'cd19_story_three_row_v30_polish.png']
    hashes = {str(p):sha(p) for p in paths}
    raw = np.asarray(Image.open(paths[0]))
    raw_dapi = np.asarray(Image.open(paths[1]))
    record = json.loads(paths[2].read_text())
    display = json.loads(paths[3].read_text())['display']
    parent_mask = np.load(paths[4])
    mask = parent_mask[320:832,384:896]
    boundary = (maximum_filter(parent_mask,3)!=minimum_filter(parent_mask,3))[320:832,384:896] & (mask>0)
    cells = pd.read_csv(paths[5])
    length = int(max(cells.CellID.max(),parent_mask.max()))+1
    lookup = np.full(length,np.nan)
    lookup[cells.CellID.to_numpy(int)] = cells.predicted_cd19.to_numpy()
    b_lookup = np.zeros(length,bool)
    b_lookup[cells.CellID.to_numpy(int)] = cells.annotated_b.to_numpy(bool)
    bmap = b_lookup[mask]
    known = np.isfinite(lookup[mask]) & (mask>0)
    dapi = np.clip(raw_dapi.astype(float)/display['dapi_hi'],0,1)**display['dapi_gamma']
    grey = np.repeat((.82*dapi)[...,None],3,axis=2)
    texture = (.85+.15*dapi)[...,None]
    observed_level = raw.astype(float)/255.
    predicted_level = np.clip((np.nan_to_num(lookup[mask],nan=1.)-1.)/.5,0,1)
    original_low,original_high = np.array([.06,.62,.29]),np.array([1.,.91,.12])

    # First reproduce the accepted views exactly, verifying cell lookup and crop.
    old_observed,_ = colors(observed_level,grey,texture,original_low,original_high)
    old_pred,old_high = colors(predicted_level,grey,texture,original_low,original_high,known,boundary)
    old_anno = grey.copy()
    old_anno[bmap] = old_high[bmap]
    with np.load(paths[8]) as z:
        assert np.array_equal(old_observed,z['observed'])
        assert np.array_equal(grey,z['dapi_background'])
    assert np.array_equal(np.round(old_pred*255).astype(np.uint8),np.asarray(Image.open(paths[6])))
    assert np.array_equal(np.round(old_anno*255).astype(np.uint8),np.asarray(Image.open(paths[7])))

    observed,_ = colors(observed_level,grey,texture,LOW,HIGH)
    predicted,high_color = colors(predicted_level,grey,texture,LOW,HIGH,known,boundary)
    annotated = grey.copy()
    annotated[bmap] = high_color[bmap]
    assert np.array_equal(observed[raw==0],grey[raw==0])
    assert np.array_equal(predicted[~known],grey[~known])
    assert np.array_equal(predicted[known & (predicted_level==0)],grey[known & (predicted_level==0)])
    assert np.array_equal(annotated[~bmap],grey[~bmap])
    (HERE/'data').mkdir(exist_ok=True)
    np.savez_compressed(HERE/'data/display_arrays.npz',observed=observed,predicted=predicted,annotated=annotated)
    for label,array in [('observed',observed),('predicted',predicted),('annotated_B',annotated)]:
        plt.imsave(HERE/f'{label}_no_labels.png',array)
    plt.rcParams.update({'font.family':'Arial','svg.fonttype':'none','text.color':'#202020'})
    cmap = LinearSegmentedColormap.from_list('CD19_grey_blue_orange',
                [(0,np.repeat(.82*.5,3)),(ANCHOR,.925*LOW),(1,.925*HIGH)])
    fig,axes = plt.subplots(1,3,figsize=(10.8,4.25))
    for ax,array,title in zip(axes,[observed,predicted,annotated],
                             ['Observed CD19','Predicted CD19','Annotated B cells']):
        show(ax,array,title)
    fig.subplots_adjust(left=.012,right=.988,top=.89,bottom=.17,wspace=.045)
    for ax,bounds,ticks,ticklabels,label in [
        (axes[0],(0,255),[0,32,64,128,255],['0','32','64','128','255'],'Acquired CD19 intensity'),
        (axes[1],(1,1.5),[1,1.125,1.25,1.5],['≤1.0','1.125','1.25','≥1.5'],'Predicted CD19 score')]:
        pos=ax.get_position()
        cax=fig.add_axes([pos.x0+.025,.12,pos.width-.05,.018])
        cb=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(*bounds),cmap=cmap),cax=cax,orientation='horizontal',ticks=ticks)
        cb.ax.set_xticklabels(ticklabels)
        cb.ax.tick_params(labelsize=8,length=2,pad=2)
        cb.outline.set_visible(False)
        cb.set_label(label,fontsize=9,labelpad=3)
    save(fig,'A_G_blue_orange_trial')

    fig,axes=plt.subplots(1,2,figsize=(7.2,3.85))
    show(axes[0],old_observed,'Previous: green–yellow')
    show(axes[1],observed,'Trial: blue–orange')
    fig.subplots_adjust(left=.015,right=.985,top=.88,bottom=.02,wspace=.04)
    save(fig,'A_palette_comparison')

    audit={'status':'PASS','inputs_sha256':hashes,
           'palette':{'neutral':'original grey DAPI','intermediate':'#3478DE','high':'#FF8A24','intermediate_position':ANCHOR},
           'observed_range':[0,255],'predicted_range':[1,1.5],
           'DAPI_display':{'hi':display['dapi_hi'],'gamma':display['dapi_gamma'],'gain':.82},
           'old_green_yellow_views_reproduced_exactly':True,
           'only_palette_changes_in_display_arrays':True,
           'acquired_pixels_and_prediction_values_unchanged':True,
           'observed_panel_uses_cell_labels_or_predictions':False,
           'same_ROI_orientation_and_scale':True,
           'cell_pixel_CD19_median':float(np.median(raw[mask>0])),
           'cell_pixel_CD19_fraction_below_32':float((raw[mask>0]<32).mean()),
           'manuscript_replaced':False}
    assert all(sha(Path(p))==v for p,v in hashes.items())
    (HERE/'DISPLAY_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    (HERE/'README.md').write_text('''# CD19 配色试稿：灰—蓝—橙

`A_G_blue_orange_trial.png/svg`：实测 CD19、预测 CD19、B-cell annotation。
`A_palette_comparison.png/svg`：Panel A 原绿黄色与新蓝橙色比较。

仅更换颜色控制点：灰色 DAPI → 蓝色 #3478DE → 橙色 #FF8A24。
B-cell annotation 同步使用高值橙色。保持 DAPI 显示、纹理、边界、ROI 和方向。
实测仍为 0–255；预测仍为 1.0–1.5；中间色控制点仍在显示范围的 25%。
色条补充低值刻度以帮助读数。实测与预测单位不同，没有套用共同数值色标。
重绘前逐像素复现原 A/G 试稿并通过检查；数据、ROI、预测模型均未改动。

灰色 DAPI 提供细胞形态背景，其亮度不是 CD19 表达。由于显示合成叠加底图，
色条表示参考底色上的强度颜色，不能凭合成图亮度精确读取原始值。
该 ROI 的细胞区域 CD19 像素中位数为 17，91.3% 低于 32。
旧正式图、旧试稿及 manuscript 均保留。

复现：`python3 render_palette.py`；参数与输入哈希见 `DISPLAY_AUDIT.json`。
''')
    print(json.dumps({k:v for k,v in audit.items() if k!='inputs_sha256'},indent=2))
