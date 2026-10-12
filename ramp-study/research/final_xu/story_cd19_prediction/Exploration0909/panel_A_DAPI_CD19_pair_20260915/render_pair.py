"""Separate the two acquired channels; retain the accepted green/yellow G."""
from pathlib import Path
import hashlib
import json
import os
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
os.environ['MPLCONFIGDIR']=str(BASE/'panel_A_dapi_trial_20260915/mpl_cache')
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OLD=BASE/'figure_layout_v30_polish'
REAL=BASE/'real_spatial0913'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def show(ax,image,title):
    if image.ndim==2:
        ax.imshow(image,cmap='gray',vmin=0,vmax=1,origin='upper',interpolation='nearest')
    else:
        ax.imshow(image,origin='upper',interpolation='nearest')
    ax.axis('off')
    ax.set_title(title,fontsize=12,pad=7)
    length=50/record['pixel_size_um']
    x,y=512-length-24,512-28
    ax.plot([x,x+length],[y,y],color='white',lw=2.1,solid_capstyle='butt')
    ax.text(x+length/2,y-10,'50 µm',ha='center',va='bottom',color='white',fontsize=10)


def save(fig,name):
    for ext in ['png','svg']:
        fig.savefig(HERE/f'{name}.{ext}',dpi=250,facecolor='white')
    plt.close(fig)


if __name__=='__main__':
    sources=[OLD/'data/H14_G_ROI_observed_CD19_raw.tif',
             OLD/'data/H14_G_ROI_observed_DAPI_raw.tif',
             OLD/'data/ROI_CORRESPONDENCE.json',
             REAL/'results/render_audit.json',
             BASE/'figure_layout_v16/panels/G_predicted_CD19_no_labels.png',
             BASE/'figure_layout_v15/panels/G_annotated_B_no_labels.png',
             OLD/'cd19_story_three_row_v30_polish.png',
             OLD/'cd19_story_three_row_v30_polish.svg',
             OLD/'cd19_story_three_row_v30_polish.pdf']
    hashes={str(p):sha(p) for p in sources}
    raw_cd19=np.asarray(Image.open(sources[0]))
    raw_dapi=np.asarray(Image.open(sources[1]))
    record=json.loads(sources[2].read_text())
    display=json.loads(sources[3].read_text())['display']
    pred=np.asarray(Image.open(sources[4]).convert('RGB'))
    anno=np.asarray(Image.open(sources[5]).convert('RGB'))
    assert raw_cd19.shape==raw_dapi.shape==(512,512)
    assert raw_cd19.dtype==raw_dapi.dtype==np.uint8
    assert pred.shape==anno.shape==(512,512,3)

    # DAPI is exactly the same grayscale anatomy context as accepted Panel G.
    dapi=.82*np.clip(raw_dapi.astype(float)/display['dapi_hi'],0,1)**display['dapi_gamma']
    # CD19 is the original grayscale 0--255 view, with no DAPI overlay.
    cd19=raw_cd19.astype(float)/255.
    neutral=np.ptp(anno.astype(int),axis=-1)==0
    assert np.array_equal(anno[neutral],np.repeat(np.round(dapi*255).astype(np.uint8)[...,None],3,axis=2)[neutral])
    assert np.array_equal(np.round(cd19*255).astype(np.uint8),raw_cd19)
    plt.rcParams.update({'font.family':'Arial','svg.fonttype':'none','text.color':'#202020'})

    # A standalone two-image draft: no squeezing into the former single-image slot.
    fig=plt.figure(figsize=(7.2,3.95))
    for x,array,title in [(.035,dapi,'DAPI'),(.530,cd19,'Observed CD19')]:
        ax=fig.add_axes([x,.035,.435,.435*7.2/3.95])
        show(ax,array,title)
    fig.text(.035,.985,'A',ha='left',va='top',fontsize=17,fontweight='bold')
    fig.text(.081,.983,'Failed CODEX CD19',ha='left',va='top',fontsize=13,fontweight='bold')
    save(fig,'panel_A_DAPI_CD19_pair')

    # Show A with the original G pixels in a matched, two-column preview.
    fig=plt.figure(figsize=(7.2,8.2))
    height=.435*7.2/8.2
    for bottom,arrays,titles in [(.535,[dapi,cd19],['DAPI','Observed CD19']),
                                  (.045,[pred,anno],['Predicted CD19','Annotated B cells'])]:
        for x,array,title in zip([.035,.530],arrays,titles):
            ax=fig.add_axes([x,bottom,.435,height])
            show(ax,array,title)
    for y,letter,title in [(.982,'A','Failed CODEX CD19'),
                           (.492,'G','Spatial distribution of predicted CD19')]:
        fig.text(.035,y,letter,ha='left',va='top',fontsize=17,fontweight='bold')
        fig.text(.081,y-.001,title,ha='left',va='top',fontsize=13,fontweight='bold')
    save(fig,'A_G_original_palette_preview')

    plt.imsave(HERE/'DAPI_no_labels.png',dapi,cmap='gray',vmin=0,vmax=1)
    plt.imsave(HERE/'observed_CD19_no_labels.png',cd19,cmap='gray',vmin=0,vmax=1)
    audit={'status':'PASS','inputs_sha256':hashes,'ROI':record['roi_xyxy'],
           'same_orientation_dimensions_and_scale':True,
           'DAPI':{'source':'acquired channel','display_upper':display['dapi_hi'],
                   'gamma':display['dapi_gamma'],'gray_gain':.82,
                   'matches_original_G_gray_background':True},
           'CD19':{'source':'acquired channel','display_range':[0,255],
                   'gamma':1,'DAPI_blending':False,'cell_masking':False,
                   'local_contrast_or_percentile_rescaling':False},
           'Panel_G':'original accepted gray-green-yellow image arrays, unchanged',
           'no_quantitative_comparison_of_brightness_between_different_channels':True,
           'raw_data_and_prediction_values_unchanged':True,
           'figure_layout_v30_and_manuscript_replaced':False}
    assert all(sha(Path(p))==digest for p,digest in hashes.items())
    (HERE/'DISPLAY_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    (HERE/'README.md').write_text('''# Panel A：DAPI 与实测 CD19 分开显示

`panel_A_DAPI_CD19_pair.png/svg` 为 Panel A 两图并排试稿。
`A_G_original_palette_preview.png/svg` 将 A 和原版 G 放在两行便于查看对应关系，
仅为空间图配对预览，不是重新排版后的完整 Figure 3。

A 左：真实 DAPI；A 右：同一区域的真实 CD19，均为原始通道像素的灰度显示。
ROI 为 x=[13440,13952)、y=[3904,4416)，原生朝向，512×512；各图标尺 50 µm。
DAPI 显示与 G 灰色底图完全一致：上限 109、gamma 0.8、灰度增益 0.82。
CD19 恢复原 A 的 0–255 线性显示，不叠加 DAPI、不使用细胞掩膜、不调局部对比度。
两者是不同采集通道，DAPI 用于确认形态和位置，不把两图亮度作为同单位表达量比较。
CD19 的特异性与失效判断由 Panel B 的独立定量结果支持。

G 直接读取原版灰—绿—黄预测图和亮黄色 B-cell annotation，像素不变。
G 正式版中的色条和 AUROC 仍保留在原 Figure 3，未被这个局部预览替换。
此前彩色叠加与定量填色试稿均保留，正式 Figure 3 和 manuscript 尚未更新。
最终合图需为 A 的两幅图分配适当宽度，不能直接各缩为原单图的一半。

复现：`python3 render_pair.py`。参数及输入哈希见 DISPLAY_AUDIT.json。
''')
    print(json.dumps({k:v for k,v in audit.items() if k!='inputs_sha256'},indent=2))
