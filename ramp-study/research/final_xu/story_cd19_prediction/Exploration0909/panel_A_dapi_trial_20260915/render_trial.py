"""Scientific display trial: acquired CD19 over the frozen Panel G DAPI field.

No image synthesis, per-cell predictions, annotations, or segmentation are used
to create the observed-CD19 panel. Raw CD19 remains linearly scaled 0--255.
"""
from pathlib import Path
import hashlib
import json
import os

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
os.environ['MPLCONFIGDIR'] = str(HERE / 'mpl_cache')
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

OLD = BASE / 'figure_layout_v30_polish'
REAL = BASE / 'real_spatial0913'
GREEN = np.array([.06, .62, .29])
YELLOW = np.array([1., .91, .12])
GREEN_POSITION = .25


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def show(ax, pixels, heading=None, channels=False):
    if pixels.ndim == 2:
        ax.imshow(pixels, cmap='gray', vmin=0, vmax=255,
                  origin='upper', interpolation='nearest')
    else:
        ax.imshow(pixels, origin='upper', interpolation='nearest')
    ax.set_axis_off()
    if heading:
        ax.set_title(heading, fontsize=12, pad=9)
    bar = 50 / record['pixel_size_um']
    x, y = 512-bar-24, 512-28
    ax.plot([x, x+bar], [y, y], color='white', lw=2.2,
            solid_capstyle='butt')
    ax.text(x+bar/2, y-10, '50 µm', color='white', fontsize=10,
            ha='center', va='bottom')
    if channels:
        ax.text(.96, .955, 'DAPI', color='#D1D1D1', transform=ax.transAxes,
                fontsize=10.5, ha='right', va='top')
        ax.text(.96, .888, 'CD19', color=YELLOW, transform=ax.transAxes,
                fontsize=10.5, ha='right', va='top')


def savefig(fig, stem):
    for ext in ['png', 'svg']:
        fig.savefig(HERE / f'{stem}.{ext}', dpi=240, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    inputs = [
        OLD / 'data/H14_G_ROI_observed_CD19_raw.tif',
        OLD / 'data/H14_G_ROI_observed_DAPI_raw.tif',
        OLD / 'data/ROI_CORRESPONDENCE.json',
        REAL / 'results/render_audit.json',
        BASE / 'figure_layout_v16/panels/G_predicted_CD19_no_labels.png',
        BASE / 'figure_layout_v15/panels/G_annotated_B_no_labels.png',
        OLD / 'cd19_story_three_row_v30_polish.png',
        OLD / 'cd19_story_three_row_v30_polish.svg',
        OLD / 'cd19_story_three_row_v30_polish.pdf',
    ]
    before = {str(p): sha(p) for p in inputs}
    raw = np.asarray(Image.open(inputs[0]))
    dapi_raw = np.asarray(Image.open(inputs[1]))
    record = json.loads(inputs[2].read_text())
    display = json.loads(inputs[3].read_text())['display']
    pred = np.asarray(Image.open(inputs[4]).convert('RGB'))
    annotation = np.asarray(Image.open(inputs[5]).convert('RGB'))
    assert raw.shape == dapi_raw.shape == (512, 512)
    assert pred.shape == annotation.shape == (512, 512, 3)
    assert raw.dtype == np.uint8

    # Exactly the same DAPI display as Panel G: hi=109, gamma=.8, gain=.82.
    dapi = np.clip(dapi_raw.astype(float)/display['dapi_hi'], 0, 1)
    dapi **= display['dapi_gamma']
    grey = np.repeat((.82*dapi)[..., None], 3, axis=2)
    texture = (.85 + .15*dapi)[..., None]
    green, yellow = texture*GREEN, texture*YELLOW
    # Reuse G's piecewise palette, but map the acquired CD19 pixel intensity.
    # There is no background threshold, percentile stretching, or cell selection.
    level = raw.astype(float)/255.
    low = level <= GREEN_POSITION
    high = ~low
    observed = grey.copy()
    alpha = (level[low]/GREEN_POSITION)[:, None]
    observed[low] = (1-alpha)*grey[low] + alpha*green[low]
    alpha = ((level[high]-GREEN_POSITION)/(1-GREEN_POSITION))[:, None]
    observed[high] = (1-alpha)*green[high] + alpha*yellow[high]
    assert np.array_equal(observed[raw == 0], grey[raw == 0])
    assert np.all((observed >= 0) & (observed <= 1))

    # Verify that the reused background really matches the unchanged G image.
    neutral = np.ptp(annotation.astype(int), axis=-1) == 0
    grey_u8 = np.round(grey*255).astype(np.uint8)
    assert neutral.sum() > raw.size/2
    assert np.array_equal(annotation[neutral], grey_u8[neutral])

    (HERE / 'data').mkdir(exist_ok=True)
    np.savez_compressed(HERE/'data/display_arrays.npz',
                        observed=observed, dapi_background=grey)
    plt.imsave(HERE/'observed_CD19_DAPI_no_labels.png', observed)
    plt.rcParams.update({'font.family':'Arial', 'svg.fonttype':'none',
                         'text.color':'#202020'})

    fig = plt.figure(figsize=(3.2, 3.55))
    ax = fig.add_axes([.045, .025, .91, .82])
    show(ax, observed, channels=True)
    fig.text(.045,.965,'A',va='top',fontsize=17,fontweight='bold')
    fig.text(.135,.962,'Failed CODEX CD19',va='top',fontsize=12.5,fontweight='bold')
    savefig(fig, 'panel_A_preview')

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.85))
    show(axes[0], raw, 'Previous: CD19 only')
    show(axes[1], observed, 'Trial: DAPI + observed CD19', channels=True)
    fig.subplots_adjust(left=.015,right=.985,top=.88,bottom=.02,wspace=.04)
    savefig(fig, 'panel_A_before_after')

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 4.2))
    show(axes[0], observed, 'Observed CD19', channels=True)
    show(axes[1], pred, 'Predicted CD19')
    show(axes[2], annotation, 'Annotated B cells')
    fig.subplots_adjust(left=.012,right=.988,top=.89,bottom=.16,wspace=.045)
    cmap = LinearSegmentedColormap.from_list('CD19_grey_green_yellow',
        [(0, np.repeat(.82*.5,3)),
         (GREEN_POSITION,(.85+.15*.5)*GREEN),
         (1,(.85+.15*.5)*YELLOW)])
    for ax, bounds, label, ticks, ticklabels in [
        (axes[0], (0,255), 'Acquired CD19 intensity', [0,128,255], ['0','128','255']),
        (axes[1], (1,1.5), 'Predicted CD19 score', [1,1.25,1.5], ['≤1.0','1.25','≥1.5']),
    ]:
        pos = ax.get_position()
        cax = fig.add_axes([pos.x0+.04, .118, pos.width-.08, .018])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(*bounds),cmap=cmap),
                          cax=cax,orientation='horizontal',ticks=ticks)
        cb.ax.set_xticklabels(ticklabels)
        cb.ax.tick_params(labelsize=8,length=2,pad=2)
        cb.outline.set_visible(False)
        cb.set_label(label,fontsize=9,labelpad=3)
    savefig(fig, 'A_G_same_ROI_trial')

    audit = {
        'status':'PASS', 'inputs_sha256':before,
        'raw_CD19_median':float(np.median(raw)),
        'raw_CD19_mean':float(raw.mean()),
        'raw_CD19_fraction_le5':float((raw<=5).mean()),
        'raw_CD19_fraction_above_green_anchor':float(high.mean()),
        'CD19_display':{'lower':0,'upper':255,'gamma':1,'green_at':63.75,
                        'yellow_at':255,'percentile_rescaling':False},
        'DAPI_display':{'upper':display['dapi_hi'],'gamma':display['dapi_gamma'],'gain':.82},
        'raw_zero_pixels_equal_DAPI_background':True,
        'DAPI_matches_G_neutral_pixels_exactly':True,
        'observed_panel_uses_predictions_or_annotations':False,
        'acquired_pixels_are_not_synthetic':True,
        'segmented_masks_used_for_observed_panel':False,
        'Panel_G_source_images_unchanged':True,
        'ROI':record['roi_xyxy'],
        'same_orientation_and_scale_as_G':True,
        'old_figure_and_manuscript_replaced':False,
    }
    assert all(sha(Path(p)) == digest for p,digest in before.items())
    (HERE/'DISPLAY_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    (HERE/'README.md').write_text('''# Figure 3 Panel A：DAPI 背景试稿

仅在本目录保存试稿，原 Figure 3、Panel G 和 manuscript 均未替换。

`panel_A_preview.png` 是新版 A；`panel_A_before_after.png` 是前后对比；
`A_G_same_ROI_trial.png` 将新版 A 与原 G 的两张图放在一起。

使用原 H14 ROI（13440,3904–13952,4416）的真实 DAPI 和真实 CD19 像素。
与 G 完全相同的 DAPI 显示：上限 109、gamma 0.8、灰色增益 0.82。
CD19 沿用原版 0–255 线性显示，按像素从灰色底图过渡到绿色（63.75）再到黄色（255）。
颜色控制点沿用 G 的相对位置，不代表阳性判定阈值。CD19 未做局部对比度、
百分位拉伸或背景扣除。真实 CD19 的中位数为 2，约 80% 的像素不超过 5。

A 是两路实测图像的显示合成，灰色底图的亮度表示 DAPI，不能解释为 CD19 表达。
G 的预测图仍使用作者分割轮廓上的预测值，B-cell 图仍显示作者注释。
A 不使用标签、预测值或分割掩膜修改 CD19 信号；三图 ROI、方向及比例尺相同。
实测 CD19 的原始强度与预测 CD19 的标准化分数单位不同，并列预览分别列出显示色标。

复现：`python3 render_trial.py`。输入哈希与显示参数见 `DISPLAY_AUDIT.json`。
''')
    print(json.dumps({k:v for k,v in audit.items() if k!='inputs_sha256'},indent=2))
    print(HERE/'A_G_same_ROI_trial.png')
