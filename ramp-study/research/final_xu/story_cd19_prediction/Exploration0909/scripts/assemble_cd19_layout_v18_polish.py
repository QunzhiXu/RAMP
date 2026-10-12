"""ADTnorm-inspired plot framing and typography, using the frozen v17 content."""
from runtime import BASE, save, sha, now
import hashlib
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib.text import Text, Annotation
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse
from matplotlib.transforms import Bbox
import assemble_cd19_layout_v17 as old

OUT = BASE / 'figure_layout_v18_polish'
QA = OUT / 'qa'
QA.mkdir(parents=True, exist_ok=True)
v, v8 = old.v, old.v8
W, H = 14.5, 10.5
INK = '#202020'
FRAME = '#535353'
# 0.92 pt at the working size corresponds to 0.45 pt at 180 mm width.
FRAME_LW = .92
TITLES = {
    'A': 'Failed CODEX CD19',
    'B': 'Detection of failed CD19',
    'C': 'Cross-platform CD19 recovery',
    'D': 'Held-out CD3 validation',
    'E': 'Prediction accuracy',
    'F': 'CD19 restoration',
    'G': 'Spatial distribution of predicted CD19',
    'H': 'Mature B-NK separation',
}
PANELS = [
    ('A', v.panel_a, (.15, .13, 2.50, 2.55)),
    ('B', v8.v6.panel_b, (2.76, .13, 5.70, 2.55)),
    ('C', v8.v3.panel_c, (8.57, .13, 5.78, 2.55)),
    ('D', v8.v3.panel_d, (.15, 2.79, 5.45, 3.50)),
    ('E', old.previous.previous.previous.panel_e, (5.695, 2.79, 4.30, 3.50)),
    ('F', v8.v7.panel_f, (10.09, 2.79, 4.26, 3.50)),
    ('G', old.panel_g, (.15, 6.40, 6.72, 3.95)),
    ('H', v8.v3.panel_h, (7.00, 6.40, 7.35, 3.95)),
]


def digest_array(array):
    array = np.ascontiguousarray(np.ma.filled(array, np.nan))
    return hashlib.sha256(array.tobytes()).hexdigest()


def payload(axes):
    """Capture the scientific artists, including color mappings and image pixels."""
    return [{
        'xlim': list(ax.get_xlim()), 'ylim': list(ax.get_ylim()),
        'collections': [{
            'offsets': digest_array(c.get_offsets()),
            'values': None if c.get_array() is None else digest_array(c.get_array()),
            'clim': list(c.get_clim()),
            'facecolors': digest_array(c.get_facecolors()),
            'alpha': c.get_alpha(),
            'paths': [digest_array(p.vertices) for p in c.get_paths()],
        } for c in ax.collections],
        'images': [digest_array(im.get_array()) for im in ax.images],
        'lines': [[digest_array(line.get_xdata()), digest_array(line.get_ydata())]
                  for line in ax.lines],
    } for ax in axes]


def frame_axis(ax, hide_ticks=False):
    ax.set_axis_on()
    if hide_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(FRAME)
        spine.set_linewidth(FRAME_LW)
    ax.tick_params(colors=INK, width=.8, labelsize=10)
    ax.xaxis.label.set(color=INK, fontsize=10.5, fontweight='normal')
    ax.yaxis.label.set(color=INK, fontsize=10.5, fontweight='normal')


def polish(fig, groups, headings):
    for text in fig.findobj(match=Text):
        text.set_fontfamily('Arial')
        text.set_fontweight('normal')
        # Keep quantitative labels legible while reducing bold text dominance.
        if text.get_text().strip():
            text.set_fontsize(max(9.5, min(text.get_fontsize(), 11.5)))
    for letter, _, rect in PANELS:
        original = headings[letter]
        original.remove()
        x, y, _, _ = rect
        v.text_at(fig, x+.02, y+.015, letter, fontsize=16, fontweight='bold', color=INK)
        v.text_at(fig, x+.30, y+.035, TITLES[letter], fontsize=12.8,
                  fontweight='bold', color=INK)
    for axes in groups.values():
        for ax in axes:
            for title in [ax.title, ax._left_title, ax._right_title]:
                title.set(fontsize=11.5, fontweight='normal', color=INK)

    for letter in ['B', 'E', 'F']:
        for ax in groups[letter]:
            frame_axis(ax)
            ax.grid(False, axis='x')
            if letter in ['B', 'E']:
                ax.yaxis.grid(True, color='#E5E5E5', linewidth=.55)
                ax.set_axisbelow(True)
    for ax in groups['D'][:2] + groups['H']:
        frame_axis(ax, hide_ticks=True)
    # Align the upper plotting edges of D, E and F without changing data limits.
    for ax in groups['D'][:2]:
        ax.apply_aspect()
    top = groups['D'][0].get_position().y1
    for ax in [groups['E'][0], *groups['F']]:
        p = ax.get_position()
        ax.set_position([p.x0, p.y0, p.width, top-p.y0])
    # Median labels below the high-valued groups avoid the new top frame.
    for text in groups['B'][1].texts:
        if text.get_text() in ['median 1.00', 'median 0.99']:
            text.set_y(.94)
            text.set_verticalalignment('top')
        if text.get_text() == 'median 0.99':
            text.set_x(1.98)

    # Neutral workflow boxes; retain the same five nodes, content and connections.
    workflow_ax = groups['C'][0]
    for patch in workflow_ax.patches:
        if isinstance(patch, FancyBboxPatch):
            patch.set(facecolor='white', edgecolor=FRAME, linewidth=FRAME_LW)
            patch.set_boxstyle('round,pad=0.01,rounding_size=0.012')
        elif isinstance(patch, FancyArrowPatch):
            patch.set(color=FRAME, linewidth=1.0, mutation_scale=11)
    for text in workflow_ax.texts:
        text.set(color=INK, fontsize=10.5)
        if text.get_text() in ['CODEX', 'References', 'Masked-marker NN',
                               'Completed CODEX', 'RAMP alignment']:
            text.set(fontweight='bold', fontsize=11.2)
        if text.get_text() == 'CODEX':
            text.set_color('#3E70B2')
        elif text.get_text() == 'References':
            text.set_color('#31916E')
        elif text.get_text() == 'Completed CODEX':
            text.set_fontsize(9.8)
        elif text.get_text() == 'RAMP alignment':
            text.set_fontsize(10.2)

    # One thin outline per highlighted population; unchanged ellipse geometry.
    for ax in groups['H']:
        for patch in ax.patches:
            if isinstance(patch, Ellipse):
                patch.set_linewidth(1.05)
        for text in ax.texts:
            text.set_fontsize(10.5)
            if isinstance(text, Annotation) and text.arrow_patch is not None:
                text.arrow_patch.set_linewidth(.95)
            if text.get_text() == 'Mature B + NK mixed':
                text.set_position((8.3, 13.1))

    # Figures on real DAPI keep the exact v17 pixels, bounds and labels.
    for ax in groups['G'][:2]:
        ax.title.set_fontsize(11.5)
    for text in fig.texts:
        if text.get_text().startswith('Mean B-cell identity AUROC:'):
            text.set(fontsize=11, fontweight='normal', color=INK)
        elif text.get_text().startswith('Mature B') and 'silhouette:' in text.get_text():
            text.set(fontsize=10.5, fontweight='normal', color=INK)
    for legend in fig.legends:
        for text in legend.get_texts():
            text.set(fontfamily='Arial', fontsize=11, fontweight='normal', color=INK)


def comparison():
    old_image = Image.open(BASE / 'figure_layout_v17/cd19_story_three_row_v17.png').convert('RGB')
    new_image = Image.open(OUT / 'cd19_story_three_row_v18_polish.png').convert('RGB')
    width = 1800
    height = round(width*H/W)
    canvas = Image.new('RGB', (2*width+60, height+85), 'white')
    font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 30)
    draw = ImageDraw.Draw(canvas)
    for x, label, im in [(20, 'Original v17', old_image),
                         (width+40, 'Polish trial v18', new_image)]:
        draw.text((x+10, 17), label, fill=INK, font=font)
        canvas.paste(im.resize((width, height), Image.Resampling.LANCZOS), (x, 65))
    canvas.save(OUT / 'v17_vs_v18_comparison.png')


def main():
    plt.rcParams.update({'font.family': 'Arial', 'pdf.fonttype': 42,
                         'svg.fonttype': 'none', 'text.color': INK,
                         'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK})
    fig = plt.figure(figsize=(W, H), facecolor='white')
    groups, headings, data_before = {}, {}, {}
    for letter, draw, rect in PANELS:
        n_axes, n_texts = len(fig.axes), len(fig.texts)
        draw(fig, rect)
        groups[letter] = fig.axes[n_axes:]
        headings[letter] = fig.texts[n_texts]
        if letter != 'C':
            data_before[letter] = payload(groups[letter])
    polish(fig, groups, headings)
    for letter in data_before:
        assert data_before[letter] == payload(groups[letter]), letter
    # Colorbars retain the numerical limits and grey-category legend of v17.
    assert groups['G'][-1].get_ylim() == (1., 1.5)
    assert any(t.get_text() == '≤1.0' for t in fig.texts)
    assert len(groups['E'][0].collections[0].get_offsets()) == 30000
    assert sum(len(c.get_offsets()) for c in groups['E'][1].collections) == 48
    assert all(sum(len(c.get_offsets()) for c in ax.collections) == 88031 for ax in groups['H'])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    visible_text = list(fig.texts)
    for ax in fig.axes:
        visible_text.extend(ax.texts)
        visible_text.extend([ax.title, ax._left_title, ax._right_title])
        if ax.axison:
            visible_text.extend([ax.xaxis.label, ax.yaxis.label])
            for axis, limits in [(ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())]:
                for tick in axis.get_major_ticks():
                    if min(limits) <= tick.get_loc() <= max(limits):
                        visible_text.extend([tick.label1, tick.label2])
    for text in visible_text:
        if text.get_visible() and text.get_text().strip():
            box = text.get_window_extent(renderer)
            if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width+1 or box.y1 > fig.bbox.height+1:
                outside.append(text.get_text())
    assert not outside, outside
    name = 'cd19_story_three_row_v18_polish'
    fig.savefig(OUT / f'{name}.png', dpi=300, facecolor='white')
    fig.savefig(OUT / f'{name}.svg', facecolor='white')
    fig.savefig(QA / 'working_size.pdf', dpi=600, facecolor='white')
    for row, limits in [('row1', (.10, H-2.72, 14.42, H-.08)),
                         ('row2', (.10, H-6.34, 14.42, H-2.74)),
                         ('row3', (.10, .08, 14.42, H-6.35))]:
        fig.savefig(OUT / f'{row}_preview.png', dpi=300, facecolor='white',
                    bbox_inches=Bbox.from_extents(*limits))
    plt.close(fig)
    # The delivery PDF is at manuscript width, with a true 0.45 pt plot frame.
    from pypdf import PdfReader, PdfWriter
    reader = PdfReader(QA / 'working_size.pdf')
    page = reader.pages[0]
    final_width_pt = 180/25.4*72
    page.scale_to(final_width_pt, final_width_pt*H/W)
    writer = PdfWriter()
    writer.add_page(page)
    writer.add_metadata({'/Title': 'CD19 story: framing and typography trial v18'})
    with (OUT / f'{name}.pdf').open('wb') as handle:
        writer.write(handle)
    comparison()
    prior = json.loads((BASE / 'figure_layout_v17/LAYOUT_AUDIT.json').read_text())
    assert all(sha(path) == digest for path, digest in prior['source_sha256'].items())
    backup_manifest = json.loads((OUT / 'backup_v17/BACKUP_SHA256.json').read_text())
    assert all(sha(path) == digest for path, digest in backup_manifest.items())
    save(OUT / 'LAYOUT_AUDIT.json', {
        'status': 'PASS', 'time_utc': now(), 'script_sha256': sha(__file__),
        'source_sha256': prior['source_sha256'], 'all_original_files_unchanged': True,
        'backup_file_count': len(backup_manifest), 'manuscript_modified': False,
        'same_panel_rectangles_as_v17': True, 'same_scientific_artist_payloads': True,
        'same_G_prediction_and_reference_pixels': True, 'same_axis_limits': True,
        'same_points_values_colors_opacities_and_ridge_curves': True,
        'no_model_or_UMAP_refit': True, 'display_bounds': [1., 1.5],
        'grey_category': '≤1.0', 'font': 'Arial', 'frame_working_linewidth_pt': FRAME_LW,
        'final_PDF_width_mm': 180, 'final_frame_linewidth_pt': FRAME_LW*(180/25.4)/W,
        'text_outside_canvas': outside, 'panels': TITLES,
    })
    (OUT / 'README_ZH.md').write_text('''# CD19 图形统一试验 v18

当前稿件使用的 v17 整图、分图、说明文件，以及 v1–v17 绘图脚本和 runtime，已在 backup_v17/ 完整备份并核对 SHA256。原版和 manuscript 均未替换。

## 视觉调整

- D/H 每张 UMAP 增加细灰黑绘图区边框；B/E 统计图和 F 每列 ridge plot 使用同一线宽。
- 字体统一为 Arial；panel 字母与标题分开，说明文字和小标题减少粗体，保留读数可辨性。
- C 五个 workflow 节点保留原有内容、位置和连接关系，改为白底和中性细边框；CODEX/References 名称保留平台色。
- H 圈选边界和引线变细，圈的位置和大小不变；混合细胞的文字标注移入图内，避开顶部边框。
- D/E/F 上缘对齐；B 高值组的 median 说明移至点下方，避开新增边框。
- 保留三行布局、完整细胞、UMAP 坐标、点层顺序、配色、透明度、KDE 曲线及全部指标。
- G 两张真实 DAPI 叠加图保持逐像素一致，色阶 1.0–1.5、独立灰色色块 ≤1.0 均保留。

## 文件

整图为 cd19_story_three_row_v18_polish.png / .svg / .pdf。PDF 宽度为 180 mm，PNG 为 4350×3150 像素。v17_vs_v18_comparison.png 为左右对比，row1/2/3_preview.png 为分行预览。

本版为待比较的独立视觉试验，不修改当前 story 根目录说明或稿件文件。LAYOUT_AUDIT.json 记录数据、备份和画布边界核验。

复现：`python3 -B Exploration0909/scripts/assemble_cd19_layout_v18_polish.py`（在 story_cd19_prediction 下运行）。
''')
    print('PASS: backup intact; identical scientific artist payloads; no clipped figure text.')
    print(OUT / f'{name}.png')


if __name__ == '__main__':
    main()
