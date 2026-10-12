"""Extend only Panel D downward within its existing figure slot."""
from pathlib import Path
import hashlib
import json
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/hnscc_CEF_compact')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import fitz

HERE = Path(__file__).resolve().parent
TRIAL = HERE.parent
OLD_LAYOUT = TRIAL/'CEF_compact/LAYOUT_AUDIT.json'
VALUES = TRIAL/'horizontal_E_spatial_left/panel_D/plotted_donor_values.csv'
INK, GRID, ORANGE, TEAL = '#182027', '#E7EBEE', '#D95F02', '#238A8D'
DONORS = ['126', 'indiv_1', 'indiv_2']
FEATURES = [('TIGIT--VSTM3', 'TIGIT'), ('CD39', 'CD39'), ('CD224', 'CD224'),
            ('CD134--OX40', 'OX40'), ('TCR-A--TCR-B', 'TCRαβ'), ('CD83-1', 'CD83')]


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def build_d():
    d = pd.read_csv(VALUES, dtype={'donor': str})
    assert len(d) == 18
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none', 'text.color': INK})
    W, H = 4.65, 3.45
    fig = plt.figure(figsize=(W, H), facecolor='white')
    fig.text(.08/W, .976, 'D', fontsize=23, weight='bold', va='top')
    fig.text(.51/W, .947, 'BIOMARKER DISCOVERY', fontsize=12, weight='bold', va='top')
    fig.text(.54, .810, '107 proteins screened | 3 donors', ha='center', fontsize=9.5)
    # Preserve the upper edge (0.730); extend the lower edge down by 0.070 H.
    ax = fig.add_axes([.154, .155, .811, .575])
    for x, (marker, name) in enumerate(FEATURES):
        b = d[d.marker_raw.eq(marker)].set_index('donor').loc[DONORS]
        v = b.association.to_numpy(float)
        color = ORANGE if x < 2 else TEAL
        ax.boxplot([v], positions=[x], widths=.47, manage_ticks=False,
                   patch_artist=True, showfliers=False, whis=(0, 100),
                   boxprops=dict(facecolor=color, edgecolor=color, alpha=.18, linewidth=1),
                   medianprops=dict(color=color, linewidth=1.4),
                   whiskerprops=dict(color=color, linewidth=1), capprops=dict(color=color, linewidth=1))
        pts = ax.scatter(x+np.array([-.10, 0, .10]), v, s=26, color=color,
                         edgecolor='white', linewidth=.5, zorder=4)
        np.testing.assert_array_equal(pts.get_offsets()[:, 1], v)
    ax.set(xlim=(-.6, 5.6), ylim=(0, .36), yticks=[0, .1, .2, .3])
    ax.set_xticks(range(6), [f[1] for f in FEATURES])
    for text, color in zip(ax.get_xticklabels(), [ORANGE]*2+[TEAL]*4):
        text.set_color(color)
        text.set_weight('bold')
    ax.set_ylabel('Association with niche score\n(Spearman ρ)', fontsize=9.3, labelpad=5)
    ax.tick_params(axis='x', labelsize=9, length=0, pad=7)
    ax.tick_params(axis='y', labelsize=8.5, length=3)
    ax.yaxis.grid(True, color=GRID, lw=.6)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    for side in ['left', 'bottom']:
        ax.spines[side].set_color('#7A838A')
    fig.legend(handles=[Line2D([], [], marker='o', linestyle='', color=c, markersize=4.2, label=l)
                        for c, l in [(ORANGE, 'Selected pair'), (TEAL, 'Other candidates')]],
               loc='center', bbox_to_anchor=(.54, .045), ncol=2, frameon=False, fontsize=8.6,
               handlelength=.6, handletextpad=.4, columnspacing=1.2)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = list(fig.texts)+[ax.yaxis.label]+ax.get_xticklabels()+ax.get_yticklabels()
    for legend in fig.legends:
        texts += legend.get_texts()
    for text in texts:
        box = text.get_window_extent(renderer)
        assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fig.bbox.width+1 and box.y1 <= fig.bbox.height+1, text.get_text()
    folder = HERE/'panel_D'
    folder.mkdir(exist_ok=True)
    stem = folder/'panel_D_taller'
    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(stem.with_suffix('.'+ext), dpi=320, facecolor='white')
    plt.close(fig)
    d.to_csv(folder/'plotted_donor_values.csv', index=False)
    with fitz.open(stem.with_suffix('.pdf')) as pdf:
        pdf[0].get_pixmap(dpi=230, alpha=False).save(folder/'pdf_preview.png')
    return stem.with_suffix('.pdf')


def main():
    old = json.loads(OLD_LAYOUT.read_text())
    paths = [OLD_LAYOUT, VALUES]+[Path(p['source']) for p in old['placements'].values()]
    before = {str(p): sha(p) for p in paths}
    replacement = build_d()
    doc = fitz.open()
    page = doc.new_page(width=16.99*72, height=6.98*72)
    placements = {}
    for letter, slot in old['placements'].items():
        source = replacement if letter == 'D' else Path(slot['source'])
        rect = fitz.Rect(*[v*72 for v in slot['bounds_inches']])
        with fitz.open(source) as panel:
            np.testing.assert_allclose([rect.width, rect.height], [panel[0].rect.width, panel[0].rect.height], atol=.01)
            page.show_pdf_page(rect, panel, 0)
        placements[letter] = dict(source=str(source), sha256=sha(source), bounds_inches=slot['bounds_inches'])
    doc.set_metadata({'title': 'HNSCC: C E F arrangement with taller biomarker discovery panel'})
    stem = HERE/'hnscc_CEF_D_taller'
    doc.save(stem.with_suffix('.pdf'), garbage=4, deflate=True)
    doc.close()
    with fitz.open(stem.with_suffix('.pdf')) as pdf:
        assert len(pdf) == 1
        pdf[0].get_pixmap(dpi=180, alpha=False).save(stem.with_suffix('.png'))
    assert before == {p: sha(Path(p)) for p in before}
    (HERE/'LAYOUT_AUDIT.json').write_text(json.dumps(dict(
        status='PASS', layout='A B D / C E F', placements=placements,
        source_layout=str(OLD_LAYOUT), size_inches=[16.99, 6.98],
        panel_D=dict(old_axes=[.154, .225, .811, .505], new_axes=[.154, .155, .811, .575],
                     height_increase_percent=100*(.575/.505-1),
                     lower_edge_moved_down_inches=.070*3.45,
                     upper_edge_unchanged=True, legend_y_old=.095, legend_y_new=.045,
                     donor_values_unchanged=True, y_range_unchanged=[0, .36]),
        other_panels_reused_without_changes=True, panel_F=old['panel_F'], inputs_unchanged=before,
        formal_figure_replaced=False, manuscript_replaced=False), indent=2)+'\n')
    (HERE/'README_ZH.md').write_text('''# HNSCC：Panel D 向下延长

整图：`hnscc_CEF_D_taller.png/.pdf`。独立 D 图和绘制数值位于 `panel_D/`。

仅将 Panel D 的绘图区向下延长 0.2415 英寸（约 6.1 mm），高度增加约 14%，上边缘和标题保持原位。底部 legend 相应下移。六个 marker 的 18 个 donor 值、字体、颜色、箱线图统计和纵轴范围 0–0.36 不变。

A、B、C、E、F 直接复用上一个版本 PDF，位置和大小不变。下排仍是 C → E → F；F 保留 343 个 CD8 的空间双图。整图仍为 16.99 × 6.98 英寸。旧版本和 manuscript 保留。
''')
    print('PASS: only Panel D extended downward; all 18 values and other panels unchanged.')


if __name__ == '__main__':
    main()
