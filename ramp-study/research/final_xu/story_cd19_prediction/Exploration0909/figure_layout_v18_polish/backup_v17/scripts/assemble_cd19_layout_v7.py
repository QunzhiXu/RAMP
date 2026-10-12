"""Tighten G's title spacing and enlarge its maps; simplify F's axis label."""
from runtime import BASE, save, sha, now
import json
import numpy as np
import matplotlib.pyplot as plt
import assemble_cd19_layout_v6 as v6

v5, v4, v3, v = v6.v5, v6.v4, v6.v3, v6.v
OUT = BASE / 'figure_layout_v7'
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'panels').mkdir(exist_ok=True)
G_GAP = .08
G_PADDING = .004
G_TOP = .53


def panel_f(fig, rect, letter='F'):
    before = len(fig.texts)
    v5.panel_f(fig, rect, letter)
    labels = [text for text in fig.texts[before:] if text.get_text() == 'CD19 (KMEANSLOW)']
    assert len(labels) == 1
    labels[0].set_text('CD19')


def panel_g(fig, rect, letter='G'):
    x, y, w, h = rect
    v.title(fig, rect, letter, 'Predicted CD19 recovers spatial B-cell locations')
    provider_b = v.spatial_md.cluster_anno_l1.eq('B-Cells').to_numpy()
    predicted_high = v.spatial_prediction > v.spatial['THRESHOLD']
    width = (w-G_GAP)/2
    pad = G_PADDING*np.ptp(v.spatial_xy, axis=0)
    lims = np.array([v.spatial_xy.min(axis=0)-pad, v.spatial_xy.max(axis=0)+pad])
    for i, (mask, color, heading) in enumerate([
        (predicted_high, v.spatial['PREDICTED_COLOR'],
         f'Predicted CD19 > 1.4 (n = {predicted_high.sum()})'),
        (provider_b, v.spatial['B_COLOR'], f'Annotated B cells (n = {provider_b.sum()})')]):
        ax = v.ax_at(fig, x+i*(width+G_GAP), y+G_TOP, width, h-G_TOP-.04)
        ax.scatter(*v.spatial_xy[~mask].T, s=5, color='#63BE7B', alpha=.62, lw=0,
                   rasterized=True, clip_on=False)
        ax.scatter(*v.spatial_xy[mask].T, s=10.5, color=color, alpha=.92, lw=0,
                   rasterized=True, clip_on=False)
        v.spatial['format_spatial_axis'](ax, tuple(lims[:, 0]), tuple(lims[:, 1]))
        # Equal spatial scale is preserved; anchor at the top so aspect-ratio
        # adjustment cannot insert empty space above the subplot headings.
        ax.set_anchor('N')
        ax.set_title(heading, fontsize=10.8, fontweight='bold', pad=6)
        assert sum(len(collection.get_offsets()) for collection in ax.collections) == len(v.spatial_xy)


def export(fig, name):
    for ext in ['png', 'svg']:
        fig.savefig(OUT / f'{name}.{ext}', dpi=300, facecolor='white')


def main():
    fig = plt.figure(figsize=(v3.W, v3.H), facecolor='white')
    panels = [
        (v.panel_a, (.15, .13, 2.50, 2.55)),
        (v6.panel_b, (2.76, .13, 5.70, 2.55)),
        (v3.panel_c, (8.57, .13, 5.78, 2.55)),
        (v3.panel_d, (.15, 2.79, 5.45, 3.50)),
        (v4.panel_e, (5.695, 2.79, 4.30, 3.50)),
        (panel_f, (10.09, 2.79, 4.26, 3.50)),
        (panel_g, (.15, 6.40, 6.72, 3.95)),
        (v3.panel_h, (7.00, 6.40, 7.35, 3.95)),
    ]
    prior_path = BASE / 'figure_layout_v6/LAYOUT_AUDIT.json'
    prior = json.loads(prior_path.read_text())
    rectangles = {draw.__name__: list(rect) for draw, rect in panels}
    assert rectangles == prior['panel_rectangles_inches_from_top_left']
    for draw, rect in panels:
        draw(fig, rect)
    export(fig, 'cd19_story_three_row_v7')
    plt.close(fig)
    for draw, name, size, rect in [
        (panel_g, 'G_compact_spatial', (6.82, 3.40), (.05, .05, 6.72, 3.30)),
        (panel_f, 'F_CD19', (4.36, 3.60), (.05, .05, 4.26, 3.50)),
    ]:
        fig = plt.figure(figsize=size, facecolor='white')
        draw(fig, rect)
        export(fig, f'panels/{name}')
        plt.close(fig)
    assert sha(v6.SCORE_SOURCE) == v6.score_hash
    assert all(sha(path) == value for path, value in v.input_hashes.items())
    save(OUT / 'LAYOUT_AUDIT.json', {
        'time_utc': now(), 'script_sha256': sha(__file__),
        'source_sha256': {**v.input_hashes, str(v6.SCORE_SOURCE): v6.score_hash},
        'base_layout_audit_sha256': sha(prior_path),
        'base_script_sha256': {str(BASE / f'scripts/assemble_cd19_layout_v{i}.py'):
                               sha(BASE / f'scripts/assemble_cd19_layout_v{i}.py') for i in range(1, 7)},
        'canvas_inches': [v3.W, v3.H], 'panel_rectangles_inches_from_top_left': rectangles,
        'F': {'x_axis_label': 'CD19', 'KDE_factor': v5.BANDWIDTH},
        'G': {'axes_anchor': 'N', 'axes_top_inches': G_TOP,
              'inter_axes_gap_inches': G_GAP, 'coordinate_padding_fraction': G_PADDING,
              'relative_map_scale_vs_v6': ((6.72-G_GAP)/1.008)/((6.72-.18)/1.03),
              'n_cells_each_map': len(v.spatial_xy),
              'coordinate_aspect': 'equal', 'background_color': '#63BE7B',
              'donor': v.spatial['DONOR'], 'prediction_threshold': v.spatial['THRESHOLD']},
        'all_input_files_unchanged': True,
    })
    (OUT / 'README_ZH.md').write_text('''# CD19 story v7

主图为 cd19_story_three_row_v7.png；同名 SVG 为矢量版。G、F 单图位于 panels。

G 的两张空间图改为顶部对齐，压紧主标题与小标题的间距；缩小图间距和坐标边距，使图形线性尺寸增加约 4%。保留相同真实坐标、等比例坐标轴、全部 2,500 个细胞、阈值 1.4 和绿色背景细胞。两图分别突出 295 个预测高值细胞和 271 个作者注释 B 细胞。

F 横轴文字简化为 CD19；继续使用 0.35 KDE 带宽。其他 panels 与整图尺寸沿用 v6，包括 B 的 CD3 低异常对照。

复现脚本：Exploration0909/scripts/assemble_cd19_layout_v7.py。所有新文件均位于 Exploration0909；原始数据与旧版图不变。
''')
    print(OUT / 'cd19_story_three_row_v7.png', flush=True)


if __name__ == '__main__':
    main()
