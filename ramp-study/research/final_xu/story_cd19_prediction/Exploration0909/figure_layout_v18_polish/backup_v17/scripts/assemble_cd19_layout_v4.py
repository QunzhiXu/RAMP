"""Keep the v3 layout; use green spatial background and a 0.4–0.9 score axis."""
from runtime import BASE, save, sha, now
import json
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import assemble_cd19_layout_v3 as previous
v = previous.v

OUT = BASE / 'figure_layout_v4'
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'panels').mkdir(exist_ok=True)
BACKGROUND = '#63BE7B'


def panel_e(fig, rect, letter='E'):
    before = len(fig.axes)
    previous.panel_e(fig, rect, letter)
    ax, = fig.axes[before:]
    ax.set_ylim(.4, .9)
    ax.set_yticks([.4, .5, .6, .7, .8, .9])
    values = v.donor_scores[['spearman', 'pearson', 'auroc', 'average_precision']]
    assert values.ge(.4).all().all() and values.le(.9).all().all()


def panel_g(fig, rect, letter='G'):
    before = len(fig.axes)
    v.panel_g(fig, rect, letter)
    axes = fig.axes[before:]
    assert len(axes) == 2
    for ax in axes:
        assert len(ax.collections) == 2
        ax.collections[0].set_color(BACKGROUND)
        assert ax.collections[0].get_alpha() == .62
        assert sum(len(sc.get_offsets()) for sc in ax.collections) == len(v.spatial_md)


def export(fig, name):
    for ext in ['png', 'svg']:
        fig.savefig(OUT / f'{name}.{ext}', dpi=300, facecolor='white')


def main():
    fig = plt.figure(figsize=(previous.W, previous.H), facecolor='white')
    panels = [
        (v.panel_a, (.15, .13, 2.50, 2.55)),
        (previous.panel_b, (2.76, .13, 5.70, 2.55)),
        (previous.panel_c, (8.57, .13, 5.78, 2.55)),
        (previous.panel_d, (.15, 2.79, 5.45, 3.50)),
        (panel_e, (5.695, 2.79, 4.30, 3.50)),
        (v.panel_f, (10.09, 2.79, 4.26, 3.50)),
        (panel_g, (.15, 6.40, 6.72, 3.95)),
        (previous.panel_h, (7.00, 6.40, 7.35, 3.95)),
    ]
    prior_path = BASE / 'figure_layout_v3/LAYOUT_AUDIT.json'
    prior = json.loads(prior_path.read_text())
    rectangles = {draw.__name__: list(rect) for draw, rect in panels}
    assert rectangles == prior['panel_rectangles_inches_from_top_left']
    for draw, rect in panels:
        draw(fig, rect)
    export(fig, 'cd19_story_three_row_v4')
    fig.savefig(OUT / 'second_row_preview.png', dpi=300, facecolor='white',
        bbox_inches=Bbox.from_extents(.10, previous.H-6.34, 14.42, previous.H-2.74))
    plt.close(fig)
    for draw, name, size in [
        (panel_e, 'E_donor_mean_accuracy_y04_y09', (4.40, 3.60)),
        (panel_g, 'G_spatial_green_background', (6.82, 4.05)),
    ]:
        fig = plt.figure(figsize=size, facecolor='white')
        draw(fig, (.05, .05, size[0]-.1, size[1]-.1))
        export(fig, f'panels/{name}')
        plt.close(fig)
    assert all(sha(path) == value for path, value in v.input_hashes.items())
    save(OUT / 'LAYOUT_AUDIT.json', {'time_utc': now(), 'script_sha256': sha(__file__),
        'source_sha256': v.input_hashes, 'base_layout_audit_sha256': sha(prior_path),
        'base_script_sha256': {str(BASE / f'scripts/assemble_cd19_layout_v{i}.py'):
                               sha(BASE / f'scripts/assemble_cd19_layout_v{i}.py') for i in [1, 2, 3]},
        'canvas_inches': [previous.W, previous.H],
        'panel_rectangles_inches_from_top_left': rectangles,
        'layout_matches_v3': True,
        'E': {'y_limits': [.4, .9], 'ticks': [.4, .5, .6, .7, .8, .9],
              'all_48_points_visible': True, 'donor_metrics_changed': False},
        'G': {'background_cell_color': BACKGROUND, 'background_alpha': .62,
              'highlight_colors': {'predicted_high': '#E04455', 'annotated_B': '#8E44AD'},
              'same_cells_coordinates_donor_threshold': True},
        'all_input_files_unchanged': True})
    (OUT / 'README_ZH.md').write_text('''# CD19 story v4

主图：cd19_story_three_row_v4.png；可编辑矢量版为同名 SVG。

沿用 v3 的 14.5 × 10.5 英寸画布与全部 panel 位置，只调整两处：

- Panel G 两张空间图的背景细胞改为绿色 #63BE7B，透明度保持 0.62。预测高值细胞和作者注释 B 细胞继续使用红色与紫色。
- Panel E 纵轴改为 0.4–0.9，刻度间隔 0.1；所有 48 个 donor × metric 点均在显示范围内。

E、G 新分图位于 panels，第二行单独预览见 second_row_preview.png。既有预测、指标、空间细胞与坐标、UMAP 以及其他图形设置沿用 v3。复现脚本为 Exploration0909/scripts/assemble_cd19_layout_v4.py，输入哈希与布局核对记录于 LAYOUT_AUDIT.json。
''')
    print(OUT / 'cd19_story_three_row_v4.png', flush=True)


if __name__ == '__main__':
    main()
