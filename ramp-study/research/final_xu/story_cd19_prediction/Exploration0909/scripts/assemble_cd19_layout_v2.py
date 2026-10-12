"""Compact layout with an enlarged, three-category Mature B/NK UMAP panel."""
from runtime import BASE, save, sha, now
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
import assemble_cd19_layout_v1 as v

OUT = BASE / 'figure_layout_v2'
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'panels').mkdir(exist_ok=True)
W, H = 16.2, 10.75
COLORS = {'Mature B': '#D81B60', 'NK': '#00A087', 'Others': '#C6CCD1'}
original_labels = v.joint_md.celltype.astype(str).to_numpy()
display_labels = np.where(original_labels == 'Mature / CODEX B', 'Mature B',
                         np.where(original_labels == 'NK', 'NK', 'Others'))
assert len(display_labels) == 88031
assert np.sum(display_labels == 'Mature B') == np.sum(original_labels == 'Mature / CODEX B')
assert np.sum(display_labels == 'NK') == np.sum(original_labels == 'NK')
counts = pd.Series(display_labels).value_counts().reindex(['Mature B', 'NK', 'Others'])


def panel_b(fig, rect, letter='B'):
    x, y, w, h = rect
    v.title(fig, rect, letter, 'QC identifies failed CD19')
    left = v.ax_at(fig, x+.43, y+.62, 2.43, h-.96)
    right = v.ax_at(fig, x+3.62, y+.62, 2.48, h-.96)
    v.qc['draw_qc'](left, v.qc_scores, v.qc_summary, v.qc_order)
    v.qc['draw_specificity'](right, v.qc_specificity)
    for ax in [left, right]:
        for text in ax.texts:
            text.set_fontsize(8.5)
        ax.tick_params(axis='y', labelsize=9)
    for text in left.texts:
        text.set_text(text.get_text().replace(' ranked #1', '\nranked #1'))
        text.set_y(1.095)
        text.set_verticalalignment('top')
        text.set_linespacing(.95)
    # Give the neighboring CyTOF/AbSeq median labels a little more separation.
    for text, offset in zip(right.texts, [0, -.08, .08]):
        text.set_x(text.get_position()[0] + offset)


def panel_d(fig, rect, letter='D'):
    x, y, w, h = rect
    v.title(fig, rect, letter, 'Held-out CD3 validation')
    axes = [v.ax_at(fig, x+.02, y+.55, 2.52, h-.62),
            v.ax_at(fig, x+2.77, y+.55, 2.52, h-.62)]
    t = v.cells.celltype.isin(v.cd3_audit['T_labels']).to_numpy()
    for mask, color in [(~t, '#E7C36B'), (t, '#266BAE')]:
        axes[0].scatter(*v.xy_cd3[mask].T, s=1.05, color=color, alpha=1, lw=0, rasterized=True)
    handles = [Line2D([], [], marker='o', linestyle='', color=color, markersize=5)
               for color in ['#266BAE', '#E7C36B']]
    axes[0].legend(handles, ['T cells', 'Other cells'], loc='upper left',
        bbox_to_anchor=(.00, .95), frameon=False, fontsize=9.2,
        handletextpad=.35, labelspacing=.3, borderpad=0)
    values = v.cells.predicted_CD3.to_numpy()
    order = np.argsort(values, kind='stable')
    sc = axes[1].scatter(*v.xy_cd3[order].T, c=values[order], s=1.05, cmap=v.cmap,
                         norm=v.norm, alpha=1, lw=0, rasterized=True)
    for ax, heading in zip(axes, ['Cell annotation', 'Predicted CD3']):
        v.format_cd3(ax)
        ax.set_title(heading, fontsize=11.5, fontweight='bold', pad=7)
    cb = fig.colorbar(sc, cax=v.ax_at(fig, x+5.55, y+1.10, .10, 1.55), ticks=[-1, 0, 1, 2, 3])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8.5, length=2)
    cb.ax.set_title('CD3', fontsize=9.5, pad=7)


def draw_focused_umap(ax, condition):
    xy = v.joint_xy[condition]
    # All cells remain present. Only the display grouping and layer order change.
    drawn = 0
    for label in ['Others', 'Mature B', 'NK']:
        mask = display_labels == label
        drawn += int(mask.sum())
        ax.scatter(*xy[mask].T, s=.55, color=COLORS[label],
                   alpha=.55 if label == 'Others' else .78,
                   linewidths=0, rasterized=True)
    assert drawn == 88031
    ax.set_xticks([]); ax.set_yticks([])
    # Match the original H panel's automatic aspect, using larger plot rectangles.
    ax.set_aspect('auto')
    for spine in ax.spines.values():
        spine.set_visible(False)
    v.joint['highlight_separation'](ax, condition)


def panel_h(fig, rect, letter='H'):
    x, y, w, h = rect
    v.title(fig, rect, letter, 'Predicted CD19 restores Mature B–NK separation')
    before = v.joint_metrics.loc['without_cd19', 'matureB_vs_NK_silhouette_umap']
    after = v.joint_metrics.loc['with_predicted_cd19', 'matureB_vs_NK_silhouette_umap']
    v.text_at(fig, x+w/2, y+.34, f'Mature B–NK silhouette: {before:.3f} → {after:.3f}',
              ha='center', fontsize=9.5, fontweight='bold')
    gap = .18
    aw = (w-gap)/2
    for i, (condition, heading) in enumerate(zip(v.joint['CONDITIONS'],
                                                ['14 shared markers', '+ predicted CD19'])):
        ax = v.ax_at(fig, x+i*(aw+gap), y+.80, aw, h-1.08)
        draw_focused_umap(ax, condition)
        ax.set_title(heading, fontsize=11, fontweight='bold', pad=7)
    handles = [Line2D([], [], marker='o', linestyle='', color=COLORS[label], markersize=6)
               for label in ['Mature B', 'NK', 'Others']]
    fw, fh = fig.get_size_inches()
    fig.legend(handles, ['Mature B', 'NK', 'Others'], loc='center',
        bbox_to_anchor=((x+w/2)/fw, 1-(y+h-.10)/fh), ncol=3,
        frameon=False, fontsize=10.3, columnspacing=1.7, handletextpad=.4,
        borderpad=0, borderaxespad=0)


def export(fig, name, dpi=300):
    for ext in ['png', 'svg']:
        fig.savefig(OUT / f'{name}.{ext}', dpi=dpi, facecolor='white')


def main():
    fig = plt.figure(figsize=(W, H), facecolor='white')
    panels = [
        (v.panel_a, (.20, .16, 2.70, 2.60)),
        (panel_b, (3.15, .16, 6.30, 2.60)),
        (v.panel_c, (9.75, .16, 6.25, 2.60)),
        (panel_d, (.20, 2.94, 6.20, 3.50)),
        (v.panel_e, (6.675, 2.94, 4.65, 3.50)),
        (v.panel_f, (11.60, 2.94, 4.40, 3.50)),
        (v.panel_g, (.20, 6.60, 7.45, 4.00)),
        (panel_h, (7.90, 6.60, 8.10, 4.00)),
    ]
    for draw, rect in panels:
        draw(fig, rect)
    export(fig, 'cd19_story_three_row_v2')
    fig.savefig(OUT / 'second_row_preview.png', dpi=300, facecolor='white',
                bbox_inches=Bbox.from_extents(.15, H-6.48, 16.07, H-2.88))
    plt.close(fig)
    for draw, name, size in [
        (panel_d, 'D_CD3_validation_pair', (6.30, 3.60)),
        (v.panel_e, 'E_donor_mean_accuracy', (4.75, 3.60)),
        (v.panel_f, 'F_square_CD19_ridges', (4.50, 3.60)),
        (panel_h, 'H_Mature_B_NK_UMAP', (8.20, 4.10)),
    ]:
        fig = plt.figure(figsize=size, facecolor='white')
        draw(fig, (.05, .05, size[0]-.1, size[1]-.1))
        export(fig, f'panels/{name}')
        plt.close(fig)
    mapping = v.joint_md.copy()
    mapping['figure_display_group'] = display_labels
    mapping.to_csv(OUT / 'H_display_metadata.csv.gz', index=False)
    mapping.groupby(['dataset', 'figure_display_group']).size().rename('n_cells').to_csv(
        OUT / 'H_group_counts_by_dataset.csv')
    assert all(sha(path) == value for path, value in v.input_hashes.items())
    save(OUT / 'LAYOUT_AUDIT.json', {'time_utc': now(),
        'source_sha256': v.input_hashes, 'script_sha256': sha(__file__),
        'base_script_sha256': sha(BASE / 'scripts/assemble_cd19_layout_v1.py'),
        'canvas_inches': [W, H], 'previous_canvas_inches': [18, 11.4],
        'canvas_area_reduction_fraction': 1-W*H/(18*11.4),
        'panel_rectangles_inches_from_top_left': {draw.__name__: list(rect) for draw, rect in panels},
        'row_gap_inches': [.18, .16],
        'D': {'n_cells': len(v.cells), 'all_cells_retained': True,
              'coordinates_changed': False, 'prediction_changed': False,
              'previous_axes_center_spacing_inches': 3.58, 'axes_center_spacing_inches': 2.75,
              'color_limits': [v.norm.vmin, v.norm.vmax]},
        'E': {'y_limits': [-.025, 1.025], 'n_donors': 12,
              'aggregation': 'arithmetic mean across all 14 markers, unchanged'},
        'H': {'n_cells': len(display_labels), 'all_cells_retained_in_both_maps': True,
              'display_counts': {str(k): int(val) for k, val in counts.items()},
              'mapping': {'Mature B': 'original Mature / CODEX B', 'NK': 'original NK',
                          'Others': 'all remaining original labels'},
              'original_labels_modified': False, 'coordinates_changed': False,
              'colors': COLORS, 'layer_order': ['Others', 'Mature B', 'NK'],
              'legend_position': 'bottom of panel, 3 columns',
              'previous_single_map_width_inches': 3.43, 'single_map_width_inches': 3.96,
              'marker_size_pt2': .55, 'alpha': {'Others': .55, 'Mature B': .78, 'NK': .78}},
        'all_input_files_unchanged': True,
        'no_model_or_UMAP_refit': True})
    (OUT / 'README_ZH.md').write_text('''# CD19 story 紧凑版 v2

主图：`cd19_story_three_row_v2.png`，矢量版为同名 SVG。`second_row_preview.png` 单独展示第二行。新分图位于 panels。

画布由 18 × 11.4 英寸收紧到 16.2 × 10.75 英寸。第二行两张 CD3 UMAP 的中心间距由 3.58 英寸缩短到 2.75 英寸，保持相同坐标、色阶和近似点云高度；准确性图略微缩小，纵轴仍保留 0–1 刻度，继续使用每位 donor 跨 14 个 marker 的平均指标。

最后的 H panel 采用三类上色：Mature B、NK、Others。Mature B 对应原始展示标签 Mature / CODEX B（11,575 个细胞）；NK 对应原始 NK（4,403 个）；其余 72,053 个细胞全部显示为灰色 Others。每张图仍包括全部 88,031 个细胞，原始 celltype 和坐标不变。映射后的展示标签另存于 H_display_metadata.csv.gz，未写回原始数据。

H panel 的 legend 移至底部，只有三个条目，两张 UMAP 的单图宽度由 3.43 英寸增至 3.96 英寸。绘制顺序为 Others → Mature B → NK，后两类沿用原来的粉红与绿色。原始三平台 UMAP 和分离度数值均沿用既定结果；本次不重新拟合。

复现脚本：Exploration0909/scripts/assemble_cd19_layout_v2.py，复用 v1 脚本读取的冻结数据和部分绘图函数。所有新输出仅位于 figure_layout_v2，v1 预览保留。详见 LAYOUT_AUDIT.json。
''')
    print(counts.to_string(), flush=True)
    print(OUT / 'cd19_story_three_row_v2.png', flush=True)


if __name__ == '__main__':
    main()
