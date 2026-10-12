"""Tighter three-row figure; accuracy axis starts at 0.4 as requested."""
from runtime import BASE, save, sha, now
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
import assemble_cd19_layout_v2 as previous
v = previous.v

OUT = BASE / 'figure_layout_v3'
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'panels').mkdir(exist_ok=True)
W, H = 14.5, 10.5
H_LIMITS = {}


def panel_b(fig, rect, letter='B'):
    x, y, w, h = rect
    v.title(fig, rect, letter, 'QC identifies failed CD19')
    left = v.ax_at(fig, x+.40, y+.60, 2.16, h-.94)
    right = v.ax_at(fig, x+3.22, y+.60, 2.32, h-.94)
    v.qc['draw_qc'](left, v.qc_scores, v.qc_summary, v.qc_order)
    v.qc['draw_specificity'](right, v.qc_specificity)
    for ax in [left, right]:
        for text in ax.texts:
            text.set_fontsize(8.3)
        ax.tick_params(axis='y', labelsize=9)
    for text in left.texts:
        text.set_text(text.get_text().replace(' ranked #1', '\nranked #1'))
        text.set_y(1.095)
        text.set_verticalalignment('top')
        text.set_linespacing(.95)
    for text, offset in zip(right.texts, [0, -.08, .08]):
        text.set_x(text.get_position()[0]+offset)


def panel_d(fig, rect, letter='D'):
    x, y, w, h = rect
    v.title(fig, rect, letter, 'Held-out CD3 validation')
    axes = [v.ax_at(fig, x, y+.55, 2.36, h-.62),
            v.ax_at(fig, x+2.42, y+.55, 2.36, h-.62)]
    t = v.cells.celltype.isin(v.cd3_audit['T_labels']).to_numpy()
    for mask, color in [(~t, '#E7C36B'), (t, '#266BAE')]:
        axes[0].scatter(*v.xy_cd3[mask].T, s=1.05, color=color, alpha=1,
                        lw=0, rasterized=True)
    handles = [Line2D([], [], marker='o', linestyle='', color=color, markersize=5)
               for color in ['#266BAE', '#E7C36B']]
    axes[0].legend(handles, ['T cells', 'Other cells'], loc='upper left',
        bbox_to_anchor=(.00, .96), frameon=False, fontsize=9.2,
        handletextpad=.35, labelspacing=.3, borderpad=0)
    values = v.cells.predicted_CD3.to_numpy()
    order = np.argsort(values, kind='stable')
    sc = axes[1].scatter(*v.xy_cd3[order].T, c=values[order], s=1.05,
                         cmap=v.cmap, norm=v.norm, alpha=1, lw=0, rasterized=True)
    lo, hi = v.xy_cd3.min(axis=0), v.xy_cd3.max(axis=0)
    pad = .012*(hi-lo)
    for ax, heading in zip(axes, ['Cell annotation', 'Predicted CD3']):
        ax.set(xlim=(lo[0]-pad[0], hi[0]+pad[0]), ylim=(lo[1]-pad[1], hi[1]+pad[1]))
        ax.set_aspect('equal'); ax.axis('off')
        ax.set_title(heading, fontsize=11.5, fontweight='bold', pad=7)
    cb = fig.colorbar(sc, cax=v.ax_at(fig, x+4.97, y+1.10, .09, 1.55), ticks=[-1, 0, 1, 2, 3])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8.5, length=2)
    cb.ax.set_title('CD3', fontsize=9.5, pad=7)


def panel_c(fig, rect, letter='C'):
    before = len(fig.axes)
    v.panel_c(fig, rect, letter)
    for ax in fig.axes[before:]:
        for text in ax.texts:
            if text.get_text() == 'Completed CODEX':
                text.set_fontsize(9.2)


def panel_e(fig, rect, letter='E'):
    # Reuse the exact donor values, box statistics, colors and jitter from v1/v2.
    before = len(fig.axes)
    v.panel_e(fig, rect, letter)
    created = fig.axes[before:]
    assert len(created) == 1
    ax = created[0]
    ax.set_ylim(.4, 1.0)
    ax.set_yticks([.4, .6, .8, 1.0])
    metric_names = ['spearman', 'pearson', 'auroc', 'average_precision']
    assert v.donor_scores[metric_names].ge(.4).all().all()
    assert v.donor_scores[metric_names].le(1.0).all().all()


def panel_h(fig, rect, letter='H'):
    x, y, w, h = rect
    v.title(fig, rect, letter, 'Predicted CD19 restores Mature B–NK separation')
    before = v.joint_metrics.loc['without_cd19', 'matureB_vs_NK_silhouette_umap']
    after = v.joint_metrics.loc['with_predicted_cd19', 'matureB_vs_NK_silhouette_umap']
    v.text_at(fig, x+w/2, y+.34, f'Mature B–NK silhouette: {before:.3f} → {after:.3f}',
              ha='center', fontsize=9.5, fontweight='bold')
    gap = .03
    aw = (w-gap)/2
    for i, (condition, heading) in enumerate(zip(v.joint['CONDITIONS'],
                                                ['14 shared markers', '+ predicted CD19'])):
        ax = v.ax_at(fig, x+i*(aw+gap), y+.80, aw, h-1.08)
        previous.draw_focused_umap(ax, condition)
        xy = v.joint_xy[condition]
        lo, hi = xy.min(axis=0), xy.max(axis=0)
        pad = (hi-lo)*np.array([.008, .02])
        xlim = (lo[0]-pad[0], hi[0]+pad[0])
        ylim = (lo[1]-pad[1], hi[1]+pad[1])
        # Preserve the complete ellipse outlines as well as every data point.
        ellipse_right = max(patch.center[0]+patch.width/2 for patch in ax.patches)
        xlim = (xlim[0], max(xlim[1], ellipse_right+.004*(hi[0]-lo[0])))
        ax.set(xlim=xlim, ylim=ylim)
        assert np.all((xy[:, 0]>=xlim[0]) & (xy[:, 0]<=xlim[1]))
        assert np.all((xy[:, 1]>=ylim[0]) & (xy[:, 1]<=ylim[1]))
        H_LIMITS[condition] = {'xlim': list(xlim), 'ylim': list(ylim)}
        ax.set_title(heading, fontsize=11, fontweight='bold', pad=7)
    handles = [Line2D([], [], marker='o', linestyle='', color=previous.COLORS[label], markersize=6)
               for label in ['Mature B', 'NK', 'Others']]
    fw, fh = fig.get_size_inches()
    fig.legend(handles, ['Mature B', 'NK', 'Others'], loc='center',
        bbox_to_anchor=((x+w/2)/fw, 1-(y+h-.10)/fh), ncol=3,
        frameon=False, fontsize=10.3, columnspacing=1.7, handletextpad=.4,
        borderpad=0, borderaxespad=0)


def export(fig, name):
    for ext in ['png', 'svg']:
        fig.savefig(OUT / f'{name}.{ext}', dpi=300, facecolor='white')


def main():
    fig = plt.figure(figsize=(W, H), facecolor='white')
    panels = [
        (v.panel_a, (.15, .13, 2.50, 2.55)),
        (panel_b, (2.76, .13, 5.70, 2.55)),
        (panel_c, (8.57, .13, 5.78, 2.55)),
        (panel_d, (.15, 2.79, 5.45, 3.50)),
        (panel_e, (5.695, 2.79, 4.30, 3.50)),
        (v.panel_f, (10.09, 2.79, 4.26, 3.50)),
        (v.panel_g, (.15, 6.40, 6.72, 3.95)),
        (panel_h, (7.00, 6.40, 7.35, 3.95)),
    ]
    for draw, rect in panels:
        draw(fig, rect)
    export(fig, 'cd19_story_three_row_v3')
    fig.savefig(OUT / 'second_row_preview.png', dpi=300, facecolor='white',
                bbox_inches=Bbox.from_extents(.10, H-6.34, 14.42, H-2.74))
    plt.close(fig)
    for draw, name, size in [
        (panel_d, 'D_CD3_validation_pair', (5.55, 3.60)),
        (panel_e, 'E_donor_mean_accuracy_y04', (4.40, 3.60)),
        (v.panel_f, 'F_square_CD19_ridges', (4.36, 3.60)),
        (panel_h, 'H_Mature_B_NK_UMAP', (7.45, 4.05)),
    ]:
        fig = plt.figure(figsize=size, facecolor='white')
        draw(fig, (.05, .05, size[0]-.1, size[1]-.1))
        export(fig, f'panels/{name}')
        plt.close(fig)
    assert all(sha(path) == value for path, value in v.input_hashes.items())
    save(OUT / 'LAYOUT_AUDIT.json', {'time_utc': now(),
        'source_sha256': v.input_hashes, 'script_sha256': sha(__file__),
        'base_script_sha256': {str(BASE / f'scripts/assemble_cd19_layout_v{i}.py'):
                               sha(BASE / f'scripts/assemble_cd19_layout_v{i}.py') for i in [1, 2]},
        'canvas_inches': [W, H], 'previous_canvas_inches': [16.2, 10.75],
        'width_reduction_from_v2_fraction': 1-W/16.2,
        'panel_rectangles_inches_from_top_left': {draw.__name__: list(rect) for draw, rect in panels},
        'gaps_inches': {'B_C': .11, 'D_E': .095, 'E_F': .095, 'H_UMAP_axes': .03},
        'D': {'axis_center_spacing_inches': 2.42, 'previous_axis_center_spacing_inches': 2.75,
              'relative_margin_each_end': .012, 'previous_relative_margin_each_end': .035,
              'n_cells': len(v.cells), 'all_cells_retained': True,
              'coordinates_changed': False, 'prediction_changed': False},
        'E': {'user_requested_y_limits': [.4, 1.0], 'n_donors': 12,
              'all_48_donor_metric_points_visible': True,
              'aggregation': 'arithmetic mean across 14 markers, unchanged'},
        'H': {'n_cells_in_each_map': 88031, 'coordinate_limits': H_LIMITS,
              'relative_margin_each_end_xy': [.008, .02],
              'limits_extended_where_needed_for_complete_ellipse_outlines': True,
              'all_cells_within_limits': True, 'coordinates_changed': False,
              'display_categories': ['Mature B', 'NK', 'Others'],
              'legend': 'bottom, three columns',
              'display_counts': {str(k): int(val) for k, val in previous.counts.items()}},
        'all_input_files_unchanged': True, 'no_model_or_UMAP_refit': True})
    (OUT / 'README_ZH.md').write_text('''# CD19 story 进一步收紧版 v3

主图：cd19_story_three_row_v3.png；矢量版为同名 SVG。第二行单独预览见 second_row_preview.png，D/E/F/H 分图位于 panels。

按本次要求，E 的纵轴显示范围改为 0.4–1.0，所有 48 个 donor × metric 点仍在范围内。计算数值、箱线图统计和点的定义沿用前一版；每列 12 位 donor，每点为该 donor 的 14 个 marker 分数均值。

画布宽度从 v2 的 16.2 英寸缩至 14.5 英寸，高度 10.5 英寸。B–C 间距从 0.30 英寸缩至 0.11 英寸，D–E 与 E–F 从 0.275 英寸缩至 0.095 英寸。D 两张 UMAP 的中心间距从 2.75 英寸缩至 2.42 英寸；坐标范围外的边距由每端 3.5% 减至 1.2%。

H 内两张 UMAP 的绘图区间距从 0.18 英寸缩至 0.03 英寸，横向坐标留白减至每端数据跨度的 0.8%，纵向为 2%。范围由全部细胞坐标的最小、最大值确定；所有 88,031 个细胞都保留且位于显示范围内。仍采用 Mature B、NK、Others 三类，legend 在底部。

沿用冻结模型输出和 UMAP 坐标，仅调整图面布局及明确指定的纵轴范围。所有新结果在 figure_layout_v3 内，v1/v2 保留。复现脚本为 Exploration0909/scripts/assemble_cd19_layout_v3.py；输入哈希、点数和坐标边界核对见 LAYOUT_AUDIT.json。
''')
    print(OUT / 'cd19_story_three_row_v3.png', flush=True)


if __name__ == '__main__':
    main()
