"""Move E/F annotations inside the plots and expand their plotting areas."""
from runtime import BASE, save, sha, now
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from pypdf import PdfReader, PdfWriter
import assemble_cd19_layout_v18_polish as previous

# Preserve a small gap between the newly lowered tick labels and the third row.
ROW3_SHIFT = .08
previous.H += ROW3_SHIFT
previous.PANELS = [(letter, draw, (x, y + (ROW3_SHIFT if letter in ['G', 'H'] else 0), w, h))
                   for letter, draw, (x, y, w, h) in previous.PANELS]

ROW1_REDUCTION = .40
previous.H -= ROW1_REDUCTION
previous.PANELS = [(letter, draw, (x, y - (ROW1_REDUCTION if letter in ['D', 'E', 'F', 'G', 'H'] else 0), w, h))
                   for letter, draw, (x, y, w, h) in previous.PANELS]

OLDER = BASE / 'figure_layout_v22_polish'
OUT = BASE / 'figure_layout_v23_polish'
QA = OUT / 'qa'
BACKUP = OUT / 'backup_v22'
for directory in [OUT, QA, BACKUP]:
    directory.mkdir(parents=True, exist_ok=True)


def main():
    before_files = {}
    for path in [*OLDER.glob('*.*'), BASE / 'scripts/assemble_cd19_layout_v22_polish.py']:
        if not path.is_file():
            continue
        before_files[str(path)] = sha(path)
        dest = BACKUP / path.name
        if dest.exists():
            assert sha(dest) == sha(path)
        else:
            shutil.copy2(path, dest)
    save(BACKUP / 'BACKUP_SHA256.json', before_files)
    plt.rcParams.update({'font.family': 'Arial', 'pdf.fonttype': 42,
                         'svg.fonttype': 'none', 'text.color': previous.INK,
                         'axes.labelcolor': previous.INK,
                         'xtick.color': previous.INK, 'ytick.color': previous.INK})
    fig = plt.figure(figsize=(previous.W, previous.H), facecolor='white')
    groups, headings = {}, {}
    for letter, draw, rect in previous.PANELS:
        n_axes, n_texts = len(fig.axes), len(fig.texts)
        draw(fig, rect)
        groups[letter] = fig.axes[n_axes:]
        headings[letter] = fig.texts[n_texts]
    previous.polish(fig, groups, headings)
    fig.canvas.draw()
    payload = {letter: previous.payload(axes) for letter, axes in groups.items()}
    bottom = groups['D'][0].get_position().y0
    top = groups['D'][0].get_position().y1
    original_bounds = {key: [list(ax.get_position().bounds) for ax in groups[key]]
                       for key in ['D', 'E', 'F']}

    lower = groups['E'][1]
    lower.set_xticks(range(4), ['Spearman ρ', 'Pearson r', 'AUROC', 'AUPRC'],
                     fontsize=10, fontweight='normal')
    p = lower.get_position()
    lower.set_position([p.x0, bottom, p.width, p.y1-bottom])
    lower.tick_params(axis='x', pad=4)

    # Move the first ridge column toward E and share the recovered width equally.
    left = groups['F'][0].get_position().x0 - .16/previous.W
    right = groups['F'][1].get_position().x1
    gap = .12/previous.W
    width = (right-left-gap)/2
    for i, (ax, title) in enumerate(zip(groups['F'], ['Observed CD19', 'Predicted CD19'])):
        ax.set_position([left+i*(width+gap), bottom, width, top-bottom])
        ax.set_title(title, fontsize=11.5, fontweight='normal', color=previous.INK, pad=7)
    removed = [text for text in fig.texts if text.get_text() == 'CD19']
    assert len(removed) == 1
    removed[0].remove()

    # Put H's legend in the empty upper-left region of the predicted-CD19 map.
    legends = [legend for legend in fig.legends
               if [text.get_text() for text in legend.get_texts()] == ['Mature B', 'NK', 'Others']]
    assert len(legends) == 1
    handles = list(legends[0].legend_handles)
    legends[0].remove()
    h_legend = groups['H'][1].legend(handles, ['Mature B', 'NK', 'Others'],
        loc='upper left', bbox_to_anchor=(.02, .985), ncol=1, frameon=False,
        fontsize=10.5, handletextpad=.4, labelspacing=.35, borderpad=0, borderaxespad=0)
    for text in h_legend.get_texts():
        text.set(fontfamily='Arial', fontweight='normal', color=previous.INK)
    g_box = groups['G'][0].get_position()
    for ax in groups['H']:
        p = ax.get_position()
        ax.set_position([p.x0, g_box.y0, p.width, g_box.height])
        ax.set_title(ax.get_title(), fontsize=11.5, fontweight='normal',
                     color=previous.INK, pad=6)
    g_metric = [text for text in fig.texts if text.get_text().startswith('Mean B-cell identity AUROC:')]
    h_metric = [text for text in fig.texts if 'silhouette:' in text.get_text()]
    assert len(g_metric) == len(h_metric) == 1
    h_metric[0].set_y(g_metric[0].get_position()[1])
    h_metric[0].set(fontsize=g_metric[0].get_fontsize(), fontweight='normal',
                    va=g_metric[0].get_va(), color=previous.INK)

    # Reclaim space between A/C headers and their content without resizing either.
    for letter in ['A', 'C']:
        ax = groups[letter][0]
        p = ax.get_position()
        shift = .25 if letter == 'A' else .30
        ax.set_position([p.x0, p.y0+shift/previous.H, p.width, p.height])
    # A slightly shorter B plot permits the entire first row to end earlier.
    for ax in groups['B']:
        p = ax.get_position()
        ax.set_position([p.x0, p.y0+.24/previous.H, p.width, p.height-.15/previous.H])

    # D now follows G: prediction on the left and independent annotations on the right.
    annotation_ax, prediction_ax, colorbar_ax = groups['D']
    left_map = annotation_ax.get_position().frozen()
    prediction_ax.set_position(left_map)
    annotation_ax.set_position([3.05/previous.W, left_map.y0, left_map.width, left_map.height])
    cp = colorbar_ax.get_position()
    colorbar_ax.set_position([2.66/previous.W, (bottom+top-cp.height)/2, cp.width, cp.height])
    # Recover the remaining whitespace by expanding E leftward at a fixed right edge.
    for ax in groups['E']:
        ep = ax.get_position()
        ax.set_position([ep.x0-.25/previous.W, ep.y0, ep.width+.25/previous.W, ep.height])
    for text in fig.texts:
        if text.get_text() in ['E', previous.TITLES['E']]:
            text.set_x(text.get_position()[0]-.25/previous.W)

    # Reclaim F's label gutter; each ridge row has its own internal label.
    old_f_width = groups['F'][0].get_position().width * previous.W
    f_left, f_right, f_gap = 10.16, groups['F'][1].get_position().x1 * previous.W, .12
    f_width = (f_right-f_left-f_gap)/2
    for i, ax in enumerate(groups['F']):
        ax.set_position([(f_left+i*(f_width+f_gap))/previous.W, bottom,
                         f_width/previous.W, top-bottom])
        ax.set_yticks([])
    f_row_labels = []
    for label, base_y in zip(['B cells', 'CD4 T', 'CD8 T'], [2, 1, 0]):
        f_row_labels.append(groups['F'][0].text(
            .035, base_y+.87, label, transform=groups['F'][0].get_yaxis_transform(),
            ha='left', va='top', fontsize=10.5, color=previous.INK))

    # Place the cohort description inside E, then distribute its freed height.
    upper = groups['E'][0]
    assert lower.get_title() == '14 markers · 12 donors'
    lower.set_title('')
    e_note = lower.text(.025, .92, '14 markers · 12 donors', transform=lower.transAxes,
                        ha='left', va='top', fontsize=10.5, color=previous.INK)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    up, lo = upper.get_position().frozen(), lower.get_position().frozen()
    old_e_gap = (up.y0-lo.y1)*previous.H
    x_text_bottom = min(t.get_window_extent(renderer).y0
                        for t in [upper.xaxis.label, *upper.get_xticklabels()] if t.get_visible())
    label_depth = (upper.get_window_extent(renderer).y0-x_text_bottom)/fig.dpi
    new_e_gap = label_depth + 3/72  # explicit clearance below the measured-CD8 label
    freed = old_e_gap-new_e_gap
    assert freed > .1, (old_e_gap, new_e_gap)
    upper_gain, lower_gain = .45*freed, .55*freed
    upper.set_position([up.x0, up.y0-upper_gain/previous.H, up.width,
                        up.height+upper_gain/previous.H])
    lower.set_position([lo.x0, lo.y0, lo.width, lo.height+lower_gain/previous.H])

    for letter, axes in groups.items():
        assert payload[letter] == previous.payload(axes), letter
    assert lower.get_xticklabels()[-1].get_text() == 'AUPRC'
    fig.canvas.draw()
    assert all(abs(ax.get_position().y0-bottom) < 1e-12
               for ax in [*groups['D'][:2], lower, *groups['F']])
    renderer = fig.canvas.get_renderer()
    f_ax_box = groups['F'][0].get_window_extent(renderer)
    f_label_curve_gaps = []
    for text, base_y in zip(f_row_labels, [2, 1, 0]):
        label_box = text.get_window_extent(renderer)
        assert f_ax_box.x0 <= label_box.x0 < label_box.x1 <= f_ax_box.x1
        local_curves = []
        for line in groups['F'][0].lines:
            xy = line.get_xydata()
            if len(xy) > 2 and np.min(xy[:, 1]) >= base_y-.01 and np.max(xy[:, 1]) <= base_y+.8:
                vertices = groups['F'][0].transData.transform(xy)
                local = vertices[(vertices[:, 0] >= label_box.x0) & (vertices[:, 0] <= label_box.x1), 1]
                if len(local):
                    local_curves.append(np.max(local))
        assert local_curves
        f_label_curve_gaps.append((label_box.y0-max(local_curves))/fig.dpi*72)
    assert min(f_label_curve_gaps) > 2, f_label_curve_gaps
    e_note_box = e_note.get_window_extent(renderer)
    e_note_covered_points = 0
    for collection in lower.collections:
        points = lower.transData.transform(collection.get_offsets())
        e_note_covered_points += int(np.sum((points[:, 0] >= e_note_box.x0) & (points[:, 0] <= e_note_box.x1)
                                           & (points[:, 1] >= e_note_box.y0) & (points[:, 1] <= e_note_box.y1)))
    assert e_note_covered_points == 0, e_note_covered_points
    e_label_gap = (upper.xaxis.label.get_window_extent(renderer).y0-lower.get_window_extent(renderer).y1)/fig.dpi*72
    assert e_label_gap >= 2.9, e_label_gap
    e_f_gap = (groups['F'][0].get_window_extent(renderer).x0-upper.get_window_extent(renderer).x1)/fig.dpi*72
    assert e_f_gap > 10, e_f_gap
    cb_box = colorbar_ax.get_tightbbox(renderer)
    left_box = prediction_ax.get_window_extent(renderer)
    right_box = annotation_ax.get_window_extent(renderer)
    colorbar_clearance = [cb_box.x0-left_box.x1, right_box.x0-cb_box.x1]
    assert min(colorbar_clearance) > 2/72*fig.dpi, colorbar_clearance
    d_e_gap = min(ax.yaxis.label.get_window_extent(renderer).x0 for ax in groups['E'])-right_box.x1
    assert d_e_gap > 4/72*fig.dpi, d_e_gap
    assert annotation_ax.get_aspect() == prediction_ax.get_aspect() == 1.0
    assert abs(left_box.width-right_box.width) < 1e-6
    assert abs(left_box.height-right_box.height) < 1e-6
    for ax in groups['H']:
        assert abs(ax.get_position().y0-g_box.y0) < 1e-12
        assert abs(ax.get_position().y1-g_box.y1) < 1e-12
    assert h_metric[0].get_window_extent(renderer).y0 == g_metric[0].get_window_extent(renderer).y0
    legend_box = h_legend.get_window_extent(renderer)
    legend_covered_cells = 0
    for collection in groups['H'][1].collections:
        points = groups['H'][1].transData.transform(collection.get_offsets())
        legend_covered_cells += int(np.sum((points[:, 0] >= legend_box.x0) & (points[:, 0] <= legend_box.x1)
                                   & (points[:, 1] >= legend_box.y0) & (points[:, 1] <= legend_box.y1)))
    assert legend_covered_cells == 0, legend_covered_cells
    # Check first-row content and headings, including the top of the second row.
    second_heading_top = max(text.get_window_extent(renderer).y1 for text in fig.texts
                             if text.get_text() in ['D', 'E', 'F'])
    first_bottom = min([groups['A'][0].get_window_extent(renderer).y0,
                        *[t.get_window_extent(renderer).y0 for ax in groups['B']
                          for t in ax.get_xticklabels() if t.get_visible()],
                        *[patch.get_window_extent(renderer).y0 for patch in groups['C'][0].patches]])
    assert first_bottom > second_heading_top + 4/72*fig.dpi
    header_gaps = {}
    for letter in ['A', 'B', 'C']:
        heading = next(t for t in fig.texts if t.get_text() == previous.TITLES[letter])
        if letter == 'A':
            content_top = groups['A'][0].get_window_extent(renderer).y1
        elif letter == 'B':
            content_top = max(ax._left_title.get_window_extent(renderer).y1 for ax in groups['B'])
        else:
            content_top = max(patch.get_window_extent(renderer).y1 for patch in groups['C'][0].patches)
        header_gaps[letter] = (heading.get_window_extent(renderer).y0-content_top)/fig.dpi*72
    assert min(header_gaps.values()) > 3, header_gaps
    # Ensure the new one-line labels fit above the next row's headings.
    next_heading_top = max(text.get_window_extent(renderer).y1 for text in fig.texts
                           if text.get_text() in ['G', 'H'])
    lowest_tick = min(t.get_window_extent(renderer).y0
                     for ax in [lower, *groups['F']] for t in ax.get_xticklabels()
                     if t.get_visible())
    assert lowest_tick > next_heading_top, (lowest_tick, next_heading_top)
    name = 'cd19_story_three_row_v23_polish'
    fig.savefig(OUT / f'{name}.png', dpi=300, facecolor='white')
    fig.savefig(OUT / f'{name}.svg', facecolor='white')
    fig.savefig(QA / 'working_size.pdf', dpi=600, facecolor='white')
    fig.savefig(OUT / 'row1_preview.png', dpi=300, facecolor='white',
                bbox_inches=Bbox.from_extents(.10, previous.H-(2.74-ROW1_REDUCTION), 14.42, previous.H-.08))
    fig.savefig(OUT / 'row2_preview.png', dpi=300, facecolor='white',
                bbox_inches=Bbox.from_extents(.10, previous.H-(6.40-ROW1_REDUCTION), 14.42, previous.H-(2.74-ROW1_REDUCTION)))
    fig.savefig(OUT / 'row3_preview.png', dpi=300, facecolor='white',
                bbox_inches=Bbox.from_extents(.10, .05, 14.42, previous.H-(6.43-ROW1_REDUCTION)))
    final_bounds = {key: [list(ax.get_position().bounds) for ax in groups[key]]
                    for key in ['D', 'E', 'F']}
    plt.close(fig)
    page = PdfReader(QA / 'working_size.pdf').pages[0]
    final_width = 180/25.4*72
    page.scale_to(final_width, final_width*previous.H/previous.W)
    writer = PdfWriter()
    writer.add_page(page)
    writer.add_metadata({'/Title': 'CD19 story v23: compact E/F with internal annotations'})
    with (OUT / f'{name}.pdf').open('wb') as handle:
        writer.write(handle)
    prior = json.loads((OLDER / 'LAYOUT_AUDIT.json').read_text())
    assert all(sha(path) == digest for path, digest in prior['source_sha256'].items())
    assert all(sha(path) == digest for path, digest in before_files.items())
    audit = {'status': 'PASS', 'time_utc': now(), 'script_sha256': sha(__file__),
        'source_sha256': prior['source_sha256'], 'previous_v22_files_unchanged': True,
        'all_scientific_artist_payloads_unchanged': True, 'manuscript_modified': False,
        'E_internal_note': e_note.get_text(), 'E_internal_note_covered_points': e_note_covered_points,
        'E_gap_before_after_inches': [old_e_gap, new_e_gap],
        'E_upper_lower_height_gain_inches': [upper_gain, lower_gain],
        'E_x_label_clearance_working_pt': e_label_gap,
        'F_row_label_positions': 'Observed CD19: B cells, CD4 T, CD8 T at the upper left of their respective rows',
        'F_row_label_curve_clearances_working_pt': f_label_curve_gaps,
        'F_column_width_before_after_inches': [old_f_width, f_width],
        'E_F_plot_gap_working_pt': e_f_gap,
        'E_last_metric_label': 'AUPRC', 'F_column_titles': ['Observed CD19', 'Predicted CD19'],
        'F_bottom_CD19_label_removed': True, 'D_E_F_bottom_edges_equal': True,
        'aligned_bottom_inches_from_top': (1-bottom)*previous.H,
        'original_axis_rectangles_normalized': original_bounds,
        'final_axis_rectangles_normalized': final_bounds,
        'tick_to_next_heading_gap_working_pt': (lowest_tick-next_heading_top)/fig.dpi*72,
        'F_left_expansion_inches': .60, 'F_column_gap_inches': .12,
        'inherited_third_row_gap_inches': ROW3_SHIFT,
        'first_row_height_reduction_inches': ROW1_REDUCTION,
        'first_row_height_reduction_fraction': ROW1_REDUCTION/(2.79-.13),
        'A_C_content_shift_up_inches': {'A': .25, 'C': .30},
        'B_plot_height_reduction_inches': .15,
        'B_plot_top_shift_up_inches': .09,
        'lower_two_rows_shift_up_inches': ROW1_REDUCTION,
        'canvas_inches': [previous.W, previous.H],
        'header_to_content_gap_working_pt': header_gaps,
        'row1_to_row2_min_gap_working_pt': (first_bottom-second_heading_top)/fig.dpi*72,
        'D_display_order': ['Predicted CD3', 'Cell annotation'],
        'D_colorbar_position': 'Between the two UMAPs',
        'D_celltype_legend': 'Inside right-hand annotation UMAP, upper left',
        'D_UMAP_aspect_and_size_unchanged': True,
        'E_left_expansion_inches': .25, 'E_width_increase_fraction': .25/3.60,
        'D_E_text_clearance_working_pt': d_e_gap/fig.dpi*72,
        'D_colorbar_clearance_working_pt': [val/fig.dpi*72 for val in colorbar_clearance],
        'H_legend_location': 'Inside predicted-CD19 UMAP, upper left',
        'H_legend_covered_cells': legend_covered_cells,
        'G_H_plot_top_and_bottom_edges_aligned': True,
        'G_H_metric_text_baselines_aligned': True,
        'H_silhouette_position': 'Bottom of panel',
        'no_model_or_UMAP_refit': True, 'final_PDF_width_mm': 180}
    save(OUT / 'LAYOUT_AUDIT.json', audit)
    (OUT / 'README_ZH.md').write_text("""# CD19 v23：E/F 图内说明与绘图区扩展

上一版整图、分行预览、说明、核验记录和脚本保存在 backup_v22。

- F 的 B cells / CD4 T / CD8 T 分别放在 Observed CD19 第一、第二、第三行的左上角，两列共用行标签。移除外侧 Y 轴类别标签，两列 ridge plot 等量加宽。
- E 的 14 markers · 12 donors 移到下图内左上角并与上边框留出间距；将腾出的竖直空间分配给上下两图，Measured CD8 轴标题与下图边框仍留有间隔。
- 行标签不覆盖 ridge 曲线，下图说明不覆盖任何 donor 点；D/E/F 底边保持对齐。
- 保留 D 预测在左、注释在右、色条居中，以及第一行压缩和 G/H 底部指标对齐。
- 数据、预测、色阶、全部散点、ridge 曲线和指标保持不变。

整图为 cd19_story_three_row_v23_polish.png / .svg / .pdf；row2_preview.png 单独展示 D/E/F。PDF 宽度为 180 mm。

复现：在 story_cd19_prediction 下运行 `python3 -B Exploration0909/scripts/assemble_cd19_layout_v23_polish.py`。

精确尺寸、文字间隔与输入核验见 LAYOUT_AUDIT.json。
""")
    print(json.dumps({k: audit[k] for k in ['status', 'aligned_bottom_inches_from_top',
          'E_gap_before_after_inches', 'F_column_width_before_after_inches',
          'F_row_label_curve_clearances_working_pt']}, indent=2))
    print(OUT / f'{name}.png')


if __name__ == '__main__':
    main()
