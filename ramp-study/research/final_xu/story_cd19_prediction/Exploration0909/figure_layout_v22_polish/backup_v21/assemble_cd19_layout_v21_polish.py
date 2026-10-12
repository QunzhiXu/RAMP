"""Remove excess first-row spacing and shift the lower rows up, preserving v20."""
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

OLDER = BASE / 'figure_layout_v20_polish'
OUT = BASE / 'figure_layout_v21_polish'
QA = OUT / 'qa'
BACKUP = OUT / 'backup_v20'
for directory in [OUT, QA, BACKUP]:
    directory.mkdir(parents=True, exist_ok=True)


def main():
    before_files = {}
    for path in [*OLDER.glob('*.*'), BASE / 'scripts/assemble_cd19_layout_v20_polish.py']:
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

    for letter, axes in groups.items():
        assert payload[letter] == previous.payload(axes), letter
    assert lower.get_xticklabels()[-1].get_text() == 'AUPRC'
    fig.canvas.draw()
    assert all(abs(ax.get_position().y0-bottom) < 1e-12
               for ax in [*groups['D'][:2], lower, *groups['F']])
    renderer = fig.canvas.get_renderer()
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
    name = 'cd19_story_three_row_v21_polish'
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
    writer.add_metadata({'/Title': 'CD19 story v21: compact first row'})
    with (OUT / f'{name}.pdf').open('wb') as handle:
        writer.write(handle)
    prior = json.loads((OLDER / 'LAYOUT_AUDIT.json').read_text())
    assert all(sha(path) == digest for path, digest in prior['source_sha256'].items())
    assert all(sha(path) == digest for path, digest in before_files.items())
    audit = {'status': 'PASS', 'time_utc': now(), 'script_sha256': sha(__file__),
        'source_sha256': prior['source_sha256'], 'previous_v20_files_unchanged': True,
        'all_scientific_artist_payloads_unchanged': True, 'manuscript_modified': False,
        'E_last_metric_label': 'AUPRC', 'F_column_titles': ['Observed CD19', 'Predicted CD19'],
        'F_bottom_CD19_label_removed': True, 'D_E_F_bottom_edges_equal': True,
        'aligned_bottom_inches_from_top': (1-bottom)*previous.H,
        'original_axis_rectangles_normalized': original_bounds,
        'final_axis_rectangles_normalized': final_bounds,
        'tick_to_next_heading_gap_working_pt': (lowest_tick-next_heading_top)/fig.dpi*72,
        'F_left_expansion_inches': .16, 'F_column_gap_inches': .12,
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
        'H_legend_location': 'Inside predicted-CD19 UMAP, upper left',
        'H_legend_covered_cells': legend_covered_cells,
        'G_H_plot_top_and_bottom_edges_aligned': True,
        'G_H_metric_text_baselines_aligned': True,
        'H_silhouette_position': 'Bottom of panel',
        'no_model_or_UMAP_refit': True, 'final_PDF_width_mm': 180}
    save(OUT / 'LAYOUT_AUDIT.json', audit)
    (OUT / 'README_ZH.md').write_text("""# CD19 v21：压缩第一行留白

上一版整图、分行预览、说明、核验记录和脚本保存在 backup_v20，v20 原目录不变。

- 第一行整体收短 0.40 英寸，约为原第一行跨度的 15%。
- A 的图像上移 0.25 英寸，C 的 workflow 内容上移 0.30 英寸，保持原有尺寸与文字大小。
- B 的绘图区顶部上移 0.09 英寸、高度减少 0.15 英寸，保留全部数据和坐标范围。
- 第二、第三行整体上移 0.40 英寸，既有尺寸、D/E/F 底边对齐及 G/H 图例和指标排布保持。
- 画布由 14.5×10.58 英寸变为 14.5×10.18 英寸。PDF 宽度仍为 180 mm。
- 空间图像像素、UMAP 坐标、点数、预测值、色阶和全部指标未改。稿件文件尚未替换。

整图为 cd19_story_three_row_v21_polish.png / .svg / .pdf；row1/2/3_preview.png 为分行预览。

复现：在 story_cd19_prediction 下运行 `python3 -B Exploration0909/scripts/assemble_cd19_layout_v21_polish.py`。

标题间隔、行间隔和数据核验见 LAYOUT_AUDIT.json。
""")
    print(json.dumps({k: audit[k] for k in ['status', 'aligned_bottom_inches_from_top',
          'first_row_height_reduction_fraction', 'header_to_content_gap_working_pt', 'row1_to_row2_min_gap_working_pt']}, indent=2))
    print(OUT / f'{name}.png')


if __name__ == '__main__':
    main()
