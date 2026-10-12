"""Align H with G and move its legend and silhouette annotation, preserving v19."""
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

OLDER = BASE / 'figure_layout_v19_polish'
OUT = BASE / 'figure_layout_v20_polish'
QA = OUT / 'qa'
BACKUP = OUT / 'backup_v19'
for directory in [OUT, QA, BACKUP]:
    directory.mkdir(parents=True, exist_ok=True)


def main():
    before_files = {}
    for path in [*OLDER.glob('*.*'), BASE / 'scripts/assemble_cd19_layout_v19_polish.py']:
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
    # Ensure the new one-line labels fit above the next row's headings.
    next_heading_top = max(text.get_window_extent(renderer).y1 for text in fig.texts
                           if text.get_text() in ['G', 'H'])
    lowest_tick = min(t.get_window_extent(renderer).y0
                     for ax in [lower, *groups['F']] for t in ax.get_xticklabels()
                     if t.get_visible())
    assert lowest_tick > next_heading_top, (lowest_tick, next_heading_top)
    name = 'cd19_story_three_row_v20_polish'
    fig.savefig(OUT / f'{name}.png', dpi=300, facecolor='white')
    fig.savefig(OUT / f'{name}.svg', facecolor='white')
    fig.savefig(QA / 'working_size.pdf', dpi=600, facecolor='white')
    fig.savefig(OUT / 'row2_preview.png', dpi=300, facecolor='white',
                bbox_inches=Bbox.from_extents(.10, previous.H-6.40, 14.42, previous.H-2.74))
    fig.savefig(OUT / 'row3_preview.png', dpi=300, facecolor='white',
                bbox_inches=Bbox.from_extents(.10, .05, 14.42, previous.H-6.43))
    final_bounds = {key: [list(ax.get_position().bounds) for ax in groups[key]]
                    for key in ['D', 'E', 'F']}
    plt.close(fig)
    page = PdfReader(QA / 'working_size.pdf').pages[0]
    final_width = 180/25.4*72
    page.scale_to(final_width, final_width*previous.H/previous.W)
    writer = PdfWriter()
    writer.add_page(page)
    writer.add_metadata({'/Title': 'CD19 story v20: aligned third-row plots and annotations'})
    with (OUT / f'{name}.pdf').open('wb') as handle:
        writer.write(handle)
    prior = json.loads((OLDER / 'LAYOUT_AUDIT.json').read_text())
    assert all(sha(path) == digest for path, digest in prior['source_sha256'].items())
    assert all(sha(path) == digest for path, digest in before_files.items())
    audit = {'status': 'PASS', 'time_utc': now(), 'script_sha256': sha(__file__),
        'source_sha256': prior['source_sha256'], 'previous_v19_files_unchanged': True,
        'all_scientific_artist_payloads_unchanged': True, 'manuscript_modified': False,
        'E_last_metric_label': 'AUPRC', 'F_column_titles': ['Observed CD19', 'Predicted CD19'],
        'F_bottom_CD19_label_removed': True, 'D_E_F_bottom_edges_equal': True,
        'aligned_bottom_inches_from_top': (1-bottom)*previous.H,
        'original_axis_rectangles_normalized': original_bounds,
        'final_axis_rectangles_normalized': final_bounds,
        'tick_to_next_heading_gap_working_pt': (lowest_tick-next_heading_top)/fig.dpi*72,
        'F_left_expansion_inches': .16, 'F_column_gap_inches': .12,
        'third_row_shift_down_inches': ROW3_SHIFT,
        'H_legend_location': 'Inside predicted-CD19 UMAP, upper left',
        'H_legend_covered_cells': legend_covered_cells,
        'G_H_plot_top_and_bottom_edges_aligned': True,
        'G_H_metric_text_baselines_aligned': True,
        'H_silhouette_position': 'Bottom of panel',
        'no_model_or_UMAP_refit': True, 'final_PDF_width_mm': 180}
    save(OUT / 'LAYOUT_AUDIT.json', audit)
    (OUT / 'README_ZH.md').write_text("""# CD19 v20：H 图例与指标位置调整

保留 v19 的 E/F 修改，并在 backup_v19 中备份上一版整图、说明、核验记录和脚本。

- H 的 Mature B / NK / Others 图例移入右侧 + predicted CD19 UMAP 的左上角，纵向排列。
- H 顶部 silhouette 指标移至底部，与 G 的 Mean B-cell identity AUROC 说明共用文字基线。
- H 两张 UMAP 的绘图区上下边缘均与 G 两张空间图对齐，数据坐标范围保持不变。
- 图例覆盖细胞数为 0；全部点、预测值、配色、空间图像素、ridge 曲线和指标均与 v19 一致。
- 原版及 manuscript 均未替换。

整图为 cd19_story_three_row_v20_polish.png / .svg / .pdf，row3_preview.png 为第三行预览。PDF 宽度为 180 mm。

复现：在 story_cd19_prediction 下运行 `python3 -B Exploration0909/scripts/assemble_cd19_layout_v20_polish.py`。

核验记录见 LAYOUT_AUDIT.json。
""")
    print(json.dumps({k: audit[k] for k in ['status', 'aligned_bottom_inches_from_top',
          'tick_to_next_heading_gap_working_pt', 'H_legend_covered_cells']}, indent=2))
    print(OUT / f'{name}.png')


if __name__ == '__main__':
    main()
