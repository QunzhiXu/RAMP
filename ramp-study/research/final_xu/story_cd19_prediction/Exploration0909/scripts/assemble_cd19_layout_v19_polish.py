"""Extend E/F to D's baseline and simplify labels; keep frozen v18 data."""
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

OUT = BASE / 'figure_layout_v19_polish'
QA = OUT / 'qa'
BACKUP = OUT / 'backup_v18'
for directory in [OUT, QA, BACKUP]:
    directory.mkdir(parents=True, exist_ok=True)


def main():
    before_files = {}
    for path in [*previous.OUT.glob('*.*'), BASE / 'scripts/assemble_cd19_layout_v18_polish.py']:
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

    for letter, axes in groups.items():
        assert payload[letter] == previous.payload(axes), letter
    assert lower.get_xticklabels()[-1].get_text() == 'AUPRC'
    fig.canvas.draw()
    assert all(abs(ax.get_position().y0-bottom) < 1e-12
               for ax in [*groups['D'][:2], lower, *groups['F']])
    renderer = fig.canvas.get_renderer()
    # Ensure the new one-line labels fit above the next row's headings.
    next_heading_top = max(text.get_window_extent(renderer).y1 for text in fig.texts
                           if text.get_text() in ['G', 'H'])
    lowest_tick = min(t.get_window_extent(renderer).y0
                     for ax in [lower, *groups['F']] for t in ax.get_xticklabels()
                     if t.get_visible())
    assert lowest_tick > next_heading_top, (lowest_tick, next_heading_top)
    name = 'cd19_story_three_row_v19_polish'
    fig.savefig(OUT / f'{name}.png', dpi=300, facecolor='white')
    fig.savefig(OUT / f'{name}.svg', facecolor='white')
    fig.savefig(QA / 'working_size.pdf', dpi=600, facecolor='white')
    fig.savefig(OUT / 'row2_preview.png', dpi=300, facecolor='white',
                bbox_inches=Bbox.from_extents(.10, previous.H-6.40, 14.42, previous.H-2.74))
    final_bounds = {key: [list(ax.get_position().bounds) for ax in groups[key]]
                    for key in ['D', 'E', 'F']}
    plt.close(fig)
    page = PdfReader(QA / 'working_size.pdf').pages[0]
    final_width = 180/25.4*72
    page.scale_to(final_width, final_width*previous.H/previous.W)
    writer = PdfWriter()
    writer.add_page(page)
    writer.add_metadata({'/Title': 'CD19 story v19: aligned second-row plot boundaries'})
    with (OUT / f'{name}.pdf').open('wb') as handle:
        writer.write(handle)
    prior = json.loads((previous.OUT / 'LAYOUT_AUDIT.json').read_text())
    assert all(sha(path) == digest for path, digest in prior['source_sha256'].items())
    assert all(sha(path) == digest for path, digest in before_files.items())
    audit = {'status': 'PASS', 'time_utc': now(), 'script_sha256': sha(__file__),
        'source_sha256': prior['source_sha256'], 'previous_v18_files_unchanged': True,
        'all_scientific_artist_payloads_unchanged': True, 'manuscript_modified': False,
        'E_last_metric_label': 'AUPRC', 'F_column_titles': ['Observed CD19', 'Predicted CD19'],
        'F_bottom_CD19_label_removed': True, 'D_E_F_bottom_edges_equal': True,
        'aligned_bottom_inches_from_top': (1-bottom)*previous.H,
        'original_axis_rectangles_normalized': original_bounds,
        'final_axis_rectangles_normalized': final_bounds,
        'tick_to_next_heading_gap_working_pt': (lowest_tick-next_heading_top)/fig.dpi*72,
        'F_left_expansion_inches': .16, 'F_column_gap_inches': .12,
        'third_row_shift_down_inches': ROW3_SHIFT,
        'no_model_or_UMAP_refit': True, 'final_PDF_width_mm': 180}
    save(OUT / 'LAYOUT_AUDIT.json', audit)
    (OUT / 'README_ZH.md').write_text('''# CD19 v19：第二行底边对齐

在 v18 细框版本上修改 E/F；v18 整图、分行图、说明和脚本保存在 backup_v18，旧版文件不变。

- E 的最后一项标签由 AUPRC (AP) 简化为 AUPRC；指标本身仍是既有 average precision，数值不变。
- E 下方统计图向下延长，底边与 D 的 UMAP 对齐。
- F 标题改为 Observed CD19 / Predicted CD19，移除底部 CD19；两列同时向下延长，与 D/E 共用底边。
- F 左边向 E 扩展 0.16 英寸，两列均分回收的宽度，列间距设为 0.12 英寸。
- 第三行整体下移 0.08 英寸、画布相应增高，给第二行刻度与第三行标题留出间隔。
- 坐标范围、细胞、预测值、色阶、全部指标、空间图像素和 KDE 曲线均不变。稿件尚未替换。

完整图为 cd19_story_three_row_v19_polish.png / .svg / .pdf，第二行预览为 row2_preview.png。PDF 宽度为 180 mm。

复现：在 story_cd19_prediction 下运行 `python3 -B Exploration0909/scripts/assemble_cd19_layout_v19_polish.py`。

底边和相邻行文字间隔的核验见 LAYOUT_AUDIT.json。
''')
    print(json.dumps({k: audit[k] for k in ['status', 'aligned_bottom_inches_from_top',
          'tick_to_next_heading_gap_working_pt']}, indent=2))
    print(OUT / f'{name}.png')


if __name__ == '__main__':
    main()
