"""Record the narrower layout without changing captions or scientific panel sources."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
FIGURE = HERE.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    abc = FIGURE/'Combined ABC'
    full = FIGURE/'Combined ABCDEF'
    path = abc/'README_ZH.md'
    text = path.read_text()
    a = text.index('当前为 ')
    b = text.index('\n\n只读', a)
    text = text[:a]+(
        '当前为 v4_tighter_rows_and_BC。保留新版 CROSS-PLATFORM ALIGNMENT 图标，'
        '将 C 比 v3 再向左移动 6 pt，压缩 B 图例与 C 之间的空白（相对最初布局累计 16 pt）。'
        'A 主体仍等比例缩小至 78%，B、C 保持原尺寸；面板字母 22 pt、标题 12.3 pt。'
        'A、B 间隔 3 pt；B、C 外画布重叠 13 pt，仅利用已有白边，不覆盖图例或坐标轴。'
        'B、C 绘图区高度、基线及 UMAP 坐标保持不变。画布 1019.04 × 198 pt '
        '（约 14.153 × 2.75 英寸），PNG 为 300 dpi，SVG 保留矢量文字和元素。'
        '替换前版本保存在 `versions/v3_before_spacing_polish/`。')+text[b:]
    path.write_text(text)
    path = full/'README_ZH.md'
    text = path.read_text().replace('v7_alignment_icon_and_synthetic_C30_ROI', 'v8_tighter_rows_and_BC')
    text = text.replace('替换前完整版本保存在 `versions/v6_before_alignment_icon_and_ROI/`。',
        '本次仅收紧布局，替换前完整版本保存在 `versions/v7_before_spacing_polish/`。')
    a = text.index('第一行直接复用')
    b = text.index('\n\n只组合', a)
    text = text[:a]+(
        '第一行复用 ABC v4，C 比上一版左移 6 pt。第二行 D、E、F 保持原图尺寸，'
        '面板外画布间距由 3 pt 改为 0 pt；已有内边距使图中元素仍有间隔。'
        '第二行整体上移 15 pt，两行实际纯白间距由 20.64 pt 压缩至 5.76 pt '
        '（300 dpi PNG 中由 86 像素变为 24 像素）。组合配置 row_gap 为 −7 pt，'
        '只重叠源画布的白边。整图 1019.04 × 389 pt，约 14.153 × 5.403 英寸，'
        'PNG 为 4246 × 1621 像素。主标题 12.3 pt、面板字母 22 pt 及独立面板内容均不变。'
        '布局检查和 manuscript 同步记录见 `../polish_spacing_20260914/`。')+text[b:]
    path.write_text(text)
    path = full/'provenance/CAPTION_SOURCES.json'
    obj = json.loads(path.read_text())
    obj['figure_version'] = 'v8_tighter_rows_and_BC'
    obj['figure_sha256'] = sha(full/'panels_ABCDEF_preview.png')
    obj['latest_layout_revision'] = dict(scope='Spacing only', caption_unchanged=True)
    path.write_text(json.dumps(obj, indent=2)+'\n')
    for folder in [abc, full]:
        path = folder/'provenance/VISUAL_REVIEW.json'
        path.write_text(json.dumps(dict(status='PASS', review_date='2026-09-14',
            scope='Tighter B/C gap and row spacing; source panel sizes and text unchanged',
            manuscript_page=10, source_panel_files_unchanged=True,
            no_label_overlap=True), indent=2)+'\n')
        entries = []
        for p in sorted(folder.rglob('*')):
            parts = p.relative_to(folder).parts
            if p.is_file() and not any(x in parts for x in ['versions', 'cache', 'tmp', '__pycache__']) and p.name != 'MANIFEST.json':
                entries.append(dict(file=str(p.relative_to(folder)), sha256=sha(p)))
        (folder/'provenance/MANIFEST.json').write_text(json.dumps(entries, indent=2)+'\n')
    installed = json.loads((HERE/'manuscript_sync/INSTALLED.json').read_text())
    (HERE/'README_ZH.md').write_text(
        '# Lung 整图间距调整（2026-09-14）\n\n'
        '当前整图：`lung_panels_ABCDEF.png` / `.svg`。\n\n'
        '- 两行纯白间距：20.64 → 5.76 pt。\n'
        '- C 左移 6 pt；第二行两个外画布间隔各减少 3 pt。\n'
        '- 所有 A–F 源图文件哈希不变，字号、图内尺寸、数据与模型不变。\n'
        '- 旧组合稿在 Combined ABC / Combined ABCDEF 的 versions 中，路径见 BACKUPS.json。\n'
        '- 两份 manuscript 均重新编译，仍为 18 页；Figure 5 及图注在第 10 页。'
        '其他 17 页逐页像素一致，所有 TeX 和 caption 文字不变。\n\n'
        'manuscript 替换前备份：`'+installed['backup']+'`。\n\n'
        '复现顺序：运行两个 Combined 文件夹的 assemble 脚本；prepare_and_verify.py；'
        '在 manuscript_sync/staging 编译两份 TeX；check_manuscript.py；'
        '目视检查第 10 页后运行 manuscript_sync/install_checked_files.py；最后 update_documentation.py。\n')
    print('Updated spacing documentation and source manifests.')


if __name__ == '__main__':
    main()
