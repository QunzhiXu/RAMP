"""Document the ridge-only revision without changing previous versions."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLAN = json.loads((HERE / 'REVISION_PLAN.json').read_text())
color_names = {'CODEX': 'blue', 'CyTOF': 'green', 'IMC': 'orange', 'IMC2': 'light orange',
               'CITE-seq': 'purple', 'AbSeq': 'magenta', 'MIBI': 'teal', 'mIF': 'rose'}

for c in PLAN:
    old, new = Path(c['old']), Path(c['new'])
    cfg = json.loads((new / 'RIDGE_CONFIG.json').read_text())
    caption = (old / 'CAPTION.md').read_text().replace('Supplementary figure draft.', 'Supplementary figure.')
    if c['name'] == 'TNBC':
        caption = caption.replace('colored by dataset as in a', 'colored by dataset according to the separate legend in b')
    elif c['name'] == 'Melanoma story':
        caption = caption.replace('Blue denotes IMC and orange CITE-seq.', 'Orange denotes IMC and purple CITE-seq.')
    elif c['name'] == 'HNSCC story':
        old_text = 'CD3, CD4 and CD8 distributions for all 77 donor/patient units, colored by platform as in a. Each ridge uses all retained cells from its unit.'
        new_text = ('CD3, CD4 and CD8 distributions for a fixed display subset of 30 donor/patient units: '
                    'all three CITE-seq donors and 27 of the 74 CODEX patients (47,488 cells in total). '
                    'Each ridge uses all frozen cells from its selected unit. Selection is determined only by '
                    'a SHA256 ordering of dataset and patient IDs with the fixed seed string 20260922, '
                    'independently of expression values or method performance. The identical selected units '
                    'are shown for every method and marker; UMAPs and quantitative evaluations retain their original scope.')
        assert old_text in caption
        caption = caption.replace(old_text, new_text)
    elif c['name'] == 'Lung HLA-ABC story':
        old_text = 'Each curve represents one reference patient, colored by platform; all 84 patients and all reference cells are retained for each method-marker combination.'
        new_text = ('Each curve represents one reference patient from a fixed display subset of ten patients '
                    'per platform (30 patients and 30,000 frozen reference cells in total). Selection is '
                    'determined only by a SHA256 ordering of dataset and patient IDs with the fixed seed '
                    'string 20260922, independently of expression values or method performance. All frozen '
                    'cells from each selected patient are retained, and the identical patients are used '
                    'for every method-marker combination. UMAPs retain all 84,000 reference cells; '
                    'quantitative evaluations are unchanged.')
        assert old_text in caption
        caption = caption.replace(old_text, new_text)
    palette = ', '.join(f"{p['name']} ({color_names[p['technology']]})" for p in cfg['platforms'])
    caption += ('\nRidge colors follow a fixed technology palette across the seven supplementary benchmark pages: '
                + palette + '. Platform colors in the UMAPs remain those of their corresponding original figures; '
                'the two blocks therefore have separate legends. Ridge legend counts denote displayed donor/patient '
                'or sample units. The left-to-right ridge legend order matches the top-to-bottom platform stacking order, '
                'with selected unit IDs sorted within each platform and held fixed across methods and markers. '
                'Display limits are computed from the complete frozen cohort for each method-marker pair, including '
                'when a subset of units is displayed. Fill opacity and platform-colored contour widths are shared '
                'across all seven pages. Each contour is continuous across the full display interval, including '
                'zero-density tails; density values and filled curves are unchanged. No alignment, UMAP fitting '
                'or benchmark scoring was repeated for this revision.\n')
    (new / 'CAPTION.md').write_text(caption)
    counts = '、'.join(f"{p['name']} {len(p['selected_unit_ids'])}/{p['available_units']}" for p in cfg['platforms'])
    (new / 'README_ZH.md').write_text(
        f"# {c['name']}：统一 ridge 版本\n\n"
        f"当前图：`output/{c['png']}`；PDF：`output/pdf/{Path(c['png']).with_suffix('.pdf')}`。\n\n"
        f"仅更新 ridge 配色、轮廓、独立图例及展示样本。UMAP 坐标、点颜色、透明度、绘制顺序、输入矩阵和评分不变。\n\n"
        f"Ridge 展示 {cfg['ridge_units']} 个样本单位、{cfg['ridge_cells']:,} 个冻结细胞。按平台：{counts}（展示数/原有数）。"
        "每个入选单位保留其全部冻结细胞；所有方法及 marker 使用同一名单。HNSCC/Lung story 的抽样只使用 ID 和固定种子，"
        "不使用表达或评分。其他五组展示全部样本。\n\n"
        "同一技术跨七页固定颜色，TNBC 的 IMC1/IMC2 使用橙色深浅区分。UMAP 沿用原图颜色；两块分别设置图例。"
        "Ridge 图例括号内为展示样本数，从左到右对应曲线从上到下的技术顺序，同一技术内按 ID 排序。"
        "所有图统一填充透明度、轮廓宽度、平滑和峰值归一化。轮廓线连续覆盖完整横轴，保留零密度尾部，不再按密度截断。横轴仍取全冻结队列的 0.5–99.5 百分位。\n\n"
        "- `RIDGE_CONFIG.json`：配色、数量、抽样与样式。\n"
        "- `ridge_unit_selection.csv`：完整候选及入选名单。\n"
        "- `ridge_display_audit.csv`：每条曲线的细胞数、范围、颜色及密度哈希。\n"
        "- `MANIFEST.json`、`QA.json`：输入/输出哈希及核验。\n"
        "- `CAPTION.md`：已同步展示范围和独立图例说明。\n\n"
        f"旧版本完整保留在 `{old}`。本目录的冻结坐标或缓存以只读使用的符号链接引用旧资源。\n\n"
        f"重画：`python3 '{new / 'build_figure.py'}'`。共用渲染逻辑见 `../../ridge_style_update_20260922/ridge_common.py`。\n")

lines = ['# Supplementary benchmark figures', '',
         '七组附录页已统一 ridge 样式，当前版本均为各目录的 `ridge_uniform_20260922/`，入口见各自 `CURRENT_PREVIEW.json`。每组一张 A4 竖版，包含七组 UMAP 和六组 marker 分布。', '',
         '本次仅统一 ridge 的技术配色、轮廓、独立图例和堆叠规则，并减少 HNSCC/Lung story 展示的患者数。原有 UMAP 内容、对齐数据和评分保持不变。', '',
         '| Benchmark | Ridge 展示样本单位 | 入选冻结细胞 | 当前目录 |',
         '|---|---:|---:|---|']
for c in PLAN:
    cfg = json.loads(Path(c['new'], 'RIDGE_CONFIG.json').read_text())
    folder = Path(c['new'])
    lines.append(f"| {c['name']} | {cfg['ridge_units']}/{cfg['full_units']} | {cfg['ridge_cells']:,} | [{folder.name}](<{folder}>) |")
lines += ['', 'Lung HLA-ABC story 每个平台从 28 位患者中选 10 位；HNSCC 保留全部 3 位 CITE-seq donor，并从 74 位 CODEX 患者中选 27 位。抽样依据固定种子与 ID 哈希顺序，不依据表达或方法表现；每个入选单位保留全部冻结细胞，所有方法和 marker 共用同一名单。其余五组不抽样。', '',
          'Ridge 固定技术配色：CODEX 蓝色、CyTOF 绿色、IMC 橙色、CITE-seq 紫色、AbSeq 品红、MIBI 青色、mIF 玫瑰色；TNBC 第二个 IMC 用浅橙色区分。UMAP 保留原配色，因此左右两块分别设置平台图例。Ridge 图例从左到右对应曲线从上到下，括号内为展示单位数。', '',
          'Melanoma 2 的 UMAP 继续展示原 story 的 40,000 个固定细胞，ridge 保留完整 165,686 个细胞；其他六页的 UMAP 展示范围不变。Lung story 与原 Lung benchmark 分别使用 Zheng mIF 和 Monkman CODEX，不能混用。', '',
          '共享脚本、统一色表、原文件哈希和 QA 总表位于 `ridge_style_update_20260922/`。各新版目录包含脚本、PNG/SVG/PDF、caption、抽样名单及核验记录。', '',
          '## 保留的上一版本', '']
for c in PLAN:
    lines.append(f"- {c['name']}：`{Path(c['old']).relative_to(HERE.parent)}`。")
lines += ['', '`data_consistency_audit_20260922/` 保留此前数据范围审计；Figure 3 的分析更新与本次展示调整分别记录。', '',
          '展示原则：同一 benchmark 的所有方法使用相同展示细胞/抽样规则和统一绘图设置；平台与细胞类型 UMAP 共用坐标。对齐拟合、展示和评分范围分别记录。', '']
(HERE / 'README_NEXT.md').write_text('\n'.join(lines))
