"""Collect the current supplementary submission without modifying the manuscript."""
from pathlib import Path
import csv
import hashlib
import json
import re
import shutil

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
SOURCE = PROJECT.parent / 'bioRxiv_Ver1'
OUT = HERE / 'Supplementary_Overleaf'
assert not OUT.exists(), 'Use a new version directory rather than overwriting an export.'
OUT.mkdir(parents=True)
copied = []

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def copy(source, destination):
    target = OUT / destination
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    assert sha(source) == sha(target)
    copied.append(dict(source=str(source), destination=destination, sha256=sha(source), bytes=source.stat().st_size))

for filename in ['supplementary_material.tex', 'supplementary_captions.tex', 'SUPPLEMENTARY_CAPTIONS.md']:
    copy(SOURCE / filename, filename)
sources = json.loads((SOURCE / 'SUPPLEMENTARY_FIGURE_SOURCES.json').read_text())
assert [s['number'] for s in sources] == [f'S{i}' for i in range(1, 14)]
for entry in sources:
    source = SOURCE / entry['destination']
    assert sha(source) == entry['sha256']
    copy(source, entry['destination'])
for source in sorted((SOURCE / 'supplementary_data').rglob('*')):
    if source.is_file() and source.name != '.DS_Store':
        copy(source, str(source.relative_to(SOURCE)))
copy(SOURCE / 'supplementary_material.pdf', 'reference/supplementary_material.pdf')
copy(SOURCE / 'SUPPLEMENTARY_FIGURE_SOURCES.json', 'reference/SUPPLEMENTARY_FIGURE_SOURCES.json')

# Preserve the separately located supplementary table without changing its placement.
main = (SOURCE / 'template.tex').read_text()
table = main[main.index(r'\section*{Supplementary table}'):main.rindex(r'\end{document}')].strip()
assert table.count(r'\begin{table}') == 1 and r'\label{tab:rank-sensitivity}' in table
(OUT / 'reference/supplementary_table_s1.tex').write_text(
    '% Extracted verbatim from the active main manuscript.\n'
    '% Already included in template.tex; not input by supplementary_material.tex.\n'
    '% Requires booktabs if included in another document.\n' + table + '\n')

dependencies = re.findall(r'\\(?:input|includepdf)(?:\[[^\]]*\])?\{([^}]+)\}',
                          (OUT / 'supplementary_material.tex').read_text())
for dep in dependencies:
    p = OUT / dep
    if not p.suffix:
        p = p.with_suffix('.tex')
    assert p.is_file(), dep
assert len(dependencies) == 14
assert not re.search(r'\\(?:bibliography|addbibresource|bibliographystyle)',
                     (OUT / 'supplementary_material.tex').read_text() +
                     (OUT / 'supplementary_captions.tex').read_text())

(OUT / 'README.md').write_text('''# RAMP supplementary material — Overleaf upload package

## 使用方式

直接上传同目录旁的 `Supplementary_Overleaf.zip` 建立独立 Overleaf 项目。
将 **Main document 设为 `supplementary_material.tex`**，使用 **pdfLaTeX** 编译。
保持文件夹结构；不需要主稿的 `template.tex`、`.bib` 或 `.bst`。

也可以把本文件夹内容加入已有主稿项目，并切换 main document 编译附录。
最终 PDF 为 17 页：4 页图注 + S1–S13 各一页。

## 内容

| 路径 | 用途 |
|---|---|
| `supplementary_material.tex` | 编译入口 |
| `supplementary_captions.tex` | 实际参与编译的图注 |
| `SUPPLEMENTARY_CAPTIONS.md` | 图注的可读/编辑副本；修改 MD 不会自动改变 TeX |
| `figure/supplementary/` | S1–S13，恰好 13 个必需 PDF |
| `supplementary_data/` | 当前稿件已有的全部 189 个附录数据、协议及说明文件；不参与 LaTeX 排版 |
| `reference/supplementary_material.pdf` | 已审核的编译参考结果 |
| `reference/supplementary_table_s1.tex` | 原样提取的 Supplementary Table S1 源码；当前仍在主稿末尾，此入口不重复排版 |
| `reference/SUPPLEMENTARY_FIGURE_SOURCES.json` | 原分析出处；绝对路径仅为出处记录，不是编译依赖 |
| `FILE_MANIFEST.csv` | 上传包内文件尺寸与 SHA-256 |
| `COMPILATION_CHECK.json` | 独立编译及与参考 PDF 的一致性核对 |

所有图及数值均来自当前 `bioRxiv_Ver1`，没有重新训练或改变结果。
数据目录中保留原有的 31 项指标审计表、旧 D–F 资源实验记录；新增图页仍只报告正文选定的 23 项指标。
旧 D–F 的适用范围已写在 `supplementary_data/trainability/README.md` 中。

本文件夹是上传快照；后续主稿更新不会自动同步到这里。创建日期：2026-09-22。
''')

(HERE / 'SOURCE_COPY_MANIFEST.json').write_text(json.dumps(dict(source=str(SOURCE), files=copied), indent=2) + '\n')
print(json.dumps(dict(output=str(OUT), figures=len(sources), copied_files=len(copied),
                      data_files=sum(p.is_file() for p in (OUT/'supplementary_data').rglob('*'))), indent=2))
