from pathlib import Path
import hashlib,json,re,shutil,csv
HERE=Path(__file__).resolve().parent;STAGE=HERE/'manuscript_stage';OUT=HERE/'Supplementary_Overleaf'
assert not OUT.exists()
OUT.mkdir()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cp(src,rel):
 dst=OUT/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);assert sha(dst)==sha(src)
for name in ['supplementary_material.tex','supplementary_captions.tex','SUPPLEMENTARY_CAPTIONS.md']:
 cp(STAGE/name,name)
source=json.loads((STAGE/'SUPPLEMENTARY_FIGURE_SOURCES.json').read_text())
assert [x['number'] for x in source]==[f'S{i}' for i in range(1,14)]
for x in source:
 assert sha(STAGE/x['destination'])==x['sha256'];cp(STAGE/x['destination'],x['destination'])
shutil.copytree(STAGE/'supplementary_data',OUT/'supplementary_data')
cp(STAGE/'supplementary_material.pdf','reference/supplementary_material.pdf')
cp(STAGE/'SUPPLEMENTARY_FIGURE_SOURCES.json','reference/SUPPLEMENTARY_FIGURE_SOURCES.json')
s=(STAGE/'template.tex').read_text();table=s[s.index(r'\section*{Supplementary table}'):s.rindex(r'\end{document}')].strip()
(OUT/'reference/supplementary_table_s1.tex').write_text('% Already included in template.tex; not input by supplementary_material.tex.\n% Requires booktabs if included in another document.\n'+table+'\n')
notes=OUT/'author_notes';notes.mkdir()
(notes/'SUPPLEMENTARY_REVIEW_ZH.md').write_text('''# 附录当前范围及检查结论

2026-09-22 更新。S1–S13 编号和数据范围完整，正文引用与文件一致，无已引用但缺失的附图。

- S1–S7：七套 benchmark，UMAP 与逐样本 marker 分布。
- S8–S9：正文 Figure 2 的 23 项主报告指标。
- S10–S11：七套 annotation benchmark、三个模型分别展示。
- S12–S13：六套 missing-marker benchmark、三个模型，Spearman / MAE 分页。

依据用户本轮决定，暂不新增逐 marker 的预测图、Lung/HNSCC 全患者验证分布。Figure 3 D–F 已从正文图、图注及关联论述移除，也未追加为附录。原始资源实验数据仍保留在 source-data 的历史目录并明确标注。

公开方法名称统一为 Harmony。旧方法标识符仅作为内部数据追溯键保留，名称映射见 supplementary_data/method_display_names.csv。除显示名称外，现有附图所有数值及样式保持不变。

Supplementary Table S1 仍位于主稿末尾；reference/ 内保留与主稿一致的源码，独立附录不重复排版。
''')
count=sum(p.is_file() for p in (OUT/'supplementary_data').rglob('*'))
(OUT/'README.md').write_text(f'''# RAMP supplementary material — Overleaf upload package

上传同目录旁的 **Supplementary_Overleaf.zip** 建立独立 Overleaf 项目。将 Main document 设为 **supplementary_material.tex**，使用 **pdfLaTeX**。保持文件夹结构，不需要主稿 template.tex、bib 或 bst。

也可将本文件夹内容加入已有主稿项目，切换 main document 后编译附录。

| 文件 | 用途 |
|---|---|
| supplementary_material.tex | 编译入口 |
| supplementary_captions.tex | 实际编译的图注 |
| SUPPLEMENTARY_CAPTIONS.md | 可读图注；编辑 MD 不自动改变 TeX |
| figure/supplementary/ | S1–S13 的 13 张必需 PDF |
| supplementary_data/ | {count} 个数据、协议、名称映射及说明文件，不参与排版 |
| reference/supplementary_material.pdf | 已核对的 17 页参考 PDF：4 页图注及 13 张图 |
| reference/supplementary_table_s1.tex | 仍位于主稿末尾的 Table S1 源码；此处不重复排版 |
| reference/SUPPLEMENTARY_FIGURE_SOURCES.json | 原始来源记录，绝对路径不是编译依赖 |
| author_notes/SUPPLEMENTARY_REVIEW_ZH.md | 当前范围和检查记录 |
| FILE_MANIFEST.csv | 全部文件的尺寸及 SHA-256 |
| COMPILATION_CHECK.json | 独立编译验证 |

本版公开名称统一为 Harmony。内部方法键及原始路径保留，统计数值不变。没有新增 S14 等附图。D–F 的旧资源实验数据保留为归档，不作为主图或附图展示。数据目录中的旧 31 指标审计记录保留；附图继续只报告正文的 23 项指标。

当前包与 bioRxiv_Ver1 同步于 2026-09-22，含本轮 Figure 1/3 调整后的相关文字。后续更新不会自动同步到上传快照。
''')
for dep in re.findall(r'\\(?:input|includepdf)(?:\[[^\]]*\])?\{([^}]+)\}',(OUT/'supplementary_material.tex').read_text()):
 p=OUT/dep
 if not p.suffix:p=p.with_suffix('.tex')
 assert p.is_file(),p
check=HERE/'upload_compile_check';shutil.copytree(OUT,check)
print('Packaged',count,'source-data files and 13 figures; independent compile directory:',check)
