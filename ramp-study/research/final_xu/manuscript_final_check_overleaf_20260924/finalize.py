from pathlib import Path
from datetime import datetime,timezone
import json,csv,hashlib
HERE=Path(__file__).resolve().parent;LIVE=HERE.parent.parent/'bioarxiv_Ver2';backup=HERE/'backup_before_revision'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
qa=json.loads((HERE/'OVERLEAF_PACKAGE_CHECK.json').read_text())
assert qa['status']=='passed' and qa['all_64_pages_pixel_identical_to_canonical']
notes='''# 最终图注、正文与 Overleaf 核对 — 2026-09-24

本次核对范围为当前 7 张主图、13 张附图与相应正文、Methods、图注的一致性，以及上传包的编译完整性。没有重算科学结果，也没有重跑模型或重新核查外部文献。

## 正文和图注

未发现需要继续修改的版本同步问题，因此 `main.tex` 和 `main.pdf` 保持原样。

- Figure 1 已为 melanoma 示例；当前 CD3/CD8/SOX10 ridge、artifact 示例、percentile shift 及 Methods 描述相符。
- Figure 3 只保留 A-C；七组 annotation 与六组 missing-marker、三模型平均、评价尺度及 S10-S13 引用相符，没有继续描述已移除的 D-F。
- CD19 图注对应同一 H14 区域的 DAPI/CD19、CD3 holdout、连续 CD19 显示和整体 identity AUROC。
- Melanoma F 已描述 AUROC 与 top-20% enrichment；Lung 使用 relative score，并明确 synthetic spatial display 以及新的样本量和结局；HNSCC 描述当前 protein/RNA donor summaries 与 LAG3 spatial correspondence。
- S1-S7 的 Raw 变换、显示单位、技术配色、compact window 范围、SOX10 左限及密度计算范围已对应当前图；S8-S13 的指标数和模型/benchmark 拆分不变。
- 数字引用、20 个图标签及附图交叉引用均正常。

## 修正的随包说明

1. 根 README 移除旧的“ZIP 暂不更新”状态，写明完整正文加附录及正确编译入口。
2. `supplementary_data/README.md` 移除对已删除 Supplementary Table S1 的旧指引，改为 Methods 与 CSV 数据。
3. `supplementary_data/ridge_display/README.md` 区分旧的 full-grid 密度审计和当前 compact viewport；添加当前 30 个显示范围及不变性核验文件。未更改旧密度数据、指标或预测。
4. `supplementary_data/model_specific_figures/README.md` 修正不存在的源清单路径，指向随包图和 `main.tex`。

## 上传包检查

- `bioarxiv_Ver2_Overleaf.zip`：227 个文件，约 42.2 MB。
- 包含 `main.tex`、`bibliography.bib`、`ramp_numeric.bst`、20 个图、203 个随附数据/记录文件及 README。
- 不含旧 bibliography style、历史 ZIP、临时编译文件或重复的成品 PDF。
- 由 ZIP 解压目录独立运行 pdfLaTeX/BibTeX，64 页编译成功，没有警告、缺失引用或外部项目文件依赖。
- 所有 64 页文字、页面尺寸和渲染像素与正式 `main.pdf` 相同。Figure 1、Figure 3、附图图注及代表附图已视觉检查。
- 检查记录：`OVERLEAF_PACKAGE_CHECK.json`；旧 ZIP 与说明文件均已备份。

本检查确认当前稿件与上传文件的一致性，不替代作者对外部数据、研究解释及投稿信息的最终确认。
'''
(HERE/'FINAL_CAPTION_CONTENT_CHECK_ZH.md').write_text(notes)
prov=json.loads((backup/'PROVENANCE.json').read_text())
prov['final_overleaf_package_20260924']=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),archive='bioarxiv_Ver2_Overleaf.zip',archive_sha256=qa['archive_sha256'],archive_files=qa['archive_files'],source_data_and_record_files=qa['source_data_and_record_files'],main_tex_unchanged=True,main_pdf_unchanged=True,all_figures_unchanged=True,scientific_scores_unchanged=True,documentation_updates=['root README','supplementary-data README: removed obsolete S1 table reference','ridge audit README and new compact-window records','model-specific figure README: corrected source pointer'],verification='OVERLEAF_PACKAGE_CHECK.json',review='FINAL_CAPTION_CONTENT_CHECK_ZH.md',backup=str(backup))
(HERE/'PROVENANCE_NEXT.json').write_text(json.dumps(prov,indent=2,ensure_ascii=False)+'\n')
check=json.loads((backup/'COMPILATION_CHECK.json').read_text())
check.update(revision='Final caption/content synchronization check and complete Overleaf archive, 2026-09-24',changed_figure_sources=[],all_source_data_files_unchanged=False,existing_numeric_analysis_data_unchanged=True,added_display_audit_files=['supplementary_data/ridge_display/COMPACT_WINDOWS_20260924.csv','supplementary_data/ridge_display/COMPACT_WINDOWS_QA_20260924.json'],manuscript_body_methods_and_captions_unchanged=True,main_tex_unchanged=True,text_changes_only=[],supplementary_figure_pages_pixel_identical=True,overleaf_zip_updated=True,overleaf_zip_status='Rebuilt at user request; independently compiled from extracted upload files.',overleaf_zip_sha256=qa['archive_sha256'],overleaf_validation='OVERLEAF_PACKAGE_CHECK.json',all_64_pages_pixel_identical_to_canonical=True,visual_review_detail_pages=[5,10,48],visual_review_contact_pages=[52,54,58],visual_review_findings='Final archive build preserves all 64 pages, 20 figure labels, 63 bibliography entries and current captions. All rendered pages are pixel-identical to canonical PDF. No missing project dependencies or compile warnings.',backup=str(backup))
(HERE/'COMPILATION_CHECK_NEXT.json').write_text(json.dumps(check,indent=2)+'\n')
repl=json.loads((HERE/'DOC_REPLACEMENTS.json').read_text())
repl.update({'README.md':str(HERE/'README_NEXT.md'),'PROVENANCE.json':str(HERE/'PROVENANCE_NEXT.json'),'COMPILATION_CHECK.json':str(HERE/'COMPILATION_CHECK_NEXT.json'),'FINAL_CAPTION_CONTENT_CHECK_ZH.md':str(HERE/'FINAL_CAPTION_CONTENT_CHECK_ZH.md'),'OVERLEAF_PACKAGE_CHECK.json':str(HERE/'OVERLEAF_PACKAGE_CHECK.json'),'bioarxiv_Ver2_Overleaf.zip':str(HERE/'bioarxiv_Ver2_Overleaf.zip')})
with (backup/'FILE_MANIFEST.csv').open(newline='') as f:rows={r['relative_path']:r for r in csv.DictReader(f)}
for rel,source in repl.items():
    p=Path(source);rows[rel]=dict(relative_path=rel,bytes=p.stat().st_size,sha256=sha(p))
with (HERE/'FILE_MANIFEST_NEXT.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['relative_path','bytes','sha256']);w.writeheader();w.writerows(rows[k] for k in sorted(rows))
repl['FILE_MANIFEST.csv']=str(HERE/'FILE_MANIFEST_NEXT.csv')
(HERE/'INSTALL_PLAN.json').write_text(json.dumps(repl,indent=2)+'\n')
print('Ready to install final archive and documentation only; no main text, figure or PDF replacements.')
