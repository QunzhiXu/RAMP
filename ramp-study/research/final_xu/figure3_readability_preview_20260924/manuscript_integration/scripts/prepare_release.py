from pathlib import Path
from datetime import datetime,timezone
import json,re,hashlib,csv
import fitz
ROOT=Path(__file__).resolve().parents[1]
DEST=Path('/Users/qxu6/Documents/New project/bioarxiv_Ver2')
stage=ROOT/'stage';build=ROOT/'build';backup=ROOT/'backup_before_revision'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
aux=(build/'main.aux').read_text();log=(build/'main.log').read_text()
labels={m[0]:{'number':m[1],'page':int(m[2])} for m in re.findall(r'\\newlabel\{(fig:[^}]+)\}\{\{([^}]+)\}\{(\d+)\}',aux)}
assert len(labels)==20 and labels['fig:trainability']['page']==10
warnings=[s for s in log.splitlines() if re.search(r'Warning|Overfull|Float too large',s)]
assert not warnings,warnings
assert len(re.findall(r'\\bibcite\{',aux))==63
assert sha(stage/'main.tex')==sha(backup/'main.tex')
oldbbl=DEST.parent/'final_xu/manuscript_numeric_citations_20260924/build/main.bbl'
assert (build/'main.bbl').read_bytes()==oldbbl.read_bytes()
doc=fitz.open(build/'main.pdf');comparison=json.loads((ROOT/'PAGE_COMPARISON.json').read_text())
for n,p in enumerate(doc,1):
 for b in p.get_text('dict')['blocks']:
  for l in b.get('lines',[]):
   for s in l['spans']:
    x0,y0,x1,y1=s['bbox']
    assert x0>=-1 and y0>=-1 and x1<=p.rect.width+1 and y1<=p.rect.height+1,(n,s['text'])
refpage=next(i+1 for i,p in enumerate(doc) if any(s['text'].strip()=='References' and s['size']>13 and s['bbox'][0]<100 for b in p.get_text('dict')['blocks'] for l in b.get('lines',[]) for s in l['spans']))
check={
 'revision':'Figure 3 readability layout, 2026-09-24',
 'total_pages':len(doc),'main_and_references_pages':47,'supplementary_legend_pages':4,'supplementary_figure_pages':13,
 'main_figures':7,'supplementary_figures':13,'one_tex_entry':True,'main_document':'main.tex','compiler':'pdfLaTeX','bibliography_engine':'BibTeX',
 'citation_style':'numeric, square brackets, sorted and compressed','bibliography_style':'ramp_numeric','bibliography_order':'first citation','reference_count':63,
 'bibliography_output_identical_to_previous_build':True,'no_compile_warnings':True,'missing_references':False,'duplicate_labels':False,
 'duplicate_hyperlink_destinations':False,'float_too_large':False,'overfull_boxes':False,'all_page_text_within_bounds':True,
 'changed_figure_sources':['figure/figure3_cross_platform_learning.pdf'],'all_other_figure_source_files_unchanged':True,
 'all_source_data_files_unchanged':True,'manuscript_body_methods_and_captions_unchanged':True,'main_tex_unchanged':True,
 'labels':labels,'reference_list_start_page':refpage,'compiled_pdf_sha256':sha(build/'main.pdf'),'revised_tex_sha256':sha(stage/'main.tex'),
 'supplementary_pages_pixel_identical':all(p in comparison['pixel_identical_pages'] for p in range(48,65)),
 'visual_review_detail_pages':[9,10,11,12,13,14,15],'visual_review_contact_pages':comparison['changed_pages'],
 'visual_review_findings':'Figure 3 and full caption fit page 10. Larger 8 pt labels and axes are legible at manuscript width. No clipping, overlaps, blank pages or float overflow. Text reflow inspected; supplementary pages unchanged.',
 'backup':str(backup),'overleaf_zip_updated':False,'overleaf_zip_status':'Retained unchanged at user request.'}
(stage/'COMPILATION_CHECK.json').write_text(json.dumps(check,indent=2)+'\n')
prov=json.loads((backup/'PROVENANCE.json').read_text())
prov['figure3_readability_revision_20260924']={
 'timestamp_utc':datetime.now(timezone.utc).isoformat(),'source':str(ROOT.parent/'output/pdf/figure3_readability_preview.pdf'),
 'destination':'figure/figure3_cross_platform_learning.pdf','sha256':sha(stage/'figure/figure3_cross_platform_learning.pdf'),
 'previous_sha256':sha(backup/'figure/figure3_cross_platform_learning.pdf'),'backup':str(backup),
 'layout':'A vertical workflow on left; B above C on right; two axes each and one shared legend.',
 'score_values_methods_benchmarks_and_axes_unchanged':True,'manuscript_text_and_caption_unchanged':True,
 'figure_page':10,'compiled_pdf_sha256':sha(build/'main.pdf'),'overleaf_zip_updated':False}
(stage/'PROVENANCE.json').write_text(json.dumps(prov,indent=2,ensure_ascii=False)+'\n')
readme=(backup/'README.md').read_text().replace('暂未同步本轮引用格式','暂未同步本轮引用格式和 Figure 3 排版')
readme+='''\n\n## Figure 3 排版更新（2026-09-24）\n\n已替换为左侧纵向 workflow、右侧 B/C 上下排列的版本，坐标数字和方法名约为 8 pt。当前合并 PDF 中位于第 10 页，图和完整 caption 同页；总页数仍为 64。统计结果、方法、benchmark、坐标范围和 caption 内容不变。Overleaf ZIP 暂不更新。\n\n可编辑图源与预览：\n`/Users/qxu6/Documents/New project/final_xu/figure3_readability_preview_20260924`\n\n替换前原图、PDF 和文件记录备份：\n`/Users/qxu6/Documents/New project/final_xu/figure3_readability_preview_20260924/manuscript_integration/backup_before_revision`\n'''
(stage/'README.md').write_text(readme)
with (backup/'FILE_MANIFEST.csv').open(newline='') as f:rows=list(csv.DictReader(f))
replacements={'figure/figure3_cross_platform_learning.pdf':stage/'figure/figure3_cross_platform_learning.pdf','main.pdf':build/'main.pdf'}
for name in ['README.md','PROVENANCE.json','COMPILATION_CHECK.json']:replacements[name]=stage/name
for row in rows:
 path=replacements.get(row['relative_path'],DEST/row['relative_path'])
 row.update(bytes=path.stat().st_size,sha256=sha(path))
with (stage/'FILE_MANIFEST.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=['relative_path','bytes','sha256']);writer.writeheader();writer.writerows(rows)
print(json.dumps({'pages':len(doc),'figure3_page':10,'reference_start_page':refpage,'warnings':warnings,'zip_updated':False},indent=2))
