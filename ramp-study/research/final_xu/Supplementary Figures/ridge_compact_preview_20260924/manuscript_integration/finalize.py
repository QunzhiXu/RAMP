from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,re,csv,difflib
import fitz
HERE=Path(__file__).resolve().parent;PREVIEW=HERE.parent;ROOT=PREVIEW.parent.parent;LIVE=ROOT.parent/'bioarxiv_Ver2'
stage=HERE/'stage';backup=HERE/'backup_before_revision';build=HERE/'build'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
write=lambda p,obj:p.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
figures=json.loads((HERE/'FIGURES.json').read_text());localpaths=json.loads((HERE/'LOCAL_PATHS.json').read_text())
qa0=json.loads((PREVIEW/'QA.json').read_text())
assert all(x['all_original_counts_and_bandwidths_identical'] and x['non_target_pixels_identical'] for x in qa0)
assert sum(x['density_hashes_verified'] for x in qa0)==3636
log=(build/'main.log').read_text();assert not re.search(r'Warning|Overfull|Float too large',log)
assert (build/'main.bbl').read_bytes()==(ROOT/'figure_raw_labels_20260924/build/main.bbl').read_bytes()
aux=(build/'main.aux').read_text();assert len(re.findall(r'\\bibcite\{',aux))==63
labels={k:dict(number=n,page=int(p)) for k,n,p in re.findall(r'\\newlabel\{(fig:[^}]+)\}\{\{([^}]+)\}\{(\d+)\}',aux)}
check=json.loads((backup/'COMPILATION_CHECK.json').read_text());assert labels==check['labels']
a=fitz.open(backup/'main.pdf');b=fitz.open(build/'main.pdf');assert len(a)==len(b)==64
changed=[i+1 for i in range(len(b)) if a[i].get_text()!=b[i].get_text()]
assert changed==[34,48,52,53,54,55,56,57,58],changed
for i,page in enumerate(b,1):
    for block in page.get_text('dict')['blocks']:
        for line in block.get('lines',[]):
            for span in line['spans']:
                x0,y0,x1,y1=span['bbox'];assert min(x0,y0)>=-1 and x1<=page.rect.width+1 and y1<=page.rect.height+1,(i,span['text'])
for page in [5,10,59,64]:
    a[page-1].get_text('dict');b[page-1].get_text('dict')
    assert a[page-1].get_pixmap(alpha=False).samples==b[page-1].get_pixmap(alpha=False).samples,page
for item in figures:assert sha(stage/item['destination'])==item['sha256']
(HERE/'main_tex.diff').write_text(''.join(difflib.unified_diff((backup/'main.tex').read_text().splitlines(True),(stage/'main.tex').read_text().splitlines(True),fromfile='previous/main.tex',tofile='updated/main.tex')))
qa=dict(pages=64,no_compile_warnings=True,reference_count=63,bibliography_unchanged=True,labels_unchanged=True,all_page_text_within_bounds=True,changed_page_text=changed,representative_unchanged_pages_pixel_verified=[5,10,59,64],density_hashes_verified=3636,accepted_figure_pdfs_exact=True,overleaf_zip_updated=False)
write(HERE/'MANUSCRIPT_QA.json',qa)
prov=json.loads((backup/'PROVENANCE.json').read_text())
for f in figures:
    r=next(x for x in prov['supplementary'] if x['number']==f['number'])
    r.update(previous_source=r['source'],previous_sha256=r['sha256'],source=f['source'],sha256=f['sha256'],change='Accepted compact ridge display windows; original density estimates, cell selection, colors and UMAPs retained.')
prov['figure1_scope']='Compact display windows accepted by user; RAMP SOX10 left limit -1.5. All cells retained in unchanged density estimation. Selected supplementary rows also use compact windows; see supplementary_compact_ridge_revision_20260924.'
rev=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),status='accepted_and_deployed',source=str(PREVIEW),backup=str(backup),scope={f['number']:[m.replace('Harmony-direct','Harmony') for m in f['targets']] for f in figures},window_rule='Union of displayed sample densities >=0.5% of own peak, padded by 4% of span within unchanged full histogram grid.',manual_override='S3 RAMP SOX10 left limit -1.5, as accepted for Figure 1.',cell_coverage_percent='99.48-100',all_density_estimates_unchanged=True,other_rows_and_umaps_unchanged=True,manuscript_text_changes=['Ridge plots Methods paragraph','Common supplementary display conventions'],compiled_pdf_sha256=sha(build/'main.pdf'),revised_tex_sha256=sha(stage/'main.tex'),overleaf_zip_updated=False)
prov['supplementary_compact_ridge_revision_20260924']=rev
write(stage/'PROVENANCE.json',prov)
check.pop('supplementary_figure_pages_pixels_identical_except_raw_label',None)
check.update(revision='Accepted S1-S7 compact ridge display windows, 2026-09-24',changed_figure_sources=[f['destination'] for f in figures],main_tex_unchanged=False,manuscript_body_methods_and_captions_unchanged=False,text_changes_only=rev['manuscript_text_changes'],visual_review_detail_pages=[34,48,54,58],visual_review_contact_pages=list(range(52,59)),visual_review_findings='Accepted previews copied exactly; Methods and common supplement caption updated with compact-window scope and unchanged density estimation. All page text stays within bounds; no changed pagination. Figure 1, Figure 3, S8 and S13 rendered identically to prior build.',backup=str(backup),compiled_pdf_sha256=sha(build/'main.pdf'),revised_tex_sha256=sha(stage/'main.tex'),density_estimates_verified=3636,all_other_figure_source_files_unchanged=True,all_source_data_files_unchanged=True)
write(stage/'COMPILATION_CHECK.json',check)
readme=(backup/'README.md').read_text()+f'\n\n## S1–S7 ridge 显示范围更新（2026-09-24）\n\n已采用用户确认的紧凑横轴预览；更新 Harmony（S1、S2、S4–S7）和 RAMP（S3、S5–S7），其余行保持原显示范围。密度仍由全部展示细胞计算，未改变数据、带宽、配色或 UMAP。Methods 与附图共同说明已同步注明规则及范围外低密度尾部。正文主图和 S8–S13 保持不变；合并 PDF 仍为 64 页，数字引用正常。Overleaf ZIP 未更新。\n\n最新图源和核验：`{PREVIEW}`。本次旧版备份：`{backup}`。\n'
(stage/'README.md').write_text(readme)
local={}
outroot=HERE/'local_docs_next';outroot.mkdir(exist_ok=True)
def save_local(rel,content):
    p=outroot/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content);local[rel]=str(p)
sup=(backup/'local_docs/Supplementary Figures/README.md').read_text()
sup=sup.replace('S1–S7 最新 PDF/PNG 位于 [figure_raw_labels_20260924/output](../figure_raw_labels_20260924/output/)，图内行名统一为 **Raw**，保留此前完整密度范围；','S1–S7 最新 PDF/PNG/SVG 位于 [ridge_compact_preview_20260924](ridge_compact_preview_20260924/)，已正式采用用户确认的紧凑横轴；图内行名仍为 **Raw**；')
sup=sup.replace('底层绘图脚本保存在各目录的 `ridge_full_range_20260922/`，其输出经文字修改脚本统一 Raw 名称；','完整范围底稿保存在各目录的 `ridge_full_range_20260922/`；当前紧凑显示脚本和输出在 `ridge_compact_preview_20260924/S1` 至 `S7`；')
sup=sup.replace('本轮修复将完整表达范围纳入直方图，并在两端增加至少 4 个平滑带宽的空间。','此前修复已将完整表达范围纳入直方图，并在两端增加至少 4 个平滑带宽的空间。当前版本保留这些密度估计，只收紧指定行的显示范围。')
sup=sup.replace('极端长尾可能使主峰在完整线性横轴上显得较窄。','Harmony（S1、S2、S4–S7）和 RAMP（S3、S5–S7）采用各样本峰高 0.5% 的范围并集，再向两侧各加该跨度的 4%；S3 RAMP SOX10 左限为 −1.5。其余行仍显示完整网格。窗口覆盖 99.48%–100% 的展示观测值；窗口外极低密度尾部仍参与计算。')
for f in figures:
    old=f"[ridge_full_range_20260922](<{ROOT}/Supplementary Figures/{f['case']}/ridge_full_range_20260922>)"
    sup=sup.replace(old,f"[{f['number']} 当前图](<{PREVIEW}/{f['number']}>)")
sup=sup.replace('当前共享脚本、范围审计、原文件哈希和 QA 总表位于 `ridge_full_range_update_20260922/`；','当前显示范围及不变性核验见 `ridge_compact_preview_20260924/`；此前完整范围修复、原文件哈希及 QA 总表位于 `ridge_full_range_update_20260922/`；')
sup=sup.replace('## 正式附录入口（2026-09-22）','## 历史独立附录入口（2026-09-22）').replace('当前 `bioRxiv_Ver1/supplementary_material.pdf` 包含','历史 `bioRxiv_Ver1/supplementary_material.pdf` 包含')
sup+='\n当前正式入口为 `../bioarxiv_Ver2/main.tex` 与 `main.pdf`（相对于 final_xu 的上一级项目目录）；正文与附录合并，S1–S7 在第 52–58 页。本轮同步图、Methods 和图注并完成编译检查，旧版备份位于 `ridge_compact_preview_20260924/manuscript_integration/backup_before_revision/`。\n'
save_local('Supplementary Figures/README.md',sup)
save_local('Supplementary Figures/CURRENT_PUBLICATION.json',json.dumps(prov['supplementary'],indent=2,ensure_ascii=False)+'\n')
for f in figures:
    rel=f"Supplementary Figures/{f['case']}/CURRENT_PREVIEW.json"
    d=json.loads((backup/'local_docs'/rel).read_text());d.update(previous_folder=d['folder'],version='ridge_compact_20260924',folder=str(PREVIEW/f['number']),pdf=f['source'],png=str(PREVIEW/'output/png'/f"{f['number'].lower()}_full_preview.png"),status='accepted_and_deployed_to_bioarxiv_Ver2',revision='Accepted compact display windows in specified method rows; all density estimates, cell selection, colors and UMAPs unchanged.',ridge_outline='Original full-grid density; compact viewport may exclude very low tails without adding artificial zero endpoints.')
    save_local(rel,json.dumps(d,indent=2,ensure_ascii=False)+'\n')
rel=str((PREVIEW/'README.md').relative_to(ROOT));s=(backup/'local_docs'/rel).read_text()
s=s.replace('Preview only; no manuscript, Overleaf ZIP, current-publication pointer or previous figure was changed.','Accepted and incorporated into `bioarxiv_Ver2/main.tex` and `main.pdf` on 2026-09-24. The initial preview stage did not change publication files; the subsequent authorized integration updated S1–S7 and the display descriptions in Methods and the common supplementary caption. The combined PDF remains 64 pages. Overleaf ZIP and all main figures are unchanged. Integration records and old-file backups are under `manuscript_integration/`.')
save_local(rel,s)
replacements={f['destination']:f['source'] for f in figures}
replacements.update({'main.pdf':str(build/'main.pdf')})
for name in ['main.tex','README.md','PROVENANCE.json','COMPILATION_CHECK.json']:replacements[name]=str(stage/name)
with (backup/'FILE_MANIFEST.csv').open(newline='') as f:rows=list(csv.DictReader(f))
for row in rows:
    p=Path(replacements.get(row['relative_path'],str(LIVE/row['relative_path'])))
    row.update(bytes=p.stat().st_size,sha256=sha(p))
with (stage/'FILE_MANIFEST.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['relative_path','bytes','sha256']);w.writeheader();w.writerows(rows)
replacements['FILE_MANIFEST.csv']=str(stage/'FILE_MANIFEST.csv')
write(HERE/'INSTALL_PLAN.json',dict(manuscript=replacements,local=local))
print(json.dumps(qa,indent=2))
