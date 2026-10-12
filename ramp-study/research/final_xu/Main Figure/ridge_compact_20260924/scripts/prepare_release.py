from pathlib import Path
from datetime import datetime,timezone
import json,re,hashlib,csv,shutil
import fitz
ROOT=Path(__file__).resolve().parents[1];LIVE=ROOT.parents[1].parent/'bioarxiv_Ver2';MAIN=ROOT.parent
stage=ROOT/'stage';build=ROOT/'build';backup=ROOT/'backup_before_revision'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
log=(build/'main.log').read_text();assert not re.search(r'Warning|Overfull|Float too large',log)
aux=(build/'main.aux').read_text();assert len(re.findall(r'\\bibcite\{',aux))==63
oldbbl=ROOT.parents[1]/'figure3_readability_preview_20260924/manuscript_integration/build/main.bbl'
assert (build/'main.bbl').read_bytes()==oldbbl.read_bytes()
labels=json.loads((ROOT/'LABELS.json').read_text());doc=fitz.open(build/'main.pdf')
for n,p in enumerate(doc,1):
 for b in p.get_text('dict')['blocks']:
  for l in b.get('lines',[]):
   for s in l['spans']:
    x0,y0,x1,y1=s['bbox'];assert min(x0,y0)>=-1 and x1<=p.rect.width+1 and y1<=p.rect.height+1,(n,s['text'])
refpage=next(i+1 for i,p in enumerate(doc) if any(s['text'].strip()=='References' and s['size']>13 and s['bbox'][0]<100 for b in p.get_text('dict')['blocks'] for l in b.get('lines',[]) for s in l['spans']))
check=json.loads((backup/'COMPILATION_CHECK.json').read_text())
check.update(revision='Figure 1 compact ridge view, RAMP SOX10 left limit -1.5, 2026-09-24',total_pages=len(doc),
 main_and_references_pages=47,labels=labels,reference_list_start_page=refpage,compiled_pdf_sha256=sha(build/'main.pdf'),revised_tex_sha256=sha(stage/'main.tex'),
 changed_figure_sources=['figure/figure1_ramp_overview.pdf'],all_other_figure_source_files_unchanged=True,
 manuscript_body_methods_and_captions_unchanged=False,main_tex_unchanged=False,
 text_changes_only=['Figure 1 caption display wording','Ridge plots Methods paragraph'],
 visual_review_detail_pages=[4,5,6,10,34],visual_review_contact_pages=[34,43,47,48],
 visual_review_findings='Figure 1 and caption fit page 5. Only ridge axes and tick pixels changed; all other Figure 1 pixels identical. Reviewed neighboring pages, retained Figure 3, new Methods wording and reflowed references. No clipping, overlap or float overflow. Supplementary figure pages pixel-identical.',
 backup=str(backup),ramp_sox10_left_limit=-1.5,figure3_source_unchanged=True,
 supplementary_figure_pages_pixel_identical=True)
check.pop('supplementary_pages_pixel_identical',None)
(stage/'COMPILATION_CHECK.json').write_text(json.dumps(check,indent=2)+'\n')
prov=json.loads((backup/'PROVENANCE.json').read_text())
prov['figure1']={'source':str(ROOT/'figure_1_melanoma.pdf'),'sha256':sha(ROOT/'figure_1_melanoma.pdf'),
 'previous_source':prov['figure1']['source'],'previous_sha256':prov['figure1']['sha256']}
prov['figure1_scope']='Compact display windows accepted by user; RAMP SOX10 left limit -1.5. All cells retained in unchanged density estimation. Supplementary ridges retain full-range axes.'
prov['figure1_compact_ridge_revision_20260924']={'timestamp_utc':datetime.now(timezone.utc).isoformat(),
 'backup':str(backup),'source':str(ROOT/'figure_1_melanoma.pdf'),'range_table':str(ROOT/'display_ranges.csv'),
 'ramp_sox10_left_limit':-1.5,'density_arrays_bitwise_identical':True,'umap_and_a_c_d_pixels_identical':True,
 'text_changes':check['text_changes_only'],'overleaf_zip_updated':False,'compiled_pdf_sha256':sha(build/'main.pdf')}
(stage/'PROVENANCE.json').write_text(json.dumps(prov,indent=2,ensure_ascii=False)+'\n')
readme=(backup/'README.md').read_text().replace('Figure 1 已采用完整密度范围的新版本。','Figure 1 已采用紧凑 ridge 显示范围，RAMP SOX10 左界为 −1.5。')
readme=readme.replace('暂未同步本轮引用格式和 Figure 3 排版','暂未同步本轮引用格式及 Figure 1、3 排版')
readme=readme.replace('总页数仍为 64。','该次更新后总页数为 64。')
readme+='''\n\n## Figure 1 ridge 显示范围（2026-09-24）\n\n已采用预览中认可的紧凑范围，RAMP SOX10 左界设为 −1.5。Figure 1 位于第 5 页，Figure 3 保留在第 10 页。当前合并 PDF 共 64 页。全部细胞仍用于原密度估计，UMAP 与 A/C/D 面板不变。Figure 1 caption 和 Ridge plots Methods 已同步；附图尚未改动。Overleaf ZIP 保持原样。\n\n图源、范围记录和备份：\n`/Users/qxu6/Documents/New project/final_xu/Main Figure/ridge_compact_20260924`\n'''
(stage/'README.md').write_text(readme)
# Keep the figure-folder entry point and detailed captions in sync as well.
(ROOT/'backup_before_revision/local_figure_docs').mkdir(exist_ok=True)
for name in ['README.md','FIGURE_CAPTION.md']:shutil.copy2(MAIN/name,backup/'local_figure_docs'/name)
rootreadme=(MAIN/'README.md').read_text().replace('更新：2026-09-22','更新：2026-09-24').replace('publication_display_20260922/','ridge_compact_20260924/')
rootreadme=rootreadme.replace('当前版本同步至 `../../bioRxiv_Ver1/`。','当前版本同步至 `../../bioarxiv_Ver2/`。')
rootreadme+='\nFigure 1B 现用紧凑显示窗口，RAMP SOX10 左界 −1.5；保留全部细胞的原密度估计和带宽。显示范围、尾部数量及像素一致性检查见 ridge_compact_20260924。附图尚未改动。\n'
(ROOT/'ROOT_README_NEXT.md').write_text(rootreadme)
cap=(MAIN/'FIGURE_CAPTION.md').read_text()
old='Ridges use pooled method–marker 0.5th–99.5th percentile display ranges, 160-bin histograms and Gaussian smoothing with sigma 6 bins;'
new='All cells enter full-range histograms; bin width is the pooled method–marker 0.5th–99.5th percentile span divided by 160, with Gaussian smoothing sigma 6 bins. Compact display windows follow the union of curve support above 0.5% of each curve’s peak, padded by 4% of that span at each end within the full histogram grid; the RAMP SOX10 left limit is −1.5. Each panel retains at least 99.8% of observed values in view, while all observations contribute to density estimation;'
assert old in cap;cap=cap.replace(old,new)
old='曲线使用 pooled 0.5%–99.5% 范围、160-bin 直方图及 sigma=6 bins 的高斯平滑；'
new='全部细胞进入完整范围的直方图，bin 宽度为 pooled 0.5%–99.5% 跨度除以 160，高斯 sigma 保持 6 bins。紧凑显示窗口覆盖任一曲线达到自身峰高 0.5% 的范围，两侧各留该跨度 4% 的缓冲，且不超出完整直方图网格；RAMP SOX10 左界为 −1.5。每图显示至少 99.8% 的实测值，全部细胞仍参与密度估计；'
assert old in cap;cap=cap.replace(old,new)
(ROOT/'FIGURE_CAPTION.md').write_text(cap);(ROOT/'ROOT_CAPTION_NEXT.md').write_text(cap)
(ROOT/'README.md').write_text('''# Figure 1 — compact ridge display\n\nAccepted 2026-09-24. Full outputs: `figure_1_melanoma.pdf`, `.png`, `.svg`. Panel previews are in `Panel A`–`Panel D`. Rebuild with `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 build_figure.py`.\n\n`display_ranges.csv` records all 12 chosen windows. RAMP SOX10 left limit is −1.5; all other windows match the preceding compact preview. All 60,000 cells / 46 units enter unchanged histograms with original bandwidth. All 552 density hashes match the previous full-range version. Only viewports and x ticks change. Some low tails lie outside the display; curves are not forced to zero at the boundary.\n\n`QA.json` confirms UMAP and A/C/D pixels are unchanged. `ridge_display_audit.csv` records outside-view counts and boundary density; histogram-grid boundaries remain distinct from display limits.\n\n`FIGURE_CAPTION.md` holds the detailed English/Chinese caption and `FIGURE1_CAPTION.tex` the current manuscript caption. Manuscript backup and compilation are retained in this folder. Overleaf ZIP is not updated.\n''')
# Manifest uses only final file versions; no staging ZIP is created.
with (backup/'FILE_MANIFEST.csv').open(newline='') as f:rows=list(csv.DictReader(f))
repl={'main.pdf':build/'main.pdf','figure/figure1_ramp_overview.pdf':ROOT/'figure_1_melanoma.pdf'}
for name in ['main.tex','README.md','COMPILATION_CHECK.json','PROVENANCE.json']:repl[name]=stage/name
for row in rows:
 p=repl.get(row['relative_path'],LIVE/row['relative_path']);row.update(bytes=p.stat().st_size,sha256=sha(p))
with (stage/'FILE_MANIFEST.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['relative_path','bytes','sha256']);w.writeheader();w.writerows(rows)
print(json.dumps({'pages':len(doc),'figure1_page':labels['fig:overview']['page'],'figure3_page':labels['fig:trainability']['page'],'references_start':refpage,'zip_updated':False},indent=2))
