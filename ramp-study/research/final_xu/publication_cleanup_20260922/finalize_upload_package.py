from pathlib import Path
import hashlib,json,csv,re,zipfile,shutil
import fitz
HERE=Path(__file__).resolve().parent;STAGE=HERE/'manuscript_stage';OUT=HERE/'Supplementary_Overleaf';LIVE=HERE.parent.parent/'bioRxiv_Ver1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
# Preserve original provenance while explicitly registering publication-label edits.
audit=json.loads((HERE/'published_data_display_edits.json').read_text())
for x in audit:x.update(original_sha256=sha(LIVE/x['path']),published_sha256=sha(STAGE/x['path']))
manifest={'change':'Public display strings only; original analysis keys, source paths and numeric cells retained.', 'tables':audit, 'original_source_manifests':'Existing SOURCE_MANIFEST records describe the unmodified analysis products; current published-file hashes are recorded here and in the upload FILE_MANIFEST.csv.'}
for root in [STAGE,OUT]:
 (root/'supplementary_data/DISPLAY_UPDATE_20260922.json').write_text(json.dumps(manifest,indent=2)+'\n')
 p=root/'supplementary_data/README.md';p.write_text(p.read_text()+'\n\n2026-09-22 publication display update: labels now use **Harmony**. Internal method identifiers are unchanged; see `method_display_names.csv`. Original `SOURCE_MANIFEST` records describe the archived analysis products. Current edited-table hashes and unchanged-numeric checks are in `DISPLAY_UPDATE_20260922.json`.\n')
 p=root/'supplementary_data/model_specific_figures/README.md';p.write_text(p.read_text()+'\nPublic labels use Harmony. The original `SOURCE_MANIFEST.json` preserves the previous rendering provenance; current publication PDFs are listed in `SUPPLEMENTARY_FIGURE_SOURCES.json` at manuscript root (or `reference/` in the upload package). Score values are unchanged.\n')
count=sum(p.is_file() for p in (OUT/'supplementary_data').rglob('*'))
p=OUT/'README.md';p.write_text(p.read_text().replace('190 个数据',f'{count} 个数据'))
# Separate clean-directory compilation must match the reference at each page.
a=fitz.open(STAGE/'supplementary_material.pdf');b=fitz.open(HERE/'upload_compile_check/supplementary_material.pdf')
assert len(a)==len(b)==17
for i in range(len(a)):
 assert a[i].get_text()==b[i].get_text(),i
 assert a[i].get_pixmap(dpi=72).samples==b[i].get_pixmap(dpi=72).samples,i
log=(HERE/'upload_compile_check/supplementary_material.log').read_text(errors='replace')
assert not re.search(r'Overfull|undefined|LaTeX Error',log)
(OUT/'COMPILATION_CHECK.json').write_text(json.dumps({'engine':'pdfLaTeX via latexmk','pages':17,'figure_pages':13,'legend_pages':4,'standalone_dependency_check':True,'independent_compilation':True,'all_page_text_and_render_pixels_match_reference':True,'missing_references':False,'overfull_boxes':False,'source_data_files':count},indent=2)+'\n')
with (OUT/'FILE_MANIFEST.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['relative_path','bytes','sha256'])
 for p in sorted(OUT.rglob('*')):
  if p.is_file() and p.name!='FILE_MANIFEST.csv':w.writerow([str(p.relative_to(OUT)),p.stat().st_size,sha(p)])
z=HERE/'Supplementary_Overleaf.zip'
with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as zipf:
 for p in sorted(OUT.rglob('*')):
  if p.is_file():zipf.write(p,p.relative_to(OUT))
with zipfile.ZipFile(z) as zipf:
 assert zipf.testzip() is None
 for p in OUT.rglob('*'):
  if p.is_file():assert hashlib.sha256(zipf.read(str(p.relative_to(OUT)))).hexdigest()==sha(p)
print(json.dumps({'upload_files':sum(p.is_file() for p in OUT.rglob('*')),'zip_mb':round(z.stat().st_size/1e6,2),'standalone_compilation':'17 pages, pixel-identical to reference','zip_sha256':sha(z)},indent=2))
