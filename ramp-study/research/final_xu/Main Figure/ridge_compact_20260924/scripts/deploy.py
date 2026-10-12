"""Deploy the accepted Figure 1 display revision while retaining the Overleaf ZIP."""
from pathlib import Path
import hashlib,json,shutil,csv
ROOT=Path(__file__).resolve().parents[1];MAIN=ROOT.parent;DEST=ROOT.parents[1].parent/'bioarxiv_Ver2'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
before=json.loads((ROOT/'BEFORE_HASHES.json').read_text())
for rel,h in before.items():assert sha(DEST/rel)==h,f'Concurrent manuscript change: {rel}'
for name in ['README.md','FIGURE_CAPTION.md']:
 assert sha(MAIN/name)==sha(ROOT/'backup_before_revision/local_figure_docs'/name),name
check=json.loads((ROOT/'stage/COMPILATION_CHECK.json').read_text());assert check['no_compile_warnings'] and check['total_pages']==64
repl={'figure/figure1_ramp_overview.pdf':ROOT/'figure_1_melanoma.pdf','main.pdf':ROOT/'build/main.pdf'}
for name in ['main.tex','README.md','COMPILATION_CHECK.json','PROVENANCE.json','FILE_MANIFEST.csv']:repl[name]=ROOT/'stage'/name
for rel,p in repl.items():shutil.copy2(p,DEST/rel)
for rel,h in before.items():
 if rel not in repl:assert sha(DEST/rel)==h,rel
with (DEST/'FILE_MANIFEST.csv').open(newline='') as f:
 for row in csv.DictReader(f):
  p=DEST/row['relative_path'];assert p.stat().st_size==int(row['bytes']) and sha(p)==row['sha256'],p
shutil.copy2(ROOT/'ROOT_README_NEXT.md',MAIN/'README.md')
shutil.copy2(ROOT/'ROOT_CAPTION_NEXT.md',MAIN/'FIGURE_CAPTION.md')
m=json.loads((ROOT/'manifest.json').read_text());m.update(manuscript_replaced=True,manuscript=str(DEST),overleaf_zip_updated=False)
(ROOT/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
report={'status':'deployed','manuscript':str(DEST/'main.pdf'),'figure1_page':5,'figure3_page':10,'pages':64,
 'ramp_sox10_left_limit':-1.5,'overleaf_zip_unchanged':True,'figure3_unchanged':True,'supplementary_sources_unchanged':True,
 'pdf_sha256':sha(DEST/'main.pdf'),'updated_manuscript_files':list(repl)}
(ROOT/'DEPLOYMENT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
