"""Install the user-requested, compiled Figure 3 update; leave TeX and ZIP unchanged."""
from pathlib import Path
import hashlib,json,shutil,csv
ROOT=Path(__file__).resolve().parents[1]
DEST=Path('/Users/qxu6/Documents/New project/bioarxiv_Ver2')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
before=json.loads((ROOT/'BEFORE_HASHES.json').read_text())
for rel,h in before.items():assert sha(DEST/rel)==h,f'Concurrent modification: {rel}'
check=json.loads((ROOT/'stage/COMPILATION_CHECK.json').read_text());assert check['no_compile_warnings']
replacements={'figure/figure3_cross_platform_learning.pdf':ROOT/'stage/figure/figure3_cross_platform_learning.pdf','main.pdf':ROOT/'build/main.pdf'}
for rel in ['README.md','COMPILATION_CHECK.json','PROVENANCE.json','FILE_MANIFEST.csv']:replacements[rel]=ROOT/'stage'/rel
for rel,src in replacements.items():shutil.copy2(src,DEST/rel)
for rel,h in before.items():
 if rel not in replacements:assert sha(DEST/rel)==h,rel
with (DEST/'FILE_MANIFEST.csv').open(newline='') as f:
 for row in csv.DictReader(f):
  p=DEST/row['relative_path'];assert p.stat().st_size==int(row['bytes']) and sha(p)==row['sha256'],p
report={'status':'deployed','manuscript':str(DEST/'main.pdf'),'figure3_page':10,'total_pages':64,'main_tex_unchanged':True,'overleaf_zip_unchanged':True,'updated_files':list(replacements),'pdf_sha256':sha(DEST/'main.pdf')}
(ROOT/'DEPLOYMENT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
