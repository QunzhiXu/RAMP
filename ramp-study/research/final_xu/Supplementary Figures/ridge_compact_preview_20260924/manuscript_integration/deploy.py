from pathlib import Path
import hashlib,json,csv,shutil
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent.parent;LIVE=ROOT.parent/'bioarxiv_Ver2'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((HERE/'INSTALL_PLAN.json').read_text());before=json.loads((HERE/'BEFORE_HASHES.json').read_text())
qa=json.loads((HERE/'MANUSCRIPT_QA.json').read_text())
assert qa['pages']==64 and qa['no_compile_warnings'] and qa['accepted_figure_pdfs_exact']
for rel,h in before.items():assert sha(LIVE/rel)==h,('Concurrent edit',rel)
for rel in plan['local']:
    assert sha(ROOT/rel)==sha(HERE/'backup_before_revision/local_docs'/rel),('Concurrent local edit',rel)
for rel,source in plan['manuscript'].items():
    assert str((LIVE/rel).resolve()).startswith(str(LIVE)+'/')
    shutil.copy2(source,LIVE/rel)
for rel,h in before.items():
    if rel not in plan['manuscript']:assert sha(LIVE/rel)==h,('Unexpected change',rel)
for rel,source in plan['manuscript'].items():assert sha(LIVE/rel)==sha(Path(source)),rel
with (LIVE/'FILE_MANIFEST.csv').open(newline='') as f:
    for row in csv.DictReader(f):
        p=LIVE/row['relative_path'];assert p.stat().st_size==int(row['bytes']) and sha(p)==row['sha256'],p
for rel,source in plan['local'].items():shutil.copy2(source,ROOT/rel)
result=dict(status='deployed',manuscript=str(LIVE/'main.pdf'),pages=64,figures_updated='S1-S7',methods_and_caption_updated=True,all_main_figures_unchanged=True,s8_s13_unchanged=True,overleaf_zip_unchanged=True,pdf_sha256=sha(LIVE/'main.pdf'),backup=str(HERE/'backup_before_revision'))
(HERE/'DEPLOYMENT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
