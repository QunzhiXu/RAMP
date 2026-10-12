from pathlib import Path
import hashlib,json,shutil,csv
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;LIVE=ROOT.parent/'bioarxiv_Ver2'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((HERE/'INSTALL_PLAN.json').read_text())
before=json.loads((HERE/'BEFORE_HASHES.json').read_text())
for rel,h in before.items():assert sha(LIVE/rel)==h,('Concurrent manuscript edit',rel)
for rel in plan['local']:
    assert sha(ROOT/rel)==sha(HERE/'backup_before_revision/local_docs'/rel),rel
qa=json.loads((HERE/'MANUSCRIPT_QA.json').read_text())
assert qa['pages']==64 and qa['no_compile_warnings'] and qa['tex_unchanged']
for rel,source in plan['manuscript'].items():shutil.copy2(source,LIVE/rel)
for rel,h in before.items():
    if rel not in plan['manuscript']:assert sha(LIVE/rel)==h,rel
with (LIVE/'FILE_MANIFEST.csv').open(newline='') as f:
    for row in csv.DictReader(f):
        p=LIVE/row['relative_path']
        assert p.stat().st_size==int(row['bytes']) and sha(p)==row['sha256'],p
for rel,source in plan['local'].items():shutil.copy2(source,ROOT/rel)
report={'status':'deployed','labels_updated':8,'tex_and_captions_unchanged':True,
        'data_curves_and_ranges_unchanged':True,'overleaf_zip_unchanged':True,
        'manuscript':str(LIVE/'main.pdf'),'pages':64,'pdf_sha256':sha(LIVE/'main.pdf')}
(HERE/'DEPLOYMENT.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
