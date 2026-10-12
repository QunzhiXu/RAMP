from pathlib import Path
import hashlib,json,shutil,csv,zipfile
HERE=Path(__file__).resolve().parent;LIVE=HERE.parent.parent/'bioarxiv_Ver2'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
before=json.loads((HERE/'BEFORE_HASHES.json').read_text());repl=json.loads((HERE/'INSTALL_PLAN.json').read_text())
for rel,h in before.items():assert sha(LIVE/rel)==h,('Concurrent edit',rel)
for rel,source in repl.items():
    assert (LIVE/rel).resolve().is_relative_to(LIVE)
    if rel not in before:assert not (LIVE/rel).exists(),rel
    assert Path(source).is_file(),source
for rel,source in repl.items():shutil.copy2(source,LIVE/rel)
for rel,h in before.items():
    if rel not in repl:assert sha(LIVE/rel)==h,rel
with (LIVE/'FILE_MANIFEST.csv').open(newline='') as f:
    for r in csv.DictReader(f):
        p=LIVE/r['relative_path'];assert p.stat().st_size==int(r['bytes']) and sha(p)==r['sha256'],r['relative_path']
with zipfile.ZipFile(LIVE/'bioarxiv_Ver2_Overleaf.zip') as z:
    assert z.testzip() is None
    for item in json.loads((HERE/'PACKAGE_MANIFEST.json').read_text()):
        assert sha(LIVE/item['path'])==item['sha256'],item['path']
result=dict(status='complete',archive=str(LIVE/'bioarxiv_Ver2_Overleaf.zip'),archive_bytes=(LIVE/'bioarxiv_Ver2_Overleaf.zip').stat().st_size,contents='main manuscript + supplementary figures and legends + associated data',main_document='main.tex',compiler='pdfLaTeX',pages=64,source_tex_and_pdf_unchanged=True,backup=str(HERE/'backup_before_revision'))
(HERE/'DEPLOYMENT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
