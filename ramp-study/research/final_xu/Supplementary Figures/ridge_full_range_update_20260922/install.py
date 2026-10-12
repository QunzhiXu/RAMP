from pathlib import Path
from datetime import datetime
import hashlib,json,shutil,sys
HERE=Path(__file__).resolve().parent;SUP=HERE.parent;LIVE=SUP.parent.parent/'bioRxiv_Ver1';STAGE=HERE/'stage'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert (HERE/'PUBLICATION_QA.json').is_file()
originals=json.loads((HERE/'original_live_hashes.json').read_text())
for rel,want in originals.items():assert sha(LIVE/rel)==want,rel
files=json.loads((HERE/'install_files.json').read_text())+['MAIN_OVERLEAF_CHECKS.json']
for rel in files:assert (STAGE/rel).is_file(),rel
for pkg in ['Main_Overleaf','Supplementary_Overleaf']:assert (HERE/(pkg+'.zip')).is_file()
if '--install' not in sys.argv:
    print('Ready: 7 figure PDFs, captions, compiled supplementary and main PDFs, Methods and both verified upload packages.');raise SystemExit
backup=LIVE/'_backups'/('ridge_full_range_20260922_'+datetime.now().strftime('%H%M%S'));backup.mkdir(parents=True)
for rel in files+['Main_Overleaf.zip','Supplementary_Overleaf.zip']:
    src=LIVE/rel
    if src.exists():
        dst=backup/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
for pkg in ['Main_Overleaf','Supplementary_Overleaf']:
    pending=LIVE/(pkg+'_pending_ridge_full_range');assert not pending.exists()
    shutil.copytree(HERE/pkg,pending)
    (LIVE/pkg).rename(backup/pkg);pending.rename(LIVE/pkg)
    shutil.copy2(HERE/(pkg+'.zip'),LIVE/(pkg+'.zip'))
for rel in files:shutil.copy2(STAGE/rel,LIVE/rel)
for rel in files:assert sha(LIVE/rel)==sha(STAGE/rel),rel
for pkg in ['Main_Overleaf','Supplementary_Overleaf']:
    assert sha(LIVE/(pkg+'.zip'))==sha(HERE/(pkg+'.zip'))
    for p in (HERE/pkg).rglob('*'):
        if p.is_file():assert sha(p)==sha(LIVE/pkg/p.relative_to(HERE/pkg))
localbackup=HERE/'prior_publication_pointers';localbackup.mkdir()
for name in ['README.md','CURRENT_PUBLICATION.json']:
    shutil.copy2(SUP/name,localbackup/name)
shutil.copy2(HERE/'README_NEXT.md',SUP/'README.md')
shutil.copy2(HERE/'CURRENT_PUBLICATION_NEXT.json',SUP/'CURRENT_PUBLICATION.json')
for item in json.loads((HERE/'REVISION_PLAN.json').read_text()):
    old,new=Path(item['old']),Path(item['new']);p=new.parent/'CURRENT_PREVIEW.json'
    shutil.copy2(p,localbackup/(item['name']+'_CURRENT_PREVIEW.json'))
    d=json.loads(p.read_text());s=json.dumps(d).replace(str(old),str(new)).replace('ridge_uniform_20260922','ridge_full_range_20260922');d=json.loads(s)
    d.update(previous_folder=str(old),revision='Complete density support and padded axes; unchanged bandwidth, cell/sample selection, colors and UMAP pixels',ridge_outline='Full-support density with endpoints at baseline')
    p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
receipt=dict(destination=str(LIVE),backup=str(backup),updated_files={rel:sha(LIVE/rel) for rel in files},zip_sha256={p:sha(LIVE/(p+'.zip')) for p in ['Main_Overleaf','Supplementary_Overleaf']},all_copies_verified=True)
(HERE/'INSTALL_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'backup':str(backup),'all_copies_verified':True,'supplementary_pdf':str(LIVE/'supplementary_material.pdf'),'supplementary_zip':str(LIVE/'Supplementary_Overleaf.zip')},indent=2))
