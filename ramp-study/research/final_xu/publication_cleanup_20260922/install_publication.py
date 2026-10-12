from pathlib import Path
from datetime import datetime
import hashlib,json,shutil,sys
HERE=Path(__file__).resolve().parent;STAGE=HERE/'manuscript_stage';LIVE=HERE.parent.parent/'bioRxiv_Ver1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
originals=json.loads((HERE/'original_live_hashes.json').read_text())
shutil.copy2(HERE/'QA.json',STAGE/'PUBLICATION_QA_20260922.json')
paths=[]
for p in STAGE.iterdir():
 if p.is_file() and (p.suffix in ['.tex','.md','.json'] or p.name in ['template.pdf','supplementary_material.pdf','template.bbl']):paths.append(p)
for folder in ['figure','supplementary_data']:paths.extend(p for p in (STAGE/folder).rglob('*') if p.is_file() and p.name!='.DS_Store')
changes=[]
for src in paths:
 rel=str(src.relative_to(STAGE));dst=LIVE/rel
 if dst.exists() and sha(src)==sha(dst):continue
 if dst.exists():assert sha(dst)==originals[rel],f'Live file changed since staging: {rel}'
 changes.append({'relative_path':rel,'old_sha256':sha(dst) if dst.exists() else None,'new_sha256':sha(src),'bytes':src.stat().st_size})
(HERE/'INSTALL_PLAN.json').write_text(json.dumps(changes,indent=2)+'\n')
if '--install' not in sys.argv:
 print(json.dumps({'changed_files':len(changes),'paths':[x['relative_path'] for x in changes]},indent=2));raise SystemExit
# Backup exact files first, then publish vetted copies. Old upload folder is retained intact.
backup=LIVE/'_backups'/('publication_cleanup_20260922_'+datetime.now().strftime('%H%M%S'))
backup.mkdir(parents=True,exist_ok=False)
for row in changes:
 rel=row['relative_path'];dst=LIVE/rel
 if dst.exists():
  b=backup/rel;b.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(dst,b);assert sha(b)==row['old_sha256']
for name in ['Supplementary_Overleaf.zip']:
 p=LIVE/name
 if p.exists():shutil.copy2(p,backup/name)
# Copy new package completely before changing the active directory.
tmp=LIVE/'Supplementary_Overleaf_new_20260922';assert not tmp.exists()
shutil.copytree(HERE/'Supplementary_Overleaf',tmp)
old=LIVE/'Supplementary_Overleaf'
if old.exists():old.rename(backup/'Supplementary_Overleaf')
tmp.rename(old)
for row in changes:
 rel=row['relative_path'];dst=LIVE/rel;dst.parent.mkdir(parents=True,exist_ok=True)
 shutil.copy2(STAGE/rel,dst);assert sha(dst)==row['new_sha256']
shutil.copy2(HERE/'Supplementary_Overleaf.zip',LIVE/'Supplementary_Overleaf.zip')
for p in (HERE/'Supplementary_Overleaf').rglob('*'):
 if p.is_file():assert sha(p)==sha(LIVE/'Supplementary_Overleaf'/p.relative_to(HERE/'Supplementary_Overleaf'))
assert sha(HERE/'Supplementary_Overleaf.zip')==sha(LIVE/'Supplementary_Overleaf.zip')
receipt=dict(destination=str(LIVE),backup=str(backup),changed_files=changes,upload_zip_sha256=sha(LIVE/'Supplementary_Overleaf.zip'),upload_folder_verified=True)
(HERE/'INSTALL_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'installed_files':len(changes),'backup':str(backup),'upload_package_updated':True},indent=2))
