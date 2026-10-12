from pathlib import Path
import json,hashlib,shutil,datetime
U=Path(__file__).resolve().parents[1];src=U/'bioRxiv_Ver1';dst=Path('/Users/qxu6/Documents/New project/bioRxiv_Ver1')
files=['template.tex','template.pdf','figure/figure6_lung_hla.png','README.md','FIGURE6_CAPTION.md','LUNG_UPDATE_20260921.md','LUNG_UPDATE_CHECKS.json']
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
before=json.loads((U/'MANUSCRIPT_BEFORE.json').read_text())
checks=json.loads((U/'PRE_INSTALL_CHECKS.json').read_text());assert checks['status']=='PASS'
# Do not overwrite any concurrent user changes.
for name in files:
 target=dst/name
 if name in before:assert target.exists() and sha(target)==before[name],f'Live file changed: {name}'
 else:assert not target.exists(),f'New file already exists: {name}'
for name,h in before.items():
 if name.startswith('figure/') and name not in files:assert sha(dst/name)==h
backup=dst/'_backups/lung_senosain140k_20260921';backup.mkdir(parents=True,exist_ok=False)
for name in files:
 target=dst/name
 if target.exists():
  q=backup/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(target,q)
# Copy the reviewed files only; do not replace the figure directory.
for name in files:
 target=dst/name;target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_name(target.name+'.lung_update_tmp')
 assert not tmp.exists();shutil.copy2(src/name,tmp);tmp.replace(target);assert sha(target)==sha(src/name)
record={'status':'PASS','installed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'target':str(dst),'backup':str(backup),'files':{name:sha(dst/name) for name in files},'checks':checks}
(U/'FINAL_VERIFICATION.json').write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'status':'PASS','installed':files,'backup':str(backup)},indent=2))
