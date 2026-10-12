"""Copy the verified, explicitly listed files; preserve every replaced file."""
from pathlib import Path
import hashlib
import json
import shutil

HERE=Path(__file__).resolve().parent
plan=json.loads((HERE/'PUBLISH_PLAN.json').read_text())
target=Path(plan['destination']);stage=Path(plan['stage'])
assert target==Path('/Users/qxu6/Documents/New project/bioRxiv_Ver1')
backup=target/'_backups/figure3_update_20260922'
assert not backup.exists(), 'Previous publication already exists; inspect before repeating.'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for row in plan['files']:
    rel=Path(row['relative']);assert not rel.is_absolute() and '..' not in rel.parts
    assert sha(stage/rel)==row['staged_sha256']
    dest=target/rel
    assert (sha(dest) if dest.exists() else None)==row['previous_sha256'],str(dest)
backup.mkdir(parents=True)
for row in plan['files']:
    rel=Path(row['relative']);dest=target/rel
    if dest.exists():
        old=backup/rel;old.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(dest,old)
    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(stage/rel,dest)
    assert sha(dest)==row['staged_sha256']
shutil.copy2(HERE/'PUBLISH_PLAN.json',backup/'PUBLISH_PLAN.json')
record=dict(status='complete',files=len(plan['files']),destination=str(target),backup=str(backup),
    figure_sha256=sha(target/'figure/figure3_cross_platform_learning.pdf'),
    manuscript_pdf_sha256=sha(target/'template.pdf'),manuscript_tex_sha256=sha(target/'template.tex'))
(HERE/'PUBLICATION.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
