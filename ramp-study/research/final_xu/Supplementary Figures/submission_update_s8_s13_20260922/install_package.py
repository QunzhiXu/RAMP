"""Install the reviewed supplement, preserving every overwritten file."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import shutil

HERE = Path(__file__).resolve().parent
LIVE = HERE.parent.parent.parent / 'bioRxiv_Ver1'
STAGE = HERE / 'bioRxiv_Ver1_stage'

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

checks = json.loads((HERE / 'BUILD_CHECKS.json').read_text())
assert checks['status'] == 'passed' and checks['visual_review_complete']
before = json.loads((HERE / 'LIVE_BEFORE.json').read_text())
targets = json.loads((HERE / 'INSTALL_FILES.json').read_text())
assert set(targets) == set(before) == set(checks['staged_sha256'])
for name in targets:
    live, staged = LIVE / name, STAGE / name
    assert live.resolve().is_relative_to(LIVE.resolve())
    assert not live.is_symlink() and not staged.is_symlink()
    assert (sha(live) if live.exists() else None) == before[name], ('Concurrent change', name)
    assert sha(staged) == checks['staged_sha256'][name], ('Unreviewed stage change', name)

stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
backup = LIVE / '_backups' / f'supplement_s8_s13_{stamp}'
backup.mkdir(parents=True, exist_ok=False)
for name in targets:
    p = LIVE / name
    if p.exists():
        q = backup / name
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, q)
        assert sha(q) == before[name]
(backup / 'BEFORE_MANIFEST.json').write_text(json.dumps(before, indent=2) + '\n')
for name in targets:
    target = LIVE / name
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f'.install-{stamp}')
    assert not temporary.exists()
    shutil.copy2(STAGE / name, temporary)
    assert sha(temporary) == checks['staged_sha256'][name]
    temporary.replace(target)
installed = {name: sha(LIVE / name) for name in targets}
assert installed == checks['staged_sha256']
for source in json.loads((LIVE / 'SUPPLEMENTARY_FIGURE_SOURCES.json').read_text()):
    assert sha(LIVE / source['destination']) == source['sha256']
receipt = dict(status='installed_and_verified', installed_at=stamp, destination=str(LIVE),
               backup_directory=str(backup), installed_file_count=len(targets),
               installed_sha256=installed, supplement_pages=17, figures='S1-S13',
               new_figures='S8-S13', main_figure_assets_modified=False)
(HERE / 'INSTALL_VERIFY.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({k:v for k,v in receipt.items() if k != 'installed_sha256'}, indent=2))
