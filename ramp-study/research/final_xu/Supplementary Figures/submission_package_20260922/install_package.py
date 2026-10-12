"""Install reviewed supplementary material into the authorized manuscript copy."""
from pathlib import Path
import datetime
import hashlib
import json
import shutil

HERE = Path(__file__).resolve().parent
STAGE = HERE / 'bioRxiv_Ver1_stage'
LIVE = HERE.parents[2] / 'bioRxiv_Ver1'

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

checks = json.loads((HERE / 'BUILD_CHECKS.json').read_text())
assert checks['supplementary_material']['pages'] == 9
assert not checks['supplementary_material']['final_log_warnings']
assert not checks['template']['final_log_warnings']
assert checks['visual_review_complete']
for path, expected in json.loads((HERE / 'LIVE_BEFORE.json').read_text()).items():
    assert sha(LIVE / path) == expected, f'Live file changed: {path}'
files = ['template.tex', 'template.pdf', 'README.md', 'supplementary_material.tex',
         'supplementary_material.pdf', 'supplementary_captions.tex', 'SUPPLEMENTARY_CAPTIONS.md',
         'SUPPLEMENTARY_FIGURE_SOURCES.json']
files += [s['destination'] for s in json.loads((STAGE / 'SUPPLEMENTARY_FIGURE_SOURCES.json').read_text())]
backup = LIVE / '_backups' / ('supplementary_figures_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
backup.mkdir(parents=True)
installed = {}
for name in files:
    target = LIVE / name
    if target.exists():
        previous = backup / name
        previous.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, previous)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(STAGE / name, target)
    assert sha(target) == sha(STAGE / name)
    installed[name] = sha(target)
record = dict(live=str(LIVE), backup=str(backup), installed_sha256=installed,
              supplementary_pages=9, main_pages=checks['template']['pages'])
(HERE / 'INSTALL_VERIFY.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(dict(installed=len(files), live=str(LIVE), backup=str(backup)), indent=2))
