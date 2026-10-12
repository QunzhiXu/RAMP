"""Publish the reviewed v21 figure and rebuilt manuscripts, preserving backups."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil

JOB = Path(__file__).resolve().parent
DEST = Path('/Users/qxu6/Documents/New project/manuscript')
STAGE = JOB / 'stage'
TARGETS = ['figure/figure3_cd19.png', 'figure/figure3_cd19.svg', 'figure/figure3_cd19.pdf',
           'manuscript.tex', 'manuscript.pdf',
           'manuscript_figure1_polished.tex', 'manuscript_figure1_polished.pdf']


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


before = json.loads((JOB / 'destination_before_sha256.json').read_text())
qa = json.loads((JOB / 'qa/PDF_CHECK.json').read_text())
assert all(r['pages'] == 18 and r['changed_render_pages'] == [6]
           and r['caption_unchanged'] and r['no_overfull_boxes'] for r in qa.values())
for name, digest in before.items():
    assert sha(DEST / name) == digest, f'Destination changed since backup: {name}'
for name in TARGETS:
    assert (STAGE / name).is_file()
    assert sha(JOB / 'backup' / name) == before[name]
for stem in ['manuscript', 'manuscript_figure1_polished']:
    original = (JOB / 'backup' / f'{stem}.tex').read_text()
    updated = (STAGE / f'{stem}.tex').read_text()
    assert updated.replace('{figure3_cd19.pdf}', '{figure3_cd19.png}') == original
for extension in ['png', 'svg', 'pdf']:
    source = JOB.parent / 'figure_layout_v21_polish' / f'cd19_story_three_row_v21_polish.{extension}'
    assert sha(source) == sha(STAGE / 'figure' / f'figure3_cd19.{extension}')
for name in TARGETS:
    shutil.copy2(STAGE / name, DEST / name)
    assert sha(STAGE / name) == sha(DEST / name)
for name, digest in before.items():
    if name not in TARGETS:
        assert sha(DEST / name) == digest, f'Unrelated file changed: {name}'
audit = {'status': 'PASS', 'time_utc': datetime.now(timezone.utc).isoformat(),
    'destination': str(DEST), 'source_figure_version': 'v21',
    'updated_sha256': {name: sha(DEST / name) for name in TARGETS},
    'backup': str(JOB / 'backup'), 'pages': 18, 'figure3_page': 6, 'caption_page': 7,
    'figure3_embedded_as': 'vector PDF', 'all_manuscript_prose_and_captions_unchanged': True,
    'only_rendered_page_changed_in_each_manuscript': 6,
    'unrelated_manuscript_files_unchanged': True}
(JOB / 'PUBLISH_AUDIT.json').write_text(json.dumps(audit, indent=2)+'\n')
print('PASS: v21 PNG/SVG/PDF and both manuscript TEX/PDF variants replaced and verified.')
print('Figure 3 is on page 6; caption remains on page 7; both documents contain 18 pages.')
print('Backup:', JOB / 'backup')
