"""Publish the verified CD19 figure replacement to the user-requested manuscript."""
from pathlib import Path
import hashlib
import json
import shutil
from datetime import datetime, timezone

JOB = Path(__file__).resolve().parent
DEST = Path('/Users/qxu6/Documents/New project/manuscript')
STAGE = JOB / 'stage'
before = json.loads((JOB / 'destination_before_sha256.json').read_text())
targets = ['figure/figure3_cd19.png', 'figure/figure3_cd19.svg', 'figure/figure3_cd19.pdf',
           'manuscript.tex', 'manuscript.pdf',
           'manuscript_figure1_polished.tex', 'manuscript_figure1_polished.pdf']


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


for name in targets:
    dst = DEST / name
    assert (STAGE / name).is_file(), name
    if name in before:
        assert dst.is_file() and sha(dst) == before[name], f'Destination changed: {name}'
    else:
        assert not dst.exists(), f'Unexpected new destination: {name}'
qa = json.loads((JOB / 'qa/PDF_CHECK.json').read_text())
assert all(record['pages'] == 18 and record['rendered_pages'] == [6, 7] for record in qa.values())
for name in targets:
    shutil.copy2(STAGE / name, DEST / name)
    assert sha(STAGE / name) == sha(DEST / name)
for name, digest in before.items():
    if name not in targets:
        assert sha(DEST / name) == digest, f'Unrelated file changed: {name}'
report = {'status': 'PASS', 'time_utc': datetime.now(timezone.utc).isoformat(),
          'destination': str(DEST), 'source_figure_version': 'v17',
          'updated_sha256': {name: sha(DEST / name) for name in targets},
          'backup': str(JOB / 'backup'), 'pdf_pages': 18, 'figure3_page': 6,
          'caption_page': 7, 'updated_caption_panels': ['E', 'G'],
          'remaining_manuscript_text_unchanged': True,
          'unrelated_manuscript_files_unchanged': True}
(JOB / 'PUBLISH_AUDIT.json').write_text(json.dumps(report, indent=2))
print('PASS: Figure 3 PNG/SVG/PDF and both manuscript TEX/PDF files updated and verified.')
print('Figure 3: page 6; updated caption: page 7; 18 pages in both variants.')
print('Backup:', JOB / 'backup')
