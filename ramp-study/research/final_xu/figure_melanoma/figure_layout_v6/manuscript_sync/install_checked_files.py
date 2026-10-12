"""Back up four manuscript files and install the compiled, reviewed Melanoma update."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import os
import shutil

HERE = Path(__file__).resolve().parent
ROOT = Path('/Users/qxu6/Documents/New project/manuscript')
STAGE = HERE/'staging'


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    old = json.loads((HERE/'original_hashes.json').read_text())
    qa = json.loads((HERE/'COMPILE_QA.json').read_text())
    assert qa['Melanoma_pages_identical_between_manuscript_variants']
    for path, expected in json.loads((HERE/'tex_hashes.json').read_text()).items():
        assert sha(Path(path)) == expected, f'TeX changed since staging: {path}'
    for path, expected in old.items():
        assert sha(Path(path)) == expected, f'Original changed since staging: {path}'
    files = [Path(path).relative_to(ROOT) for path in old]
    assert len(files) == 4
    for rel in files:
        assert (STAGE/rel).is_file()
    timestamp = datetime.now(ZoneInfo('America/Chicago')).strftime('%Y%m%d_%H%M%S')
    backup = ROOT/'backups'/f'melanoma_before_{timestamp}'
    backup.mkdir(parents=True, exist_ok=False)
    # Complete and verify every backup before replacing any current file.
    for rel in files:
        target = backup/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/rel, target)
        assert sha(target) == old[str(ROOT/rel)]
    changes = []
    for rel in files:
        target = ROOT/rel
        temporary = target.with_name(target.name+'.melanoma_sync_tmp')
        assert not temporary.exists()
        shutil.copy2(STAGE/rel, temporary)
        assert sha(temporary) == sha(STAGE/rel)
        os.replace(temporary, target)
        assert sha(target) == sha(STAGE/rel)
        changes.append(dict(file=str(target), old_sha256=old[str(target)], new_sha256=sha(target)))
    audit = dict(status='PASS', backup=str(backup), timestamp=timestamp, files=changes,
                 figure_page=8, caption_page=9, pages=18,
                 source_figure=str(HERE.parent/'melanoma_three_row_redesign_v6.png'),
                 caption_scope='TeX and captions unchanged; only the Figure 4 icon updated')
    (backup/'SYNC_MANIFEST.json').write_text(json.dumps(audit, indent=2)+'\n')
    (HERE/'INSTALLED.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
