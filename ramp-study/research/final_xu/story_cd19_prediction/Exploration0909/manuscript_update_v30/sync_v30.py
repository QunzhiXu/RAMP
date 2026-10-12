"""Stage, validate and install the approved same-ROI CD19 Figure 3."""
from pathlib import Path
import datetime
import difflib
import hashlib
import json
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
DEST = Path('/Users/qxu6/Documents/New project/manuscript')
FIG = HERE.parent / 'figure_layout_v30_polish'
STAGE = HERE / 'stage'
STEMS = ['manuscript', 'manuscript_figure1_polished']

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(name, obj):
    (HERE / name).write_text(json.dumps(obj, indent=2) + '\n')

def stage():
    originals = [p for p in DEST.iterdir() if p.is_file()]
    originals += [p for p in (DEST / 'figure').iterdir() if p.is_file()]
    snapshot = HERE / 'destination_before_sha256.json'
    assert not snapshot.exists(), 'Do not overwrite an existing snapshot'
    dump(snapshot.name, {str(p): sha(p) for p in originals})
    shutil.copytree(DEST / 'figure', STAGE / 'figure')
    for ext in ['pdf', 'png', 'svg']:
        shutil.copy2(FIG / f'cd19_story_three_row_v30_polish.{ext}',
                     STAGE / 'figure' / f'figure3_cd19.{ext}')
    old = ('\\textbf{(A)} Grayscale image of the failed observed CODEX CD19 channel. Scale\n'
           'bar, 50~\\textmu m.')
    new = ('\\textbf{(A)} Acquired CD19 from the registered H14 CODEX image, using the\n'
           'same 512~$\\times$~512-pixel field and orientation as G. Native 8-bit\n'
           'intensities are displayed linearly from 0 to 255. Scale bar, 50~\\textmu m.')
    for stem in STEMS:
        before = (DEST / f'{stem}.tex').read_text()
        assert before.count(old) == 1
        after = before.replace(old, new)
        (STAGE / f'{stem}.tex').write_text(after)
        (HERE / f'{stem}.diff').write_text(''.join(difflib.unified_diff(
            before.splitlines(keepends=True), after.splitlines(keepends=True),
            fromfile=f'{stem}:before', tofile=f'{stem}:v30')))
    print('Staged: Figure 3 assets and Panel A caption only.', flush=True)

def compile_all():
    for stem in STEMS:
        cmd = ['/Library/TeX/texbin/latexmk', '-pdf',
               '-pdflatex=/Library/TeX/texbin/pdflatex %O %S',
               '-interaction=nonstopmode', '-halt-on-error', f'{stem}.tex']
        with (HERE / f'{stem}_build.log').open('w') as log:
            subprocess.run(cmd, cwd=STAGE, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f'Compiled {stem}', flush=True)

def verify():
    import fitz
    qa = {}
    for stem in STEMS:
        old = fitz.open(DEST / f'{stem}.pdf')
        new = fitz.open(STAGE / f'{stem}.pdf')
        assert len(old) == len(new) == 18, (stem, len(old), len(new))
        changed = []
        # Render before text extraction for deterministic PyMuPDF comparison.
        for i in range(len(new)):
            a = old[i].get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
            b = new[i].get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
            if a.samples != b.samples:
                changed.append(i + 1)
        for page in changed:
            new[page - 1].get_pixmap(matrix=fitz.Matrix(1.6, 1.6), alpha=False).save(
                HERE / f'{stem}_page{page}.png')
        assert set(changed) <= {5, 6, 7}, changed
        assert 5 in changed
        text = ''.join(p.get_text() for p in new)
        assert 'Acquired CD19 from the registered H14 CODEX image' in text
        log = (STAGE / f'{stem}.log').read_text()
        assert 'Overfull' not in log
        assert 'undefined references' not in log
        qa[stem] = {'pages': len(new), 'changed_pages': changed,
                    'other_pages_pixel_identical': True,
                    'caption_updated': True, 'no_overfull_or_undefined_references': True}
    for ext in ['pdf', 'png', 'svg']:
        assert sha(FIG / f'cd19_story_three_row_v30_polish.{ext}') == sha(
            STAGE / 'figure' / f'figure3_cd19.{ext}')
    dump('QA.json', qa)
    print(json.dumps(qa, indent=2))

def install():
    original = json.loads((HERE / 'destination_before_sha256.json').read_text())
    assert all(sha(p) == h for p, h in original.items()), 'Manuscript changed during staging'
    qa = json.loads((HERE / 'QA.json').read_text())
    assert (HERE / 'VISUAL_QA.md').is_file(), 'Visual inspection is required'
    assert all(q['pages'] == 18 for q in qa.values())
    pairs = [(STAGE / f'{s}.{ext}', DEST / f'{s}.{ext}')
             for s in STEMS for ext in ['tex', 'pdf']]
    pairs += [(STAGE / 'figure' / f'figure3_cd19.{ext}',
               DEST / 'figure' / f'figure3_cd19.{ext}') for ext in ['pdf', 'png', 'svg']]
    pairs += [(HERE / 'manuscript_figure1_polished_page5.png', DEST / 'figure3_latest_page.png'),
              (HERE / 'FIGURE3_UPDATE.md', DEST / 'FIGURE3_UPDATE.md')]
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    backup = DEST / 'backups' / f'figure3_before_v30_{timestamp}'
    backup.mkdir(parents=True)
    records = []
    for source, target in pairs:
        assert source.is_file()
        old_hash = sha(target) if target.exists() else None
        if target.exists():
            saved = backup / target.relative_to(DEST)
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, saved)
            assert sha(saved) == old_hash
        records.append({'file': str(target), 'old_sha256': old_hash, 'new_sha256': sha(source)})
    record = {'status': 'backed_up', 'backup': str(backup), 'files': records, 'qa': qa}
    (backup / 'SYNC_MANIFEST.json').write_text(json.dumps(record, indent=2) + '\n')
    for source, target in pairs:
        shutil.copy2(source, target)
        assert sha(source) == sha(target)
    changed = {str(target) for _, target in pairs}
    untouched = {p: h for p, h in original.items() if p not in changed}
    assert all(sha(p) == h for p, h in untouched.items())
    record.update(status='PASS', unchanged_original_files=len(untouched))
    (backup / 'SYNC_MANIFEST.json').write_text(json.dumps(record, indent=2) + '\n')
    dump('INSTALL_VERIFY.json', record)
    print(json.dumps({'status': 'PASS', 'backup': str(backup), 'installed': len(pairs),
                      'unchanged_original_files': len(untouched)}, indent=2))

if __name__ == '__main__':
    {'stage': stage, 'compile': compile_all, 'verify': verify, 'install': install}[sys.argv[1]]()
