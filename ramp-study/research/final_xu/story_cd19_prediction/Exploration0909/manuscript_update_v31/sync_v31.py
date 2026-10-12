"""Stage and verify Figure 3 v31; install with backups after visual review."""
from pathlib import Path
import datetime
import difflib
import hashlib
import json
import re
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEST = ROOT.parent / 'manuscript'
FIG = HERE.parent / 'figure_layout_v31_DAPI_pair'
STAGE = HERE / 'stage'
STEMS = ['manuscript', 'manuscript_figure1_polished']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def stage():
    assert not (HERE / 'destination_before_sha256.json').exists()
    originals = [p for p in DEST.iterdir() if p.is_file()]
    originals += [p for p in (DEST / 'figure').iterdir() if p.is_file()]
    dump('destination_before_sha256.json', {str(p): sha(p) for p in originals})
    shutil.copytree(DEST / 'figure', STAGE / 'figure')
    for ext in ['pdf', 'png', 'svg']:
        shutil.copy2(FIG / f'cd19_story_three_row_v31_DAPI_pair.{ext}',
                     STAGE / 'figure' / f'figure3_cd19.{ext}')
    old = ('\\textbf{(A)} Acquired CD19 from the registered H14 CODEX image, using the\n'
           'same 512~$\\times$~512-pixel field and orientation as G. Native 8-bit\n'
           'intensities are displayed linearly from 0 to 255. Scale bar, 50~\\textmu m.')
    new = ('\\textbf{(A)} Acquired DAPI (left) and CD19 (right) from the registered H14\n'
           'CODEX image, using the same 512~$\\times$~512-pixel field and orientation\n'
           'as G. DAPI provides morphology; CD19 retains native 8-bit intensities\n'
           'displayed linearly from 0 to 255. Scale bars, 50~\\textmu m.')
    for stem in STEMS:
        before = (DEST / f'{stem}.tex').read_text()
        assert before.count(old) == 1
        after = before.replace(old, new)
        (STAGE / f'{stem}.tex').write_text(after)
        (HERE / f'{stem}.diff').write_text(''.join(difflib.unified_diff(
            before.splitlines(keepends=True), after.splitlines(keepends=True),
            fromfile=f'{stem}:before', tofile=f'{stem}:v31')))
    rows = [
        ('1', 'RAMP overview', 'figure1_ramp_overview.pdf', 'Main Figure/icon_update_20260915'),
        ('2', 'Benchmark', 'figure2_benchmark.pdf', 'Main Figure/Figure 2/louvain_ari_fastmnn_20260914/Combined'),
        ('3', 'CD19 recovery', 'figure3_cd19.pdf', 'story_cd19_prediction/Exploration0909/figure_layout_v31_DAPI_pair'),
        ('4', 'Melanoma', 'figure4_melanoma.png', 'figure_melanoma/figure_layout_v8'),
        ('5', 'Lung HLA-ABC', 'figure5_lung_hla.png', 'story_lung_HLA_ABC/Figure/Combined ABCDEF'),
        ('6', 'HNSCC', 'figure6_hnscc.pdf', 'figure_HNSCC/big_figure/polish_20260913/CEF_D_colorbar_aligned_20260914'),
    ]
    readme = ('# Figure index\n\n'
              '[Manuscript PDF](manuscript.pdf) · [TeX source](manuscript.tex)\n\n'
              '| Figure | Manuscript figure | Source folder |\n'
              '|---|---|---|\n')
    for number, title, asset, source in rows:
        assert (STAGE / 'figure' / asset).is_file()
        assert (ROOT / source).is_dir()
        readme += f'| {number} | [{title}](figure/{asset}) | [Folder](<../final_xu/{source}/>) |\n'
    (STAGE / 'README.md').write_text(readme)
    (STAGE / 'FIGURE3_UPDATE.md').write_text(
        '# Figure 3 / CD19 update — v31\n\n'
        'Panel A now shows acquired DAPI and observed CD19 side by side, '
        'using the same ROI and orientation as Panel G. Panels B/C are compacted; '
        'D–H and all analysis results are unchanged. Both current manuscript '
        'sources and PDFs include the updated Panel A caption.\n\n'
        'Source: [Figure 3 v31](<../final_xu/story_cd19_prediction/Exploration0909/'
        'figure_layout_v31_DAPI_pair/>). '
        'Previous files are preserved under `backups/figure3_before_v31_*/`.\n')
    print('Staged Figure 3 v31, Panel A caption and concise figure index.', flush=True)


def compile_all():
    for stem in STEMS:
        command = ['/Library/TeX/texbin/latexmk', '-pdf',
                   '-pdflatex=/Library/TeX/texbin/pdflatex %O %S',
                   '-interaction=nonstopmode', '-halt-on-error', f'{stem}.tex']
        with (HERE / f'{stem}_build.log').open('w') as log:
            subprocess.run(command, cwd=STAGE, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f'Compiled {stem}', flush=True)


def verify():
    import fitz
    qa = {}
    for stem in STEMS:
        old = fitz.open(DEST / f'{stem}.pdf')
        new = fitz.open(STAGE / f'{stem}.pdf')
        assert len(old) == len(new) == 18, (stem, len(old), len(new))
        changed = [i + 1 for i in range(18)
                   if old[i].get_pixmap().samples != new[i].get_pixmap().samples]
        assert 5 in changed and set(changed) <= {5, 6}, changed
        for page in [5, 6]:
            new[page - 1].get_pixmap(matrix=fitz.Matrix(1.8, 1.8)).save(
                HERE / f'{stem}_page{page}.png')
        text = ' '.join(p.get_text() for p in new)
        assert 'Acquired DAPI (left) and CD19 (right)' in text
        log = (STAGE / f'{stem}.log').read_text()
        assert 'Overfull' not in log
        assert 'undefined references' not in log
        qa[stem] = {'pages': 18, 'changed_pages': changed,
                    'other_pages_pixel_identical': True,
                    'caption_updated': True, 'no_overfull_or_undefined_references': True}
    for asset in (DEST / 'figure').iterdir():
        if asset.is_file() and not asset.name.startswith('figure3_cd19.'):
            assert sha(asset) == sha(STAGE / 'figure' / asset.name)
    for ext in ['pdf', 'png', 'svg']:
        assert sha(FIG / f'cd19_story_three_row_v31_DAPI_pair.{ext}') == sha(
            STAGE / 'figure' / f'figure3_cd19.{ext}')
    for href in re.findall(r'\]\(([^\n]+?)\)', (STAGE / 'README.md').read_text()):
        target = href.strip('<>')
        test = (DEST / target) if target.startswith('../') else (STAGE / target)
        assert test.exists(), target
    qa['all_other_figure_assets_unchanged'] = True
    qa['readme_links_valid'] = True
    dump('QA.json', qa)
    print(json.dumps(qa, indent=2), flush=True)


def install():
    original = json.loads((HERE / 'destination_before_sha256.json').read_text())
    assert all(sha(p) == h for p, h in original.items()), 'Destination changed since staging'
    assert not (DEST / 'README.md').exists(), 'Unexpected README appeared'
    assert (HERE / 'VISUAL_QA.md').read_text().startswith('PASS\n')
    qa = json.loads((HERE / 'QA.json').read_text())
    pairs = [(STAGE / f'{s}.{ext}', DEST / f'{s}.{ext}')
             for s in STEMS for ext in ['tex', 'pdf']]
    pairs += [(STAGE / 'figure' / f'figure3_cd19.{ext}', DEST / 'figure' / f'figure3_cd19.{ext}')
              for ext in ['pdf', 'png', 'svg']]
    pairs += [(STAGE / name, DEST / name) for name in ['README.md', 'FIGURE3_UPDATE.md']]
    pairs += [(HERE / 'manuscript_page5.png', DEST / 'figure3_latest_page.png')]
    backup = DEST / 'backups' / ('figure3_before_v31_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    backup.mkdir(parents=True)
    records = []
    for source, target in pairs:
        before = sha(target) if target.exists() else None
        if target.exists():
            saved = backup / target.relative_to(DEST)
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, saved)
            assert sha(saved) == before
        records.append({'file': str(target), 'old_sha256': before, 'new_sha256': sha(source)})
    record = {'status': 'BACKED_UP', 'backup': str(backup), 'files': records, 'qa': qa}
    (backup / 'SYNC_MANIFEST.json').write_text(json.dumps(record, indent=2) + '\n')
    for source, target in pairs:
        temporary = target.with_name(target.name + '.v31_tmp')
        shutil.copy2(source, temporary)
        assert sha(temporary) == sha(source)
        temporary.replace(target)
        assert sha(target) == sha(source)
    changed = {str(target) for _, target in pairs}
    assert all(sha(p) == h for p, h in original.items() if p not in changed)
    record['status'] = 'PASS'
    (backup / 'SYNC_MANIFEST.json').write_text(json.dumps(record, indent=2) + '\n')
    dump('INSTALL_VERIFY.json', record)
    print(json.dumps({'status': 'PASS', 'backup': str(backup), 'installed_files': len(pairs)}, indent=2))


if __name__ == '__main__':
    {'stage': stage, 'compile': compile_all, 'verify': verify, 'install': install}[sys.argv[1]]()
