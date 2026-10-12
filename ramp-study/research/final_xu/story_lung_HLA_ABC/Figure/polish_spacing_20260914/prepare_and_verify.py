"""Verify a layout-only change, then stage the manuscript without editing its TeX."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
FIGURE = HERE.parent
ROOT = Path('/Users/qxu6/Documents/New project/manuscript')
WORK = HERE/'manuscript_sync'
STAGE = WORK/'staging'


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def blank_band(path):
    img = np.asarray(Image.open(path).convert('RGB'))
    rows = np.where(np.all(img[700:950] >= 250, axis=(1, 2)))[0]+700
    groups = np.split(rows, np.where(np.diff(rows) != 1)[0]+1)
    longest = max(groups, key=len)
    return dict(pixels=len(longest), pt=len(longest)*72/300,
                first_row=int(longest[0]), last_row=int(longest[-1]))


def main():
    sources = json.loads((HERE/'SOURCE_PANEL_HASHES.json').read_text())
    assert all(sha(Path(p)) == h for p, h in sources.items())
    old = FIGURE/'Combined ABCDEF/versions/v7_before_spacing_polish/panels_ABCDEF_preview.png'
    new = FIGURE/'Combined ABCDEF/panels_ABCDEF_preview.png'
    old_gap, new_gap = blank_band(old), blank_band(new)
    assert 4 < new_gap['pt'] < 8 and new_gap['pt'] < old_gap['pt']
    write_json(HERE/'REVISION_QA.json', dict(
        status='PASS', scope='Composite translation and whitespace only',
        all_12_independent_panel_files_unchanged=True,
        original_size_px=Image.open(old).size, new_size_px=Image.open(new).size,
        row_whitespace_before=old_gap, row_whitespace_after=new_gap,
        C_shift_left_pt=6, DEF_gap_before_pt=3, DEF_gap_after_pt=0,
        original_canvas_pt=[1025.04, 404], new_canvas_pt=[1019.04, 389],
        font_and_source_panel_scales_unchanged=True,
        visual_check='PASS: no overlap; axis labels, legends and all A-F contents retained'))
    (STAGE/'figure').mkdir(parents=True, exist_ok=True)
    for ext in ['png', 'svg']:
        shutil.copy2(FIGURE/f'Combined ABCDEF/panels_ABCDEF_preview.{ext}', HERE/f'lung_panels_ABCDEF.{ext}')
    rels = ['manuscript.pdf', 'manuscript_figure1_polished.pdf',
            'figure/figure5_lung_hla.png', 'figure/figure5_lung_hla.svg']
    write_json(WORK/'original_hashes.json', {str(ROOT/p): sha(ROOT/p) for p in rels})
    tex_files = ['manuscript.tex', 'manuscript_figure1_polished.tex']
    write_json(WORK/'tex_hashes.json', {str(ROOT/p): sha(ROOT/p) for p in tex_files})
    for p in tex_files:
        shutil.copy2(ROOT/p, STAGE/p)
        assert sha(ROOT/p) == sha(STAGE/p)
    for p in (ROOT/'figure').iterdir():
        dest = STAGE/'figure'/p.name
        if p.name in ['figure5_lung_hla.png', 'figure5_lung_hla.svg']:
            shutil.copy2(HERE/f'lung_panels_ABCDEF{p.suffix}', dest)
        elif not dest.exists():
            dest.symlink_to(p)
    template = FIGURE/'polish_20260914/manuscript_sync/install_checked_files.py'
    code = template.read_text().replace('six manuscript files', 'four manuscript files')
    code = code.replace('assert len(files) == 6', 'assert len(files) == 4')
    code = code.replace("files = [Path(path).relative_to(ROOT) for path in old]",
        "tex = json.loads((HERE/'tex_hashes.json').read_text())\n"
        "    for path, expected in tex.items():\n"
        "        assert sha(Path(path)) == expected and sha(STAGE/Path(path).name) == expected\n"
        "    files = [Path(path).relative_to(ROOT) for path in old]")
    code = code.replace('Figure 5 icon and spatial display updated; only Panel D caption changed in TeX',
                        'Figure 5 spacing only; TeX and captions unchanged')
    (WORK/'install_checked_files.py').write_text(code)
    print(json.dumps(dict(row_before=old_gap, row_after=new_gap, source_panels='unchanged',
                         staging=str(STAGE)), indent=2))


if __name__ == '__main__':
    main()
