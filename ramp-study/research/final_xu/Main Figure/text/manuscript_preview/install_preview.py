"""Copy the reviewed Figure 1 trial into the user-requested manuscript folder."""
from pathlib import Path
import hashlib
import json
import shutil

stage = Path(__file__).resolve().parent
target = Path('/Users/qxu6/Documents/New project/manuscript')
manifest = json.loads((stage / 'source_manifest.json').read_text())

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

assert target.is_dir()
assert all(sha(p) == s for p, s in manifest['source_sha256'].items())
files = {
    'figure/figure1_ramp_overview_polished.pdf': 'figure/figure1_ramp_overview_polished.pdf',
    'manuscript_figure1_polished.tex': 'manuscript_figure1_polished.tex',
    'manuscript_figure1_polished.pdf': 'manuscript_figure1_polished.pdf',
    'polished_page-02.png': 'figure1_polished_page.png',
    'figure1_before_after_in_manuscript.png': 'figure1_before_after_in_manuscript.png',
    'FIGURE1_POLISH_PREVIEW.md': 'FIGURE1_POLISH_PREVIEW.md',
}
for source, dest in files.items():
    src, dst = stage / source, target / dest
    assert src.is_file()
    if dst.exists():
        assert sha(src) == sha(dst), f'Existing file differs: {dst}'
for source, dest in files.items():
    src, dst = stage / source, target / dest
    if not dst.exists():
        shutil.copy2(src, dst)
    assert sha(src) == sha(dst)
    print(dst)
assert all(sha(p) == s for p, s in manifest['source_sha256'].items())
