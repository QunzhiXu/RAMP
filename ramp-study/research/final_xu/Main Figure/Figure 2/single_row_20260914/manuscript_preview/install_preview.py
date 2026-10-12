"""Install only new, separately named preview files requested by the user."""
from pathlib import Path
import hashlib,json,shutil

p=Path(__file__).resolve().parent
destination=Path('/Users/qxu6/Documents/New project/manuscript')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
old=json.loads((p/'original_manuscript_hashes.json').read_text())
assert all(sha(path)==h for path,h in old.items()),'Original manuscript changed during preview build'
pairs=[
    (p/'build/manuscript_figure2_single_row.pdf',destination/'manuscript_figure2_single_row.pdf'),
    (p/'manuscript_figure2_single_row.tex',destination/'manuscript_figure2_single_row.tex'),
    (p/'figure2_benchmark_single_row.pdf',destination/'figure/figure2_benchmark_single_row.pdf'),
    (p.parent/'Combined/figure_2_single_row.png',destination/'figure/figure2_benchmark_single_row.png'),
    (p/'figure2_in_manuscript.png',destination/'figure2_single_row_page.png'),
    (p/'FIGURE2_SINGLE_ROW_PREVIEW.md',destination/'FIGURE2_SINGLE_ROW_PREVIEW.md'),
]
for source,target in pairs:
    assert source.is_file()
    assert not target.exists() or sha(source)==sha(target),f'Existing different preview must be preserved: {target}'
for source,target in pairs:
    if not target.exists():shutil.copy2(source,target)
    assert sha(source)==sha(target)
assert all(sha(path)==h for path,h in old.items())
(p/'INSTALL_VERIFY.json').write_text(json.dumps({'original_files_unchanged':len(old),
    'installed':{str(t):sha(t) for _,t in pairs},'figure2_pdf_page':4},indent=2)+'\n')
print('Installed separate preview. Original manuscript files unchanged.')
for _,target in pairs:print(target)
