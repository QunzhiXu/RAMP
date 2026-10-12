"""Stage the approved icon/ROI replacement and the corresponding Lung D caption."""
from pathlib import Path
import hashlib,json,shutil,difflib

HERE=Path(__file__).resolve().parent
ROOT=Path('/Users/qxu6/Documents/New project/manuscript')
STORY=HERE.parents[1]
WORK=HERE/'manuscript_sync'
STAGE=WORK/'staging'

D_CAPTION=r'''\textbf{(D)} Measured HLA-A (left) and predicted HLA-ABC state (right) in a
previously selected 192~$\times$~192~\textmu m region of C\_30/P\_26
(389 cells; raw-cell Spearman $\rho=0.834$). Original positions and measured
nuclear geometry constrain shared synthetic cell shapes and DAPI background,
using independent bone-marrow nuclear textures; these are numerical cell
annotations, not acquired Zheng images. Measured HLA-A is patient-ranked;
predictions are unchanged. The previously selected shared color range is
0.25--0.55 (saturated endpoints); scale bars, 50~\textmu m. Across 693,693
cells (119 cores, 28 patients), median within-patient $\rho=0.692$; all core
correlations were positive (0.239--0.883). Metrics use unclipped cell values.
'''

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    (STAGE/'figure').mkdir(parents=True,exist_ok=True)
    src=STORY/'Figure/Combined ABCDEF'
    for ext in ['png','svg']:
        shutil.copy2(src/f'panels_ABCDEF_preview.{ext}',HERE/f'lung_panels_ABCDEF.{ext}')
    rels=[Path(p) for p in ['manuscript.tex','manuscript.pdf','manuscript_figure1_polished.tex',
          'manuscript_figure1_polished.pdf','figure/figure5_lung_hla.png','figure/figure5_lung_hla.svg']]
    hashes={str(ROOT/p):sha(ROOT/p) for p in rels}
    (WORK/'original_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    for p in (ROOT/'figure').iterdir():
        dest=STAGE/'figure'/p.name
        if p.name in ['figure5_lung_hla.png','figure5_lung_hla.svg']:
            shutil.copy2(HERE/f'lung_panels_ABCDEF{p.suffix}',dest)
        elif not dest.exists():dest.symlink_to(p)
    diffs=[]
    for name in ['manuscript.tex','manuscript_figure1_polished.tex']:
        old=(ROOT/name).read_text()
        start=old.index(r'\includegraphics[width=\textwidth]{figure5_lung_hla.png}')
        a=old.index(r'\textbf{(D)}',start);b=old.index(r'\textbf{(E)}',a)
        new=old[:a]+D_CAPTION+old[b:]
        (STAGE/name).write_text(new)
        assert new[:a]==old[:a] and new[new.index(r'\textbf{(E)}',a):]==old[b:]
        diffs.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=str(ROOT/name),tofile=str(STAGE/name)))
    (WORK/'caption_changes.diff').write_text(''.join(diffs))
    template=STORY.parent/'figure_melanoma/figure_layout_v7/manuscript_sync/install_checked_files.py'
    code=template.read_text().replace('Melanoma','Lung').replace('melanoma_before_','lung_before_').replace('.melanoma_sync_tmp','.lung_sync_tmp')
    code=code.replace('figure_page=8, caption_page=9','figure_page=10, caption_page=10')
    code=code.replace('melanoma_three_row_redesign_v7.png','lung_panels_ABCDEF.png')
    code=code.replace('Figure 4 panels F-H and corresponding caption descriptions updated; all other TeX unchanged','Figure 5 icon and spatial display updated; only Panel D caption changed in TeX')
    (WORK/'install_checked_files.py').write_text(code)
    print('Prepared six manuscript files; only Lung panel D caption text changed.')

if __name__=='__main__':main()
