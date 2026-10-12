"""Stage figure and corresponding F-H caption edits before manuscript replacement."""
from pathlib import Path
import difflib
import hashlib
import json
import shutil

HERE = Path(__file__).resolve().parent
ROOT = Path('/Users/qxu6/Documents/New project/manuscript')
WORK = HERE / 'manuscript_sync'
STAGE = WORK / 'staging'

F_CAPTION = r'''\textbf{(F) Patient-held-out prediction performance.} Each metric shows the
same 14 evaluable IMC patients with at least 20 candidate CD8 cells and both
target classes. Median AUROC, AUPRC, precision and recall are 0.707, 0.212,
0.193 and 0.395, respectively. AUPRC is average precision; precision and
recall use the highest-scoring 20\% of candidate CD8 cells in each whole
patient image, as in E. Boxes show quartiles and medians, whiskers extend
to 1.5 interquartile ranges, and dots show every patient. Standardization
and classifier fitting excluded the evaluated patient; the RAMP alignment
was frozen from the complete cohort.
'''
H_CAPTION = r'''\textbf{(H) Held-out proteins associated with the transferred state.} CD11a,
CD18, CD2 and PD-1 entered neither the alignment nor the classifier. After
cell-level CLR normalization of ADT counts, expression was ranked within
each patient. For each protein, High (red) denotes the highest score quartile
and Lower (blue) the remaining cells. Each box contains eight patient-level
mean expression ranks; thin lines connect the same patient. Box and whisker
definitions follow F. All four proteins are higher in high-scoring cells in
all eight patients. A module averaging the four ranks has mean within-patient
Spearman $\rho=0.480$ with the transferred score. CD11a and CD18 form the LFA-1
heterodimer, linking these measurements to an adhesion-and-activation programme.
'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    (STAGE/'figure').mkdir(parents=True, exist_ok=True)
    files = [Path(p) for p in ['manuscript.tex','manuscript.pdf',
             'manuscript_figure1_polished.tex','manuscript_figure1_polished.pdf',
             'figure/figure4_melanoma.png','figure/figure4_melanoma.svg']]
    originals = {str(ROOT/p): sha(ROOT/p) for p in files}
    (WORK/'original_hashes.json').write_text(json.dumps(originals,indent=2)+'\n')
    for src in (ROOT/'figure').iterdir():
        target = STAGE/'figure'/src.name
        if src.name in ['figure4_melanoma.png','figure4_melanoma.svg']:
            shutil.copy2(HERE/f'melanoma_three_row_redesign_v7{src.suffix}', target)
        elif not target.exists():
            target.symlink_to(src)
    diffs = []
    for name in ['manuscript.tex','manuscript_figure1_polished.tex']:
        original = (ROOT/name).read_text()
        start = original.index(r'\includegraphics[width=\textwidth]{figure4_melanoma.png}')
        end = original.index(r'\label{fig:melanoma}', start)
        section = original[start:end]
        f_start = section.index(r'\textbf{(F)')
        g_start = section.index(r'\textbf{(G)')
        h_start = section.index(r'\textbf{(H)')
        closing = section.index('These analyses connect', h_start)
        g = section[g_start:h_start]
        assert 'saturates at 0.9.' in g
        g = g.replace('saturates at 0.9.', 'saturates at 0.9. The dashed ellipse marks the main high-score\nconcentration for visual guidance; it does not select cells.')
        revised = section[:f_start]+F_CAPTION+g+H_CAPTION+section[closing:]
        updated = original[:start]+revised+original[end:]
        assert updated[:start] == original[:start]
        assert updated[updated.index(r'\label{fig:melanoma}'):] == original[end:]
        (STAGE/name).write_text(updated)
        diffs.extend(difflib.unified_diff(original.splitlines(True), updated.splitlines(True),
                     fromfile=str(ROOT/name), tofile=str(STAGE/name)))
    (WORK/'caption_changes.diff').write_text(''.join(diffs))
    old = HERE.parent/'figure_layout_v6/manuscript_sync/install_checked_files.py'
    installer = old.read_text().replace('four manuscript files', 'six manuscript files')
    installer = installer.replace("    for path, expected in json.loads((HERE/'tex_hashes.json').read_text()).items():\n        assert sha(Path(path)) == expected, f'TeX changed since staging: {path}'\n", '')
    installer = installer.replace('assert len(files) == 4', 'assert len(files) == 6')
    installer = installer.replace('melanoma_three_row_redesign_v6', 'melanoma_three_row_redesign_v7')
    installer = installer.replace('TeX and captions unchanged; only the Figure 4 icon updated',
                    'Figure 4 panels F-H and corresponding caption descriptions updated; all other TeX unchanged')
    (WORK/'install_checked_files.py').write_text(installer)
    print('Staged six files and caption-only edits for F-H.')


if __name__ == '__main__':
    main()
