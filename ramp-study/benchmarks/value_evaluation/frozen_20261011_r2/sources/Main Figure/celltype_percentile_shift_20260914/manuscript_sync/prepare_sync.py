"""Stage the current Figure 1, its caption, and directly related Methods."""
from pathlib import Path
import hashlib
import json
import shutil
import difflib

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
DEST = Path('/Users/qxu6/Documents/New project/manuscript')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    snapshot = HERE/'original_manuscript_hashes.json'
    if not snapshot.exists():
        files = [p for p in DEST.iterdir() if p.is_file()]
        files += [p for p in (DEST/'figure').iterdir() if p.is_file()]
        snapshot.write_text(json.dumps({str(p):sha(p) for p in files},indent=2)+'\n')
    original = json.loads(snapshot.read_text())
    assert all(sha(p)==s for p,s in original.items()), 'Manuscript changed during staging'
    stage = HERE/'staged'
    figures = stage/'figure'
    figures.mkdir(parents=True,exist_ok=True)
    for source in (DEST/'figure').iterdir():
        if source.is_file():
            shutil.copy2(source,figures/source.name)
    for ext in ['png','pdf','svg']:
        shutil.copy2(BASE/f'figure_1_celltype_percentile_shift.{ext}',figures/f'figure1_ramp_overview.{ext}')
    shutil.copy2(BASE/'figure_1_celltype_percentile_shift.pdf',figures/'figure1_ramp_overview_polished.pdf')
    caption=(HERE/'figure1_caption.tex').read_text().strip()
    addition=r'''\paragraph{Cell-type marker distribution shifts.}
For Figure~\ref{fig:overview}D, the common baseline was the frozen raw marker
matrix. Raw and corrected values were ranked independently among all cells
in each sample and marker, with average ranks for tied values, and converted
to $100\times\mathrm{rank}/(n+1)$. For each
sample~$\times$~cell type~$\times$~marker combination, we computed the
signed and absolute changes in the 25th percentile, median and 75th
percentile, using linear quantile interpolation. The displayed statistic
was the absolute median change in percentile points. The same 6,173
combinations were used for every method, retaining all shared markers and
resolved groups with at least 50 cells per sample. Smaller groups and
unresolved/non-joint labels were omitted from the summary but retained in
the sample-wide ranking reference and complete tables. Combinations were
weighted equally; benchmarks, patients and cell counts were not separately
reweighted, and no significance tests treated these combinations as
independent patients. Changes relative to method-specific preprocessed
inputs were retained separately. The display range was 0--60 points;
observations above 60 were represented by right-pointing boundary markers,
while box and whisker statistics used full values. Changes in ties can
alter average-rank percentiles without a strict order inversion. Cell-type
labels were used only for evaluation, not for alignment.

'''
    for stem in ['manuscript','manuscript_figure1_polished']:
        old=(DEST/f'{stem}.tex').read_text()
        start=old.index('% ---------------------------------------------------------------- Fig 1')
        end=old.index('% ---------------------------------------------------------------- Fig 2',start)
        figure='figure1_ramp_overview.pdf' if stem=='manuscript' else 'figure1_ramp_overview_polished.pdf'
        block=('% ---------------------------------------------------------------- Fig 1\n'
               '\\begin{center}\n\\includegraphics[width=\\textwidth]{'+figure+'}\n\\end{center}\n'
               +caption+'\n\\label{fig:overview}\n\\clearpage\n\n\n')
        text=old[:start]+block+old[end:]
        anchor=r'\subsection*{Display settings}'
        assert text.count(anchor)==1
        text=text.replace(anchor,addition+anchor)
        # Preserve the old strict-order audit, which still supports the existing
        # abstract. Do not rewrite unrelated claims during a figure insertion.
        assert text[:start]==old[:start]
        figure2=text[text.index('% ---------------------------------------------------------------- Fig 2'):text.index('% ---------------------------------------------------------------- Fig 3')]
        oldfigure2=old[old.index('% ---------------------------------------------------------------- Fig 2'):old.index('% ---------------------------------------------------------------- Fig 3')]
        assert figure2==oldfigure2
        (stage/f'{stem}.tex').write_text(text)
        (HERE/f'{stem}.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),text.splitlines(True),
                                        fromfile=f'old/{stem}.tex',tofile=f'new/{stem}.tex')))
    (HERE/'STAGING_MANIFEST.json').write_text(json.dumps(dict(
        figure_source=str(BASE),figure_sha256={ext:sha(BASE/f'figure_1_celltype_percentile_shift.{ext}') for ext in ['pdf','png','svg']},
        changed_sections=['Figure 1 image reference and A-D caption','Directly related cell-type shift Methods paragraph'],
        preserved='Abstract, Figure 2 and other story text, original strict-order audit, all non-Figure-1 assets',
        manuscript_write_performed=False),indent=2)+'\n')
    print(stage)


if __name__=='__main__':
    main()
