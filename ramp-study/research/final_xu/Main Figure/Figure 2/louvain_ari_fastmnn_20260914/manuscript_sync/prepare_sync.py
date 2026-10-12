"""Stage the current manuscript with only the agreed Figure 2 changes."""
from pathlib import Path
import json, hashlib, shutil, difflib

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
DEST = Path('/Users/qxu6/Documents/New project/manuscript')

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)

def main():
    originals = [p for p in DEST.iterdir() if p.is_file()]
    originals += [p for p in (DEST/'figure').iterdir() if p.is_file()]
    snap = HERE/'original_manuscript_hashes.json'
    if not snap.exists():
        snap.write_text(json.dumps({str(p): sha(p) for p in originals}, indent=2)+'\n')
    original = json.loads(snap.read_text())
    assert all(sha(p) == h for p,h in original.items())
    stage = HERE/'staged';stage.mkdir(exist_ok=True)
    figs = stage/'figure';figs.mkdir(exist_ok=True)
    for source in (DEST/'figure').iterdir():
        if source.is_file():shutil.copy2(source, figs/source.name)
    for ext in ['pdf','png','svg']:
        shutil.copy2(BASE/'Combined'/f'figure_2_louvain_fastmnn.{ext}',figs/f'figure2_benchmark.{ext}')
    for stem in ['manuscript','manuscript_figure1_polished']:
        before = (DEST/f'{stem}.tex').read_text()
        text = once(before,'Cell-type agreement (Leiden adjusted Rand index, ARI)',
                    'Cell-type agreement (Louvain adjusted Rand index, ARI)')
        text = once(text,'separation (macro-averaged cell-type silhouette in UMAP space), respectively.',
                    'separation (macro-averaged reference-cell-type silhouette in UMAP space), respectively.')
        text = once(text,
            'Tasks can share source cohorts and are not independent patient replicates.\n\\textbf{(C)}',
            'Tasks may share source cohorts. Louvain ARI was selected after metric review.\n\\textbf{(C)}')
        text = once(text,
            'Each colored point is one of seven integration tasks: the original bone-marrow,\nTNBC, melanoma and lung-cancer benchmarks, plus the lung, melanoma and HNSCC\nstory cohorts.',
            'Each colored point is one of seven tasks: the four original benchmarks\nand the lung, melanoma and HNSCC story cohorts.')
        text = once(text,
            "benchmarks equally. RAMP's overall BA is 0.869. Raw is a separate baseline;\nembedding-only fastMNN and MaxFuse are inapplicable to single-marker gates.",
            "benchmarks equally. Overall BA is 0.869 for RAMP and 0.835 for reconstructed\nfastMNN. F uses standard PCA-fastMNN reconstructed marker values, whereas\nA--D use corrected PCs; MaxFuse has no corresponding marker output.\nRaw is a separate baseline.")
        text = once(text,
            'not enter D. No significance tests based on these correlated summaries are\nshown.}',
            'not enter D.}')
        text = once(text,
            'method. Panels A and B display the original metric means. All seven tasks use\n20 matched resamples from frozen alignments, not independently refitted models.',
            'method. Panels A and B show Louvain ARI and reference-cell-type UMAP\nsilhouette, respectively, averaging 20 matched resamples from frozen alignments\nrather than independently refitted models.')
        text = once(text,
            'baseline; marker-gate evaluation is inapplicable to the frozen fastMNN and\nMaxFuse embeddings.',
            'baseline. Standard PCA-fastMNN reconstructed marker values use the same\ngating protocol; its corrected PCs remain in A--D. MaxFuse has no marker output.')
        text = once(text,
            'For Figure~\\ref{fig:overview}C,D, fastMNN marker coordinates were reconstructed',
            'For Figure~\\ref{fig:overview}C,D and Figure~\\ref{fig:benchmark}F,\nfastMNN marker coordinates were reconstructed')
        (stage/f'{stem}.tex').write_text(text)
        (HERE/f'{stem}.diff').write_text(''.join(difflib.unified_diff(
            before.splitlines(keepends=True),text.splitlines(keepends=True),
            fromfile=str(DEST/f'{stem}.tex'),tofile=str(stage/f'{stem}.tex'))))
        assert before[:before.index('% ---------------------------------------------------------------- Fig 2')] == text[:text.index('% ---------------------------------------------------------------- Fig 2')]
        a='% ---------------------------------------------------------------- Fig 3';b='\\subsection*{Evaluation}'
        assert before[before.index(a):before.index(b)] == text[text.index(a):text.index(b)]
    (HERE/'STAGING_MANIFEST.json').write_text(json.dumps({
        'source_figure':str(BASE/'Combined/figure_2_louvain_fastmnn.pdf'),
        'changes':['Figure 2 assets','Louvain ARI caption and metric description',
                   'fastMNN reconstructed gating caption and directly related Methods'],
        'other_figures_and_abstract_unchanged':True,
        'originals_unchanged':all(sha(p)==h for p,h in original.items())},indent=2)+'\n')
    print(stage)

if __name__=='__main__':main()
