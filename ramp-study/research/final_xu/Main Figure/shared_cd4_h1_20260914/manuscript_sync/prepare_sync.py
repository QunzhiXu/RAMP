"""Stage Figure 1 and its directly related caption/method edits locally."""
from pathlib import Path
import hashlib, json, shutil, difflib

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
DEST=Path('/Users/qxu6/Documents/New project/manuscript')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

files=[p for p in DEST.iterdir() if p.is_file()]
files += [p for p in (DEST/'figure').iterdir() if p.is_file()]
snapshot=HERE/'original_manuscript_hashes.json'
if not snapshot.exists():
    snapshot.write_text(json.dumps({str(p):sha(p) for p in files},indent=2)+'\n')
original=json.loads(snapshot.read_text())
assert all(sha(p)==v for p,v in original.items())
stage=HERE/'staged';(stage/'figure').mkdir(parents=True,exist_ok=True)
for source in (DEST/'figure').iterdir():
    if source.is_file(): shutil.copy2(source,stage/'figure'/source.name)
for ext in ['pdf','png','svg']:
    shutil.copy2(BASE/f'figure_1_shared_cd4.{ext}',stage/'figure'/f'figure1_ramp_overview.{ext}')
shutil.copy2(BASE/'figure_1_shared_cd4.pdf',stage/'figure/figure1_ramp_overview_polished.pdf')

caption_c=r'''\textbf{(C) Selected marker-distribution changes in comparison methods.}
Upper, CD4 in the same bone-marrow CyTOF sample H1 across Raw,
Harmony-direct, CytoNorm and RAMP. The groups are CD4 T cells ($n=4{,}029$),
CD8 T cells ($n=2{,}883$), B cells ($n=2{,}756$) and monocytes ($n=3{,}608$).
CD4-T-versus-monocyte AUROC is 0.926, 0.378, 0.322 and 0.926, respectively;
CytoNorm also changes CD4-T-versus-B-cell AUROC from 0.968 to 0.403.
Lower left, CD8 in Shiao CODEX sample h17: CD4 T cells ($n=164$), CD8 T cells
($n=120$) and macrophages ($n=420$). With cyCombine, CD8-T-versus-macrophage
AUROC changes from 0.976 to 0.610 and the CD8 T-cell median percentile from
0.983 to 0.371. Lower right, CD14 in bone-marrow CyTOF sample H2: CD4 T cells
($n=4{,}264$), CD8 T cells ($n=3{,}563$), B cells ($n=2{,}185$) and monocytes
($n=2{,}600$). Monocyte-versus-B-cell AUROC changes from 0.811 to 0.149 in
the standard PCA-fastMNN reconstructed marker output. RAMP retains the
original percentile profiles. Monocytes (purple) and macrophages (gold)
retain their distinct provider annotations; plasma cells are not relabelled
as B cells. Each method's values are converted to average rank$/(n+1)$ among
all cells in the sample before selecting displayed groups. Boxes show the
median and interquartile range, with 1.5-IQR whiskers and individual outliers
omitted. Axes share ticks at 0, 0.5 and 1 and an upper display limit of 1.04.
Colored arrows identify the Raw reference population; shading identifies
comparison methods. The CD4 example shows reversal of relative ordering,
without defining monocytes as absolutely CD4-negative; the CD8 example
shows reduced discrimination rather than complete reversal. Examples were
selected by exploratory examination of the four frozen benchmarks and do
not estimate artifact prevalence.
'''

for stem in ['manuscript','manuscript_figure1_polished']:
    old=(DEST/f'{stem}.tex').read_text()
    start=old.index('% ---------------------------------------------------------------- Fig 1')
    end=old.index('% ---------------------------------------------------------------- Fig 2',start)
    block=old[start:end]
    c=block.index(r'\textbf{(C)');d=block.index(r'\textbf{(D)',c)
    block=block[:c]+caption_c+block[d:]
    block=block.replace('The same 6,173 combinations from 104\nsample units are evaluated for each method',
        'The same 6,173 combinations from 103 eligible units out of 104 loaded\nsample units are evaluated for each of six methods')
    block=block.replace('(31 for Harmony-direct and 7 for cyCombine; none for the other methods).',
        '(31 for Harmony-direct, 7 for cyCombine and 23 for reconstructed fastMNN).')
    block=block.replace('in C. Embedding-only fastMNN and MaxFuse lack comparable marker axes and\nare excluded from D.',
        'in C. FastMNN uses its reconstructed marker coordinates, a diagnostic\nrepresentation rather than calibrated original-scale protein abundance.\nMaxFuse has no corresponding corrected marker output and is excluded.')
    assert 'in Meyer IMC sample B12.64192' not in block
    assert '103 eligible' in block and '23 for reconstructed fastMNN' in block
    text=old[:start]+block+old[end:]
    prior='fastMNN and MaxFuse return latent embeddings; the remaining methods return\nmarker-valued matrices.'
    replacement="Integration evaluation uses fastMNN and MaxFuse embeddings and the other\nmethods' marker-valued matrices; the additional fastMNN reconstruction is\ndescribed below."
    assert prior in text;text=text.replace(prior,replacement)
    text=text.replace('For every method returning marker-valued output, the same 1,151',
        'For the original marker-output audit, the same 1,151')
    text=text.replace('any resulting biological artifact. Embedding methods are excluded because\ntheir output dimensions do not correspond one-to-one to measured markers.',
        'any resulting biological artifact. The original audit excluded fastMNN\nand MaxFuse because their integration embeddings do not have measured-marker\naxes; the additional fastMNN reconstruction diagnostic below is separate.')
    prior='combinations were used for every method, retaining all shared markers and\nresolved groups with at least 50 cells per sample.'
    replacement='combinations from 103 eligible units out of 104 loaded sample units were\nused for each of six methods, retaining all shared markers and resolved\ngroups with at least 50 cells per sample.'
    assert prior in text;text=text.replace(prior,replacement)
    anchor=r'\subsection*{Display settings}'
    addition=r'''For Figure~\ref{fig:overview}C,D, fastMNN marker coordinates were reconstructed
as corrected principal-component scores multiplied by the transpose of the
saved rotation matrix, without restoring original-scale centers or inverting
cosine normalization. This is the standard PCA reconstruction, not the
$d=\mathrm{NA}$ experiment, and serves as a marker-coordinate diagnostic;
integration metrics retain corrected principal components. MaxFuse has no
analogous corrected marker output.

'''
    assert text.count(anchor)==1;text=text.replace(anchor,addition+anchor)
    assert text[:start]==old[:start], 'Abstract changed'
    a=text.index('% ---------------------------------------------------------------- Fig 2');b=text.index(r'\subsection*{Comparison methods}')
    ao=old.index('% ---------------------------------------------------------------- Fig 2');bo=old.index(r'\subsection*{Comparison methods}')
    assert text[a:b]==old[ao:bo], 'Other figures/story content changed'
    (stage/f'{stem}.tex').write_text(text)
    (HERE/f'{stem}.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),text.splitlines(True),fromfile='old/'+stem+'.tex',tofile='new/'+stem+'.tex')))

(HERE/'STAGING_MANIFEST.json').write_text(json.dumps(dict(figure_source=str(BASE),
    changed_sections=['Figure 1 asset','Figure 1 C/D caption','Directly related fastMNN/shift Methods'],
    abstract_and_other_figures_preserved=True,manuscript_write_performed=False),indent=2)+'\n')
print(stage)
