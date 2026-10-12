"""Prepare a separate manuscript preview; original manuscript remains intact."""
from pathlib import Path
import sys,json,hashlib,shutil

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
MANUSCRIPT=Path('/Users/qxu6/Documents/New project/manuscript')
sys.path.insert(0,str(BASE))
import common
import build_all

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

common.setup()
common.plt.rcParams['pdf.fonttype']=42
# Preserve all previous exported data as well as the frozen inputs.
common.save_data=lambda *args,**kwargs:None
mods={c:build_all.load(c) for c in 'ABCDEF'}
fig=common.plt.figure(figsize=(180/25.4,71/25.4),facecolor='white')
positions={'A':(0,0,26.5,65),'B':(27.5,0,26.5,65),'C':(55.5,0,34,65),
    'D':(91,0,29,65),'E':(121.5,0,30,65),'F':(153,0,27,65)}
for c,m in mods.items():m.draw(fig,positions[c])
common.case_legend(fig,68)
asset=HERE/'figure2_benchmark_single_row.pdf'
fig.savefig(asset,dpi=360)
common.plt.close(fig)

source=MANUSCRIPT/'manuscript_figure1_polished.tex'
text=source.read_text()
start=text.index('% ---------------------------------------------------------------- Fig 2')
end=text.index('% ---------------------------------------------------------------- Fig 3',start)
caption=(HERE/'figure2_caption.tex').read_text()
replacement=r'''% ---------------------------------------------------------------- Fig 2
\begin{center}
\includegraphics[width=\textwidth]{figure2_benchmark_single_row.pdf}
\end{center}
'''+caption+'\n\\label{fig:benchmark}\n\\clearpage\n\n\n'
text=text[:start]+replacement+text[end:]
# Keep the general 31-metric evaluation description; describe the displayed
# five-category summary separately so its numbers match the new Figure 2.
old=r'''The aggregate reported in
Figure~\ref{fig:benchmark}D averages the source category ranks over all 24
benchmark~$\times$~category cells, not the integer heatmap positions, and
weights categories equally so that batch mixing does not dominate by virtue of
contributing 16 of the 31 metrics.'''
new=r'''The display in Figure~\ref{fig:benchmark}C,D uses the 30 metrics in the
five categories other than Geometry. This display scope was selected after
inspection of the full results. Panel D averages per-metric ranks within each
benchmark and category, then gives equal weight to all 20
benchmark~$\times$~category combinations. The complete 31-metric evaluation
is retained separately.'''
assert old in text
text=text.replace(old,new)
old=r'''Display refinements in
Figure~\ref{fig:benchmark}B and C alter point rendering and layer order only
and are applied identically to all methods within a panel.'''
new=r'''Display refinements alter point rendering and layer order only and
are applied identically to all methods within a panel.'''
assert old in text
text=text.replace(old,new)
(HERE/'manuscript_figure2_single_row.tex').write_text(text)
figdir=HERE/'figure';figdir.mkdir(exist_ok=True)
for p in (MANUSCRIPT/'figure').iterdir():
    if p.is_file() and not (figdir/p.name).exists():
        (figdir/p.name).symlink_to(p)
local=figdir/asset.name
if not local.exists():local.symlink_to(asset)
originals=[p for p in MANUSCRIPT.iterdir() if p.is_file()]
originals += [p for p in (MANUSCRIPT/'figure').iterdir() if p.is_file()]
(HERE/'original_manuscript_hashes.json').write_text(json.dumps({str(p):digest(p) for p in originals},indent=2)+'\n')
(HERE/'preview_provenance.json').write_text(json.dumps({
    'source_tex':str(source),'source_tex_sha256':digest(source),
    'changes':['Replace only Figure 2 with the new single row at full text width, preserving polished Figure 1.',
        'Update Figure 2 caption and directly affected Methods references.',
        'Keep original manuscript files intact; output has a separate preview filename.'],
    'input_sha256':common.READS,'native_figure_width_mm':180,
    'text_width_mm':170.18,'effective_tick_font_pt':6*170.18/180,
    'output_pdf':'manuscript_figure2_single_row.pdf'},indent=2)+'\n')
print(HERE/'manuscript_figure2_single_row.tex')
