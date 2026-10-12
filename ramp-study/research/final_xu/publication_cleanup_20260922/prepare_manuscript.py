from pathlib import Path
import hashlib, json, re, shutil, csv
ROOT=Path(__file__).resolve().parent.parent
HERE=Path(__file__).resolve().parent
LIVE=ROOT.parent/'bioRxiv_Ver1'
STAGE=HERE/'manuscript_stage'
STAGE.mkdir(exist_ok=True)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cp(src,dst):
 dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
def renamed(s): return s.replace('Harmony-direct','Harmony').replace('Harmony direct','Harmony')
def once(s,old,new):
 assert s.count(old)==1,(old[:100],s.count(old));return s.replace(old,new)
# Copy only current publication dependencies; old records remain untouched.
for p in LIVE.iterdir():
 if p.is_file() and p.suffix in ['.tex','.md','.bib','.bst']:
  cp(p,STAGE/p.name)
for name in ['figure','supplementary_data']:
 shutil.copytree(LIVE/name,STAGE/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('.DS_Store'))
originals={str(p.relative_to(LIVE)):sha(p) for p in LIVE.rglob('*') if p.is_file() and not any(x in p.parts for x in ['_backups','Supplementary_Overleaf'])}
(HERE/'original_live_hashes.json').write_text(json.dumps(originals,indent=2)+'\n')
s=(LIVE/'template.tex').read_text()
s=once(s,'test how preprocessing affects cross-platform annotation, missing-marker\nprediction, label requirements, model capacity and multi-source training\nunder matched supervision.', 'test how preprocessing affects cross-platform annotation and missing-marker\nprediction under matched supervision.')
s=once(s,r'''\textbf{(B)} Alignment of 62,701 breast cancer cells using 11 proteins.
IMC1, IMC2 and CODEX denote \citet{tietscher2023}, \citet{meyer2025} and
\citet{shiao2024}, respectively. Sample-level ridges accompany identical UMAP''',r'''\textbf{(B)} Alignment of 60,000 melanoma cells using 12 shared proteins.
CODEX, IMC and MIBI denote \citet{hickey2023}, \citet{hoch2022} and
\citet{vesely2025mibimelanoma}, respectively. Ridges of CD3, CD8 and SOX10
across 46 sample units accompany identical UMAP''')
s=once(s,'units in four benchmarks, using identical combinations and Raw references.\nBoundary triangles', 'units in four benchmarks, using identical combinations and Raw references\nfor Harmony, cyCombine, CytoNorm, reconstructed fastMNN and RAMP.\nBoundary triangles')
s=once(s,'senosain2023,cords2024,monkman2024}. In the breast cancer benchmark, containing\n62,701 cells and 11 shared proteins, source datasets differed in their marker', 'senosain2023,cords2024,monkman2024}. In the three-platform melanoma benchmark,\ncontaining 60,000 cells and 12 shared proteins, source datasets differed in their marker')
s=once(s,'RAMP supports cross-platform prediction across model families and supervision budgets.', 'RAMP supports cross-platform prediction across model families.')
a=s.index(r'\textbf{(D)} Neural-network annotation macro-F1'); b=s.index('}{fig:trainability}',a)
s=s[:a]+'''Three random seeds were used, with source training and validation support
checked before comparison. Benchmark configurations share some source cohorts;
points are descriptive summaries, not independent-patient replicates.
'''+s[b:]
a=s.index('Separately, the original-scope experiments varied source supervision and');b=s.index(r'\subsection{Reference-based completion',a)
s=s[:a]+s[b:]
s=once(s,'supervision, RAMP improved cross-platform annotation, supported smaller\nnetworks and fewer training labels, and produced the highest combined-source\nclassification endpoint across the three model families tested. The benefit', 'supervision, RAMP improved cross-platform annotation across the three model\nfamilies tested. The benefit')
s=once(s,'incomparable. D--F retain the initial class sets and original Lung reference\npool and are separate resource experiments, not continuations of the revised B\nclassifier. Missing-marker prediction uses all frozen cells without the', 'incomparable. Missing-marker prediction uses all frozen cells without the')
a=s.index(r'\paragraph{Annotation budgets, capacity and source-cohort diversity.}');b=s.index(r'\paragraph{Scoring, aggregation and source records.}',a)
s=s[:a]+s[b:]
s=once(s,'the three model families equally for each benchmark point; D,E use neural\nnetworks and F shows model families separately. Resource curves retain matched\ndirection support across the tested levels. Boxplots and curves are\ndescriptive summaries over overlapping benchmark configurations; random seeds\nand configurations are not independent-patient replicates. Per-model scores,\nresource curves, source/target mappings, parameter counts, eligibility records\nand input provenance are supplied under', 'the three model families equally for each benchmark point. Boxplots are\ndescriptive summaries over overlapping benchmark configurations; random seeds\nand configurations are not independent-patient replicates. Per-model scores,\nsource/target mappings, eligibility records and input provenance are supplied under')
s=once(s,'used for RAMP, Harmony-direct, Harmony-arcsinh, cyCombine, CytoNorm and', 'used for RAMP, Harmony, cyCombine, CytoNorm and')
s=renamed(s)
for forbidden in ['Harmony-direct','Harmony-arcsinh','and supervision budgets.','D--F retain',r'\ref{fig:trainability}D',r'\ref{fig:trainability}E',r'\ref{fig:trainability}F']:
 assert forbidden not in s,forbidden
(STAGE/'template.tex').write_text(s)
def texmd(t):
 t=re.sub(r'\\textbf\{([^{}]*)\}',r'**\1**',t)
 t=re.sub(r'\\paragraph\{([^{}]*)\}',r'### \1\n',t)
 t=t.replace(r'Figure~\ref{fig:trainability}','Figure 3').replace(r'Figure~\ref{fig:benchmark}','Figure 2').replace(r'Figure~\ref{fig:overview}','Figure 1')
 return t.replace('~',' ')
cap=s.split(r'\paperfig{figure3_cross_platform_learning}{%',1)[1].split('}{fig:trainability}',1)[0].strip()
(STAGE/'FIGURE3_CAPTION.md').write_text('# Figure 3 caption — A–C\n\n'+texmd(cap)+'\n')
cap1=s.split(r'\paperfig{figure1_ramp_overview}{%',1)[1].split('}{fig:overview}',1)[0].strip()
(STAGE/'FIGURE1_CAPTION.md').write_text('# Figure 1 caption — A–D\n\n'+texmd(cap1)+'\n')
res=s.split(r'\subsection{RAMP supports cross-platform predictive learning}\label{res-trainability}',1)[1].split(r'\subsection{Reference-based completion',1)[0]
res=re.sub(r'\\paperfig\{figure3_cross_platform_learning\}\{%.+?\}\{fig:trainability\}', '',res,flags=re.S).strip()
methods=s.split(r'\subsection{Cross-platform learning experiments}\label{methods-trainability}',1)[1].split(r'\subsection{Within-sample marker-order audit}',1)[0].strip()
disc=s.split('Marker-level alignment and learned representations address different',1)[1].split('\n\nThese experiments evaluate',1)[0]
(STAGE/'CROSS_PLATFORM_LEARNING_SECTION.md').write_text('# Cross-platform learning — current Figure 3 A–C\n\nSynced from the active manuscript, 2026-09-22. Archived D–F resource experiments are not included in the main figure or the supplement.\n\n## Results\n\n'+texmd(res)+'\n\n## Methods\n\n'+texmd(methods)+'\n\n## Discussion\n\nMarker-level alignment and learned representations address different'+texmd(disc)+'\n')
for name in ['supplementary_captions.tex','SUPPLEMENTARY_CAPTIONS.md','FIGURE2_CAPTION.md']:
 p=STAGE/name;p.write_text(renamed(p.read_text()))
# Figure copies and provenance.
fig1=ROOT/'Main Figure/publication_display_20260922/figure_1_melanoma.pdf'
fig3=ROOT/'cross_platform_trainability_v2_20260919/Figure/Combined ABC/figure_cross_platform_learning_ABC.pdf'
cp(fig1,STAGE/'figure/figure1_ramp_overview.pdf');cp(fig3,STAGE/'figure/figure3_cross_platform_learning.pdf')
sources=json.loads((LIVE/'SUPPLEMENTARY_FIGURE_SOURCES.json').read_text())
for entry in sources:
 old=entry.copy();p=ROOT/'Supplementary Figures/harmony_display_20260922/pdf'/Path(entry['destination']).name
 cp(p,STAGE/entry['destination']);entry.update(source=str(p),sha256=sha(p),previous_source=old['source'],previous_sha256=old['sha256'],change='Display label Harmony; numeric results and other figure content unchanged.')
(STAGE/'SUPPLEMENTARY_FIGURE_SOURCES.json').write_text(json.dumps(sources,indent=2)+'\n')
# Rename display fields in publication CSVs only; do not alter machine identifiers or provenance paths.
data_edits=[]
for p in (STAGE/'supplementary_data').rglob('*'):
 if not p.is_file():continue
 if p.suffix=='.csv':
  with p.open(newline='') as f: rows=list(csv.reader(f))
  count=sum(v in ['Harmony-direct','Harmony direct'] for row in rows for v in row)
  if count:
   new=[['Harmony' if v in ['Harmony-direct','Harmony direct'] else v for v in row] for row in rows]
   with p.open('w',newline='') as f: csv.writer(f).writerows(new)
   data_edits.append(dict(path=str(p.relative_to(STAGE)),display_cells_changed=count,numeric_values_unchanged=True))
 if p.suffix=='.md':p.write_text(renamed(p.read_text()))
mapfile=STAGE/'supplementary_data/method_display_names.csv'
mapfile.write_text('internal_identifier,publication_label\nharmony_direct,Harmony\nHarmony-direct,Harmony\nHarmony-direct[source_scale],Harmony\nHarmony-direct_source_scale,Harmony\n')
p=STAGE/'supplementary_data/trainability/README.md';t=p.read_text();t=t.replace('- `Panel_D/`, `Panel_E/`, `Panel_F/`: unchanged original-v0.5 resource experiments,', '- `Panel_D/`, `Panel_E/`, `Panel_F/`: archived original-v0.5 resource experiments,\n  removed from the current main figure and not included in supplementary figures,')
t+='\nDisplay names use **Harmony**. The internal identifier `harmony_direct` and source paths are retained for reproducibility; see `../method_display_names.csv`. No numerical results were changed.\n';p.write_text(t)
p=STAGE/'README.md';t=p.read_text()
t=t.replace('Copied unchanged from Biorxiv','`../final_xu/Main Figure/publication_display_20260922/`; melanoma example, five marker-valued methods in D')
t=t.replace('Combined%20ABCDEF/','Combined%20ABC/')
t=re.sub(r'Current manuscript: 50 pages.*?\n\n', 'Current manuscript: PAGE_COUNT_PLACEHOLDER pages after the 2026-09-22 figure-scope and display-name update.\n\n',t,flags=re.S)
t=once(t,'Figure 3 B/C and its Results, Methods, abstract and source tables were updated on 2026-09-22. D–F retain the original resource-experiment scope; see [update record](FIGURE3_UPDATE_20260922.md).','Figure 3 now contains A–C only. Its caption, Results, Methods and Discussion match this scope; D–F remain archived, without being added to the supplement. B/C numerical results are unchanged. See [current update](PUBLICATION_UPDATE_20260922.md).')
t=t.replace('the existing CSV source tables are unchanged.', 'numerical source-table values are unchanged; public method labels now use Harmony.')
t=t.replace('Main figure assets are unchanged.', 'Current main-figure changes are recorded in PUBLICATION_UPDATE_20260922.md.')
t+='\nThe current standalone Overleaf upload package is [Supplementary_Overleaf.zip](Supplementary_Overleaf.zip).\n';p.write_text(t)
notes='''# Publication update — 2026-09-22

- Figure 1B now uses the accepted three-platform melanoma example: 60,000 frozen cells, 46 sample units, 12 shared markers; CD3/CD8/SOX10 ridges. Coordinates, cells and all method settings are unchanged from that saved example.
- Figure 1D now shows Harmony, cyCombine, CytoNorm, reconstructed fastMNN and RAMP. Harmony with fixed arcsinh preprocessing is omitted from this display. The same 6,173 observations and untruncated box statistics are retained for each remaining method; C examples are unchanged.
- Public labels use Harmony throughout main and supplementary figures, legends, text and display columns. Internal identifiers and original source paths remain unchanged.
- Figure 3 contains A–C only. Removed D–F descriptions and related resource-efficiency/capacity claims from caption, Results, Methods and Discussion. The original D–F analyses remain archived; no new supplementary figure was added.
- Figure 3 B/C, Figure 2's 23-metric scope and S1–S13 numeric results are unchanged.
- Per-marker prediction plots and Lung/HNSCC all-patient validation plots remain deferred, as requested.
- Original current publication files and Overleaf package are backed up before installation. See the synchronization receipt in the source work directory.
'''
(STAGE/'PUBLICATION_UPDATE_20260922.md').write_text(notes)
p=STAGE/'REVISION_NOTES.md';p.write_text(p.read_text()+'\n\n'+notes)
(HERE/'published_data_display_edits.json').write_text(json.dumps(data_edits,indent=2)+'\n')
(HERE/'new_figure_sources.json').write_text(json.dumps({'figure1':str(fig1),'figure3':str(fig3)},indent=2)+'\n')
print('Prepared publication stage:',STAGE)
print('Renamed display columns in',len(data_edits),'tables; values unchanged.')
