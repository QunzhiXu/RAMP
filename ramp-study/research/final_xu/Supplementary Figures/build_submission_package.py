"""Stage the supplementary figures and concise legends for bioRxiv_Ver1."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parent
LIVE = ROOT.parent.parent / 'bioRxiv_Ver1'
PACKAGE = ROOT / 'submission_package_20260922'
STAGE = PACKAGE / 'bioRxiv_Ver1_stage'
STAGE.mkdir(parents=True, exist_ok=True)

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def tex(s):
    return s.replace('&', r'\&').replace('%', r'\%').replace('_', r'\_').replace('#', r'\#')

common = (
    'Each figure shows (A) joint UMAPs colored by dataset and existing cell-type annotations and '
    '(B) sample-level distributions of three marker features. A includes Raw, Harmony-direct, cyCombine, '
    'CytoNorm, fastMNN, MaxFuse and RAMP; B omits MaxFuse because its output in this analysis is an '
    'embedding without marker-valued coordinates. fastMNN ridges use PCA-reconstructed marker coordinates, '
    'not original intensity units. Within each task, all methods use the same display cells and sample '
    'selection. Each ridge is normalized to its own maximum, so heights describe distribution shape, '
    'not cell abundance. Histograms use 160 bins and Gaussian smoothing with sigma 6 bins. Horizontal '
    'limits are the pooled 0.5th-99.5th percentiles of the complete frozen cohort for each method-marker '
    'pair; scales are method-specific. Ridge colors are fixed by technology across figures, and legend '
    'counts denote displayed sample units. UMAPs retain their original colors and have separate legends. '
    'Application-cohort UMAPs retain their story display protocols; these do not replace the archived '
    'evaluation coordinates underlying Figure 2. Dataset references, corresponding-channel definitions, '
    'alignment settings and evaluation protocols are given in the main Methods.'
)
entries = [
    ('Bone marrow', 'Bone marrow benchmark',
     'CODEX, CyTOF and Triana AbSeq: 88,031 cells, 17 sample units and 14 shared markers. '
     '(A) All frozen cells and the original display labels are retained. (B) CD3, CD4 and CD8 '
     'distributions for all 17 units (12 CODEX, two CyTOF and three AbSeq). Raw values are displayed '
     'after arcsinh transformation with cofactor 1. This benchmark is distinct from the CD19-recovery application.'),
    ('TNBC', 'TNBC benchmark',
     'Tietscher IMC (IMC1), Meyer IMC (IMC2) and Shiao CODEX: 62,701 cells, 11 sample units and '
     '11 shared features. (A) All frozen cells and the original display labels are retained. '
     '(B) CD3, CD8 and PANCK distributions for all 11 units (one IMC1, five IMC2 and five CODEX). '
     'Raw values are displayed after arcsinh transformation with cofactor 1; IMC1 and IMC2 use different orange shades.'),
    ('Melanoma three-platform', 'Three-platform melanoma benchmark',
     'Hickey CODEX, Hoch IMC and Ptacek MIBI: 60,000 cells, 46 sample units and 12 shared features. '
     '(A) All frozen cells are displayed. (B) CD3, CD8 and SOX10 distributions for all 46 units '
     '(six CODEX, 17 IMC and 23 MIBI). Raw values are displayed after arcsinh transformation with '
     'cofactor 1. The Hickey SOX10/S100 channel serves as the corresponding SOX10 feature.'),
    ('Lung cancer benchmark', 'Original lung cancer benchmark',
     'Senosain CyTOF, Cords IMC and Monkman CODEX: 60,000 cells, 30 sample units and eight shared '
     'features. (A) All frozen cells are displayed. (B) CD3, vimentin and the corresponding '
     'epithelial-marker feature (EPI) for all 30 units, ten per platform. Raw values are displayed '
     'after arcsinh transformation with cofactor 1. EPI denotes corresponding platform-specific '
     'channels rather than identical antibody measurements. This task is distinct from Figure S5.'),
    ('Lung HLA-ABC story', 'Lung HLA-ABC application cohort',
     'Senosain CyTOF, Cords IMC and Zheng mIF: 84,000 reference cells from 84 patients and nine '
     'corresponding shared features. (A) All reference cells are displayed using the current story '
     'UMAP protocol. (B) CD3, CD4 and CD8 for a fixed subset of ten patients per platform '
     '(30,000 cells). Patients were selected by deterministic ID-hash ordering, independently of '
     'expression or performance; each selected patient contributes all 1,000 frozen reference cells. '
     'Raw display transformations are arcsinh with cofactor 5 for CyTOF and cofactor 1 for IMC/mIF.'),
    ('Melanoma story', 'Melanoma IMC and CITE-seq application cohort',
     'Moldoveanu IMC and Engelbrecht CITE-seq: 165,686 cells, 38 donors and 15 shared proteins. '
     '(A) All methods display the same 40,000 cells used in the story, 20,000 per platform, with '
     'approximately balanced donor contributions within each platform. (B) CD3, CD8 and CD20 '
     'distributions retain all 165,686 frozen cells and all 38 donors (30 IMC and eight CITE-seq). '
     'Raw display transformations are arcsinh with cofactor 1 for IMC and log1p for CITE-seq protein counts.'),
    ('HNSCC story', 'HNSCC application cohort',
     'Rahim CITE-seq and Haist CODEX: 94,488 cells, 77 donor/patient units and 22 shared proteins. '
     '(A) All frozen cells are displayed using the story UMAP protocol. (B) CD3, CD4 and CD8 '
     'for all three CITE-seq donors and a fixed subset of 27 of 74 CODEX patients (47,488 cells). '
     'CODEX patients were selected by deterministic ID-hash ordering, independently of expression '
     'or performance; all frozen cells from selected units are retained. Raw ridges use log1p '
     'CITE-seq protein counts and provider-supplied signed CODEX values without an additional transformation.'),
]
sources = []
target = STAGE / 'figure/supplementary'
target.mkdir(parents=True, exist_ok=True)
captions = []
for i, (case, title, body) in enumerate(entries, 1):
    pointer = json.loads((ROOT / case / 'CURRENT_PREVIEW.json').read_text())
    source = Path(pointer['pdf'])
    filename = f'figure_s{i}_{case.lower().replace(" ", "_").replace("-", "_")}.pdf'
    shutil.copy2(source, target / filename)
    sources.append(dict(number=f'S{i}', case=case, source=str(source),
                        destination=f'figure/supplementary/{filename}', sha256=sha(source)))
    captions.append(dict(number=f'S{i}', title=title, text=body))
md = '# Supplementary figure legends\n\n## Common display conventions for Figures S1-S7\n\n' + common + '\n\n'
body_tex = r'\section*{Common display conventions for Figures S1--S7}' + '\n' + tex(common) + '\n\n'
for i, cap in enumerate(captions):
    if i == 3:
        body_tex += '\\newpage\n\\section*{Supplementary figure legends (continued)}\n'
    title = f"Supplementary Figure {cap['number']}. {cap['title']}"
    md += f"## {title}\n\n{cap['text']}\n\n"
    body_tex += '\\subsection*{'+tex(title)+'}\n'+tex(cap['text'])+'\n\n'
appendix_note = ('For the four original three-platform tasks, MaxFuse UMAPs show the fixed first-platform '
                 'hub (CODEX in S1 and S3, IMC1 in S2, and CyTOF in S4); S5 shows the CyTOF hub. '
                 'Figure 2 scores for three-platform tasks aggregate all three hub orientations. '
                 'S6 and S7 use pairwise MaxFuse. Figures S1-S7 are descriptive displays; '
                 'no alignments or quantitative benchmark scores were refitted for this supplement.')
md += '## Additional display note\n\n' + appendix_note + '\n'
body_tex += r'\subsection*{Additional display note}' + '\n' + tex(appendix_note) + '\n'
(STAGE / 'SUPPLEMENTARY_CAPTIONS.md').write_text(md)
(STAGE / 'supplementary_captions.tex').write_text(body_tex)
document = r'''\documentclass[11pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern,microtype,graphicx,pdfpages,eso-pic}
\usepackage[margin=18mm]{geometry}
\usepackage{titlesec}
\usepackage[hidelinks,pdftitle={RAMP: Supplementary figures and legends}]{hyperref}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\setlength{\emergencystretch}{2em}
\titleformat{\section}{\large\bfseries}{}{0pt}{}
\titleformat{\subsection}{\normalsize\bfseries}{}{0pt}{}
\titlespacing*{\section}{0pt}{8pt}{5pt}
\titlespacing*{\subsection}{0pt}{9pt}{3pt}
\newcommand{\suppHeader}[1]{%
  \thispagestyle{empty}%
  \AddToShipoutPictureFG*{\AtPageUpperLeft{%
    \put(\LenToUnit{9mm},\LenToUnit{-6mm}){%
      \fontsize{8}{9}\selectfont\sffamily\bfseries Supplementary Figure #1}}}}
\begin{document}
{\Large\bfseries Supplementary figures and legends}\par
{\normalsize\bfseries Rank-Preserving Alignment of Multi-Platform Protein Data for Single-Cell and Spatial Proteomics}\par
\input{supplementary_captions}
\clearpage
'''
for s in sources:
    document += '\\includepdf[pages=1,fitpaper=true,noautoscale=true,scale=1,pagecommand={\\suppHeader{'+s['number']+'}}]{'+s['destination']+'}\n'
document += '\\end{document}\n'
(STAGE / 'supplementary_material.tex').write_text(document)
# Stage a manuscript rebuild without modifying the live manuscript yet.
for filename in ['bibliography.bib', 'ramp_harvard.bst', 'agsm.bst']:
    shutil.copy2(LIVE / filename, STAGE / filename)
for p in (LIVE / 'figure').iterdir():
    if p.is_file():
        q = STAGE / 'figure' / p.name
        if not q.exists():
            q.symlink_to(p.resolve())
text = (LIVE / 'template.tex').read_text()
needle = 'are not seven independent patient cohorts.'
assert text.count(needle) == 1
text = text.replace(needle, needle + '\nSupplementary Figures~S1--S7 provide the complete UMAP and marker-distribution\ndisplays for these seven integration tasks; figure-specific display subsets\nand shared plotting conventions are given in the separate supplementary file.')
(STAGE / 'template.tex').write_text(text)
readme = (LIVE / 'README.md').read_text()
readme += '\n\n## Supplementary figures\n\n- [Supplementary PDF: Figures S1-S7 and concise legends](supplementary_material.pdf)\n- [Supplementary LaTeX](supplementary_material.tex) and [legend source](supplementary_captions.tex)\n- [Editable caption Markdown](SUPPLEMENTARY_CAPTIONS.md)\n- Individual figure PDFs: `figure/supplementary/`\n\nBuild with `latexmk -pdf -interaction=nonstopmode -halt-on-error supplementary_material.tex`. Upload these two supplementary TeX files and the `figure/supplementary/` directory together for Overleaf. The seven source figures remain in `../final_xu/Supplementary Figures/`, with current source paths recorded in `SUPPLEMENTARY_FIGURE_SOURCES.json`. Supplementary Table S1 remains in the main manuscript; the existing CSV source tables are unchanged.\n'
(STAGE / 'README.md').write_text(readme)
(STAGE / 'SUPPLEMENTARY_FIGURE_SOURCES.json').write_text(json.dumps(sources, indent=2) + '\n')
before = {name: sha(LIVE / name) for name in ['template.tex', 'template.pdf', 'README.md']}
(PACKAGE / 'LIVE_BEFORE.json').write_text(json.dumps(before, indent=2) + '\n')
(PACKAGE / 'CAPTION_RECORDS.json').write_text(json.dumps(dict(common=common, captions=captions, additional_note=appendix_note), indent=2) + '\n')
print(STAGE)
