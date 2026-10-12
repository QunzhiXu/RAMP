from pathlib import Path
import csv,hashlib,json,re,shutil,zipfile
ROOT=Path(__file__).resolve().parent.parent;HERE=Path(__file__).resolve().parent;LIVE=ROOT.parent/'bioarxiv_Ver2'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (HERE/'BEFORE_HASHES.json').exists(),'Backup already exists; do not replace'
backup=HERE/'backup_before_revision';backup.mkdir()
before={str(p.relative_to(LIVE)):sha(p) for p in LIVE.rglob('*') if p.is_file()}
(HERE/'BEFORE_HASHES.json').write_text(json.dumps(before,indent=2)+'\n')
for name in ['main.tex','README.md','COMPILATION_CHECK.json','PROVENANCE.json','FILE_MANIFEST.csv','bioarxiv_Ver2_Overleaf.zip']:
    shutil.copy2(LIVE/name,backup/name)
tex=(LIVE/'main.tex').read_text()
assert r'\usepackage[numbers,square,sort&compress]{natbib}' in tex
assert r'\bibliographystyle{ramp_numeric}' in tex
assert not re.search(r'\b(TODO|FIXME|TBD|PLACEHOLDER|XXX)\b|/Users/|Harmony-direct|Harmony\s*\(arcsinh\)',tex)
assert len(re.findall(r'\\paperfig\{',tex))==7
assert len(re.findall(r'\\suppfigurepage\{',tex))==13
# Resolve all actual figure dependencies, including the paperfig wrapper.
assets=[]
for stem in re.findall(r'\\paperfig\{([^}]+)\}',tex):
    hits=[LIVE/'figure'/(stem+ext) for ext in ['.pdf','.png','.jpg'] if (LIVE/'figure'/(stem+ext)).is_file()]
    assert len(hits)==1,(stem,hits);assets.append(hits[0])
assets += [LIVE/r for r in re.findall(r'\\suppfigurepage\{([^}]+)\}',tex)]
assert len(assets)==20 and all(p.is_file() for p in assets)
# All internal label references resolve in this single source.
labels=set(re.findall(r'\\label\{([^}]+)\}',tex))
labels.update(re.findall(r'\}\{(fig:[^}]+)\}',tex))
refs=set(re.findall(r'\\(?:ref|eqref)\{([^}]+)\}',tex))
assert not refs-labels,refs-labels
assert r'\input{' not in tex and r'\include{' not in tex
# Publication pointers agree with the installed supplementary assets.
prov=json.loads((LIVE/'PROVENANCE.json').read_text())
for item in prov['supplementary']:
    assert sha(LIVE/item['destination'])==item['sha256'],item['number']
assert sha(assets[0])==prov['figure1']['sha256']
readme='''# RAMP manuscript and supplementary material

This Overleaf project contains the complete manuscript and supplementary material in one LaTeX source.

## Compile in Overleaf

1. Upload `bioarxiv_Ver2_Overleaf.zip` as a new project.
2. Set the main document to `main.tex` and the compiler to **pdfLaTeX**.
3. Recompile. Overleaf runs BibTeX automatically; if references remain unresolved after replacing an existing project, use **Recompile from scratch**.

The current checked output has **64 pages**, **7 main figures**, **13 supplementary figures (S1-S13)** and **63 numbered references**. Both main and supplementary content are included; this is not a supplementary-only project.

## Files

- `main.tex`: complete text, Methods, figure legends and supplementary section.
- `bibliography.bib`: bibliography database.
- `ramp_numeric.bst`: required numeric bibliography style, ordered by first citation.
- `figure/`: the 20 figure files used by `main.tex`.
- `supplementary_data/`: 201 accompanying data and analysis-record files; retained with the submission source, but not needed by the LaTeX compiler.

No other project-specific `.tex`, `.sty` or `.cls` files are required. Standard packages are supplied by Overleaf/TeX Live. The unused legacy `ramp_harvard.bst`, generated PDF, local backups and compilation intermediates are excluded from the upload ZIP. The compiled `main.pdf` is provided separately in the manuscript folder.

## Current figure and caption status (2026-09-24)

- Figure 1 uses the accepted compact ridge windows, with the RAMP SOX10 left limit at -1.5.
- Figure 3 uses the readable A-C layout and corresponding legends; removed D-F panels are not referenced.
- Figure 1 and S1-S7 use **Raw** as the ridge row label; exact input transformations remain in Methods and legends.
- S1-S7 use the accepted compact windows for Harmony in S1, S2, S4-S7 and RAMP in S3, S5-S7. Other rows retain the full histogram grid. Density estimates, cells, bandwidths, colors and UMAPs are unchanged; all values, including tails outside the display windows, contribute to density estimation.
- Main-figure and supplementary legends have been checked against the current assets. The Overleaf ZIP includes these updates and the numeric citation style.

## Cross-references

Use `(Fig.~\\ref{fig:overview})` and `(Supplementary Fig.~\\ref{fig:s3})` in text. All main and supplementary labels resolve within `main.tex`; no external-document cross-reference package is needed.

Main figure labels: `fig:overview`, `fig:benchmark`, `fig:trainability`, `fig:cd19`, `fig:melanoma`, `fig:lunghla`, `fig:hnscc`. Supplementary labels: `fig:s1` through `fig:s13`.

For local compilation: `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`.
'''
(HERE/'README_NEXT.md').write_text(readme)
package=HERE/'package';package.mkdir()
for name in ['main.tex','bibliography.bib','ramp_numeric.bst']:shutil.copy2(LIVE/name,package/name)
for p in assets:
    out=package/p.relative_to(LIVE);out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,out)
for p in (LIVE/'supplementary_data').rglob('*'):
    if p.is_file() and not p.name.startswith('.'):
        out=package/p.relative_to(LIVE);out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,out)
shutil.copy2(HERE/'README_NEXT.md',package/'README.md')
files=sorted(p for p in package.rglob('*') if p.is_file())
assert len(list((package/'supplementary_data').rglob('*')))>200
out=HERE/'bioarxiv_Ver2_Overleaf.zip'
with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in files:z.write(p,p.relative_to(package).as_posix())
extract=HERE/'zip_compile';extract.mkdir()
with zipfile.ZipFile(out) as z:
    assert z.testzip() is None
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
    z.extractall(extract)
for p in files:assert sha(p)==sha(extract/p.relative_to(package))
(HERE/'PACKAGE_MANIFEST.json').write_text(json.dumps([dict(path=p.relative_to(package).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files],indent=2)+'\n')
print(json.dumps(dict(zip=str(out),files=len(files),figures=len(assets),data_files=len([p for p in files if p.relative_to(package).parts[0]=='supplementary_data']),size_mb=round(out.stat().st_size/1e6,2),tex_unchanged=sha(package/'main.tex')==before['main.tex'],unresolved_labels=[]),indent=2))
