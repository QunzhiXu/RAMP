from pathlib import Path
import json,hashlib,shutil
HERE=Path(__file__).resolve().parent; PREVIEW=HERE.parent; ROOT=PREVIEW.parent.parent
LIVE=ROOT.parent/'bioarxiv_Ver2'
stage=HERE/'stage'; backup=HERE/'backup_before_revision'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for p in [stage,backup,HERE/'build',HERE/'renders']:p.mkdir(exist_ok=True)
assert not (HERE/'BEFORE_HASHES.json').exists(),'Do not overwrite original backup'
before={str(p.relative_to(LIVE)):sha(p) for p in LIVE.rglob('*') if p.is_file()}
(HERE/'BEFORE_HASHES.json').write_text(json.dumps(before,indent=2)+'\n')
protected=json.loads((PREVIEW/'PROTECTED_HASHES.json').read_text())
print('Protected snapshot format',type(protected).__name__)
plan=json.loads((PREVIEW/'PLAN.json').read_text())
pub=json.loads((ROOT/'Supplementary Figures/CURRENT_PUBLICATION.json').read_text())
figures=[]
for item in plan:
    source=list((Path(item['output'])/'output/pdf').glob('*.pdf'))
    assert len(source)==1,source
    prior=next(p for p in pub if p['number']==item['number'])
    dest=Path(prior['destination'])
    if dest.is_absolute():
        # Publication records can point to an older manuscript version; keep the filename only.
        dest=Path('figure/supplementary')/dest.name
    assert (LIVE/dest).exists(),dest
    assert sha(LIVE/dest)==sha(Path(item['old_publication']))
    figures.append(dict(number=item['number'],case=item['case'],source=str(source[0]),destination=str(dest),sha256=sha(source[0]),targets=item['targets']))
replaced={f['destination']:Path(f['source']) for f in figures}
for name in ['main.tex','main.pdf','README.md','PROVENANCE.json','COMPILATION_CHECK.json','FILE_MANIFEST.csv']:
    shutil.copy2(LIVE/name,backup/name)
for rel in replaced:
    (backup/rel).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(LIVE/rel,backup/rel)
for p in LIVE.rglob('*'):
    if p.is_file() and (p.relative_to(LIVE).parts[0] in ['figure','supplementary_data'] or p.suffix in ['.bib','.bst','.cls','.sty']):
        rel=str(p.relative_to(LIVE));target=stage/rel;target.parent.mkdir(parents=True,exist_ok=True)
        if rel in replaced:shutil.copy2(replaced[rel],target)
        else:target.symlink_to(p)
text=(LIVE/'main.tex').read_text()
old='''Curves are peak-normalized; height does not
represent cell number. Figure~\\ref{fig:overview}B shows the union of sample-curve
ranges above 0.5\\% of peak height, padded by 4\\% of this span within the grid;
the RAMP SOX10 left limit is $-1.5$. Each window retains at least 99.8\\% of
observed values; all values contribute to density estimation. Supplementary
ridge axes show the full grid.'''
new='''Curves are peak-normalized; height does not
represent cell number. Compact axes in Figure~\\ref{fig:overview}B and selected
supplementary rows (specified in their legends) show the union of displayed
sample-curve ranges above 0.5\\% of peak height, padded by 4\\% within the grid.
RAMP SOX10 has a $-1.5$ left limit in Figure~\\ref{fig:overview}B and Supplementary
Fig.~\\ref{fig:s3}; other supplementary axes show the full grid. All values
contribute to density estimation, including tails outside compact windows.'''
assert text.count(old)==1
text=text.replace(old,new)
old='Axes span the complete frozen cohort range plus at least four smoothing bandwidths at each end, retaining full tails; scales are method-specific.'
new='Compact axes follow the density-based rule in Methods for Harmony in Supplementary Figs.~\\ref{fig:s1}, \\ref{fig:s2}, \\ref{fig:s4}--\\ref{fig:s7} and RAMP in Supplementary Figs.~\\ref{fig:s3}, \\ref{fig:s5}--\\ref{fig:s7}; other rows show the full histogram grid. Compact windows retain 99.48\\%--100\\% of displayed observations; all observations contribute to density estimation. Scales are method-specific.'
assert text.count(old)==1
text=text.replace(old,new)
(stage/'main.tex').write_text(text)
local_paths=['Supplementary Figures/README.md','Supplementary Figures/CURRENT_PUBLICATION.json',str((PREVIEW/'README.md').relative_to(ROOT))]
local_paths += [f"Supplementary Figures/{f['case']}/CURRENT_PREVIEW.json" for f in figures]
for rel in local_paths:
    out=backup/'local_docs'/rel;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,out)
(HERE/'FIGURES.json').write_text(json.dumps(figures,indent=2)+'\n')
(HERE/'LOCAL_PATHS.json').write_text(json.dumps(local_paths,indent=2)+'\n')
print('Staged seven accepted PDFs and two TeX paragraph edits. Backup complete.')
