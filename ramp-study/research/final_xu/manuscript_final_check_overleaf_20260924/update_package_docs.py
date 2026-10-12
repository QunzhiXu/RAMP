from pathlib import Path
import hashlib,json,shutil,zipfile
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;LIVE=ROOT.parent/'bioarxiv_Ver2';package=HERE/'package';backup=HERE/'backup_before_revision';P=ROOT/'Supplementary Figures/ridge_compact_preview_20260924'
changes={}
for rel in ['supplementary_data/README.md','supplementary_data/ridge_display/README.md','supplementary_data/model_specific_figures/README.md']:
    dest=backup/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(LIVE/rel,dest)
s=(LIVE/'supplementary_data/README.md').read_text().replace('for the four selection/weighting combinations shown in Supplementary Table S1.','for the four selection/weighting combinations described in Methods. These are supplied as CSV data; there is no Supplementary Table S1 in the current manuscript.')
s+='\nRidge audit files under `ridge_display/` distinguish the historical full-grid density calculation from the compact display windows accepted on 2026-09-24; see its README and `COMPACT_WINDOWS_20260924.csv`. Numeric integration and prediction results are unchanged.\n'
changes['supplementary_data/README.md']=s
changes['supplementary_data/ridge_display/README.md']='''# Ridge density and display-window records

The existing `RANGE_AUDIT.csv`, `s1_ridge_display_audit.csv` through `s7_ridge_display_audit.csv`, and `QA_SUMMARY.json` describe the **2026-09-22 full-grid density calculation**. All selected cell values enter the histograms; `n_below`/`n_above` are zero relative to that original grid. The `old_n_below`/`old_n_above` fields refer to still earlier display limits. Gaussian sigma remains 6 bins, with the original expression-unit bandwidth and zero endpoints after full-grid padding. These archived values were not rewritten.

## Current display windows (2026-09-24)

`COMPACT_WINDOWS_20260924.csv` records all 30 updated axes in S1-S7: Harmony in S1, S2, S4-S7 and RAMP in S3, S5-S7. Each window takes the union of positions where any displayed sample's peak-normalized density reaches 0.5% of its own maximum, then adds 4% of the span to each side within the full histogram grid. S3 RAMP SOX10 has the accepted left limit of -1.5. Other axes retain their full-grid ranges.

The compact windows contain 99.48%-100% of displayed observations. Values outside the viewport still contribute to the unchanged density calculation; no artificial zero endpoints are added. `COMPACT_WINDOWS_QA_20260924.json` records the verification of 3,636 unchanged density arrays, sample counts and bandwidths, plus unchanged pixels outside the targeted axes. UMAPs, underlying data, sample selection and quantitative benchmark scores are unchanged.
'''
s=(LIVE/'supplementary_data/model_specific_figures/README.md').read_text().replace('current publication PDFs are listed in `SUPPLEMENTARY_FIGURE_SOURCES.json` at manuscript root (or `reference/` in the upload package).','current publication PDFs are included under `figure/supplementary/` at the project root, and their inclusion and captions are defined in `main.tex`.')
changes['supplementary_data/model_specific_figures/README.md']=s
for rel,s in changes.items():(package/rel).write_text(s)
rel='supplementary_data/ridge_display/COMPACT_WINDOWS_20260924.csv';shutil.copy2(P/'DISPLAY_WINDOW_SUMMARY.csv',package/rel)
old=json.loads((P/'QA.json').read_text())
qa=dict(revision='Accepted supplementary compact ridge windows, 2026-09-24',all_cells_retained_in_density_estimation=True,all_original_counts_and_bandwidths_identical=True,non_target_pixels_identical=True,density_hashes_verified=sum(x['density_hashes_verified'] for x in old),details=[{k:x[k] for k in ['figure','density_hashes_verified','all_original_counts_and_bandwidths_identical','non_target_pixels_identical']} for x in old],display_window_table='COMPACT_WINDOWS_20260924.csv',other_axes='Original full histogram grid',current_figures='figure/supplementary/ at the project root')
(package/'supplementary_data/ridge_display/COMPACT_WINDOWS_QA_20260924.json').write_text(json.dumps(qa,indent=2)+'\n')
readme=(package/'README.md').read_text().replace('201 accompanying data and analysis-record files','203 accompanying data and analysis-record files')
(package/'README.md').write_text(readme);(HERE/'README_NEXT.md').write_text(readme)
files=sorted(p for p in package.rglob('*') if p.is_file());out=HERE/'bioarxiv_Ver2_Overleaf.zip'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in files:z.write(p,p.relative_to(package).as_posix())
# Check the final ZIP, not just the pre-update draft. These changes affect only Markdown/audit files.
with zipfile.ZipFile(out) as z:
    assert z.testzip() is None
    z.extractall(HERE/'zip_compile')
for p in files:assert sha(p)==sha(HERE/'zip_compile'/p.relative_to(package))
(HERE/'PACKAGE_MANIFEST.json').write_text(json.dumps([dict(path=p.relative_to(package).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files],indent=2)+'\n')
(HERE/'DOC_REPLACEMENTS.json').write_text(json.dumps({rel:str(package/rel) for rel in [*changes,'supplementary_data/ridge_display/COMPACT_WINDOWS_20260924.csv','supplementary_data/ridge_display/COMPACT_WINDOWS_QA_20260924.json']},indent=2)+'\n')
print('Final ZIP:',len(files),'files;',out.stat().st_size,'bytes. Three stale documentation references corrected; two display-audit files added.')
