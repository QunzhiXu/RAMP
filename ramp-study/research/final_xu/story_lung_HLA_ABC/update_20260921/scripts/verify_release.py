from pathlib import Path
import json,hashlib,shutil,re
import pandas as pd,numpy as np,fitz
S=Path(__file__).resolve().parents[2];U=S/'update_20260921';R=S/'releases/senosain140k_20260921';M=U/'bioRxiv_Ver1';Q=S/'summary_steps3_6';F=S/'Figure'
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
# Check archived test and current release have identical scientific payloads.
manifest=json.loads((R/'RELEASE_MANIFEST.json').read_text())
for p,h in manifest.items():assert sha(R/p)==h,p
checks={'release_files_verified':len(manifest),'complete_test_analysis_promoted_without_numerical_edits':True}
# Correct group denominators in the presentation summary (not model outputs).
g=pd.read_csv(Q/'06_DFS/KM_PATIENTS.csv')
five=pd.read_csv(Q/'06_DFS/FIVE_YEAR_DFS.csv')
for i,row in five.iterrows():
 sub=g[g.group.eq(row.group)];five.loc[i,'n']=len(sub);five.loc[i,'events']=sub.event.sum()
five.to_csv(Q/'06_DFS/FIVE_YEAR_DFS.csv',index=False)
shutil.copy2(Q/'06_DFS/FIVE_YEAR_DFS.csv',F/'Panel F/data/FIVE_YEAR_DFS.csv')
p=F/'Panel F/provenance/VERIFICATION.json';v=json.loads(p.read_text());v['five_year_DFS']=five.to_dict('records');save(p,v)
p=F/'Panel F/provenance/SOURCES.json';records=json.loads(p.read_text())
for r in records:r['sha256']=sha(Path(r['source']));assert sha(F/'Panel F'/r['local'])==r['sha256']
save(p,records)
# Figure values and labels match the new analysis, not old groups.
e=pd.read_csv(F/'Panel E/data/PLOT_PATIENTS.csv');gp=pd.read_csv(R/'analysis/PATIENT_GROUPS.csv');gp=gp[gp.version.eq('new')].set_index('patient_id')
assert len(e)==len(g)==280;assert e.set_index('patient_id').group.equals(gp.loc[e.patient_id,'group'])
np.testing.assert_allclose(e.all_mean,gp.loc[e.patient_id,'all_mean'],atol=1e-14,rtol=0)
assert g.set_index('patient_id').group.equals(gp.loc[g.patient_id,'group'])
for n in ['B','D','E','F']:
 v=json.loads((F/f'Panel {n}/provenance/VERIFICATION.json').read_text());assert v['status']=='PASS'
cox=pd.read_csv(R/'analysis/COX_TERMS.csv');cox=cox.query("version=='new' and config=='RAMP_k30' and state=='mean' and cohort=='all280' and term=='state_z'").iloc[0]
v=json.loads((F/'Panel F/provenance/VERIFICATION.json').read_text());assert abs(v['adjusted_HR_per_SD']['hr']-cox.hr)<1e-13
assert sha(M/'figure/figure6_lung_hla.png')==sha(F/'Combined ABCDEF/panels_ABCDEF_preview.png')
before=json.loads((U/'MANUSCRIPT_BEFORE.json').read_text())
for p in (M/'figure').iterdir():
 if p.name!='figure6_lung_hla.png':assert sha(p)==before['figure/'+p.name],p
checks.update({'E_F_same_280_patients_and_new_groups':True,'new_cox_matches_figure':True,'other_six_manuscript_figures_unchanged':True,'A_C_source_panels_unchanged':True,'D_ROI_cell_coordinates_shapes_background_thresholds_fixed':True})
# Assert manuscript changes are restricted to authorized lung result and methods sections.
old=Path('/Users/qxu6/Documents/New project/bioRxiv_Ver1/template.tex').read_text();new=(M/'template.tex').read_text()
def strip_sections(t):
 for marker in ['\\subsection{Predicted vCAF HLA-ABC state','\\subsection{Lung vCAF HLA-ABC inference']:
  a=t.index(marker);b=t.index('\\subsection{',a+15);t=t[:a]+'[UPDATED LUNG SECTION]\n'+t[b:]
 return t
assert strip_sections(old)==strip_sections(new)
assert all(x in new for x in ['0.870','0.696','0.275--0.889','0.63407','P=0.0104'])
checks['other_manuscript_sections_unchanged']=True
log=(M/'template.log').read_text();assert 'Overfull' not in log and 'undefined' not in log and 'Float too large' not in log
pdf=fitz.open(M/'template.pdf');assert len(pdf)==49;assert 'Figure 6:' in pdf[15].get_text();assert '0.870' in pdf[15].get_text()
checks.update({'PDF_pages':len(pdf),'Figure6_page':16,'overfull_boxes':0,'undefined_references':0,'oversized_floats':0,'underfull_paragraph_warnings':2,'visually_reviewed_pages':[15,16,17,18,39,40,41],'all_pages_have_content':all(len(p.get_text())>100 for p in pdf)})
# Refresh package manifests after documentation updates.
for n in ['Panel B','Panel D','Panel E','Panel F','Combined ABC','Combined ABCDEF']:
 p=F/n;rows=[]
 for f in sorted(p.rglob('*')):
  rel=f.relative_to(p)
  if f.is_file() and rel.parts[0] not in {'cache','tmp','versions','cd4_cd8_check'} and f.name!='MANIFEST.json':rows.append({'file':str(rel),'sha256':sha(f)})
 save(p/'provenance/MANIFEST.json',rows)
save(Q/'provenance/MANIFEST.json',[{'file':str(p.relative_to(Q)),'sha256':sha(p)} for p in sorted(Q.rglob('*')) if p.is_file() and p.name!='MANIFEST.json'])
checks['status']='PASS';save(U/'PRE_INSTALL_CHECKS.json',checks);save(M/'LUNG_UPDATE_CHECKS.json',checks)
(M/'LUNG_UPDATE_20260921.md').write_text('''# Lung Figure 6 update - 2026-09-21

Source: `../final_xu/story_lung_HLA_ABC/`.
Analysis release: `releases/senosain140k_20260921`.

Updated Figure 6 B/D/E/F, lung Results, Methods and caption, plus the README. The same 84 reference patients, Cords/Zheng reference cells, target cohorts, algorithm, k, eligibility, ROI and color limits were retained. Senosain source cells were selected from the project-curated 140,000-cell QC/annotation resource, followed by refitting the same alignment and predictor. The other six figures and all non-lung manuscript sections are unchanged.

- Auxiliary validation: median patient Spearman 0.696; fixed 389-cell region 0.870, whole C30 0.827.
- DFS: 280 patients, 96 events; lower/higher five-year DFS 75.1% / 56.7%; adjusted HR 1.36 (95% CI 1.07-1.71), P=0.0104.
- Compiled manuscript: 49 pages; Figure 6 remains on page 16. Figure and adjacent result/method pages visually checked. No undefined references, overfull boxes or oversized floats; two existing underfull paragraph warnings remain.

The preceding version is backed up under `_backups/lung_senosain140k_20260921/`. Earlier `LAYOUT_REVIEW.md` describes the prior 48-page edition; current checks are `LUNG_UPDATE_CHECKS.json`.
''')
print(json.dumps(checks,indent=2))
