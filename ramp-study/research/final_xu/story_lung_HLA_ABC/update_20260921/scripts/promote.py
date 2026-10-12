from pathlib import Path
import shutil,json,hashlib,datetime
import pandas as pd
S=Path(__file__).resolve().parents[2]; U=S/'update_20260921'; T=S/'test/new_senosain_20260921'; R=S/'releases/senosain140k_20260921'
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
assert json.loads((T/'STATUS.json').read_text())['stage']=='complete'
assert not R.exists()
# Independent release copy. Test and original frozen datasets remain immutable.
shutil.copytree(T,R,ignore=shutil.ignore_patterns('cache','tmp','__pycache__','logs'))
for d in ['tmp','logs']: (R/d).mkdir()
for name in ['Panel B','Panel D','Panel E','Panel F','Combined ABC','Combined ABCDEF']:
 p=S/'Figure'/name; dest=p/'versions/before_senosain140k_20260921'; dest.mkdir(parents=True)
 for q in list(p.iterdir()):
  if q.name in ['versions','cache','tmp']:continue
  if q.is_dir():shutil.copytree(q,dest/q.name)
  else:shutil.copy2(q,dest/q.name)
back=U/'backup_story';back.mkdir()
shutil.copy2(S/'FIGURE_CAPTION.md',back/'FIGURE_CAPTION.md')
if (S/'README.md').exists():shutil.copy2(S/'README.md',back/'README.md')
# Preserve the previous derived summary in full before constructing its replacement.
shutil.move(S/'summary_steps3_6',back/'summary_steps3_6')
Q=S/'summary_steps3_6'
for d in ['03_projection','04_vcaf_markers','05_immune_environment','06_DFS','provenance','scripts']:(Q/d).mkdir(parents=True,exist_ok=True)
# Current patient groups, unchanged original immune counts, new adjusted comparisons.
g=pd.read_csv(R/'analysis/PATIENT_GROUPS.csv');g=g[g.version.eq('new')].copy();assert len(g)==280
old=pd.read_csv(back/'summary_steps3_6/05_immune_environment/PATIENT_SUMMARY.csv')
for col in ['group','high_HLA','all_mean','n_vcaf']:
 old[col]=old.patient_id.map(g.set_index('patient_id')[col])
assert not old[['group','high_HLA','all_mean','n_vcaf']].isna().any().any()
old.to_csv(Q/'05_immune_environment/PATIENT_SUMMARY.csv',index=False)
c=pd.read_csv(R/'analysis/IMMUNE_COMPOSITION.csv');c=c[c.version.eq('new')].copy();c.to_csv(Q/'05_immune_environment/COMPOSITION_COMPARISONS.csv',index=False)
c['full_CAF_covariates']=False;c.to_csv(Q/'05_immune_environment/COMPOSITION_ADJUSTED.csv',index=False)
for f in ['IMMUNE_EXPRESSION.csv','IMMUNE_COMPOSITION.csv']:
 d=pd.read_csv(R/'analysis'/f);d[d.version.eq('new')].to_csv(Q/'05_immune_environment'/f,index=False)
for f in ['VCAF_ROI_MARKER_ASSOCIATIONS.csv','VCAF_PATIENT_MARKER_ASSOCIATIONS.csv','VCAF_MARKER_SUMMARY.csv']:
 d=pd.read_csv(R/'analysis'/f);d[d.version.eq('new')].to_csv(Q/'04_vcaf_markers'/f,index=False)
for f in ['NEW_ALL_PATIENT_SCORES.csv','NEW_HIGH_STATE_THRESHOLDS.csv','COHORT_FLOW.csv','COHORT_CHECKS.json','MOLECULAR_FREEZE.json']:shutil.copy2(R/'analysis'/f,Q/'03_projection'/f)
g[['patient_id','time_years','event','group']].to_csv(Q/'06_DFS/KM_PATIENTS.csv',index=False)
for f in ['KM_SUMMARY.csv','FIVE_YEAR_DFS.csv','COX_TERMS.csv']:
 d=pd.read_csv(R/'analysis'/f);d=d[d.version.eq('new')&d.config.eq('RAMP_k30')&d.state.eq('mean')&d.cohort.eq('all280')].copy()
 d['id']='RAMP_k30__mean__all280';d['n']=280;d['events']=96
 if f=='FIVE_YEAR_DFS.csv':
  d['n']=d['group'].map(g.groupby('group').size())
  d['events']=d['group'].map(g.groupby('group').event.sum())
 if f=='COX_TERMS.csv':d['model']='matched_full';d['id']+='__matched_full'
 d.to_csv(Q/'06_DFS'/f,index=False)
# Stage latest manuscript without modifying the live directory.
M=Path('/Users/qxu6/Documents/New project/bioRxiv_Ver1');stage=U/'bioRxiv_Ver1';stage.mkdir()
for p in M.iterdir():
 if p.is_file() and p.suffix in ['.tex','.bib','.bst','.cls','.sty','.bbl','.md']:shutil.copy2(p,stage/p.name)
shutil.copytree(M/'figure',stage/'figure')
save(U/'MANUSCRIPT_BEFORE.json',{str(p.relative_to(M)):sha(p) for p in M.rglob('*') if p.is_file() and (p.parent==M or p.parent==M/'figure')})
save(S/'CURRENT_ANALYSIS.json',{'version':'senosain140k_20260921','release':'releases/senosain140k_20260921','figure':'Figure/Combined ABCDEF/panels_ABCDEF_preview.png','source_test':'test/new_senosain_20260921','summary':'summary_steps3_6','legacy_frozen_packages':['data_umap','umap','validation','data_cords','survival','mechanism_support'],'note':'Legacy packages preserve the previous reproducible analysis; current scientific outputs are the versioned release.'})
save(R/'RELEASE_MANIFEST.json',{str(p.relative_to(R)):sha(p) for p in R.rglob('*') if p.is_file()})
print('Backups, independent release, current summary, and manuscript staging complete.')
