"""Materialize full native panels for EXACT frozen story cells, without refitting RAMP."""
import os,sys,json,hashlib,argparse,subprocess,tempfile
from pathlib import Path
sys.dont_write_bytecode=True
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[k]='1'
import numpy as np
import pandas as pd
import h5py
from scipy import sparse
MODULE=Path(__file__).resolve().parent
from lung_story_support import check_inputs, claim, publish_new, fail, check_export, SOURCE_HASHES
HERE=COMP=CELL_METADATA=SENOSAIN_ROOT=CORDS_RDS=ZHENG_H5AD=None
SOURCES={}
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def track(p):
 p=Path(p).resolve();SOURCES[str(p)]=sha(p);return p
def read(p,**kw):return pd.read_csv(track(p),low_memory=False,float_precision='round_trip',**kw)
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
CFG=MD=REF=SHARED=None
views=[]
def obs(f,key):
 d=f['obs'][key]
 if isinstance(d,h5py.Group):return d['categories'].asstr()[:][d['codes'][:]]
 if '__categories' in f['obs'] and key in f['obs/__categories']:return f['obs/__categories'][key].asstr()[:][d[:]]
 return d.asstr()[:] if d.dtype.kind in 'OSU' else d[:]
def positions(source_ids,expected):
 idx=pd.Index(np.asarray(source_ids,dtype=str));assert idx.is_unique
 at=idx.get_indexer(np.asarray(expected,dtype=str));assert (at>=0).all();return at

def save_view(ds,slug,raw,names,ids,donors,mapping,op,excluded,extra=None):
 names=list(names);raw=np.asarray(raw,dtype=np.float64)
 rows=MD.dataset.eq(ds).to_numpy();md=MD.loc[rows].reset_index(drop=True)
 assert raw.shape==(len(md),len(names)) and len(set(names))==len(names)
 assert np.array_equal(np.asarray(ids,dtype=str),md.cell_id.to_numpy())
 assert np.isfinite(raw).all(),slug
 mapped=[names.index(mapping[m]) for m in SHARED]
 err=float(np.abs(raw[:,mapped]-REF[rows]).max());assert err<1e-10,(slug,'source shared values differ',err)
 # Verify patient identity from the provider independently of the marker values.
 expected_donors=md.unit_key.str.split('|').str[-1] if CFG['case']=='lung_story' else None
 if expected_donors is None:
  if CFG['case']=='hnscc_story':expected_donors=md.unit_key.str.removeprefix('GSE212797_').str.removeprefix('HAIST_')
  else:expected_donors=md.unit_key.str.split('::').str[-1]
 assert np.array_equal(np.asarray(donors,dtype=str),expected_donors.to_numpy()),(slug,'donor mismatch')
 p=HERE/'data'/slug;p.mkdir(exist_ok=True)
 np.save(p/'raw_all.npy',raw);(p/'raw_feature_names.txt').write_text('\n'.join(names)+'\n')
 np.save(p/'cell_ids.npy',md.cell_id.to_numpy(dtype=str));np.save(p/'global_row_indices.npy',np.flatnonzero(rows))
 md.assign(donor=np.asarray(donors,dtype=str)).to_csv(p/'cell_metadata.csv.gz',index=False,compression={'method':'gzip','mtime':0})
 keep=[j for j,n in enumerate(names) if n not in excluded]
 x=raw[:,keep]
 if op=='asinh5':ready=np.arcsinh(x/5)
 elif op=='asinh1':ready=np.arcsinh(x)
 elif op=='log1p':assert (x>=0).all();ready=np.log1p(x)
 elif op=='identity':ready=x.copy()
 else:raise ValueError(op)
 scale=ready.std(axis=0,ddof=1);good=scale>1e-12
 for j,ok in enumerate(good):
  if not ok:excluded[names[keep[j]]]='zero variance on the frozen cells (predefined numerical guard)'
 keep=np.asarray(keep)[good];ready=ready[:,good];active_names=[names[i] for i in keep]
 mu=ready.mean(axis=0);sd=ready.std(axis=0,ddof=1);active=(ready-mu)/sd
 assert np.isfinite(active).all() and (active.std(axis=1)>1e-12).all()
 jj=[active_names.index(mapping[m]) for m in SHARED];shared=active[:,jj]
 for n,a in [('active_raw',raw[:,keep]),('active_preprocessed',ready),('active',active),('shared',shared),('mean',mu),('scale',sd)]:np.save(p/(n+'.npy'),a)
 (p/'active_feature_names.txt').write_text('\n'.join(active_names)+'\n');(p/'shared_feature_names.txt').write_text('\n'.join(SHARED)+'\n')
 pd.DataFrame([{'raw_feature':n,'in_active':n in active_names,'exclusion_reason':excluded.get(n,'')} for n in names]).to_csv(p/'feature_inventory.csv',index=False)
 pd.DataFrame({'shared_marker':SHARED,'native_feature':[mapping[m] for m in SHARED],'active_index':jj}).to_csv(p/'shared_mapping.csv',index=False)
 record={'dataset':ds,'slug':slug,'cells':len(raw),'donors':len(set(donors)),'raw_features':len(names),'active_features':len(active_names),'shared_features':len(SHARED),'preprocessing':op+' then dataset-wise marker z-score (sample SD, ddof=1)','shared_is_exact_active_slice':True,'raw_shared_max_absolute_error_vs_original_RAMP_input':err,'cell_order_verified':True,'donor_identity_verified':True,'excluded_features':excluded,'extra':extra or {}}
 dump(p/'manifest.json',record);views.append(record);print('PREPARED',slug,raw.shape,'active',len(active_names),'shared error',err,flush=True)

def lung():
 md=read(CELL_METADATA,dtype={'patient_id':str})
 assert np.array_equal(md.cell_key.astype(str),MD.cell_id)
 ds=CFG['datasets'][0];m=md.loc[md.dataset.eq(ds)];src=SENOSAIN_ROOT
 names=read(src/'senosain_140k_markers35.csv').marker.tolist();at=m.source_row_id.to_numpy(int)
 raw=np.load(track(src/'senosain_140k_raw35.npy'))[at]
 sm=read(src/'senosain_140k_cell_metadata.csv.gz',usecols=['source_cell_id','patient_id'],dtype=str).iloc[at]
 assert np.array_equal(sm.source_cell_id.to_numpy(),m.source_cell_id.to_numpy())
 save_view(ds,'senosain_cytof',raw,names,m.cell_key.to_numpy(),sm.patient_id.to_numpy(),{m:('CYTOKERATIN' if m=='EPI' else m) for m in SHARED},'asinh5',{'CD90':'not uniformly measured across provider cohort; same exclusion as old MaxFuse lung benchmark'})
 ds=CFG['datasets'][1];m=md.loc[md.dataset.eq(ds)];p=HERE/'data/cords_imc';record=json.loads((p/'EXPORT_COMPLETE.json').read_text());SOURCES[str(CORDS_RDS)]=record['source_sha256']
 names=(p/'raw_feature_names.txt').read_text().splitlines();raw=np.fromfile(p/'counts.float64.bin',dtype='<f8').reshape(len(m),len(names));identity=pd.read_csv(p/'source_identity.csv',dtype={'patient_id':str})
 assert np.array_equal(identity.cell_id.astype(str),m.cell_key.astype(str)) and np.array_equal(identity.source_row_id,m.source_row_id)
 mapping={'CD3':'CD3','CD4':'CD4','CD8':'CD8a','CD31':'vWF + CD31','KI67':'Ki-67','VIMENTIN':'Vimentin','EPI':'Pan Cytokeratin + Keratin Epithelial','CD45':'CD45RA + CD45R0','HLA_DR':'HLA-DR'}
 save_view(ds,'cords_imc',raw,names,m.cell_key.to_numpy(),identity.patient_id.to_numpy(),mapping,'asinh1',{n:'DNA channel' for n in names if n in ['Iridium_191','Iridium_193']},extra={'provider_c_counts_asinh_also_saved':'c_counts_asinh.float64.bin','source_scale_choice':'asinh(counts), identical to initial story comparison; no additional percentile censoring introduced'})
 ds=CFG['datasets'][2];m=md.loc[md.dataset.eq(ds)];p=track(ZHENG_H5AD);at=m.source_row_id.to_numpy(int)
 with h5py.File(p,'r') as f:
  names=f['var/_index'].asstr()[:].tolist();order=np.argsort(at);raw=f['X'][at[order],:][np.argsort(order)];donors=obs(f,'patient_id')[at]
  assert np.array_equal(obs(f,'CellID')[at].astype(str),m.source_cell_id.astype(str))
 z=['CD3e','CD4','CD8','CD31','Ki67','Vimentin','Pan-Cytokeratin','CD45','HLA-DR']
 save_view(ds,'zheng_mif',raw,names,m.cell_key.to_numpy(),donors,dict(zip(SHARED,z)),'asinh1',{})


def export_cords(rscript):
 # Derive exactly the old requested rows, not a new donor/cell selection.
 metadata=read(CELL_METADATA,dtype={'patient_id':str})
 request=metadata.loc[metadata.dataset.eq('Cords IMC'),['cell_key','source_row_id','patient_id','source_cell_id']]
 folder=HERE/'data/cords_imc';folder.mkdir(exist_ok=False)
 request.to_csv(folder/'requested_cells.csv',index=False)
 expected=json.loads((MODULE/'source/historical_manifest.json').read_text())['output_sha256']
 if sha(folder/'requested_cells.csv')!=expected['cords_imc/requested_cells.csv']:raise ValueError("Cords requested rows changed")
 cmd=[str(rscript),'--vanilla',str(MODULE/'source/export_cords.R'),str(CORDS_RDS),str(folder)]
 dump(HERE/'CORDS_EXPORT_COMMAND.json',{'command':cmd,'script_sha256':SOURCE_HASHES['source/export_cords.R']})
 # R 4.3 rejects spaces in R_TempDir; only disposable R scratch goes here.
 # All selected-cell exports and failure logs remain in the owned run directory.
 with tempfile.TemporaryDirectory(prefix='ramp_lung_export_',dir='/private/tmp') as scratch:
  env=os.environ.copy();env['TMPDIR']=scratch
  with (HERE/'export_cords.stdout.log').open('x') as stdout,(HERE/'export_cords.stderr.log').open('x') as stderr:
   result=subprocess.run(cmd,cwd=HERE,env=env,stdout=stdout,stderr=stderr)
 if result.returncode:raise RuntimeError("Cords R export failed; see retained logs")
 check_export(folder,CORDS_RDS)
 print('Cords provider export verified',flush=True)

def main(argv=None):
 global HERE,COMP,CELL_METADATA,SENOSAIN_ROOT,CORDS_RDS,ZHENG_H5AD,CFG,MD,REF,SHARED,SOURCES,views
 parser=argparse.ArgumentParser(description=__doc__)
 for name in ('comparison-root','cell-metadata','senosain-root','cords-rds','zheng-h5ad','rscript','output'):
  parser.add_argument('--'+name,type=Path,required=True)
 args=parser.parse_args(argv)
 COMP=args.comparison_root.resolve(strict=True)
 CELL_METADATA=args.cell_metadata.resolve(strict=True);SENOSAIN_ROOT=args.senosain_root.resolve(strict=True)
 CORDS_RDS=args.cords_rds.resolve(strict=True);ZHENG_H5AD=args.zheng_h5ad.resolve(strict=True)
 rscript=args.rscript.absolute()
 if not rscript.is_file():raise FileNotFoundError(rscript)
 inputs=check_inputs(COMP,CELL_METADATA,SENOSAIN_ROOT,CORDS_RDS,ZHENG_H5AD)
 out=claim(args.output,COMP,CELL_METADATA,SENOSAIN_ROOT,CORDS_RDS,ZHENG_H5AD,inputs,rscript)
 HERE=out/'_work';HERE.mkdir();(HERE/'data').mkdir()
 SOURCES={};views=[]
 try:
  CFG=json.loads(track(COMP/'run_config.json').read_text())
  if CFG['case']!='lung_story':raise ValueError("Only lung story is supported")
  MD=read(COMP/'input/fit_metadata.csv.gz',dtype=str)
  REF=np.load(track(COMP/'input/raw.npy'))
  SHARED=CFG['markers']
  export_cords(rscript)
  lung()
  assert len(views)==len(CFG['datasets'])
  for p,h in SOURCES.items():assert sha(p)==h
  done=HERE/'data/manifest.json'
  outputs={str(p.relative_to(HERE/'data')):sha(p) for p in (HERE/'data').rglob('*') if p.is_file() and p!=done}
  dump(done,{'case':CFG['case'],'views':views,'source_sha256':SOURCES,'output_sha256':outputs,'fit_uses_labels':False,'RNA_used':False,'story_targets_in_active':'All usable native proteins, including story targets where present. Integration benchmark only; not a held-out target prediction test.','protocol':'shared unchanged (9/15/22); active full usable native proteins; exact original cells; original per-assay transforms extended to full panel; provider QC exclusions; zscore sample-SD','original_input_manifest_sha256':sha(COMP/'input/manifest.json')})
  dump(HERE/'run_config.json',{**CFG,'comparison':str(out),'scope':'full native active protein panel; exact original story cells and shared marker list; per-assay transforms unchanged','full_input_manifest_sha256':sha(done)})
  check_inputs(COMP,CELL_METADATA,SENOSAIN_ROOT,CORDS_RDS,ZHENG_H5AD)
  publish_new(HERE,out)
 except BaseException as error:
  fail(out,error);raise
 print(out,flush=True)

if __name__=='__main__':main()
