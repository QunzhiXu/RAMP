"""Materialize full native panels for EXACT frozen story cells, without refitting RAMP."""
import os,sys,json,hashlib,ast,argparse
from types import SimpleNamespace
from pathlib import Path
sys.dont_write_bytecode=True
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[k]='1'
import numpy as np
import pandas as pd
import h5py
from scipy import sparse
MODULE=Path(__file__).resolve().parent
from hnscc_support import check_inputs, claim, publish_new, fail
HERE=COMP=ROOT=RAHIM_ADT=HAIST_CELLS=None
STORY=MODULE/'source'
SOURCES={}
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def track(p):
 p=Path(p).resolve();SOURCES[str(p)]=sha(p);return p
def read(p,**kw):return pd.read_csv(track(p),low_memory=False,float_precision='round_trip',**kw)
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def module(p,name):
 # Extract only the pinned literal marker map; never execute the historical sampler.
 if Path(p).resolve()!=MODULE/'source/alignment/01_prepare_alignment_input.py':raise ValueError("Unexpected marker-map source")
 tree=ast.parse(track(p).read_text())
 node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MARKER_MAP' for t in n.targets))
 return SimpleNamespace(MARKER_MAP=ast.literal_eval(node.value),RAHIM_ADT=RAHIM_ADT,HAIST_CELLS=HAIST_CELLS)
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

def hnscc():
 mod=module(STORY/'alignment/01_prepare_alignment_input.py','original_input')
 ds=CFG['datasets'][0];wanted=MD.loc[MD.dataset.eq(ds),'cell_id'].to_numpy()
 with h5py.File(track(mod.RAHIM_ADT),'r') as f:
  names=f['var/_index'].asstr()[:].tolist();src=f['obs/_index'].asstr()[:];at=positions('RAHIM|'+pd.Series(src),wanted)
  g=f['X'];raw=sparse.csr_matrix((g['data'][:],g['indices'][:],g['indptr'][:]),shape=(len(src),len(names))).toarray()[at]
  donors=obs(f,'patient')[at]
 save_view(ds,'rahim_adt',raw,names,wanted,donors,{m:mod.MARKER_MAP[m][0] for m in SHARED},'log1p',{n:'isotype control' for n in names if 'isotype' in n.lower()})
 ds=CFG['datasets'][1];wanted=MD.loc[MD.dataset.eq(ds),'cell_id'].to_numpy();p=track(mod.HAIST_CELLS)
 header=pd.read_csv(p,nrows=0).columns.tolist();names=header[:header.index('UID')];pieces=[];wanted_source={i.removeprefix('HAIST|') for i in wanted}
 for chunk in pd.read_csv(p,usecols=[*names,'UID','Pat_ID'],dtype={'UID':str},chunksize=100000,low_memory=False,float_precision='round_trip'):
  pieces.append(chunk.loc[chunk.UID.isin(wanted_source)].copy())
 table=pd.concat(pieces,ignore_index=True);at=positions('HAIST|'+table.UID,wanted);table=table.iloc[at]
 save_view(ds,'haist_codex',table[names].to_numpy(float),names,wanted,table.Pat_ID.astype(int).astype(str),{m:mod.MARKER_MAP[m][1] for m in SHARED},'identity',{})


def main(argv=None):
 global HERE,COMP,RAHIM_ADT,HAIST_CELLS,CFG,MD,REF,SHARED,SOURCES,views
 parser=argparse.ArgumentParser(description=__doc__)
 for name in ('comparison-root','rahim-adt','haist-cells','output'):
  parser.add_argument('--'+name,type=Path,required=True)
 args=parser.parse_args(argv)
 COMP=args.comparison_root.resolve(strict=True)
 RAHIM_ADT=args.rahim_adt.resolve(strict=True);HAIST_CELLS=args.haist_cells.resolve(strict=True)
 inputs=check_inputs(COMP,RAHIM_ADT,HAIST_CELLS)
 out=claim(args.output,COMP,RAHIM_ADT,HAIST_CELLS,inputs)
 HERE=out/'_work';HERE.mkdir();(HERE/'data').mkdir()
 SOURCES={};views=[]
 try:
  CFG=json.loads(track(COMP/'run_config.json').read_text())
  if CFG['case']!='hnscc_story':raise ValueError("Only HNSCC is supported")
  MD=read(COMP/'input/fit_metadata.csv.gz',dtype=str)
  REF=np.load(track(COMP/'input/raw.npy'))
  SHARED=CFG['markers']
  hnscc()
  assert len(views)==len(CFG['datasets'])
  for p,h in SOURCES.items():assert sha(p)==h
  done=HERE/'data/manifest.json'
  outputs={str(p.relative_to(HERE/'data')):sha(p) for p in (HERE/'data').rglob('*') if p.is_file() and p!=done}
  dump(done,{'case':CFG['case'],'views':views,'source_sha256':SOURCES,'output_sha256':outputs,'fit_uses_labels':False,'RNA_used':False,'story_targets_in_active':'All usable native proteins, including story targets where present. Integration benchmark only; not a held-out target prediction test.','protocol':'shared unchanged (9/15/22); active full usable native proteins; exact original cells; original per-assay transforms extended to full panel; provider QC exclusions; zscore sample-SD','original_input_manifest_sha256':sha(COMP/'input/manifest.json')})
  dump(HERE/'run_config.json',{**CFG,'comparison':str(out),'scope':'full native active protein panel; exact original story cells and shared marker list; per-assay transforms unchanged','full_input_manifest_sha256':sha(done)})
  check_inputs(COMP,RAHIM_ADT,HAIST_CELLS)
  publish_new(HERE,out)
 except BaseException as error:
  fail(out,error);raise
 print(out,flush=True)

if __name__=='__main__':main()
