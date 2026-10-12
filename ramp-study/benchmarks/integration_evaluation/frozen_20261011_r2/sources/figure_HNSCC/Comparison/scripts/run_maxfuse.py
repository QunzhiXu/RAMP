import os,sys,json,subprocess,hashlib,time
from pathlib import Path
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parent.parent
ALG=ROOT/'algorithm/maxfuse'
ENV=os.environ.copy();ENV.update(PYTHONDONTWRITEBYTECODE='1',NUMBA_CACHE_DIR=str(HERE/'.cache/numba'),MPLCONFIGDIR=str(HERE/'.cache/matplotlib'),XDG_CACHE_HOME=str(HERE/'.cache/xdg'))
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def main():
 cfg=json.loads((HERE/'run_config.json').read_text());datasets=cfg['datasets'];markers=np.array(cfg['markers'],dtype=str)
 x=np.load(HERE/'input/maxfuse_input.npy');md=pd.read_csv(HERE/'input/fit_metadata.csv.gz');ids=np.load(HERE/'input/cell_ids.npy')
 profilepath=ROOT/'benchmark_results/bone_marrow/maxfuse_v2/pairwise_config.json'
 profile=json.loads(profilepath.read_text());dims=x.shape[1]-1
 for block in ['graph','initial','refine','propagation']:
  for k in profile[block]:
   if ('svd_components' in k or k=='cca_components') and profile[block][k] is not None:profile[block][k]=min(profile[block][k],dims)
 profilepathnew=HERE/'input/maxfuse_pairwise_config.json';dump(profilepathnew,profile)
 dump(HERE/'input/maxfuse_design.json',{'profile_source':str(profilepath),'profile_source_sha256':sha(profilepath),'adaptation':'SVD/CCA dimensions capped explicitly at shared marker count minus one; active=shared; all other locked parameters retained. Propagation not required for full-cell CCA embeddings.','dimensions_cap':dims,'within_dataset_marker_zscore_ddof':1,'original_row_order':True})
 names=['view'+str(i) for i in range(len(datasets))]
 payload={}
 for i,ds in enumerate(datasets):
  rows=md.dataset.eq(ds).to_numpy()
  if len(datasets)==2:
   for v in ['shared','active']:
    payload[f'{v}{i+1}']=x[rows];payload[f'{v}{i+1}_feature_names']=markers
   payload[f'cell_ids{i+1}']=ids[rows]
  else:
   for v in ['shared','active']:
    payload[f'view{i}_{v}']=x[rows];payload[f'view{i}_{v}_feature_names']=markers
   payload[f'view{i}_cell_ids']=ids[rows]
 archive=HERE/'input/maxfuse_input.npz'
 if not archive.exists():np.savez_compressed(archive,**payload)
 hubs=(range(3) if len(datasets)==3 else [None])
 if len(sys.argv)>1:
  assert len(datasets)==3
  hubs=[int(sys.argv[1])];assert hubs[0] in range(3)
 for hub in hubs:
  name='maxfuse' if hub is None else f'maxfuse_hub{hub}'
  output=HERE/'methods'/name;complete=output/'COMPLETE.json'
  if complete.exists():
   info=json.loads(complete.read_text());assert sha(output/'values.npy')==info['values_sha256'];print(name,'cached',flush=True);continue
  base=[str(ALG/'.venv/bin/python')]
  if hub is None:
   command=base+[str(ALG/'run_maxfuse_pairwise.py'),'--input',str(archive),'--config',str(profilepathnew),'--output',str(output)]
  else:
   ordered=[hub]+[i for i in range(3) if i!=hub]
   ordered_names=[names[i] for i in ordered]
   rotated={}
   for j,i in enumerate(ordered):
    for suffix in ['shared','active','shared_feature_names','active_feature_names','cell_ids']:rotated[f'view{j}_{suffix}']=payload[f'view{i}_{suffix}']
   archive=HERE/'input'/f'maxfuse_hub{hub}_input.npz'
   if not archive.exists():np.savez_compressed(archive,**rotated)
   pcmap=HERE/'input'/f'maxfuse_hub{hub}_configs.json';dump(pcmap,{n:profile for n in ordered_names[1:]})
   command=base+[str(ALG/'run_maxfuse_three_view_extension.py'),'--input',str(archive),'--views',*ordered_names,'--hub',names[hub],'--pair-config-map',str(pcmap),'--svd-components','none','none','none','--gcca-components',str(dims),'--outer-reuse-policy','allow','--output',str(output)]
  print('START',name,flush=True);t=time.time()
  if not (output/'manifest.json').exists():
   with (HERE/'logs'/(name+'.log')).open('w') as f:r=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,env=ENV,cwd=HERE)
   if r.returncode:print('FAILED',name,'see log',flush=True);continue
  ys=[];ii=[]
  for i in range(len(datasets)):
   stem=f'cca_embedding{i+1}' if hub is None else f'view{i}_gcca_embedding'
   idstem=stem+'_cell_ids' if hub is None else f'view{i}_gcca_cell_ids'
   ys.append(np.load(output/(stem+'.npy')));ii.append(np.load(output/(idstem+'.npy')))
  merged=np.concatenate(ys);outputids=np.concatenate(ii)
  assert len(outputids)==len(ids) and len(np.unique(outputids))==len(ids)
  order=pd.Index(outputids).get_indexer(ids);assert (order>=0).all()
  values=merged[order];assert np.isfinite(values).all();np.save(output/'values.npy',values)
  dump(complete,{'method':name,'values_sha256':sha(output/'values.npy'),'cells':len(values),'dimensions':values.shape[1],'input_manifest_sha256':sha(HERE/'input/manifest.json'),'input_npz_sha256':sha(archive),'cell_ids_sha256':sha(HERE/'input/cell_ids.npy'),'command':command,'elapsed_seconds':time.time()-t,'output_type':'embedding','active_equals_shared':True})
  print('COMPLETE',name,values.shape,flush=True)
if __name__=='__main__':main()
