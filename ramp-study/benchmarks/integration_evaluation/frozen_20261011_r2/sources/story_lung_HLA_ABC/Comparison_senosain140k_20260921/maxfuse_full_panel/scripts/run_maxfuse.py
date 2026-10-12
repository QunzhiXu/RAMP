"""Original locked MaxFuse profile with joint shared and full native active panels."""
import os,sys,json,subprocess,hashlib,time,ast,copy
from pathlib import Path
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parents[1];COMP=HERE.parent;ROOT=COMP.parent.parent
ALG=ROOT/'Algorithm/maxfuse'
ENV=os.environ.copy();ENV.update(PYTHONDONTWRITEBYTECODE='1',NUMBA_CACHE_DIR=str(HERE/'.cache/numba'),MPLCONFIGDIR=str(HERE/'.cache/matplotlib'),XDG_CACHE_HOME=str(HERE/'.cache/xdg'),TMPDIR=str(HERE/'tmp'))
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def resolve(profile,a,b,p):
 # Execute exactly the existing pure dimension resolver, with its original constant.
 source=ROOT/'benchmark_results/_maxfuse_benchmark.py';tree=ast.parse(source.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='resolve_pairwise_dimensions']
 ctx={'copy':copy,'Any':object,'PAIRWISE_RANK_ADAPTATION':'clip_non_null_official_targets_to_each_view_feature_rank'}
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ctx)
 return ctx['resolve_pairwise_dimensions'](profile,policy=ctx['PAIRWISE_RANK_ADAPTATION'],shared_width_first=p,shared_width_second=p,active_width_first=a,active_width_second=b)
def main():
 cfg=json.loads((HERE/'run_config.json').read_text());manifest=json.loads((HERE/'data/manifest.json').read_text());views=manifest['views'];sharednames=np.asarray(cfg['markers'],dtype=str)
 assert sha(COMP/'input/manifest.json')==manifest['original_input_manifest_sha256']
 for p,h in manifest['output_sha256'].items():assert sha(HERE/'data'/p)==h
 ids=np.load(COMP/'input/cell_ids.npy');payload={};data=[]
 for v in views:
  p=HERE/'data'/v['slug'];data.append({'shared':np.load(p/'shared.npy'),'active':np.load(p/'active.npy'),'shared_feature_names':sharednames,'active_feature_names':np.array((p/'active_feature_names.txt').read_text().splitlines(),dtype=str),'cell_ids':np.load(p/'cell_ids.npy')})
 profilepath=ROOT/'benchmark_results/bone_marrow/maxfuse_v2/pairwise_config.json';profile=json.loads(profilepath.read_text())
 hubs=range(3) if len(views)==3 else [None]
 if len(sys.argv)>1:hubs=[int(sys.argv[1])];assert len(views)==3 and hubs[0] in range(3)
 for hub in hubs:
  name='maxfuse' if hub is None else f'maxfuse_hub{hub}';out=HERE/'methods'/name;done=out/'COMPLETE.json'
  if done.exists():
   prior=json.loads(done.read_text())
   assert sha(out/'values.npy')==prior['values_sha256']
   assert sha(HERE/'data/manifest.json')==prior['input_manifest_sha256'], 'Cached model input differs'
   assert sha(profilepath)==prior['source_profile_sha256'], 'Cached model profile differs'
   print(name,'CACHED',flush=True);continue
  if hub is None:
   payload={f'{k}{i+1}' if k in ['shared','active','cell_ids'] else f'{k.split("_feature_names")[0]}{i+1}_feature_names':v for i,view in enumerate(data) for k,v in view.items()}
   config,audit=resolve(profile,len(data[0]['active_feature_names']),len(data[1]['active_feature_names']),len(sharednames));configpath=HERE/'data/pairwise_config.json';dump(configpath,config);dump(HERE/'data/pairwise_dimension_audit.json',audit)
   archive=HERE/'data/pairwise_input.npz'
   if not archive.exists():np.savez_compressed(archive,**payload)
   command=[str(ALG/'.venv/bin/python'),str(ALG/'run_maxfuse_pairwise.py'),'--input',str(archive),'--config',str(configpath),'--output',str(out)]
  else:
   # Exactly the cyclic hub/order policy used by the old four benchmarks.
   order=[(hub+j)%3 for j in range(3)];names=[f'view{i}' for i in order]
   payload={f'view{j}_{k}':v for j,i in enumerate(order) for k,v in data[i].items()}
   configs={};audits={}
   for i,n in zip(order[1:],names[1:]):configs[n],audits[n]=resolve(profile,len(data[hub]['active_feature_names']),len(data[i]['active_feature_names']),len(sharednames))
   configpath=HERE/'data'/f'hub{hub}_pairwise_configs.json';dump(configpath,configs);dump(HERE/'data'/f'hub{hub}_dimension_audit.json',audits)
   archive=HERE/'data'/f'hub{hub}_input.npz'
   if not archive.exists():np.savez_compressed(archive,**payload)
   command=[str(ALG/'.venv/bin/python'),str(ALG/'run_maxfuse_three_view_extension.py'),'--input',str(archive),'--views',*names,'--hub',names[0],'--pair-config-map',str(configpath),'--svd-components','none','none','none','--gcca-components','15','--outer-reuse-policy','allow','--output',str(out)]
  start=time.time();record={'method':name,'command':command,'source_profile':str(profilepath),'source_profile_sha256':sha(profilepath),'source_resolver_sha256':sha(ROOT/'benchmark_results/_maxfuse_benchmark.py'),'input_npz_sha256':sha(archive),'input_manifest_sha256':sha(HERE/'data/manifest.json'),'shared_markers':cfg['markers'],'active_widths':[v['active_features'] for v in views],'active_equals_shared':False,'start_unix':start}
  dump(HERE/'logs'/(name+'_execution.json'),record);print('START',cfg['case'],name,flush=True)
  if not (out/'manifest.json').exists():
   with (HERE/'logs'/(name+'.log')).open('w') as f:r=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,env=ENV,cwd=HERE)
   if r.returncode:raise RuntimeError(f'{name} failed with {r.returncode}; see log')
  arrays=[];cellids=[]
  for i in range(len(views)):
   stem=f'cca_embedding{i+1}' if hub is None else f'view{i}_gcca_embedding';idstem=stem+'_cell_ids' if hub is None else f'view{i}_gcca_cell_ids'
   arrays.append(np.load(out/(stem+'.npy')));cellids.append(np.load(out/(idstem+'.npy')))
  allids=np.concatenate(cellids);assert len(allids)==len(ids) and pd.Index(allids).is_unique
  order=pd.Index(allids).get_indexer(ids);assert (order>=0).all();values=np.concatenate(arrays)[order];assert np.isfinite(values).all()
  np.save(out/'values.npy',values);dump(done,{**record,'elapsed_seconds':time.time()-start,'cells':len(values),'dimensions':values.shape[1],'values_sha256':sha(out/'values.npy'),'cell_ids_sha256':sha(COMP/'input/cell_ids.npy'),'output_type':'embedding'})
  print('COMPLETE',cfg['case'],name,values.shape,flush=True)
if __name__=='__main__':
 import fcntl
 locks=HERE/'.locks';locks.mkdir(exist_ok=True)
 key=('hub'+sys.argv[1]) if len(sys.argv)>1 else 'default'
 with (locks/(key+'.lock')).open('a') as lock:
  fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
  main()
