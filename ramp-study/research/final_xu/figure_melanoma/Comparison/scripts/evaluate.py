"""Shared frozen 31-metric protocol, full-cell UMAP and 20 matched resamples."""
import os,sys,json,hashlib,importlib.util,time
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parent.parent
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
os.environ['RAMP_EVALUATION_CACHE']=str(HERE/'.cache/evaluation')
os.environ['MPLBACKEND']='Agg'
import numpy as np
import pandas as pd
SRC=ROOT/'benchmark_results/comparison_evaluation/scripts/evaluate_benchmark.py'
spec=importlib.util.spec_from_file_location('frozen_benchmark_evaluation',SRC)
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
OUT=HERE/'evaluation'
LABELS={'raw':'Unaligned','ramp':'RAMP','harmony':'Harmony-direct','fastmnn':'fastMNN','cycombine':'cyCombine','cytonorm':'CytoNorm','maxfuse':'MaxFuse',**{f'maxfuse_hub{i}':f'MaxFuse hub {i+1}' for i in range(3)}}
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def plot_umap(x,md,key):
 import matplotlib.pyplot as plt
 from matplotlib.lines import Line2D
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,'svg.fonttype':'none'})
 fig,axes=plt.subplots(1,2,figsize=(10.2,4.4))
 palette=['#348ABD','#E69F00','#CC6677','#009E73','#AA4499','#44AA99','#88CCEE','#DDCC77','#882255','#999999','#332288','#669900']
 for ax,col,title in zip(axes,['dataset','celltype'],['Dataset','Cell type']):
  categories=sorted(md[col].astype(str).unique());handles=[]
  for j,c in enumerate(categories):
   rows=md[col].astype(str).eq(c).to_numpy();color=palette[j%len(palette)]
   if any(w in c.lower() for w in ['uncertain','ambiguous','non-joint']):color='#c7c7c7'
   ax.scatter(x[rows,0],x[rows,1],s=.5,c=color,alpha=.62,edgecolors='none',rasterized=True)
   handles.append(Line2D([],[],ls='',marker='o',markersize=4,color=color,label=c))
  ax.set_title(title);ax.set_xticks([]);ax.set_yticks([])
  for spine in ax.spines.values():spine.set_visible(False)
  ax.legend(handles=handles,loc='upper left',bbox_to_anchor=(1,1),frameon=False,fontsize=7,handletextpad=.3)
 fig.suptitle(f'{LABELS[key]} · {len(md):,} cells',fontweight='bold',y=.98)
 fig.subplots_adjust(left=.02,right=.78,bottom=.04,top=.89,wspace=1.25)
 dest=OUT/'umap/plots';dest.mkdir(exist_ok=True)
 fig.savefig(dest/(key+'.png'),dpi=200,bbox_inches='tight');fig.savefig(dest/(key+'.svg'),dpi=180,bbox_inches='tight');plt.close(fig)
def run(key):
 start=time.time();name=LABELS[key]
 path=HERE/'input'/(key+'.npy') if key in ['raw','ramp'] else HERE/'methods'/key/'values.npy'
 mdpath=HERE/'input/evaluation_metadata.csv.gz';md=pd.read_csv(mdpath,low_memory=False)
 config=json.loads((HERE/'run_config.json').read_text())
 recordsdir=OUT/'replicates';recordsdir.mkdir(exist_ok=True)
 resultfile=recordsdir/(key+'.csv');done=recordsdir/(key+'_COMPLETE.json')
 fp={'source_values_sha256':sha(path),'metadata_sha256':sha(mdpath),'evaluator_source_sha256':sha(SRC),'adapter_sha256':sha(__file__),'input_manifest_sha256':sha(HERE/'input/manifest.json'),'seeds':list(e.SEEDS),'cells_per_dataset':5000,'umap_protocol':e.UMAP_PROTOCOL}
 fingerprint=hashlib.sha256(json.dumps(fp,sort_keys=True).encode()).hexdigest()
 if done.exists():
  prior=json.loads(done.read_text());assert prior['fingerprint']==fingerprint and prior['replicates_sha256']==sha(resultfile)
  print(key,'EVALUATION cached',flush=True);return
 values=e.standardized(np.load(path),name);assert len(values)==len(md)
 raw=e.standardized(np.load(HERE/'input/raw.npy'),'Unaligned')
 coordinates=e.fit_or_load_umap(values,name,OUT)
 plot_umap(coordinates,md,key)
 manifestpath=recordsdir/(key+'_config.json')
 records=[]
 if resultfile.exists():
  assert json.loads(manifestpath.read_text())['fingerprint']==fingerprint
  records=pd.read_csv(resultfile).to_dict('records')
 dump(manifestpath,{'fingerprint':fingerprint,'configuration':fp})
 completed={int(r['sampling_seed']) for r in records}
 for i,seed in enumerate(e.SEEDS):
  if seed in completed:continue
  rows=e.sample_rows(md,seed,5000);local=md.iloc[rows]
  celltype=local.celltype.astype(str).to_numpy();dataset=local.dataset.astype(str).to_numpy();unit=local.unit_key.astype(str).to_numpy()
  flag=local.joint_evaluation.to_numpy(dtype=bool)
  counts=pd.Series(celltype[flag]).value_counts()
  present=pd.DataFrame({'celltype':celltype[flag],'dataset':dataset[flag]}).groupby('celltype').dataset.nunique()
  levels=present[(present>=2)&(counts>=30)].index.astype(str).to_numpy();valid=flag&np.isin(celltype,levels)
  assert len(levels)>=2
  print(config['case'],key,'resample',i+1,'/20',flush=True)
  result=e.evaluate_one(values[rows],raw[rows],celltype,dataset,unit,valid,seed,coordinates[rows])
  records.append({'benchmark':config['case'],'method':name,'method_key':key,'sampling_seed':seed,'n_cells':len(rows),'n_joint_celltypes':len(levels),'sampled_rows_sha256':e.sha256_array(rows),'fingerprint':fingerprint,**result})
  e.write_atomic_csv(pd.DataFrame(records),resultfile)
 assert len(records)==20
 dump(done,{'fingerprint':fingerprint,'replicates_sha256':sha(resultfile),'seconds':time.time()-start,'n_resamples':20})
 print(key,'EVALUATION COMPLETE',round(time.time()-start),flush=True)
if __name__=='__main__':run(sys.argv[1])
