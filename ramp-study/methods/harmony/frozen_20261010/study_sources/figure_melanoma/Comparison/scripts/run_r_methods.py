import os,sys,json,subprocess,hashlib,uuid,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parents[1]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def run(method):
 out=HERE/'methods'/method;out.mkdir(exist_ok=True)
 done=out/'COMPLETE.json'
 if done.exists():
  m=json.loads(done.read_text());assert sha(out/'values.npy')==m['values_sha256'];return method,'cached'
 env=os.environ.copy()
 env.update({k:'1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS']})
 tmp=HERE/'tmp'/method;tmp.mkdir(exist_ok=True)
 link=Path('/private/tmp')/('ramp_story_'+uuid.uuid4().hex);link.symlink_to(tmp,target_is_directory=True)
 env['TMPDIR']=str(link);env['PYTHONDONTWRITEBYTECODE']='1'
 try:
  if not (out/'R_COMPLETE.json').exists():
   with (HERE/'logs'/(method+'.log')).open('w') as f:
    p=subprocess.run(['Rscript',str(HERE/'scripts/run_r.R'),str(HERE),method],stdout=f,stderr=subprocess.STDOUT,env=env,cwd=HERE)
   if p.returncode:return method,'FAILED; see logs/'+method+'.log'
  result=pd.read_csv(out/'corrected_expression.csv.gz',low_memory=False)
  ids=np.load(HERE/'input/cell_ids.npy')
  assert np.array_equal(result.cell_id.astype(str),ids)
  values=result.iloc[:,1:].to_numpy(float);assert np.isfinite(values).all()
  np.save(out/'values.npy',values)
  info=json.loads((out/'R_COMPLETE.json').read_text())
  done.write_text(json.dumps({'method':method,'cells':len(values),'dimensions':values.shape[1],'values_sha256':sha(out/'values.npy'),'cell_ids_sha256':sha(HERE/'input/cell_ids.npy'),'input_manifest_sha256':sha(HERE/'input/manifest.json'),'runtime':info},indent=2)+'\n')
  return method,'COMPLETE '+str(values.shape)
 finally:link.unlink(missing_ok=True)
if __name__=='__main__':
 methods=sys.argv[1:] or ['harmony','fastmnn','cycombine','cytonorm']
 with ThreadPoolExecutor(max_workers=2) as pool:
  for f in as_completed([pool.submit(run,m) for m in methods]):print(*f.result(),flush=True)
