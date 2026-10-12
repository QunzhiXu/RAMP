"""Local, resumable fitting/evaluation. No network, no scheduler, no external edits."""
from pathlib import Path
import os,sys,subprocess,time,json,threading,concurrent.futures,fcntl
C=Path(__file__).resolve().parents[1];MF=C/'maxfuse_full_panel';PY=sys.executable
ENV=os.environ.copy();ENV.update({k:'1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']});ENV['PYTHONDONTWRITEBYTECODE']='1';ENV['TMPDIR']=str(C/'tmp')
state={'status':'running','started_unix':time.time(),'pid':os.getpid(),'jobs':{}};lock=threading.Lock()
def update(name,**kw):
 with lock:
  state['jobs'].setdefault(name,{}).update(kw);state['updated_unix']=time.time();p=C/'STATUS.tmp.json';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(C/'STATUS.json')
def run(name,args,done):
 if done.exists():update(name,status='complete_cached');return
 update(name,status='running',started_unix=time.time(),command=args)
 with (C/'logs'/f'{name}.log').open('a') as f:
  proc=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,cwd=C,env=ENV);update(name,child_pid=proc.pid);code=proc.wait()
 if code or not done.exists():update(name,status='failed',returncode=code,ended_unix=time.time());raise RuntimeError(name+' failed')
 update(name,status='complete',ended_unix=time.time())
def rmodels():
 for key in ['harmony','fastmnn','cycombine','cytonorm']:run('fit_'+key,[PY,'-u',str(C/'scripts/run_r_methods.py'),key],C/'methods'/key/'COMPLETE.json')
def maxfuse(hub):run('fit_maxfuse_hub'+str(hub),[PY,'-u',str(MF/'scripts/run_maxfuse.py'),str(hub)],MF/'methods'/f'maxfuse_hub{hub}'/'COMPLETE.json')
def evaluations():
 pending=['ramp','raw','harmony','fastmnn','cycombine','cytonorm','maxfuse_hub0','maxfuse_hub1','maxfuse_hub2']
 while pending:
  key=next((k for k in pending if k in ['raw','ramp'] or ((MF if k.startswith('maxfuse') else C)/'methods'/k/'COMPLETE.json').exists()),None)
  if key is None:
   with lock:
    if any(v.get('status')=='failed' for n,v in state['jobs'].items() if n.startswith('fit_')):raise RuntimeError('Fit failed; remaining evaluations held')
   time.sleep(5);continue
  base=MF if key.startswith('maxfuse') else C
  run('eval_'+key,[PY,'-u',str(base/'scripts/evaluate.py'),key],base/'evaluation/replicates'/f'{key}_COMPLETE.json');pending.remove(key)
with (C/'RUN.lock').open('a') as handle:
 fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert (MF/'data/manifest.json').exists();update('orchestrator',status='running')
 errors=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
  fs=[pool.submit(rmodels),pool.submit(evaluations),*[pool.submit(maxfuse,i) for i in range(3)]]
  for f in concurrent.futures.as_completed(fs):
   try:f.result()
   except Exception as ex:errors.append(str(ex));update('errors',messages=errors,status='failed')
 state['status']='failed' if errors else 'models_and_evaluations_complete';update('orchestrator',status=state['status'],ended_unix=time.time())
 if errors:raise RuntimeError(errors)
 print('All model fits and all 9 evaluations finished.',flush=True)
