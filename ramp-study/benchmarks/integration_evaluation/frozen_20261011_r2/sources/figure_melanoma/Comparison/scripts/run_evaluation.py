import os,sys,json,time,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
HERE=Path(__file__).resolve().parents[1]
config=json.loads((HERE/'run_config.json').read_text())
keys=['ramp','raw','fastmnn','cycombine','cytonorm','harmony']+(['maxfuse_hub0','maxfuse_hub1','maxfuse_hub2'] if len(config['datasets'])==3 else ['maxfuse'])
def run(key):
 env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
 with (HERE/'logs'/('eval_'+key+'.log')).open('w') as f:
  result=subprocess.run([sys.executable,str(HERE/'scripts/evaluate.py'),key],stdout=f,stderr=subprocess.STDOUT,env=env,cwd=HERE)
 return result.returncode
pending=keys.copy();running={};failed=[]
with ThreadPoolExecutor(max_workers=2) as pool:
 while pending or running:
  for key,f in list(running.items()):
   if f.done():
    status=f.result();print(key,'DONE' if status==0 else 'FAILED',flush=True)
    if status:failed.append(key)
    del running[key]
  for key in pending.copy():
   if len(running)>=2:break
   ready=key in ['raw','ramp'] or (HERE/'methods'/key/'COMPLETE.json').exists()
   if ready:running[key]=pool.submit(run,key);pending.remove(key)
  if pending or running:time.sleep(5)
print('Evaluation finished; failed=',failed,flush=True)
