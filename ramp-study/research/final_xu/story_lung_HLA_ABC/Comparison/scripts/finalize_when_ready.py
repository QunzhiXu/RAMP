import os,sys,json,time,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
cfg=json.loads((HERE/'run_config.json').read_text())
keys=['raw','ramp','harmony','cycombine','cytonorm','fastmnn']+(['maxfuse_hub0','maxfuse_hub1','maxfuse_hub2'] if len(cfg['datasets'])==3 else ['maxfuse'])
while not all((HERE/'evaluation/replicates'/(key+'_COMPLETE.json')).exists() for key in keys):time.sleep(5)
env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['OPENBLAS_NUM_THREADS']='1'
for script in ['summarize.py','render_overview.py']:
 subprocess.run([sys.executable,str(HERE/'scripts'/script)],env=env,cwd=HERE,check=True)
print('ALL RESULTS FINALIZED',cfg['case'],flush=True)
