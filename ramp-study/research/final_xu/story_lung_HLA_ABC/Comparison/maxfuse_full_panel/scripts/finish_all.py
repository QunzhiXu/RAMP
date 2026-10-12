"""Bounded local orchestration of five fixed fits and their existing evaluations."""
import os,sys,json,time,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import psutil
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parents[2]
CASES={s:ROOT/s/'Comparison/maxfuse_full_panel' for s in ['figure_HNSCC','figure_melanoma','story_lung_HLA_ABC']}
TASKS=[('figure_HNSCC','maxfuse'),('figure_melanoma','maxfuse')]+[('story_lung_HLA_ABC',f'maxfuse_hub{i}') for i in range(3)]
external=set(TASKS[:3]);pending=TASKS[3:];fitting={};evaluating={};evaluated=set();errors=[]
ENV=os.environ.copy();ENV['PYTHONDONTWRITEBYTECODE']='1'
def job(case,key,stage):
 b=CASES[case];command=[sys.executable,str(b/'scripts'/('run_maxfuse.py' if stage=='fit' else 'evaluate.py'))]
 if stage=='fit':command.append(key[-1])
 else:command.append(key)
 with (b/'logs'/f'{stage}_{key}_managed.log').open('w') as f:r=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,env=ENV,cwd=b)
 return r.returncode
with ThreadPoolExecutor(max_workers=6) as pool:
 while pending or external or fitting or evaluating or len(evaluated)<len(TASKS):
  for task in list(external):
   case,key=task;b=CASES[case]
   if (b/'methods'/key/'COMPLETE.json').exists():external.remove(task);print('FIT COMPLETE',*task,flush=True)
   else:
    lp=b/'logs'/('run_hub0.log' if key.endswith('hub0') else 'run.log')
    if lp.exists() and 'Traceback (most recent call last)' in lp.read_text(errors='replace'):raise RuntimeError(f'External fit failed: {task}')
  for task,future in list(fitting.items()):
   if future.done():
    del fitting[task]
    if future.result()!=0:raise RuntimeError(f'Fit failed: {task}')
    print('FIT COMPLETE',*task,flush=True)
  for task,future in list(evaluating.items()):
   if future.done():
    del evaluating[task]
    if future.result()!=0:raise RuntimeError(f'Evaluation failed: {task}')
    evaluated.add(task);print('EVALUATION COMPLETE',*task,flush=True)
  for task in TASKS:
   if len(evaluating)>=2:break
   if task not in evaluated and task not in evaluating and (CASES[task[0]]/'methods'/task[1]/'COMPLETE.json').exists():
    if psutil.virtual_memory().available<3*1024**3:break
    evaluating[task]=pool.submit(job,*task,'eval');print('EVALUATION START',*task,flush=True)
  while pending and len(external)+len(fitting)<4 and psutil.virtual_memory().available>5*1024**3:
   task=pending.pop(0);fitting[task]=pool.submit(job,*task,'fit');print('FIT START',*task,flush=True)
  status={'external_fits':[list(t) for t in external],'managed_fits':[list(t) for t in fitting],'pending':[list(t) for t in pending],'evaluating':[list(t) for t in evaluating],'evaluated':[list(t) for t in evaluated],'updated_unix':time.time()}
  (HERE/'ORCHESTRATION.json').write_text(json.dumps(status,indent=2)+'\n')
  if len(evaluated)==len(TASKS):break
  time.sleep(5)
for case,b in CASES.items():
 with (b/'logs/finalize.log').open('w') as f:r=subprocess.run([sys.executable,str(b/'scripts/summarize.py')],stdout=f,stderr=subprocess.STDOUT,env=ENV,cwd=b)
 if r.returncode:raise RuntimeError('Summarize failed: '+case)
 print('FINALIZED',case,flush=True)
print('ALL THREE FULL-PANEL COMPARISONS COMPLETE',flush=True)
