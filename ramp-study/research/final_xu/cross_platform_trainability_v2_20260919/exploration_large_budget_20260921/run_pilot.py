"""RAMP-only exploratory extension; frozen source inputs/results remain read-only."""
from pathlib import Path
import os,sys
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMBA_NUM_THREADS','LOKY_MAX_CPU_COUNT']:os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
import json,time,traceback,argparse,copy
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor,as_completed
import pandas as pd
from trainability.core import file_sha256,digest
from trainability.real_data import load_task
from trainability.runner import make_plan,run_supervised
from trainability.adapters import fit_representation
from trainability.heads import Candidate

def save(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def prepare():
 feasible=pd.read_csv(OUT/'FEASIBILITY.csv');jobs=[];baselines=[];exclusions=[];input_hashes={}
 for name,rows in feasible.groupby('task',sort=False):
  cfgpath=ROOT/f'config/formal_v0_5/{name}__ramp.json';cfg=json.loads(cfgpath.read_text());task=load_task(name)
  byid={j['id']:j for j in cfg['jobs']};input_hashes[str(cfgpath)]=file_sha256(cfgpath)
  for p in [ROOT/f'data/real_inputs_v0_3_r2/{name}/manifest.json',ROOT/f'representations/formal_v0_5/{name}/complete/ramp/COMPLETE.json']:input_hashes[str(p)]=file_sha256(p)
  for row in rows.to_dict('records'):
   base=byid[row['baseline_job_id']]
   # Reuse comparable one-seed 500/2k anchors; no changed sampling or re-training.
   for b,w in [(500,8),(500,32),(500,128),(2000,32)]:
    matches=[j for j in cfg['jobs'] if j['task_type']=='J1' and j['family']=='mlp' and j['source']==row['source'] and j['target']==row['target'] and j['seed']==20260919 and j['fit_budget']==b and j['width']==w and not j.get('labelled_patient_count')]
    assert len(matches)==1,(name,row['source'],row['target'],b,w)
    old=matches[0];res=Path(cfg['output'])/'jobs'/old['id']/'RESULT.json';ledger=res.with_name('ledger.json')
    assert res.is_file() and ledger.is_file()
    oldplan,mode=make_plan(task,row['source'],row['target'],fit_budget=b,validation_budget=512,seed=20260919)
    assert digest(dict(fit=oldplan.fit,validation=oldplan.validation,calibration=oldplan.calibration))==old['supervision_sha256']
    baselines.append(dict(task=name,source=row['source'],target=row['target'],seed=20260919,fit_budget=b,width=w,result=str(res),result_sha256=file_sha256(res)))
   grid=[(5000,w) for w in [8,32,128,512]]+[(b,32) for b in [10000,20000,50000,100000]]
   for budget,width in grid:
    if budget>row['fit_available'] or row['validation_available']<512:
     exclusions.append(dict(task=name,source=row['source'],target=row['target'],fit_budget=budget,width=width,fit_available=row['fit_available'],reason='insufficient_frozen_source_training_cells'));continue
    plan,mode=make_plan(task,row['source'],row['target'],fit_budget=budget,validation_budget=512,seed=20260919)
    anchor=json.loads((Path(cfg['output'])/'jobs'/base['id']/'ledger.json').read_text())
    assert list(plan.fit[:2000])==anchor['fit'] and list(plan.validation)==anchor['validation']
    j=dict(task=name,source=row['source'],target=row['target'],seed=20260919,fit_budget=budget,width=width,validation_budget=512,method='ramp',split_mode=mode,candidates=[dict(c,width=width) for c in base['candidates']],supervision_sha256=digest(dict(fit=plan.fit,validation=plan.validation,calibration=plan.calibration)))
    j['id']='explore_'+digest(j)[:20];jobs.append(j)
 jobs.sort(key=lambda j:(j['task'],j['width']!=32,j['fit_budget'],j['width'],j['source'],j['target']))
 manifest=dict(exploratory=True,seed_count=1,seed=20260919,method='ramp',fit_budgets_requested=[10000,20000,50000,100000],additional_common_budget=5000,capacity_fit_budget=5000,widths=[8,32,128,512],validation_budget=512,source_only_selection=True,full_frozen_unlabeled_pool=True,parameters_unchanged_except_fit_budget_and_width=True,code_hashes={str(p):file_sha256(p) for p in (ROOT/'src').rglob('*') if p.suffix in ['.py','.R']},input_hashes=input_hashes,jobs=jobs,baselines=baselines,exclusions=exclusions)
 save(OUT/'MANIFEST.json',manifest)
 print('Prepared',len(jobs),'new jobs,',len(baselines),'reused baseline records,',len(exclusions),'infeasible grid points',flush=True)
 return manifest

def worker(name,jobs,manifest_hash):
 started=time.monotonic();task=load_task(name)
 cache=ROOT/f'representations/formal_v0_5/{name}/complete/ramp'
 assert (cache/'COMPLETE.json').is_file(),'Do not refit or write to the frozen cache'
 rep=fit_representation(task.input('ramp'),'ramp',fitting_pool=task.pool,out=cache,input_contract=task.receipt,family=f'trainability_v0.3:{name}')
 # Same frozen cells/metadata/representation, explicitly mark outputs exploratory.
 task=replace(task,receipt=dict(task.receipt,pilot=True,exploration='large_budget_20260921'))
 log=OUT/f'logs/{name}.log';log.parent.mkdir(exist_ok=True)
 records=[]
 for j in jobs:
  dest=OUT/'jobs'/j['id'];done=dest/'PILOT_COMPLETE.json';t=time.monotonic()
  try:
   if done.exists():
    old=json.loads(done.read_text());assert old['job_sha256']==digest(j) and old['manifest_sha256']==manifest_hash
    assert old['result_sha256']==file_sha256(dest/'RESULT.json');status='cached'
   else:
    assert not dest.exists(),'Partial output exists; inspect before retry'
    plan,mode=make_plan(task,j['source'],j['target'],fit_budget=j['fit_budget'],validation_budget=512,seed=j['seed'])
    assert digest(dict(fit=plan.fit,validation=plan.validation,calibration=plan.calibration))==j['supervision_sha256']
    run_supervised(task,rep,plan,sources=[j['source']],target=j['target'],marker=None,candidates=[Candidate(**c) for c in j['candidates']],seed=j['seed'],out=dest,split_mode=mode)
    save(done,dict(job_sha256=digest(j),manifest_sha256=manifest_hash,result_sha256=file_sha256(dest/'RESULT.json'),exploratory=True));status='passed'
   result=json.loads((dest/'RESULT.json').read_text())
   record=dict(id=j['id'],task=name,status=status,budget=j['fit_budget'],width=j['width'],source=j['source'],target=j['target'],macro_f1=result['score']['macro_f1'],seconds=time.monotonic()-t)
  except Exception as e:
   record=dict(id=j['id'],task=name,status='failed',budget=j['fit_budget'],width=j['width'],error=repr(e))
   save(OUT/'failures'/f"{j['id']}.json",dict(job=j,traceback=traceback.format_exc()))
  records.append(record)
  with log.open('a') as f:f.write(json.dumps(record)+'\n')
  save(OUT/f'status/{name}.json',dict(task=name,completed=len(records),total=len(jobs),elapsed_seconds=time.monotonic()-started,records=records))
 return dict(task=name,completed=len(records),failed=sum(r['status']=='failed' for r in records),seconds=time.monotonic()-started)

def main():
 p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');p.add_argument('--workers',type=int,default=2);args=p.parse_args()
 if not (OUT/'MANIFEST.json').exists():m=prepare()
 else:m=json.loads((OUT/'MANIFEST.json').read_text())
 if args.prepare_only:return
 for path,h in {**m['code_hashes'],**m['input_hashes']}.items():assert file_sha256(path)==h,path
 for b in m['baselines']:assert file_sha256(b['result'])==b['result_sha256']
 save(OUT/'RUN.json',dict(pid=os.getpid(),started_unix=time.time(),script_sha256=file_sha256(__file__),manifest_sha256=file_sha256(OUT/'MANIFEST.json'),workers=args.workers,exploratory=True))
 done=[]
 with ProcessPoolExecutor(max_workers=args.workers) as ex:
  futures={ex.submit(worker,name,[j for j in m['jobs'] if j['task']==name],file_sha256(OUT/'MANIFEST.json')):name for name in sorted({j['task'] for j in m['jobs']})}
  for f in as_completed(futures):
   try:r=f.result()
   except Exception as e:r=dict(task=futures[f],failed=True,error=repr(e),traceback=traceback.format_exc())
   done.append(r);print(json.dumps(r),flush=True);save(OUT/'TASK_STATUS.json',dict(tasks=done))
 save(OUT/'RUN_COMPLETE.json',dict(tasks=done,all_passed=all(not x['failed'] for x in done),finished_unix=time.time()))

if __name__=='__main__':main()
