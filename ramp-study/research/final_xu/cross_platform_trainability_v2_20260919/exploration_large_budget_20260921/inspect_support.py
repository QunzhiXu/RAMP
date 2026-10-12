from pathlib import Path
import sys,json
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import pandas as pd
from trainability.real_data import load_task
from trainability.sampling import source_split
from trainability.runner import eligible_ids
out=Path(__file__).parent;rows=[]
for name in [x['task'] for x in json.loads((ROOT/'data/real_inputs_v0_3_r2/SUMMARY.json').read_text())['tasks']]:
 task=load_task(name);md=task.metadata.set_index('cell_id')
 cfg=json.loads((ROOT/f'config/formal_v0_5/{name}__ramp.json').read_text())
 jobs=[j for j in cfg['jobs'] if j['task_type']=='J1' and j['family']=='mlp' and j['fit_budget']==2000 and j['seed']==20260919 and j['width']==32 and not j.get('labelled_patient_count')]
 for j in jobs:
  ids=eligible_ids(task,j['source'],True)
  pats=md.patient_id.to_dict() if md.loc[list(ids),'identity_level'].eq('patient').all() else md.unit_key.to_dict()
  split=source_split(ids,pats,.2,j['seed'],patient_mapping_verified=True)
  row=dict(task=name,source=j['source'],target=j['target'],seed=j['seed'],eligible_source=len(ids),fit_available=len(split['fit_pool']),validation_available=len(split['validation_pool']),baseline_job_id=j['id'])
  for b in [10000,20000,50000,100000]:row[f'supports_{b}']=len(split['fit_pool'])>=b and len(split['validation_pool'])>=512
  rows.append(row)
 print(name,'directions',len(jobs),flush=True)
df=pd.DataFrame(rows);df.to_csv(out/'FEASIBILITY.csv',index=False)
print(df[['task','source','target','fit_available']].to_string(index=False))
print('Support counts', {b:int(df[f'supports_{b}'].sum()) for b in [10000,20000,50000,100000]})
