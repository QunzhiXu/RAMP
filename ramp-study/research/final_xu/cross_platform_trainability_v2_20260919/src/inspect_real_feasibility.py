"""Count feasible registered supervision plans without fitting or test outcomes."""
import sys
sys.dont_write_bytecode=True
import itertools,json
import pandas as pd
from trainability.core import ROOT,write_json,ProtocolError
from trainability.real_data import load_task
from trainability.runner import make_plan,make_j3_plans
from trainability.j4 import PretrainConfig,make_plans

folder=ROOT/'data/real_inputs_v0_3_r2'
out=ROOT/'validation/feasibility_v0_3';out.mkdir(exist_ok=False)
tasks=json.loads((folder/'SUMMARY.json').read_text())['tasks'];rows=[];contracts=[]
for info in tasks:
    t=load_task(info['task'],folder);datasets=list(dict.fromkeys(t.pool.datasets))
    for ds,c in t.contracts.items():contracts.append(dict(task=t.name,dataset=ds,**c,evidence_path=t.receipt['manifest_path']))
    for source,target in itertools.permutations(datasets,2):
        for kind in ['J1','J2']:
            for n in [2,5,10,20,50,100,200,500,1000,2000,5000]:
                for seed in [20260919,20260920,20260921]:
                    r=dict(task=t.name,task_type=kind,source=source,target=target,fit_budget=n,validation_budget=512,seed=seed)
                    try:
                        marker=t.markers[0] if kind=='J2' else None
                        if marker and not all(c['j2_eligible'] for c in t.contracts.values()):raise ProtocolError('Unresolved upstream dependence in full U')
                        p,mode=make_plan(t,source,target,fit_budget=n,validation_budget=512,seed=seed,
                            marker=marker,calibration_per_unit=64 if marker else 0)
                        r.update(status='feasible',split_mode=mode,total_supervision=p.total,
                            calibration_cells=len(p.calibration),markers_covered=len(t.markers) if marker else 0)
                    except ProtocolError as e:r.update(status='not_evaluable',reason=str(e))
                    rows.append(r)
    if len(datasets)==3:
        for target in datasets:
            a,b=[d for d in datasets if d!=target]
            for kind in ['J3','J4']:
                r=dict(task=t.name,task_type=kind,source=a+' + '+b,target=target,seed=20260919)
                try:
                    if kind=='J3':
                        p=make_j3_plans(t,a,b,target,fit_quotas={c:25 for c in t.receipt['classes']},
                            validation_quotas={c:10 for c in t.receipt['classes']},seed=20260919)
                        r.update(total_supervision=p['A+B'].total)
                    else:make_plans(t,a,b,target,PretrainConfig(),20260919)
                    r.update(status='feasible')
                except ProtocolError as e:r.update(status='not_evaluable',reason=str(e))
                rows.append(r)
    print(t.name,'feasibility checked',flush=True)
pd.DataFrame(rows).to_csv(out/'supervision_feasibility.csv',index=False)
write_json(ROOT/'config/input_contracts.v0_3.json',contracts)
write_json(out/'SUMMARY.json',dict(total_plans=len(rows),feasible=sum(x['status']=='feasible' for x in rows),
    basis='source identity/counts only; no performance screening',
    proposed_validation_budget=512,proposed_calibration_per_unit=64,
    numeric_configuration_frozen=False,j3_fit_quota_per_class=25,j3_validation_quota_per_class=10,
    tasks=tasks))
