"""One guarded head retraining. Never opens historical predictions or scores."""
import argparse
import sys
sys.dont_write_bytecode = True
from contract import *

LUNG_MAPPING={('Senosain CyTOF','CD4 T'):'CD4 T',('Senosain CyTOF','CD8 T'):'CD8 T',
 ('Cords IMC','CD4'):'CD4 T',('Cords IMC','CD8'):'CD8 T',('Zheng mIF','Th'):'CD4 T',
 ('Zheng mIF','Treg'):'CD4 T',('Zheng mIF','Tc'):'CD8 T',('Zheng mIF','DNT'):'DNT'}

def cohort(bundle, spec, case):
    import numpy as np
    import pandas as pd
    from trainability.inputs import FittingPool
    from trainability.real_data import RealTask
    md=pd.read_csv(member(bundle,spec['metadata']),compression='gzip',dtype=str,keep_default_na=False)
    if case=='lung_story':
        md=md.rename(columns={'cell_key':'cell_id'});md['patient_id']=md.dataset+'::'+md.patient_id
        md['identity_level']='patient'
        md['celltype']=[LUNG_MAPPING.get((d,p),b) for d,p,b in zip(md.dataset,md.provider_cell_type,md.broad_type)]
        require(int((md.dataset.eq('Senosain CyTOF') & md.parent_cluster.eq('C02')).sum())==3171,'Lung cohort changed')
    x=np.load(member(bundle,spec['native']),allow_pickle=False)
    pool=FittingPool(spec['pool_id'],tuple(md.cell_id),tuple(md.dataset),tuple(md.unit_key))
    require(md.cell_id.is_unique and pool.sha256==spec['pool_sha256'],'Cohort identity differs')
    require(x.shape==(len(md),len(spec['markers'])) and np.isfinite(x).all(),'Invalid native matrix')
    return RealTask(case,x,md,tuple(spec['markers']),{},dict(classes=spec['classes'],canonical_markers=spec['canonical_markers']),pool)

def check_ledger(task, entry, ledger, frozen_plan):
    j=entry['row']['job']; marker=j.get('marker') or None
    require(ledger['sources']==[j['source']] and ledger['target']==j['target'] and j['source']!=j['target'],'Direction differs')
    require(ledger['marker']==marker and ledger['seed']==j['seed'] and ledger['split_mode']==j['split_mode'],'Task differs')
    require(ledger['pool_sha256']==task.pool.sha256 and digest(ledger['candidates'])==entry['row']['candidates_sha256'],'Candidate/pool differs')
    roles={k:ledger[k] for k in ('fit','validation','calibration')}
    require(digest(roles)==j['supervision_sha256'],'Supervision hash differs')
    require(len(roles['fit'])==j['fit_budget'] and len(roles['validation'])==j['validation_budget'],'Budget differs')
    ids=sum(roles.values(),[]); require(len(ids)==len(set(ids)) and len(ids)==j['total_supervision']==ledger['total_supervision'],'Role overlap/budget')
    md=task.metadata.set_index('cell_id'); require(set(ids)<=set(md.index),'Unknown supervision ID')
    require(md.loc[ids,'dataset'].eq(j['source']).all(),'Target leakage into supervision')
    if marker is None:
        require(not roles['calibration'] and md.loc[ids,'celltype'].isin(task.receipt['classes']).all(),'J1 class/calibration differs')
    else:
        counts=md.loc[roles['calibration'],'unit_key'].value_counts()
        require(len(counts)>0 and counts.eq(j['calibration_per_unit']).all(),'Calibration differs')
    if frozen_plan is not None:
        for k,v in roles.items(): require(v==frozen_plan.get(k,[]),'Frozen plan differs')
    mask=task.metadata.dataset.eq(j['target'])
    if marker is None: mask &=task.metadata.celltype.isin(task.receipt['classes'])
    require(ledger['target_cell_ids']==task.metadata.loc[mask,'cell_id'].tolist(),'Target identity/order differs')
    require(ledger['mask']==([] if marker is None else [task.receipt['canonical_markers'][marker]]),'Mask differs')
    require(all(c['family']==j['family'] for c in ledger['candidates']),'Wrong head family')

def execute(bundle, run, ident):
    bundle=Path(bundle).resolve();run=Path(run).resolve();plan=load_plan()
    require(ident in plan['tasks'],'Unknown pilot job');entry=plan['tasks'][ident]
    req=read(run/'REQUEST.json')
    require(req==dict(id=ident,bundle=str(bundle),bundle_manifest=record(bundle/'MANIFEST.json'),plan_sha256=PLAN_SHA,code_records=codes()),'Worker request differs')
    write(run/'STARTED.json',dict(id=ident))
    try:
        before=environment();write(run/'environment_before.json',before)
        used={n:v for n,v in plan['training_files'].items() if n.startswith('code/')}
        for n,v in used.items(): copy(member(bundle,n),run/'runtime'/Path(n).relative_to('code'),v['record'])
        spec=plan['cohorts'][entry['row']['job']['task']]
        names=[spec['native'],spec['metadata'],entry['ledger'],*entry['representation'].values()]
        if entry['plan']:names.append(entry['plan'])
        used.update({n:plan['training_files'][n] for n in names})
        for n,v in used.items():require(record(member(bundle,n))==v['record'],'Bundle input changed')
        sys.path.insert(0,str(run/'runtime/src'))
        from trainability.runner import run_supervised
        from trainability.heads import Candidate
        from trainability.sampling import Supervision
        from trainability.inputs import PreparedRepresentation
        import numpy as np
        task=cohort(bundle,spec,entry['row']['job']['task'])
        ledger=checked_json(bundle,entry['ledger'],plan)
        check_ledger(task,entry,ledger,checked_json(bundle,entry['plan'],plan) if entry['plan'] else None)
        task.receipt['pilot']=not ledger['formal_benchmark']
        ids=np.load(member(bundle,entry['representation']['ids']),allow_pickle=False)
        require(tuple(ids)==task.pool.cell_ids,'Representation row order differs')
        x=np.load(member(bundle,entry['representation']['values']),allow_pickle=False)
        rep=PreparedRepresentation(x,task.pool.cell_ids,task.pool.sha256,entry['row']['job']['method'],tuple(ledger['mask']))
        supervision=Supervision(*(tuple(ledger[k]) for k in ('fit','validation','calibration')))
        run_supervised(task,rep,supervision,sources=ledger['sources'],target=ledger['target'],marker=ledger['marker'],
            candidates=[Candidate(**c) for c in ledger['candidates']],seed=ledger['seed'],out=run/'runtime/results',split_mode=ledger['split_mode'])
        require(read(run/'runtime/results/ledger.json')==ledger,'Recreated ledger differs')
        require(codes()==req['code_records'],'Controller changed during fit')
        for n,v in used.items():require(record(member(bundle,n))==v['record'],'Bundle changed during fit')
        for n,v in used.items():
            if n.startswith('code/'):require(record(run/'runtime'/Path(n).relative_to('code'))==v['record'],'Runtime code changed')
        after=environment();require(after==before,'Environment changed');write(run/'environment_after.json',after)
        # Parent seals only after subprocess streams close, so logs are fully hashed.
        write(run/'WORKER_DONE.json',dict(status='complete',id=ident,inputs={n:v['record'] for n,v in used.items()}))
    except BaseException as e:
        write(run/'FAILED.json',dict(error=repr(e)));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--id',required=True);a=p.parse_args()
    execute(a.bundle,a.run,a.id)
