"""Fit a frozen representation group; no historical predictions/results are read."""
import argparse
import sys
import time
sys.dont_write_bytecode=True
from full_common import *
import worker as pilot_worker

def execute(lockdir,source,run,ident):
    source=Path(source).resolve();run=Path(run).resolve();root=load_lock(lockdir);g=group_lock(lockdir,root,ident);initial=codes()
    require(read(run/'REQUEST.json')==expected_request(ident,lockdir,source,initial),'Group request changed')
    write(run/'STARTED.json',dict(id=ident))
    try:
        before=environment();write(run/'environment_before.json',before)
        for n,r in g['inputs'].items():require(record(member(source,n))==r,'Group input changed: '+n)
        for n,r in root['scientific'].items():copy(member(source,n),run/'runtime/src/trainability'/Path(n).name,r)
        sys.path.insert(0,str(run/'runtime/src'))
        from trainability.runner import run_supervised
        from trainability.heads import Candidate
        from trainability.sampling import Supervision
        from trainability.inputs import PreparedRepresentation
        import numpy as np
        task=pilot_worker.cohort(source,g['cohort'],g['representation']['case'])
        ids=np.load(member(source,g['representation']['ids']),allow_pickle=False)
        require(tuple(ids)==task.pool.cell_ids,'Representation cell order differs')
        x=np.load(member(source,g['representation']['values']),allow_pickle=False)
        marker=g['representation']['marker'];mask=() if not marker else (g['cohort']['canonical_markers'][marker],)
        rep=PreparedRepresentation(x,task.pool.cell_ids,task.pool.sha256,g['representation']['method'],mask)
        summaries={};started=time.monotonic()
        for jobid,row in g['tasks'].items():
            disk_check(run,8*1024**2);require(codes()==initial,'Controller changed before fit')
            ledger=bound_json(source,row['folder']+'/ledger.json',g['inputs'])
            frozen=bound_json(source,row['plan'],g['inputs']) if row['plan'] else None
            pilot_worker.check_ledger(task,dict(row=row),ledger,frozen)
            task.receipt['pilot']=not ledger['formal_benchmark']
            plan=Supervision(*(tuple(ledger[k]) for k in ('fit','validation','calibration')))
            dest=run/'runtime/results'/jobid
            run_supervised(task,rep,plan,sources=ledger['sources'],target=ledger['target'],marker=ledger['marker'],
                candidates=[Candidate(**c) for c in ledger['candidates']],seed=ledger['seed'],out=dest,split_mode=ledger['split_mode'])
            require(codes()==initial,'Controller changed during fit')
            summaries[jobid]=verify_job(dest,row,ledger)
            print('JOB_COMPLETE',jobid,flush=True)
        for n,r in g['inputs'].items():require(record(member(source,n))==r,'Group input changed during fitting: '+n)
        for n,r in root['scientific'].items():require(record(run/'runtime/src/trainability'/Path(n).name)==r,'Scientific code changed')
        require(codes()==initial,'Controller changed during group')
        after=environment();require(after==before,'Runtime changed');write(run/'environment_after.json',after)
        write(run/'WORKER_DONE.json',dict(status='complete',id=ident,jobs=summaries,code_records=initial,inputs=g['inputs'],elapsed_seconds=time.monotonic()-started))
    except BaseException as e:write(run/'FAILED.json',dict(error=repr(e)));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lockdir',type=Path,required=True);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--group',required=True);a=p.parse_args()
    execute(a.lockdir,a.source_root,a.run,a.group)
