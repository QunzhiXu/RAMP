"""Small predeclared head-retraining pilot, never a full Figure 3 launch."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys
sys.dont_write_bytecode=True
from contract import *

def prepare(source,bundle):
    source=Path(source).resolve();plan=load_plan()
    # All reads below are limited to the pinned source catalog's training subset.
    for v in plan['training_files'].values():require(record(member(source,v['source']))==v['record'],'Source changed before preparation')
    out=reserve(bundle,[source,HERE/'locks'])
    try:
        for n,v in plan['training_files'].items():copy(member(source,v['source']),out/n,v['record'])
        for v in plan['training_files'].values():require(record(member(source,v['source']))==v['record'],'Source changed during preparation')
        write(out/'MANIFEST.json',dict(plan_sha256=PLAN_SHA,files=plan['training_files']))
        require(files(out)=={n:v['record'] for n,v in plan['training_files'].items()}|{'MANIFEST.json':record(out/'MANIFEST.json')},'Bundle changed')
        write(out/'COMPLETE.json',dict(status='complete',manifest=record(out/'MANIFEST.json')))
        return dict(files=len(plan['training_files']),tasks=len(plan['tasks']))
    except BaseException as e:write(out/'FAILED.json',dict(error=repr(e)));raise

def verify_job(bundle,run,ident,plan,unsealed=False):
    import numpy as np
    run=Path(run);m=verify_seal(run) if not unsealed else None;entry=plan['tasks'][ident];req=read(run/'REQUEST.json')
    require(req['id']==ident and req['plan_sha256']==PLAN_SHA and req['bundle_manifest']==record(Path(bundle)/'MANIFEST.json'),'Run identity/bundle differs')
    require(req['code_records']==codes(),'Unknown controller')
    if m is not None:require(m['details']==dict(id=ident,code_records=req['code_records']),'Run detail differs')
    require(not(run/'FAILED.json').exists(),'Failed job')
    command=read(run/'COMMAND.json')
    require(command['argv'][1:]==['-B',command['worker'],'--bundle',req['bundle'],'--run',command['run'],'--id',ident],'Command differs')
    require(Path(command['worker']).name=='worker.py' and command['worker_sha256']==codes()['worker.py']['sha256'],'Worker binding differs')
    require(command['threads']=={n:'1' for n in THREADS},'Command threads differ')
    before=read(run/'environment_before.json');require(before==read(run/'environment_after.json'),'Environment changed')
    require(before['versions']==VERSIONS and before['threads']=={n:'1' for n in THREADS} and before['pools'] and all(p['num_threads']==1 for p in before['pools']),'Recorded runtime differs')
    expected={n:v['record'] for n,v in plan['training_files'].items() if n.startswith('code/')}
    spec=plan['cohorts'][entry['row']['job']['task']]
    for n in [spec['native'],spec['metadata'],entry['ledger'],*entry['representation'].values(),*([entry['plan']] if entry['plan'] else [])]:expected[n]=plan['training_files'][n]['record']
    require(read(run/'WORKER_DONE.json')==dict(status='complete',id=ident,inputs=expected),'Worker completion differs')
    for n,r in expected.items():
        if n.startswith('code/'):require(record(run/'runtime'/Path(n).relative_to('code'))==r,'Runtime copy differs')
    result=run/'runtime/results';ledger=checked_json(bundle,entry['ledger'],plan)
    require(read(result/'ledger.json')==ledger,'Run ledger differs')
    require(set(p.name for p in result.iterdir())=={'ledger.json','predictions.npz','predictions.json','RESULT.json'},'Scientific payload differs')
    receipt=read(result/'predictions.json');score=read(result/'RESULT.json')
    require(receipt['sha256']==record(result/'predictions.npz')['sha256'] and receipt['target_answers_included'] is False,'Prediction receipt differs')
    require(receipt['provenance']==dict(ledger_sha256=digest(ledger),pool_sha256=ledger['pool_sha256'],selection=score['selection']),'Prediction binding differs')
    require(score['prediction_receipt']==receipt and receipt['n_cells']==len(ledger['target_cell_ids']),'Result receipt differs')
    with np.load(result/'predictions.npz',allow_pickle=False) as data:
        require(set(data.files)=={'cell_ids','predictions'} and data['cell_ids'].tolist()==ledger['target_cell_ids'],'Prediction identities differ')
        pred=data['predictions'];require(pred.ndim==1 and len(pred)==receipt['n_cells'],'Prediction shape differs')
        if ledger['marker']:require(pred.dtype.kind=='f' and np.isfinite(pred).all(),'Invalid regression prediction')
        else:require(pred.dtype.kind in 'US' and set(pred)<=set(spec['classes']),'Invalid classification prediction')
    for r,k in [('fit','fit_cells'),('validation','validation_cells'),('calibration','calibration_cells')]:require(score[k]==len(ledger[r]),'Result budget differs')
    require(score['total_supervision']==ledger['total_supervision'],'Total budget differs')
    return dict(id=ident,predictions=receipt['n_cells'],code_records=req['code_records'])

def fit(bundle,output,python,workers):
    bundle=Path(bundle).resolve();plan=verify_bundle(bundle);require(1<=workers<=3,'Pilot worker limit')
    out=reserve(output,[bundle,HERE/'locks']);initial=codes()
    request=dict(scope='pilot45',plan_sha256=PLAN_SHA,bundle_manifest=record(bundle/'MANIFEST.json'),code_records=initial,tasks=list(plan['tasks']))
    write(out/'BATCH_REQUEST.json',request)
    try:
        def one(ident):
            require(codes()==initial,'Controller changed before job')
            run=out/ident;run.mkdir(exist_ok=False)
            write(run/'REQUEST.json',dict(id=ident,bundle=str(bundle),bundle_manifest=record(bundle/'MANIFEST.json'),plan_sha256=PLAN_SHA,code_records=initial))
            command=[str(python),'-B',str(HERE/'worker.py'),'--bundle',str(bundle),'--run',str(run),'--id',ident]
            write(run/'COMMAND.json',dict(argv=command,worker=str(HERE/'worker.py'),worker_sha256=initial['worker.py']['sha256'],run=str(run),threads={n:'1' for n in THREADS}))
            with (run/'stdout.txt').open('x') as stdout,(run/'stderr.txt').open('x') as stderr:
                rc=subprocess.run(command,env=child_environment(),stdout=stdout,stderr=stderr).returncode
            require(rc==0,'Worker failed: '+ident)
            require(codes()==initial,'Controller changed during batch')
            report=verify_job(bundle,run,ident,plan,unsealed=True)
            seal(run,dict(id=ident,code_records=initial))
            print('FINISHED',ident,flush=True);return report
        with ThreadPoolExecutor(max_workers=workers) as pool:reports=list(pool.map(one,plan['tasks']))
        verify_bundle(bundle);require(codes()==initial,'Controller changed')
        check_grid([r['id'] for r in reports],plan['tasks'])
        write(out/'report.json',dict(scope='pilot45',jobs=reports,identical_request_code_records=all(r['code_records']==initial for r in reports)))
        seal(out,dict(scope='pilot45',plan_sha256=PLAN_SHA,code_records=initial))
        return dict(status='completed',scope='pilot45',jobs=len(reports),predictions=sum(r['predictions'] for r in reports))
    except BaseException as e:
        if not(out/'FAILED.json').exists():write(out/'FAILED.json',dict(error=repr(e)))
        raise

def verify_run(bundle,run):
    plan=verify_bundle(bundle);run=Path(run);m=verify_seal(run)
    req=read(run/'BATCH_REQUEST.json');report=read(run/'report.json')
    require(req==dict(scope='pilot45',plan_sha256=PLAN_SHA,bundle_manifest=record(Path(bundle)/'MANIFEST.json'),code_records=codes(),tasks=list(plan['tasks'])),'Batch request differs')
    require(m['details']==dict(scope='pilot45',plan_sha256=PLAN_SHA,code_records=codes()),'Batch detail differs')
    check_grid([r['id'] for r in report['jobs']],plan['tasks'])
    require({p.name for p in run.iterdir() if p.is_dir()}==set(plan['tasks']),'Unexpected job directories')
    reports=[verify_job(bundle,run/i,i,plan) for i in plan['tasks']]
    require(report==dict(scope='pilot45',jobs=reports,identical_request_code_records=all(r['code_records']==codes() for r in reports)),'Batch report differs')
    return dict(status='structurally_verified',scope='pilot45',jobs=len(reports),predictions=sum(r['predictions'] for r in reports),not_a_historical_numerical_comparison=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','verify','fit','verify-run']);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--source-root',type=Path);p.add_argument('--output',type=Path);p.add_argument('--run',type=Path);p.add_argument('--python',default=sys.executable);p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    if a.mode=='prepare':result=prepare(a.source_root,a.bundle)
    elif a.mode=='verify':plan=verify_bundle(a.bundle);result=dict(tasks=len(plan['tasks']),files=len(plan['training_files']))
    elif a.mode=='fit':result=fit(a.bundle,a.output,a.python,a.workers)
    else:result=verify_run(a.bundle,a.run)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
