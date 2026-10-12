"""Create-only continuation of exactly 15 previously unaccepted groups."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import subprocess
import sys
import time
sys.dont_write_bytecode=True
from resume_common import *

def verify_group(lockdir,source,run,ident,root,unsealed=False):
    run=Path(run);g=group_lock(lockdir,root,ident)
    if not unsealed:
        require(verify_seal(run)['details']==dict(scope='full32370',id=ident,lock_sha256=LOCK_SHA,code_records=codes()),'Group seal identity differs')
    require(not(run/'FAILED.json').exists(),'Failed group')
    req=read(run/'REQUEST.json')
    require(req.get('continuation_of')==INTERRUPTION_SHA and req.get('resource_policy')==RESOURCE_POLICY,'Continuation policy differs')
    require(req['scope']=='full32370' and req['id']==ident and req['lock_sha256']==LOCK_SHA and req['code_records']==codes(),'Group identity/controller differs')
    # Paths may relocate; records and actual input hashes bind the supplied roots.
    before=read(run/'environment_before.json');require(before==read(run/'environment_after.json'),'Environment differs')
    require(before['versions']==VERSIONS and before['threads']=={n:'1' for n in THREADS} and before['pools'] and all(p['num_threads']==1 for p in before['pools']),'Recorded runtime differs')
    command=read(run/'COMMAND.json');argv=command['argv']
    require(argv[1:]==['-B',command['worker'],'--lockdir',req['lockdir'],'--source-root',req['source_root'],'--run',command['run'],'--group',ident],'Worker command differs')
    require(Path(command['worker']).name=='resume_worker.py' and command['worker_record']==codes()['resume_worker.py'] and command['threads']=={n:'1' for n in THREADS},'Worker command binding differs')
    done=read(run/'WORKER_DONE.json')
    require(done['status']=='complete' and done['id']==ident and done['code_records']==codes() and done['inputs']==g['inputs'],'Worker completion differs')
    check_grid(list(done['jobs']),g['tasks']);require({p.name for p in (run/'runtime/results').iterdir()}==set(g['tasks']),'Result grid differs')
    for n,r in root['scientific'].items():require(record(run/'runtime/src/trainability'/Path(n).name)==r,'Runtime scientific code differs')
    summaries={}
    for jobid,row in g['tasks'].items():
        ledger=bound_json(source,row['folder']+'/ledger.json',g['inputs'])
        summaries[jobid]=verify_job(run/'runtime/results'/jobid,row,ledger)
    require(summaries==done['jobs'],'Job report differs')
    return dict(id=ident,jobs=len(summaries),predictions=sum(r['predictions'] for r in summaries.values()),code_records=done['code_records'])


def binding(lockdir,run):
    root=load_lock(lockdir);run=Path(run);todo=remaining(root)
    require(not(run/'FAILED.json').exists() and (run/'COMPLETE.json').is_file(),'Continuation incomplete')
    req=read(run/'BATCH_REQUEST.json');m=read(run/'BATCH_MANIFEST.json')
    require(set(req)=={'scope','lock_sha256','code_records','groups','source_root','lockdir','workers','python','continuation_of','resource_policy','original_run'},'Continuation request schema differs')
    require(set(m)=={'scope','lock_sha256','code_records','groups','jobs','predictions','continuation_of','resource_policy'},'Continuation manifest schema differs')
    for obj in (req,m):
        require(obj['scope']=='continuation810' and obj['lock_sha256']==LOCK_SHA and obj['code_records']==codes() and obj['continuation_of']==INTERRUPTION_SHA and obj['resource_policy']==RESOURCE_POLICY,'Continuation identity/controller differs')
        check_grid(list(obj['groups']),todo)
    require(type(req['workers']) is int and 1<=req['workers']<=2,'Continuation worker count differs')
    require(type(m['jobs']) is int and m['jobs']==810 and type(m['predictions']) is int and m['predictions']>0,'Continuation totals differ')
    require(read(run/'COMPLETE.json')==dict(status='complete',manifest=record(run/'BATCH_MANIFEST.json')),'Continuation seal differs')
    require({p.name for p in run.iterdir()}==set(todo)|{'BATCH_REQUEST.json','BATCH_MANIFEST.json','COMPLETE.json'},'Continuation inventory differs')
    for i in todo:require(record(member(run,i)/'RUN_MANIFEST.json')==m['groups'][i],'Continuation group manifest differs')
    return root,m

def verify_run(lockdir,source,run):
    root,m=binding(lockdir,run)
    reports=[verify_group(lockdir,source,member(run,i),i,root) for i in remaining(root)]
    require(sum(r['jobs'] for r in reports)==m['jobs'] and sum(r['predictions'] for r in reports)==m['predictions'],'Continuation counted totals differ')
    return dict(status='structurally_verified',groups=15,jobs=m['jobs'],predictions=m['predictions'])

def fit(lockdir,source,original_run,output,python,workers):
    lockdir=Path(lockdir).resolve();source=Path(source).resolve();root=load_lock(lockdir);initial=codes()
    require(type(workers) is int and 1<=workers<=2,'Continuation worker limit is 2')
    prior_binding(original_run,root);todo=remaining(root)
    needed=0
    for i in todo:
        g=group_lock(lockdir,root,i)
        for row in g['tasks'].values():
            needed+=g['inputs'][row['folder']+'/ledger.json']['bytes']
            needed+=sum(g['references'][row['folder']+'/'+n]['bytes'] for n in ('predictions.npz','predictions.json','RESULT.json'))
    disk_check(HERE,needed+256*1024**2)
    out=reserve(output,[lockdir,source,original_run]);write(out/'BATCH_REQUEST.json',dict(scope='continuation810',lock_sha256=LOCK_SHA,code_records=initial,
        groups=todo,source_root=str(source),lockdir=str(lockdir),workers=workers,python=str(python),original_run=str(Path(original_run).resolve()),continuation_of=INTERRUPTION_SHA,resource_policy=RESOURCE_POLICY))
    try:
        def one(ident):
            require(codes()==initial,'Controller changed before launch');disk_check(out)
            run=out/ident;run.mkdir(exist_ok=False);write(run/'REQUEST.json',expected_request(ident,lockdir,source,initial))
            cmd=[str(python),'-B',str(HERE/'resume_worker.py'),'--lockdir',str(lockdir),'--source-root',str(source),'--run',str(run),'--group',ident]
            write(run/'COMMAND.json',dict(argv=cmd,worker=str(HERE/'resume_worker.py'),worker_record=initial['resume_worker.py'],run=str(run),threads={n:'1' for n in THREADS}))
            with (run/'stdout.txt').open('x') as stdout,(run/'stderr.txt').open('x') as stderr:
                r=subprocess.run(cmd,env=child_environment(),stdout=stdout,stderr=stderr)
            require(r.returncode==0,'Continuation worker failed: '+ident)
            report=verify_group(lockdir,source,run,ident,root,unsealed=True)
            require(codes()==initial,'Controller changed during continuation')
            seal(run,dict(scope='full32370',id=ident,lock_sha256=LOCK_SHA,code_records=initial));return report
        reports={};started=time.monotonic()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures={pool.submit(one,i):i for i in todo}
            try:
                for f in as_completed(futures):
                    r=f.result();reports[r['id']]=r
                    print(json.dumps(dict(finished_groups=len(reports),jobs=sum(x['jobs'] for x in reports.values()),total=810,last=r['id'],seconds=round(time.monotonic()-started,1))),flush=True)
            except BaseException:
                for f in futures:f.cancel()
                raise
        check_grid(list(reports),todo);require(sum(r['jobs'] for r in reports.values())==810,'Incomplete continuation')
        require(codes()==initial,'Controller changed before completion');prior_binding(original_run,root);disk_check(out)
        write(out/'BATCH_MANIFEST.json',dict(scope='continuation810',lock_sha256=LOCK_SHA,code_records=initial,
            groups={i:record(out/i/'RUN_MANIFEST.json') for i in todo},jobs=810,predictions=sum(r['predictions'] for r in reports.values()),continuation_of=INTERRUPTION_SHA,resource_policy=RESOURCE_POLICY))
        write(out/'COMPLETE.json',dict(status='complete',manifest=record(out/'BATCH_MANIFEST.json')))
        return dict(status='completed',jobs=810,groups=15)
    except BaseException as e:write(out/'FAILED.json',dict(error=repr(e)));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['fit','verify-run']);p.add_argument('--lockdir',type=Path,required=True);p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--original-run',type=Path);p.add_argument('--python',default=sys.executable);p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    print(fit(a.lockdir,a.source_root,a.original_run,a.run,a.python,a.workers) if a.mode=='fit' else verify_run(a.lockdir,a.source_root,a.run))

