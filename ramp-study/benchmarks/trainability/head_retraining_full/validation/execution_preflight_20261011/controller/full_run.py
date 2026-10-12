"""Exclusive full-grid batch with bounded parallel groups and fixed scientific code."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import subprocess
import sys
import time
sys.dont_write_bytecode=True
from full_common import *

def verify_group(lockdir,source,run,ident,root,unsealed=False):
    run=Path(run);g=group_lock(lockdir,root,ident)
    if not unsealed:
        require(verify_seal(run)['details']==dict(scope='full32370',id=ident,lock_sha256=LOCK_SHA,code_records=codes()),'Group seal identity differs')
    require(not(run/'FAILED.json').exists(),'Failed group')
    req=read(run/'REQUEST.json')
    require(req['scope']=='full32370' and req['id']==ident and req['lock_sha256']==LOCK_SHA and req['code_records']==codes(),'Group identity/controller differs')
    # Paths may relocate; records and actual input hashes bind the supplied roots.
    before=read(run/'environment_before.json');require(before==read(run/'environment_after.json'),'Environment differs')
    require(before['versions']==VERSIONS and before['threads']=={n:'1' for n in THREADS} and before['pools'] and all(p['num_threads']==1 for p in before['pools']),'Recorded runtime differs')
    command=read(run/'COMMAND.json');argv=command['argv']
    require(argv[1:]==['-B',command['worker'],'--lockdir',req['lockdir'],'--source-root',req['source_root'],'--run',command['run'],'--group',ident],'Worker command differs')
    require(Path(command['worker']).name=='full_worker.py' and command['worker_record']==codes()['full_worker.py'] and command['threads']=={n:'1' for n in THREADS},'Worker command binding differs')
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

def verify_run(lockdir,source,run):
    root=load_lock(lockdir);run=Path(run);req=read(run/'BATCH_REQUEST.json');m=read(run/'BATCH_MANIFEST.json')
    require(req['scope']=='full32370' and req['lock_sha256']==LOCK_SHA and req['code_records']==codes(),'Batch identity/controller differs')
    check_grid(req['groups'],root['groups']);check_grid(list(m['groups']),root['groups'])
    require(read(run/'COMPLETE.json')==dict(status='complete',manifest=record(run/'BATCH_MANIFEST.json')) and not(run/'FAILED.json').exists(),'Batch incomplete')
    require(set(p.name for p in run.iterdir())==set(root['groups'])|{'BATCH_REQUEST.json','BATCH_MANIFEST.json','COMPLETE.json'},'Unexpected batch files')
    reports=[]
    for ident in root['groups']:
        require(record(run/ident/'RUN_MANIFEST.json')==m['groups'][ident],'Group manifest changed')
        reports.append(verify_group(lockdir,source,run/ident,ident,root))
    total=sum(r['jobs'] for r in reports)
    require(total==32370 and m['jobs']==total and m['predictions']==sum(r['predictions'] for r in reports),'Batch totals differ')
    return dict(status='structurally_verified',groups=len(reports),jobs=total,predictions=m['predictions'],code_records_identical=all(r['code_records']==req['code_records'] for r in reports))

def fit(lockdir,source,output,python,workers):
    lockdir=Path(lockdir).resolve();source=Path(source).resolve();root=load_lock(lockdir);initial=codes()
    require(1<=workers<=8,'Worker limit is 8')
    disk_check(HERE,int(root['expected_result_bytes'])+2*1024**3)
    # Validate every group lock before any output ownership or launch.
    for ident in root['groups']:group_lock(lockdir,root,ident)
    out=reserve(output,[lockdir,source]);write(out/'BATCH_REQUEST.json',dict(scope='full32370',lock_sha256=LOCK_SHA,code_records=initial,
        groups=list(root['groups']),source_root=str(source),lockdir=str(lockdir),workers=workers,python=str(python)))
    try:
        def one(ident):
            require(codes()==initial,'Controller changed before launch');disk_check(out)
            run=out/ident;run.mkdir(exist_ok=False);write(run/'REQUEST.json',expected_request(ident,lockdir,source,initial))
            cmd=[str(python),'-B',str(HERE/'full_worker.py'),'--lockdir',str(lockdir),'--source-root',str(source),'--run',str(run),'--group',ident]
            write(run/'COMMAND.json',dict(argv=cmd,worker=str(HERE/'full_worker.py'),worker_record=initial['full_worker.py'],run=str(run),threads={n:'1' for n in THREADS}))
            with (run/'stdout.txt').open('x') as stdout,(run/'stderr.txt').open('x') as stderr:
                r=subprocess.run(cmd,env=child_environment(),stdout=stdout,stderr=stderr)
            require(r.returncode==0,'Group worker failed: '+ident)
            report=verify_group(lockdir,source,run,ident,root,unsealed=True)
            require(codes()==initial,'Controller changed during group')
            seal(run,dict(scope='full32370',id=ident,lock_sha256=LOCK_SHA,code_records=initial));return report
        reports={};started=time.monotonic()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures={pool.submit(one,ident):ident for ident in root['groups']}
            try:
                for f in as_completed(futures):
                    r=f.result();reports[r['id']]=r
                    print(json.dumps(dict(finished_groups=len(reports),jobs=sum(x['jobs'] for x in reports.values()),total=32370,last=r['id'],seconds=round(time.monotonic()-started,1))),flush=True)
            except BaseException:
                for f in futures:f.cancel()
                raise
        check_grid(list(reports),root['groups']);require(sum(r['jobs'] for r in reports.values())==32370,'Incomplete task grid')
        require(codes()==initial,'Controller changed before batch completion');load_lock(lockdir)
        write(out/'BATCH_MANIFEST.json',dict(scope='full32370',lock_sha256=LOCK_SHA,code_records=initial,groups={i:record(out/i/'RUN_MANIFEST.json') for i in root['groups']},
            jobs=sum(r['jobs'] for r in reports.values()),predictions=sum(r['predictions'] for r in reports.values())))
        write(out/'COMPLETE.json',dict(status='complete',manifest=record(out/'BATCH_MANIFEST.json')))
        return dict(status='completed',jobs=32370,groups=len(reports))
    except BaseException as e:write(out/'FAILED.json',dict(error=repr(e)));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['fit','verify-run']);p.add_argument('--lockdir',type=Path,required=True);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--python',default=sys.executable);p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    print(fit(a.lockdir,a.source_root,a.run,a.python,a.workers) if a.mode=='fit' else verify_run(a.lockdir,a.source_root,a.run))
