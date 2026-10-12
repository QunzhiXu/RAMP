"""Two declared full groups exercise new orchestration before the complete batch."""
import argparse
import subprocess
import sys
sys.dont_write_bytecode=True
from full_common import *
from full_run import verify_group
from full_compare import compare_group

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lockdir',type=Path,required=True);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    lockdir=a.lockdir.resolve();source=a.source_root.resolve();root=load_lock(lockdir);initial=codes()
    chosen=[min(i for i,v in root['groups'].items() if v['representation'].startswith(prefix)) for prefix in ('tnbc/complete/ramp','bone_marrow/CD117/clipped_zscore')]
    out=reserve(a.output,[lockdir,source]);write(out/'REQUEST.json',dict(groups=chosen,code_records=initial,scope='two_group_smoke_not_full_batch'))
    try:
        reports=[]
        for ident in chosen:
            run=out/ident;run.mkdir();write(run/'REQUEST.json',expected_request(ident,lockdir,source,initial))
            command=[sys.executable,'-B',str(HERE/'full_worker.py'),'--lockdir',str(lockdir),'--source-root',str(source),'--run',str(run),'--group',ident]
            write(run/'COMMAND.json',dict(argv=command,worker=str(HERE/'full_worker.py'),worker_record=initial['full_worker.py'],run=str(run),threads={n:'1' for n in THREADS}))
            with (run/'stdout.txt').open('x') as stdout,(run/'stderr.txt').open('x') as stderr:
                rc=subprocess.run(command,env=child_environment(),stdout=stdout,stderr=stderr).returncode
            require(rc==0,'Smoke worker failed')
            checked=verify_group(lockdir,source,run,ident,root,unsealed=True)
            seal(run,dict(scope='full32370',id=ident,lock_sha256=LOCK_SHA,code_records=initial))
            comparison=compare_group(lockdir,source,run,ident,root)
            require(comparison['exact_jobs']==len(comparison['jobs']),'Smoke numerical difference; stop before full batch')
            reports.append(dict(structural=checked,comparison=comparison));print('SMOKE',ident,checked['jobs'],flush=True)
        require(codes()==initial,'Controller changed during smoke');write(out/'report.json',dict(status='passed',scope='smoke108_only',groups=reports,code_records=initial))
        seal(out,dict(scope='smoke108_only'))
    except BaseException as e:write(out/'FAILED.json',dict(error=repr(e)));raise
