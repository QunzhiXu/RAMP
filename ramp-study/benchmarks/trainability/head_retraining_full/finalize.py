"""One finalizer for the existing batch; never creates or restarts scientific fits."""
import subprocess
import sys
import time
sys.dont_write_bytecode=True
from full_common import *

if __name__=='__main__':
    lockdir=HERE/'locks/full_20261011';source=STUDY.parent/'final_xu';run=HERE/'runs/full_20261011'
    comparison=HERE/'validation/comparison_20261011';aggregation=HERE/'validation/aggregation_20261011';handoff=HERE/'validation/handoff_20261011'
    out=reserve(HERE/'validation/finalization_20261011',[run,lockdir,source])
    commands=[
        [sys.executable,'-B',str(HERE/'full_compare.py'),'--lockdir',str(lockdir),'--source-root',str(source),'--run',str(run),'--output',str(comparison)],
        [sys.executable,'-B',str(HERE/'aggregate_results.py'),'--lockdir',str(lockdir),'--run',str(run),'--comparison',str(comparison),'--output',str(aggregation)],
        [sys.executable,'-B',str(HERE/'validate_full.py'),'validate','--lockdir',str(lockdir),'--source-root',str(source),'--run',str(run),'--comparison',str(comparison),'--aggregation',str(aggregation),'--output',str(handoff)]
    ]
    write(out/'REQUEST.json',dict(watched_run=str(run),commands=commands,starts_or_restarts_fitting=False,reporting_code={n:record(HERE/n) for n in ('full_compare.py','aggregate_results.py','validate_full.py','finalize.py')}))
    try:
        while not(run/'COMPLETE.json').is_file():
            require(not(run/'FAILED.json').is_file(),'Full batch failed; finalization not started')
            time.sleep(10)
        require(read(out/'REQUEST.json')['reporting_code']=={n:record(HERE/n) for n in ('full_compare.py','aggregate_results.py','validate_full.py','finalize.py')},'Reporting code changed after finalizer start')
        results=[]
        for i,command in enumerate(commands):
            with (out/f'{i}.stdout').open('x') as stdout,(out/f'{i}.stderr').open('x') as stderr:
                r=subprocess.run(command,env=child_environment(),stdout=stdout,stderr=stderr)
            results.append(dict(command=command,returncode=r.returncode));require(r.returncode==0,'Finalization step failed: '+str(i))
            print('FINALIZATION STEP',i,'DONE',flush=True)
        write(out/'report.json',dict(status='passed',commands=results));seal(out,dict(scope='full_batch_finalization'))
    except BaseException as e:write(out/'FAILED.json',dict(error=repr(e)));raise
