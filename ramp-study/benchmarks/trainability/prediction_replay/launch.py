"""Archive the fixed controller and record a single all-task replay invocation."""
import argparse
import os
import subprocess
import sys
import time
from common import *

THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMBA_NUM_THREADS','LOKY_MAX_CPU_COUNT','NUMEXPR_NUM_THREADS')

def main():
    p=argparse.ArgumentParser();p.add_argument('--index',type=Path,required=True);p.add_argument('--anchor',required=True);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    evidence=output(a.output,a.source_root,[a.index,a.run]);before=code_records()
    try:
        for n in CODE:
            with (evidence/n).open('xb') as f:f.write((HERE/n).read_bytes())
        env=dict(os.environ);env.update({k:'1' for k in THREADS});env['PYTHONDONTWRITEBYTECODE']='1'
        command=[sys.executable,'-B',str(HERE/'replay.py'),'run','--index',str(a.index.resolve()),'--anchor',a.anchor,'--source-root',str(a.source_root.resolve()),'--output',str(a.run.resolve()),'--workers','3']
        write(evidence/'REQUEST.json',dict(command=command,code_records=before,thread_environment={k:env[k] for k in THREADS},index=record(a.index),started_unix=time.time()))
        with (evidence/'stdout.log').open('xb') as out,(evidence/'stderr.log').open('xb') as err:
            result=subprocess.run(command,stdout=out,stderr=err,env=env)
        require(result.returncode==0,'Rescore subprocess failed: '+str(result.returncode))
        require(code_records()==before,'Controlled files changed')
        write(evidence/'COMPLETE.json',dict(status='complete',finished_unix=time.time(),files={p.name:record(p) for p in evidence.iterdir()}))
        print('ALL-TASK REPLAY FINISHED',flush=True)
    except BaseException as e:write(evidence/'FAILED.json',dict(error=repr(e)));raise

if __name__=='__main__':main()
