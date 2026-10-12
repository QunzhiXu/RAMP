"""Single-use reporting finalizer. Waits for the existing batch; never starts one."""
import subprocess
import sys
import time
from common import *

def main():
    source=PARENT.parents[2]/'final_xu';index=HERE/'bindings/current_predictions_20261011/catalog.json'
    anchor='82eafb7e4e41652b23844b53247081e3242163cc606004771734d2ba51ea7c4d'
    run=HERE/'runs/all_predictions_20261011'
    dest=output(HERE/'validation/finalization_20261011',source,[index,run])
    args=['--index',str(index),'--anchor',anchor,'--run',str(run)]
    aggregate=HERE/'validation/aggregation_20261011';per_job=HERE/'validation/per_job_20261011'
    commands=[
        [sys.executable,'-B',str(HERE/'compare.py'),*args,'--output',str(per_job)],
        [sys.executable,'-B',str(HERE/'replay.py'),'aggregate',*args,'--output',str(aggregate)],
        [sys.executable,'-B',str(HERE/'validate.py'),*args,'--source-root',str(source),'--comparison',str(aggregate),'--per-job',str(per_job),'--output',str(HERE/'validation/handoff_20261011')],
    ]
    try:
        write(dest/'REQUEST.json',dict(commands=commands,watched_batch=str(run),starts_scientific_batch=False))
        while not (run/'COMPLETE.json').is_file():
            require(not (run/'FAILED.json').is_file(),'Scientific replay failed; no reporting attempted')
            time.sleep(10)
        results=[]
        for i,command in enumerate(commands):
            with (dest/f'{i}.stdout').open('xb') as out,(dest/f'{i}.stderr').open('xb') as err:
                result=subprocess.run(command,stdout=out,stderr=err)
            results.append(dict(command=command,returncode=result.returncode))
            require(result.returncode==0,'Reporting step failed: '+str(i))
            print('FINALIZATION STEP',i,'DONE',flush=True)
        write(dest/'COMPLETE.json',dict(status='complete',commands=results,files={p.name:record(p) for p in dest.iterdir()}))
    except BaseException as e:write(dest/'FAILED.json',dict(error=repr(e)));raise

if __name__=='__main__':main()
