"""One create-only reporting pipeline after the continuation; never launches fits."""
import subprocess
import sys
sys.dont_write_bytecode=True
from resume_common import *

def finalize():
    import validate_combined
    lock=HERE/'locks/full_20261011';source=STUDY.parent/'final_xu';old=HERE/'runs/full_20261011';new=HERE/'runs/continuation_20261011'
    require((new/'COMPLETE.json').exists() and not(new/'FAILED.json').exists(),'Continuation not complete')
    out=reserve(HERE/'validation/continuation_finalization_20261011')
    initial={n:record(HERE/n) for n in validate_combined.REPORTING}
    write(out/'REQUEST.json',dict(controller=codes(),reporting_code=initial,continuation=record(new/'BATCH_MANIFEST.json')))
    index=HERE/'validation/combined_full_20261011';comparison=HERE/'validation/combined_comparison_20261011';aggregation=HERE/'validation/combined_aggregation_20261011';handoff=HERE/'validation/combined_handoff_20261011'
    flags=['--lockdir',str(lock),'--source-root',str(source),'--original-run',str(old),'--continuation',str(new),'--run',str(index)]
    commands=[['combined.py','assemble',*flags],['compare_combined.py',*flags,'--output',str(comparison)],
        ['aggregate_combined.py','--lockdir',str(lock),'--run',str(index),'--original-run',str(old),'--continuation',str(new),'--comparison',str(comparison),'--output',str(aggregation)],
        ['validate_combined.py',*flags,'--comparison',str(comparison),'--aggregation',str(aggregation),'--output',str(handoff)]]
    try:
        records=[]
        for i,args in enumerate(commands):
            require(initial=={n:record(HERE/n) for n in initial},'Reporting code changed')
            cmd=[sys.executable,'-B',str(HERE/args[0]),*args[1:]]
            print('REPORTING_STAGE',i,args[0],flush=True)
            with (out/f'stage_{i}.stdout').open('x') as stdout,(out/f'stage_{i}.stderr').open('x') as stderr:
                p=subprocess.run(cmd,env=child_environment(),stdout=stdout,stderr=stderr)
            records.append(dict(command=cmd,returncode=p.returncode));write(out/f'command_{i}.json',records[-1])
            require(p.returncode==0,'Reporting stage failed: '+args[0])
        require(initial=={n:record(HERE/n) for n in initial},'Reporting code changed')
        write(out/'report.json',dict(status='completed_pending_external_audit',commands=records,handoff=record(handoff/'report.json')));seal(out,dict(scope='continuation_finalization'))
        print('FINALIZED_PENDING_EXTERNAL_AUDIT',flush=True)
    except BaseException as e:write(out/'FAILED.json',dict(error=repr(e)));raise

if __name__=='__main__':finalize()
