"""Read-only prerequisites, frozen controller archive, then the single authorized full batch."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import subprocess
import sys
sys.dont_write_bytecode=True
from full_common import *
from full_run import fit
import setup_phase

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lockdir',type=Path,required=True);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True);p.add_argument('--python',required=True);p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    root=load_lock(a.lockdir);initial=codes();out=reserve(a.evidence,[a.lockdir,a.source_root,a.run])
    try:
        require(setup_phase.capture()==read(HERE/'validation/preflight_20261011/report.json'),'Accepted baseline changed')
        for n,r in root['vendor'].items():require(record(HERE/'vendor/pilot'/n)==r,'Frozen pilot implementation changed')
        smoke=read(HERE/'validation/group_smoke_20261011/report.json');require(smoke['status']=='passed' and smoke['code_records']==initial,'Smoke/controller mismatch')
        # Pin all 35k scientific inputs before fitting; no historical scores are opened.
        with ThreadPoolExecutor(max_workers=4) as pool:
            names=list(root['source_files'])
            for i,(n,r) in enumerate(zip(names,pool.map(lambda n:record(member(a.source_root,n)),names))):
                require(r==root['source_files'][n],'Bound training input changed: '+n)
                if (i+1)%5000==0:print('PREFLIGHT_HASHED',i+1,'/',len(names),flush=True)
        commands=[]
        with (out/'tests.stdout').open('x') as stdout,(out/'tests.stderr').open('x') as stderr:
            cmd=[a.python,'-B','-m','unittest','discover','-s',str(HERE/'tests'),'-v'];r=subprocess.run(cmd,env=child_environment(),stdout=stdout,stderr=stderr)
        require(r.returncode==0,'Preflight tests failed');commands.append(dict(command=cmd,returncode=r.returncode))
        probe=subprocess.run([a.python,'-B','-c','import sys,json;sys.path.insert(0,sys.argv[1]);from full_common import environment;print(json.dumps(environment()))',str(HERE)],env=child_environment(),capture_output=True,text=True)
        require(probe.returncode==0,'Runtime probe failed: '+probe.stderr);runtime=json.loads(probe.stdout)
        require(codes()==initial,'Controller changed during preflight')
        for n,r in initial.items():copy(HERE/n,out/'controller'/n,r)
        free=disk_check(HERE,root['expected_result_bytes']+2*1024**3)
        import psutil
        memory=psutil.virtual_memory();require(memory.available>=10*1024**3,'Insufficient available RAM for eight-worker batch')
        write(out/'report.json',dict(status='ready',code_records=initial,lock_sha256=LOCK_SHA,source_files=len(names),training_input_bytes=sum(r['bytes'] for r in root['source_files'].values()),
            expected_result_bytes=root['expected_result_bytes'],disk_free=free,disk_reserve=RESERVE_BYTES,memory_available=memory.available,cpu_count=os.cpu_count(),workers=a.workers,
            runtime=runtime,smoke=record(HERE/'validation/group_smoke_20261011/report.json'),commands=commands))
        seal(out,dict(scope='full_execution_preflight'))
        print(fit(a.lockdir,a.source_root,a.run,a.python,a.workers),flush=True)
    except BaseException as e:
        if not(out/'COMPLETE.json').exists():write(out/'FAILED.json',dict(error=repr(e)))
        raise
