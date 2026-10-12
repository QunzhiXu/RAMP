"""Bounded two-worker dispatcher for an already frozen/released suite."""
import sys
sys.dont_write_bytecode=True
import argparse,fcntl,json,os,signal,subprocess,time,shutil
from pathlib import Path
from datetime import datetime,timezone
from trainability.core import ROOT,require,file_sha256,validate_release,digest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--suite',required=True);args=parser.parse_args()
    suite_path=Path(args.suite);suite=json.loads(suite_path.read_text());profile=json.loads(Path(suite['profile']).read_text())
    require(suite['status']=='released','Suite is not released')
    require(file_sha256(suite['profile'])==suite['profile_sha256'],'Profile changed')
    hashes={str(p.relative_to(ROOT)):file_sha256(p) for p in (ROOT/'src').rglob('*') if p.suffix in {'.py','.R'}}
    code_sha=digest(hashes)
    for item in suite['shards']:
        require(file_sha256(item['config'])==item['sha256'],'Shard manifest changed')
        cfg=json.loads(Path(item['config']).read_text());release=json.loads(Path(cfg['release_evidence']).read_text())
        validate_release(cfg,release)
        require(release['configuration_frozen']['configuration_sha256']==item['sha256'],'Release config mismatch')
        require(release['configuration_frozen']['code_sha256']==code_sha,'Code changed since release')
    root=Path(suite['output']);require(root.resolve().is_relative_to(ROOT),'Output outside v2')
    root.mkdir(parents=True,exist_ok=True);logs=root/'logs';logs.mkdir(exist_ok=True)
    lock=(root/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock.write(str(os.getpid()));lock.flush()
    state_path=root/'RUNNER_STATUS.json';started=time.monotonic();queue=list(suite['shards']);active={};finished=[]
    env=os.environ.copy()
    env.update({k:'1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMBA_NUM_THREADS','LOKY_MAX_CPU_COUNT']})
    env['PYTHONDONTWRITEBYTECODE']='1'
    stopped=False
    def signal_stop(signum,frame):
        nonlocal stopped
        stopped=True  # Finish current shards; resume remaining ones explicitly.
    signal.signal(signal.SIGTERM,signal_stop);signal.signal(signal.SIGINT,signal_stop)
    def status(phase):
        value=dict(phase=phase,pid=os.getpid(),updated_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=time.monotonic()-started,pending_shards=len(queue),
            running=[dict(task=i['task'],method=i['method'],pid=p.pid,log=str(log)) for p,i,h,log in active.values()],
            finished=finished,free_gib=shutil.disk_usage(root).free/2**30,
            suite_sha256=file_sha256(suite_path),code_sha256=code_sha)
        tmp=root/'RUNNER_STATUS.tmp';tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(state_path)
    while queue or active:
        if time.monotonic()-started>profile['supervisor_max_hours']*3600:stopped=True
        if shutil.disk_usage(root).free<profile['minimum_free_gib']*2**30:stopped=True
        while queue and len(active)<profile['workers'] and not stopped:
            item=queue.pop(0);name=item['task']+'__'+item['method']
            log=logs/(name+'_'+str(time.time_ns())+'.log');handle=log.open('w')
            p=subprocess.Popen([sys.executable,str(ROOT/'src/run_experiment.py'),'--config',item['config']],
                cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            active[p.pid]=(p,item,handle,log)
        for pid,(p,item,handle,log) in list(active.items()):
            code=p.poll()
            if code is not None:
                handle.close();finished.append(dict(task=item['task'],method=item['method'],returncode=code,log=str(log)))
                del active[pid]
        status('draining_then_paused' if stopped else 'running')
        if stopped and not active:break
        if queue or active:time.sleep(10)
    if queue:
        status('paused_with_unstarted_shards');return 2
    status('summarizing')
    output=root/('summary_'+str(time.time_ns()))
    code=subprocess.call([sys.executable,str(ROOT/'src/summarize_formal_suite.py'),'--suite',str(suite_path),'--output',str(output)],cwd=ROOT,env=env)
    phase='complete' if code==0 and all(x['returncode']==0 for x in finished) else 'finished_with_failures'
    status(phase)
    (root/'LATEST_SUMMARY.txt').write_text(str(output)+'\n')
    return 0 if phase=='complete' else 2


if __name__=='__main__':raise SystemExit(main())
