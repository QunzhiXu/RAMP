"""Small preflight evidence; never copies scientific input/result directories."""
import subprocess
import sys
import tempfile
sys.dont_write_bytecode=True
from resume_common import *

def prepare():
    root=load_lock(HERE/'locks/full_20261011');prior_binding(HERE/'runs/full_20261011',root)
    out=reserve(HERE/'validation/continuation_preflight_20261011')
    try:
        initial=codes()
        for n,r in initial.items():copy(HERE/n,out/'controller'/n,r)
        initial_files={p.relative_to(HERE).as_posix():record(p) for p in HERE.glob('*.py')}
        initial_files.update({p.relative_to(HERE).as_posix():record(p) for p in HERE.glob('*.md')})
        # Old artifacts stay in place; only test code, tiny locks and ten source modules are copied.
        with tempfile.TemporaryDirectory(prefix='head_resume_tests_') as t:
            temp=Path(t).resolve();mod=temp/'ramp-study/benchmarks/trainability/head_retraining_full';mod.mkdir(parents=True)
            for p in HERE.glob('*.py'):shutil.copy2(p,mod/p.name)
            for n in ('vendor','tests','locks'):shutil.copytree(HERE/n,mod/n)
            p='validation/interruption_20261011/report.json';copy(HERE/p,mod/p,record(HERE/p))
            for n,r in root['scientific'].items():copy(STUDY.parent/'final_xu'/n,temp/'final_xu'/n,r)
            cmd=[sys.executable,'-B','-m','unittest','discover','-s',str(mod/'tests'),'-v']
            result=subprocess.run(cmd,env=child_environment(),capture_output=True,text=True)
            write(out/'tests.json',dict(command=cmd,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
            require(result.returncode==0,'Continuation preflight tests failed')
        require(codes()==initial,'Continuation code changed during preflight')
        report=dict(status='passed',code_records=initial,prior_files=initial_files,interruption_sha256=INTERRUPTION_SHA,
            original_groups=845,original_jobs=31560,remaining_groups=remaining(root),remaining_jobs=810,resource_policy=RESOURCE_POLICY,free_bytes=disk_check(HERE),
            policy='No old result copies or deletion; original failed batch stays failed. Two versioned controller sets, same scientific execute function.')
        write(out/'report.json',report);seal(out,dict(scope='continuation_preflight'));print(report['status'],len(report['remaining_groups']),flush=True)
    except BaseException as e:write(out/'FAILED.json',dict(error=repr(e)));raise

if __name__=='__main__':prepare()
