"""Reporting-only handoff validation. Never fits heads or modifies completed artifacts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys
import tempfile
sys.dont_write_bytecode=True
from full_common import *
from full_run import verify_run
import setup_phase

def strict_batch_binding(lockdir,run):
    root=load_lock(lockdir);run=Path(run);req=read(run/'BATCH_REQUEST.json');m=read(run/'BATCH_MANIFEST.json')
    require(set(req)=={'scope','lock_sha256','code_records','groups','source_root','lockdir','workers','python'},'Batch request schema differs')
    require(set(m)=={'scope','lock_sha256','code_records','groups','jobs','predictions'},'Batch manifest schema differs')
    require(req['scope']==m['scope']=='full32370' and req['lock_sha256']==m['lock_sha256']==LOCK_SHA and req['code_records']==m['code_records']==codes(),'Batch control binding differs')
    require(type(req['workers']) is int and 1<=req['workers']<=8,'Batch worker count differs')
    check_grid(req['groups'],root['groups']);check_grid(list(m['groups']),root['groups'])
    require(type(m['jobs']) is int and m['jobs']==32370 and type(m['predictions']) is int and m['predictions']>0,'Batch counts differ')
    return root

def invoke(cmd,path,success=True):
    r=subprocess.run(cmd,env=child_environment(),capture_output=True,text=True)
    write(path,dict(command=cmd,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
    require((r.returncode==0)==success,'Unexpected command outcome');return r

def validate(lockdir,source,run,comparison,aggregate,outpath):
    out=reserve(outpath,[lockdir,source,run,comparison,aggregate]);root=strict_batch_binding(lockdir,run)
    try:
        baseline=setup_phase.capture();require(baseline==read(HERE/'validation/preflight_20261011/report.json'),'Prior baseline changed')
        compare_manifest=verify_seal(comparison);report=read(Path(comparison)/'report.json')
        require(report['batch_manifest']==record(Path(run)/'BATCH_MANIFEST.json'),'Compared batch changed')
        require(report['jobs']==32370 and report['status']=='exact' and report['exact_jobs']==32370 and report['changed_predictions']==0,'Full numerical comparison is not exact')
        verified=verify_run(lockdir,source,run)
        require(verified==report['structural'],'Structural verification differs')
        verify_seal(aggregate);agg=read(Path(aggregate)/'report.json')
        require(agg['status']=='passed' and agg['rows']==67800 and agg['jobs']==32370 and agg['comparison_report']==record(Path(comparison)/'report.json'),'Aggregation evidence differs')
        source_count=0
        with ThreadPoolExecutor(max_workers=4) as pool:
            names=list(root['source_files'])
            for n,r in zip(names,pool.map(lambda n:record(member(source,n)),names)):
                require(r==root['source_files'][n],'Bound training input changed after fitting: '+n);source_count+=1
        before=dict(batch=record(Path(run)/'BATCH_MANIFEST.json'),comparison=record(Path(comparison)/'report.json'),aggregate=record(Path(aggregate)/'report.json'))
        with tempfile.TemporaryDirectory(prefix='full_head_audit_') as temp:
            temp=Path(temp).resolve();mod=temp/'ramp-study/benchmarks/trainability/head_retraining_full';mod.mkdir(parents=True)
            for p in HERE.glob('*.py'):shutil.copy2(p,mod/p.name)
            shutil.copytree(HERE/'vendor',mod/'vendor');shutil.copytree(HERE/'tests',mod/'tests');shutil.copytree(lockdir,mod/'locks/full_20261011')
            for n,r in root['scientific'].items():copy(member(source,n),temp/'final_xu'/n,r)
            tests=invoke([sys.executable,'-B','-m','unittest','discover','-s',str(mod/'tests'),'-v'],out/'tests.json')
            import re
            match=re.search(r'Ran (\d+) tests',tests.stderr);require(match is not None,'Missing test count');n_tests=int(match[1])
            moved=temp/'moved lock with spaces';shutil.copytree(lockdir,moved)
            invoke([sys.executable,'-B',str(mod/'full_run.py'),'verify-run','--lockdir',str(moved),'--source-root',str(source),'--run',str(run)],out/'moved_lock.json')
            negative=[]
            for name in ('duplicate','missing','foreign','unknown_controller','manifest_scope','zero_count'):
                bad=temp/name;bad.mkdir();req=read(Path(run)/'BATCH_REQUEST.json');m=read(Path(run)/'BATCH_MANIFEST.json')
                if name=='duplicate':req['groups']=[req['groups'][0]]*860
                elif name=='missing':req['groups'].pop()
                elif name=='foreign':req['groups'][-1]='group_unknown'
                elif name=='unknown_controller':req['code_records']['full_worker.py']['sha256']='0'*64
                elif name=='manifest_scope':m['scope']='pilot45'
                else:m['jobs']=0
                write(bad/'BATCH_REQUEST.json',req);write(bad/'BATCH_MANIFEST.json',m);write(bad/'COMPLETE.json',dict(status='complete',manifest=record(bad/'BATCH_MANIFEST.json')))
                fingerprint=files(bad)
                r=invoke([sys.executable,'-B',str(HERE/'validate_full.py'),'check-binding','--lockdir',str(moved),'--run',str(bad)],out/('negative_'+name+'.json'),success=False)
                require('Batch control binding differs' in r.stderr or 'Full task grid differs' in r.stderr or 'Batch counts differ' in r.stderr,'Wrong refusal reason')
                require(files(bad)==fingerprint,'Read-only verification wrote to malformed batch');negative.append(name)
        probe=invoke([sys.executable,'-B','-c','import sys,json;sys.path.insert(0,sys.argv[1]);from full_common import environment;print(json.dumps(environment()))',str(HERE)],out/'current_runtime.json')
        current=json.loads(probe.stdout)
        for ident in root['groups']:require(current==read(Path(run)/ident/'environment_before.json'),'Current runtime differs from recorded runtime')
        require(before==dict(batch=record(Path(run)/'BATCH_MANIFEST.json'),comparison=record(Path(comparison)/'report.json'),aggregate=record(Path(aggregate)/'report.json')),'Completed evidence changed')
        require(setup_phase.capture()==baseline,'Previously accepted files changed during validation')
        result=dict(status='passed_internal_validation_pending_external_audit',tests=n_tests,negative_cli=negative,structural=verified,
            comparison={k:report[k] for k in ('jobs','exact_jobs','predictions','changed_predictions','candidate_records','byte_identical_npz')},published_tables=agg['comparisons'],
            source_files_checked=source_count,old_trainability_files=len(baseline['trainability_files']),external_non_run_files=len(baseline['external']),prior=baseline['prior'],
            evidence=before,code_records=codes(),reporting_code={n:record(HERE/n) for n in ('full_compare.py','aggregate_results.py','validate_full.py','finalize.py')},
            limitations='Frozen 860 representations; no representation refit or serialized head-weight comparison; no clean installation/cross-host proof; historical post-hoc choices retained')
        write(out/'report.json',result);seal(out,dict(scope='full_head_retraining_handoff'));print({k:result[k] for k in ('status','tests','comparison')})
    except BaseException as e:write(out/'FAILED.json',dict(error=repr(e)));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['validate','check-binding']);p.add_argument('--lockdir',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--source-root',type=Path);p.add_argument('--comparison',type=Path);p.add_argument('--aggregation',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    if a.mode=='check-binding':strict_batch_binding(a.lockdir,a.run);print('Strict batch binding verified')
    else:validate(a.lockdir.resolve(),a.source_root.resolve(),a.run.resolve(),a.comparison.resolve(),a.aggregation.resolve(),a.output)
