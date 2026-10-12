"""Read-only full handoff checks, with small temporary audit fixtures only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import re
import subprocess
import sys
import tempfile
sys.dont_write_bytecode=True
from full_common import *
import combined
import setup_phase

REPORTING=('combined.py','compare_combined.py','aggregate_combined.py','validate_combined.py','finalize_continuation.py','make_handoff_summary.py','full_compare.py','aggregate_results.py')

def invoke(cmd,path,success=True):
    r=subprocess.run(cmd,env=child_environment(),capture_output=True,text=True)
    write(path,dict(command=cmd,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
    require((r.returncode==0)==success,'Unexpected CLI outcome');return r

def validate(lockdir,source,original_run,continuation,run,comparison,aggregation,outpath):
    root,batch=combined.binding(lockdir,original_run,continuation,run)
    out=reserve(outpath,[lockdir,source,original_run,continuation,run,comparison,aggregation])
    try:
        code={n:record(HERE/n) for n in REPORTING}
        baseline=setup_phase.capture();require(baseline==read(HERE/'validation/preflight_20261011/report.json'),'Previously accepted baseline changed')
        print('PRIOR_BASELINE_OK',flush=True)
        pre=read(HERE/'validation/continuation_preflight_20261011/report.json')
        for n,r in pre['prior_files'].items():require(record(HERE/n)==r,'Pre-continuation file changed: '+n)
        failed_before={i:files(Path(original_run)/i) for i in combined.interruption()['failed_groups']}
        verify_seal(comparison);proof=read(Path(comparison)/'report.json')
        require(proof['status']=='exact' and proof['jobs']==proof['exact_jobs']==32370 and proof['changed_predictions']==0,'Full comparison not exact')
        require(proof['batch_manifest']==record(Path(run)/'BATCH_MANIFEST.json'),'Compared combined manifest changed')
        # Independently count every comparison job record, not just top-level self reports.
        from make_handoff_summary import summarize
        summary=summarize(lockdir,comparison,out/'branch_summary')
        require(summary['totals']['candidate_records']==280540,'Candidate count differs')
        verify_seal(aggregation);agg=read(Path(aggregation)/'report.json')
        require(agg['status']=='passed' and agg['jobs']==32370 and agg['rows']==67800 and agg['comparison_report']==record(Path(comparison)/'report.json'),'Aggregation binding differs')
        with ThreadPoolExecutor(max_workers=4) as pool:
            names=list(root['source_files'])
            for i,(n,r) in enumerate(zip(names,pool.map(lambda n:record(member(source,n)),names))):
                require(r==root['source_files'][n],'Training input changed: '+n)
                if (i+1)%10000==0:print('TRAINING_INPUTS_CHECKED',i+1,flush=True)
        with tempfile.TemporaryDirectory(prefix='combined_heads_audit_') as t:
            temp=Path(t).resolve();mod=temp/'ramp-study/benchmarks/trainability/head_retraining_full';mod.mkdir(parents=True)
            for p in HERE.glob('*.py'):shutil.copy2(p,mod/p.name)
            for n in ('vendor','tests','locks'):shutil.copytree(HERE/n,mod/n)
            for n in ('validation/interruption_20261011/report.json','validation/continuation_preflight_20261011/report.json'):copy(HERE/n,mod/n,record(HERE/n))
            for n,r in root['scientific'].items():copy(member(source,n),temp/'final_xu'/n,r)
            test=invoke([sys.executable,'-B','-m','unittest','discover','-s',str(mod/'tests'),'-v'],out/'tests.json')
            m=re.search(r'Ran (\d+) tests',test.stderr);require(m is not None,'Missing test count');ntests=int(m[1])
            moved=temp/'moved locks with spaces';shutil.copytree(lockdir,moved)
            base=[sys.executable,'-B',str(mod/'combined.py')]
            flags=['--lockdir',str(moved),'--source-root',str(source),'--original-run',str(original_run),'--continuation',str(continuation)]
            moved_result=invoke(base+['verify-run',*flags,'--run',str(run)],out/'moved_verification.json')
            import ast
            require(ast.literal_eval(moved_result.stdout.strip().splitlines()[-1])==proof['structural'],'Moved verification differs')
            negative=[]
            for name in ('duplicate','missing','extra','foreign','traversal','swapped_origin','unknown_controller','zero_jobs','wrong_predictions','pilot_scope','swapped_hash','missing_complete','failed'):
                bad=temp/name;bad.mkdir();value=read(Path(run)/'BATCH_MANIFEST.json')
                if name=='duplicate':value['groups']=[value['groups'][0]]*860
                elif name=='missing':value['groups'].pop()
                elif name=='extra':value['groups'].append(dict(id='extra',origin='continuation',manifest={}))
                elif name=='foreign':value['groups'][0]['id']='foreign'
                elif name=='traversal':value['groups'][0]['id']='../group_0000'
                elif name=='swapped_origin':value['groups'][0]['origin']='continuation'
                elif name=='unknown_controller':value['controllers']['original']['full_worker.py']['sha256']='0'*64
                elif name=='zero_jobs':value['jobs']=0
                elif name=='wrong_predictions':value['predictions']=0
                elif name=='pilot_scope':value['scope']='pilot45'
                elif name=='swapped_hash':value['groups'][0]['manifest']=value['groups'][1]['manifest']
                write(bad/'BATCH_MANIFEST.json',value);write(bad/'verification.json',read(Path(run)/'verification.json'))
                if name!='missing_complete':seal(bad,dict(scope='combined_full32370'))
                if name=='failed':write(bad/'FAILED.json',dict(error='injected failure'))
                before=files(bad)
                r=invoke(base+['check-binding',*flags,'--run',str(bad)],out/('negative_'+name+'.json'),False)
                require(any(s in r.stderr for s in ('Combined','Full task grid differs','Incomplete artifact','COMPLETE.json')),'Unexpected negative reason')
                rejected_output=temp/('aggregate_'+name)
                invoke([sys.executable,'-B',str(HERE/'aggregate_combined.py'),'--lockdir',str(moved),'--run',str(bad),'--comparison',str(comparison),'--output',str(rejected_output),'--original-run',str(original_run),'--continuation',str(continuation)],out/('negative_aggregate_'+name+'.json'),False)
                require(not rejected_output.exists(),'Rejected aggregation created output')
                require(files(bad)==before,'Verifier wrote to negative fixture');negative.append(name)
        print('MOVED_AND_NEGATIVE_CHECKS_OK',flush=True)
        probe=invoke([sys.executable,'-B','-c','import sys,json;sys.path.insert(0,sys.argv[1]);from full_common import environment;print(json.dumps(environment()))',str(HERE)],out/'current_runtime.json')
        runtime=json.loads(probe.stdout)
        for route in batch['groups']:require(read(combined.group_path(original_run,continuation,route)/'environment_before.json')==runtime,'Current runtime differs')
        require(failed_before=={i:files(Path(original_run)/i) for i in failed_before},'Failed partial artifacts changed')
        require(setup_phase.capture()==baseline,'Previously accepted files changed during validation')
        require(code=={n:record(HERE/n) for n in REPORTING},'Reporting code changed during validation')
        combined.binding(lockdir,original_run,continuation,run)
        result=dict(status='passed_internal_validation_pending_external_audit',tests=ntests,negative_cli=negative,structural=proof['structural'],
            comparison=summary['totals'],published_tables=agg['comparisons'],source_files_checked=len(names),old_trainability_files=len(baseline['trainability_files']),
            external_non_run_files=len(baseline['external']),prior=baseline['prior'],pre_continuation_files=len(pre['prior_files']),failed_partial_files_checked=sum(len(v) for v in failed_before.values()),
            evidence=dict(combined=record(Path(run)/'BATCH_MANIFEST.json'),comparison=record(Path(comparison)/'report.json'),aggregation=record(Path(aggregation)/'report.json')),
            reporting_code=code,controllers=batch['controllers'],disk_free_bytes=shutil.disk_usage(HERE).free,
            limits='Frozen representations; head weights not serialized; no clean install/cross-host, no clinical/UMAP/figures; historical exploratory choices retained. No external audit approval.')
        write(out/'report.json',result);seal(out,dict(scope='full_head_continuation_handoff'));print(result['status'],result['comparison'],flush=True)
    except BaseException as e:write(out/'FAILED.json',dict(error=repr(e)));raise

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('lockdir','source-root','original-run','continuation','run','comparison','aggregation','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();validate(a.lockdir,a.source_root,a.original_run,a.continuation,a.run,a.comparison,a.aggregation,a.output)
