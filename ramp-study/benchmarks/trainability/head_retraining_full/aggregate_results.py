"""Published-table reaggregation from newly retrained RESULT trees, not old predictions."""
import argparse
import importlib.util
import sys
sys.dont_write_bytecode=True
from full_common import *

def aggregate(lockdir,run,comparison,output):
    import pandas as pd
    root=load_lock(lockdir);comparison=Path(comparison);verified=verify_seal(comparison);proof=read(comparison/'report.json')
    require(proof['jobs']==32370 and proof['status']=='exact','Full exact comparison is required before published-table reaggregation')
    require(proof['batch_manifest']==record(Path(run)/'BATCH_MANIFEST.json'),'Compared batch changed')
    batch=read(Path(run)/'BATCH_MANIFEST.json')
    # Import the unchanged accepted pure aggregation implementation under its own name.
    pre=read(HERE/'validation/preflight_20261011/report.json')
    require(record(PARENT/'contract.py')==pre['trainability_files']['contract.py'],'Accepted aggregation changed')
    sys.path.insert(0,str(PARENT))
    spec=importlib.util.spec_from_file_location('accepted_trainability_aggregation',PARENT/'contract.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    rows=[]
    for ident in root['groups']:
        g=group_lock(lockdir,root,ident)
        require(record(Path(run)/ident/'RUN_MANIFEST.json')==batch['groups'][ident],'Compared group changed')
        group_manifest=read(Path(run)/ident/'RUN_MANIFEST.json')
        require(record(Path(run)/ident/'WORKER_DONE.json')==group_manifest['files']['WORKER_DONE.json'],'Compared job records changed')
        saved=read(Path(run)/ident/'WORKER_DONE.json')['jobs']
        for jobid,row in g['tasks'].items():
            j=row['job'];path=Path(run)/ident/'runtime/results'/jobid/'RESULT.json'
            # Group verification/comparison has bound these files; recheck each hash now.
            require(record(path)==saved[jobid]['result'],'Compared scientific result changed')
            result=read(path);score=result['score']
            for metric in (('macro_f1','balanced_accuracy','accuracy') if j['task_type']=='J1' else ('effective_spearman','mae')):
                item={k:j.get(k,'') for k in ('task','task_type','method','marker','source','target','family','seed')}
                item.update(direction=j['source']+' -> '+j['target'],job_id=j['id'],metric=metric,score=score[metric]);rows.append(item)
    require(len(rows)==67800,'Metric row count differs');df=pd.DataFrame(rows)
    require(not df.duplicated(['task','task_type','method','marker','source','target','family','seed','metric']).any(),'Duplicate metric identities')
    primary=old.aggregate(df);release=PARENT/'frozen_20261011_r2/tree'/old.inv.ROOT/old.inv.FINAL
    checks={'primary_means':old.compare_scores(primary,pd.DataFrame(read(release/'primary_means.json')))}
    bench=primary[primary.task!='ALL'].groupby(['task_type','task','metric','method'],as_index=False).score.mean()
    overall=bench.groupby(['task_type','metric','method'],as_index=False).score.mean();tables=dict(primary_means=primary,PER_BENCHMARK=bench,OVERALL=overall)
    for name,keys in [('PER_BENCHMARK',['task_type','task','metric','method']),('OVERALL',['task_type','metric','method'])]:checks[name]=old.compare_scores(tables[name],old.frame(release/(name+'.csv')),keys)
    for kind in ('J1','J2'):
        b=bench[bench.task_type==kind].drop(columns='task_type').copy()
        b['rank']=b.groupby(['task','metric']).score.transform(lambda s:s.rank(ascending=s.name[1]=='mae',method='min')).astype(int)
        o=b.groupby(['method','metric'],as_index=False).score.mean()
        o['rank']=o.groupby('metric').score.transform(lambda s:s.rank(ascending=s.name=='mae',method='min')).astype(int)
        for suffix,t,keys in [('PER_BENCHMARK',b,['task','metric','method']),('OVERALL',o,['method','metric'])]:
            name=kind+'_'+suffix;checks[name]=old.compare_scores(t,old.frame(release/(name+'.csv')),keys);tables[name]=t
    out=reserve(output,[lockdir,run,comparison,release]);
    for name,t in tables.items():
        with (out/(name+'.csv')).open('x') as f:t.to_csv(f,index=False)
    with (out/'per_job_metrics.csv').open('x') as f:df.to_csv(f,index=False)
    report=dict(status='passed',scope='Newly retrained 32370 heads to 67800 metrics to published tables',comparison_report=record(comparison/'report.json'),
        rows=67800,jobs=32370,comparisons=checks,aggregation_code=record(PARENT/'contract.py'))
    write(out/'report.json',report);seal(out,dict(scope='full_new_prediction_reaggregation'));return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lockdir',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--comparison',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(aggregate(a.lockdir,a.run,a.comparison,a.output))
