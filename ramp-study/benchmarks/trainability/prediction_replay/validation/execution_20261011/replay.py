"""All-current-prediction rescore; never refit a model or use historical scores as inputs."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import gzip
import io
import json
import multiprocessing
from pathlib import Path
import time
import numpy as np
import pandas as pd
from common import *
import binding
from scores import Cohort, score_job

def representation_check(catalog,source):
    result=[];cohorts={}
    for name,r in catalog['representations'].items():
        c=cohorts.setdefault(r['case'],None)
        if c is None:
            c=cohorts[r['case']]=Cohort(r['case'],catalog['cohorts'][r['case']],source,catalog['files'])
        records=catalog['files']
        values=np.load(io.BytesIO(verified_bytes(source,r['values'],records)),allow_pickle=False)
        ids=np.load(io.BytesIO(verified_bytes(source,r['ids'],records)),allow_pickle=False)
        require(np.array_equal(ids,c.arrays['cell_id']),'Representation cell order: '+name)
        require(values.ndim==2 and len(values)==len(ids) and np.isfinite(values).all(),'Representation matrix: '+name)
        proof=bound_json(source,r['proof'],records)
        if r['case']=='lung_story' and not r['marker'] and r['method']=='ramp':
            release=bound_json(source,binding.LUNG+'/RELEASE_MANIFEST.json',records)
            require(release['alignment/RAMP_SHARED9.npy']==records[r['values']]['sha256'],'Lung release RAMP differs')
            pilot=bound_json(source,binding.ROOT+'/'+phase1.PILOT+'/MANIFEST.json',records)
            require(phase1.bind(pilot['representation'])==r['values'] and pilot['pool_sha256']==c.spec['pool_sha256'],'Lung RAMP route')
        else:
            require(proof['values_sha256']==records[r['values']]['sha256'] and proof['cell_ids_sha256']==records[r['ids']]['sha256'],'Representation proof: '+name)
            if 'shape' in proof:require(proof['shape']==list(values.shape),'Representation shape proof')
            if 'pool_sha256' in proof:require(proof['pool_sha256']==c.spec['pool_sha256'],'Representation pool')
            if 'labels_used' in proof:require(proof['labels_used'] is False,'Representation uses labels')
        if r['marker']:
            start=bound_json(source,r['folder']+'/START.json',records)
            canonical=c.spec['canonical_markers'][r['marker']]
            require(start['masked']==[canonical] and start['pool_sha256']==c.spec['pool_sha256'],'Masked representation differs')
            require(start['key']==proof['key'] and proof['method']==r['method'],'Masked representation identity')
            require(proof['input_contract']['canonical_markers'][r['marker']]==canonical,'Mask alias differs')
            n=len(c.spec['markers'])-(2 if r['method'] in ('fastmnn','maxfuse') else 1)
            require(values.shape[1]==n,'Masked feature count')
        else:
            n=len(c.spec['markers'])-(1 if r['method'] in ('fastmnn','maxfuse') else 0)
            if r['method']=='maxfuse':n=min(20,n)
            require(values.shape[1]==n,'Complete feature count: '+name)
        result.append(dict(representation=name,shape=list(values.shape),dtype=str(values.dtype),values_sha256=records[r['values']]['sha256'],cell_ids_sha256=records[r['ids']]['sha256']))
    require(len(result)==860 and sum(bool(r['marker']) for r in catalog['representations'].values())==790,'Representation total')
    return result

METRIC_FIELDS=('task','task_type','method','marker','source','target','direction','family','seed','job_id','metric','score')

def branch(index,anchor,source,out,case,kind,expected_code,limit=None):
    start=time.monotonic();require(code_records()==expected_code,'Controller changed before branch')
    catalog=binding.load(index,anchor);jobs=[r for r in catalog['jobs'] if r['job']['task']==case and r['job']['task_type']==kind]
    if limit is not None:jobs=jobs[:limit]
    folder=output(out,source,[index,*[HERE/n for n in CODE]])
    try:
        write(folder/'REQUEST.json',dict(case=case,task_type=kind,job_keys=[list(key(r)) for r in jobs],index_sha256=anchor,code_records=expected_code,environment=environment(),pilot=limit is not None))
        cohort=Cohort(case,catalog['cohorts'][case],source,catalog['files']);maxdiff=0.0;metric_count=0
        with (folder/'scores.jsonl.gz').open('xb') as raw, gzip.GzipFile(fileobj=raw,mode='wb',filename='',mtime=0,compresslevel=1) as zipped, (folder/'metrics.csv').open('x',newline='') as table:
            writer=csv.DictWriter(table,fieldnames=METRIC_FIELDS);writer.writeheader()
            for i,row in enumerate(jobs):
                result=score_job(row,cohort,source,catalog['files']);maxdiff=max(maxdiff,result['max_abs_difference'])
                zipped.write((json.dumps(result,sort_keys=True,allow_nan=False,separators=(',',':'))+'\n').encode())
                j=row['job']; fields={k:j.get(k,'') for k in ('task','task_type','method','marker','source','target','family','seed')}
                fields.update(direction=j['source']+' -> '+j['target'],job_id=j['id'])
                for metric in (('macro_f1','balanced_accuracy','accuracy') if kind=='J1' else ('effective_spearman','mae')):
                    require(result['score'][metric] is not None,'Missing primary score')
                    writer.writerow(fields|dict(metric=metric,score=result['score'][metric]));metric_count+=1
                if (i+1)%100==0:print(f'{kind}/{case}: {i+1}/{len(jobs)}',flush=True)
        require(code_records()==expected_code,'Controller changed during branch')
        report=dict(case=case,task_type=kind,jobs=len(jobs),metric_rows=metric_count,max_abs_difference=maxdiff,elapsed_seconds=time.monotonic()-start,environment=environment())
        require(report['environment']==read(folder/'REQUEST.json')['environment'],'Environment changed')
        write(folder/'report.json',report)
        manifest={p.name:record(p) for p in folder.iterdir() if p.is_file()}
        write(folder/'RUN_MANIFEST.json',manifest)
        write(folder/'COMPLETE.json',dict(status='complete',manifest=record(folder/'RUN_MANIFEST.json')))
        return report
    except BaseException as e:
        write(folder/'FAILED.json',dict(error=repr(e)));raise

def branch_names(catalog):
    return sorted({(r['job']['task'],r['job']['task_type']) for r in catalog['jobs']})

def verify_branch(folder,catalog,anchor,expected_code,case,kind,pilot=False):
    done=read(folder/'COMPLETE.json');require(done['status']=='complete' and done['manifest']==record(folder/'RUN_MANIFEST.json'),'Incomplete branch')
    manifest=read(folder/'RUN_MANIFEST.json')
    require(set(p.name for p in folder.iterdir())==set(manifest)|{'RUN_MANIFEST.json','COMPLETE.json'},'Branch file set')
    for n,r in manifest.items():require(record(member(folder,n))==r,'Branch payload changed')
    request=read(folder/'REQUEST.json');require(request['case']==case and request['task_type']==kind,'Branch swapped')
    require(request['index_sha256']==anchor and request['code_records']==expected_code and request['pilot']==pilot,'Branch controller/input differs')
    expected=[r for r in catalog['jobs'] if r['job']['task']==case and r['job']['task_type']==kind]
    if pilot:expected=expected[:1]
    require(request['job_keys']==[list(key(r)) for r in expected],'Branch task grid')
    actual=[];count=0;delta=0.0
    with gzip.open(folder/'scores.jsonl.gz','rt') as stream:
        for line in stream:
            r=json.loads(line);actual.append(r);count+=1;delta=max(delta,r['max_abs_difference'])
    check_grid(actual,expected)
    require([r['job'] for r in actual]==[r['job'] for r in expected],'Branch job records changed')
    report=read(folder/'report.json')
    require(report['jobs']==count and report['max_abs_difference']==delta,'Branch report differs')
    table=pd.read_csv(folder/'metrics.csv',keep_default_na=False,float_precision='round_trip')
    wanted=[]
    for r in actual:
        j=r['job']
        for metric in (('macro_f1','balanced_accuracy','accuracy') if kind=='J1' else ('effective_spearman','mae')):
            wanted.append(dict({k:j.get(k,'') for k in ('task','task_type','method','marker','source','target','family','seed')},direction=j['source']+' -> '+j['target'],job_id=j['id'],metric=metric,score=r['score'][metric]))
    pd.testing.assert_frame_equal(table,pd.DataFrame(wanted)[list(METRIC_FIELDS)],check_exact=True)
    return report

def verify_run(folder,catalog,anchor,pilot=False):
    require((folder/'COMPLETE.json').is_file(),'Batch incomplete')
    complete=read(folder/'COMPLETE.json');require(complete['status']=='complete' and complete['manifest']==record(folder/'BATCH_MANIFEST.json'),'Batch manifest differs')
    manifest=read(folder/'BATCH_MANIFEST.json');request=read(folder/'REQUEST.json')
    require(request['index_sha256']==anchor and request['pilot']==pilot and request['code_records']==code_records(),'Batch identity/controller')
    expected=[f'{k}__{c}' for c,k in branch_names(catalog)]
    require(manifest['branches']==expected and len(set(manifest['branches']))==len(expected),'Batch branch grid differs')
    require(set(p.name for p in (folder/'branches').iterdir())==set(expected),'Branch directory grid differs')
    require(set(manifest['files'])=={'REQUEST.json','representations.json','report.json'}|{f'branches/{b}/RUN_MANIFEST.json' for b in expected},'Batch file index differs')
    for n,r in manifest['files'].items():require(record(member(folder,n))==r,'Batch payload changed')
    reports=[]
    for c,k in branch_names(catalog):reports.append(verify_branch(folder/'branches'/f'{k}__{c}',catalog,anchor,request['code_records'],c,k,pilot))
    return dict(status='passed',branches=len(reports),jobs=sum(r['jobs'] for r in reports),metric_rows=sum(r['metric_rows'] for r in reports),max_abs_difference=max(r['max_abs_difference'] for r in reports))

def run(index,anchor,source,out,workers=3,pilot=False):
    require(workers in range(1,5),'Worker limit');catalog=binding.load(index,anchor);before=code_records()
    folder=output(out,source,[index,*[HERE/n for n in CODE]])
    try:
        write(folder/'REQUEST.json',dict(index_sha256=anchor,code_records=before,source_root=str(source.resolve()),environment=environment(),workers=workers,pilot=pilot))
        write(folder/'representations.json',representation_check(catalog,source))
        print('REPRESENTATIONS: 860 verified',flush=True)
        reports=[]
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures={pool.submit(branch,index,anchor,source,folder/'branches'/f'{k}__{c}',c,k,before,1 if pilot else None):(c,k) for c,k in branch_names(catalog)}
            for f in as_completed(futures):reports.append(f.result());print('FINISHED',futures[f],flush=True)
        require(before==code_records(),'Controller changed during batch')
        source_verification=binding.verify_files(catalog,source)
        for c,k in branch_names(catalog):verify_branch(folder/'branches'/f'{k}__{c}',catalog,anchor,before,c,k,pilot)
        write(folder/'report.json',dict(status='passed',branches=reports,source_verification=source_verification,code_records=before,pilot=pilot,models_refitted=0))
        names=['REQUEST.json','representations.json','report.json']+[f'branches/{k}__{c}/RUN_MANIFEST.json' for c,k in branch_names(catalog)]
        write(folder/'BATCH_MANIFEST.json',dict(branches=[f'{k}__{c}' for c,k in branch_names(catalog)],files={n:record(member(folder,n)) for n in names}))
        write(folder/'COMPLETE.json',dict(status='complete',manifest=record(folder/'BATCH_MANIFEST.json')))
        print('COMPLETE',folder,flush=True)
    except BaseException as e:
        write(folder/'FAILED.json',dict(error=repr(e)));raise

def aggregate(folder,catalog,anchor,out):
    import contract
    verified=verify_run(folder,catalog,anchor)
    rows=pd.concat([pd.read_csv(folder/'branches'/f'{k}__{c}'/'metrics.csv',keep_default_na=False,float_precision='round_trip') for c,k in branch_names(catalog)],ignore_index=True)
    require(len(rows)==67800,'Metric grid total')
    primary=contract.aggregate(rows);root=SNAPSHOT/'tree'/phase1.ROOT/phase1.FINAL
    reports={'primary_means':contract.compare_scores(primary,pd.DataFrame(read(root/'primary_means.json')))}
    bench=primary[primary.task!='ALL'].groupby(['task_type','task','metric','method'],as_index=False).score.mean()
    overall=bench.groupby(['task_type','metric','method'],as_index=False).score.mean();tables={'primary_means':primary,'PER_BENCHMARK':bench,'OVERALL':overall}
    for n,ks in [('PER_BENCHMARK',['task_type','task','metric','method']),('OVERALL',['task_type','metric','method'])]:reports[n]=contract.compare_scores(tables[n],contract.frame(root/(n+'.csv')),ks)
    for kind in ('J1','J2'):
        b=bench[bench.task_type==kind].drop(columns='task_type').copy()
        b['rank']=b.groupby(['task','metric']).score.transform(lambda s:s.rank(ascending=s.name[1]=='mae',method='min')).astype(int)
        o=b.groupby(['method','metric'],as_index=False).score.mean()
        o['rank']=o.groupby('metric').score.transform(lambda s:s.rank(ascending=s.name=='mae',method='min')).astype(int)
        for suffix,df,ks in [('PER_BENCHMARK',b,['task','metric','method']),('OVERALL',o,['method','metric'])]:
            name=kind+'_'+suffix;reports[name]=contract.compare_scores(df,contract.frame(root/(name+'.csv')),ks);tables[name]=df
    destination=output(out,Path(read(folder/'REQUEST.json')['source_root']),[folder,SNAPSHOT])
    try:
        for name,df in tables.items():
            with (destination/(name+'.csv')).open('x') as f:df.to_csv(f,index=False)
        write(destination/'report.json',dict(status='passed',run=verified,comparisons=reports,metrics_rows=len(rows),scope='Independent saved-prediction rescoring; accepted phase-1 nested aggregation; no retraining'))
        write(destination/'COMPLETE.json',dict(status='complete',files={p.name:record(p) for p in destination.iterdir()}))
        print(json.dumps(reports,indent=2))
    except BaseException as e:write(destination/'FAILED.json',dict(error=repr(e)));raise

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['run','verify-run','aggregate']);p.add_argument('--index',type=Path,required=True);p.add_argument('--anchor',required=True);p.add_argument('--source-root',type=Path);p.add_argument('--run',type=Path);p.add_argument('--output',type=Path);p.add_argument('--workers',type=int,default=3);p.add_argument('--pilot',action='store_true')
    a=p.parse_args()
    if a.mode=='run':require(a.source_root is not None and a.output is not None,'Source/output required');run(a.index,a.anchor,a.source_root,a.output,a.workers,a.pilot)
    elif a.mode=='verify-run':print(verify_run(a.run,binding.load(a.index,a.anchor),a.anchor,a.pilot))
    else:require(not a.pilot and a.output is not None,'Full completed run required');aggregate(a.run,binding.load(a.index,a.anchor),a.anchor,a.output)

if __name__=='__main__':main()
