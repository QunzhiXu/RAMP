"""All-current-prediction rescore; never refit a model or use historical scores as inputs."""
import argparse
from collections import Counter
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

# Verification-only compatibility with the exact four-file controller archived in
# validation/execution_20261011. Never accept an arbitrary historical controller.
# New execution still uses code_records() and retains its existing change guards.
ACCEPTED_REPLAY_CODE = {
    'binding.py': {'bytes': 9586, 'sha256': '8d716aed7090468d64f3feb680f1d477d07d429e367af92f3552fb04b14b9676'},
    'common.py': {'bytes': 3582, 'sha256': 'f641e49fe500e22fc060c6a59fc1b3a544fb6f4c0d09910331e98d55c3fac051'},
    'replay.py': {'bytes': 15052, 'sha256': '6d84f72426c215cc00264fcb11319a3469b78c7bbef7c39a3a6c94a229eb98d4'},
    'scores.py': {'bytes': 13981, 'sha256': '69e0c3413292828b11ccf28d05692cbf94e81c06644be894815384b59795cb6e'},
}
SCORE_TOLERANCE = 2e-14  # Unchanged from scores.compare_tree and the original handoff.


def check_controller_records(records):
    require(records == code_records() or records == ACCEPTED_REPLAY_CODE,
            'Batch identity/controller')


def fields(value, names, label):
    require(isinstance(value, dict) and set(value) == set(names), label + ' fields differ')


def finite_number(value, label, low=None, high=None):
    require(type(value) in (int, float) and np.isfinite(value), label + ' must be finite numeric')
    require((low is None or value >= low) and (high is None or value <= high),
            label + ' outside allowed range')
    return value


def integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, label + ' must be an integer with valid support')
    return value


def identity(value, label):
    require(isinstance(value, str) and value.strip() and value not in ('None', 'nan'), label + ' missing')


def close_number(actual, expected, label):
    if expected is None:
        require(actual is None, label + ' must be null')
    else:
        finite_number(actual, label)
        require(abs(actual - expected) <= SCORE_TOLERANCE, label + ' inconsistent with diagnostics')


def difference(value):
    return finite_number(value, 'max_abs_difference', 0, SCORE_TOLERANCE)


def confusion(value, size):
    require(isinstance(value, list) and len(value) == size and
            all(isinstance(row, list) and len(row) == size for row in value), 'Confusion matrix shape')
    for row in value:
        for x in row:
            finite_number(x, 'Confusion probability', 0, 1)
    matrix = np.asarray(value, dtype=float)
    close_number(float(matrix.sum()), 1.0, 'Confusion probability mass')
    return matrix


def classification_diagnostics(score, classes):
    fields(score, ('macro_f1', 'balanced_accuracy', 'accuracy', 'confusion_matrix',
                   'patients', 'cells', 'classes', 'zero_division', 'patient_confusion_matrices'), 'J1 score')
    require(score['classes'] == classes and len(classes) > 0 and len(set(classes)) == len(classes), 'J1 class order')
    require(type(score['zero_division']) is int and score['zero_division'] == 0, 'J1 zero_division')
    patients = integer(score['patients'], 'J1 patients', 1)
    cells = integer(score['cells'], 'J1 cells', patients)
    detail = score['patient_confusion_matrices']
    require(isinstance(detail, list) and len(detail) == patients, 'J1 patient support differs')
    matrices, ids = [], []
    for row in detail:
        fields(row, ('patient', 'matrix'), 'J1 patient diagnostic')
        identity(row['patient'], 'J1 patient'); ids.append(row['patient'])
        matrices.append(confusion(row['matrix'], len(classes)))
    require(ids == sorted(set(ids)), 'J1 patient identities duplicate or unordered')
    matrix = confusion(score['confusion_matrix'], len(classes))
    require(np.max(np.abs(matrix - np.mean(matrices, axis=0))) <= SCORE_TOLERANCE,
            'J1 confusion matrix is not patient-balanced')
    diagonal = matrix.diagonal(); denominator = matrix.sum(0) + matrix.sum(1); support = matrix.sum(1)
    expected = dict(accuracy=float(diagonal.sum()),
                    macro_f1=float(np.divide(2*diagonal, denominator, out=np.zeros(len(classes)), where=denominator>0).mean()),
                    balanced_accuracy=float(np.divide(diagonal, support, out=np.zeros(len(classes)), where=support>0).mean()))
    for name, value in expected.items():
        finite_number(score[name], 'J1 ' + name, -SCORE_TOLERANCE, 1 + SCORE_TOLERANCE)
        close_number(score[name], value, 'J1 ' + name)
    return dict(cells=cells, patients=ids, classes=classes)


def regression_diagnostics(score, baseline=False):
    names = {'status', 'effective_spearman', 'mae', 'mae_rho_support', 'rho_patients',
             'mae_patients', 'units', 'rho_status_counts'}
    fields(score, names if baseline else names | {'source_constant_baseline'}, 'J2 score')
    require(score['status'] == 'ok', 'J2 unsuccessful score')
    rows = score['units']; require(isinstance(rows, list) and rows, 'J2 units missing')
    ids, support = [], []
    for row in rows:
        fields(row, ('unit', 'patient', 'cells', 'mae', 'rho', 'effective_rho', 'rho_status'), 'J2 unit diagnostic')
        identity(row['unit'], 'J2 unit'); identity(row['patient'], 'J2 patient')
        ids.append(row['unit']); cells = integer(row['cells'], 'J2 unit cells', 1)
        finite_number(row['mae'], 'J2 unit MAE', 0, 1)
        status = row['rho_status']
        require(status in ('ok', 'constant_prediction', 'constant_truth', 'insufficient_truth_support'), 'J2 rho status')
        require((cells < 30) == (status == 'insufficient_truth_support'), 'J2 minimum truth support')
        if status == 'ok':
            require(not baseline, 'J2 source-constant baseline is not constant')
            finite_number(row['rho'], 'J2 unit rho', -1, 1)
            close_number(row['effective_rho'], row['rho'], 'J2 effective rho')
        elif status == 'constant_prediction':
            require(row['rho'] is None, 'J2 constant prediction rho must be null')
            close_number(row['effective_rho'], 0.0, 'J2 constant prediction effective rho')
        else:
            require(row['rho'] is None and row['effective_rho'] is None, 'J2 unsupported rho must be null')
        support.append((row['unit'], row['patient'], cells,
                        'supported_truth' if status in ('ok', 'constant_prediction') else status))
    require(ids == sorted(set(ids)), 'J2 unit identities duplicate or unordered')
    counts = score['rho_status_counts']
    require(isinstance(counts, dict) and counts == dict(Counter(r['rho_status'] for r in rows)), 'J2 status counts differ')
    for count in counts.values():integer(count, 'J2 status count', 1)
    def average(field, eligible=False):
        people = {}
        for row in rows:
            if row[field] is not None and (not eligible or row['effective_rho'] is not None):
                people.setdefault(row['patient'], []).append(row[field])
        return (float(np.mean([np.mean(v) for v in people.values()])) if people else None), len(people)
    rho, nr = average('effective_rho'); mae, nm = average('mae'); eligible_mae, _ = average('mae', True)
    integer(score['rho_patients'], 'J2 rho patients'); integer(score['mae_patients'], 'J2 MAE patients', 1)
    require(score['rho_patients'] == nr and score['mae_patients'] == nm, 'J2 patient support differs')
    for name, value in [('effective_spearman', rho), ('mae', mae), ('mae_rho_support', eligible_mae)]:
        close_number(score[name], value, 'J2 ' + name)
    if not baseline:
        require(regression_diagnostics(score['source_constant_baseline'], True) == support,
                'J2 baseline target support differs')
    return support


def score_record(result, expected, catalog):
    fields(result, ('job', 'score', 'max_abs_difference', 'supervision_sha256',
                    'target_ids_sha256', 'prediction_sha256', 'ledger_sha256'), 'Saved score record')
    job = expected['job']; require(result['job'] == job, 'Branch job records changed')
    difference(result['max_abs_difference'])
    for name in ('supervision_sha256', 'target_ids_sha256', 'prediction_sha256', 'ledger_sha256'):
        value = result[name]
        require(isinstance(value, str) and len(value) == 64 and set(value) <= set('0123456789abcdef'), 'Invalid ' + name)
    require(result['supervision_sha256'] == job['supervision_sha256'], 'Score supervision differs')
    require(result['prediction_sha256'] == catalog['files'][expected['folder']+'/predictions.npz']['sha256'], 'Score prediction binding differs')
    if job['task_type'] == 'J1':
        return classification_diagnostics(result['score'], catalog['cohorts'][job['task']]['classes'])
    support = regression_diagnostics(result['score'])
    finite_number(result['score']['effective_spearman'], 'J2 primary rho', -1, 1)
    finite_number(result['score']['mae'], 'J2 primary MAE', 0, 1)
    return support

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
    check_controller_records(expected_code)
    done=read(folder/'COMPLETE.json');require(done['status']=='complete' and done['manifest']==record(folder/'RUN_MANIFEST.json'),'Incomplete branch')
    manifest=read(folder/'RUN_MANIFEST.json')
    require(set(manifest)=={'REQUEST.json','scores.jsonl.gz','metrics.csv','report.json'},'Branch payload index differs')
    require(set(p.name for p in folder.iterdir())==set(manifest)|{'RUN_MANIFEST.json','COMPLETE.json'},'Branch file set')
    for n,r in manifest.items():require(record(member(folder,n))==r,'Branch payload changed')
    request=read(folder/'REQUEST.json');require(request['case']==case and request['task_type']==kind,'Branch swapped')
    require(request['index_sha256']==anchor and request['code_records']==expected_code and request['pilot']==pilot,'Branch controller/input differs')
    expected=[r for r in catalog['jobs'] if r['job']['task']==case and r['job']['task_type']==kind]
    if pilot:expected=expected[:1]
    require(request['job_keys']==[list(key(r)) for r in expected],'Branch task grid')
    actual=[];count=0;delta=0.0;target_support={};target_ids={}
    with gzip.open(folder/'scores.jsonl.gz','rt') as stream:
        for line in stream:
            require(count<len(expected),'Extra saved score record')
            r=json.loads(line);support=score_record(r,expected[count],catalog)
            job=r['job'];support_key=(job['target'],job.get('marker',''))
            require(target_support.setdefault(support_key,support)==support,'Target diagnostic support differs across jobs')
            require(target_ids.setdefault(job['target'],r['target_ids_sha256'])==r['target_ids_sha256'],'Target ID hash differs across jobs')
            actual.append(r);count+=1;delta=max(delta,r['max_abs_difference'])
    check_grid(actual,expected)
    require([r['job'] for r in actual]==[r['job'] for r in expected],'Branch job records changed')
    report=read(folder/'report.json')
    fields(report,('case','task_type','jobs','metric_rows','max_abs_difference','elapsed_seconds','environment'),'Branch report')
    integer(report['jobs'],'Branch jobs');integer(report['metric_rows'],'Branch metric rows')
    difference(report['max_abs_difference']);finite_number(report['elapsed_seconds'],'Branch elapsed seconds',0)
    require(report['case']==case and report['task_type']==kind and report['environment']==request['environment'],'Branch report identity/environment')
    require(report['jobs']==count and report['max_abs_difference']==delta,'Branch report differs')
    table=pd.read_csv(folder/'metrics.csv',keep_default_na=False,float_precision='round_trip')
    wanted=[]
    for r in actual:
        j=r['job']
        for metric in (('macro_f1','balanced_accuracy','accuracy') if kind=='J1' else ('effective_spearman','mae')):
            wanted.append(dict({k:j.get(k,'') for k in ('task','task_type','method','marker','source','target','family','seed')},direction=j['source']+' -> '+j['target'],job_id=j['id'],metric=metric,score=r['score'][metric]))
    pd.testing.assert_frame_equal(table,pd.DataFrame(wanted)[list(METRIC_FIELDS)],check_exact=True)
    require(len(table)==len(expected)*(3 if kind=='J1' else 2)==report['metric_rows'],'Branch metric count differs')
    return report

def verify_run(folder,catalog,anchor,pilot=False):
    require((folder/'COMPLETE.json').is_file(),'Batch incomplete')
    complete=read(folder/'COMPLETE.json');require(complete['status']=='complete' and complete['manifest']==record(folder/'BATCH_MANIFEST.json'),'Batch manifest differs')
    manifest=read(folder/'BATCH_MANIFEST.json');request=read(folder/'REQUEST.json')
    require(request['index_sha256']==anchor and type(request['pilot']) is bool and request['pilot']==pilot,'Batch identity/controller')
    check_controller_records(request['code_records'])
    expected=[f'{k}__{c}' for c,k in branch_names(catalog)]
    require(manifest['branches']==expected and len(set(manifest['branches']))==len(expected),'Batch branch grid differs')
    require(set(p.name for p in (folder/'branches').iterdir())==set(expected),'Branch directory grid differs')
    require(set(manifest['files'])=={'REQUEST.json','representations.json','report.json'}|{f'branches/{b}/RUN_MANIFEST.json' for b in expected},'Batch file index differs')
    for n,r in manifest['files'].items():require(record(member(folder,n))==r,'Batch payload changed')
    reports=[]
    for c,k in branch_names(catalog):reports.append(verify_branch(folder/'branches'/f'{k}__{c}',catalog,anchor,request['code_records'],c,k,pilot))
    summary=read(folder/'report.json')
    fields(summary,('status','branches','source_verification','code_records','pilot','models_refitted'),'Batch report')
    require(summary['status']=='passed' and summary['code_records']==request['code_records'] and
            type(summary['pilot']) is bool and summary['pilot']==pilot and
            type(summary['models_refitted']) is int and summary['models_refitted']==0,'Batch report identity')
    require(summary['source_verification']==dict(files=len(catalog['files']),bytes=sum(r['bytes'] for r in catalog['files'].values())),
            'Batch source verification counts differ')
    require(isinstance(summary['branches'],list) and len(summary['branches'])==len(reports),'Batch report branch count')
    # Compare complete branch reports, not only their self-reported totals.
    require(sorted(summary['branches'],key=lambda r:(r['case'],r['task_type']))==reports,'Batch report branches differ')
    jobs=sum(r['jobs'] for r in reports);metrics=sum(r['metric_rows'] for r in reports)
    selected=[r for r in catalog['jobs']] if not pilot else [next(r for r in catalog['jobs'] if (r['job']['task'],r['job']['task_type'])==(c,k)) for c,k in branch_names(catalog)]
    require(jobs==len(selected) and metrics==sum(3 if r['job']['task_type']=='J1' else 2 for r in selected),'Batch derived metric/job counts differ')
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
