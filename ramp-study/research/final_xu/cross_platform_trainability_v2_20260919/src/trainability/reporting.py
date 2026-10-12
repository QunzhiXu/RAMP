"""Manifest-bound summaries and paired Bayesian patient-cluster uncertainty.

Exponential weights are shared by registered global patient ID across every
method, seed, marker, direction and task. This is conditional target-patient
uncertainty, not a retraining bootstrap or uncertainty over new studies.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .core import require,seed_for,file_sha256,write_json,output_path
from .aggregation import resource_crossings


def eligible_patient_count(score,metric):
    return score['patients'] if 'macro_f1' in score else score['rho_patients' if metric=='effective_spearman' else 'mae_patients']


def patient_interval_allowed(score,identity_level,metric):
    return identity_level=='patient' and eligible_patient_count(score,metric)>=2


class PatientBootstrap:
    def __init__(self,draws=500,seed=20260920):
        require(draws>=2,'Insufficient bootstrap draws')
        self.draws=draws;self.seed=seed;self.cache={}

    def weights(self,patients):
        require(len(patients)>0 and len(set(patients))==len(patients),'Duplicate or empty patient support')
        for p in patients:
            if p not in self.cache:
                self.cache[p]=np.random.default_rng(seed_for(self.seed,'patient_bootstrap',p)).exponential(size=self.draws)
        return np.stack([self.cache[p] for p in patients],axis=1)

    def classification(self,score):
        rows=sorted(score['patient_confusion_matrices'],key=lambda r:r['patient'])
        matrices=np.array([r['matrix'] for r in rows]);k=matrices.shape[1]
        require(matrices.shape==(len(rows),k,k) and np.isfinite(matrices).all(),'Invalid patient confusion matrices')
        w=self.weights([r['patient'] for r in rows]);w=w/w.sum(axis=1,keepdims=True)
        cm=(w@matrices.reshape(len(rows),-1)).reshape(self.draws,k,k)
        tp=np.diagonal(cm,axis1=1,axis2=2);support=cm.sum(axis=2);den=support+cm.sum(axis=1)
        return dict(macro_f1=np.divide(2*tp,den,out=np.zeros_like(tp),where=den>0).mean(axis=1),
            balanced_accuracy=np.divide(tp,support,out=np.zeros_like(tp),where=support>0).mean(axis=1),
            accuracy=tp.sum(axis=1))

    def regression(self,score):
        result={}
        for key,metric in [('effective_rho','effective_spearman'),('mae','mae')]:
            grouped={}
            for row in score['units']:
                if row[key] is not None:grouped.setdefault(row['patient'],[]).append(row[key])
            if not grouped:result[metric]=None;continue
            pats=sorted(grouped);v=np.array([np.mean(grouped[p]) for p in pats]);w=self.weights(pats)
            result[metric]=(w@v)/w.sum(axis=1)
        return result


def nested_average(rows,value_key='score'):
    """Average repeats, markers within direction, directions, then tasks equally."""
    require(bool(rows),'Empty aggregation')
    current=[dict(r) for r in rows]
    for retain in [('task','direction','marker'),('task','direction'),('task',),()]:
        groups={}
        for r in current:groups.setdefault(tuple(r[k] for k in retain),[]).append(r[value_key])
        current=[dict(zip(retain,key),**{value_key:np.mean(values,axis=0)}) for key,values in sorted(groups.items())]
    return current[0][value_key]


def paired_rows(left,right):
    keys=lambda r:(r['task'],r['direction'],r['marker'],r['seed'])
    a={keys(r):r for r in left};b={keys(r):r for r in right}
    require(len(a)==len(left) and len(b)==len(right),'Duplicate primary comparison rows')
    require(a.keys()==b.keys(),'Comparison support differs')
    require(all(r['score'] is not None for r in left+right),'Missing score in paired comparison')
    diffs=[]
    for key in sorted(a):
        x,y=a[key],b[key]
        d=dict(x,score=x['score']-y['score'])
        d['bootstrap']=None if x['bootstrap'] is None or y['bootstrap'] is None else x['bootstrap']-y['bootstrap']
        diffs.append(d)
    return diffs


def paired_summary(left,right):
    diffs=paired_rows(left,right)
    estimate=float(nested_average(diffs));interval=None
    if all(r['bootstrap'] is not None for r in diffs):
        interval=np.quantile(nested_average(diffs,'bootstrap'),[.025,.975]).tolist()
    return dict(difference=estimate,interval95=interval,rows=len(diffs),tasks=len({r['task'] for r in diffs}),
        interval_type='paired Bayesian patient bootstrap' if interval else 'unavailable: unit-descriptive data or fewer than two eligible patients')


def summarize_suite(suite_path,*,destination):
    suite_path=Path(suite_path);suite=json.loads(suite_path.read_text())
    profile=json.loads(Path(suite['profile']).read_text())
    require(file_sha256(suite['profile'])==suite['profile_sha256'],'Profile changed')
    out=output_path(destination);require(not out.exists(),'Summary output exists');out.mkdir(parents=True)
    bootstrap=PatientBootstrap(**{k:profile['bootstrap'][k] for k in ['draws','seed']})
    coverage=[];rows=[];primary=[];scopes={};pretraining=[];contrast_rows=[]
    for shard in suite['shards']:
        require(file_sha256(shard['config'])==shard['sha256'],'Manifest changed')
        config=json.loads(Path(shard['config']).read_text());root=Path(config['output'])
        for job in config['jobs']:
            folder=root/'jobs'/job['id'];record=dict(job_id=job['id'],task=job['task'],method=job['method'],
                task_type=job['task_type'],status='not_run')
            if not (folder/'JOB_COMPLETE.json').exists():
                if (root/'failures'/(job['id']+'.json')).exists():record['status']='failed'
                coverage.append(record);continue
            complete=json.loads((folder/'JOB_COMPLETE.json').read_text())
            # Verify exactly the artifacts declared by the completed job; no glob-based discovery.
            for name,h in complete['artifact_sha256'].items():
                require(file_sha256(folder/name)==h,'Completed artifact changed: '+str(folder/name))
            record['status']='passed';coverage.append(record)
            if job['task_type']=='J3':
                payloads=[(arm,json.loads((folder/arm/'RESULT.json').read_text())) for arm in ['A-only','B-only','A+B']]
            elif job['task_type']=='J4':
                r=json.loads((folder/'RESULT.json').read_text());payloads=[]
                for name,score in r['scores'].items():
                    payloads.append((name,dict(score=score,target_identity_level=r['target_identity_level'],
                        selection=r['training'][name],total_supervision=r['total_probe_supervision'])))
                pretraining.append(dict(task=job['task'],method=job['method'],seed=job['seed'],target=job['target'],
                    shared_baseline=r['shared_baseline'],comparison_scope=r['comparison_scope'],training=r['training']))
            else:payloads=[('',json.loads((folder/'RESULT.json').read_text()))]
            for arm,result in payloads:
                score=result['score'];classification='macro_f1' in score
                metrics=['macro_f1','balanced_accuracy','accuracy'] if classification else ['effective_spearman','mae']
                draw=bootstrap.classification(score) if classification else bootstrap.regression(score)
                for metric in metrics:
                    n_patients=eligible_patient_count(score,metric)
                    patient_interval=patient_interval_allowed(score,result['target_identity_level'],metric)
                    row=dict(job_id=job['id'],task=job['task'],task_type=job['task_type'],method=job['method'],
                        family=job['family'],direction=job.get('source',' + '.join(job.get('sources',[])))+' -> '+job['target'],
                        marker=job.get('marker',''),seed=job['seed'],arm=arm,metric=metric,score=score[metric],
                        fit_budget=job.get('fit_budget'),width=job.get('width'),
                        labelled_patient_count=job.get('labelled_patient_count'),
                        total_supervision=result['total_supervision'],
                        parameters=result['selection'].get('trainable_head_parameters',result['selection'].get('trainable_parameters')),
                        identity_level=result['target_identity_level'],
                        eligible_patients=n_patients,patient_interval_available=patient_interval,
                        training_seconds=result.get('training_seconds_including_selection'),
                        inference_seconds=result.get('inference_seconds'))
                    rows.append(row)
                    scopes[job['id']]=job['analysis_scopes']
                    if 'primary' in job['analysis_scopes']:
                        primary.append(dict(row,bootstrap=draw[metric] if patient_interval else None))
                    if job['task_type'] in {'J3','J4'} and metric=='macro_f1':
                        contrast_rows.append(dict(row,bootstrap=draw[metric] if patient_interval else None))
    pd.DataFrame(coverage).to_csv(out/'coverage.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    complete=all(r['status']=='passed' for r in coverage)
    comparisons=[];means=[];curves=[];contrasts=[];interactions=[];seed_variation=[]
    # Full inferential/curve summaries require completed manifest coverage.
    # Partial/failing runs still expose every job status and individual score.
    if complete:
        for arm in ['A-only_finetuned','A+B_finetuned']:
            for task in ['ALL']+sorted({r['task'] for r in contrast_rows if r['task_type']=='J4'}):
                selected=[r for r in contrast_rows if r['task_type']=='J4' and (task=='ALL' or r['task']==task)]
                effects={}
                for method in profile['methods']:
                    ft=[r for r in selected if r['method']==method and r['arm']==arm]
                    rs=[r for r in selected if r['method']==method and r['arm']=='random_supervised']
                    if ft and rs:effects[method]=paired_rows(ft,rs)
                if 'ramp' in effects:
                    for method in effects:
                        if method!='ramp':interactions.append(dict(task=task,arm=arm,left='ramp',right=method,
                            definition='(FT-RS)_RAMP - (FT-RS)_comparator',**paired_summary(effects['ramp'],effects[method])))
        for kind,pairs in [('J3',[('A+B','A-only'),('A+B','B-only')]),
                           ('J4',[('A-only_finetuned','random_supervised'),('A+B_finetuned','random_supervised'),
                                  ('A+B_finetuned','A-only_finetuned')])]:
            for method in profile['methods']:
                for family in sorted({r['family'] for r in contrast_rows if r['task_type']==kind}):
                    chosen=[r for r in contrast_rows if r['task_type']==kind and r['method']==method and r['family']==family]
                    for task in ['ALL']+sorted({r['task'] for r in chosen}):
                        subset=[r for r in chosen if task=='ALL' or r['task']==task]
                        for left,right in pairs:
                            a=[r for r in subset if r['arm']==left];b=[r for r in subset if r['arm']==right]
                            if a and b:contrasts.append(dict(task=task,task_type=kind,method=method,family=family,
                                left=left,right=right,**paired_summary(a,b)))
        for kind in ['J1','J2']:
            for family in ['linear','knn','mlp']:
                metrics=['macro_f1','balanced_accuracy','accuracy'] if kind=='J1' else ['effective_spearman','mae']
                for metric in metrics:
                    selected=[r for r in primary if r['task_type']==kind and r['family']==family and r['metric']==metric]
                    for task in ['ALL']+sorted({r['task'] for r in selected}):
                        subset=[r for r in selected if task=='ALL' or r['task']==task]
                        # None Spearman is truth-ineligible, not a failed prediction.
                        eligibility={ (r['task'],r['direction'],r['marker'],r['seed']) for r in subset if r['score'] is not None }
                        for method in profile['methods']:
                            a=[r for r in subset if r['method']==method and (r['task'],r['direction'],r['marker'],r['seed']) in eligibility]
                            if not a:continue
                            require(all(r['score'] is not None for r in a),'Method-specific missing score')
                            expected=len(eligibility);require(len(a)==expected,'Incomplete primary method coverage')
                            means.append(dict(task=task,task_type=kind,family=family,metric=metric,method=method,
                                score=float(nested_average(a)),rows=expected))
                            by_seed=[float(nested_average([r for r in a if r['seed']==s])) for s in profile['seeds'] if any(r['seed']==s for r in a)]
                            seed_variation.append(dict(task=task,task_type=kind,family=family,metric=metric,method=method,
                                repeat_means=by_seed,standard_deviation=float(np.std(by_seed,ddof=1)) if len(by_seed)>1 else None,
                                interpretation='model/sampling variation; not independent biological replicates'))
                            if method=='ramp':continue
                            b=[r for r in subset if r['method']=='ramp' and (r['task'],r['direction'],r['marker'],r['seed']) in eligibility]
                            comparisons.append(dict(task=task,task_type=kind,family=family,metric=metric,
                                left='ramp',right=method,**paired_summary(b,a)))
        frame=pd.DataFrame(rows)
        if not frame.empty:
            for scope,resource in [('learning','total_supervision'),('capacity','parameters'),('donors','labelled_patient_count')]:
                f=frame[frame.job_id.map(lambda j:scope in scopes[j]) & frame.metric.isin(['macro_f1','effective_spearman'])]
                keys=['task','task_type','method','family','direction','marker','metric']
                for key,g in f.groupby(keys,dropna=False,sort=True):
                    points=[]
                    for x,h in g.groupby(resource,dropna=False):
                        ok=set(h.seed)==set(profile['seeds']) and len(h)==len(profile['seeds']) and h.score.notna().all()
                        points.append((x,float(h.score.mean()) if ok else None))
                    points.sort()
                    if not points or any(not np.isfinite(x) or x<=0 for x,_ in points):continue
                    # Expected resource grid is encoded in the manifest; missing/infeasible
                    # points are not interpolated. Publish the tested range with every curve.
                    for crossing in resource_crossings([p[0] for p in points],[p[1] for p in points]):
                        curves.append(dict(zip(keys,key),scope=scope,resource=resource,
                            tested_resources=[p[0] for p in points],**crossing))
        write_json(out/'primary_means.json',means)
        write_json(out/'paired_comparisons.json',comparisons)
        write_json(out/'resource_crossings.json',curves)
        write_json(out/'cohort_pretraining_contrasts.json',contrasts)
        write_json(out/'pretraining_interactions.json',interactions)
        write_json(out/'seed_variation.json',seed_variation)
    write_json(out/'pretraining_diagnostics.json',pretraining)
    write_json(out/'REPORT.json',dict(status='complete' if complete else 'incomplete_no_aggregate_claim',
        jobs=len(coverage),passed=sum(r['status']=='passed' for r in coverage),
        failed=sum(r['status']=='failed' for r in coverage),not_run=sum(r['status']=='not_run' for r in coverage),
        score_rows=len(rows),primary_comparisons=len(comparisons),suite_sha256=file_sha256(suite_path),
        uncertainty=profile['bootstrap'],excluded_plans='validation/formal_preflight_v0_5/EXCLUSIONS.json',
        caveat='Primary means are descriptive over frozen tasks. Patient intervals condition on fitted models. Missing jobs prevent aggregate release; truth-ineligible correlations are reported separately from failures.'))
    return out
