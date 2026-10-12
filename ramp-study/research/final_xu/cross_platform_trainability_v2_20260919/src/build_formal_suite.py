"""Build explicit jobs and exclusions from source support, never model performance."""
import sys
sys.dont_write_bytecode=True
import itertools,json
from collections import Counter
from dataclasses import asdict
import numpy as np
from trainability.core import ROOT,Q_GRID,ProtocolError,digest,write_json,file_sha256,require
from trainability.real_data import load_task
from trainability.runner import make_plan,make_j3_plans
from trainability.j4 import PretrainConfig,make_plans
from trainability.heads import Candidate
from trainability.scoring import regression_score

METHODS=['unaligned','ramp','harmony_direct','cycombine','cytonorm','fastmnn','maxfuse','clipped_zscore','minmax','percentile']
SEEDS=[20260919,20260920,20260921]


def candidates(family,width=32):
    result=[]
    for scale in ['identity','standard']:
        if family=='linear':
            result += [asdict(Candidate(family,scale=scale,logistic_c=c,ridge_alpha=c)) for c in [.1,1.,10.]]
        elif family=='knn':
            result += [asdict(Candidate(family,scale=scale,neighbors=k,weights=w))
                       for k in [5,15,50] for w in ['uniform','distance']]
        else:
            result += [asdict(Candidate(family,scale=scale,width=width,mlp_alpha=a,learning_rate=lr,
                       max_epochs=100,patience=15,batch_size=128)) for a in [.001,1.] for lr in [.001,.01]]
    return result


def main():
    folder=ROOT/'config/formal_v0_5';folder.mkdir(exist_ok=False)
    out=ROOT/'validation/formal_preflight_v0_5';out.mkdir(exist_ok=False)
    budget=json.loads((ROOT/'validation/formal_budget_v0_5/REPORT.json').read_text())
    require(budget['status']=='passed' and not budget['target_scores_computed'],'Budget pilot not eligible')
    profile=dict(methods=METHODS,seeds=SEEDS,primary_fit_cells=500,validation_cells=512,
        calibration_per_unit=64,learning_fit_cells=[20,100,500,2000],capacity_widths=[8,32,128],
        learning_width=32,labelled_patients=[2,4,8,16,32],
        j2_primary_markers='all eligible shared markers, all ordered directions',
        j2_resource_directions='first registered dataset to last registered dataset, all eligible shared markers',
        j1_resource_directions='all ordered directions',
        j3_fit_quota_per_class=25,j3_validation_quota_per_class=10,
        j4=asdict(PretrainConfig(width=32,probe_fit=500,probe_validation=512,
            supervised_updates=budget['selected_supervised_updates'],pretrain_updates=300,validation_every=100)),
        bootstrap=dict(kind='paired Bayesian patient-cluster bootstrap',draws=500,seed=20260920,
            identity='registered global dataset::patient identity; common patients receive identical exponential weights across methods, directions, markers, seeds and tasks',
            scope='conditional on frozen models/splits; unit-descriptive directions have no patient CI'),
        workers=2,minimum_free_gib=40,supervisor_max_hours=72,
        formal_scope='finite grid; no gating; no target-performance screening',
        source_budget_report_sha256=file_sha256(ROOT/'validation/formal_budget_v0_5/REPORT.json'))
    write_json(folder/'PROFILE.json',profile)
    infos=json.loads((ROOT/'data/real_inputs_v0_3_r2/SUMMARY.json').read_text())['tasks']
    exclusions=[];shards=[];counts=Counter();source_hashes={};pool_records=[];direction_records=[]
    for info in infos:
        task=load_task(info['task']);name=task.name;md=task.metadata.set_index('cell_id')
        lookup={x:i for i,x in enumerate(task.pool.cell_ids)}
        datasets=list(dict.fromkeys(task.pool.datasets));base=[];plan_cache={};validation_cache={}
        source_hashes.update(task.receipt['sources'])
        pool_records.append(dict(task=name,pool_sha256=task.pool.sha256,cells=len(md),
            manifest=task.receipt['manifest_path'],manifest_sha256=file_sha256(task.receipt['manifest_path'])))
        def add(job):
            try:
                if job['task_type'] in {'J1','J2'}:
                    marker=job.get('marker');key=(job['source'],marker is not None,job['fit_budget'],
                        job.get('labelled_patient_count'),job['seed'])
                    if marker and not all(c['j2_eligible'] for c in task.contracts.values()):
                        raise ProtocolError('Ptacek upstream dependency unresolved in full U; strict J2 excluded')
                    if key not in plan_cache:
                        plan_cache[key]=make_plan(task,job['source'],job['target'],fit_budget=job['fit_budget'],
                            validation_budget=512,seed=job['seed'],marker=marker,
                            calibration_per_unit=64 if marker else 0,labelled_patient_count=job.get('labelled_patient_count'))
                    plan,mode=plan_cache[key]
                    if marker:
                        vk=(job['source'],marker,job['seed'])
                        if vk not in validation_cache:
                            ids=list(plan.validation);truth=task.native[[lookup[i] for i in ids],task.markers.index(marker)]
                            v=regression_score(truth,np.zeros(len(ids)),md.loc[ids,'unit_key'].to_numpy(),md.loc[ids,'patient_id'].to_numpy())
                            validation_cache[vk]=v['effective_spearman'] is not None
                        require(validation_cache[vk],'No eligible varying source-validation unit for this marker')
                    job.update(supervision_sha256=digest(asdict(plan)),split_mode=mode,
                        total_supervision=plan.total,validation_budget=512,calibration_per_unit=64 if marker else 0)
                elif job['task_type']=='J3':
                    a,b=job['sources'];p=make_j3_plans(task,a,b,job['target'],
                        fit_quotas={c:25 for c in task.receipt['classes']},
                        validation_quotas={c:10 for c in task.receipt['classes']},seed=job['seed'])
                    job['supervision_sha256']=digest({k:asdict(v) for k,v in p.items()})
                else:
                    a,b=job['sources'];p,_,_,_=make_plans(task,a,b,job['target'],PretrainConfig(**profile['j4']),job['seed'])
                    require(md.loc[list(p.fit+p.validation),'identity_level'].eq('patient').all(),'J4 source probe patient mapping unresolved')
                    job['supervision_sha256']=digest(asdict(p))
                base.append(job)
            except ProtocolError as exc:
                exclusions.append(dict(task=name,job=job,status='not_evaluable_predeclared',reason=str(exc),applies_to_all_eligible_methods=True))
        for source,target in itertools.permutations(datasets,2):
            direction_records.append(dict(task=name,source=source,target=target,
                target_identity_level=md.loc[md.dataset.eq(target),'identity_level'].iloc[0],
                j2_resource_direction=source==datasets[0] and target==datasets[-1]))
            for seed in SEEDS:
                for marker in [None]+list(task.markers):
                    kind='J1' if marker is None else 'J2'
                    resource=marker is None or (source==datasets[0] and target==datasets[-1])
                    cells=profile['learning_fit_cells'] if resource else [500]
                    specs=[('linear',500,32),('knn',500,32)]+[('mlp',n,32) for n in cells]
                    if resource:specs += [('mlp',500,w) for w in [8,128]]
                    for family,n,width in specs:
                        scopes=[]
                        if n==500 and width==32:scopes.append('primary')
                        if resource and family=='mlp' and width==32:scopes.append('learning')
                        if resource and family=='mlp' and n==500:scopes.append('capacity')
                        j=dict(task=name,task_type=kind,source=source,target=target,seed=seed,
                            family=family,width=width,fit_budget=n,analysis_scopes=scopes,candidates=candidates(family,width))
                        if marker:j['marker']=marker
                        add(j)
                for donor_n in profile['labelled_patients']:
                    add(dict(task=name,task_type='J1',source=source,target=target,seed=seed,family='mlp',
                        width=32,fit_budget=500,labelled_patient_count=donor_n,analysis_scopes=['donors'],candidates=candidates('mlp')))
        if len(datasets)==3:
            for target in datasets:
                a,b=[d for d in datasets if d!=target]
                for seed in SEEDS:
                    for family in ['linear','knn','mlp']:
                        add(dict(task=name,task_type='J3',sources=[a,b],target=target,seed=seed,family=family,
                            width=32,fit_quota_per_class=25,validation_quota_per_class=10,
                            analysis_scopes=['cohorts'],candidates=candidates(family)))
                    add(dict(task=name,task_type='J4',sources=[a,b],target=target,seed=seed,
                        family='encoder',width=32,analysis_scopes=['pretraining'],pretrain=profile['j4']))
        for method in METHODS:
            jobs=[]
            for j in base:
                if j['task_type']=='J4' and method in ['fastmnn','maxfuse']:continue
                row=dict(j,method=method);row['id']=j['task_type']+'_'+digest(row)[:20];jobs.append(row)
                counts[j['task_type']]+=1
            jobs.sort(key=lambda j:(j.get('marker',''),j['task_type'],j['id']))
            p=folder/(name+'__'+method+'.json')
            config=dict(name='formal_v0_5',mode='formal',status='frozen',formal_run_enabled=True,
                q_grid=Q_GRID,seed=SEEDS[0],jobs=jobs,input_root=str(ROOT/'data/real_inputs_v0_3_r2'),
                output=str(ROOT/'results/formal_v0_5'/name/method),
                representation_root=str(ROOT/'representations/formal_v0_5'),alignment_timeout_seconds=7200,
                minimum_free_bytes=40*2**30,max_runtime_seconds=72*3600,
                release_evidence=str(folder/(name+'__'+method+'.release.json')),
                profile_sha256=file_sha256(folder/'PROFILE.json'))
            write_json(p,config);shards.append(dict(config=str(p),sha256=file_sha256(p),jobs=len(jobs),task=name,method=method))
        print(name,len(base),'base jobs',flush=True)
    write_json(out/'EXCLUSIONS.json',exclusions)
    write_json(out/'INPUTS.json',dict(pools=pool_records,original_source_hashes=source_hashes))
    write_json(out/'DIRECTIONS.json',direction_records)
    write_json(out/'REPORT.json',dict(status='passed_with_predeclared_exclusions',job_counts=dict(counts),
        total_jobs=sum(counts.values()),shards=len(shards),excluded_base_jobs=len(exclusions),
        exclusion_reasons=dict(Counter(x['reason'] for x in exclusions)),
        basis='registered identity, source counts and source truth support only; no target performance',
        profile_sha256=file_sha256(folder/'PROFILE.json')))
    write_json(folder/'SUITE.json',dict(status='awaiting_release_checks',profile=str(folder/'PROFILE.json'),
        profile_sha256=file_sha256(folder/'PROFILE.json'),shards=shards,
        output=str(ROOT/'results/formal_v0_5'),preflight=str(out/'REPORT.json')))
    print('TOTAL',dict(counts),flush=True)


if __name__=='__main__':main()
