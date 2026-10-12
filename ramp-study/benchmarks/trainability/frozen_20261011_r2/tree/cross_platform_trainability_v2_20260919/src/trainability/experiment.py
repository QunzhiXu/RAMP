"""Manifest-driven dispatcher. Formal execution requires all release evidence."""
from __future__ import annotations
import json
from pathlib import Path
import traceback
import time
from dataclasses import asdict
from .core import ROOT,require,output_path,write_json,digest,file_sha256,validate_release
from .real_data import load_task,pilot_indices
from .adapters import fit_representation
from .heads import Candidate
from .runner import make_plan,make_j3_plans,run_supervised
from .j4 import PretrainConfig,run_j4


def execute(config_path,*,pilot=False):
    config_path=Path(config_path)
    cfg=json.loads(config_path.read_text())
    require(cfg['mode']==('pilot' if pilot else 'formal'),'Requested mode differs from manifest')
    require(len({j['id'] for j in cfg['jobs']})==len(cfg['jobs']),'Duplicate job IDs')
    if not pilot:
        release=json.loads(Path(cfg['release_evidence']).read_text())
        validate_release(cfg,release)
        require(cfg.get('formal_run_enabled') is True,'Formal execution disabled')
        require(release['configuration_frozen'].get('configuration_sha256')==file_sha256(config_path),
                'Release evidence does not bind this exact run manifest')
    out=output_path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    manifest_hash=file_sha256(config_path)
    code_hashes={str(p.relative_to(ROOT)):file_sha256(p)
                 for p in (ROOT/'src').rglob('*') if p.suffix in {'.py','.R'}}
    code_sha256=digest(code_hashes)
    if not pilot:
        require(release['configuration_frozen'].get('code_sha256')==code_sha256,
                'Formal source code differs from released version')
    if (out/'RUN.json').exists():
        old=json.loads((out/'RUN.json').read_text())
        require(old['configuration_sha256']==manifest_hash,'Cannot resume with changed config')
        require(old.get('code_sha256')==code_sha256,'Cannot resume training outputs with changed/unrecorded code')
    else:write_json(out/'RUN.json',dict(configuration=cfg,configuration_sha256=manifest_hash,
        code_sha256=code_sha256,code_hashes=code_hashes,formal=not pilot))
    tasks={};reps={};results=[]
    started=time.monotonic()
    import shutil
    for job in cfg['jobs']:
        if (time.monotonic()-started>cfg.get('max_runtime_seconds',float('inf'))
                or shutil.disk_usage(out).free<cfg.get('minimum_free_bytes',0)):
            write_json(out/'PAUSED.json',dict(reason='runtime_or_disk_budget',completed_jobs=len(results)),replace=True)
            break
        name=job['task']
        if name not in tasks:
            tasks.clear();reps.clear()  # Bound resident data to the current task.
            task=load_task(name,cfg.get('input_root'))
            if pilot:task=task.subset(pilot_indices(task,cfg['pilot_cells_per_dataset'],cfg['seed']),cfg['name'])
            tasks[name]=task
        task=tasks[name];method=job['method'];marker=job.get('marker')
        key=(name,method,marker)
        dest=out/'jobs'/job['id']
        print('START',job['id'],flush=True)
        if (dest/'JOB_COMPLETE.json').exists():
            r=json.loads((dest/'JOB_COMPLETE.json').read_text())
            require(r['job_sha256']==digest(job),'Completed job definition changed')
            for p,h in r['artifact_sha256'].items():require(file_sha256(dest/p)==h,'Completed job artifact changed')
            results.append(dict(id=job['id'],status='cached'));continue
        try:
            if key not in reps:
                reps.clear()  # Jobs are grouped by representation; old matrices stay on disk.
                canonical=task.receipt['canonical_markers'][marker] if marker else None
                reps[key]=fit_representation(task.input(method,canonical),method,fitting_pool=task.pool,
                    out=output_path(cfg.get('representation_root',out/'representations'))/name/(marker or 'complete')/method,
                    input_contract=task.receipt,family=f'trainability_v0.3:{name}',timeout=cfg.get('alignment_timeout_seconds',7200))
            rep=reps[key]
            if job['task_type']=='alignment':dest.mkdir(parents=True,exist_ok=False)
            elif job['task_type'] in {'J1','J2'}:
                plan,mode=make_plan(task,job['source'],job['target'],fit_budget=job['fit_budget'],
                    validation_budget=job['validation_budget'],seed=job['seed'],marker=marker,
                    calibration_per_unit=job.get('calibration_per_unit',0),
                    labelled_patient_count=job.get('labelled_patient_count'))
                if 'supervision_sha256' in job:
                    require(digest(asdict(plan))==job['supervision_sha256'],'Supervision differs from frozen plan')
                run_supervised(task,rep,plan,sources=[job['source']],target=job['target'],marker=marker,
                    candidates=[Candidate(**c) for c in job['candidates']],seed=job['seed'],out=dest,split_mode=mode)
            elif job['task_type']=='J3':
                a,b=job['sources'];classes=task.receipt['classes']
                plans=make_j3_plans(task,a,b,job['target'],fit_quotas={c:job['fit_quota_per_class'] for c in classes},
                    validation_quotas={c:job['validation_quota_per_class'] for c in classes},seed=job['seed'])
                if 'supervision_sha256' in job:
                    require(digest({k:asdict(v) for k,v in plans.items()})==job['supervision_sha256'],'J3 supervision differs from frozen plan')
                dest.mkdir(parents=True,exist_ok=False)
                for condition,plan in plans.items():
                    src={'A-only':[a],'B-only':[b],'A+B':[a,b]}[condition]
                    run_supervised(task,rep,plan,sources=src,target=job['target'],marker=None,
                        candidates=[Candidate(**c) for c in job['candidates']],seed=job['seed'],
                        out=dest/condition,split_mode='equal_class_budget_source_patient_split')
            elif job['task_type']=='J4':
                a,b=job['sources']
                run_j4(task,rep,source_a=a,source_b=b,target=job['target'],cfg=PretrainConfig(**job['pretrain']),seed=job['seed'],out=dest,
                    expected_supervision_sha256=job.get('supervision_sha256'))
            else:raise ValueError('Unknown task type')
            hashes={str(p.relative_to(dest)):file_sha256(p) for p in dest.rglob('*') if p.is_file()}
            write_json(dest/'JOB_COMPLETE.json',dict(job_sha256=digest(job),artifact_sha256=hashes))
            results.append(dict(id=job['id'],status='passed'))
        except Exception as e:
            # Preserve failures as records, never remove them from a summary average.
            err=out/'failures'/(job['id']+'.json')
            write_json(err,dict(job=job,error=repr(e),traceback=traceback.format_exc()),replace=True)
            results.append(dict(id=job['id'],status='failed',error=repr(e)))
        print(results[-1],flush=True)
        write_json(out/'SUMMARY.json',dict(results=results,formal=not pilot,
            all_passed=len(results)==len(cfg['jobs']) and all(r['status'] in {'passed','cached'} for r in results)),replace=True)
    return results
