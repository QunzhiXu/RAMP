"""Fixed, label-blind donor-cap pilot; production RAMP and frozen inputs are read-only."""
import os
import sys
sys.dont_write_bytecode = True
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMBA_NUM_THREADS', 'LOKY_MAX_CPU_COUNT']:
    os.environ[key] = '1'
from pathlib import Path
OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
os.environ['MPLCONFIGDIR'] = str(OUT/'matplotlib')
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT.parent/'algorithm'))
import argparse
import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from ramp import ramp_ver2 as alg
from trainability.core import digest, file_sha256
from trainability.real_data import load_task
from trainability.inputs import PreparedRepresentation
from trainability.runner import make_plan, run_supervised
from trainability.heads import Candidate

TASKS = ['hnscc_story', 'melanoma_story', 'lung_story']
ALL_TASKS = ['bone_marrow','tnbc','melanoma','lung_cancer',
             'lung_story','melanoma_story','hnscc_story']
DIAG = ROOT/'reports/biomarker_diagnosis_20260921'
VARIANTS = ['old_alignment_same_subset', 'refit_cap20']

def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temp.replace(path)

def support(md, request):
    count, _ = alg.adaptive_pair_fit_size(md, request['datasets'])
    rows = []
    minimum = max(40, 4*alg.CURVE_MIN_BIN_CELLS)
    for ds in request['datasets']:
        indices = alg.unit_balanced_fit_indices(md, ds, count)
        for ref in request['datasets']:
            if ds == ref: continue
            sampled = md.iloc[indices].copy()
            sampled['role'] = alg.deterministic_roles(request['family'], ds, ref, indices)
            for unit in md.loc[md.dataset.eq(ds), 'unit_key'].unique():
                group = sampled[sampled.unit_key.eq(unit)]
                n = int(group.role.eq('curve_fit').sum())
                rows.append(dict(dataset=ds, reference=ref, unit_key=unit,
                    pair_fit=count, sampled_cells=len(group), curve_cells=n,
                    required_curve_cells=minimum, support_passes=n>=minimum))
    return pd.DataFrame(rows)

def primary_jobs(name):
    cfg = json.loads((ROOT/f'config/formal_v0_5/{name}__ramp.json').read_text())
    return cfg, [j for j in cfg['jobs'] if j['task_type']=='J1'
                 and 'primary' in j['analysis_scopes']]

def prepare():
    assert alg.__version__ == '2.3.0' and alg.SEED == 42
    selection = pd.read_csv(DIAG/'DONOR_CAP_SELECTIONS.csv')
    selection = selection[selection.seed.eq(20260921) & selection.cap.eq(20)]
    old_rows, old_scores, details = [], [], {}
    protected = {str(p):file_sha256(p) for p in [DIAG/'DONOR_CAP_SELECTIONS.csv',
        DIAG/'DONOR_CAP_PER_UNIT.csv',DIAG/'DONOR_CAP_CHECKS.json']}
    for name in ALL_TASKS:
        task = load_task(name)
        base = ROOT/f'representations/formal_v0_5/{name}/complete/ramp'
        req = json.loads((base/'request.json').read_text())
        ids = np.load(base/'cell_ids.npy').astype(str)
        assert np.array_equal(ids, np.asarray(task.pool.cell_ids))
        values = np.load(base/'values.npy'); baseline = np.load(base/'kmeanslow.npy')
        audit = support(task.metadata, req)
        delta_rows = []
        for unit, group in task.metadata.groupby('unit_key'):
            delta = np.abs(values[group.index]-baseline[group.index])
            delta_rows.append(dict(unit_key=unit, changed_markers=int((delta>1e-8).any(axis=0).sum()),
                exact_baseline=bool(np.array_equal(values[group.index],baseline[group.index])),
                maximum_change=float(delta.max())))
        audit = audit.merge(pd.DataFrame(delta_rows), on='unit_key', validate='many_to_one')
        audit.insert(0,'task',name); old_rows.append(audit)
        cfg, jobs = primary_jobs(name)
        for j in jobs:
            path = Path(cfg['output'])/'jobs'/j['id']/'RESULT.json'
            result = json.loads(path.read_text())
            old_scores.append(dict(task=name, variant='original_full_cohort', id=j['id'],
                source=j['source'],target=j['target'],family=j['family'],seed=j['seed'],
                **{m:result['score'][m] for m in ['macro_f1','balanced_accuracy','accuracy']}))
            protected[str(path)] = file_sha256(path)
        for p in [base/f for f in ['request.json','values.npy','kmeanslow.npy','cell_ids.npy','COMPLETE.json']]:
            protected[str(p)] = file_sha256(p)
        folder = ROOT/f'data/real_inputs_v0_3_r2/{name}'
        for p in [folder/f for f in ['manifest.json','metadata.csv.gz','native.npy']]:
            protected[str(p)] = file_sha256(p)
        protected[str(ROOT/f'config/formal_v0_5/{name}__ramp.json')] = file_sha256(ROOT/f'config/formal_v0_5/{name}__ramp.json')
        if name not in TASKS: continue
        chosen = selection[selection.task.eq(name)].copy()
        for ds in req['datasets']:
            units = np.asarray(sorted(task.metadata.loc[task.metadata.dataset.eq(ds),'unit_key'].unique()))
            rng = np.random.default_rng(alg.stable_seed('donor-cap-preflight',name,ds,20260921))
            assert set(chosen.loc[chosen.dataset.eq(ds),'unit_key']) == set(rng.permutation(units)[:20])
        index = np.flatnonzero(task.metadata.unit_key.isin(chosen.unit_key).to_numpy())
        sub = task.subset(index, 'donor_cap20_seed20260921')
        for ds, group in sub.metadata.groupby('dataset'):
            assert set(task.receipt['classes']) <= set(group.celltype), (name,ds)
        dest = OUT/'inputs'/name; dest.mkdir(parents=True, exist_ok=False)
        np.save(dest/'original_row_indices.npy', index)
        np.save(dest/'native.npy', sub.native)
        sub.metadata.to_csv(dest/'metadata.csv.gz', index=False)
        chosen.to_csv(dest/'selected_donors.csv', index=False)
        np.save(dest/'old_aligned_same_subset.npy', values[index])
        np.save(dest/'old_kmeanslow_same_subset.npy', baseline[index])
        new_support = support(sub.metadata, req)
        assert new_support.support_passes.all(), name
        new_support.to_csv(dest/'support.csv', index=False)
        # Verify identical rows to the donor-only preflight performed before any model score.
        preflight = pd.read_csv(DIAG/'DONOR_CAP_PER_UNIT.csv')
        preflight = preflight[(preflight.task==name)&(preflight.seed==20260921)&(preflight.cap==20)]
        assert preflight.selected_original_rows_sha256.nunique() == 1
        import hashlib
        assert hashlib.sha256(index.astype('<i8').tobytes()).hexdigest() == preflight.selected_original_rows_sha256.iloc[0]
        prepared_jobs=[]
        for j in jobs:
            plan, mode = make_plan(sub,j['source'],j['target'],fit_budget=j['fit_budget'],
                validation_budget=j['validation_budget'],seed=j['seed'])
            prepared_jobs.append(dict(j, original_supervision_sha256=j['supervision_sha256'],
                supervision_sha256=digest(dict(fit=plan.fit,validation=plan.validation,calibration=plan.calibration)),
                split_mode=mode))
        details[name] = dict(request=req, pool_sha256=sub.pool.sha256,
            original_cells=len(task.metadata), cells=len(sub.metadata),
            markers=list(task.markers), classes=task.receipt['classes'],
            original_donors=task.metadata.groupby('dataset').unit_key.nunique().to_dict(),
            donors=sub.metadata.groupby('dataset').unit_key.nunique().to_dict(), jobs=prepared_jobs,
            files={str(p):file_sha256(p) for p in dest.iterdir() if p.is_file()})
    pd.concat(old_rows,ignore_index=True).to_csv(OUT/'ALL_SEVEN_ORIGINAL_COVERAGE.csv',index=False)
    pd.DataFrame(old_scores).to_csv(OUT/'ORIGINAL_FULL_J1_SCORES.csv',index=False)
    code_paths=list((ROOT/'src/trainability').glob('*.py'))+list((ROOT.parent/'algorithm/ramp').glob('*.py'))+[Path(__file__)]
    manifest=dict(exploratory=True, scope='J1 classification pilot, coverage audit of all seven benchmarks',
        tasks=details, selection_seed=20260921, cap=20, model_seeds=[20260919,20260920,20260921],
        variants=VARIANTS, source_only_model_selection=True, unchanged_algorithm=True,
        unchanged_markers_and_classes=True, selected_whole_real_donors=True,
        subset_selected_before_performance=True, no_comparator_ranking_from_old_cohort=True,
        comparison='Same subset/supervision: old full-cohort RAMP representation versus RAMP refit on subset.',
        protected_files=protected, code_hashes={str(p):file_sha256(p) for p in code_paths},
        created_unix=time.time())
    save(OUT/'MANIFEST.json',manifest)
    print('PREPARED', {n:(d['cells'],len(d['jobs'])*2) for n,d in details.items()},flush=True)
    return manifest

def worker(name, info, manifest_hash):
    started=time.monotonic(); dest=OUT/'representations'/name
    def status(stage, **kw):
        save(OUT/'status'/f'{name}.json',dict(task=name,stage=stage,elapsed=time.monotonic()-started,**kw))
    full=load_task(name); index=np.load(OUT/'inputs'/name/'original_row_indices.npy')
    task=full.subset(index,'donor_cap20_seed20260921')
    assert task.pool.sha256==info['pool_sha256']
    assert np.array_equal(task.native,np.load(OUT/'inputs'/name/'native.npy'))
    req=info['request']
    if not (dest/'COMPLETE.json').exists():
        assert not dest.exists(),f'Inspect partial representation: {dest}'
        dest.mkdir(parents=True)
        save(dest/'START.json',dict(manifest_sha256=manifest_hash,request=req,started_unix=time.time()))
        status('alignment_running')
        t=time.monotonic()
        # The exact production entry point: no changed constants, filters or monkeypatches.
        result=alg.fit_transform(task.native,task.metadata[['dataset','unit_key']],list(task.markers),
            datasets=req['datasets'],family=req['family'])
        assert result.strict_rank_inversions==0
        assert np.isfinite(result.aligned).all()
        np.save(dest/'aligned.npy',result.aligned)
        np.save(dest/'kmeanslow.npy',result.baseline)
        for key in ['coupling_audit','strength_audit','evidence_audit','reference_weight_audit',
                    'profile_audit','algorithm_audit']:
            getattr(result,key).to_csv(dest/(key+'.csv'),index=False)
        oldbase=np.load(OUT/'inputs'/name/'old_kmeanslow_same_subset.npy')
        max_base_difference=float(np.abs(result.baseline-oldbase).max())
        assert max_base_difference<1e-10, 'Donor-wise baseline unexpectedly changed'
        save(dest/'COMPLETE.json',dict(manifest_sha256=manifest_hash,seconds=time.monotonic()-t,
            pool_sha256=task.pool.sha256,strict_rank_inversions=0,
            maximum_baseline_difference=max_base_difference,
            files={str(p):file_sha256(p) for p in dest.iterdir() if p.is_file()}))
        print(name,'ALIGNMENT_COMPLETE',round(time.monotonic()-t,1),flush=True)
    receipt=json.loads((dest/'COMPLETE.json').read_text())
    assert receipt['manifest_sha256']==manifest_hash
    for p,h in receipt['files'].items():assert file_sha256(p)==h,p
    arrays={'old_alignment_same_subset':np.load(OUT/'inputs'/name/'old_aligned_same_subset.npy'),
            'refit_cap20':np.load(dest/'aligned.npy')}
    reps={v:PreparedRepresentation(x,task.pool.cell_ids,task.pool.sha256,'ramp') for v,x in arrays.items()}
    records=[]
    for j in info['jobs']:
        plan,mode=make_plan(task,j['source'],j['target'],fit_budget=j['fit_budget'],
            validation_budget=j['validation_budget'],seed=j['seed'])
        assert digest(dict(fit=plan.fit,validation=plan.validation,calibration=plan.calibration))==j['supervision_sha256']
        for v in VARIANTS:
            jd=OUT/'jobs'/name/v/j['id'];done=jd/'PILOT_COMPLETE.json'
            status('heads_running',completed=len(records),total=2*len(info['jobs']),current=j['id'],variant=v)
            if done.exists():
                record=json.loads(done.read_text())
                assert record['manifest_sha256']==manifest_hash and record['result_sha256']==file_sha256(jd/'RESULT.json')
                result=json.loads((jd/'RESULT.json').read_text())
            else:
                assert not jd.exists(),f'Inspect partial head: {jd}'
                result=run_supervised(task,reps[v],plan,sources=[j['source']],target=j['target'],
                    marker=None,candidates=[Candidate(**c) for c in j['candidates']],seed=j['seed'],out=jd,split_mode=mode)
                record=dict(manifest_sha256=manifest_hash,result_sha256=file_sha256(jd/'RESULT.json'))
                save(done,record)
            row=dict(task=name,variant=v,id=j['id'],source=j['source'],target=j['target'],family=j['family'],seed=j['seed'],
                **{m:result['score'][m] for m in ['macro_f1','balanced_accuracy','accuracy']})
            records.append(row)
            pd.DataFrame(records).to_csv(OUT/f'{name}_SCORES.csv',index=False)
            if len(records)%12==0:print(name,'HEADS',len(records),'/',len(info['jobs'])*2,flush=True)
    status('complete',completed=len(records),total=2*len(info['jobs']))
    return dict(task=name,complete=True,jobs=len(records),seconds=time.monotonic()-started)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--workers',type=int,default=2);args=parser.parse_args()
    m=json.loads((OUT/'MANIFEST.json').read_text()) if (OUT/'MANIFEST.json').exists() else prepare()
    for p,h in {**m['protected_files'],**m['code_hashes']}.items():assert file_sha256(p)==h,p
    for info in m['tasks'].values():
        for p,h in info['files'].items():assert file_sha256(p)==h,p
    if args.prepare_only:return
    mh=file_sha256(OUT/'MANIFEST.json');reports=[]
    save(OUT/'RUN.json',dict(pid=os.getpid(),started_unix=time.time(),manifest_sha256=mh,workers=args.workers))
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures={executor.submit(worker,n,m['tasks'][n],mh):n for n in TASKS}
        for future in as_completed(futures):
            try:record=future.result()
            except Exception as exc:
                record=dict(task=futures[future],complete=False,error=repr(exc),traceback=traceback.format_exc())
            reports.append(record);save(OUT/'TASK_STATUS.json',reports);print(json.dumps(record),flush=True)
    for p,h in {**m['protected_files'],**m['code_hashes']}.items():assert file_sha256(p)==h,p
    save(OUT/'RUN_COMPLETE.json',dict(tasks=reports,all_passed=all(r['complete'] for r in reports),
        protected_file_hashes_unchanged=True,finished_unix=time.time()))

if __name__=='__main__':main()
