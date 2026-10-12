"""Paired, resumable J2 loss trial. Writes only below this experiment directory."""
from __future__ import annotations
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse, fcntl, hashlib, json, sys, time, traceback
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from rank_head import LAMBDAS, TEMPERATURE, SourceRankScore, train, predict

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT/'src'))
from trainability.core import file_sha256, digest
from trainability.targets import SourceCalibration
from trainability.scoring import freeze_predictions, regression_score


def write(path, value):
    path=Path(path); assert path.resolve().is_relative_to(HERE)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def prepare():
    manifest_path=HERE/'MANIFEST.json'
    if manifest_path.exists():
        d=json.loads(manifest_path.read_text())
        for p,h in d['frozen_scripts'].items(): assert file_sha256(p)==h, f'Changed script: {p}'
        return d
    all_jobs={}
    for method in ['ramp','percentile']:
        for p in sorted((ROOT/'config/formal_v0_5').glob(f'*__{method}.json')):
            cfg=json.loads(p.read_text())
            for j in cfg['jobs']:
                if j['task_type']=='J2' and j['family']=='mlp' and 'primary' in j['analysis_scopes']:
                    key=(j['task'],j['marker'],j['source'],j['target'],j['seed'])
                    all_jobs.setdefault(key,{})[method]=j
    jobs=[]
    for key, methods in sorted(all_jobs.items()):
        assert set(methods)=={'ramp','percentile'}
        ledgers={m:json.loads((ROOT/'results/formal_v0_5'/key[0]/m/'jobs'/j['id']/'ledger.json').read_text()) for m,j in methods.items()}
        for field in ['fit','validation','calibration','target_cell_ids','mask','pool_sha256','seed']:
            assert ledgers['ramp'][field]==ledgers['percentile'][field],(key,field)
        jobs.append(dict(key=digest(key)[:24],task=key[0],marker=key[1],source=key[2],target=key[3],seed=key[4],methods=methods))
    assert len(jobs)==977 and len({x['task'] for x in jobs})==6
    frozen={str(p):file_sha256(p) for p in [HERE/'run.py',HERE/'rank_head.py',HERE/'test_rank_head.py',HERE/'PLAN.md']}
    protected={str(p):file_sha256(p) for p in (ROOT/'src').rglob('*') if p.suffix in {'.py','.R'}}
    d=dict(created_utc=datetime.now(timezone.utc).isoformat(),purpose='exploratory matched loss experiment',
        jobs=jobs,n_pairs=len(jobs),task_markers=len({(j['task'],j['marker']) for j in jobs}),
        lambdas=list(LAMBDAS),temperature=TEMPERATURE,frozen_scripts=frozen,
        original_code_hashes=protected,selection='source validation only; ties prefer smaller lambda',
        original_results_unchanged=True)
    write(manifest_path,d);return d


@lru_cache(maxsize=1)
def load_task(name):
    path=ROOT/'data/real_inputs_v0_3_r2'/name
    receipt=json.loads((path/'manifest.json').read_text())
    for f,h in receipt['outputs'].items(): assert file_sha256(path/f)==h
    md=pd.read_csv(path/'metadata.csv.gz').set_index('cell_id',drop=False)
    native=np.load(path/'native.npy',mmap_mode='r')
    lookup={s:i for i,s in enumerate(md.index)}
    return receipt,md,native,lookup


@lru_cache(maxsize=2)
def load_representation(name, marker, method):
    receipt,md,native,lookup=load_task(name)
    canonical=receipt['canonical_markers'][marker]
    path=ROOT/'representations/formal_v0_5'/name/marker/method
    start=json.loads((path/'START.json').read_text());done=json.loads((path/'COMPLETE.json').read_text())
    assert start['masked']==[canonical] and done['pool_sha256']==receipt['pool_sha256']
    assert file_sha256(path/'values.npy')==done['values_sha256']
    assert file_sha256(path/'cell_ids.npy')==done['cell_ids_sha256']
    ids=np.load(path/'cell_ids.npy',allow_pickle=False)
    assert np.array_equal(ids,md.index.to_numpy())
    x=np.load(path/'values.npy',mmap_mode='r')
    assert x.shape==(len(md),len(receipt['markers'])-1)
    return x,dict(values_sha256=done['values_sha256'],cell_ids_sha256=done['cell_ids_sha256'],
                  masked=canonical,pool_sha256=done['pool_sha256'])


def run_method(job, method):
    out=HERE/'results'/job['key']/method
    if (out/'RESULT.json').exists():
        return json.loads((out/'RESULT.json').read_text())
    started=time.monotonic()
    old=ROOT/'results/formal_v0_5'/job['task']/method/'jobs'/job['methods'][method]['id']
    ledger=json.loads((old/'ledger.json').read_text())
    original=json.loads((old/'RESULT.json').read_text())
    selection=original['selection'];candidate=selection['candidate']
    receipt,md,native,lookup=load_task(job['task'])
    x,rep_receipt=load_representation(job['task'],job['marker'],method)
    ids={k:ledger[k] for k in ['fit','validation','calibration','target_cell_ids']}
    assert not set(ids['fit']+ids['validation']+ids['calibration']).intersection(ids['target_cell_ids'])
    assert not set(ids['fit']).intersection(ids['validation'])
    assert not set(ids['calibration']).intersection(ids['fit']+ids['validation'])
    for k in ['fit','validation','calibration']:
        assert set(md.loc[ids[k],'dataset'])=={job['source']}
    assert set(md.loc[ids['target_cell_ids'],'dataset'])=={job['target']}
    col=receipt['markers'].index(job['marker'])
    # Only source hidden-marker slices are extracted before prediction freezing.
    idx={k:np.array([lookup[i] for i in v]) for k,v in ids.items()}
    values=lambda k: np.asarray(native[idx[k],col])
    cal=SourceCalibration.fit(ids['calibration'],values('calibration'),md.loc[ids['calibration'],'unit_key'],ids['calibration'])
    units=md.loc[ids['fit'],'unit_key'].to_numpy(str)
    y=cal.transform(ids['fit'],values('fit'),units)
    fit_measured=values('fit');validation_measured=values('validation')
    vx=np.asarray(x[idx['validation']]);fx=np.asarray(x[idx['fit']])
    scaler=StandardScaler().fit(fx) if candidate['scale']=='standard' else None
    if scaler is not None: fx=scaler.transform(fx);vx=scaler.transform(vx)
    vu=md.loc[ids['validation'],'unit_key'].to_numpy(str)
    vp=md.loc[ids['validation'],'patient_id'].to_numpy(str) if ledger['identity_level']=='patient' else vu
    source_score=SourceRankScore(validation_measured,vu,vp)
    models=[];candidates=[]
    for lam in LAMBDAS:
        model,info=train(fx,y,fit_measured,units,vx,source_score,candidate,ledger['seed'],lam)
        oracle=regression_score(validation_measured,predict(model,vx),vu,vp)['effective_spearman']
        assert abs(oracle-info['source_validation_score'])<1e-12
        models.append(model);candidates.append(info)
    # Source-only selection completed before target prediction/truth access.
    winner=max(range(len(LAMBDAS)),key=lambda i:(candidates[i]['source_validation_score'],-LAMBDAS[i]))
    assert abs(candidates[0]['source_validation_score']-selection['source_validation_score'])<1e-10
    assert candidates[0]['selected_epoch']==selection['selected_epoch'],(job['key'],method,'baseline epoch')
    source_record=dict(task=job['task'],method=method,marker=job['marker'],source=job['source'],target=job['target'],
        seed=job['seed'],candidate=candidate,candidates=candidates,selected_column=winner,
        selected_lambda=LAMBDAS[winner],representation=rep_receipt,ledger_sha256=file_sha256(old/'ledger.json'),
        original_result_sha256=file_sha256(old/'RESULT.json'),target_answers_used_for_selection=False)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'SELECTION.json').exists():
        assert json.loads((out/'SELECTION.json').read_text())==source_record
    else: write(out/'SELECTION.json',source_record)
    xt=np.asarray(x[idx['target_cell_ids']]);xt=scaler.transform(xt) if scaler is not None else xt
    predictions=np.column_stack([predict(m,xt) for m in models])
    pp=out/'predictions.npz'
    if pp.exists():
        frozen=np.load(pp,allow_pickle=False)
        assert np.array_equal(frozen['cell_ids'],np.asarray(ids['target_cell_ids']))
        np.testing.assert_allclose(frozen['predictions'],predictions,rtol=1e-12,atol=1e-12)
        pred_receipt=json.loads(pp.with_suffix('.json').read_text())
        assert file_sha256(pp)==pred_receipt['sha256']
    else:
        pred_receipt=freeze_predictions(pp,ids['target_cell_ids'],predictions,
            dict(selection_sha256=file_sha256(out/'SELECTION.json'),column_lambdas=list(LAMBDAS),selected_column=winner))
    weights={f'lambda_{i}_parameter_{j}':p for i,m in enumerate(models) for j,p in enumerate(m)}
    if scaler is not None: weights.update(scaler_mean=scaler.mean_,scaler_scale=scaler.scale_)
    np.savez_compressed(out/'weights.npz',**weights)
    # Compare against old predictions (not truths) to verify exact MSE control.
    original_predictions=np.load(old/'predictions.npz',allow_pickle=False)
    assert np.array_equal(original_predictions['cell_ids'],np.asarray(ids['target_cell_ids']))
    error=float(np.max(np.abs(predictions[:,0]-original_predictions['predictions'])))
    np.testing.assert_allclose(predictions[:,0],original_predictions['predictions'],rtol=1e-8,atol=1e-8)
    # Target answers first accessed after write-once prediction and selection artifacts.
    truth=values('target_cell_ids')
    target_md=md.loc[ids['target_cell_ids']];tu=target_md.unit_key.to_numpy(str)
    tp=target_md.patient_id.to_numpy(str) if target_md.identity_level.eq('patient').all() else tu
    scores=[regression_score(truth,predictions[:,i],tu,tp) for i in range(len(LAMBDAS))]
    result=dict(**source_record,scores=scores,selected_score=scores[winner],prediction_receipt=pred_receipt,
        original_control_max_abs_error=error,elapsed_seconds=time.monotonic()-started,
        source_calibration_cells=len(ids['calibration']),fit_cells=len(ids['fit']),validation_cells=len(ids['validation']),
        total_supervision=ledger['total_supervision'])
    write(out/'RESULT.json',result);return result


def run_pair(job):
    started=time.monotonic()
    try:
        for method in ['ramp','percentile']: run_method(job,method)
        record=dict(status='complete',key=job['key'],seconds=time.monotonic()-started)
        write(HERE/'results'/job['key']/'COMPLETE.json',record)
        return record
    except Exception:
        record=dict(status='failed',key=job['key'],traceback=traceback.format_exc())
        write(HERE/'results'/job['key']/'FAILURE.json',record)
        return record


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--limit',type=int);parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    lock=(HERE/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=prepare()
    if args.prepare:
        print(json.dumps(dict(pairs=manifest['n_pairs'],task_markers=manifest['task_markers'])));return 0
    jobs=[j for j in manifest['jobs'] if not (HERE/'results'/j['key']/'COMPLETE.json').exists()]
    if args.limit: jobs=jobs[:args.limit]
    started=time.monotonic();done=0;failed=[];it=iter(jobs);pending={}
    total=manifest['n_pairs']-len([j for j in manifest['jobs'] if not (HERE/'results'/j['key']/'COMPLETE.json').exists()])
    status=dict(phase='running',pid=os.getpid(),completed_pairs=total,expected_pairs=manifest['n_pairs'],
                done_this_run=0,elapsed_seconds=0,active=[],failures=[])
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for _ in range(args.workers):
            j=next(it,None)
            if j:pending[pool.submit(run_pair,j)]=j
        while pending:
            finished,_=wait(pending,timeout=10,return_when=FIRST_COMPLETED)
            for f in finished:
                j=pending.pop(f);r=f.result();done+=1
                if r['status']!='complete':
                    failed.append(r);print(json.dumps(r),flush=True)
                else:
                    nxt=next(it,None) if not failed else None
                    if nxt:pending[pool.submit(run_pair,nxt)]=nxt
            total=sum((HERE/'results'/j['key']/'COMPLETE.json').exists() for j in manifest['jobs'])
            status=dict(phase='failed' if failed else 'running',pid=os.getpid(),completed_pairs=total,
                expected_pairs=manifest['n_pairs'],done_this_run=done,elapsed_seconds=time.monotonic()-started,
                active=[dict(task=j['task'],marker=j['marker'],source=j['source'],target=j['target'],seed=j['seed']) for j in pending.values()],
                updated_utc=datetime.now(timezone.utc).isoformat(),failures=failed)
            write(HERE/'STATUS.json',status)
            if finished and done%10==0: print(json.dumps({k:status[k] for k in ['completed_pairs','expected_pairs','elapsed_seconds']}),flush=True)
    for p,h in manifest['original_code_hashes'].items():assert file_sha256(p)==h, f'Original code changed: {p}'
    status['phase']='failed' if failed else ('complete' if total==manifest['n_pairs'] else 'partial')
    write(HERE/'STATUS.json',status)
    print(json.dumps(status),flush=True)
    return 1 if failed else 0


if __name__=='__main__': raise SystemExit(main())
