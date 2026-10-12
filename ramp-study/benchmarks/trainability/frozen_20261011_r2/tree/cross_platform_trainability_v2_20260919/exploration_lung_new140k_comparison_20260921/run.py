"""Nine controls paired to the completed RAMP Senosain140k J1 pilot."""
import os
import sys
sys.dont_write_bytecode = True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
          'VECLIB_MAXIMUM_THREADS','NUMBA_NUM_THREADS','LOKY_MAX_CPU_COUNT']:
    os.environ[k] = '1'
from pathlib import Path
OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
PILOT = ROOT / 'exploration_lung_new140k_ramp_c02_20260921'
COMPARISON = ROOT.parent / 'story_lung_HLA_ABC/Comparison_senosain140k_20260921'
sys.path.insert(0, str(ROOT / 'src'))
import importlib.util
spec = importlib.util.spec_from_file_location('lung_ramp_pilot', PILOT / 'run.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
import time
import traceback
import numpy as np
import pandas as pd
from trainability.adapters import fit_representation, source_files
from trainability.core import digest, file_sha256, seed_for
from trainability.inputs import LabelFreeInput, PreparedRepresentation
from trainability.heads import Candidate
from trainability.runner import run_supervised, eligible_ids
from trainability.sampling import Supervision

CLASSES = pilot.CLASSES
ARMS = pilot.ARMS
METHODS = ['maxfuse','unaligned','clipped_zscore','minmax','percentile',
           'harmony_direct','cycombine','cytonorm','fastmnn']
REUSE = {'harmony_direct':'harmony','cycombine':'cycombine','cytonorm':'cytonorm','fastmnn':'fastmnn'}


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def task_for(arm):
    t = pilot.task_for(arm)
    return replace(t, receipt=dict(t.receipt, manifest_path=str(OUT / 'MANIFEST.json')))


def ready_input(task, method):
    x = task.native.copy()
    if method != 'harmony_direct':
        for ds, cofactor in [('Senosain CyTOF',5),('Cords IMC',1),('Zheng mIF',1)]:
            ix = task.metadata.dataset.eq(ds).to_numpy()
            x[ix] = np.arcsinh(x[ix]/cofactor)
    return LabelFreeInput(x,task.pool.cell_ids,task.pool.datasets,task.pool.units,task.markers)


def prepare():
    assert not (OUT / 'MANIFEST.json').exists()
    old = json.loads((PILOT / 'MANIFEST.json').read_text())
    pilot.verify_inputs(old)
    assert json.loads((PILOT / 'CHECKS.json').read_text())['predictions_independently_rescored']
    protected = dict(old['protected_files']) | dict(old['experiment_hashes'])
    def protect(p):
        protected[str(p)] = file_sha256(p)
    for f in ['MANIFEST.json','RUN_COMPLETE.json','CHECKS.json','SUMMARY.csv']:
        protect(PILOT / f)
    for p in (PILOT / 'jobs').rglob('*'):
        if p.is_file(): protect(p)
    task = task_for('keep_c02')
    assert task.pool.sha256 == old['pool_sha256']
    input_receipt = json.loads((COMPARISON / 'input/manifest.json').read_text())
    protect(COMPARISON / 'input/manifest.json')
    for name in ['cell_ids.npy','fit_metadata.csv.gz','raw.npy','analysis_ready.npy','markers.txt']:
        p = COMPARISON / 'input' / name
        assert file_sha256(p) == input_receipt['output_sha256'][name], name
        protect(p)
    assert np.array_equal(np.load(COMPARISON / 'input/cell_ids.npy'),np.asarray(task.pool.cell_ids))
    assert np.array_equal(np.load(COMPARISON / 'input/raw.npy'),task.native)
    assert (COMPARISON / 'input/markers.txt').read_text().splitlines() == list(task.markers)
    ident = pd.read_csv(COMPARISON / 'input/fit_metadata.csv.gz',dtype=str,keep_default_na=False)
    for col, values in [('cell_id',task.pool.cell_ids),('dataset',task.pool.datasets),('unit_key',task.pool.units)]:
        assert tuple(ident[col]) == values
    assert np.array_equal(np.load(COMPARISON / 'input/analysis_ready.npy'),ready_input(task,'unaligned').x)
    reuse = {}
    for method, dirname in REUSE.items():
        folder = COMPARISON / 'methods' / dirname
        receipt = json.loads((folder / 'COMPLETE.json').read_text())
        assert receipt['input_manifest_sha256'] == file_sha256(COMPARISON / 'input/manifest.json')
        assert receipt['values_sha256'] == file_sha256(folder / 'values.npy')
        assert receipt['cell_ids_sha256'] == file_sha256(COMPARISON / 'input/cell_ids.npy')
        if method in {'cycombine','cytonorm'}:
            helper = ROOT / 'src/trainability/vendor' / (method + '_repro.R')
        else:
            helper = ROOT.parent / 'algorithm' / dirname / (dirname + ('_direct_repro.R' if method=='harmony_direct' else '_repro.R'))
        assert file_sha256(helper) == receipt['runtime']['helper_sha256']
        for p in [folder/'COMPLETE.json', folder/'R_COMPLETE.json', folder/'values.npy', helper]: protect(p)
        reuse[method] = dict(values=str(folder/'values.npy'), receipt=str(folder/'COMPLETE.json'),
            protocol='Identical shared9 inputs/order/units; same official helper and original Figure3 defaults',
            labels_used=False, output_type='embedding' if method=='fastmnn' else 'marker_values')
    for method in METHODS:
        protected.update(source_files(method))
        cfg = ROOT / f'config/formal_v0_5/lung_story__{method}.json'
        protect(cfg)
        jobs = [j for j in json.loads(cfg.read_text())['jobs'] if j['task_type']=='J1' and 'primary' in j['analysis_scopes']]
        lookup = {(j['source'],j['target'],j['seed'],j['family']):j for j in jobs}
        assert len(lookup)==54
        for j in old['jobs']:
            original = lookup[(j['source'],j['target'],j['seed'],j['family'])]
            assert original['candidates']==j['candidates']
            assert original['fit_budget']==500 and original['validation_budget']==512
    artifacts = [OUT/'run.py',OUT/'summarize.py',OUT/'DESIGN_ZH.md']
    save(OUT/'MANIFEST.json',dict(methods=['ramp']+METHODS,new_methods=METHODS,arms=ARMS,classes=CLASSES,
        jobs=old['jobs'],jobs_per_method=108,new_jobs=972,total_jobs_including_ramp=1080,
        ramp_pilot=str(PILOT),pool_sha256=task.pool.sha256,alignment_cells=84000,shared_markers=list(task.markers),
        reusable_representations=reuse,protected_files=protected,
        experiment_hashes={str(p):file_sha256(p) for p in artifacts},
        maxfuse='Original Figure3 shared=active=9; first dataset fixed hub; original pairwise config and dimension clipping',
        primary_scope='keep_c02',sensitivity_scope='exclude_c02_supervision',
        scope_selection_history='Both arms retained; primary retention follows biomarker review and the already observed RAMP pilot. Not preregistered.',
        fit_budget=500,validation_budget=512,source_only_selection=True,exploratory=True,
        formal_figure_updated=False,created_unix=time.time()))
    print('PREPARED: nine controls, 972 new jobs, 1080 including frozen RAMP.',flush=True)


def verify_inputs(m):
    for p,h in {**m['protected_files'],**m['experiment_hashes']}.items():
        assert file_sha256(p)==h,p


def representation(task, method, m):
    dest = OUT/'representations'/method
    if method in REUSE:
        info = m['reusable_representations'][method]
        values = np.load(info['values'],allow_pickle=False)
        ids = np.asarray(task.pool.cell_ids)
        if not (dest/'COMPLETE.json').exists():
            dest.mkdir(parents=True,exist_ok=False)
            # Link via provenance rather than copying or editing the original matrix.
            save(dest/'COMPLETE.json',dict(method=method,reused=True,**info,
                pool_sha256=task.pool.sha256,values_sha256=file_sha256(info['values']),
                cell_ids_sha256=digest(ids.tolist()),shape=list(values.shape)))
        else:
            proof=json.loads((dest/'COMPLETE.json').read_text())
            assert proof['values_sha256']==file_sha256(info['values']) and proof['pool_sha256']==task.pool.sha256
        return PreparedRepresentation(values,task.pool.cell_ids,task.pool.sha256,method)
    contract=dict(name='lung_story_new140k_J1',manifest_path=str(OUT/'MANIFEST.json'),
        pool_sha256=task.pool.sha256,shared_markers=list(task.markers),labels_passed=False,
        input_operators={'Senosain CyTOF':'arcsinh /5','Cords IMC':'arcsinh /1','Zheng mIF':'arcsinh /1'},
        j2_scope='not evaluated in this experiment')
    return fit_representation(ready_input(task,method),method,fitting_pool=task.pool,
        out=dest,input_contract=contract,family='trainability_v0.3:lung_story',timeout=7200)


def run_method(method, manifest_hash):
    m=json.loads((OUT/'MANIFEST.json').read_text())
    start=time.monotonic()
    task=task_for('keep_c02')
    save(OUT/'status'/f'{method}.json',dict(method=method,phase='representation',completed=0,total=108))
    rep=representation(task,method,m)
    align_seconds=time.monotonic()-start
    print(method,'representation ready',round(align_seconds,1),'s',flush=True)
    tasks={a:task_for(a) for a in ARMS}
    for i,j in enumerate(m['jobs']):
        task=tasks[j['arm']]
        plan_path=PILOT/'plans'/f"{j['id']}.json"
        saved=json.loads(plan_path.read_text())
        plan=Supervision(tuple(saved['fit']),tuple(saved['validation']))
        assert saved['pool_sha256']==task.pool.sha256==rep.fitting_pool_sha256
        assert list(eligible_ids(task,j['target'],True))==saved['target_ids']
        assert digest(dict(fit=plan.fit,validation=plan.validation,calibration=()))==j['supervision_sha256']
        dest=OUT/'jobs'/method/j['arm']/j['id'];done=dest/'EXPERIMENT_COMPLETE.json'
        if done.exists():
            proof=json.loads(done.read_text());assert proof['manifest_sha256']==manifest_hash
            for p,h in proof['files'].items():assert file_sha256(p)==h,p
        else:
            assert not dest.exists(),f'Inspect partial job: {dest}'
            result=run_supervised(task,rep,plan,sources=[j['source']],target=j['target'],marker=None,
                candidates=[Candidate(**c) for c in j['candidates']],seed=j['seed'],out=dest,
                split_mode='patient_disjoint_validation')
            assert result['score']['classes']==CLASSES
            save(done,dict(manifest_sha256=manifest_hash,method=method,
                files={str(p):file_sha256(p) for p in dest.iterdir() if p.is_file()}))
        save(OUT/'status'/f'{method}.json',dict(method=method,phase='classification',completed=i+1,total=108,
            seconds=time.monotonic()-start,representation_seconds=align_seconds))
        if (i+1)%54==0:print(method,i+1,'/ 108',flush=True)
    result=dict(method=method,jobs=108,seconds=time.monotonic()-start,representation_seconds=align_seconds)
    save(OUT/'status'/f'{method}.json',dict(phase='complete',**result))
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--workers',type=int,default=3);args=parser.parse_args()
    if not (OUT/'MANIFEST.json').exists():prepare()
    m=json.loads((OUT/'MANIFEST.json').read_text());verify_inputs(m)
    if args.prepare_only:return
    mh=file_sha256(OUT/'MANIFEST.json')
    save(OUT/'RUN.json',dict(pid=os.getpid(),manifest_sha256=mh,started_unix=time.time()))
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        fs={executor.submit(run_method,method,mh):method for method in METHODS}
        for f in as_completed(fs):
            try:result=dict(complete=True,**f.result())
            except Exception as exc:result=dict(method=fs[f],complete=False,error=repr(exc),traceback=traceback.format_exc())
            results.append(result);save(OUT/'METHOD_COMPLETIONS.json',results);print(result,flush=True)
    verify_inputs(m)
    save(OUT/'RUN_COMPLETE.json',dict(results=results,all_passed=all(r['complete'] for r in results),
        protected_files_unchanged=True,finished_unix=time.time()))
    assert all(r['complete'] for r in results),'Some methods failed; inspect logs'
    print('All nine controls finished; original files unchanged.',flush=True)


if __name__=='__main__':main()
