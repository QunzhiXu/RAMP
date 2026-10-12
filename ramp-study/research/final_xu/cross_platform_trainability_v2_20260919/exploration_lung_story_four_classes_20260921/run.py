"""Frozen all-cell Lung representations; four-class exploratory J1 refitting."""
import os
import sys
sys.dont_write_bytecode = True
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMBA_NUM_THREADS', 'LOKY_MAX_CPU_COUNT']:
    os.environ[key] = '1'
from pathlib import Path
OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
sys.path.insert(0, str(ROOT / 'src'))
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
import time
import traceback
import numpy as np
import pandas as pd
from trainability.core import digest, file_sha256, seed_for
from trainability.heads import Candidate
from trainability.inputs import PreparedRepresentation
from trainability.real_data import load_task
from trainability.runner import eligible_ids, make_plan, run_supervised
from trainability.sampling import Supervision, nested_draws, source_split

TASK = 'lung_story'
CLASSES = ['CD4 T', 'CD8 T', 'Myeloid / APC', 'Tumor / epithelial']
METHODS = ['ramp', 'unaligned', 'clipped_zscore', 'minmax', 'percentile',
           'harmony_direct', 'cycombine', 'cytonorm', 'fastmnn', 'maxfuse']
PROVIDER = ROOT.parent / 'story_lung_HLA_ABC/data_umap/data/CELL_METADATA.csv.gz'
MAPPING = {('Senosain CyTOF', 'CD4 T'): 'CD4 T', ('Senosain CyTOF', 'CD8 T'): 'CD8 T',
           ('Cords IMC', 'CD4'): 'CD4 T', ('Cords IMC', 'CD8'): 'CD8 T',
           ('Zheng mIF', 'Th'): 'CD4 T', ('Zheng mIF', 'Treg'): 'CD4 T',
           ('Zheng mIF', 'Tc'): 'CD8 T', ('Zheng mIF', 'DNT'): 'DNT'}


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def config(method):
    return json.loads((ROOT / f'config/formal_v0_5/{TASK}__{method}.json').read_text())


def job_key(j):
    return j['source'], j['target'], j['seed'], j['family']


def primary_jobs(method):
    return sorted([j for j in config(method)['jobs'] if j['task_type'] == 'J1'
                   and 'primary' in j['analysis_scopes']], key=job_key)


def restricted_task(full):
    provider = pd.read_csv(PROVIDER, dtype=str, keep_default_na=False)
    md = full.metadata.merge(provider[['cell_key', 'provider_cell_type']],
        left_on='cell_id', right_on='cell_key', how='left', validate='one_to_one')
    assert len(md) == 84000 and md.provider_cell_type.notna().all()
    assert np.array_equal(md.cell_id, full.metadata.cell_id)
    md['original_celltype'] = md.celltype
    md['celltype'] = [MAPPING.get((d, p), c) for d, p, c in
                     zip(md.dataset, md.provider_cell_type, md.original_celltype)]
    changed = md.celltype.ne(md.original_celltype)
    assert md.loc[changed, 'original_celltype'].eq('T cell').all()
    assert not md.celltype.eq('T cell').any(), 'Unresolved T labels'
    assert md.celltype.eq('DNT').sum() == 587
    # All non-T labels, including Cycling tumor within Tumor, remain unchanged.
    return replace(full, metadata=md, receipt=dict(full.receipt, classes=CLASSES,
        pilot=True, evaluation_scope='CD4 / CD8 / broad Myeloid-APC / Tumor-epithelial'))


def fixed_plan(full, task, job):
    original = full.metadata.set_index('cell_id')
    md = task.metadata.set_index('cell_id')
    source, target, seed = job['source'], job['target'], job['seed']
    split = source_split(eligible_ids(full, source, True), original.patient_id.to_dict(),
                         .2, seed, patient_mapping_verified=True)
    old, mode = make_plan(full, source, target, fit_budget=500, validation_budget=512, seed=seed)
    assert mode == split['mode'] == 'patient_disjoint_validation'
    assert digest(dict(fit=old.fit, validation=old.validation,
                       calibration=old.calibration)) == job['supervision_sha256']
    parents = {role: sorted(set(original.loc[list(split[role + '_pool']), 'patient_id']))
               for role in ['fit', 'validation']}
    pools = {role: tuple(i for i in split[role + '_pool'] if md.at[i, 'celltype'] in CLASSES)
             for role in ['fit', 'validation']}
    plan = Supervision(nested_draws(pools['fit'], [500], seed_for(seed, 'fit'))[500],
                       nested_draws(pools['validation'], [512], seed_for(seed, 'validation'))[512])
    plan.assert_allowed(eligible_ids(task, source, True), eligible_ids(task, target, True))
    assert not set(parents['fit']) & set(parents['validation'])
    for role in ['fit', 'validation']:
        chosen = md.loc[list(getattr(plan, role))]
        assert set(chosen.celltype) == set(CLASSES)
        assert set(chosen.patient_id) <= set(parents[role])
        assert chosen.celltype.value_counts().min() >= 10
    return plan, parents, pools


def prepare():
    protected = {}
    def protect(p): protected[str(p)] = file_sha256(p)
    full = load_task(TASK)
    task = restricted_task(full)
    protect(PROVIDER)
    for f in ['manifest.json', 'metadata.csv.gz', 'native.npy']:
        protect(ROOT / 'data/real_inputs_v0_3_r2' / TASK / f)
    for p in (ROOT / 'src/trainability').glob('*.py'): protect(p)
    for folder in ['lung_story_scope_audit_20260921', 'lung_story_label_verification_20260921',
                   'seven_benchmark_scope_consistency_20260921']:
        protect(ROOT / 'reports' / folder / 'REVIEW_ZH.md')
    base = primary_jobs('ramp')
    assert len(base) == 54
    arms = []
    for method in METHODS:
        protect(ROOT / f'config/formal_v0_5/{TASK}__{method}.json')
        jobs = primary_jobs(method)
        assert [job_key(j) for j in jobs] == [job_key(j) for j in base]
        for j, b in zip(jobs, base):
            assert j['candidates'] == b['candidates']
            assert j['supervision_sha256'] == b['supervision_sha256']
            assert j['fit_budget'] == b['fit_budget'] == 500
            assert j['validation_budget'] == b['validation_budget'] == 512
            old = Path(config(method)['output']) / 'jobs' / j['id']
            for f in ['RESULT.json', 'ledger.json', 'predictions.npz']: protect(old / f)
        rep = ROOT / f'representations/formal_v0_5/{TASK}/complete/{method}'
        receipt = json.loads((rep / 'COMPLETE.json').read_text())
        assert receipt['pool_sha256'] == full.pool.sha256 and not receipt['labels_used']
        assert receipt['values_sha256'] == file_sha256(rep / 'values.npy')
        assert receipt['cell_ids_sha256'] == file_sha256(rep / 'cell_ids.npy')
        assert np.array_equal(np.load(rep / 'cell_ids.npy'), np.asarray(full.pool.cell_ids))
        for f in ['values.npy', 'cell_ids.npy', 'COMPLETE.json']: protect(rep / f)
        arms.append(dict(method=method, values=str(rep / 'values.npy'),
                         representation_type=receipt['output_type']))
    mapping = task.metadata[['cell_id', 'dataset', 'patient_id', 'provider_cell_type',
                             'original_celltype', 'celltype']].copy()
    mapping['eligible'] = mapping.celltype.isin(CLASSES)
    mapping.to_csv(OUT / 'LABEL_MAPPING.csv.gz', index=False,
                   compression={'method': 'gzip', 'mtime': 0})
    task.metadata.groupby(['dataset', 'celltype']).agg(cells=('cell_id', 'size'),
        patients=('patient_id', 'nunique')).reset_index().to_csv(OUT / 'CELL_COUNTS.csv', index=False)
    md = task.metadata.set_index('cell_id')
    support, planned = [], []
    for j in base:
        plan, parents, pools = fixed_plan(full, task, j)
        newj = dict(j, id='LUNG4_' + digest(job_key(j))[:16], original_job_id=j['id'],
            original_supervision_sha256=j['supervision_sha256'],
            supervision_sha256=digest(dict(fit=plan.fit, validation=plan.validation, calibration=plan.calibration)))
        planned.append(newj)
        save(OUT / 'plans' / (newj['id'] + '.json'), dict(job=newj, fit=list(plan.fit),
            validation=list(plan.validation), calibration=[],
            target_ids=list(eligible_ids(task, j['target'], True)),
            parent_source_patients=parents, pool_sha256=full.pool.sha256))
        for role in ['fit', 'validation']:
            for ct in CLASSES:
                chosen = md.loc[list(getattr(plan, role))]
                support.append(dict(source=j['source'], target=j['target'], seed=j['seed'],
                    family=j['family'], role=role, celltype=ct,
                    cells=int(chosen.celltype.eq(ct).sum()),
                    patients=int(chosen.loc[chosen.celltype.eq(ct), 'patient_id'].nunique()),
                    parent_patients=len(parents[role]),
                    eligible_pool_cells=int(md.loc[list(pools[role]), 'celltype'].eq(ct).sum())))
    support = pd.DataFrame(support)
    support.to_csv(OUT / 'SUPERVISION_COUNTS.csv', index=False)
    support.groupby(['source', 'role', 'celltype']).agg(min_cells=('cells', 'min'),
        max_cells=('cells', 'max'), min_patients=('patients', 'min')).reset_index().to_csv(
        OUT / 'SUPPORT_RANGES.csv', index=False)
    hashes = {str(p): file_sha256(p) for p in (OUT / 'plans').glob('*.json')}
    for p in [Path(__file__), OUT / 'LABEL_MAPPING.csv.gz']: hashes[str(p)] = file_sha256(p)
    manifest = dict(task=TASK, classes=CLASSES, exploratory=True, prespecified=False,
        refit_alignment=False, all_cells_retained_in_alignment=True, fitting_cells=len(full.metadata),
        retained_original_patient_split=True, only_T_labels_refined=True,
        scope='Original 28 Senosain + 28 Cords + 28 Zheng patients; four-class annotation.',
        sampling='Original five-class source patient split; filter within roles; uniform 500 fit / 512 validation.',
        source_selection='Original candidates and source-patient-equal macro-F1; no target tuning.',
        aggregation='Equal models, seeds and six directions; cell and original patient weighting.',
        limitations=['Senosain Cycling tumor labels retained unchanged despite lineage uncertainty.',
                     'Myeloid/APC is a broad label with different subtype compositions.',
                     'Cords CD4 minimum training support is 10.',
                     'New four-class F1 is not directly comparable to original five-class F1.'],
        arms=arms, jobs=planned, total_jobs=540, protected_files=protected,
        experiment_hashes=hashes, created_unix=time.time())
    save(OUT / 'MANIFEST.json', manifest)
    print('PREPARED 540 jobs; fixed patient split; all four classes >=10 in every fit and validation.', flush=True)
    return manifest


def run_method(arm, jobs, manifest_hash):
    start = time.monotonic()
    full = load_task(TASK)
    task = restricted_task(full)
    rep = PreparedRepresentation(np.load(arm['values']), full.pool.cell_ids, full.pool.sha256, arm['method'])
    for n, j in enumerate(jobs):
        saved = json.loads((OUT / 'plans' / (j['id'] + '.json')).read_text())
        plan = Supervision(tuple(saved['fit']), tuple(saved['validation']))
        assert digest(dict(fit=plan.fit, validation=plan.validation, calibration=plan.calibration)) == j['supervision_sha256']
        assert list(eligible_ids(task, j['target'], True)) == saved['target_ids']
        dest = OUT / 'jobs' / arm['method'] / j['id']
        done = dest / 'EXPERIMENT_COMPLETE.json'
        if done.exists():
            receipt = json.loads(done.read_text())
            assert receipt['manifest_sha256'] == manifest_hash
            for p, h in receipt['files'].items(): assert file_sha256(p) == h
        else:
            assert not dest.exists(), f'Inspect partial job: {dest}'
            result = run_supervised(task, rep, plan, sources=[j['source']], target=j['target'], marker=None,
                candidates=[Candidate(**c) for c in j['candidates']], seed=j['seed'], out=dest,
                split_mode='patient_disjoint_validation')
            assert result['score']['classes'] == CLASSES
            assert result['fit_cells'] == 500 and result['validation_cells'] == 512
            save(done, dict(manifest_sha256=manifest_hash,
                files={str(p): file_sha256(p) for p in dest.iterdir() if p.is_file()}))
        save(OUT / 'status' / (arm['method'] + '.json'),
             dict(method=arm['method'], completed=n+1, total=len(jobs), seconds=time.monotonic()-start))
        if (n+1) % 18 == 0: print(arm['method'], n+1, '/', len(jobs), flush=True)
    return dict(method=arm['method'], complete=True, jobs=len(jobs), seconds=time.monotonic()-start)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    mp = OUT / 'MANIFEST.json'
    manifest = json.loads(mp.read_text()) if mp.exists() else prepare()
    for p, h in {**manifest['protected_files'], **manifest['experiment_hashes']}.items():
        assert file_sha256(p) == h, p
    if args.prepare_only: return
    mh = file_sha256(mp)
    save(OUT / 'RUN.json', dict(pid=os.getpid(), manifest_sha256=mh, started_unix=time.time()))
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_method, arm, manifest['jobs'], mh): arm['method'] for arm in manifest['arms']}
        for future in as_completed(futures):
            try: result = future.result()
            except Exception as exc:
                result = dict(method=futures[future], complete=False, error=repr(exc), traceback=traceback.format_exc())
            results.append(result)
            save(OUT / 'METHOD_STATUS.json', results)
            print(json.dumps(result), flush=True)
    for p, h in {**manifest['protected_files'], **manifest['experiment_hashes']}.items():
        assert file_sha256(p) == h, p
    save(OUT / 'RUN_COMPLETE.json', dict(methods=results, all_passed=all(r['complete'] for r in results),
        protected_files_unchanged=True, finished_unix=time.time()))
    if not all(r['complete'] for r in results): raise RuntimeError('Incomplete methods')


if __name__ == '__main__':
    main()
