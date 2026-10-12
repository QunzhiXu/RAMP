"""Three-class original spatial melanoma J1; preserve the original four-class source patient split."""
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
from functools import lru_cache
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

TASK = 'melanoma'
CLASSES = ['Myeloid', 'Stroma', 'Tumor']
DISPLAY_LABELS = {c: c for c in CLASSES}
METHODS = ['ramp', 'unaligned', 'clipped_zscore', 'minmax', 'percentile',
           'harmony_direct', 'cycombine', 'cytonorm', 'fastmnn', 'maxfuse']
AUDIT = ROOT / 'reports/celltype_support_audit_20260921'


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


@lru_cache(maxsize=None)
def config(method):
    return json.loads((ROOT / f'config/formal_v0_5/{TASK}__{method}.json').read_text())


def job_key(j):
    return j['source'], j['target'], j['seed'], j['family']


@lru_cache(maxsize=None)
def primary_jobs(method):
    return sorted([j for j in config(method)['jobs'] if j['task_type'] == 'J1'
                   and 'primary' in j['analysis_scopes']], key=job_key)


def restricted_task(full):
    # Keep all cells in the frozen representation; restrict supervision and evaluation only.
    return replace(full, receipt=dict(full.receipt, classes=CLASSES, pilot=True,
        evaluation_scope='Myeloid / Stroma / Tumor; original spatial three-platform benchmark, original label IDs retained'))


def fixed_plan(full, job):
    """Split the original four-class source first, then restrict eligible cells."""
    md = full.metadata.set_index('cell_id')
    source, target, seed = job['source'], job['target'], job['seed']
    original_ids = eligible_ids(full, source, True)
    split = source_split(original_ids, md.patient_id.to_dict(), .2, seed,
                         patient_mapping_verified=True)
    assert split['mode'] == job['split_mode']
    assert md.loc[list(original_ids), 'identity_level'].eq('patient').all()
    original_plan, _ = make_plan(full, source, target, fit_budget=500,
                                 validation_budget=512, seed=seed)
    assert digest(dict(fit=original_plan.fit, validation=original_plan.validation,
                       calibration=original_plan.calibration)) == job['supervision_sha256']
    parent_patients = {role: sorted(set(md.loc[list(split[role + '_pool']), 'patient_id']))
                       for role in ['fit', 'validation']}
    filtered = {role: tuple(i for i in split[role + '_pool'] if md.at[i, 'celltype'] in CLASSES)
                for role in ['fit', 'validation']}
    plan = Supervision(nested_draws(filtered['fit'], [500], seed_for(seed, 'fit'))[500],
                       nested_draws(filtered['validation'], [512], seed_for(seed, 'validation'))[512])
    task = restricted_task(full)
    plan.assert_allowed(eligible_ids(task, source, True), eligible_ids(task, target, True))
    assert split['mode'] == 'patient_disjoint_validation'
    assert not set(parent_patients['fit']) & set(parent_patients['validation'])
    for role in ['fit', 'validation']:
        ids = getattr(plan, role)
        assert set(md.loc[list(ids), 'celltype']) == set(CLASSES)
        assert set(md.loc[list(ids), 'patient_id']) <= set(parent_patients[role])
    return plan, parent_patients, filtered


def prepare():
    protected = {}
    def protect(p): protected[str(p)] = file_sha256(p)
    full = load_task(TASK)
    for f in ['manifest.json', 'metadata.csv.gz', 'native.npy']:
        protect(ROOT / 'data/real_inputs_v0_3_r2' / TASK / f)
    for p in (ROOT / 'src/trainability').glob('*.py'): protect(p)
    for f in ['REVIEW_ZH.md', 'CHECKS.json', 'CLASS_SETS.json', 'CLASS_SUPPORT.csv',
              'PER_BENCHMARK_SCORES.csv', 'PROVIDER_LABEL_COMPOSITION.csv']:
        protect(AUDIT / f)
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
    task = restricted_task(full)
    md = full.metadata.set_index('cell_id')
    support, planned = [], []
    prior_classes = json.loads((AUDIT / 'CLASS_SETS.json').read_text())[TASK]['support_ge10']
    assert prior_classes == CLASSES
    for j in base:
        plan, parent_patients, pools = fixed_plan(full, j)
        newj = dict(j, id='MEL3_' + digest(job_key(j))[:16], original_job_id=j['id'],
            original_supervision_sha256=j['supervision_sha256'],
            supervision_sha256=digest(dict(fit=plan.fit, validation=plan.validation, calibration=plan.calibration)))
        planned.append(newj)
        save(OUT / 'plans' / (newj['id'] + '.json'), dict(job=newj,
            fit=list(plan.fit), validation=list(plan.validation), calibration=[],
            target_ids=list(eligible_ids(task, j['target'], True)),
            parent_source_patients=parent_patients, pool_sha256=full.pool.sha256))
        for role in ['fit', 'validation']:
            for ct in CLASSES:
                n = int(md.loc[list(getattr(plan, role)), 'celltype'].eq(ct).sum())
                assert n >= 10, (j['source'], j['seed'], role, ct, n)
                support.append(dict(source=j['source'], seed=j['seed'], family=j['family'],
                    role=role, celltype=ct, cells=n, split_mode=j['split_mode'],
                    parent_patients=len(parent_patients[role]),
                    eligible_pool_cells=int(md.loc[list(pools[role]), 'celltype'].eq(ct).sum())))
    pd.DataFrame(support).to_csv(OUT / 'SUPERVISION_COUNTS.csv', index=False)
    counts = full.metadata.groupby(['dataset', 'celltype']).agg(cells=('cell_id', 'size'),
                                                               patients=('patient_id', 'nunique')).reset_index()
    counts['eligible'] = counts.celltype.isin(CLASSES)
    counts.to_csv(OUT / 'CELL_COUNTS.csv', index=False)
    hashes = {str(p): file_sha256(p) for p in (OUT / 'plans').glob('*.json')}
    hashes[str(Path(__file__))] = file_sha256(__file__)
    hashes[str(OUT / 'DESIGN_ZH.md')] = file_sha256(OUT / 'DESIGN_ZH.md')
    manifest = dict(task=TASK, classes=CLASSES, display_labels=DISPLAY_LABELS,
        exploratory=True, prespecified=False, refit_alignment=False,
        all_cells_retained_in_alignment=True, fitting_cells=len(full.metadata),
        retained_original_patient_split=True, label_values_and_order_unchanged=True,
        scope='Complete original spatial melanoma pool: 6 Hickey CODEX, 17 Hoch IMC, 23 Ptacek MIBI registered patients; three-class supervision/evaluation.',
        sampling='Reproduce original four-class patient split before restricting each role to three classes; original uniform sampling, 500 fit / 512 validation, no quotas.',
        source_selection='Original candidate grids and patient-equal macro-F1, no target tuning.',
        aggregation='Equal models, seeds and directions; report cell-equal and original patient-equal metrics.',
        limitation='CD8 is shared but sparsely supported under the original budget; exclusion is task restriction, not proof of biological incomparability. Stroma and Myeloid are broad compartments. Ptacek upstream preprocessing still precludes strict J2 holdout.',
        arms=arms, jobs=planned, total_jobs=540, protected_files=protected,
        experiment_hashes=hashes, created_unix=time.time())
    save(OUT / 'MANIFEST.json', manifest)
    print('PREPARED 540 jobs; original four-class source patient partitions preserved.', flush=True)
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
        dest = OUT / 'jobs' / arm['method'] / j['id']
        done = dest / 'EXPERIMENT_COMPLETE.json'
        if done.exists():
            receipt = json.loads(done.read_text())
            assert receipt['manifest_sha256'] == manifest_hash
            for p, h in receipt['files'].items(): assert file_sha256(p) == h
        else:
            assert not dest.exists(), f'Inspect partial job: {dest}'
            result = run_supervised(task, rep, plan, sources=[j['source']], target=j['target'],
                marker=None, candidates=[Candidate(**c) for c in j['candidates']], seed=j['seed'],
                out=dest, split_mode=j['split_mode'])
            assert result['score']['classes'] == CLASSES
            assert result['fit_cells'] == 500 and result['validation_cells'] == 512
            save(done, dict(manifest_sha256=manifest_hash,
                files={str(p): file_sha256(p) for p in dest.iterdir() if p.is_file()}))
        save(OUT / 'status' / (arm['method'] + '.json'),
             dict(method=arm['method'], completed=n+1, total=len(jobs), seconds=time.monotonic()-start))
        if (n+1) % 6 == 0: print(arm['method'], n+1, '/ 54', flush=True)
    return dict(method=arm['method'], complete=True, jobs=len(jobs), seconds=time.monotonic()-start)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    mp = OUT / 'MANIFEST.json'
    m = json.loads(mp.read_text()) if mp.exists() else prepare()
    for p, h in {**m['protected_files'], **m['experiment_hashes']}.items(): assert file_sha256(p) == h, p
    if args.prepare_only: return
    mh = file_sha256(mp)
    save(OUT / 'RUN.json', dict(pid=os.getpid(), manifest_sha256=mh, started_unix=time.time()))
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_method, arm, m['jobs'], mh): arm['method'] for arm in m['arms']}
        for future in as_completed(futures):
            try: result = future.result()
            except Exception as exc:
                result = dict(method=futures[future], complete=False, error=repr(exc), traceback=traceback.format_exc())
            results.append(result)
            save(OUT / 'METHOD_STATUS.json', results)
            print(json.dumps(result), flush=True)
    for p, h in {**m['protected_files'], **m['experiment_hashes']}.items(): assert file_sha256(p) == h, p
    save(OUT / 'RUN_COMPLETE.json', dict(methods=results, all_passed=all(r['complete'] for r in results),
        protected_files_unchanged=True, finished_unix=time.time()))
    if not all(r['complete'] for r in results): raise RuntimeError('Incomplete methods')


if __name__ == '__main__':
    main()
