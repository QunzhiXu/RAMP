"""Exploratory CD4/CD8/B annotation, with immutable all-cell representations."""
import os
import sys
sys.dont_write_bytecode = True
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
          'VECLIB_MAXIMUM_THREADS', 'NUMBA_NUM_THREADS', 'LOKY_MAX_CPU_COUNT'):
    os.environ[k] = '1'
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
from trainability.core import digest, file_sha256
from trainability.heads import Candidate
from trainability.inputs import PreparedRepresentation
from trainability.real_data import load_task
from trainability.runner import make_plan, run_supervised, eligible_ids

TASK = 'hnscc_story'
CLASSES = ['B', 'CD4 T', 'CD8 T']
METHODS = ['ramp', 'unaligned', 'clipped_zscore', 'minmax', 'percentile',
           'harmony_direct', 'cycombine', 'cytonorm', 'fastmnn', 'maxfuse']
CAP = ROOT / 'exploration_donor_cap20_20260921'


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def job_key(j):
    return (j['source'], j['target'], j['seed'], j['family'])


def config(method):
    return json.loads((ROOT / f'config/formal_v0_5/{TASK}__{method}.json').read_text())


def primary_jobs(method):
    return sorted([j for j in config(method)['jobs'] if j['task_type'] == 'J1'
                   and 'primary' in j['analysis_scopes']], key=job_key)


def restricted_task(cohort):
    full = load_task(TASK)
    if cohort == 'cap20':
        ix = np.load(CAP / 'inputs' / TASK / 'original_row_indices.npy')
        full = full.subset(ix, 'donor_cap20_seed20260921')
    else:
        assert cohort == 'full'
    # Restrict only supervised eligibility; all cells outside these three classes remain in U.
    return replace(full, receipt=dict(full.receipt, classes=CLASSES, pilot=True,
        evaluation_scope='B versus CD4 versus CD8 within annotated B/CD4/CD8 cells; Plasma excluded'))


def prepare():
    protected = {}
    def protect(p):
        protected[str(p)] = file_sha256(p)
    for p in (ROOT / 'data/real_inputs_v0_3_r2' / TASK).iterdir():
        if p.is_file(): protect(p)
    for p in (ROOT / 'src/trainability').glob('*.py'): protect(p)
    previous = ROOT / 'exploration_hnscc_cd4_cd8_20260921'
    for filename in ['run.py', 'summarize.py', 'MANIFEST.json', 'RUN_COMPLETE.json',
                     'FULL_COHORT_SUMMARY.csv', 'SCORES_PER_JOB.csv']:
        protect(previous / filename)
    base = primary_jobs('ramp')
    assert len(base) == 18
    arms = []
    full = load_task(TASK)
    for method in METHODS:
        p = ROOT / f'config/formal_v0_5/{TASK}__{method}.json'
        protect(p)
        jobs = primary_jobs(method)
        assert [job_key(j) for j in jobs] == [job_key(j) for j in base]
        for j, b in zip(jobs, base):
            assert j['candidates'] == b['candidates']
            assert j['fit_budget'] == b['fit_budget'] == 500
            assert j['validation_budget'] == b['validation_budget'] == 512
        rp = ROOT / f'representations/formal_v0_5/{TASK}/complete/{method}'
        receipt = json.loads((rp / 'COMPLETE.json').read_text())
        assert receipt['pool_sha256'] == full.pool.sha256
        assert receipt['labels_used'] is False
        assert receipt['values_sha256'] == file_sha256(rp / 'values.npy')
        assert receipt['cell_ids_sha256'] == file_sha256(rp / 'cell_ids.npy')
        assert np.array_equal(np.load(rp / 'cell_ids.npy'), np.asarray(full.pool.cell_ids))
        for f in ['values.npy', 'cell_ids.npy', 'COMPLETE.json']: protect(rp / f)
        arms.append(dict(id='full__' + method, cohort='full', method=method,
            values=str(rp / 'values.npy'), representation_type=receipt['output_type'],
            ranking_group='full_cohort_all_methods', original_jobs=jobs))
    for label, p in [('original', CAP / 'inputs' / TASK / 'old_aligned_same_subset.npy'),
                     ('refit', CAP / 'representations' / TASK / 'aligned.npy')]:
        protect(p)
        arms.append(dict(id='cap20__ramp_' + label, cohort='cap20', method='ramp',
            values=str(p), representation_type='marker_values',
            ranking_group='cap20_ramp_only_paired_diagnostic', original_jobs=base))
    for p in [CAP / 'inputs' / TASK / 'original_row_indices.npy',
              CAP / 'inputs' / TASK / 'metadata.csv.gz', CAP / 'MANIFEST.json',
              CAP / 'representations' / TASK / 'COMPLETE.json']:
        protect(p)
    plans, counts, supports = {}, [], []
    for cohort in ['full', 'cap20']:
        task = restricted_task(cohort)
        md = task.metadata.set_index('cell_id')
        plans[cohort] = []
        for (ds, ct), g in task.metadata.groupby(['dataset', 'celltype']):
            counts.append(dict(cohort=cohort, dataset=ds, celltype=ct, cells=len(g),
                patients=g.patient_id.nunique(), eligible=ct in CLASSES))
        for j in base:
            plan, mode = make_plan(task, j['source'], j['target'],
                fit_budget=500, validation_budget=512, seed=j['seed'])
            assert mode == 'patient_disjoint_validation'
            assert not set(md.loc[list(plan.fit), 'patient_id']).intersection(
                md.loc[list(plan.validation), 'patient_id'])
            for role, ids in [('fit', plan.fit), ('validation', plan.validation)]:
                assert set(md.loc[list(ids), 'celltype']) == set(CLASSES)
                for ct in CLASSES:
                    supports.append(dict(cohort=cohort, source=j['source'], target=j['target'],
                        seed=j['seed'], family=j['family'], role=role, celltype=ct,
                        cells=int(md.loc[list(ids), 'celltype'].eq(ct).sum())))
            tj = dict(j, id='TCB_' + digest(job_key(j))[:16],
                original_job_id=j['id'], original_supervision_sha256=j['supervision_sha256'],
                supervision_sha256=digest(dict(fit=plan.fit, validation=plan.validation,
                                               calibration=plan.calibration)), split_mode=mode)
            plans[cohort].append(tj)
            save(OUT / 'plans' / cohort / (tj['id'] + '.json'), dict(job=tj,
                fit=list(plan.fit), validation=list(plan.validation), calibration=[],
                target_ids=list(eligible_ids(task, j['target'], True)), pool_sha256=task.pool.sha256))
    pd.DataFrame(counts).to_csv(OUT / 'CELL_COUNTS.csv', index=False)
    pd.DataFrame(supports).to_csv(OUT / 'SUPERVISION_COUNTS.csv', index=False)
    hashes = {str(p): file_sha256(p) for p in (OUT / 'plans').rglob('*.json')}
    hashes[str(Path(__file__))] = file_sha256(__file__)
    manifest = dict(task=TASK, classes=CLASSES, exploratory=True, prespecified=False,
        rationale='Shared CD3/CD4/CD8/CD20, mapped labels, sufficient cells; exploratory extension after two-class audit.',
        all_cells_retained_in_alignment=True, refit_alignment=False,
        sampling='Original patient split and uniform random draws within the three eligible classes.',
        source_selection='Patient-equal macro-F1; original candidate grids and early stopping.',
        target_labels_used_for='Fixed evaluation eligibility and scoring only, not head fitting or tuning.',
        metrics=['overall_cell_accuracy', 'overall_cell_macro_f1', 'patient_equal_accuracy',
                 'patient_equal_macro_f1', 'overall_cell_balanced_accuracy'],
        aggregation='Equal directions, model families and seeds; no biological independence claim for seeds.',
        primary_comparison='Original full-U representations for all ten methods.',
        secondary_comparison='Only RAMP original/refit representations on the fixed cap20 donor subset.',
        arms=arms, jobs=plans, total_jobs=216, protected_files=protected,
        experiment_hashes=hashes, created_unix=time.time())
    save(OUT / 'MANIFEST.json', manifest)
    print('PREPARED', len(arms), 'arms /', manifest['total_jobs'], 'jobs', flush=True)
    return manifest


def run_arm(arm, manifest_hash, jobs):
    started = time.monotonic()
    task = restricted_task(arm['cohort'])
    if arm['cohort'] == 'cap20':
        reference_md = pd.read_csv(CAP / 'inputs' / TASK / 'metadata.csv.gz', dtype=str)
        assert task.metadata.cell_id.tolist() == reference_md.cell_id.tolist()
    x = np.load(arm['values'], allow_pickle=False)
    rep = PreparedRepresentation(x, task.pool.cell_ids, task.pool.sha256, arm['method'])
    for n, j in enumerate(jobs):
        plan, mode = make_plan(task, j['source'], j['target'], fit_budget=500,
                              validation_budget=512, seed=j['seed'])
        assert digest(dict(fit=plan.fit, validation=plan.validation,
                           calibration=plan.calibration)) == j['supervision_sha256']
        out = OUT / 'jobs' / arm['id'] / j['id']
        done = out / 'EXPERIMENT_COMPLETE.json'
        if done.exists():
            receipt = json.loads(done.read_text())
            assert receipt['manifest_sha256'] == manifest_hash
            for p, h in receipt['files'].items(): assert file_sha256(p) == h
        else:
            assert not out.exists(), f'Inspect partial job before resuming: {out}'
            result = run_supervised(task, rep, plan, sources=[j['source']], target=j['target'],
                marker=None, candidates=[Candidate(**c) for c in j['candidates']],
                seed=j['seed'], out=out, split_mode=mode)
            assert result['score']['classes'] == CLASSES
            assert result['fit_cells'] == 500 and result['validation_cells'] == 512
            save(done, dict(manifest_sha256=manifest_hash,
                files={str(p): file_sha256(p) for p in out.iterdir() if p.is_file()}))
        save(OUT / 'status' / (arm['id'] + '.json'), dict(arm=arm['id'],
            complete=n+1, total=len(jobs), elapsed_seconds=time.monotonic()-started))
        if (n+1) % 6 == 0: print(arm['id'], n+1, '/', len(jobs), flush=True)
    return dict(arm=arm['id'], complete=True, jobs=len(jobs), seconds=time.monotonic()-started)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    mp = OUT / 'MANIFEST.json'
    m = json.loads(mp.read_text()) if mp.exists() else prepare()
    for p, h in {**m['protected_files'], **m['experiment_hashes']}.items():
        assert file_sha256(p) == h, p
    if args.prepare_only: return
    mh = file_sha256(mp)
    save(OUT / 'RUN.json', dict(pid=os.getpid(), started_unix=time.time(), manifest_sha256=mh))
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_arm, a, mh, m['jobs'][a['cohort']]): a for a in m['arms']}
        for f in as_completed(futures):
            try: result = f.result()
            except Exception as exc:
                result = dict(arm=futures[f]['id'], complete=False, error=repr(exc),
                              traceback=traceback.format_exc())
            results.append(result)
            save(OUT / 'ARM_STATUS.json', results)
            print(json.dumps(result), flush=True)
    for p, h in {**m['protected_files'], **m['experiment_hashes']}.items():
        assert file_sha256(p) == h, p
    save(OUT / 'RUN_COMPLETE.json', dict(arms=results,
        all_passed=all(r['complete'] for r in results),
        protected_files_unchanged=True, finished_unix=time.time()))
    if not all(r['complete'] for r in results): raise RuntimeError('Some arms failed')


if __name__ == '__main__':
    main()
