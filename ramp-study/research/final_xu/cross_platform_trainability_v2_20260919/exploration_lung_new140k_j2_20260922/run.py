"""Frozen Lung J2 input-version update; original data/algorithms stay read-only."""
import os
import sys
sys.dont_write_bytecode = True
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMBA_NUM_THREADS', 'LOKY_MAX_CPU_COUNT']:
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
from pathlib import Path
OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
FX = ROOT.parent
RELEASE = FX / 'story_lung_HLA_ABC/releases/senosain140k_20260921'
ANN = FX / 'raw_data/lung_cancer/Senosain_2025_CyTOF_and_omics/annotation'
sys.path.insert(0, str(ROOT / 'src'))
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import fcntl
import json
import platform
import shutil
import time
import traceback
import numpy as np
import pandas as pd
from trainability.core import digest, file_sha256
from trainability.real_data import RealTask, load_task
from trainability.inputs import FittingPool, simple_representation
from trainability.adapters import fit_representation, source_files, SIMPLE
from trainability.heads import Candidate
from trainability.runner import eligible_ids, make_plan, run_supervised
from trainability.sampling import Supervision

METHODS = ['ramp', 'unaligned', 'harmony_direct', 'cycombine', 'cytonorm',
           'fastmnn', 'maxfuse', 'clipped_zscore', 'minmax', 'percentile']
SEEDS = [20260919, 20260920, 20260921]
OLD_MEANS = ROOT / 'results/formal_v0_5/summary_1790002465230065000/primary_means.json'


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def task_for():
    contract = json.loads((OUT / 'INPUT_CONTRACT.json').read_text())
    md = pd.read_csv(RELEASE / 'inputs/CELL_METADATA.csv.gz', dtype=str, keep_default_na=False)
    md = md.rename(columns={'cell_key': 'cell_id'})
    md['patient_id'] = md.dataset + '::' + md.patient_id
    md['identity_level'] = 'patient'
    md['celltype'] = md.broad_type  # Unused by J2; no classification eligibility filter.
    pool = FittingPool('lung_story:senosain140k_release_20260921:all84000',
                      tuple(md.cell_id), tuple(md.dataset), tuple(md.unit_key))
    native = np.load(RELEASE / 'inputs/RAW_SHARED9.npy', allow_pickle=False)
    receipt = dict(contract, manifest_path=str(OUT / 'INPUT_CONTRACT.json'),
                   pool_id=pool.pool_id, pool_sha256=pool.sha256, classes=[], pilot=False)
    return RealTask('lung_story', native, md, tuple(contract['markers']),
                    contract['contracts'], receipt, pool)


def original_jobs(method):
    d = json.loads((ROOT / f'config/formal_v0_5/lung_story__{method}.json').read_text())
    return [j for j in d['jobs'] if j['task_type'] == 'J2' and 'primary' in j['analysis_scopes']]


def prepare():
    assert not (OUT / 'MANIFEST.json').exists(), 'Already frozen'
    protected = {}
    def protect(p):
        p = Path(p)
        protected[str(p)] = file_sha256(p)
    release_manifest = json.loads((RELEASE / 'RELEASE_MANIFEST.json').read_text())
    for rel in ['inputs/CELL_METADATA.csv.gz', 'inputs/RAW_SHARED9.npy', 'inputs/DESIGN.json',
                'scripts/align.py']:
        p = RELEASE / rel
        assert file_sha256(p) == release_manifest[rel], rel
        protect(p)
    protect(RELEASE / 'RELEASE_MANIFEST.json')
    provenance = json.loads((RELEASE / 'provenance/ALIGNMENT_INPUTS.json').read_text())
    names = ['senosain_140k_raw35.npy', 'senosain_140k_markers35.csv',
             'senosain_140k_cell_metadata.csv.gz',
             'rough_annotation/senosain_140k_final_cell_annotations.csv.gz']
    for name in names:
        p = ANN / name
        assert file_sha256(p) == provenance[str(p)], str(p)
        protect(p)
    for p in [ANN / '01_sample_140k.py', ANN / 'senosain_140k_manifest.json',
              ANN / 'senosain_140k_provider_fcs77.npy', RELEASE / 'provenance/ALIGNMENT_INPUTS.json',
              ROOT / 'data/real_inputs_v0_3_r2/lung_story/manifest.json', OLD_MEANS,
              ROOT / 'CURRENT_J1_RESULTS.json', ROOT / 'Figure/Panel C/PROVENANCE.json']:
        protect(p)
    current_j1 = json.loads((ROOT / 'CURRENT_J1_RESULTS.json').read_text())
    j1_means = Path(current_j1['directory']) / 'primary_means.json'
    assert file_sha256(j1_means) == current_j1['primary_means_sha256']
    protect(j1_means)
    old = load_task('lung_story')
    for name in ['native.npy', 'metadata.csv.gz']:
        protect(ROOT / 'data/real_inputs_v0_3_r2/lung_story' / name)
    contracts = {ds: dict(c) for ds, c in old.contracts.items()}
    for c in contracts.values():
        c.update(reason='Explicit column selection; source values verified against frozen provider-layer inputs',
                 caveat='Simulated masking AFTER frozen provider processing, QC and cell selection; not instrument-raw or prospective panel absence')
    contract = dict(name='lung_story', version='senosain140k_J2_20260922', markers=list(old.markers),
        canonical_markers=old.receipt['canonical_markers'], contracts=contracts,
        source_files={k:v for k,v in protected.items() if '/src/' not in k},
        cell_selection='All frozen 84000 cells; no J1 label eligibility; C02 retained',
        input_contract_scope='Frozen source matrix onward; provider normalization/denoising and QC stay fixed')
    save(OUT / 'INPUT_CONTRACT.json', contract)
    task = task_for()
    md = task.metadata
    assert task.native.shape == (84000, 9) and np.isfinite(task.native).all()
    assert md.cell_id.is_unique and md.groupby('dataset').size().eq(28000).all()
    assert md.groupby('dataset').patient_id.nunique().eq(28).all()
    assert md.groupby('unit_key').size().eq(1000).all()
    assert task.pool.sha256 == 'bf038c92760745bbcb14f6be28b25b15a8fbf9f1ef0d1c613fd7f48950295f71'
    assert tuple(json.loads((RELEASE / 'inputs/DESIGN.json').read_text())['markers']) == task.markers
    sen = md.dataset.eq('Senosain CyTOF').to_numpy()
    assert ((md.parent_cluster == 'C02') & sen).sum() == 3171
    assert md.loc[sen, 'qc_recommendation'].eq('keep').all()
    other = ~sen
    lookup = old.metadata.set_index('cell_id')
    order = pd.Index(old.pool.cell_ids).get_indexer(md.loc[other, 'cell_id'])
    assert (order >= 0).all()
    assert np.array_equal(task.native[other], old.native[order])
    for col in ['dataset', 'unit_key', 'patient_id']:
        assert list(md.loc[other, col]) == list(lookup.loc[md.loc[other, 'cell_id'], col])
    raw35 = np.load(ANN / 'senosain_140k_raw35.npy', mmap_mode='r')
    annmd = pd.read_csv(ANN / 'senosain_140k_cell_metadata.csv.gz', dtype=str, keep_default_na=False)
    rr = md.loc[sen, 'source_row_id'].to_numpy(int)
    assert list(md.loc[sen, 'source_cell_id']) == list(annmd.iloc[rr].source_cell_id)
    names = pd.read_csv(ANN / 'senosain_140k_markers35.csv')
    cc = [list(names.marker).index('CYTOKERATIN' if m == 'EPI' else m) for m in task.markers]
    assert np.array_equal(task.native[sen], raw35[rr][:, cc])
    # Direct provider FCS77 channel extraction is documented and checked below.
    fcs77 = np.load(ANN / 'senosain_140k_provider_fcs77.npy', mmap_mode='r')
    index_col = 'fcs_parameter_one_based'
    fc = names[index_col].to_numpy(int) - 1
    assert np.array_equal(raw35, fcs77[:, fc])
    mask_checks = []
    for marker in task.markers:
        canonical = task.receipt['canonical_markers'][marker]
        col = task.markers.index(marker)
        corrupted = task.native.copy()
        corrupted[:, col] = np.nan
        modified = replace(task, native=corrupted)
        for method in METHODS:
            a, b = task.input(method, canonical), modified.input(method, canonical)
            assert np.array_equal(a.x, b.x) and a.markers == b.markers
            assert a.x.shape == (84000, 8) and marker not in a.markers
            assert a.masked_markers == (canonical,)
            task.pool.validate(a)
            if method in SIMPLE:
                assert np.array_equal(simple_representation(a, method, fitting_pool=task.pool),
                                      simple_representation(b, method, fitting_pool=task.pool))
            mask_checks.append(dict(marker=marker, method=method, exact_masked_input_invariance=True,
                simple_representation_invariance=method in SIMPLE,
                external_fit_invariance='boundary bytes identical; adapters receive only masked input' if method not in SIMPLE else None))
    plans = {}
    datasets = list(dict.fromkeys(task.pool.datasets))
    for source in datasets:
        target = next(d for d in datasets if d != source)
        for seed in SEEDS:
            plan, mode = make_plan(task, source, target, fit_budget=500, validation_budget=512,
                                   seed=seed, marker=task.markers[0], calibration_per_unit=64)
            assert (len(plan.fit), len(plan.validation), len(plan.calibration)) == (500, 512, 1792)
            mm = md.set_index('cell_id')
            assert not set(mm.loc[list(plan.fit), 'patient_id']) & set(mm.loc[list(plan.validation), 'patient_id'])
            key = digest([source, seed])[:20]
            record = dict(source=source, seed=seed, mode=mode, **asdict(plan), sha256=digest(asdict(plan)))
            if source != 'Senosain CyTOF':
                oldplan, oldmode = make_plan(old, source, target, fit_budget=500, validation_budget=512,
                                            seed=seed, marker=task.markers[0], calibration_per_unit=64)
                assert plan == oldplan and mode == oldmode
            save(OUT / 'plans' / f'{key}.json', record)
            plans[(source, seed)] = (key, record)
    jobs, signatures = [], []
    for method in METHODS:
        protect(ROOT / f'config/formal_v0_5/lung_story__{method}.json')
        template = original_jobs(method)
        assert len(template) == 486
        sig = []
        for j in template:
            assert j['fit_budget'] == 500 and j['validation_budget'] == 512 and j['calibration_per_unit'] == 64
            key, plan = plans[(j['source'], j['seed'])]
            sig.append({k:v for k,v in j.items() if k not in ['id', 'method', 'supervision_sha256']})
            ident = 'NEWLUNGJ2_' + digest([method, j['marker'], j['source'], j['target'], j['seed'], j['family']])[:20]
            jobs.append(dict(j, id=ident, original_job_id=j['id'], plan_file=f'plans/{key}.json',
                             supervision_sha256=plan['sha256']))
        signatures.append(sorted(sig, key=lambda x: (x['marker'], x['source'], x['target'], x['seed'], x['family'])))
        protected.update(source_files(method))
    assert all(s == signatures[0] for s in signatures)
    assert len(jobs) == len({j['id'] for j in jobs}) == 4860
    for p in (ROOT / 'src/trainability').rglob('*'):
        if p.suffix in {'.py', '.R'}: protect(p)
    save(OUT / 'PREFLIGHT.json', dict(status='passed', cells=84000, patients_per_platform=28,
        source_values_equal_raw35=True, raw35_equal_provider_fcs77=True,
        Cords_Zheng_values_and_ids_unchanged=True, Cords_Zheng_source_plans_unchanged=True,
        C02_retained=3171, markers=list(task.markers), mask_checks=mask_checks,
        same_candidate_grids_all_methods=True, labels_not_used_for_J2=True,
        fit_validation_patients_disjoint=True, calibration_cells=1792,
        caveat=contract['input_contract_scope']))
    artifacts = list((OUT / 'plans').glob('*.json')) + [OUT / n for n in
        ['run.py', 'summarize.py', 'DESIGN_ZH.md', 'INPUT_CONTRACT.json', 'PREFLIGHT.json']]
    save(OUT / 'MANIFEST.json', dict(name=OUT.name, task='lung_story', task_type='J2',
        pool_sha256=task.pool.sha256, methods=METHODS, markers=list(task.markers), seeds=SEEDS,
        jobs=jobs, total_jobs=4860, total_representations=90, family='trainability_v0.3:lung_story',
        protected_files=protected, experiment_hashes={str(p):file_sha256(p) for p in artifacts},
        original_primary_means=str(OLD_MEANS), source_contract=str(OUT / 'INPUT_CONTRACT.json'),
        accepted_j1_primary_means=str(j1_means),
        model_scope='Original primary linear/knn/mlp; no resource curves rerun',
        environment=dict(python=sys.version, platform=platform.platform()), created_unix=time.time()))
    print('PREFLIGHT PASS: 4860 paired J2 jobs, 90 masked representations.', flush=True)


def verify(manifest):
    for p, h in {**manifest['protected_files'], **manifest['experiment_hashes']}.items():
        assert file_sha256(p) == h, f'Frozen input/code changed: {p}'


def verify_job(dest, job, mh):
    proof = json.loads((dest / 'JOB_COMPLETE.json').read_text())
    assert proof['manifest_sha256'] == mh and proof['job_sha256'] == digest(job)
    for name, h in proof['artifact_sha256'].items():
        assert file_sha256(dest / name) == h, str(dest / name)


def audit_adapter_inputs(method, marker, folder, expected):
    """Inspect persisted boundary inputs; target must not be recoverable as a feature."""
    if method == 'ramp':
        req = json.loads((folder / 'request.json').read_text())
        assert tuple(req['markers']) == expected.markers
        assert np.array_equal(np.load(folder / 'input.npy'), expected.x)
    elif method == 'maxfuse':
        with np.load(folder / 'input.npz', allow_pickle=False) as z:
            for key in z.files:
                if key.endswith('_feature_names'):
                    assert tuple(z[key]) == expected.markers and marker not in z[key]
                elif key.endswith(('_active', '_shared')):
                    assert z[key].shape[1] == 8 and np.isfinite(z[key]).all()
    elif method not in SIMPLE:
        assert (folder / 'markers.txt').read_text().splitlines() == list(expected.markers)
        columns = list(pd.read_csv(folder / 'input.csv.gz', nrows=0).columns)
        assert columns == ['cell_id', 'dataset', 'unit_key', *expected.markers]
    save(folder / 'MASK_BOUNDARY_AUDIT.json', dict(passed=True, marker=marker,
        feature_names=list(expected.markers), canonical_mask=list(expected.masked_markers),
        all_platforms_masked=True, input_columns=8, cells=84000))


def run_shard(method, marker, mh):
    manifest = json.loads((OUT / 'MANIFEST.json').read_text())
    assert file_sha256(OUT / 'MANIFEST.json') == mh
    task = task_for()
    assert task.pool.sha256 == manifest['pool_sha256']
    jobs = sorted([j for j in manifest['jobs'] if j['method'] == method and j['marker'] == marker],
                  key=lambda j: (j['source'], j['target'], j['seed'], j['family']))
    assert len(jobs) == 54
    status_path = OUT / 'status' / f'{method}__{marker}.json'
    start = time.monotonic()
    status = dict(method=method, marker=marker, pid=os.getpid(), completed=0, total=54, stage='alignment')
    def progress(**extra):
        status.update(extra, elapsed_seconds=time.monotonic()-start, updated_unix=time.time())
        save(status_path, status)
    progress()
    try:
        assert shutil.disk_usage(OUT).free >= 40 * 2**30, 'Less than 40 GiB free; pause for review'
        masked = task.input(method, task.receipt['canonical_markers'][marker])
        repdir = OUT / 'representations' / marker / method
        rep = fit_representation(masked, method, fitting_pool=task.pool, out=repdir,
            input_contract=task.receipt, family=manifest['family'], timeout=7200)
        audit_adapter_inputs(method, marker, repdir, masked)
        for i, job in enumerate(jobs):
            assert shutil.disk_usage(OUT).free >= 40 * 2**30, 'Less than 40 GiB free; pause for review'
            progress(stage='prediction', current_job=job['id'])
            p = json.loads((OUT / job['plan_file']).read_text())
            plan = Supervision(tuple(p['fit']), tuple(p['validation']), tuple(p['calibration']))
            assert digest(asdict(plan)) == job['supervision_sha256']
            dest = OUT / 'jobs' / method / job['id']
            if (dest / 'JOB_COMPLETE.json').exists():
                verify_job(dest, job, mh)
            else:
                assert not dest.exists(), f'Partial job requires inspection: {dest}'
                r = run_supervised(task, rep, plan, sources=[job['source']], target=job['target'],
                    marker=marker, candidates=[Candidate(**c) for c in job['candidates']],
                    seed=job['seed'], out=dest, split_mode=p['mode'])
                assert r['fit_cells'] == 500 and r['validation_cells'] == 512 and r['calibration_cells'] == 1792
                assert r['target_identity_level'] == 'patient'
                save(dest / 'JOB_COMPLETE.json', dict(manifest_sha256=mh, job_sha256=digest(job),
                    artifact_sha256={p.name:file_sha256(p) for p in dest.iterdir() if p.is_file()}))
            progress(completed=i+1)
        progress(stage='complete')
        print(f'DONE {method} {marker}: 54/54; {time.monotonic()-start:.0f}s', flush=True)
        return dict(method=method, marker=marker, status='passed', jobs=54)
    except Exception as exc:
        progress(stage='failed', error=repr(exc), traceback=traceback.format_exc())
        print(f'FAILED {method} {marker}: {exc}', flush=True)
        return dict(method=method, marker=marker, status='failed', error=repr(exc))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    assert args.prepare_only or args.run
    assert 1 <= args.workers <= 2
    with (OUT / 'RUN.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not (OUT / 'MANIFEST.json').exists(): prepare()
        manifest = json.loads((OUT / 'MANIFEST.json').read_text())
        verify(manifest)
        if args.prepare_only: return
        mh = file_sha256(OUT / 'MANIFEST.json')
        save(OUT / 'RUN.json', dict(pid=os.getpid(), started_unix=time.time(), workers=args.workers,
             manifest_sha256=mh, total_jobs=4860, total_representations=90, network_required=False))
        # RAMP and percentile first for an early pipeline check; then expensive fits.
        ordered = ['ramp', 'percentile', 'maxfuse', 'unaligned', 'clipped_zscore', 'minmax',
                   'harmony_direct', 'cycombine', 'cytonorm', 'fastmnn']
        results = []
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(run_shard, m, marker, mh) for m in ordered for marker in manifest['markers']]
            for future in as_completed(futures):
                results.append(future.result())
                save(OUT / 'RUN_STATUS.json', dict(completed_shards=len(results), total_shards=90,
                    passed_shards=sum(r['status']=='passed' for r in results),
                    failed_shards=sum(r['status']=='failed' for r in results), results=results,
                    updated_unix=time.time()))
        verify(manifest)
        if not all(r['status'] == 'passed' for r in results):
            save(OUT / 'INCOMPLETE.json', dict(results=results, no_aggregate_release=True))
            raise RuntimeError('Incomplete experiment; preserved failures require inspection')
        import summarize
        summarize.main()
        save(OUT / 'COMPLETE.json', dict(manifest_sha256=mh, jobs=4860, representations=90,
             results=results, completed_unix=time.time(), original_inputs_and_code_unchanged=True))


if __name__ == '__main__': main()
