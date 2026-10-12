"""RAMP-only J1 pilot on the released Senosain140k reference; no source edits."""
import os
import sys
sys.dont_write_bytecode = True
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMBA_NUM_THREADS', 'LOKY_MAX_CPU_COUNT']:
    os.environ[key] = '1'
from pathlib import Path
OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
RELEASE = ROOT.parent / 'story_lung_HLA_ABC/releases/senosain140k_20260921'
OLD = ROOT / 'exploration_lung_story_four_classes_20260921'
sys.path.insert(0, str(ROOT / 'src'))
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import platform
import time
import numpy as np
import pandas as pd
import sklearn
from trainability.core import digest, file_sha256, seed_for
from trainability.heads import Candidate
from trainability.inputs import FittingPool, PreparedRepresentation
from trainability.real_data import RealTask
from trainability.runner import eligible_ids, run_supervised
from trainability.sampling import Supervision, nested_draws

CLASSES = ['CD4 T', 'CD8 T', 'Myeloid / APC', 'Tumor / epithelial']
ARMS = ['keep_c02', 'exclude_c02_supervision']
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


def task_for(arm):
    assert arm in ARMS
    md = pd.read_csv(RELEASE / 'inputs/CELL_METADATA.csv.gz', dtype=str, keep_default_na=False)
    md = md.rename(columns={'cell_key': 'cell_id'})
    md['patient_id'] = md.dataset + '::' + md.patient_id
    md['identity_level'] = 'patient'
    md['original_broad_type'] = md.broad_type
    md['celltype'] = [MAPPING.get((d, p), c) for d, p, c in
                      zip(md.dataset, md.provider_cell_type, md.broad_type)]
    md['is_c02'] = md.dataset.eq('Senosain CyTOF') & md.parent_cluster.eq('C02')
    assert md.is_c02.sum() == 3171
    if arm == 'exclude_c02_supervision':
        md.loc[md.is_c02, 'celltype'] = 'C02 / uncertain supervision'
    md['joint_evaluation'] = md.celltype.isin(CLASSES)
    design = json.loads((RELEASE / 'inputs/DESIGN.json').read_text())
    markers = tuple(design['markers'])
    assert len(markers) == 9 and len(md) == 84000 and md.cell_id.is_unique
    assert md.groupby('dataset').patient_id.nunique().eq(28).all()
    native = np.load(RELEASE / 'inputs/RAW_SHARED9.npy', allow_pickle=False)
    assert native.shape == (84000, 9) and np.isfinite(native).all()
    pool = FittingPool('lung_story:senosain140k_release_20260921:all84000',
                       tuple(md.cell_id), tuple(md.dataset), tuple(md.unit_key))
    # J1 only: no J2 input contract is asserted or used in this pilot.
    receipt = dict(classes=CLASSES, canonical_markers={m: m for m in markers},
                   pilot=True, annotation_arm=arm, j2_not_run=True,
                   manifest_path=str(OUT / 'MANIFEST.json'))
    return RealTask('lung_story', native, md, markers, {}, receipt, pool)


def original_jobs():
    path = ROOT / 'config/formal_v0_5/lung_story__ramp.json'
    jobs = [j for j in json.loads(path.read_text())['jobs'] if
            j['task_type'] == 'J1' and 'primary' in j['analysis_scopes']]
    return sorted(jobs, key=lambda j: (j['source'], j['target'], j['seed'], j['family']))


def parent_roles():
    roles = {}
    for path in sorted((OLD / 'plans').glob('*.json')):
        j = json.loads(path.read_text())
        key = j['job']['source'], j['job']['seed']
        if key in roles:
            assert roles[key] == j['parent_source_patients']
        else:
            roles[key] = j['parent_source_patients']
    assert len(roles) == 9
    return roles


def prepare():
    assert not (OUT / 'MANIFEST.json').exists()
    protected = {}
    def protect(p):
        protected[str(p)] = file_sha256(p)
    release_manifest = json.loads((RELEASE / 'RELEASE_MANIFEST.json').read_text())
    for rel in ['inputs/CELL_METADATA.csv.gz', 'inputs/RAW_SHARED9.npy',
                'inputs/DESIGN.json', 'alignment/RAMP_SHARED9.npy',
                'alignment/COMPLETE.json', 'scripts/align.py']:
        p = RELEASE / rel
        assert file_sha256(p) == release_manifest[rel], rel
        protect(p)
    protect(RELEASE / 'RELEASE_MANIFEST.json')
    protect(ROOT / 'config/formal_v0_5/lung_story__ramp.json')
    for p in (ROOT / 'src/trainability').glob('*.py'):
        protect(p)
    for p in (OLD / 'plans').glob('*.json'):
        protect(p)
    for f in ['FULL_COHORT_SUMMARY.csv', 'BY_MODEL.csv', 'BY_DIRECTION.csv']:
        protect(OLD / f)
    protect(ROOT / 'reports/senosain_new_annotation_review_20260921/c02_marker_recheck/README_ZH.md')
    roles = parent_roles()
    base = original_jobs()
    assert len(base) == 54
    planned, support, counts = [], [], []
    pool_hash = None
    for arm in ARMS:
        task = task_for(arm)
        if pool_hash is None:
            pool_hash = task.pool.sha256
        assert task.pool.sha256 == pool_hash
        md = task.metadata.set_index('cell_id')
        task.metadata.to_csv(OUT / f'METADATA_{arm}.csv.gz', index=False,
                            compression={'method': 'gzip', 'mtime': 0})
        for (ds, ct), g in task.metadata.groupby(['dataset', 'celltype']):
            counts.append(dict(arm=arm, dataset=ds, celltype=ct, cells=len(g),
                               patients=g.patient_id.nunique(), eligible=ct in CLASSES))
        for j in base:
            source, target, seed = j['source'], j['target'], j['seed']
            parents = roles[(source, seed)]
            assert not set(parents['fit']) & set(parents['validation'])
            source_md = md[md.dataset.eq(source) & md.celltype.isin(CLASSES)]
            assert set(source_md.patient_id) <= set(parents['fit'] + parents['validation'])
            selected = {}
            for role, budget in [('fit', 500), ('validation', 512)]:
                ids = source_md.index[source_md.patient_id.isin(parents[role])].tolist()
                selected[role] = nested_draws(ids, [budget], seed_for(seed, role))[budget]
                chosen = md.loc[list(selected[role])]
                assert set(chosen.celltype) == set(CLASSES)
                if arm == 'exclude_c02_supervision':
                    assert not chosen.is_c02.any()
                for ct in CLASSES:
                    support.append(dict(arm=arm, source=source, target=target, seed=seed,
                        family=j['family'], role=role, celltype=ct,
                        cells=int(chosen.celltype.eq(ct).sum()),
                        patients=int(chosen.loc[chosen.celltype.eq(ct), 'patient_id'].nunique())))
            plan = Supervision(selected['fit'], selected['validation'])
            plan.assert_allowed(eligible_ids(task, source, True), eligible_ids(task, target, True))
            ident = 'NEWLUNG_' + digest([arm, source, target, seed, j['family']])[:18]
            newj = dict(j, id=ident, arm=arm, original_job_id=j['id'],
                supervision_sha256=digest(dict(fit=plan.fit, validation=plan.validation, calibration=())))
            save(OUT / 'plans' / f'{ident}.json', dict(job=newj, fit=list(plan.fit),
                validation=list(plan.validation), target_ids=list(eligible_ids(task, target, True)),
                parent_source_patients=parents, pool_sha256=pool_hash))
            planned.append(newj)
    pd.DataFrame(counts).to_csv(OUT / 'CELL_COUNTS.csv', index=False)
    sp = pd.DataFrame(support)
    sp.to_csv(OUT / 'SUPERVISION_COUNTS.csv', index=False)
    sp.groupby(['arm', 'source', 'role', 'celltype']).agg(min_cells=('cells', 'min'),
        max_cells=('cells', 'max'), min_patients=('patients', 'min')).reset_index().to_csv(
        OUT / 'SUPPORT_RANGES.csv', index=False)
    rep = np.load(RELEASE / 'alignment/RAMP_SHARED9.npy', allow_pickle=False)
    assert rep.shape == (84000, 9) and np.isfinite(rep).all()
    artifacts = list((OUT / 'plans').glob('*.json')) + list(OUT.glob('METADATA_*.csv.gz'))
    artifacts += [OUT / 'run.py', OUT / 'summarize.py', OUT / 'DESIGN_ZH.md']
    save(OUT / 'MANIFEST.json', dict(method='ramp', task='J1 classification only',
        classes=CLASSES, arms=ARMS, jobs=planned, total_jobs=108, pool_sha256=pool_hash,
        alignment_cells=84000, refit_alignment=False, all_c02_retained_in_alignment=True,
        representation=str(RELEASE / 'alignment/RAMP_SHARED9.npy'),
        source_only_selection='Original candidate grids and patient-equal macro-F1',
        sampling='Original source patient roles; uniform 500 fit/512 validation after scope filtering',
        same_target_control='Rescore keep-C02 predictions on exact exclude-C02 target IDs',
        exploratory=True, formal_figure_changed=False,
        protected_files=protected, experiment_hashes={str(p): file_sha256(p) for p in artifacts},
        environment=dict(python=sys.version, platform=platform.platform(), sklearn=sklearn.__version__,
                         numpy=np.__version__, pandas=pd.__version__), created_unix=time.time()))
    print('Prepared 108 RAMP jobs; all 84000 cells retained in alignment.', flush=True)


def verify_inputs(manifest):
    for p, h in {**manifest['protected_files'], **manifest['experiment_hashes']}.items():
        assert file_sha256(p) == h, p


def run_arm(arm, manifest_hash):
    manifest = json.loads((OUT / 'MANIFEST.json').read_text())
    task = task_for(arm)
    representation = PreparedRepresentation(np.load(manifest['representation']),
        task.pool.cell_ids, task.pool.sha256, 'ramp')
    jobs = [j for j in manifest['jobs'] if j['arm'] == arm]
    start = time.monotonic()
    for i, j in enumerate(jobs):
        saved = json.loads((OUT / 'plans' / f"{j['id']}.json").read_text())
        plan = Supervision(tuple(saved['fit']), tuple(saved['validation']))
        assert digest(dict(fit=plan.fit, validation=plan.validation, calibration=())) == j['supervision_sha256']
        assert list(eligible_ids(task, j['target'], True)) == saved['target_ids']
        dest = OUT / 'jobs' / arm / j['id']
        done = dest / 'EXPERIMENT_COMPLETE.json'
        if done.exists():
            proof = json.loads(done.read_text())
            assert proof['manifest_sha256'] == manifest_hash
            for p, h in proof['files'].items():
                assert file_sha256(p) == h
        else:
            assert not dest.exists(), f'Partial job requires inspection: {dest}'
            result = run_supervised(task, representation, plan, sources=[j['source']],
                target=j['target'], marker=None, candidates=[Candidate(**c) for c in j['candidates']],
                seed=j['seed'], out=dest, split_mode='patient_disjoint_validation')
            assert result['score']['classes'] == CLASSES
            assert result['fit_cells'] == 500 and result['validation_cells'] == 512
            save(done, dict(manifest_sha256=manifest_hash,
                files={str(p): file_sha256(p) for p in dest.iterdir() if p.is_file()}))
        save(OUT / 'status' / f'{arm}.json', dict(completed=i+1, total=len(jobs),
                                                seconds=time.monotonic()-start))
        if (i+1) % 18 == 0:
            print(arm, i+1, '/', len(jobs), flush=True)
    return dict(arm=arm, jobs=len(jobs), seconds=time.monotonic()-start)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if not (OUT / 'MANIFEST.json').exists():
        prepare()
    manifest = json.loads((OUT / 'MANIFEST.json').read_text())
    verify_inputs(manifest)
    if args.prepare_only:
        return
    mh = file_sha256(OUT / 'MANIFEST.json')
    save(OUT / 'RUN.json', dict(pid=os.getpid(), started_unix=time.time(), manifest_sha256=mh))
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run_arm, arm, mh) for arm in ARMS]
        results = [f.result() for f in as_completed(futures)]
    verify_inputs(manifest)
    save(OUT / 'RUN_COMPLETE.json', dict(all_passed=True, results=results,
        protected_files_unchanged=True, finished_unix=time.time()))
    print('COMPLETE: 108/108; protected files unchanged.', flush=True)


if __name__ == '__main__':
    main()
