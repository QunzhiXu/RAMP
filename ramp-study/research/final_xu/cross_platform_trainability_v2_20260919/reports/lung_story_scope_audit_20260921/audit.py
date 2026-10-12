"""Audit Lung-story labels, marker support and frozen J1 results; no refitting."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FX = ROOT.parent
sys.path.insert(0, str(ROOT / 'src'))
import json
from dataclasses import replace
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from trainability.real_data import load_task
from trainability.core import digest, file_sha256, seed_for
from trainability.runner import eligible_ids, make_plan
from trainability.sampling import source_split, nested_draws, Supervision

TASK = 'lung_story'
METHODS = ['ramp', 'unaligned', 'clipped_zscore', 'minmax', 'percentile',
           'harmony_direct', 'cycombine', 'cytonorm', 'fastmnn', 'maxfuse']
HASHES = {}


def track(p):
    p = Path(p)
    HASHES[str(p)] = file_sha256(p)
    return p


def readj(p): return json.loads(track(p).read_text())


def csv(p, **kwargs): return pd.read_csv(track(p), **kwargs)


def export(name, rows):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(HERE / name, index=False)


def primary(cfg):
    return sorted([j for j in cfg['jobs'] if j['task_type'] == 'J1' and 'primary' in j['analysis_scopes']],
                  key=lambda j: (j['source'], j['target'], j['seed'], j['family']))


def main():
    task = load_task(TASK)
    for name in ['manifest.json', 'metadata.csv.gz', 'native.npy']:
        track(ROOT / 'data/real_inputs_v0_3_r2' / TASK / name)
    for p in (ROOT / 'src/trainability').glob('*.py'): track(p)
    source = csv(FX / 'story_lung_HLA_ABC/data_umap/data/CELL_METADATA.csv.gz',
                 dtype=str, keep_default_na=False)
    for p in ['data/MARKER_MAPPING.csv', 'data/DESIGN.json', 'README_ZH.md']:
        track(FX / 'story_lung_HLA_ABC/data_umap' / p)
    track(FX / 'overnight0909/scripts/export_full527_cells.R')
    md = task.metadata.merge(source[['cell_key', 'source_cell_id', 'provider_cell_type',
                                    'paper_mesenchymal_gate']], left_on='cell_id', right_on='cell_key',
                             how='left', validate='one_to_one')
    assert len(md) == 84000 and md.provider_cell_type.notna().all()
    assert np.array_equal(md.cell_id, task.metadata.cell_id)
    mapping = {('Senosain CyTOF', 'CD4 T'): 'CD4 T', ('Senosain CyTOF', 'CD8 T'): 'CD8 T',
               ('Cords IMC', 'CD4'): 'CD4 T', ('Cords IMC', 'CD8'): 'CD8 T',
               ('Zheng mIF', 'Th'): 'CD4 T', ('Zheng mIF', 'Treg'): 'CD4 T',
               ('Zheng mIF', 'Tc'): 'CD8 T', ('Zheng mIF', 'DNT'): 'DNT'}
    md['refined_label'] = [mapping.get((ds, label), broad) for ds, label, broad in
                            zip(md.dataset, md.provider_cell_type, md.celltype)]
    export('PROVIDER_COMPOSITION.csv', md.groupby(['dataset', 'celltype', 'provider_cell_type']).size().rename('cells').reset_index())
    counts = md.groupby(['dataset', 'celltype']).agg(cells=('cell_id', 'size'), patients=('patient_id', 'nunique')).reset_index()
    export('CLASS_COUNTS.csv', counts)
    export('REFINED_CLASS_COUNTS.csv', md.groupby(['dataset', 'refined_label']).agg(
        cells=('cell_id', 'size'), patients=('patient_id', 'nunique')).reset_index())
    export('PATIENT_CLASS_COUNTS.csv', md.groupby(['dataset', 'patient_id', 'celltype']).size().rename('cells').reset_index())
    # This is an overlap diagnostic with another project-derived annotation,
    # NOT an independent truth set and never a replacement for frozen labels.
    newer = csv(FX / 'raw_data/lung_cancer/Senosain_2025_CyTOF_and_omics/annotation/rough_annotation/senosain_140k_final_cell_annotations.csv.gz',
                dtype=str, keep_default_na=False)
    overlap = md[md.dataset.eq('Senosain CyTOF')].merge(newer[['source_cell_id', 'final_cell_type', 'annotation_confidence', 'rationale_short']],
        on='source_cell_id', how='inner', validate='one_to_one')
    export('SENOSAIN_NEWER_ANNOTATION_OVERLAP.csv', overlap[['cell_id', 'source_cell_id', 'provider_cell_type', 'final_cell_type', 'annotation_confidence', 'rationale_short']])
    export('SENOSAIN_ANNOTATION_CROSSTAB.csv', overlap.groupby(['provider_cell_type', 'final_cell_type']).size().rename('cells').reset_index())
    track(FX / 'story_test_lung/shared_inputs/senosain/senosain_exact70k_provenance.json')
    track(FX / 'raw_data/lung_cancer/Senosain_2025_CyTOF_and_omics/annotation/rough_annotation/final_annotation_manifest.json')
    # Fixed broad-identity and within-T contrasts, decided without selecting
    # thresholds or cells according to any method's prediction performance.
    contrasts = [('CD3', 'T cell', None, 'celltype'), ('CD31', 'Endothelial', None, 'celltype'),
        ('VIMENTIN', 'Fibroblast / stroma', None, 'celltype'),
        ('VIMENTIN', 'Fibroblast / stroma', 'Tumor / epithelial', 'celltype'),
        ('EPI', 'Tumor / epithelial', None, 'celltype'),
        ('HLA_DR', 'Myeloid / APC', 'T cell', 'celltype'),
        ('CD45', 'Myeloid / APC', 'T cell', 'celltype'),
        ('CD4', 'CD4 T', 'CD8 T', 'refined_label'), ('CD8', 'CD8 T', 'CD4 T', 'refined_label')]
    auc_rows = []
    for marker, positive, negative, field in contrasts:
        for (ds, patient), g in md.groupby(['dataset', 'patient_id']):
            g = g[g[field].isin(task.receipt['classes'] if negative is None else [positive, negative])]
            ispos = g[field].eq(positive).to_numpy()
            npos, nneg = int(ispos.sum()), int((~ispos).sum())
            enough = min(npos, nneg) >= 5
            auc_rows.append(dict(dataset=ds, patient=patient, marker=marker, positive=positive,
                negative=negative or 'other original eligible classes', positive_cells=npos,
                negative_cells=nneg, eligible=enough,
                auroc=float(roc_auc_score(ispos, task.native[g.index, task.markers.index(marker)])) if enough else None))
    auc = pd.DataFrame(auc_rows)
    export('MARKER_CONTRASTS_PER_PATIENT.csv', auc)
    export('MARKER_CONTRASTS_SUMMARY.csv', auc[auc.eligible].groupby(['dataset', 'marker', 'positive', 'negative']).agg(
        mean_auroc=('auroc', 'mean'), median_auroc=('auroc', 'median'), patients=('patient', 'size')).reset_index())
    cfg = readj(ROOT / f'config/formal_v0_5/{TASK}__ramp.json')
    jobs = primary(cfg)
    assert len(jobs) == 54
    original_md = task.metadata.set_index('cell_id')
    candidate_md = md.set_index('cell_id')
    original_classes = task.receipt['classes']
    # Candidate support is inspected without training or choosing a winning subset.
    scopes = {
        'original_five': ('celltype', original_classes),
        'original_support_four': ('celltype', [c for c in original_classes if c != 'Endothelial']),
        'split_T_six_no_DNT': ('refined_label', [c for c in original_classes if c != 'T cell'] + ['CD4 T', 'CD8 T']),
        'CD4_CD8_epithelial_support_only': ('refined_label', ['CD4 T', 'CD8 T', 'Tumor / epithelial'])}
    fit_rows = []
    seen = set()
    for job in jobs:
        skey = job['source'], job['seed']
        if skey in seen: continue
        seen.add(skey)
        plan, mode = make_plan(task, job['source'], job['target'], fit_budget=500, validation_budget=512, seed=job['seed'])
        assert digest(dict(fit=plan.fit, validation=plan.validation, calibration=plan.calibration)) == job['supervision_sha256']
        split = source_split(eligible_ids(task, job['source'], True), original_md.patient_id.to_dict(), .2, job['seed'], patient_mapping_verified=True)
        for scope, (field, classes) in scopes.items():
            roles = {}
            for role, budget in [('fit', 500), ('validation', 512)]:
                pool = tuple(i for i in split[role + '_pool'] if candidate_md.at[i, field] in classes)
                chosen = nested_draws(pool, [budget], seed_for(job['seed'], role))[budget]
                roles[role] = chosen
                if scope == 'original_five': assert chosen == getattr(plan, role)
                for label in classes:
                    n = int(candidate_md.loc[list(chosen), field].eq(label).sum())
                    fit_rows.append(dict(scope=scope, source=job['source'], seed=job['seed'], role=role,
                        celltype=label, cells=n, pool_cells=int(candidate_md.loc[list(pool), field].eq(label).sum()),
                        patients=int(candidate_md.loc[list(chosen)].loc[lambda d: d[field].eq(label), 'patient_id'].nunique())))
            assert not set(candidate_md.loc[list(roles['fit']), 'patient_id']) & set(candidate_md.loc[list(roles['validation']), 'patient_id'])
    support = pd.DataFrame(fit_rows)
    export('EXISTING_AND_CANDIDATE_SUPPORT.csv', support)
    export('SUPPORT_RANGES.csv', support.groupby(['scope', 'source', 'role', 'celltype']).agg(
        minimum_cells=('cells', 'min'), maximum_cells=('cells', 'max'), min_patients=('patients', 'min')).reset_index())
    # Audit frozen numerical representation coverage without fitting anything.
    repdir = ROOT / f'representations/formal_v0_5/{TASK}/complete/ramp'
    receipt = readj(repdir / 'COMPLETE.json')
    x = np.load(track(repdir / 'values.npy'))
    baseline = np.load(track(repdir / 'kmeanslow.npy'))
    ids = np.load(track(repdir / 'cell_ids.npy'))
    assert np.array_equal(ids, np.asarray(task.pool.cell_ids))
    assert file_sha256(repdir / 'values.npy') == receipt['values_sha256']
    coverage = []
    for (ds, unit), g in md.groupby(['dataset', 'unit_key']):
        for k, marker in enumerate(task.markers):
            coverage.append(dict(dataset=ds, unit=unit, marker=marker,
                exactly_baseline=bool(np.array_equal(x[g.index,k], baseline[g.index,k]))))
    coverage = pd.DataFrame(coverage)
    export('BASELINE_ONLY_MARKERS.csv', coverage)
    export('BASELINE_ONLY_UNITS.csv', coverage.groupby(['dataset', 'unit']).exactly_baseline.all().reset_index())
    # All 540 original primary jobs; preserve full confusion matrices, including
    # false positives from excluded true classes in the four-class diagnostic.
    scores, class_rows, confusion = [], [], []
    reference = {(j['source'], j['target'], j['seed'], j['family']): j for j in jobs}
    for method in METHODS:
        c = readj(ROOT / f'config/formal_v0_5/{TASK}__{method}.json')
        jm = primary(c)
        assert len(jm) == 54
        for j in jm:
            key = j['source'], j['target'], j['seed'], j['family']
            assert j['supervision_sha256'] == reference[key]['supervision_sha256']
            assert j['candidates'] == reference[key]['candidates']
            r = readj(Path(c['output']) / 'jobs' / j['id'] / 'RESULT.json')
            s = r['score']; assert s['classes'] == original_classes
            cm = np.asarray(s['confusion_matrix']);tp = np.diag(cm)
            sup, pred = cm.sum(1), cm.sum(0)
            f1 = np.divide(2*tp, sup+pred, out=np.zeros(len(tp)), where=sup+pred > 0)
            recall = np.divide(tp, sup, out=np.zeros(len(tp)), where=sup > 0)
            precision = np.divide(tp, pred, out=np.zeros(len(tp)), where=pred > 0)
            assert abs(f1.mean()-s['macro_f1']) < 1e-12
            base = dict(method=method, source=j['source'], target=j['target'], seed=j['seed'], family=j['family'])
            scores.append(dict(**base, patient_macro_f1=s['macro_f1'], patient_accuracy=s['accuracy'],
                patient_balanced_accuracy=s['balanced_accuracy'], selected_four_f1=float(f1[1:].mean())))
            for k, label in enumerate(original_classes):
                class_rows.append(dict(**base, celltype=label, f1=f1[k], recall=recall[k], precision=precision[k]))
                if method == 'ramp':
                    for q, output in enumerate(original_classes):
                        confusion.append(dict(**base, truth=label, prediction=output,
                            conditional_fraction=float(cm[k,q]/sup[k]) if sup[k] else 0.))
    scores = pd.DataFrame(scores)
    export('FROZEN_SCORES_PER_JOB.csv', scores)
    metric_cols = ['patient_macro_f1', 'patient_accuracy', 'patient_balanced_accuracy', 'selected_four_f1']
    summary = scores.groupby('method')[metric_cols].mean().reset_index()
    for col in metric_cols: summary[col + '_rank'] = summary[col].rank(ascending=False, method='min').astype(int)
    export('METHOD_SUMMARY.csv', summary.sort_values('patient_macro_f1', ascending=False))
    previous = csv(ROOT / 'reports/celltype_support_audit_20260921/PER_BENCHMARK_SCORES.csv')
    previous = previous[previous.task.eq(TASK) & previous.rule.eq('support_ge10') & previous.metric.eq('macro_f1')].set_index('method')
    assert all(abs(r.selected_four_f1-previous.loc[r.method, 'score']) < 1e-12 for r in summary.itertuples())
    export('BY_DIRECTION.csv', scores.groupby(['method', 'source', 'target'])[metric_cols].mean().reset_index())
    classes = pd.DataFrame(class_rows)
    export('PER_CLASS_PER_JOB.csv', classes)
    export('PER_CLASS_MEANS.csv', classes.groupby(['method', 'celltype'])[['f1', 'recall', 'precision']].mean().reset_index())
    export('PER_CLASS_BY_DIRECTION.csv', classes.groupby(['method', 'source', 'target', 'celltype'])[['f1', 'recall', 'precision']].mean().reset_index())
    export('RAMP_CONFUSION.csv', pd.DataFrame(confusion).groupby(['source', 'target', 'truth', 'prediction']).conditional_fraction.mean().reset_index())
    for p,h in HASHES.items(): assert file_sha256(p) == h,p
    (HERE/'CHECKS.json').write_text(json.dumps(dict(task=TASK, frozen_cells=84000, original_jobs_audited=540,
        source_annotations_joined_by_exact_ID=True, newer_annotation_overlap_cells=len(overlap),
        changed_labels=False, trained_models=False, refit_alignment=False, original_files_unchanged=True,
        new_annotation_is_project_derived_not_ground_truth=True, source_patient_splits_preserved_for_candidate_support=True,
        candidates_are_support_audits_only=True, protected_files=HASHES,
        audit_script_sha256=file_sha256(__file__)), indent=2)+'\n')
    print(summary.to_string(index=False), flush=True)
    print('AUDIT COMPLETE; no new labels, models or alignment.', flush=True)


if __name__ == '__main__':
    main()
