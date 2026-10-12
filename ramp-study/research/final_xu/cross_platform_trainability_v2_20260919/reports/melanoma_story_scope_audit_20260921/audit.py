"""Read-only audit of the melanoma-story J1 ontology, support and frozen predictions."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dataclasses import replace
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from trainability.core import digest, file_sha256, seed_for
from trainability.real_data import load_task
from trainability.runner import make_plan, eligible_ids
from trainability.sampling import source_split, nested_draws, Supervision

TASK = 'melanoma_story'
KEEP = ['B lineage', 'CD8 T', 'Other T / CD8-']
CAP = ROOT / 'exploration_donor_cap20_20260921'
METHODS = ['ramp', 'unaligned', 'clipped_zscore', 'minmax', 'percentile',
           'harmony_direct', 'cycombine', 'cytonorm', 'fastmnn', 'maxfuse']
PROTECTED = {}


def track(p):
    p = Path(p)
    PROTECTED[str(p)] = file_sha256(p)
    return p


def readj(p):
    return json.loads(track(p).read_text())


def csv(p, **kw):
    return pd.read_csv(track(p), **kw)


def table(df):
    lines = ['| ' + ' | '.join(df.columns) + ' |', '| ' + ' | '.join(['---'] * len(df.columns)) + ' |']
    for row in df.itertuples(index=False, name=None):
        lines.append('| ' + ' | '.join(f'{x:.4f}' if isinstance(x, (float, np.floating)) else str(x) for x in row) + ' |')
    return '\n'.join(lines)


def main():
    full = load_task(TASK)
    for f in ['manifest.json', 'metadata.csv.gz', 'native.npy']:
        track(ROOT / 'data/real_inputs_v0_3_r2' / TASK / f)
    md = full.metadata
    marker_index = {m: k for k, m in enumerate(full.markers)}
    provider = []
    base = ROOT.parent / 'figure_melanoma/inputs/cohort_package'
    cite = csv(base / 'engelbrecht_citeseq/locked_metadata.csv.gz', dtype=str, keep_default_na=False)
    imc = csv(base / 'moldoveanu_imc/locked_metadata.csv.gz', dtype=str, keep_default_na=False)
    for frame, key, label in [(cite, 'source_cell_id', 'Broad_cell.type.1'),
                               (imc, 'cell_uid', 'Cluster.v2')]:
        provider.append(frame[[key, label]].rename(columns={key: 'cell_id', label: 'provider_label'}))
    joined = md.merge(pd.concat(provider, ignore_index=True), on='cell_id', how='left', validate='one_to_one')
    assert joined.provider_label.notna().all() and len(joined) == len(md)
    composition = joined.groupby(['dataset', 'celltype', 'provider_label']).size().rename('cells').reset_index()
    composition.to_csv(HERE / 'PROVIDER_COMPOSITION.csv', index=False)
    tcite = cite[cite['Broad_cell.type.1'].isin(['CD4_T', 'T_prolif'])]
    assert len(tcite) == 19196
    assert tcite['Broad_celltype3'].eq('CD4').all()
    tcite.groupby(['Broad_cell.type.1', 'Broad_celltype3', 'cell_type']).size().rename('cells').reset_index().to_csv(
        HERE / 'CITE_CD4_HIERARCHY.csv', index=False)
    paper = track(ROOT.parent / 'raw_data/melanoma/moldoveanu_imc/Moldoveanu_2022.md')
    track(base / 'DATA_SELECTION.md')
    configs = {}
    rows, classes, confusion = [], [], []
    for method in METHODS:
        cfg = readj(ROOT / f'config/formal_v0_5/{TASK}__{method}.json')
        jobs = [j for j in cfg['jobs'] if j['task_type'] == 'J1' and 'primary' in j['analysis_scopes']]
        assert len(jobs) == 18
        configs[method] = jobs
        for j in jobs:
            result = readj(Path(cfg['output']) / 'jobs' / j['id'] / 'RESULT.json')
            score = result['score']
            labels = score['classes']
            cm = np.asarray(score['confusion_matrix'])
            support, predicted, tp = cm.sum(1), cm.sum(0), cm.diagonal()
            f1 = np.divide(2*tp, support+predicted, out=np.zeros(len(labels)), where=support+predicted > 0)
            recall = np.divide(tp, support, out=np.zeros(len(labels)), where=support > 0)
            precision = np.divide(tp, predicted, out=np.zeros(len(labels)), where=predicted > 0)
            assert abs(f1.mean() - score['macro_f1']) < 1e-12
            ix = [labels.index(c) for c in KEEP]
            common = dict(method=method, source=j['source'], target=j['target'], family=j['family'], seed=j['seed'])
            rows.append(dict(common, original_five_class_f1=f1.mean(), selected_class_f1=f1[ix].mean(),
                             original_accuracy=score['accuracy']))
            for k, cl in enumerate(labels):
                classes.append(dict(common, celltype=cl, f1=f1[k], recall=recall[k], precision=precision[k]))
                if method == 'ramp':
                    for l, predicted_label in enumerate(labels):
                        confusion.append(dict(common, truth=cl, prediction=predicted_label,
                                              conditional_fraction=cm[k,l]/support[k] if support[k] else 0))
    scores = pd.DataFrame(rows)
    scores.to_csv(HERE / 'FROZEN_SCORES_PER_JOB.csv', index=False)
    summary = scores.groupby('method')[['original_five_class_f1', 'selected_class_f1']].mean().reset_index()
    summary['selected_class_rank'] = summary.selected_class_f1.rank(ascending=False, method='min').astype(int)
    summary = summary.sort_values('selected_class_f1', ascending=False)
    summary.to_csv(HERE / 'REPRODUCED_METHOD_SUMMARY.csv', index=False)
    previous = csv(ROOT / 'reports/celltype_support_audit_20260921/PER_BENCHMARK_SCORES.csv')
    previous = previous[(previous.task == TASK) & (previous.rule == 'support_ge10') & (previous.metric == 'macro_f1')]
    check = summary.merge(previous[['method', 'score']], on='method', validate='one_to_one')
    assert len(check) == 10 and np.max(np.abs(check.score-check.selected_class_f1)) < 1e-12
    perclass = pd.DataFrame(classes)
    perclass.to_csv(HERE / 'PER_CLASS_PER_JOB.csv', index=False)
    direction = perclass.groupby(['method', 'source', 'target', 'celltype'])[['f1', 'recall', 'precision']].mean().reset_index()
    direction.to_csv(HERE / 'PER_CLASS_BY_DIRECTION.csv', index=False)
    perclass.groupby(['method', 'celltype'])[['f1', 'recall', 'precision']].mean().to_csv(HERE / 'PER_CLASS_MEANS.csv')
    pd.DataFrame(confusion).groupby(['source', 'target', 'truth', 'prediction']).conditional_fraction.mean().to_csv(HERE / 'RAMP_CONFUSION.csv')
    cap_indices = np.load(track(CAP / 'inputs' / TASK / 'original_row_indices.npy'))
    pools = {'full': full, 'cap20': full.subset(cap_indices, 'donor_cap20_seed20260921')}
    count_rows, donor_rows, plan_rows, coverage = [], [], [], []
    for cohort, task in pools.items():
        for (ds, ct), g in task.metadata.groupby(['dataset', 'celltype']):
            count_rows.append(dict(cohort=cohort, dataset=ds, celltype=ct, cells=len(g),
                                   patients=g.patient_id.nunique(), retained=ct in KEEP))
        for (ds, pat, ct), g in task.metadata.groupby(['dataset', 'patient_id', 'celltype']):
            donor_rows.append(dict(cohort=cohort, dataset=ds, patient=pat, celltype=ct, cells=len(g)))
        for scope, allowed in [('existing_five', task.receipt['classes']), ('candidate_three', KEEP),
                               ('candidate_three_preserve_original_patients', KEEP)]:
            restricted = replace(task, receipt=dict(task.receipt, classes=allowed))
            lookup = task.metadata.set_index('cell_id')
            for source in sorted(task.metadata.dataset.unique()):
                target = next(d for d in task.metadata.dataset.unique() if d != source)
                for seed in [20260919, 20260920, 20260921]:
                    ids = eligible_ids(task if scope.endswith('preserve_original_patients') else restricted,
                                       source, True)
                    split = source_split(ids, lookup.patient_id.to_dict(), .2, seed, patient_mapping_verified=True)
                    if scope.endswith('preserve_original_patients'):
                        split['fit_pool'] = tuple(i for i in split['fit_pool'] if lookup.at[i, 'celltype'] in KEEP)
                        split['validation_pool'] = tuple(i for i in split['validation_pool'] if lookup.at[i, 'celltype'] in KEEP)
                        plan = Supervision(
                            nested_draws(split['fit_pool'], [500], seed_for(seed, 'fit'))[500],
                            nested_draws(split['validation_pool'], [512], seed_for(seed, 'validation'))[512])
                        mode = split['mode']
                    else:
                        plan, mode = make_plan(restricted, source, target, fit_budget=500, validation_budget=512, seed=seed)
                    for ct in allowed:
                        plan_rows.append(dict(cohort=cohort, scope=scope, source=source, seed=seed, celltype=ct,
                            fit_pool=int(lookup.loc[list(split['fit_pool']), 'celltype'].eq(ct).sum()),
                            validation_pool=int(lookup.loc[list(split['validation_pool']), 'celltype'].eq(ct).sum()),
                            sampled_fit=int(lookup.loc[list(plan.fit), 'celltype'].eq(ct).sum()),
                            sampled_validation=int(lookup.loc[list(plan.validation), 'celltype'].eq(ct).sum()),
                            mode=mode))
        if cohort == 'full':
            folder = ROOT / f'representations/formal_v0_5/{TASK}/complete/ramp'
            rep = np.load(track(folder / 'values.npy'))
            baseline = np.load(track(folder / 'kmeanslow.npy'))
            assert np.array_equal(np.load(track(folder / 'cell_ids.npy')), np.asarray(task.pool.cell_ids))
        else:
            folder = CAP / 'representations' / TASK
            rep = np.load(track(folder / 'aligned.npy'))
            baseline = np.load(track(folder / 'kmeanslow.npy'))
        for (ds, unit), g in task.metadata.groupby(['dataset', 'unit_key']):
            unchanged = np.array_equal(rep[g.index], baseline[g.index])
            coverage.append(dict(cohort=cohort, dataset=ds, unit=unit, all_markers_exact_baseline=unchanged))
    counts = pd.DataFrame(count_rows)
    counts.to_csv(HERE / 'CLASS_COUNTS.csv', index=False)
    pd.DataFrame(donor_rows).to_csv(HERE / 'PATIENT_CLASS_COUNTS.csv', index=False)
    plans = pd.DataFrame(plan_rows)
    plans.to_csv(HERE / 'EXISTING_AND_CANDIDATE_PLAN_SUPPORT.csv', index=False)
    pd.DataFrame(coverage).to_csv(HERE / 'BASELINE_ONLY_UNITS.csv', index=False)
    # Marker-label consistency, within donor; not independent prediction validation.
    contrasts = [('CD20', 'B lineage', 'CD8 T'), ('CD20', 'B lineage', 'Other T / CD8-'),
                 ('CD8', 'CD8 T', 'Other T / CD8-'), ('CD4', 'Other T / CD8-', 'CD8 T'),
                 ('CD14', 'Myeloid', 'B lineage'), ('CD31', 'Endothelial', 'Myeloid')]
    auc_rows = []
    for marker, positive, negative in contrasts:
        for (ds, unit), g in md.groupby(['dataset', 'unit_key']):
            sub = g[g.celltype.isin([positive, negative])]
            pos = sub.celltype.eq(positive).to_numpy()
            npos, nneg = int(pos.sum()), int((~pos).sum())
            enough = min(npos, nneg) >= 5
            auc_rows.append(dict(dataset=ds, unit=unit, marker=marker, positive=positive,
                negative=negative, positive_cells=npos, negative_cells=nneg, eligible=enough,
                auroc=float(roc_auc_score(pos, full.native[sub.index, marker_index[marker]])) if enough else None))
    auc = pd.DataFrame(auc_rows)
    auc.to_csv(HERE / 'MARKER_CONTRASTS_PER_DONOR.csv', index=False)
    auc_summary = auc[auc.eligible].groupby(['dataset', 'marker', 'positive', 'negative']).agg(
        mean_patient_auroc=('auroc', 'mean'), donors=('auroc', 'count')).reset_index()
    auc_summary.to_csv(HERE / 'MARKER_CONTRASTS_SUMMARY.csv', index=False)
    for p, h in PROTECTED.items(): assert file_sha256(p) == h, p
    (HERE / 'CHECKS.json').write_text(json.dumps(dict(
        original_files_unchanged=True, trained_new_heads=False, reran_alignment=False,
        all_frozen_cells_joined_to_provider=True, frozen_cells=len(md),
        all_19196_CITE_other_T_have_provider_CD4_annotation=True,
        table_reproduced_to_1e12=True, existing_jobs_audited=180,
        hypothetical_three_class_plans_only=True, protected_files=PROTECTED,
        audit_script_sha256=file_sha256(__file__)), indent=2) + '\n')
    print(summary.to_string(index=False))
    print(auc_summary.to_string(index=False))
    print(plans[plans.scope.str.startswith('candidate_three') & (plans.celltype == 'B lineage')].to_string(index=False))


if __name__ == '__main__':
    main()
