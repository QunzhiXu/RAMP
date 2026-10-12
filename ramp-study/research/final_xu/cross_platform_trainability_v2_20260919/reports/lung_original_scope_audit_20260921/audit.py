"""Read-only original lung benchmark marker, label and old-head diagnostics."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
TASK = 'lung_cancer'
AUDIT = ROOT / 'reports/celltype_support_audit_20260921'
METHODS = ['ramp', 'unaligned', 'clipped_zscore', 'minmax', 'percentile', 'harmony_direct', 'cycombine', 'cytonorm', 'fastmnn', 'maxfuse']
sys.path.insert(0, str(ROOT / 'src'))
from trainability.real_data import load_task

def config(method):
    return json.loads((ROOT / f'config/formal_v0_5/{TASK}__{method}.json').read_text())

def primary_jobs(method):
    return [j for j in config(method)['jobs'] if j['task_type'] == 'J1' and 'primary' in j['analysis_scopes']]

def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from trainability.core import file_sha256


def main():
    full = load_task(TASK)
    md, markers = full.metadata, list(full.markers)
    all_classes = full.receipt['classes']
    ramp = ROOT / f'representations/formal_v0_5/{TASK}/complete/ramp'
    assert np.array_equal(np.load(ramp / 'cell_ids.npy'), md.cell_id.to_numpy(str))
    values = {'native': full.native, 'ramp': np.load(ramp / 'values.npy')}
    support = pd.read_csv(AUDIT / 'CLASS_SUPPORT.csv')
    support[support.task.eq(TASK)].to_csv(OUT / 'ORIGINAL_CLASS_SUPPORT.csv', index=False)
    provider = pd.read_csv(AUDIT / 'PROVIDER_LABEL_COMPOSITION.csv')
    provider = provider[provider.task.eq(TASK)].copy()
    provider['within_class_fraction'] = provider.cells / provider.groupby(['dataset', 'celltype']).cells.transform('sum')
    provider.to_csv(OUT / 'PROVIDER_LABEL_COMPOSITION.csv', index=False)
    rows, profiles = [], []
    for (ds, patient), block in md[md.celltype.isin(all_classes)].groupby(['dataset', 'patient_id']):
        ix = block.index.to_numpy()
        labels = block.celltype.to_numpy()
        for representation, x in values.items():
            ranks = rankdata(x[ix], axis=0)
            for ct in all_classes:
                positive = labels == ct
                n, negatives = int(positive.sum()), int((~positive).sum())
                if n >= 5 and negatives >= 20:
                    aucs = (ranks[positive].sum(axis=0) - n * (n + 1) / 2) / (n * negatives)
                    for marker, auc in zip(markers, aucs):
                        rows.append(dict(dataset=ds, patient=patient, representation=representation,
                            celltype=ct, marker=marker, positive_cells=n, negative_cells=negatives, auc=float(auc)))
                if n:
                    q = np.quantile(x[ix[positive]], [.25, .5, .75], axis=0)
                    for col, marker in enumerate(markers):
                        profiles.append(dict(dataset=ds, patient=patient, representation=representation,
                            celltype=ct, marker=marker, cells=n, q25=q[0, col], median=q[1, col], q75=q[2, col]))
    auc = pd.DataFrame(rows)
    auc.to_csv(OUT / 'MARKER_AUC_PER_PATIENT.csv', index=False)
    auc_summary = auc.groupby(['dataset', 'representation', 'celltype', 'marker'], as_index=False).agg(
        mean_patient_auc=('auc', 'mean'), patients=('patient', 'nunique'), minimum_patient_auc=('auc', 'min'))
    auc_summary.to_csv(OUT / 'MARKER_AUC_SUMMARY.csv', index=False)
    selected = [('CD4 T', 'CD4'), ('CD4 T', 'CD3'), ('CD8 T', 'CD8'), ('CD8 T', 'CD3'),
                ('B-like', 'HLA_DR'), ('Myeloid', 'HLA_DR'), ('Endothelial', 'CD31'), ('Tumor/Epithelial', 'EPI')]
    expected = auc_summary.merge(pd.DataFrame(selected, columns=['celltype', 'marker']), on=['celltype', 'marker'])
    expected.to_csv(OUT / 'EXPECTED_MARKER_AUC.csv', index=False)
    pd.DataFrame(profiles).to_csv(OUT / 'MARKER_QUARTILES_PER_PATIENT.csv', index=False)
    heads, class_rows, confusion_rows = [], [], []
    input_hashes = {}
    def track(p):
        p = Path(p); input_hashes[str(p)] = file_sha256(p)
    for f in ['manifest.json', 'metadata.csv.gz', 'native.npy']:
        track(ROOT / 'data/real_inputs_v0_3_r2' / TASK / f)
    for f in ['values.npy', 'cell_ids.npy', 'COMPLETE.json']:
        track(ramp / f)
    fitrows = []
    mindex = md.set_index('cell_id')
    for method in METHODS:
        cfg = config(method)
        track(ROOT / f'config/formal_v0_5/{TASK}__{method}.json')
        for j in primary_jobs(method):
            p = Path(cfg['output']) / 'jobs' / j['id'] / 'RESULT.json'
            track(p)
            result = json.loads(p.read_text())
            if method == 'ramp':
                lp = p.parent / 'ledger.json'; track(lp)
                ledger = json.loads(lp.read_text())
                for role in ['fit', 'validation']:
                    block = mindex.loc[ledger[role]]
                    for ct in all_classes:
                        sub = block[block.celltype.eq(ct)]
                        fitrows.append(dict(source=j['source'], target=j['target'], family=j['family'], seed=j['seed'],
                            role=role, celltype=ct, cells=len(sub), patients_or_units=sub.patient_id.nunique(),
                            split_mode=j['split_mode'], identity_level=';'.join(sorted(block.identity_level.unique()))))
            score = result['score']
            assert score['classes'] == all_classes
            cm = np.asarray(score['confusion_matrix'])
            tp, truth, pred = np.diag(cm), cm.sum(1), cm.sum(0)
            f1 = np.divide(2 * tp, truth + pred, out=np.zeros(len(tp)), where=truth+pred > 0)
            recall = np.divide(tp, truth, out=np.zeros(len(tp)), where=truth > 0)
            assert np.isclose(f1.mean(), score['macro_f1'], atol=1e-12)
            key = dict(method=method, source=j['source'], target=j['target'], family=j['family'], seed=j['seed'])
            heads.append(dict(**key, source_validation_f1=result['selection']['source_validation_score'],
                              target_f1=score['macro_f1'], target_ba=score['balanced_accuracy'],
                              target_accuracy=score['accuracy'], split_mode=j['split_mode']))
            for i, ct in enumerate(all_classes):
                class_rows.append(dict(**key, celltype=ct, f1=f1[i], recall=recall[i],
                    truth_fraction=truth[i], predicted_fraction=pred[i]))
                for k, predicted in enumerate(all_classes):
                    confusion_rows.append(dict(**key, truth=ct, predicted=predicted,
                        conditional_fraction=float(cm[i,k] / truth[i]) if truth[i] else np.nan))
    pd.DataFrame(heads).to_csv(OUT / 'OLD_HEAD_SCORES_PER_JOB.csv', index=False)
    pd.DataFrame(heads).groupby(['method', 'source', 'target'])[
        ['source_validation_f1', 'target_f1', 'target_ba', 'target_accuracy']].mean().to_csv(OUT / 'OLD_HEAD_BY_DIRECTION.csv')
    pd.DataFrame(class_rows).groupby(['method', 'celltype'])[
        ['f1', 'recall', 'truth_fraction', 'predicted_fraction']].mean().to_csv(OUT / 'OLD_CLASS_MEANS.csv')
    pd.DataFrame(class_rows).groupby(['method', 'source', 'target', 'celltype'])[
        ['f1', 'recall', 'truth_fraction', 'predicted_fraction']].mean().to_csv(OUT / 'OLD_CLASS_BY_DIRECTION.csv')
    pd.DataFrame(confusion_rows).groupby(['method', 'source', 'target', 'truth', 'predicted']).conditional_fraction.mean().to_csv(
        OUT / 'OLD_CONFUSION_BY_DIRECTION.csv')
    pd.DataFrame(fitrows).to_csv(OUT / 'SUPERVISION_COUNTS.csv', index=False)
    pivot = auc.pivot(index=['dataset', 'patient', 'celltype', 'marker'], columns='representation', values='auc')
    save(OUT / 'DIAGNOSIS_CHECKS.json', dict(old_jobs=len(heads), all_original_f1_reproduced=True,
        no_refit=True, original_inputs_read_only=True,
        maximum_native_vs_ramp_patient_auc_difference=float((pivot.native-pivot.ramp).abs().max()),
        script_sha256=file_sha256(__file__)))
    assert all(file_sha256(p) == h for p,h in input_hashes.items())
    save(OUT / 'INPUT_HASHES.json', input_hashes)
    print(expected[expected.representation.eq('native')][['dataset', 'celltype', 'marker', 'mean_patient_auc', 'patients']].to_string(index=False))


if __name__ == '__main__':
    from pathlib import Path
    main()
