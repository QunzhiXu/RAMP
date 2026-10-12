"""Prepare patient-level plotting tables from frozen scores and rank summaries."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'analysis/all_cd8_prediction'
OUT = HERE / 'data'
OUT.mkdir(exist_ok=True)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths = [SOURCE / name for name in [
        'patient_metrics.csv', 'outer_patient_predictions.csv.gz',
        'biological_interpretation/all_heldout_adt_effects_by_patient.csv',
        'citeseq_projection/engelbrecht_all_cd8_projection.csv.gz']]
    hashes = {str(p): sha(p) for p in paths}
    metrics, predictions, effects, query = [pd.read_csv(p) for p in paths]
    eligible = metrics.loc[metrics.evaluable_discrimination].copy()
    assert len(eligible) == 14
    records = []
    keys = ['AUROC', 'AUPRC', 'Precision', 'Recall']
    for row in eligible.itertuples():
        cells = predictions.loc[predictions.patient.eq(row.held_out_patient)]
        y = cells.target_30um.to_numpy(bool)
        score = cells.model_score.to_numpy(float)
        called = cells.called_top20.to_numpy(bool)
        expected = np.zeros(len(cells), dtype=bool)
        expected[np.argsort(-score, kind='mergesort')[:int(np.ceil(.2 * len(cells)))]] = True
        assert np.array_equal(expected, called)
        assert len(cells) == row.n_all_cd8 and y.sum() == row.n_target
        values = [roc_auc_score(y, score), average_precision_score(y, score),
                  float((called & y).sum() / called.sum()),
                  float((called & y).sum() / y.sum())]
        assert np.allclose(values, [row.AUROC, row.AUPRC, row.top20_precision, row.top20_recall])
        for metric, value in zip(keys, values):
            records.append(dict(patient=row.held_out_patient, metric=metric, value=value,
                                n_cd8=len(cells), n_target=int(y.sum()),
                                positive_call_rule='top 20% within whole held-out patient'))
    f = pd.DataFrame(records)
    f.to_csv(OUT / 'panel_F_four_metrics_by_patient.csv', index=False)
    fs = f.groupby('metric').value.agg(['count', 'mean', 'median', 'min', 'max']).reindex(keys)
    fs.to_csv(OUT / 'panel_F_metric_summary.csv')

    marker_map = {'CD11a-AB': 'CD11a', 'CD18-AB': 'CD18', 'CD2-AB': 'CD2', 'CD279-AB': 'PD-1'}
    selected = effects.loc[effects.normalization.eq('classical_clr') & effects.adt_feature.isin(marker_map)]
    assert len(selected) == 32
    records = []
    for row in selected.itertuples():
        local = query.loc[query.Patient_ID.eq(row.patient)]
        assert len(local) == row.n_cells
        assert int((local.within_patient_score_percentile >= .75).sum()) == row.n_high
        # Average-tie ranks divided by n+1 have overall mean exactly 0.5.
        # The frozen within-patient difference and group sizes uniquely recover
        # BOTH group means; these are not synthetic or cell-level observations.
        fraction = row.n_high / row.n_cells
        delta = row.top25_minus_rest75_rank
        higher = .5 + (1-fraction) * delta
        lower = .5 - fraction * delta
        assert np.isclose(higher-lower, delta)
        assert np.isclose(fraction*higher+(1-fraction)*lower, .5)
        assert 0 <= lower <= 1 and 0 <= higher <= 1
        for group, mean, n in [('Lower', lower, row.n_cells-row.n_high), ('High', higher, row.n_high)]:
            records.append(dict(patient=row.patient, marker=marker_map[row.adt_feature],
                                adt_feature=row.adt_feature, group=group,
                                mean_expression_rank=mean, n_group_cells=n,
                                n_patient_cd8=row.n_cells, normalization='classical_clr',
                                grouping='High: score rank >=0.75; Lower: remaining cells'))
    h = pd.DataFrame(records)
    assert len(h) == 64 and h.groupby(['marker','group']).size().eq(8).all()
    h.to_csv(OUT / 'panel_H_group_means_by_patient.csv', index=False)
    assert all(sha(Path(p)) == digest for p,digest in hashes.items())
    audit = dict(status='PASS', source_sha256=hashes, all_sources_unchanged=True,
                 models_refitted=False, F=dict(n_patients=14, points=56,
                 metrics=keys, metrics_recomputed_and_match_frozen_results=True,
                 call_threshold='top 20% in whole held-out patient',
                 summary=fs.reset_index().to_dict('records')),
                 H=dict(n_patients=8, markers=list(marker_map.values()), points=64,
                 units='patient mean of within-patient expression ranks after cell-level CLR',
                 grouping='top score quartile vs remaining cells, unchanged',
                 recovery_formula='High=0.5+(1-n_high/n)*difference; Lower=0.5-(n_high/n)*difference',
                 basis='Mean of average-tie ranks divided by n+1 is exactly 0.5',
                 group_counts_verified_against_query_cells=True))
    (OUT / 'DATA_QA.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(fs.to_string())
    print(h.groupby(['marker','group']).mean_expression_rank.agg(['count','mean']).to_string())


if __name__ == '__main__':
    main()
