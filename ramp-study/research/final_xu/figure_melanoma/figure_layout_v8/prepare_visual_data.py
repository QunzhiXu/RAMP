"""Recompute the two displayed metrics from the existing held-out predictions."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'analysis/all_cd8_prediction'
OUT=HERE/'data'
OUT.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    paths=[SOURCE/'patient_metrics.csv',SOURCE/'outer_patient_predictions.csv.gz']
    hashes={str(p):sha(p) for p in paths}
    m,pred=[pd.read_csv(p) for p in paths]
    eligible=m.loc[m.evaluable_discrimination].sort_values('held_out_patient')
    assert len(eligible)==14
    records=[];full=[]
    for row in eligible.itertuples():
        g=pred.loc[pred.patient.eq(row.held_out_patient)]
        y=g.target_30um.to_numpy(bool)
        score=g.model_score.to_numpy(float)
        calls=g.called_top20.to_numpy(bool)
        expected=np.zeros(len(g),bool)
        expected[np.argsort(-score,kind='mergesort')[:int(np.ceil(.2*len(g)))]]=True
        assert np.array_equal(calls,expected)
        precision=float(y[calls].mean());prevalence=float(y.mean())
        recall=float((y&calls).sum()/y.sum())
        auc=float(roc_auc_score(y,score));ap=float(average_precision_score(y,score))
        enrichment=precision/prevalence
        assert np.allclose([auc,ap,precision,recall,enrichment],
            [row.AUROC,row.AUPRC,row.top20_precision,row.top20_recall,row.top20_enrichment])
        assert len(g)==row.n_all_cd8 and y.sum()==row.n_target
        common=dict(patient=row.held_out_patient,n_cd8=len(g),n_target=int(y.sum()),n_called=int(calls.sum()),prevalence=prevalence)
        for metric,value in [('AUROC',auc),('Enrichment',enrichment)]:
            records.append(dict(**common,metric=metric,value=value))
        full.append(dict(**common,AUROC=auc,AUPRC=ap,precision_top20=precision,recall_top20=recall,enrichment_top20=enrichment))
    data=pd.DataFrame(records)
    data.to_csv(OUT/'panel_F_two_metrics_by_patient.csv',index=False)
    pd.DataFrame(full).to_csv(OUT/'panel_F_all_metrics_by_patient.csv',index=False)
    summary=data.groupby('metric').value.agg(['count','mean','median','min','max'])
    summary.to_csv(OUT/'panel_F_metric_summary.csv')
    assert all(sha(Path(p))==h for p,h in hashes.items())
    qa={'status':'PASS','source_sha256':hashes,'source_files_unchanged':True,
        'models_refitted':False,'threshold_changed':False,'top_fraction':.2,
        'selection_unit':'whole held-out patient, before any ROI display',
        'n_patients':14,'points_per_metric':14,'enrichment_definition':'precision_top20 / patient target prevalence',
        'reference_lines':{'AUROC':.5,'Enrichment':1.0},
        'independent_axes':True,'all_original_metrics_retained':'panel_F_all_metrics_by_patient.csv',
        'summary':summary.reset_index().to_dict('records')}
    (OUT/'DATA_QA.json').write_text(json.dumps(qa,indent=2)+'\n')
    print(json.dumps(qa,indent=2))

if __name__=='__main__':main()
