"""Count feasible patient-disjoint splits; do not choose or apply a split."""
from check import ROOT, OUT, PILOT, save
import itertools
import json
import numpy as np
import pandas as pd

def main():
    results=[]
    for task in ['melanoma_story','hnscc_story','lung_story']:
        classes=json.loads((ROOT/f'data/real_inputs_v0_3_r2/{task}/manifest.json').read_text())['classes']
        md=pd.read_csv(PILOT/f'inputs/{task}/metadata.csv.gz',dtype=str,keep_default_na=False)
        md=md[md.celltype.isin(classes)]
        for ds,g in md.groupby('dataset'):
            assert g.identity_level.eq('patient').all()
            counts=pd.crosstab(g.patient_id,g.celltype).reindex(columns=classes,fill_value=0)
            n=len(counts);k=min(n-1,max(1,round(n*.2)));x=counts.to_numpy();tot=x.sum(0)
            tested=0;feasible={v:0 for v in [5,10,20]}
            # All combinations are examined using SOURCE class counts only.
            for subset in itertools.combinations(range(n),k):
                tested+=1;vc=x[list(subset)].sum(0);fc=tot-vc
                for v in feasible:
                    if (fc>=20).all() and (vc>=v).all():feasible[v]+=1
            for v,c in feasible.items():
                results.append(dict(task=task,source=ds,patients=n,validation_patients=k,
                    fit_floor=20,validation_floor=v,possible_patient_splits=tested,
                    feasible_splits=c,feasible_fraction=c/tested))
            print(task,ds,tested,feasible,flush=True)
    pd.DataFrame(results).to_csv(OUT/'POSSIBLE_PATIENT_SPLITS.csv',index=False)
    save('SPLIT_ENUMERATION_CHECKS.json',dict(no_splits_selected_or_applied=True,
        no_target_labels_or_prediction_scores_read=True,source_label_counts_only=True,
        preserve_current_donor_subset_and_validation_donor_count=True,
        note='Diagnostic thresholds only; feasible counts do not establish statistical adequacy.'))

if __name__=='__main__':main()
