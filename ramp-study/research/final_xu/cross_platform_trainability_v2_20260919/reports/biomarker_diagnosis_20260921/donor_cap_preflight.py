"""Assess real-donor subsampling using unmodified RAMP sampling functions.

No alignments, model fits, score-based selection, donor relabeling or data
replacement. Original cells of selected donors are retained in original order.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT.parent/'algorithm'))
from ramp import ramp_ver2 as alg

TASKS=['bone_marrow','tnbc','melanoma','lung_cancer','lung_story','melanoma_story','hnscc_story']
CAPS=[20,24,25,30]
SEEDS=[20260921,20260922,20260923]


def main():
    records=[];selections=[];summary=[]
    for task in TASKS:
        folder=ROOT/f'representations/formal_v0_5/{task}/complete/ramp'
        req=json.loads((folder/'request.json').read_text())
        full=pd.read_csv(folder/'identity.csv',dtype=str)
        for seed in SEEDS:
            orders={}
            for ds in req['datasets']:
                units=np.asarray(sorted(full.loc[full.dataset.eq(ds),'unit_key'].unique()))
                rng=np.random.default_rng(alg.stable_seed('donor-cap-preflight',task,ds,seed))
                orders[ds]=rng.permutation(units)
            for cap in CAPS:
                chosen={ds:set(units[:cap]) for ds,units in orders.items()}
                keep=np.array([u in chosen[d] for d,u in zip(full.dataset,full.unit_key)])
                md=full.loc[keep].reset_index(drop=True)
                for ds,units in chosen.items():
                    for u in sorted(units):selections.append(dict(task=task,seed=seed,cap=cap,dataset=ds,unit_key=u))
                n,audit=alg.adaptive_pair_fit_size(md,req['datasets'])
                rowdigest=hashlib.sha256(np.flatnonzero(keep).astype('<i8').tobytes()).hexdigest()
                for ds in req['datasets']:
                    ix=alg.unit_balanced_fit_indices(md,ds,n)
                    for ref in req['datasets']:
                        if ref==ds:continue
                        sample=md.iloc[ix].copy()
                        sample['role']=alg.deterministic_roles(req['family'],ds,ref,ix)
                        for unit,g in sample.groupby('unit_key'):
                            count=int(g.role.eq('curve_fit').sum())
                            records.append(dict(task=task,seed=seed,cap=cap,dataset=ds,reference=ref,unit_key=unit,
                                retained_dataset_cells=int(md.dataset.eq(ds).sum()),pair_fit=n,sampled_cells=len(g),
                                curve_fit_cells=count,required_curve_cells=max(40,4*alg.CURVE_MIN_BIN_CELLS),
                                numeric_support_passes=count>=max(40,4*alg.CURVE_MIN_BIN_CELLS),
                                ramp_family=req['family'],selected_original_rows_sha256=rowdigest))
    frame=pd.DataFrame(records)
    frame.to_csv(HERE/'DONOR_CAP_PER_UNIT.csv',index=False)
    pd.DataFrame(selections).to_csv(HERE/'DONOR_CAP_SELECTIONS.csv',index=False)
    for (task,cap,ds),g in frame.groupby(['task','cap','dataset']):
        by=g.groupby(['seed','unit_key']).numeric_support_passes.agg(['all','any'])
        summary.append(dict(task=task,cap=cap,dataset=ds,draws=g.seed.nunique(),donors_per_draw=g.groupby('seed').unit_key.nunique().min(),
            checked_donor_draws=len(by),donor_draws_with_all_directions_passing=int(by['all'].sum()),
            donor_draws_with_any_direction_passing=int(by['any'].sum()),
            minimum_curve_fit_cells=int(g.curve_fit_cells.min()),maximum_curve_fit_cells=int(g.curve_fit_cells.max())))
    sm=pd.DataFrame(summary);sm.to_csv(HERE/'DONOR_CAP_SUMMARY.csv',index=False)
    checks=dict(no_alignment_run=True,no_models_fit=True,no_scores_read=True,unit_names_unchanged=True,
        cell_values_unchanged=True,retained_original_cell_order=True,seed=42,selection_seeds=SEEDS,
        selection_scope='True original units/donors sampled without expression, labels or outcomes; all frozen cells of retained units kept.',
        limitation='Counts are necessary, not sufficient: ties, poor fit quality, strength and compatibility may still limit correction.',
        algorithm_sha256={name:hashlib.sha256((ROOT.parent/'algorithm/ramp'/name).read_bytes()).hexdigest() for name in ['ramp.py','ramp_ver2.py']})
    (HERE/'DONOR_CAP_CHECKS.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(sm[sm.task.isin(['hnscc_story','melanoma_story','lung_story'])].to_string(index=False))
    g=sm[sm.cap.eq(20)]
    print('Cap20 all seven tasks all directions support:',bool((g.checked_donor_draws==g.donor_draws_with_all_directions_passing).all()))


if __name__=='__main__':main()
