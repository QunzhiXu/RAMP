"""Predeclared source-only J4 budget calibration; no target predictions/scores."""
import sys
sys.dont_write_bytecode=True
import json,time
from dataclasses import replace
import numpy as np
from trainability.core import ROOT,write_json,require,file_sha256
from trainability.real_data import load_task
from trainability.adapters import fit_representation
from trainability.j4 import PretrainConfig,run_j4


def main():
    out=ROOT/'validation/formal_budget_v0_5';out.mkdir(exist_ok=False)
    spec=dict(tasks=['bone_marrow','lung_story'],methods=['unaligned','ramp'],seed=20260919,
        supervised_candidates=[500,1000,2000,3000],max_updates=3000,
        validation_every=100,pretrain_updates=300,probe_fit=500,probe_validation=512,
        selection='Smallest candidate with median remaining source-F1 gap <= .005 and maximum gap <= .02 across all matched RS/FT curves; otherwise maximum budget',
        median_gap_tolerance=.005,max_gap_tolerance=.02,target_scores_forbidden=True,
        full_registered_U=True,source_script_sha256=file_sha256(__file__))
    write_json(out/'SPECIFICATION.json',spec)
    rows=[];started=time.monotonic()
    for name in spec['tasks']:
        full=load_task(name);a,b,target=list(dict.fromkeys(full.pool.datasets))
        for method in spec['methods']:
            print('BUDGET',name,method,flush=True)
            rep=fit_representation(full.input(method),method,fitting_pool=full.pool,
                out=out/'representations'/name/method,input_contract=full.receipt,
                family=f'trainability_v0.3:{name}')
            task=replace(full,receipt=dict(full.receipt,pilot=True,budget_calibration=True))
            cfg=PretrainConfig(width=32,probe_fit=500,probe_validation=512,
                supervised_updates=3000,validation_every=100,pretrain_updates=300)
            r=run_j4(task,rep,source_a=a,source_b=b,target=target,cfg=cfg,seed=spec['seed'],
                out=out/'jobs'/name/method,source_only=True)
            for readout in ['random_supervised','A-only_finetuned','A+B_finetuned']:
                log=r['training'][readout];h=log['source_history']
                for budget in spec['supervised_candidates']:
                    best=max(x['source_macro_f1'] for x in h if x['updates']<=budget)
                    rows.append(dict(task=name,method=method,readout=readout,budget=budget,
                        best_source_score=best,remaining_source_gap=max(x['source_macro_f1'] for x in h)-best,
                        full_training_seconds=log['seconds']))
    summaries=[]
    for budget in spec['supervised_candidates']:
        gaps=[r['remaining_source_gap'] for r in rows if r['budget']==budget]
        summaries.append(dict(budget=budget,median_gap=float(np.median(gaps)),max_gap=max(gaps),
            passes_rule=np.median(gaps)<=.005 and max(gaps)<=.02))
    selected=next(r['budget'] for r in summaries if r['passes_rule'])
    write_json(out/'REPORT.json',dict(status='passed',selected_supervised_updates=selected,
        pretrain_updates=300,summaries=summaries,curves=rows,elapsed_seconds=time.monotonic()-started,
        target_scores_computed=False,formal_experiment_run=False,
        note='Finite source-validation budget calibration, not a convergence certificate. Fixed pretraining budget 300; selected common supervised budget applies to every formal J4 method.'))
    print('SELECTED',selected,flush=True)


if __name__=='__main__':main()
