"""Apply the existing value-evaluation protocol to fastMNN reconstructed values.

This is a separately labelled diagnostic extension. Existing results, algorithms,
source gates and data are read-only; target labels never fit source thresholds.
"""
from pathlib import Path
import sys, os, importlib.util, json, time
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=ROOT/'Main Figure/Figure 2/value_evaluation_20260914'
os.environ['MPLCONFIGDIR']=str(HERE/'.cache/matplotlib')
import run_audit as a
import numpy as np
import pandas as pd
spec=importlib.util.spec_from_file_location('frozen_values',OLD/'evaluate_values.py')
ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
os.environ['MPLCONFIGDIR']=str(HERE/'.cache/matplotlib')
ev.HERE=HERE/'value_evaluation'
ev.METHODS=['Raw','fastMNN reconstructed']
METRICS=['balanced_accuracy','f1','sensitivity','specificity','precision','auc']

def mean(d,keys,columns):
    return d.groupby(keys,observed=True,dropna=False)[columns].mean().reset_index()

def main():
    started=time.time()
    a.track(__file__);a.track(OLD/'evaluate_values.py');a.track(OLD/'PROTOCOL.md')
    allscores=[];allaucs=[];allemd=[];allscales=[]
    regression=[]
    for case in a.CASES:
        out=HERE/'value_evaluation'/case;out.mkdir(parents=True,exist_ok=True)
        meta=a.read_csv(HERE/case/'cell_metadata.csv.gz',dtype={'cell_id':str})
        label_col='celltype_joint_evaluation' if 'celltype_joint_evaluation' in meta else 'celltype'
        meta['eval_label']=meta[label_col].fillna('Unannotated').astype(str)
        markers=a.track(HERE/case/'markers.txt').read_text().splitlines()
        matrices={'Raw':np.load(a.track(ROOT/f'benchmark_results/{case}/ramp/alignment/input/raw.npy')),
                  'fastMNN reconstructed':np.load(a.track(HERE/case/'reconstructed.npy'))}
        ev.evaluate_gates(case,meta,markers,matrices,out)
        # Raw is rerun only for regression checking; distribution baseline stays frozen.
        ev.evaluate_distributions(case,meta,markers,{'fastMNN reconstructed':matrices['fastMNN reconstructed']},out)
        new=a.read_csv(out/'gate_target_unit_scores.csv')
        old=a.read_csv(OLD/'results'/case/'gate_target_unit_scores.csv')
        key=['benchmark','marker','source_dataset','target_dataset','target_unit']
        nr=new[new.method.eq('Raw')].sort_values(key).reset_index(drop=True)
        oo=old[old.method.eq('Raw')].sort_values(key).reset_index(drop=True)
        assert len(nr)==len(oo) and nr[key].equals(oo[key])
        error=float(np.nanmax(np.abs(nr[METRICS].to_numpy()-oo[METRICS].to_numpy())))
        assert error<1e-12,(case,error)
        regression.append(dict(case=case,raw_score_rows=len(nr),maximum_error=error))
        allscores.append(pd.concat([old,new[~new.method.eq('Raw')]],ignore_index=True))
        for filename,acc in [('within_unit_auc_preservation.csv',allaucs),('emd_unit_pairs.csv.gz',allemd),('marker_scales.csv',allscales)]:
            old=a.read_csv(OLD/'results'/case/filename);new=a.read_csv(out/filename)
            acc.append(pd.concat([old,new[new.method.eq('fastMNN reconstructed')]],ignore_index=True))
        print('Extended value metrics:',case,f'{time.time()-started:.1f}s',flush=True)
    out=HERE/'value_evaluation/summary';out.mkdir(exist_ok=True)
    scores=pd.concat(allscores,ignore_index=True);primary=scores[scores.primary_eligible].copy()
    counts=primary.groupby(['benchmark','marker','source_dataset','target_dataset','target_unit']).method.nunique()
    assert counts.eq(6).all()
    directions=mean(primary,['benchmark','marker','method','source_dataset','target_dataset'],METRICS)
    tasks=mean(directions,['benchmark','marker','method'],METRICS)
    assert len(tasks)==66
    case_scores=mean(tasks,['benchmark','method'],METRICS)
    overall=mean(case_scores,['method'],METRICS)
    for d,n in [(scores,'gate_target_unit_scores.csv.gz'),(directions,'gate_direction_means.csv'),
                (tasks,'gate_task_means.csv'),(case_scores,'gate_case_means.csv'),(overall,'gate_overall_equal_case.csv')]:
        d.to_csv(out/n,index=False)
    aucs=pd.concat(allaucs,ignore_index=True);aucs=aucs[aucs.primary_eligible].copy()
    aucs['absolute_auc_change']=aucs.auc_change.abs()
    aucs['auc_loss']=-aucs.auc_change.clip(upper=0)
    cols=['auc','raw_auc','auc_change','absolute_auc_change','auc_loss']
    ap=mean(aucs,['benchmark','marker','dataset','method'],cols)
    at=mean(ap,['benchmark','marker','method'],cols)
    ac=mean(at,['benchmark','method'],cols)
    ao=mean(ac,['method'],cols)
    ac.to_csv(out/'auc_preservation_case_means.csv',index=False)
    ao.to_csv(out/'auc_preservation_overall.csv',index=False)
    emd=pd.concat(allemd,ignore_index=True)
    scales=pd.concat(allscales,ignore_index=True)
    valid=scales.groupby(['benchmark','marker']).valid_scale.all().rename('all_valid').reset_index()
    scales.to_csv(out/'marker_scales_all.csv',index=False)
    valid.to_csv(out/'marker_scale_eligibility.csv',index=False)
    emd=emd.merge(valid,on=['benchmark','marker']);emd=emd[emd.all_valid]
    ep=mean(emd,['benchmark','method','marker','celltype','platform_left','platform_right'],['emd_normalized'])
    et=mean(ep,['benchmark','method','marker','celltype'],['emd_normalized'])
    em=mean(et,['benchmark','method','marker'],['emd_normalized'])
    ec=mean(em,['benchmark','method'],['emd_normalized'])
    eo=mean(ec,['method'],['emd_normalized'])
    for d,n in [(em,'emd_marker_means.csv'),(ec,'emd_case_means.csv'),(eo,'emd_overall_equal_case.csv')]:d.to_csv(out/n,index=False)
    overview=overall.merge(eo,on='method').merge(ao[['method','auc_change','auc_loss','absolute_auc_change']],on='method')
    overview.to_csv(out/'RESULTS_OVERVIEW.csv',index=False)
    pd.DataFrame(regression).to_csv(out/'raw_regression_checks.csv',index=False)
    assert all(a.sha(p)==h for p,h in a.READS.items())
    (out/'manifest.json').write_text(json.dumps(dict(status='PASS',protocol_source=str(OLD/'PROTOCOL.md'),
        new_method='fastMNN reconstructed',scope='Diagnostic extension of the existing four-benchmark marker-value evaluation',
        raw_regression=regression,source_gates='Same frozen source-only fitting; no target threshold tuning',
        existing_method_values_reused=True,inputs_unchanged=True,input_sha256=a.READS,elapsed_seconds=time.time()-started),indent=2)+'\n')
    print(overview.to_string(index=False),flush=True)

if __name__=='__main__':main()
