"""Rebuild eligibility from all 527 patients and join frozen molecular scores."""
import package_runtime as rt
from package_runtime import BASE
import json
import numpy as np
import pandas as pd

OUT=BASE/'results'
CORDS=BASE.parent/'data_cords'
ROOT=BASE.parents[1]
CLINICAL=ROOT/'story_test_lung/2_dataset/outputs/common8/results/locked_scores_with_outcomes.csv'
CONFIGS=['RAMP_k30','RAMP_k100','baseline_k30','baseline_k100']


def main():
    paths=[CORDS/'results/ALL_PATIENT_SCORES.csv',CORDS/'results/HIGH_STATE_THRESHOLDS.csv',
           CORDS/'results/PREDICTION_FREEZE.json',CORDS/'results/REPRODUCTION.json',CLINICAL,
           BASE/'reference/PREVIOUS_PATIENT_INPUTS.csv',BASE/'reference/LEGACY_PATIENT_INPUTS.csv']
    hashes={str(p):rt.sha(p) for p in paths}
    assert json.loads(paths[3].read_text())['status']=='PASS'
    scores=pd.read_csv(paths[0],dtype={'patient_id':str},float_precision='round_trip')
    assert len(scores)==2108 and scores.groupby('config').patient_id.nunique().eq(527).all()
    clinical=pd.read_csv(CLINICAL,dtype={'patient_id':str})
    cols=['patient_id','age','sex','stage_group','Stage','dfs_days','dfs_event','caf_fraction','median_score']
    clinical=clinical[cols].copy();assert clinical.patient_id.is_unique
    clinical.to_csv(BASE/'data/CLINICAL_INPUTS.csv',index=False)
    clinical=clinical.rename(columns={'median_score':'old_global_CAF_score'})
    legacy=pd.read_csv(BASE/'reference/LEGACY_PATIENT_INPUTS.csv',dtype={'patient_id':str})
    previous=pd.read_csv(BASE/'reference/PREVIOUS_PATIENT_INPUTS.csv',dtype={'patient_id':str},float_precision='round_trip')
    frames=[];allframes=[];flow=[];comparisons=[]
    for config in CONFIGS:
        p=scores[scores.config.eq(config)].merge(clinical,on='patient_id',how='left',validate='one_to_one')
        p['early_stage_ok']=p.stage_group.eq('I-II')
        p['vcaf_count_ok']=p.n_vcaf.ge(15)
        p['caf_count_ok']=p.n_full_caf.ge(50)
        p['dfs_ok']=np.isfinite(p[['dfs_days','dfs_event']]).all(axis=1)&p.dfs_days.gt(0)&p.dfs_event.isin([0,1])
        p['covariates_ok']=np.isfinite(p[['age','sex','caf_fraction','old_global_CAF_score']]).all(axis=1)
        keep=pd.Series(True,index=p.index)
        for name,rule in [('all_patients',keep.copy()),('early_stage_I_II',p.early_stage_ok),
                          ('at_least_15_vCAF',p.vcaf_count_ok),('at_least_50_CAF',p.caf_count_ok),
                          ('valid_DFS',p.dfs_ok),('complete_existing_covariates',p.covariates_ok)]:
            before=int(keep.sum());keep &= rule
            flow.append({'config':config,'step':name,'n_patients':int(keep.sum()),
                         'excluded_at_step':before-int(keep.sum()),'known_events':int(p.loc[keep,'dfs_event'].sum())})
        p['eligible']=keep;p['time_years']=p.dfs_days/365.25;p['event']=p.dfs_event
        allframes.append(p)
        d=p[p.eligible].copy();d['event']=d.event.astype(int)
        assert len(d)==280 and d.event.sum()==96 and d.Stage.notna().all() and d.sex.isin([1,2]).all()
        assert np.isfinite(d[['all_mean','all_high','vcaf_fraction','current_global_CAF_median']]).all().all()
        kk=int(config.split('_k')[1])
        old=legacy[legacy.k.eq(kk)][['patient_id','all_mean','all_high']].rename(columns={'all_mean':'old_mean','all_high':'old_high'})
        d=d.merge(old,on='patient_id',validate='one_to_one')
        prior=previous[previous.config.eq(config)].set_index('patient_id')
        assert set(d.patient_id)==set(prior.index)
        prior=prior.loc[d.patient_id]
        errors={c:float(np.abs(d[c].to_numpy()-prior[c].to_numpy()).max()) for c in
                ['all_mean','all_high','n_vcaf','n_full_caf','vcaf_fraction','current_global_CAF_median',
                 'old_global_CAF_score','time_years','event','age','sex','Stage','caf_fraction','old_mean','old_high']}
        assert max(errors.values())<1e-12,errors
        assert np.array_equal(d.reference_donor,prior.reference_donor)
        comparisons.append({'config':config,'n_patients':280,'events':96,'reference_donors':int(d.reference_donor.sum()),
                            'maximum_score_covariate_outcome_error':max(errors.values())})
        frames.append(d)
    pd.concat(frames,ignore_index=True).to_csv(OUT/'PATIENT_INPUTS.csv',index=False)
    pd.concat(allframes,ignore_index=True).to_csv(OUT/'ALL_PATIENTS.csv',index=False)
    pd.DataFrame(flow).to_csv(OUT/'COHORT_FLOW.csv',index=False)
    pd.DataFrame(comparisons).to_csv(OUT/'COHORT_REPRODUCTION.csv',index=False)
    rt.save_json(OUT/'DESIGN.json',{'time_utc':rt.now(),'primary_config':'RAMP_k30','primary_state':'mean',
        'companion_state':'high-state vCAF fraction','reference_score':'source-patient relative HLA-ABC rank',
        'n_initial_patients':527,'eligibility':'>=15 author vCAFs; >=50 full CAFs; provider stage_group I-II; positive DFS; event 0/1; complete age, sex, original CAF fraction and global CAF score',
        'T_or_SMA_requirements':False,'clinical_threshold_optimization':False,'cell_thresholds':json.loads((CORDS/'results/PREDICTION_FREEZE.json').read_text())['thresholds'],
        'KM_split':'patient median for each state, >=median Higher','continuous_Cox':'per 1 SD, age remains in years',
        'adjustment':['age','sex','provider Stage (categorical code)','original CAF fraction','original global CAF score','vCAF/all-CAF fraction'],
        'sensitivity':['k100','same-input baseline','updated global CAF median','exclude 22 reference donors (258 remaining)'],
        'interpretation':'Fixed reproduction of previously explored association; not causal or independent survival validation',
        'input_sha256':hashes})
    assert all(rt.sha(p)==h for p,h in hashes.items())
    rt.save_json(OUT/'PREPARATION_VERIFICATION.json',{'status':'PASS','time_utc':rt.now(),'n_initial_patients':527,
        'n_survival_patients':280,'events':96,'n_configurations':4,'same_patients_scores_covariates_outcomes':True,
        'T_or_SMA_fields_used':False,'cohort_rebuilt_from_all_patients':True,'comparisons':comparisons})
    print(pd.DataFrame(flow).query("config == 'RAMP_k30'").to_string(index=False),flush=True)


if __name__=='__main__':main()
