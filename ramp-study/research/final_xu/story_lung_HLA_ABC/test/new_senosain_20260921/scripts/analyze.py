import runtime as rt
from runtime import BASE,STORY,CONFIGS
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import sys
OUT=BASE/'analysis'

def rho(a,b):return float(spearmanr(a,b).statistic)

def validation():
    assert (BASE/'validation/COMPLETE.json').exists()
    rt.status('Zheng evaluation')
    p=STORY/'validation/results/PREDICTIONS_AND_TRUTH.csv.gz';h=rt.hashes([p])
    meta=pd.read_csv(BASE/'validation/CELL_METADATA.csv.gz',dtype={'patient_id':str})
    new=np.load(BASE/'validation/PREDICTIONS.npy',mmap_mode='r')
    cols=['source_row_id','patient_id','core_id','observed_HLA_A','observed_HLA_A_patient_rank',*CONFIGS]
    old=pd.read_csv(p,usecols=cols,dtype={'patient_id':str},float_precision='round_trip').set_index('source_row_id').loc[meta.source_row_id]
    assert np.array_equal(old.patient_id,meta.patient_id) and np.array_equal(old.core_id,meta.core_id)
    truth=old.observed_HLA_A.to_numpy();rank=old.observed_HLA_A_patient_rank.to_numpy()
    rows=[];stability=[]
    for level,keys in [('patient',['patient_id']),('core',['patient_id','core_id']),('patient_celltype',['patient_id','cell_type'])]:
        for key,ix in meta.groupby(keys,sort=True).indices.items():
            if level=='patient_celltype' and len(ix)<20:continue
            if not isinstance(key,tuple):key=(key,)
            info={'level':level,**dict(zip(keys,key)),'n_cells':len(ix)}
            for j,c in enumerate(CONFIGS):
                for version,pred in [('old',old[c].to_numpy()[ix]),('new',new[ix,j])]:
                    rows.append(dict(**info,config=c,version=version,cell_spearman=rho(truth[ix],pred),relative_score_MAE=float(np.abs(rank[ix]-pred).mean())))
                stability.append(dict(**info,config=c,old_new_spearman=rho(old[c].to_numpy()[ix],new[ix,j]),mean_abs_change=float(np.abs(old[c].to_numpy()[ix]-new[ix,j]).mean())))
    metrics=pd.DataFrame(rows);metrics.to_csv(OUT/'ZHENG_METRICS.csv',index=False)
    pd.DataFrame(stability).to_csv(OUT/'ZHENG_PREDICTION_STABILITY.csv',index=False)
    summaries=[]
    for key,g in metrics[metrics.level.isin(['patient','core'])].groupby(['version','level','config']):
        v,l,c=key;summaries.append(dict(version=v,level=l,config=c,n_units=len(g),median_spearman=g.cell_spearman.median(),mean_spearman=g.cell_spearman.mean(),mean_relative_score_MAE=g.relative_score_MAE.mean(),min_spearman=g.cell_spearman.min(),max_spearman=g.cell_spearman.max()))
    summary=pd.DataFrame(summaries);summary.to_csv(OUT/'ZHENG_SUMMARY.csv',index=False)
    prior=pd.read_csv(STORY/'validation/results/SUMMARY_METRICS.csv');check=summary[summary.version.eq('old')].merge(prior,on=['level','config'],suffixes=('','_prior'))
    assert np.max(np.abs(check.median_spearman-check.median_spearman_prior))<1e-10
    sample=meta.core_id.eq('C_30');d=meta[sample].copy();d['measured_relative_score']=rank[sample]
    d['old_prediction']=old.RAMP_k30.to_numpy()[sample];d['new_prediction']=new[sample,0]
    d.to_csv(OUT/'C30_SPATIAL.csv.gz',index=False)
    rt.unchanged(h);rt.save(OUT/'VALIDATION_CHECKS.json',dict(truth_inputs_unchanged=True,old_summary_reproduced=True,prediction_frozen_before_truth=True,input_hashes=h))
    print(summary[summary.config.eq('RAMP_k30')].to_string(index=False),flush=True)

def scores():
    assert (BASE/'cords/COMPLETE.json').exists()
    rt.status('Cords molecular aggregation')
    meta=pd.read_csv(BASE/'cords/CELL_METADATA.csv.gz',dtype={'patient_id':str})
    pred=np.load(BASE/'cords/PREDICTIONS.npy',mmap_mode='r')
    samplepath=STORY/'data_cords/reference/THRESHOLD_CAF_CELL_IDS.csv.gz';h=rt.hashes([samplepath])
    sample=pd.read_csv(samplepath).source_row_id
    sf=meta.source_row_id.isin(set(sample));cf=meta.cell_category.eq('Fibroblast');vf=cf&meta.cell_type.eq('vCAF')
    assert sf.sum()==150024 and cf.sum()==530184 and vf.sum()==52328 and cf[sf].all()
    cuts={c:float(np.median(pred[sf,j])) for j,c in enumerate(CONFIGS)}
    counts=meta.loc[cf].groupby('patient_id').size().rename('n_full_caf')
    frames=[]
    for j,c in enumerate(CONFIGS):
        v=meta.loc[vf,['patient_id']].copy();v['value']=pred[vf,j];v['high']=v.value>=cuts[c]
        pv=v.groupby('patient_id').agg(n_vcaf=('value','size'),all_mean=('value','mean'),all_high=('high','mean'))
        d=counts.to_frame().join(pv);d['n_vcaf']=d.n_vcaf.fillna(0).astype(int);d['vcaf_fraction']=d.n_vcaf/d.n_full_caf
        sampled=pd.DataFrame({'patient_id':meta.loc[sf,'patient_id'].to_numpy(),'value':pred[sf,j]})
        d['current_global_CAF_median']=sampled.groupby('patient_id').value.median()
        refs=set(meta.loc[meta.reference_donor,'patient_id']);d['reference_donor']=d.index.isin(refs)
        d['config']=c;d['method']=c.split('_k')[0];d['k']=int(c.split('_k')[1]);frames.append(d.reset_index())
    new=pd.concat(frames,ignore_index=True);new.to_csv(OUT/'NEW_ALL_PATIENT_SCORES.csv',index=False)
    pd.DataFrame([dict(config=c,threshold=t) for c,t in cuts.items()]).to_csv(OUT/'NEW_HIGH_STATE_THRESHOLDS.csv',index=False)
    rt.save(OUT/'MOLECULAR_FREEZE.json',dict(time_utc=rt.now(),thresholds=cuts,score_sha256=rt.sha(OUT/'NEW_ALL_PATIENT_SCORES.csv'),clinical_outcomes_read=False))
    # Only now connect the clinical table; same eligibility and patients as before.
    clinicalpath=STORY/'survival/data/CLINICAL_INPUTS.csv';oldpath=STORY/'survival/results/PATIENT_INPUTS.csv'
    h.update(rt.hashes([clinicalpath,oldpath,STORY/'data_cords/results/ALL_PATIENT_SCORES.csv']))
    clinical=pd.read_csv(clinicalpath,dtype={'patient_id':str}).rename(columns={'median_score':'old_global_CAF_score'})
    old=pd.read_csv(oldpath,dtype={'patient_id':str},float_precision='round_trip');old['version']='old'
    inputs=[];flow=[]
    for c,g in new.groupby('config',sort=False):
        p=g.merge(clinical,on='patient_id',how='left',validate='one_to_one');keep=np.ones(len(p),bool)
        rules=[('all',keep.copy()),('I-II',p.stage_group.eq('I-II')),('vCAF>=15',p.n_vcaf.ge(15)),('CAF>=50',p.n_full_caf.ge(50)),('valid_DFS',p.dfs_days.gt(0)&p.dfs_event.isin([0,1])),('complete_covariates',np.isfinite(p[['age','sex','Stage','caf_fraction','old_global_CAF_score','current_global_CAF_median']]).all(axis=1))]
        for name,mask in rules:keep&=np.asarray(mask);flow.append(dict(config=c,step=name,n=int(keep.sum())))
        p=p[keep].copy();p['time_years']=p.dfs_days/365.25;p['event']=p.dfs_event.astype(int);p['eligible']=True;p['version']='new'
        ref=old[old.config.eq(c)].set_index('patient_id').loc[p.patient_id]
        assert len(p)==280 and p.event.sum()==96 and set(p.patient_id)==set(old[old.config.eq(c)].patient_id)
        for col in ['n_full_caf','n_vcaf','vcaf_fraction','time_years','event','age','sex','Stage','caf_fraction']:
            assert np.max(np.abs(p[col].to_numpy()-ref[col].to_numpy()))<1e-12,col
        assert np.isfinite(p[['all_mean','all_high','current_global_CAF_median']]).all().all()
        inputs.append(p)
    allinputs=pd.concat([old,*inputs],ignore_index=True);allinputs.to_csv(OUT/'SURVIVAL_INPUTS.csv',index=False)
    pd.DataFrame(flow).to_csv(OUT/'COHORT_FLOW.csv',index=False)
    rows=[];pairs=[]
    for c in CONFIGS:
        a=old[old.config.eq(c)].set_index('patient_id');b=allinputs[allinputs.version.eq('new')&allinputs.config.eq(c)].set_index('patient_id').loc[a.index]
        for state in ['mean','high']:
            av=a['all_'+state];bv=b['all_'+state];ag=av>=av.median();bg=bv>=bv.median()
            rows.append(dict(config=c,state=state,n=280,old_new_spearman=rho(av,bv),mean_abs_score_change=float(np.abs(av-bv).mean()),old_cut=av.median(),new_cut=bv.median(),changed_group=int((ag!=bg).sum()),same_group_fraction=float((ag==bg).mean())))
            pairs.extend(dict(patient_id=p,config=c,state=state,old_score=float(av.loc[p]),new_score=float(bv.loc[p]),old_high=bool(ag.loc[p]),new_high=bool(bg.loc[p])) for p in a.index)
    pd.DataFrame(rows).to_csv(OUT/'VCAF_SCORE_STABILITY.csv',index=False);pd.DataFrame(pairs).to_csv(OUT/'VCAF_PATIENT_SCORE_PAIRS.csv',index=False)
    rt.unchanged(h);rt.save(OUT/'COHORT_CHECKS.json',dict(n=280,events=96,same_patients=True,same_clinical_covariates=True,global_CAF_updated_per_version=True,source_files_unchanged=True,input_hashes=h))
    print(pd.DataFrame(rows).to_string(index=False),flush=True)

if __name__=='__main__':
    if sys.argv[1]=='validation':validation()
    elif sys.argv[1]=='scores':scores()
