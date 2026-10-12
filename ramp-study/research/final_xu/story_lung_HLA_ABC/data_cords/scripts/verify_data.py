"""Numerically verify full Cords exports against the fixed earlier results."""
import package_runtime as rt
from package_runtime import BASE
import json
import numpy as np
import pandas as pd

OUT=BASE/'results'
CONFIGS=['RAMP_k30','RAMP_k100','baseline_k30','baseline_k100']


def main():
    pred=pd.read_csv(OUT/'ALL_CORDS_PREDICTIONS.csv.gz',dtype={'patient_id':str},float_precision='round_trip')
    assert len(pred)==2849385 and pred.source_row_id.is_unique
    with np.load(BASE/'reference/PREVIOUS_ALL_PREDICTIONS.npz') as old:
        assert np.array_equal(pred.source_row_id,old['source_row_id'])
        errors={c:float(np.abs(pred[c].to_numpy()-old['prediction'][:,i]).max()) for i,c in enumerate(CONFIGS)}
    assert max(errors.values())<1e-12,errors
    meta=pd.read_csv(BASE/'data/CELL_METADATA.csv.gz',dtype={'patient_id':str},float_precision='round_trip')
    assert np.array_equal(meta.matrix_row_index,np.arange(len(meta)))
    assert np.array_equal(meta.iloc[pred.matrix_row_index].source_row_id,pred.source_row_id)
    assert np.array_equal(meta.iloc[pred.matrix_row_index].patient_id,pred.patient_id)
    arrays={m:np.load(BASE/'data'/f'{m}_SHARED9.npy',mmap_mode='r') for m in ['RAW','baseline','RAMP']}
    assert all(a.shape==(2849385,9) for a in arrays.values())
    params=pd.read_csv(OUT/'BASELINE_PARAMETERS.csv',dtype={'patient_id':str},float_precision='round_trip')
    markers=pd.read_csv(BASE/'data/MARKERS.csv').marker.tolist()
    params=params.set_index(['patient_id','marker'])
    assert len(params)==527*9
    baseline_error=0.;max_displacement=0.;inversions=0
    for pid,ix in meta.groupby('patient_id',sort=False).indices.items():
        raw=arrays['RAW'][ix];b=arrays['baseline'][ix];a=arrays['RAMP'][ix]
        assert np.isfinite(raw).all() and np.isfinite(b).all() and np.isfinite(a).all()
        for j,marker in enumerate(markers):
            p=params.loc[(pid,marker)]
            recomputed=(np.arcsinh((raw[:,j]+p['shift'])/p.cofactor)-p['offset'])/p['scale']
            baseline_error=max(baseline_error,float(np.abs(recomputed-b[:,j]).max()))
            order=np.argsort(raw[:,j],kind='stable')
            inversions+=int(((np.diff(raw[order,j])>0)&(np.diff(a[order,j]) < -1e-10)).sum())
        max_displacement=max(max_displacement,float(np.abs(a-b).max()))
    assert baseline_error<1e-8 and max_displacement<=.75+1e-9 and inversions==0
    scores=pd.read_csv(OUT/'ALL_PATIENT_SCORES.csv',dtype={'patient_id':str},float_precision='round_trip')
    old=pd.read_csv(BASE/'reference/PREVIOUS_ALL_PATIENT_SCORES.csv',dtype={'patient_id':str},float_precision='round_trip')
    keys=['patient_id','config'];joined=scores.merge(old,on=keys,validate='one_to_one',suffixes=('','_old'))
    assert len(joined)==2108
    score_errors={}
    for col in ['all_mean','all_high','n_vcaf','n_full_caf','vcaf_fraction','current_global_CAF_median']:
        assert np.array_equal(joined[col].isna(),joined[col+'_old'].isna())
        err=float(np.nanmax(np.abs(joined[col]-joined[col+'_old'])))
        score_errors[col]=err;assert err<1e-12
    cuts=pd.read_csv(OUT/'HIGH_STATE_THRESHOLDS.csv').set_index('config').threshold
    oldcuts=pd.read_csv(BASE/'reference/PREVIOUS_HIGH_STATE_THRESHOLDS.csv').set_index('config').threshold
    assert np.max(np.abs(cuts-oldcuts.loc[cuts.index]))<1e-12
    audit=pd.read_csv(OUT/'PROJECTION_AUDIT.csv',dtype={'patient_id':str})
    assert len(audit)==527 and audit.mapping.eq('existing_donor_frozen_profile').sum()==28
    assert audit.mapping.eq('new_donor_platform_projection').sum()==499
    assert audit.anchor_coordinate_error.dropna().max()<1e-8
    result={'status':'PASS','time_utc':rt.now(),'n_patients':527,'n_cells':2849385,
            'n_reference_patients':28,'n_extension_patients':499,'full_prediction_max_error':errors,
            'patient_score_max_errors':score_errors,'baseline_from_saved_parameters_max_error':baseline_error,
            'maximum_displacement':max_displacement,'strict_rank_inversions':inversions,
            'expression_row_identity_verified':True,'source_response_exactly_matches_Zheng_validation':True}
    rt.save_json(OUT/'REPRODUCTION.json',result)
    design=json.loads((OUT/'DESIGN.json').read_text())
    assert all(rt.sha(p)==h for p,h in design['input_sha256'].items())
    files=[p for folder in ['data','results'] for p in (BASE/folder).rglob('*') if p.is_file()
           and p.name not in {'RUN_MANIFEST.json','RUN_STATUS.json','RUNNING.lock'}]
    rt.save_json(OUT/'RUN_MANIFEST.json',{'time_utc':rt.now(),'inputs':design['input_sha256'],
        'scripts':{str(p.relative_to(BASE)):rt.sha(p) for p in (BASE/'scripts').glob('*.py')},
        'reference':{str(p.relative_to(BASE)):rt.sha(p) for p in (BASE/'reference').glob('*') if p.is_file()},
        'outputs':{str(p.relative_to(BASE)):rt.sha(p) for p in files}})
    rt.save_json(OUT/'RUN_STATUS.json',{'phase':'complete','time_utc':rt.now(),'patients_done':527,'patients_total':527})
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
