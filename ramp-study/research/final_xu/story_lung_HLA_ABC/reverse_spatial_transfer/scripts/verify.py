"""Independent identity, kNN reconstruction, donor summaries, and immutable input checks."""
import runtime as rt
from runtime import BASE
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from PIL import Image
OUT=BASE/'results'

def main():
    design=json.loads((OUT/'DESIGN.json').read_text());association=json.loads((OUT/'ASSOCIATION_INPUTS.json').read_text());freeze=json.loads((OUT/'PREDICTION_FREEZE.json').read_text())
    for record in [design,association]:
        for key,h in record['input_sha256'].items():assert rt.sha(record['input_paths'][key])==h,key
    for name,h in freeze['output_sha256'].items():assert rt.sha(OUT/name)==h,name
    q=pd.read_csv(OUT/'CORDS_SPATIAL_STATES.csv.gz',dtype={'patient_id':str},float_precision='round_trip')
    sm=pd.read_csv(OUT/'SENOSAIN_REVERSE_PREDICTIONS.csv.gz',dtype={'patient_id':str},float_precision='round_trip')
    measured=pd.read_csv(OUT/'SENOSAIN_MEASURED_HELDOUT_PROTEINS.csv.gz',dtype={'patient_id':str},float_precision='round_trip')
    a=pd.read_csv(design['input_paths']['anchors'],dtype={'patient_id':str},low_memory=False)
    z=np.load(OUT/'REVERSE_NEIGHBORS.npz');ci=z['source_reference_index'];si=z['target_reference_index']
    assert np.array_equal(sm.source_row_id,a.iloc[si].source_row_id);assert np.array_equal(sm.source_row_id,measured.source_row_id)
    loc=pd.Index(q.source_row_id).get_indexer(a.iloc[ci].source_row_id);assert (loc>=0).all()
    names=[c.removeprefix('k30__') for c in sm if c.startswith('k30__')];train=q.iloc[loc][names].to_numpy()
    sampled=np.random.default_rng(20260909).choice(len(sm),31,replace=False);err=0;distance_error=0
    x=np.load(design['input_paths']['aligned']);scaled=(x-z['scale_mean'])/z['scale_scale']
    for target in sampled:
        ix=z['indices'][target];d=z['distances'][target]
        expected_distance=np.sqrt(np.sum((scaled[ci[ix]]-scaled[si[target]])**2,axis=1))
        distance_error=max(distance_error,float(np.max(np.abs(expected_distance-d))))
        for k in [30,100]:
            w=np.exp(-(d[:k]/max(d[k-1],1e-8))**2)
            for j,name in enumerate(names):
                yy=train[ix[:k],j];ok=np.isfinite(yy);expected=np.average(yy[ok],weights=w[ok])
                err=max(err,abs(expected-sm.iloc[target][f'k{k}__{name}']))
    assert err<1e-12 and distance_error<1e-10
    for k in [30,100]:assert np.allclose(sm[[f'k{k}__niche__N{i}' for i in range(1,7)]].sum(axis=1),1)
    raw=np.load(association['input_paths']['raw_proteins'],mmap_mode='r');markers=pd.read_csv(association['input_paths']['marker_names']).marker.tolist()
    direct=raw[sm.source_row_id.to_numpy(int),markers.index('HLA_ABC')]
    # Pandas writes float32 using a shortest round-trip decimal; compare at source precision.
    raw_serialization_error=float(np.max(np.abs(direct.astype(float)-measured.measured__HLA_ABC.to_numpy())))
    assert np.array_equal(direct,measured.measured__HLA_ABC.to_numpy().astype(direct.dtype))
    rankerr=0;contrast_error=0
    pairs=pd.read_csv(OUT/'SENOSAIN_DONOR_PROTEIN_CONTRASTS.csv.gz',dtype={'patient_id':str},float_precision='round_trip')
    reference=pairs.query("population=='paper_mesenchymal' and k==30 and state=='r40__CAF_fraction' and marker=='HLA_ABC'").set_index('patient_id')
    for pid,ix in sm.groupby('patient_id').indices.items():
        y=rankdata(direct[ix])/(len(ix)+1);rankerr=max(rankerr,float(np.max(np.abs(y-measured.iloc[ix].rank__HLA_ABC))))
        mask=sm.iloc[ix].paper_mesenchymal_gate.to_numpy(bool)
        if mask.sum()<15:continue
        xx=sm.iloc[ix].k30__r40__CAF_fraction.to_numpy()[mask];yy=y[mask];l,h=np.quantile(xx,[1/3,2/3]);delta=yy[xx>=h].mean()-yy[xx<=l].mean()
        contrast_error=max(contrast_error,abs(delta-reference.loc[pid,'difference']))
    assert rankerr<1e-12 and contrast_error<1e-12
    clinical=pd.read_csv(OUT/'SURVIVAL_INPUTS.csv',dtype={'patient_id':str},float_precision='round_trip')
    original=pd.read_csv(association['input_paths']['clinical'],dtype={'patient_id':str},float_precision='round_trip').query("config=='RAMP_k30'").set_index('patient_id')
    assert set(clinical.patient_id)==set(original.index) and len(clinical)==280 and clinical.event.sum()==96
    for col in ['all_mean','all_high','time_years','event','age','sex','Stage','caf_fraction','old_global_CAF_score','vcaf_fraction']:
        assert np.array_equal(clinical[col],original.loc[clinical.patient_id,col]),col
    coxerr=pd.read_csv(OUT/'COX_REPRODUCTION.csv').reference_coef_error.iloc[0];assert coxerr<1e-10
    issues=pd.read_csv(OUT/'MODEL_ISSUES.csv');assert issues.empty
    figures=[]
    for p in sorted((OUT/'figures').glob('*.png')):
        with Image.open(p) as im:
            im.load();arr=np.asarray(im.convert('RGB'));assert im.width>1000 and im.height>500 and arr.std()>5
            figures.append(dict(file=p.name,width=im.width,height=im.height,sha256=rt.sha(p)))
        assert p.with_suffix('.svg').exists()
    assert len(figures)==5
    identity=json.loads((OUT/'IDENTITY_VERIFICATION.json').read_text())
    rt.save(OUT/'VERIFICATION.json',dict(status='pass',finished=rt.now(),source_hashes_unchanged=True,frozen_spatial_and_reverse_outputs_unchanged=True,
        n_source=28000,n_target=28000,n_cords_vcaf=int(q.cell_type.eq('vCAF').sum()),n_dfs=280,dfs_events=96,
        independent_neighbor_distance_max_error=distance_error,independent_knn_prediction_max_error=err,
        measured_HLA_exact_at_original_dtype=str(direct.dtype),raw_HLA_CSV_decimal_error=raw_serialization_error,
        independent_measured_HLA_rank_max_error=rankerr,independent_donor_contrast_max_error=contrast_error,
        original_clinical_and_HLA_inputs_exact=True,original_Cox_coefficient_max_error=float(coxerr),model_issues=0,
        original_gate_membership_exact=identity['original_gate_membership_exact'],figures=figures,
        visual_review='See FIGURE_QA.json for manual image inspection status.'))
    manifest={str(p.relative_to(BASE)):rt.sha(p) for p in sorted(BASE.rglob('*')) if p.is_file() and not any(x in p.relative_to(BASE).parts for x in ['cache','tmp']) and p.name!='MANIFEST.json' and p.suffix!='.log'}
    rt.save(OUT/'MANIFEST.json',manifest)
    print('PASS: immutable inputs, reverse KNN, measured HLA ranks, donor contrasts, clinical cohort, original Cox, and five figure pairs.')

if __name__=='__main__':main()
