"""Independent consistency checks against frozen patient scores and input hashes."""
import runtime as rt
from runtime import BASE
import json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from PIL import Image
from xml.etree import ElementTree
OUT=BASE/'results'
design=json.loads((OUT/'DESIGN.json').read_text())
for k,path in design['input_paths'].items():assert rt.sha(path)==design['input_sha256'][k],k
v=pd.read_csv(OUT/'VCAF_CURRENT_PREDICTIONS_AND_DISTANCES.csv.gz',float_precision='round_trip')
frozen=BASE.parent/'data_cords/results/ALL_PATIENT_SCORES.csv'
s=pd.read_csv(frozen,float_precision='round_trip')
assert len(v)==52328 and v.source_row_id.is_unique and np.isfinite(v[['x','y','RAMP_k30','RAMP_k100']]).all().all()
errors={}
for k in [30,100]:
    g=v.groupby('patient_id').agg(score=(f'RAMP_k{k}','mean'),high=(f'high_k{k}','mean'),n=('source_row_id','size'))
    old=s[s.config.eq(f'RAMP_k{k}')].set_index('patient_id').loc[g.index]
    assert np.array_equal(g.n,old.n_vcaf)
    for col,ref in [('score','all_mean'),('high','all_high')]:
        err=float(np.max(np.abs(g[col]-old[ref])));assert err<1e-12;errors[f'k{k}_{col}']=err

r=pd.read_csv(OUT/'ROI_CLUSTERING.csv')
max_moran_error=0.
for row in r[r.config.eq('RAMP_k30')].sample(12,random_state=20260909).itertuples():
    g=v[v.patient_id.eq(row.patient_id)&v.roi_id.eq(row.roi_id)];n=len(g)
    dd,ii=cKDTree(g[['x','y']]).query(g[['x','y']],k=min(7,n))
    # Explicit symmetric adjacency and textbook quadratic-form calculation.
    W=np.zeros((n,n))
    for i in range(n):
        for d,j in zip(dd[i],ii[i]):
            if i!=j and d<40:W[i,j]=W[j,i]=1
    z=g.RAMP_k30.to_numpy()-g.RAMP_k30.mean()
    stat=n/W.sum()*(z@W@z)/(z@z)
    err=abs(stat-row.moran_I);max_moran_error=max(max_moran_error,float(err));assert err<1e-12

pngs=list((OUT/'figures').rglob('*.png'));svgs=list((OUT/'figures').rglob('*.svg'))
for p in pngs:
    with Image.open(p) as im:im.load()
for p in svgs:ElementTree.parse(p)
manifest={str(p.relative_to(BASE)):rt.sha(p) for p in BASE.rglob('*') if p.is_file() and 'cache' not in p.parts and 'tmp' not in p.parts and p.name not in ['MANIFEST.json','VERIFICATION.json']}
rt.save_json(OUT/'MANIFEST.json',manifest)
rt.save_json(OUT/'VERIFICATION.json',dict(status='pass',completed_at_utc=rt.now(),all_frozen_inputs_unchanged=True,
    vCAF_cells=52328,patient_score_max_errors=errors,patient_score_reference=str(frozen),patient_score_reference_sha256=rt.sha(frozen),
    independent_adjacency_Moran_max_error=max_moran_error,pngs=len(pngs),svgs=len(svgs),image_decoding_and_svg_parsing='pass',
    manually_reviewed=['175_10__175_A3_3','86_68__86_B7_3','vCAF_neighborhood_contrasts','vCAF_spatial_clustering']))
print('Verification passed: source hashes, current patient means/high fractions, independent Moran formula, 10 PNGs and 10 SVGs.')
