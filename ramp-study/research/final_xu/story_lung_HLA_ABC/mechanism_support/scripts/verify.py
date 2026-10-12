"""Independent arithmetic checks and source immutability verification."""
import runtime as rt
from runtime import BASE, INPUTS
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from PIL import Image

OUT=BASE/'results'
def main():
    design=json.loads((OUT/'DESIGN.json').read_text())
    unchanged={key:rt.sha(path)==design['input_sha256'][key] for key,path in INPUTS.items()}
    assert all(unchanged.values()),unchanged
    mapping=pd.read_csv(BASE/'data/MARKER_MAPPING.csv')
    novel=mapping[~mapping.used_in_HLA_prediction].marker.tolist()
    assert novel==['CD73','IDO','PD1','TCF1','FOXP3']
    meas=pd.read_csv(BASE/'data/MEASURED_MARKERS.csv.gz',float_precision='round_trip')
    v=pd.read_csv(OUT/'VCAF_MEASURED_AND_PREDICTED.csv.gz',float_precision='round_trip')
    t=pd.read_csv(OUT/'T_CELL_NEAREST_VCAF.csv.gz',float_precision='round_trip')
    assert len(meas)==209301 and len(v)==52328 and len(t)==156973
    assert v.source_row_id.is_unique and t.source_row_id.is_unique
    assert set(v.source_row_id).isdisjoint(t.source_row_id)
    truth=meas.set_index('source_row_id')
    for d in [v,t]:
        for col in novel:
            assert np.array_equal(d[col].to_numpy(),truth.loc[d.source_row_id,col].to_numpy())
    r=pd.read_csv(OUT/'ROI_ASSOCIATIONS.csv',float_precision='round_trip')
    p=pd.read_csv(OUT/'PATIENT_ASSOCIATIONS.csv',float_precision='round_trip')
    checked=[]
    eligible=r[(r.config=='RAMP_k30')&r.endpoint.isin(['vCAF_CD73','vCAF_IDO','nearby_CD8_PD1','nearby_CD8_TCF1'])&r.rho.notna()]
    for ep,sub in eligible.groupby('endpoint'):
        for _,row in sub.sample(n=5,random_state=42).iterrows():
            if ep.startswith('vCAF_'):
                g=v[v.patient_id.eq(row.patient_id)&v.roi_id.eq(row.roi_id)]
                marker=ep.removeprefix('vCAF_');y=truth.loc[g.source_row_id,marker].to_numpy()
            else:
                g=t[t.patient_id.eq(row.patient_id)&t.roi_id.eq(row.roi_id)&t.cell_type.eq('CD8')&t.distance_vcaf.le(row.radius)]
                marker=ep.removeprefix('nearby_CD8_');y=truth.loc[g.source_row_id,marker].to_numpy()
            # Independent Pearson correlation of ranks from rejoined provider values.
            direct=float(np.corrcoef(rankdata(g.RAMP_k30),rankdata(y))[0,1])
            assert abs(direct-row.rho)<1e-12,(ep,direct,row.rho)
            checked.append({'endpoint':ep,'patient_id':row.patient_id,'roi_id':row.roi_id,'radius':int(row.radius),'absolute_error':abs(direct-row.rho)})
    # Check a paired contrast and patient aggregation directly from raw measurements.
    cchecks=[]
    for ep in ['vCAF_CD73','vCAF_IDO']:
        row=r[(r.endpoint==ep)&(r.config=='RAMP_k30')&r.difference.notna()].sample(n=1,random_state=8).iloc[0]
        g=v[v.patient_id.eq(row.patient_id)&v.roi_id.eq(row.roi_id)]
        values=rankdata(truth.loc[g.source_row_id,ep.removeprefix('vCAF_')])/(len(g)+1)
        low,high=np.quantile(g.RAMP_k30,[1/3,2/3]);diff=values[g.RAMP_k30>=high].mean()-values[g.RAMP_k30<=low].mean()
        assert abs(diff-row.difference)<1e-12
        patient=p[(p.endpoint==ep)&(p.config=='RAMP_k30')&(p.patient_id==row.patient_id)].iloc[0]
        rois=r[(r.endpoint==ep)&(r.config=='RAMP_k30')&(r.patient_id==row.patient_id)]
        assert abs(patient.rho-rois.rho.mean())<1e-12
        assert abs(patient.difference-rois.difference.mean())<1e-12
        cchecks.append({'endpoint':ep,'patient_id':row.patient_id,'roi_id':row.roi_id,'error':abs(diff-row.difference)})
    s=pd.read_csv(OUT/'ASSOCIATION_SUMMARY.csv')
    for (pop,config,metric),g in s[s.primary_endpoint].groupby(['population','config','metric']):
        assert len(g)==4
        a=g.p.to_numpy();order=np.argsort(a);expected=np.empty(4)
        expected[order]=np.minimum.accumulate((a[order]*4/np.arange(1,5))[::-1])[::-1].clip(0,1)
        assert np.max(np.abs(expected-g.q_primary))<1e-12
    clinical=pd.read_csv(OUT/'SURVIVAL_INPUTS.csv')
    assert len(clinical)==280 and clinical.patient_id.is_unique and clinical.event.sum()==96
    assert pd.read_csv(OUT/'COX_REPRODUCTION.csv').original_HLA_coef_error.max()<1e-10
    figs={}
    for png in sorted((OUT/'figures').glob('*.png')):
        assert png.with_suffix('.svg').is_file()
        with Image.open(png) as im:
            arr=np.asarray(im.convert('RGB'));assert arr.std()>10
            figs[png.name]={'size':list(im.size),'nonblank':True,'svg_exists':True}
    assert len(figs)==3
    rt.save(OUT/'VERIFICATION.json',{'status':'PASS','time_utc':rt.now(),'sources_unchanged':unchanged,'all_measured_values_exact':True,'independent_rank_correlations':checked,'independent_paired_contrasts':cchecks,'four_endpoint_BH_verified':True,'fixed_DFS_patients':280,'DFS_events':96,'original_HLA_Cox_reproduced':True,'figures':figs})
    files=[path for path in BASE.rglob('*') if path.is_file() and not any(part in {'cache','tmp','logs','__pycache__'} for part in path.relative_to(BASE).parts) and path.name!='MANIFEST.json']
    rt.save(OUT/'MANIFEST.json',{'time_utc':rt.now(),'input_sha256':design['input_sha256'],'outputs_sha256':{str(path.relative_to(BASE)):rt.sha(path) for path in files}})
    print('PASS: measured values, ranks, patient aggregation, neighborhoods, original survival, figures and frozen inputs.',flush=True)

if __name__=='__main__':main()
