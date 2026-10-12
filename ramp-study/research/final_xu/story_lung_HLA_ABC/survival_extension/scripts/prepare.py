"""Aggregate frozen cell predictions without changing the survival cohort."""
import runtime as rt
from runtime import BASE
import pandas as pd
import numpy as np
OUT=BASE/'results';OUT.mkdir(exist_ok=True)
PATHS={'cells':BASE.parent/'data_cords/results/ALL_CORDS_PREDICTIONS.csv.gz',
       'cohort':BASE.parent/'survival/results/PATIENT_INPUTS.csv',
       'cox_reference':BASE.parent/'survival/results/COX_TERMS.csv'}
def main():
    hashes={k:rt.sha(p) for k,p in PATHS.items()}
    rt.save(OUT/'DESIGN.json',dict(started_at_utc=rt.now(),input_paths={k:str(p) for k,p in PATHS.items()},input_sha256=hashes,
        scope='Frozen RAMP k30 primary, k100 companion. Existing 280 patients/96 DFS events. No new imputation, no survival cutoff search.',
        questions=['Does HLA state add information beyond vCAF abundance?','Is the association concentrated in vCAF or shared by other compartments?','How does adjusted risk vary with continuous state?'],
        abundance='Original n_vCAF/n_all_CAF; continuous adjustment and patient-median split for a descriptive 2x2 analysis.',
        dose_response='Linear adjusted Cox; fixed natural cubic spline with internal knot at state median and boundary knots at 10th/90th percentiles; fixed patient score quartiles. No knot/cutoff optimization.',
        compartments=['vCAF','Other CAF','All CAF','Tumour','Immune','Vessel','All non-vCAF cells'],
        specificity='At least 15 cells of each evaluated compartment; compare on identical patients, same covariates. Separate models and joint vCAF+compartment models. Different P values alone do not prove specificity.',
        adjustment='Age, sex, provider categorical Stage, original CAF fraction, original global CAF score, vCAF/CAF abundance. Mean HLA score standardized per analysis cohort.',
        exploratory=True))
    inp=pd.read_csv(PATHS['cohort'],float_precision='round_trip')
    inp=inp[inp.config.isin(['RAMP_k30','RAMP_k100'])].copy();assert len(inp)==560
    inp.to_csv(OUT/'PATIENT_INPUTS.csv',index=False)
    cols=['patient_id','source_row_id','cell_type','cell_category','RAMP_k30','RAMP_k100']
    d=pd.read_csv(PATHS['cells'],usecols=cols,float_precision='round_trip');assert len(d)==2849385
    masks={'vCAF':d.cell_type.eq('vCAF'),'Other CAF':d.cell_category.eq('Fibroblast')&~d.cell_type.eq('vCAF'),
           'All CAF':d.cell_category.eq('Fibroblast'),'Tumour':d.cell_category.eq('Tumour'),
           'Immune':d.cell_category.isin(['Immune','T cell']),'Vessel':d.cell_category.eq('vessel'),
           'All non-vCAF cells':~d.cell_type.eq('vCAF')}
    rows=[]
    for label,mask in masks.items():
        a=d[mask].groupby('patient_id').agg(n_cells=('source_row_id','size'),RAMP_k30=('RAMP_k30','mean'),RAMP_k100=('RAMP_k100','mean')).reset_index()
        a['compartment']=label;rows.append(a)
    a=pd.concat(rows,ignore_index=True);a.to_csv(OUT/'ALL_PATIENT_COMPARTMENT_SCORES.csv',index=False)
    errors={}
    for k in [30,100]:
        g=inp[inp.config.eq(f'RAMP_k{k}')].merge(a[a.compartment.eq('vCAF')],on='patient_id',validate='one_to_one')
        err=float(np.max(np.abs(g.all_mean-g[f'RAMP_k{k}'])));assert err<1e-12
        assert np.array_equal(g.n_cells,g.n_vcaf);errors[f'k{k}']=err
    assert all(rt.sha(PATHS[k])==h for k,h in hashes.items())
    rt.save(OUT/'PREPARATION_VERIFICATION.json',dict(status='pass',source_files_unchanged=True,vCAF_mean_max_error=errors,patients=280,events=96))
    print('Prepared frozen cohort and seven compartment scores.',flush=True)
if __name__=='__main__':main()
