import runtime as rt
from runtime import BASE,STORY,ROOT,MARKERS
import numpy as np
import pandas as pd
import json
from project import anchor_map
from frozen_maps import apply_maps

def main():
    checks={};h={}
    for p in [BASE/'alignment/COMPLETE.json',BASE/'validation/COMPLETE.json',BASE/'cords/COMPLETE.json']:
        d=json.loads(p.read_text());h.update(d.get('source_inputs_sha256',d.get('input_hashes',{})))
    rt.unchanged(h)
    a=pd.read_csv(BASE/'inputs/CELL_METADATA.csv.gz',low_memory=False,dtype={'patient_id':str})
    oldmd=pd.read_csv(STORY/'data_umap/data/CELL_METADATA.csv.gz',low_memory=False,dtype={'patient_id':str})
    s=a[a.dataset.eq('Senosain CyTOF')].copy();s['new_row']=s.index
    o=oldmd[oldmd.dataset.eq('Senosain CyTOF')].copy();o['old_row']=o.index
    common=s.merge(o[['source_cell_id','old_row']],on='source_cell_id',validate='one_to_one')
    raw=np.load(BASE/'inputs/RAW_SHARED9.npy');oldraw=np.load(STORY/'data_umap/data/RAW_SHARED9.npy')
    err=float(np.abs(raw[common.new_row]-oldraw[common.old_row]).max());assert err==0
    newy=pd.read_csv(BASE/'inputs/SOURCE_RESPONSE.csv.gz',float_precision='round_trip')
    oldy=pd.read_csv(STORY/'validation/data/SOURCE_RESPONSE.csv.gz',float_precision='round_trip')
    err_y=float(np.abs(newy.observed_HLA_ABC.to_numpy()[common.new_row]-oldy.observed_HLA_ABC.to_numpy()[common.old_row]).max());assert err_y==0
    checks.update(overlap_cells=len(common),overlap_marker_max_error=err,overlap_HLA_ABC_max_error=err_y)
    # Independent regression check: the projection implementation reconstructs old frozen outputs.
    oldb=np.load(STORY/'data_umap/reference/baseline_SHARED9.npy');olda=np.load(STORY/'data_umap/reference/RAMP_SHARED9.npy')
    with np.load(STORY/'data_umap/reference/REFERENCE_PROFILES.npz') as z:profiles={k:z[k] for k in z.files}
    cordmd=pd.read_csv(STORY/'data_cords/data/CELL_METADATA.csv.gz',dtype={'patient_id':str})
    cordx=np.load(STORY/'data_cords/data/RAW_SHARED9.npy',mmap_mode='r');cordb=np.load(STORY/'data_cords/data/baseline_SHARED9.npy',mmap_mode='r');corda=np.load(STORY/'data_cords/data/RAMP_SHARED9.npy',mmap_mode='r')
    pid=oldmd.loc[oldmd.dataset.eq('Cords IMC'),'patient_id'].iloc[0]
    ri=np.flatnonzero(oldmd.dataset.eq('Cords IMC')&oldmd.patient_id.eq(pid));ix=np.flatnonzero(cordmd.patient_id.eq(pid))
    b,z=anchor_map(np.asarray(cordx[ix]),oldraw[ri],oldb[ri],olda[ri],f'Cords IMC|{pid}',profiles,np.asarray(cordb[ix]))
    err=float(np.abs(z-corda[ix]).max());assert err<1e-8;checks['old_reference_Cords_projection_error']=err
    with np.load(STORY/'data_cords/reference/FROZEN_PLATFORM_MAPS.npz') as q:maps=[(q[f'x{j}'],q[f'y{j}']) for j in range(9)]
    pid=cordmd.loc[~cordmd.reference_donor,'patient_id'].iloc[0];ix=np.flatnonzero(cordmd.patient_id.eq(pid))
    err=float(np.abs(apply_maps(np.array(cordb[ix]),maps)-corda[ix]).max());assert err<1e-10;checks['old_new_donor_Cords_projection_error']=err
    parameters=pd.read_csv(STORY/'data_cords/results/BASELINE_PARAMETERS.csv',dtype={'patient_id':str})
    maxerr=0.
    for pid in cordmd.loc[~cordmd.reference_donor,'patient_id'].drop_duplicates().iloc[::50]:
        ix=np.flatnonzero(cordmd.patient_id.eq(pid));q=parameters[parameters.patient_id.eq(pid)].set_index('marker').loc[MARKERS]
        restored=(np.arcsinh((cordx[ix]+q['shift'].to_numpy())/q.cofactor.to_numpy())-q.offset.to_numpy())/q.scale.to_numpy()
        maxerr=max(maxerr,float(np.abs(restored-cordb[ix]).max()))
    assert maxerr<1e-9;checks['reused_baseline_parameter_reconstruction_error']=maxerr
    # Compare refitted old Cox estimates to the already corrected historical model.
    source=ROOT.parent/'lung_caf_covariate_fix_20260917/COX_TERMS_corrected.csv'
    h[str(source)]=rt.sha(source)
    prior=pd.read_csv(source);new=pd.read_csv(BASE/'analysis/COX_TERMS.csv')
    v=new[(new.version=='old')&(new.config=='RAMP_k30')&(new.state=='mean')&(new.cohort=='all280')].merge(prior,on='term',suffixes=('','_prior'))
    assert len(v)==9
    err=float(np.abs(v.coef-v.coef_prior).max());assert err<1e-6;checks['old_corrected_Cox_max_coef_error']=err
    checks['input_hashes']=h;checks['source_files_unchanged']=True
    checks['code_sha256']={p.name:rt.sha(p) for p in (BASE/'scripts').glob('*') if p.is_file()}
    rt.save(BASE/'provenance/FINAL_CHECKS.json',checks)
    print('All checks passed',flush=True)
if __name__=='__main__':main()
