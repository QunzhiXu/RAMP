import runtime as rt
from runtime import BASE,STORY,ROOT,MARKERS
import time,json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from ramp import ramp_ver2 as ramp
from frozen_maps import fit_maps

def main():
    out=BASE/'alignment'
    if (out/'COMPLETE.json').exists():print('Alignment already complete',flush=True);return
    t=time.time();rt.status('prepare new reference')
    ann=ROOT/'raw_data/lung_cancer/Senosain_2025_CyTOF_and_omics/annotation'
    old=STORY/'data_umap'
    paths=[ann/'senosain_140k_cell_metadata.csv.gz',ann/'rough_annotation/senosain_140k_final_cell_annotations.csv.gz',ann/'senosain_140k_raw35.npy',ann/'senosain_140k_markers35.csv',old/'data/CELL_METADATA.csv.gz',old/'data/RAW_SHARED9.npy',old/'data/DESIGN.json',old/'data/SELECTED_DONORS.csv',old/'reference/baseline_SHARED9.npy',old/'reference/RAMP_SHARED9.npy',BASE/'vendor/ramp/ramp_ver2.py',BASE/'vendor/ramp/ramp.py']
    h=rt.hashes(paths);rt.save(BASE/'provenance/ALIGNMENT_INPUTS.json',h)
    om=pd.read_csv(old/'data/CELL_METADATA.csv.gz',dtype={'patient_id':str})
    ox=np.load(old/'data/RAW_SHARED9.npy');design=json.loads((old/'data/DESIGN.json').read_text())
    am=pd.read_csv(ann/'senosain_140k_cell_metadata.csv.gz',dtype={'patient_id':str})
    a=pd.read_csv(ann/'rough_annotation/senosain_140k_final_cell_annotations.csv.gz',dtype={'patient_id':str},keep_default_na=False)
    assert am.source_cell_id.equals(a.source_cell_id)
    assert np.array_equal(am.sampled_row_zero_based,np.arange(140000))
    names=pd.read_csv(ann/'senosain_140k_markers35.csv').marker.tolist()
    xx=np.load(ann/'senosain_140k_raw35.npy',mmap_mode='r')
    wanted=set(om.loc[om.dataset.eq('Senosain CyTOF'),'patient_id'])
    am=am.merge(a[['source_cell_id','parent_cluster','subcluster','final_cell_type','annotation_confidence','qc_recommendation']],on='source_cell_id',validate='one_to_one',sort=False)
    pick=am[am.patient_id.isin(wanted)&am.qc_recommendation.eq('keep')].sort_values(['patient_id','sample_rank_within_patient']).groupby('patient_id',sort=False).head(1000).copy().reset_index(drop=True)
    assert len(pick)==28000 and pick.groupby('patient_id').size().eq(1000).all()
    candidate=ROOT/'cross_platform_trainability_v2_20260919/reports/senosain_new_annotation_review_20260921/CANDIDATE_REFERENCE_IDS_REVIEW_ONLY.csv.gz'
    assert set(pick.source_cell_id)==set(pd.read_csv(candidate).source_cell_id)
    h[str(candidate)]=rt.sha(candidate)
    rr=pick.sampled_row_zero_based.to_numpy(int)
    source=np.array(xx[rr][:,[names.index('CYTOKERATIN' if m=='EPI' else m) for m in MARKERS]],float)
    sm=pd.DataFrame(index=range(len(pick)),columns=om.columns)
    for col in ['source_cell_id','patient_id','source_file']:sm[col]=pick[col]
    sm['source_row_id']=rr;sm['roi_id']=pick.source_file;sm['dataset']='Senosain CyTOF'
    sm['provider_cell_type']=pick.final_cell_type
    mapping={'CD4 T':'T cell','CD8 T':'T cell','Myeloid':'Myeloid / APC','Tumor/Epithelial':'Tumor / epithelial','Endothelial':'Endothelial','B-like':'B cell','Mix':'Other / mixed'}
    sm['broad_type']=pick.final_cell_type.map(mapping);assert sm.broad_type.notna().all()
    sm['unit_key']='Senosain CyTOF|'+pick.patient_id
    sm['cell_key']='Senosain CyTOF|new140k|'+pick.sampled_row_zero_based.astype(str)
    for c in ['stage_group','selection_batch']:
        m=om[om.dataset.eq('Senosain CyTOF')].drop_duplicates('patient_id').set_index('patient_id')[c];sm[c]=pick.patient_id.map(m)
    sm['annotation_confidence']=pick.annotation_confidence;sm['parent_cluster']=pick.parent_cluster;sm['qc_recommendation']=pick.qc_recommendation
    non=~om.dataset.eq('Senosain CyTOF')
    md=pd.concat([sm,om[non]],ignore_index=True);raw=np.vstack([source,ox[non]])
    assert raw.shape==(84000,9) and md.cell_key.is_unique and np.isfinite(raw).all()
    assert md.groupby('unit_key').size().eq(1000).all()
    # Source responses belong to the selected cells, never inferred from labels.
    yr=np.asarray(xx[rr,names.index('HLA_ABC')]);y=np.empty(len(pick))
    for _,ix in pick.groupby('patient_id').indices.items():y[ix]=rankdata(yr[ix])/(len(ix)+1)
    response=sm[['cell_key','source_cell_id','patient_id','source_row_id']].copy()
    response['observed_HLA_ABC']=yr;response['source_patient_rank']=y
    response.to_csv(BASE/'inputs/SOURCE_RESPONSE.csv.gz',index=False)
    pick.to_csv(BASE/'inputs/SELECTED_SENOSAIN_CELLS.csv.gz',index=False)
    md.to_csv(BASE/'inputs/CELL_METADATA.csv.gz',index=False);np.save(BASE/'inputs/RAW_SHARED9.npy',raw)
    design['cell_selection']='New Senosain: first 1000 QC=keep in frozen random order per original patient; Cords/Zheng unchanged'
    rt.save(BASE/'inputs/DESIGN.json',design)
    overlap=len(set(sm.source_cell_id)&set(om.loc[om.dataset.eq('Senosain CyTOF'),'source_cell_id']))
    rt.save(BASE/'inputs/SAMPLING_CHECKS.json',dict(n_cells=84000,new_source_cells=28000,overlap_old_source=overlap,C02_cells=int(pick.parent_cluster.eq('C02').sum()),all_QC_keep=True,candidate_ids_match=True,other_platform_cells_unchanged=True,label_only_changes_would_not_change_predictions=True))
    print('New reference prepared',len(pick),'C02',pick.parent_cluster.eq('C02').sum(),'overlap',overlap,flush=True)
    rt.status('fitting canonical RAMP 2.3.0');captured={};original=ramp.apply_monotone_profiles
    def capture(b,m,p):
        captured.update({k:np.asarray(v).copy() for k,v in p.items()});return original(b,m,p)
    ramp.apply_monotone_profiles=capture
    try:r=ramp.fit_transform(raw,md[['dataset','unit_key']],MARKERS,datasets=design['datasets'],family=design['family'])
    finally:ramp.apply_monotone_profiles=original
    assert r.strict_rank_inversions==0
    assert np.max(np.abs(r.aligned-r.baseline))<=.75+1e-9
    assert np.max(np.abs(r.baseline[28000:]-np.load(old/'reference/baseline_SHARED9.npy')[28000:]))<1e-10
    for name,x in [('baseline',r.baseline),('RAMP',r.aligned)]:np.save(out/f'{name}_SHARED9.npy',x)
    np.savez_compressed(out/'REFERENCE_PROFILES.npz',**{f'{u}::{j}':p for (u,j),p in captured.items()})
    for name in ['coupling_audit','strength_audit','evidence_audit','reference_weight_audit','profile_audit','algorithm_audit']:getattr(r,name).to_csv(out/f'{name.upper()}.csv',index=False)
    ci=np.flatnonzero(md.dataset.eq('Cords IMC'));maps=fit_maps(r.baseline[ci],r.aligned[ci])
    np.savez_compressed(out/'FROZEN_PLATFORM_MAPS.npz',**{f'{axis}{j}':v for j,pair in enumerate(maps) for axis,v in zip(['x','y'],pair)})
    rt.unchanged(h)
    rt.save(out/'COMPLETE.json',dict(seconds=time.time()-t,source_inputs_sha256=h,source_files_unchanged=True,rank_inversions=0,n_profiles=len(captured),other_platform_baseline_unchanged=True,code_sha256=rt.sha(__file__)))
    rt.status('alignment complete',seconds=time.time()-t)
    print('Alignment complete',round(time.time()-t,1),flush=True)
if __name__=='__main__':main()
