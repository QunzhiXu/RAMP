#!/usr/bin/env python3
import sys
sys.dont_write_bytecode=True
from common import *
import anndata as ad
from scipy import sparse


def main():
    screen_path=FORMAL/'biomarker_discovery/screen/marker_screen_all_107_all_variants.csv'
    raw_path=helper.HAIST_CELL_TABLE
    adt_path=FORMAL/'sampled_data/rahim_citeseq/complete/GSE212797_adata_ADT.h5ad'
    inputs={'plan':HERE/'ANALYSIS_PLAN.md','helper':HELPER,'screen':screen_path,'full_codex':raw_path,'adt':adt_path,'alignment_raw':helper.ALIGNMENT_INPUT/'raw_shared_markers.float64.npy','alignment_output':helper.ALIGNMENT_OUTPUT/'aligned_ramp_ver2.float32.npy','metadata':helper.ALIGNMENT_INPUT/'cell_metadata.csv.gz','selected_patients':helper.ALIGNMENT_INPUT/'selected_74_patients.csv','previous_scaling':FORMAL/'biomarker_discovery/projection/feature_scaling_audit.csv','original_program':FORMAL/'biomarker_discovery/screen/rahim_tumor_cd8_biomarker_programs.csv.gz'}
    before={k:sha(p) for k,p in inputs.items()}
    screen=pd.read_csv(screen_path)
    candidates=screen[(screen.paper_clr__positive_donors==3)&(screen.isotype_adjusted__positive_donors==3)&(screen.global_adt_loo_adjusted__positive_donors>=2)].sort_values('robust_median_rank_primary_variants').copy()
    assert len(candidates)==12 and set(candidates.marker_raw)==set(ALIASES)
    candidates['alias']=candidates.marker_raw.map(ALIASES)
    candidates['extension_tier']=np.where(candidates.passes_frozen_rule_pre_redundancy,'original_robust','direction_consistent_exploratory')
    candidates.to_csv(HERE/'candidate_audit.csv',index=False)
    marker_names=candidates.marker_raw.tolist();aliases=candidates.alias.tolist()
    strict=candidates[candidates.passes_frozen_rule_pre_redundancy].marker_raw.tolist()
    assert len(strict)==8
    programs={'pair':['TIGIT--VSTM3','CD39']}
    for marker in marker_names:
        if marker not in programs['pair']:programs['plus_'+ALIASES[marker]]=programs['pair']+[marker]
    programs['robust8']=strict
    registry=[]
    for name,members in programs.items():
        tier='reference' if name=='pair' else 'original_robust' if set(members)<=set(strict) else 'direction_consistent_exploratory'
        registry.append(dict(program=name,members=';'.join(members),markers=len(members),tier=tier))
    pd.DataFrame(registry).to_csv(HERE/'program_registry.csv',index=False)
    w=np.zeros((len(marker_names),len(programs)))
    for j,members in enumerate(programs.values()):
        for marker in members:w[marker_names.index(marker),j]=1/len(members)
    save_json(dict(markers=marker_names,aliases=aliases,programs=programs,response_columns=['marker_'+a for a in aliases]+list(programs)),HERE/'definitions.json')
    raw,aligned,meta,shared=helper.read_formal_alignment()
    selected=helper.read_selected_patients()
    curves,audit=helper.build_haist_curves(raw,aligned,meta,shared)
    source_rows=np.flatnonzero(meta.dataset.eq('GSE212797_ADT')&meta.tissue.eq('tumor'))
    source=meta.iloc[source_rows][['cell_uid','donor','provider_cell_type']].reset_index(drop=True)
    source.cell_uid=source.cell_uid.str.removeprefix('RAHIM|')
    source['broad']=source.provider_cell_type.map(SOURCE_BROAD)
    assert len(source)==11674 and source.broad.notna().all()
    sx=np.asarray(aligned[source_rows],dtype=float)
    adt=ad.read_h5ad(adt_path)
    pos=adt.obs_names.get_indexer(source.cell_uid)
    assert (pos>=0).all()
    assert adt.obs.iloc[pos].patient.astype(str).tolist()==source.donor.tolist()
    values=adt.X[pos]
    counts=(values.toarray() if sparse.issparse(values) else np.asarray(values)).astype(float)
    markers=adt.var_names.astype(str)
    bg=np.array(['isotype' not in m.lower() and m!='CD223--LAG-3' for m in markers]);assert bg.sum()==129
    logs=np.log1p(counts)
    marker_clr=logs[:,markers.get_indexer(marker_names)]-logs[:,bg].mean(axis=1,keepdims=True)
    all_pcts=pct_matrix(marker_clr,source.donor.to_numpy())
    cd8=source.provider_cell_type.eq('CD8 T cell').to_numpy()
    cd8_pcts=pct_matrix(marker_clr[cd8],source.donor.to_numpy()[cd8])
    all_y=np.column_stack([all_pcts,all_pcts@w]);cd8_y=np.column_stack([cd8_pcts,cd8_pcts@w])
    previous=pd.read_csv(inputs['original_program'],dtype={'donor':str})
    check=source[cd8][['cell_uid','donor']].assign(rebuilt=cd8_y[:,len(marker_names)]).merge(previous,on=['cell_uid','donor'],validate='one_to_one')
    error=float(np.max(abs(check.rebuilt-check.compact_top5pct_clr_program)))
    assert error<5e-15
    source.to_csv(HERE/'source_all_metadata.csv.gz',index=False)
    np.save(HERE/'source_all_aligned.npy',sx)
    np.save(HERE/'source_all_responses.npy',all_y)
    np.save(HERE/'source_cd8_responses.npy',cd8_y)
    np.save(HERE/'source_adt_counts.npy',counts)
    save_json(markers.tolist(),HERE/'all_adt_names.json')
    use=['UID','Pat_ID','unique_region','x','y','Tissue_type_broad','cell_types_mediumres']+shared+READOUTS
    blocks=[];ids=set(selected.CaseID.astype(str))
    print('Reading all cell types from full CODEX PT cohort.',flush=True)
    for chunk in pd.read_csv(raw_path,usecols=use,chunksize=150000):
        patients=chunk.Pat_ID.astype(str)
        mask=patients.isin(ids)&chunk.Tissue_type_broad.eq('PT')
        if mask.any():
            b=chunk[mask].copy();b['Pat_ID']=patients[mask].to_numpy();blocks.append(b)
    target=pd.concat(blocks,ignore_index=True)
    target=target.rename(columns={'UID':'cell_uid','Pat_ID':'patient','cell_types_mediumres':'cell_type'})
    assert target.cell_uid.is_unique and target.patient.nunique()==74
    assert target.cell_type.eq('CD8T').sum()==38299
    target['broad']=target.cell_type.map(TARGET_BROAD);assert target.broad.notna().all()
    hx=np.empty((len(target),len(shared)),float);outside=np.zeros(len(target),int)
    for patient,rows in target.groupby('patient').groups.items():
        rr=np.asarray(rows);vv=target.loc[rr,shared].to_numpy(float)
        for j,marker in enumerate(shared):
            cx,cy=curves[(str(patient),marker)]
            hx[rr,j]=np.interp(vv[:,j],cx,cy)
            outside[rr]+=(vv[:,j]<cx[0])|(vv[:,j]>cx[-1])
    target['curve_outside_markers']=outside
    target.drop(columns=shared+['Tissue_type_broad']).to_csv(HERE/'target_all_metadata.csv.gz',index=False)
    np.save(HERE/'target_all_aligned.npy',hx)
    center,scale,scaler=helper.equal_donor_patient_scaler(sx,source.donor.to_numpy(),hx,target.patient.to_numpy(),shared)
    scaler.to_csv(HERE/'allcell_scaler.csv',index=False)
    oldscale=pd.read_csv(inputs['previous_scaling']);assert oldscale.marker.tolist()==shared
    np.savez(HERE/'scalers.npz',all_center=center,all_scale=scale,cd8_center=oldscale.combined_equal_domain_mean.to_numpy(),cd8_scale=oldscale.combined_equal_domain_scale.to_numpy())
    sample=np.flatnonzero(meta.dataset.eq('Haist_CODEX'))
    meta.iloc[sample][['cell_uid','donor','provider_cell_type']].to_csv(HERE/'target_sample_metadata.csv.gz',index=False)
    np.save(HERE/'target_sample_aligned.npy',np.asarray(aligned[sample],float))
    assert before=={k:sha(p) for k,p in inputs.items()}
    manifest=dict(inputs={k:dict(path=str(p),sha256=before[k]) for k,p in inputs.items()},inputs_unchanged=True,source_all_cells=len(source),source_cd8_cells=int(cd8.sum()),source_donor_counts=source.donor.value_counts().to_dict(),target_cells=len(target),target_patients=74,target_rois=target.unique_region.nunique(),target_counts=target.cell_type.value_counts().to_dict(),original_source_pair_error=error,clinical_outcomes_read=False)
    save_json(manifest,HERE/'prepare_manifest.json');print(json.dumps({k:v for k,v in manifest.items() if k!='inputs'},indent=2),flush=True)


if __name__=='__main__':main()
