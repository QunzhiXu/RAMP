#!/usr/bin/env python3
"""Frozen-neighbor comparison of seven signed molecular programs; no outcomes."""
from pathlib import Path
import hashlib, importlib.util, json, os, sys
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / 'output'
OUT.mkdir(exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(HERE / '.cache/matplotlib'))
os.environ.setdefault('MPLBACKEND','Agg')
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

PROGRAMS = {
    'pair': ['TIGIT','CD39'],
    'protein3': ['LAG3','CD127','CD62L'],
    'rna3': ['CTLA4','CXCL13','IL7R'],
    'pair_protein3': ['TIGIT','CD39','LAG3','CD127','CD62L'],
    'pair_rna3': ['TIGIT','CD39','CTLA4','CXCL13','IL7R'],
    'protein3_rna3': ['LAG3','CD127','CD62L','CTLA4','CXCL13','IL7R'],
    'pair_protein3_rna3': ['TIGIT','CD39','LAG3','CD127','CD62L','CTLA4','CXCL13','IL7R'],
}
NAMES = {'pair':'CD39/TIGIT', 'protein3':'3 proteins', 'rna3':'3 RNA',
         'pair_protein3':'Pair + 3 proteins','pair_rna3':'Pair + 3 RNA',
         'protein3_rna3':'3 proteins + 3 RNA','pair_protein3_rna3':'Pair + 3 proteins + 3 RNA'}
NEGATIVE = {'CD127','CD62L','IL7R'}
ADT = ROOT / 'sampled_data/rahim_citeseq/complete/GSE212797_adata_ADT.h5ad'
RNA = ROOT / 'sampled_data/rahim_citeseq/complete/GSE212797_adata.h5ad'
SOURCE = ROOT / 'biomarker_discovery/screen/rahim_tumor_cd8_biomarker_programs.csv.gz'
BASELINE = ROOT / 'spatial_return_survival/output/haist74_pt_cd8_tigit_cd39_return.csv.gz'
HELPER = ROOT / 'biomarker_discovery/01_project_spatial_state.py'
RETURN_HELPER = ROOT / 'spatial_return_survival/01_return_fixed_program.py'
INPUTS = [ADT,RNA,SOURCE,BASELINE,HELPER,RETURN_HELPER,
          ROOT/'alignment/input/raw_shared_markers.float64.npy',ROOT/'alignment/input/cell_metadata.csv.gz',
          ROOT/'alignment/input/markers.txt',ROOT/'alignment/input/selected_74_patients.csv',
          ROOT/'alignment/output/aligned_ramp_ver2.float32.npy',
          ROOT/'sampled_data/haist_codex/complete/HNSCCdiscovery_CODEX.csv',
          ROOT/'survival_motivation/output/pt_cd8_cell_spatial_niche_scores.csv.gz']


def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def dense(x):
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def source_features(source, helper):
    donors=source.donor.to_numpy(str)
    a=ad.read_h5ad(ADT)
    pos=a.obs_names.get_indexer(source.cell_uid)
    assert (pos>=0).all() and a.obs.iloc[pos].patient.astype(str).tolist()==list(donors)
    names=np.asarray(a.var_names,str)
    counts=dense(a.X[pos,:]).astype(float)
    logs=np.log1p(counts)
    isotype=np.char.find(np.char.lower(names),'isotype')>=0
    strict_bg=~isotype & (names!='CD223--LAG-3')
    pairfree_bg=~isotype & ~np.isin(names,['TIGIT--VSTM3','CD39'])
    strict=logs-logs[:,strict_bg].mean(axis=1,keepdims=True)
    pairfree=logs-logs[:,pairfree_bg].mean(axis=1,keepdims=True)
    raw={'TIGIT':strict[:,a.var_names.get_loc('TIGIT--VSTM3')],
         'CD39':strict[:,a.var_names.get_loc('CD39')]}
    for key,marker in [('LAG3','CD223--LAG-3'),('CD127','CD127--IL-7Ra'),('CD62L','CD62L')]:
        raw[key]=pairfree[:,a.var_names.get_loc(marker)]
    rna=ad.read_h5ad(RNA,backed='r')
    pos=rna.obs_names.get_indexer(source.cell_uid)
    assert (pos>=0).all()
    assert rna.obs.iloc[pos].patient.astype(str).tolist()==list(donors)
    assert rna.obs.iloc[pos].cell_type.astype(str).eq('CD8 T cell').all()
    assert rna.obs.iloc[pos].tissue.astype(str).eq('tumor').all()
    genes=['CTLA4','CXCL13','IL7R']; cols=rna.raw.var_names.get_indexer(genes)
    assert (cols>=0).all()
    order=np.argsort(pos)
    expr=dense(rna.raw.X[pos[order],:][:,cols])[np.argsort(order)]
    rna.file.close()
    for j,gene in enumerate(genes):raw[gene]=expr[:,j]
    signed={}
    feature_table=source[['cell_uid','donor']].copy()
    for name,v in raw.items():
        assert np.isfinite(v).all()
        rank=helper.percentile_midrank(v,donors)
        signed[name]=1-rank if name in NEGATIVE else rank
        feature_table[name+'_measured']=v
        feature_table[name+'_signed_rank']=signed[name]
    response=np.column_stack([np.mean([signed[m] for m in ms],axis=0) for ms in PROGRAMS.values()])
    np.testing.assert_allclose(response[:,0],source.compact_top5pct_clr_program,atol=1e-14,rtol=0)
    # Read original values directly for bit-level baseline transfer verification.
    response[:,0]=source.compact_top5pct_clr_program.to_numpy(float)
    for i,key in enumerate(PROGRAMS):feature_table[key]=response[:,i]
    feature_table.to_csv(OUT/'source_features_and_programs.csv.gz',index=False)
    source_audit=[]
    for donor in sorted(set(donors)):
        rows=donors==donor
        for i,key in enumerate(PROGRAMS):
            source_audit.append(dict(donor=donor,program=key,cells=int(rows.sum()),
                score_sd=float(response[rows,i].std()),
                rho_with_original_pair=float(spearmanr(response[rows,i],response[rows,0]).statistic),
                rho_with_forward_niche_score=float(spearmanr(response[rows,i],source.projected_roi_adjusted_exposure.to_numpy()[rows]).statistic)))
    pd.DataFrame(source_audit).to_csv(OUT/'source_donor_program_associations.csv',index=False)
    return response


def patient_metrics(target, prediction, helper, programs):
    patients=target.patient.to_numpy(str)
    rows=[]
    rank_matrix=np.column_stack([helper.percentile_midrank(prediction[:,j],patients) for j in range(prediction.shape[1])])
    for patient,ix in target.groupby('patient',sort=False).groups.items():
        ix=np.asarray(ix,int)
        truth=target.spatial_y.to_numpy()[ix]; niche=truth>0
        for j,key in enumerate(programs):
            value=prediction[ix,j]; rank=rank_matrix[ix,j]; high=rank>=.60
            q1,q5=rank<.20,rank>=.80
            rho=float(spearmanr(value,truth).statistic) if np.ptp(truth)>0 and np.ptp(value)>0 else np.nan
            auc=float(roc_auc_score(niche,value)) if len(np.unique(niche))==2 else np.nan
            hi=float(niche[high].mean()) if high.any() else np.nan
            lo=float(niche[~high].mean()) if (~high).any() else np.nan
            rows.append(dict(patient=patient,program=key,cells=len(ix),spearman=rho,niche_auroc=auc,
                high_cells=int(high.sum()),high_in_niche=hi,other_in_niche=lo,
                high_minus_other=hi-lo,q5_minus_q1=float(niche[q5].mean()-niche[q1].mean()) if q1.any() and q5.any() else np.nan,
                rho_with_pair=float(spearmanr(value,prediction[ix,0]).statistic)))
    return pd.DataFrame(rows),rank_matrix


def summarize(metrics):
    baseline=metrics[metrics.program.eq('pair')].set_index('patient')
    rows=[]
    for key in PROGRAMS:
        b=metrics[metrics.program.eq(key)].set_index('patient')
        r=dict(program=key,label=NAMES[key],patients=len(b),
               spearman_positive_patients=int((b.spearman>0).sum()),
               positive_enrichment_patients=int((b.high_minus_other>0).sum()),
               median_rho_with_pair=float(b.rho_with_pair.median()))
        for metric in ['spearman','high_minus_other','niche_auroc','q5_minus_q1']:
            pair=b[[metric]].join(baseline[[metric]],rsuffix='_baseline').dropna()
            v=pair[metric].to_numpy(float); delta=v-pair[metric+'_baseline'].to_numpy(float)
            indices=np.random.default_rng(20260913).integers(0,len(v),(10000,len(v)))
            ci=np.quantile(v[indices].mean(axis=1),[.025,.975])
            dc=np.quantile(delta[indices].mean(axis=1),[.025,.975])
            r.update({metric+'_n':len(v),metric+'_mean':float(v.mean()),metric+'_median':float(np.median(v)),
                      metric+'_ci_low':float(ci[0]),metric+'_ci_high':float(ci[1]),metric+'_delta_vs_pair':float(delta.mean()),
                      metric+'_delta_ci_low':float(dc[0]),metric+'_delta_ci_high':float(dc[1])})
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    before={str(p):sha(p) for p in INPUTS}
    helper=module(HELPER,'frozen_projection')
    ret=module(RETURN_HELPER,'frozen_return')
    raw,aligned,metadata,markers=helper.read_formal_alignment()
    source_meta,source_rows,_,_=ret.read_locked_source(metadata)
    source=pd.read_csv(SOURCE,dtype={'cell_uid':str,'donor':str}).set_index('cell_uid').loc[source_meta.cell_uid].reset_index()
    response=source_features(source,helper)
    print('Seven source programs constructed; original Pair reproduced.',flush=True)
    selected=helper.read_selected_patients()
    curves,curve_audit=helper.build_haist_curves(raw,aligned,metadata,markers)
    cells,target_x,_=helper.load_full_haist_pt_cd8(selected,curves,markers,curve_audit)
    source_x=np.asarray(aligned[source_rows],float)
    donors=source.donor.to_numpy(str); patients=cells.Pat_ID.to_numpy(str)
    center,scale,scaler=helper.equal_donor_patient_scaler(source_x,donors,target_x,patients,markers)
    source_z,target_z=(source_x-center)/scale,(target_x-center)/scale
    scaler.to_csv(OUT/'feature_scaling_audit.csv',index=False)
    target=pd.read_csv(BASELINE,dtype={'cell_uid':str,'patient':str}).set_index('cell_uid').loc[cells.UID].reset_index()
    assert target.patient.tolist()==list(patients)
    by_donor=[]
    for donor in sorted(set(donors)):
        ix=np.flatnonzero(donors==donor);tree=cKDTree(source_z[ix])
        distance,neighbor=tree.query(target_z,k=30,workers=1)
        local_scale=np.maximum(np.median(distance,axis=1,keepdims=True),1e-6)
        weight=np.exp(-distance/local_scale)
        numerator=np.sum(weight[:,:,None]*response[ix][neighbor,:],axis=1)
        pred=numerator/np.maximum(weight.sum(axis=1),1e-12)[:,None]
        np.testing.assert_allclose(pred[:,0],helper.weighted_predict(tree,response[ix,0],target_z),atol=5e-16,rtol=0)
        by_donor.append(pred)
        np.savez_compressed(OUT/f'neighbors_donor_{donor}.npz',source_row_indices=ix,neighbor=neighbor,distance=distance,weight=weight)
        print(f'Donor {donor}: all programs predicted with the identical neighbor graph.',flush=True)
    donor_matrix=np.stack(by_donor,axis=2)
    prediction=np.median(donor_matrix,axis=2)
    np.testing.assert_allclose(prediction[:,0],target.predicted_score,atol=2e-14,rtol=0)
    baseline_error=float(np.max(np.abs(prediction[:,0]-target.predicted_score.to_numpy())))
    np.savez_compressed(OUT/'donor_predictions.npz',predictions=donor_matrix,programs=np.array(list(PROGRAMS)),donors=np.array(sorted(set(donors))))
    metrics,ranks=patient_metrics(target,prediction,helper,list(PROGRAMS))
    np.testing.assert_allclose(ranks[:,0],target.within_patient_rank,atol=2e-14,rtol=0)
    metrics.to_csv(OUT/'patient_metrics.csv',index=False)
    summary=summarize(metrics);summary.to_csv(OUT/'program_summary.csv',index=False)
    out=target[['cell_uid','patient','unique_region','x','y','spatial_y','in_niche']].copy()
    for j,key in enumerate(PROGRAMS):
        out[key+'_predicted']=prediction[:,j];out[key+'_rank']=ranks[:,j]
    out.to_csv(OUT/'all_program_predictions.csv.gz',index=False)
    sensitivity=[]
    for leave,donor in enumerate(sorted(set(donors))):
        keep=[j for j in range(3) if j!=leave]
        pred=np.median(donor_matrix[:,:,keep],axis=2)
        m,_=patient_metrics(target,pred,helper,list(PROGRAMS))
        s=summarize(m);s.insert(0,'omitted_source_donor',donor);sensitivity.append(s)
    pd.concat(sensitivity,ignore_index=True).to_csv(OUT/'source_donor_sensitivity.csv',index=False)
    assert all(sha(Path(p))==h for p,h in before.items())
    audit=dict(status='PASS',inputs_sha256=before,inputs_unchanged=True,programs=PROGRAMS,
               negative_markers=sorted(NEGATIVE),source_cells=4788,source_donors=3,target_cells=len(target),
               target_patients=target.patient.nunique(),shared_markers=markers,k=30,
               weights='exp(-distance / query-specific median distance)',aggregation='median of three donor predictions',
               baseline_max_absolute_error=baseline_error,baseline_ranks_reproduced=True,
               high_threshold=.6,parameter_search=False,survival_outcomes_read=False,
               primary_metric='Equal-patient mean Spearman with observed spatial_y',
               best_new_program=summary[summary.program.ne('pair')].sort_values('spearman_mean',ascending=False).iloc[0].program)
    (OUT/'RUN_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(summary[['program','spearman_mean','spearman_median','spearman_delta_vs_pair','spearman_delta_ci_low','spearman_delta_ci_high','high_minus_other_mean']].to_string(index=False),flush=True)
    print('PASS: baseline reproduced and all source hashes unchanged.',flush=True)


if __name__=='__main__':main()
