from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata, wilcoxon, false_discovery_control

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RAW = ROOT/'figure_HNSCC/sampled_data/haist_codex/complete/HNSCCdiscovery_CODEX.csv'
RETURN = ROOT/'figure_HNSCC/spatial_return_survival/output/haist74_pt_cd8_tigit_cd39_return.csv.gz'
CACHE = ROOT/'story_test_hnscc/hnscc_tigit_cd39_spatial_ecology_20260904/output/pt_cd8_spatial_ecology_features.csv.gz'
SEED = 20260913

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def summarize(v):
    v=np.asarray(v,float); v=v[np.isfinite(v)]
    if not len(v):return dict(n=0,mean=np.nan,median=np.nan,positive=0,negative=0,ci_low=np.nan,ci_high=np.nan,p=np.nan)
    boot=np.random.default_rng(SEED).choice(v,(10000,len(v)),replace=True).mean(1)
    return dict(n=len(v),mean=v.mean(),median=np.median(v),positive=int((v>1e-12).sum()),negative=int((v < -1e-12).sum()),ci_low=np.quantile(boot,.025),ci_high=np.quantile(boot,.975),p=wilcoxon(v).pvalue if np.any(v) else 1.)

def main():
    start={str(p):sha(p) for p in [RAW,RETURN,CACHE]}
    f=pd.read_csv(CACHE,dtype={'cell_uid':str,'patient':str,'unique_region':str})
    q=pd.read_csv(RETURN,dtype={'cell_uid':str,'patient':str,'unique_region':str})
    merged=f.merge(q,on='cell_uid',validate='one_to_one',suffixes=('_old','_current'))
    assert len(merged)==len(f)==len(q)==38299
    for c in ['predicted_score','within_patient_rank','spatial_y','x','y']:
        np.testing.assert_array_equal(merged[c+'_old'],merged[c+'_current'])
    assert f.patient.nunique()==74 and f.unique_region.nunique()==176
    print('Verified all 38,299 current predictions against old spatial features.',flush=True)
    groups={'Tumor':['tumor','tumor_CA9'],'CAF':['CAF'],'Macrophage':['macrophages'],'CAF + macrophage':['CAF','macrophages']}
    # Reconstruct patient-level and within-ROI high-low effects without changed thresholds.
    contrasts=[]; quintiles=[]; correlations=[]
    f['state']=np.select([f.within_patient_rank<=.4,f.within_patient_rank>=.6],['Lower','Higher'],default='Middle')
    f['quintile']=np.minimum((f.within_patient_rank*5).astype(int)+1,5)
    for k in [15,30]:
        for label,types in groups.items():
            for metric in ['raw_fraction','roi_adjusted_difference']:
                endpoint=f'k{k}_{metric}__{label}'
                f[endpoint]=f[[f'k{k}_{metric}__{t}' for t in types]].sum(axis=1)
                means=f[f.state.ne('Middle')].groupby(['patient','state'])[endpoint].mean().unstack()
                for patient,r in means.iterrows():
                    contrasts.append(dict(k=k,endpoint=label,metric=metric,aggregation='patient',patient=patient,low=r.Lower,high=r.Higher,difference=r.Higher-r.Lower))
                roi=f[f.state.ne('Middle')].groupby(['patient','unique_region','state'])[endpoint].agg(['mean','count']).unstack()
                # Five focal cells per group in each ROI; no intensity/score-based exclusions.
                valid=(roi[('count','Lower')]>=5)&(roi[('count','Higher')]>=5)
                roi=roi.loc[valid]
                out=pd.DataFrame({'low':roi[('mean','Lower')],'high':roi[('mean','Higher')]})
                out=out.groupby('patient').mean()
                for patient,r in out.iterrows():
                    contrasts.append(dict(k=k,endpoint=label,metric=metric,aggregation='within_roi_equal',patient=patient,low=r.low,high=r.high,difference=r.high-r.low))
                b=f.groupby(['patient','quintile'])[endpoint].mean()
                for (p,qi),v in b.items():quintiles.append(dict(k=k,endpoint=label,metric=metric,patient=p,quintile=qi,value=v))
            for p,g in f.groupby('patient'):
                a=g.predicted_score.to_numpy();b=g[f'k{k}_roi_adjusted_difference__{label}'].to_numpy()
                correlations.append(dict(k=k,endpoint=label,patient=p,rho=spearmanr(a,b).statistic if np.ptp(a)>0 and np.ptp(b)>0 else np.nan))
    con=pd.DataFrame(contrasts);con.to_csv(HERE/'cell_neighbor_patient_contrasts.csv',index=False)
    cr=[]
    for keys,g in con.groupby(['k','endpoint','metric','aggregation']):
        cr.append(dict(zip(['k','endpoint','metric','aggregation'],keys),**summarize(g.difference),mean_low=g.low.mean(),mean_high=g.high.mean()))
    cr=pd.DataFrame(cr);cr['q_this_review']=false_discovery_control(cr.p.to_numpy());cr.to_csv(HERE/'cell_neighbor_summary.csv',index=False)
    pd.DataFrame(quintiles).to_csv(HERE/'cell_neighbor_patient_quintiles.csv',index=False)
    co=pd.DataFrame(correlations);co.to_csv(HERE/'cell_neighbor_correlations.csv',index=False)
    print(cr[(cr.k==15)&cr.metric.eq('raw_fraction')][['endpoint','aggregation','n','mean_low','mean_high','mean','ci_low','ci_high']].to_string(index=False),flush=True)
    # Read real tissue coordinates and labels, no protein values or outcome files.
    use=['UID','unique_region','Pat_ID','x','y','Tissue_type_broad','cell_types_mediumres']
    blocks=[]
    for c in pd.read_csv(RAW,usecols=use,dtype={'UID':str,'unique_region':str,'Pat_ID':str},chunksize=150000):
        blocks.append(c[c.Tissue_type_broad.eq('PT')&c.unique_region.isin(set(q.unique_region))].copy())
    allcells=pd.concat(blocks,ignore_index=True).rename(columns={'Pat_ID':'patient','UID':'cell_uid'})
    assert len(allcells)==452282
    assert set(allcells.loc[allcells.cell_types_mediumres.eq('CD8T'),'cell_uid'])==set(q.cell_uid)
    allcells=allcells.merge(q[['cell_uid','predicted_score','spatial_y']],on='cell_uid',how='left',validate='one_to_one')
    allcells['n_cd8']=allcells.cell_types_mediumres.eq('CD8T').astype(int)
    allcells['n_non_cd8']=1-allcells.n_cd8
    for label,types in groups.items(): allcells[label]=allcells.cell_types_mediumres.isin(types).astype(int)
    tiles=[];metrics=[];counts=[]
    for size in [100,200,400]:
        for shift in [0,.5]:
            a=allcells.copy();a['tile_x']=np.floor((a.x-shift*size)/size).astype(int);a['tile_y']=np.floor((a.y-shift*size)/size).astype(int)
            b=a.groupby(['patient','unique_region','tile_x','tile_y']).agg(n_cells=('cell_uid','size'),n_cd8=('n_cd8','sum'),n_non_cd8=('n_non_cd8','sum'),state=('predicted_score','mean'),original_niche=('spatial_y','mean'),**{k:(k,'sum') for k in groups}).reset_index()
            for label in groups:b[label]=b[label]/b.n_non_cd8.replace(0,np.nan)
            b['eligible']=(b.n_cd8>=10)&(b.n_non_cd8>=30)
            b['edge_px']=size;b['offset_fraction']=shift;tiles.append(b)
            use=b[b.eligible].copy()
            # All-patient inventory including those with no eligible tiles.
            for patient in sorted(q.patient.unique()):
                p=use[use.patient.eq(patient)]
                counts.append(dict(edge_px=size,offset_fraction=shift,patient=patient,n_tiles=len(p),n_cd8=int(p.n_cd8.sum()),n_rois=p.unique_region.nunique()))
                if len(p)<5:continue
                for endpoint in list(groups)+['original_niche']:
                    rho=spearmanr(p.state,p[endpoint]).statistic if p.state.nunique()>1 and p[endpoint].nunique()>1 else np.nan
                    metrics.append(dict(edge_px=size,offset_fraction=shift,patient=patient,endpoint=endpoint,aggregation='within_patient',n_tiles=len(p),rho=rho))
                    centered=[]
                    for _,r in p.groupby('unique_region'):
                        if len(r)<2:continue
                        x=(rankdata(r.state)-.5)/len(r)-.5;y=(rankdata(r[endpoint])-.5)/len(r)-.5
                        centered.append(np.column_stack([x,y]))
                    if centered:
                        ar=np.concatenate(centered)
                        rh=np.corrcoef(ar.T)[0,1] if len(ar)>=5 and (ar.std(0)>0).all() else np.nan
                        metrics.append(dict(edge_px=size,offset_fraction=shift,patient=patient,endpoint=endpoint,aggregation='within_roi_centered_ranks',n_tiles=len(ar),rho=rh))
            print(f'Tiles {size}px shift {shift}: {len(use)} eligible, {use.patient.nunique()} patients',flush=True)
    pd.concat(tiles,ignore_index=True).to_csv(HERE/'all_grid_tiles.csv.gz',index=False)
    pd.DataFrame(counts).to_csv(HERE/'all74_grid_eligibility.csv',index=False)
    met=pd.DataFrame(metrics);met.to_csv(HERE/'regional_patient_correlations.csv',index=False)
    out=[]
    for keys,g in met.groupby(['edge_px','offset_fraction','endpoint','aggregation']):
        out.append(dict(zip(['edge_px','offset_fraction','endpoint','aggregation'],keys),**summarize(g.rho)))
    out=pd.DataFrame(out);ok=out.p.notna();out.loc[ok,'q_this_review']=false_discovery_control(out.loc[ok,'p'].to_numpy());out.to_csv(HERE/'regional_summary.csv',index=False)
    print('PRIMARY 200 PX',out[out.edge_px.eq(200)&out.offset_fraction.eq(0)].to_string(index=False),flush=True)
    end={str(p):sha(p) for p in [RAW,RETURN,CACHE]};assert start==end
    (HERE/'AUDIT.json').write_text(json.dumps(dict(status='PASS',cells=38299,patients=74,rois=176,all_matching_tissue_cells=len(allcells),hashes_unchanged=start,model_or_outcome_tuning=False,source_prediction_identity='exact'),indent=2)+'\n')

if __name__=='__main__':main()
