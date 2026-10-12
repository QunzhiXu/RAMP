from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, false_discovery_control
from analyze_spatial_scores import HERE, ROOT, RAW, RETURN, sha, summarize

MARKERS=['LAG3','IL10','GranzymeB','Ki67']
OLD=ROOT/'story_test_hnscc/final_return_exploration_20260910'

def main():
    paths=[RAW,RETURN,OLD/'target_all_metadata.csv.gz',OLD/'target_all_aligned.npy']
    before={str(p):sha(p) for p in paths}
    q=pd.read_csv(RETURN,dtype={'cell_uid':str,'patient':str,'unique_region':str})
    meta=pd.read_csv(OLD/'target_all_metadata.csv.gz',dtype={'cell_uid':str,'patient':str,'unique_region':str})
    aligned=np.load(OLD/'target_all_aligned.npy',mmap_mode='r');assert aligned.shape==(len(meta),22)
    meta['shared_mean']=aligned.mean(axis=1)
    c=meta.set_index('cell_uid').loc[q.cell_uid].reset_index()
    for col in ['patient','unique_region','x','y']:np.testing.assert_array_equal(c[col],q[col])
    blocks=[]
    for r in pd.read_csv(RAW,usecols=['UID']+MARKERS,dtype={'UID':str},chunksize=150000):
        blocks.append(r[r.UID.isin(set(q.cell_uid))])
    raw=pd.concat(blocks).set_index('UID').loc[q.cell_uid]
    for m in MARKERS:np.testing.assert_allclose(c[m],raw[m].to_numpy(),rtol=1e-12,atol=1e-12)
    q[MARKERS]=raw[MARKERS].to_numpy();q['shared_mean']=c.shared_mean.to_numpy()
    q['state']=np.select([q.within_patient_rank<=.4,q.within_patient_rank>=.6],['Lower','Higher'],default='Middle')
    for m in MARKERS:
        q[m+'_patient_rank']=q.groupby('patient')[m].transform(lambda v:(rankdata(v)-.5)/len(v))
        q[m+'_roi_rank']=q.groupby('unique_region')[m].transform(lambda v:(rankdata(v)-.5)/len(v))
    records=[];cors=[]
    for m in MARKERS:
        b=q[q.state.ne('Middle')].groupby(['patient','state'])[m+'_patient_rank'].mean().unstack()
        for p,r in b.iterrows():records.append(dict(marker=m,patient=p,aggregation='patient',low=r.Lower,high=r.Higher,difference=r.Higher-r.Lower))
        b=q[q.state.ne('Middle')].groupby(['patient','unique_region','state'])[m+'_roi_rank'].agg(['mean','count']).unstack()
        b=b[(b[('count','Lower')]>=5)&(b[('count','Higher')]>=5)]
        b=pd.DataFrame({'low':b[('mean','Lower')],'high':b[('mean','Higher')]}).groupby('patient').mean()
        for p,r in b.iterrows():records.append(dict(marker=m,patient=p,aggregation='within_roi_equal',low=r.low,high=r.high,difference=r.high-r.low))
    for p,g in q.groupby('patient'):
        for m in MARKERS:
            rho=spearmanr(g.predicted_score,g[m]).statistic if g[m].nunique()>1 else np.nan
            cors.append(dict(marker=m,patient=p,aggregation='unadjusted',rho=rho,cells=len(g)))
        if len(g)<40:continue
        roi=pd.get_dummies(g.unique_region,drop_first=True).to_numpy(float)
        intensity=rankdata(g.shared_mean);intensity=(intensity-intensity.mean())/max(intensity.std(),1e-12)
        design=np.column_stack([np.ones(len(g)),roi,intensity])
        r=rankdata(g[['predicted_score']+MARKERS].to_numpy(),axis=0)
        residual=r-design@np.linalg.lstsq(design,r,rcond=None)[0]
        for j,m in enumerate(MARKERS):
            rho=np.corrcoef(residual[:,0],residual[:,j+1])[0,1]
            cors.append(dict(marker=m,patient=p,aggregation='ROI_and_shared_intensity_adjusted',rho=rho,cells=len(g)))
    d=pd.DataFrame(records);d.to_csv(HERE/'codex_protein_patient_contrasts.csv',index=False)
    cor=pd.DataFrame(cors);cor.to_csv(HERE/'codex_protein_patient_correlations.csv',index=False)
    rows=[]
    for keys,g in d.groupby(['marker','aggregation']):rows.append(dict(zip(['marker','aggregation'],keys),**summarize(g.difference),mean_low=g.low.mean(),mean_high=g.high.mean()))
    summary=pd.DataFrame(rows);summary['q_this_review']=false_discovery_control(summary.p.to_numpy());summary.to_csv(HERE/'codex_protein_contrast_summary.csv',index=False)
    rows=[]
    for keys,g in cor.groupby(['marker','aggregation']):rows.append(dict(zip(['marker','aggregation'],keys),**summarize(g.rho)))
    assoc=pd.DataFrame(rows);assoc['q_this_review']=false_discovery_control(assoc.p.to_numpy());assoc.to_csv(HERE/'codex_protein_correlation_summary.csv',index=False)
    q.to_csv(HERE/'codex_measured_protein_cells.csv.gz',index=False)
    after={str(p):sha(p) for p in paths};assert before==after
    (HERE/'PROTEIN_AUDIT.json').write_text(json.dumps(dict(status='PASS',cells=len(q),patients=q.patient.nunique(),formal_cd8_model=True,cache_matches_raw=True,inputs_unchanged=before),indent=2)+'\n')
    print(summary.to_string(index=False));print(assoc.to_string(index=False))

if __name__=='__main__':main()
