"""Fixed spatial-scale, geometric segregation and inter-ROI state exploration."""
import runtime as rt
from runtime import BASE
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree, distance
from scipy.stats import rankdata

OUT=BASE/'results/spatial_architecture';OUT.mkdir(exist_ok=True)
SOURCE=BASE/'results/VCAF_CURRENT_PREDICTIONS_AND_DISTANCES.csv.gz'
SEED=20260909
BINS=[(0,20),(20,40),(40,80),(80,160),(160,np.inf)]

def boot(x):
    x=np.asarray(x,float);rng=np.random.default_rng(SEED)
    return tuple(np.quantile(x[rng.integers(len(x),size=(2000,len(x)))].mean(axis=1),[.025,.975]))

def auc(h,s):
    ranks=rankdata(s,axis=-1);nhi=h.sum(axis=-1);n=s.shape[-1]
    return ((ranks*h).sum(axis=-1)-nhi*(nhi+1)/2)/(nhi*(n-nhi))

def bh(p):
    p=np.asarray(p,float);idx=np.argsort(p);n=len(p);q=np.empty(n)
    q[idx]=np.minimum(1,np.minimum.accumulate((p[idx]*n/np.arange(1,n+1))[::-1])[::-1]);return q

def main():
    source_hash=rt.sha(SOURCE)
    rt.save_json(OUT/'DESIGN.json',dict(started_at_utc=rt.now(),source=str(SOURCE),source_sha256=source_hash,
        current_model='Frozen three-dataset RAMP k30, k100 companion; all 52,328 author vCAFs; no new prediction model or cutoff.',
        pair_distance='ROI >=30 vCAFs. All unordered pairs, fixed distance bins 0-20,20-40,40-80,80-160,>=160 provider units; >=10 pairs per bin. Compare mean absolute score difference with all-pair mean absolute difference in SAME ROI (exact expectation under random score permutation). Average ROI ratios within patient; patients equally weighted.',
        geometric_separation='ROI >=30 vCAFs, >=5 fixed high and >=5 fixed low. For each cell, estimate state by fraction high among its SIX nearest other vCAFs in real x/y coordinates (no distance cutoff, no self). AUC summarizes separation; it is a spatial descriptive score, not HLA prediction validation. 499 within-ROI state permutations, recomputing spatial neighbor scores. No tuning.',
        gradient='ROI >=30 vCAFs; five fixed shuffled folds, fit linear score ~ x+y on remaining folds. Held-out R2 measures a broad linear spatial gradient. Negative R2 retained.',
        between_ROI='Patient with >=2 ROIs containing >=15 vCAFs each. Compare max-min ROI mean and variance fraction eta2 attributable to ROI. Randomly redistribute scores among fixed ROI slots 499 times within patient. No direct distances across different ROIs.',
        large_effect_descriptions='Report full distributions and descriptive counts at AUC 0.7/0.8, held-out gradient R2 0.25, between-ROI score range 0.1; thresholds chosen here to summarize magnitude, not used to select new prediction parameters.',
        no_outcomes_used=True,seed=SEED))
    d=pd.read_csv(SOURCE,usecols=['source_row_id','patient_id','roi_id','x','y','RAMP_k30','RAMP_k100','high_k30','high_k100'],float_precision='round_trip')
    assert len(d)==52328 and d.source_row_id.is_unique
    pairs=[];regions=[];rng=np.random.default_rng(SEED)
    for i,((pid,roi),g) in enumerate(d.groupby(['patient_id','roi_id'],sort=True)):
        n=len(g)
        if n<30:continue
        xy=g[['x','y']].to_numpy();pdst=distance.pdist(xy)
        inds=cKDTree(xy).query(xy,k=7)[1]
        neighbors=np.array([ii[ii!=j][:6] for j,ii in enumerate(inds)])
        assert neighbors.shape==(n,6) and not np.any(neighbors==np.arange(n)[:,None])
        x=(xy-xy.mean(axis=0))/np.maximum(xy.std(axis=0),1e-10)
        x=np.column_stack([np.ones(n),x]);fold=np.empty(n,int)
        fold[np.random.default_rng(SEED).permutation(n)]=np.arange(n)%5
        for k in [30,100]:
            z=g[f'RAMP_k{k}'].to_numpy();h=g[f'high_k{k}'].to_numpy();nh=int(h.sum());nl=n-nh
            diff=distance.pdist(z[:,None],metric='cityblock');expected=float(diff.mean())
            if expected<=1e-12:continue
            for lo,hi in BINS:
                keep=(pdst>=lo)&(pdst<hi);nb=int(keep.sum())
                if nb<10:continue
                val=float(diff[keep].mean())
                pairs.append(dict(patient_id=pid,roi_id=roi,config=f'RAMP_k{k}',distance_bin=f'{lo}-{hi:g}',lo=lo,hi=hi,n_vcaf=n,n_pairs=nb,mean_absolute_difference=val,random_expected=expected,ratio=val/expected,reduction=1-val/expected))
            held=np.full(n,np.nan)
            for f in range(5):
                tr=fold!=f;te=~tr;coef=np.linalg.lstsq(x[tr],z[tr],rcond=None)[0];held[te]=x[te]@coef
            den=float(np.sum((z-z.mean())**2));r2=1-float(np.sum((z-held)**2))/den
            row=dict(patient_id=pid,roi_id=roi,config=f'RAMP_k{k}',n_vcaf=n,n_high=nh,n_low=nl,gradient_cv_r2=r2,spatial_auc=np.nan,permutation_auc_mean=np.nan,auc_p=np.nan)
            if min(nh,nl)>=5:
                score=h[neighbors].mean(axis=1);obs=float(auc(h,score))
                null=[]
                for start in range(0,499,50):
                    hh=np.vstack([rng.permutation(h) for _ in range(min(50,499-start))])
                    ss=hh[:,neighbors].mean(axis=2);null.extend(auc(hh,ss))
                row.update(spatial_auc=obs,permutation_auc_mean=float(np.mean(null)),auc_p=(1+np.sum(np.asarray(null)>=obs))/500)
            regions.append(row)
        if i%200==0:print(f'Spatial scale and separation: ROI {i+1}/975',flush=True)
    p=pd.DataFrame(pairs);p.to_csv(OUT/'ROI_DISTANCE_DECAY.csv',index=False)
    pp=p.groupby(['patient_id','config','distance_bin','lo'],sort=False).agg(reduction=('reduction','mean'),mean_absolute_difference=('mean_absolute_difference','mean'),random_expected=('random_expected','mean'),n_roi=('roi_id','size')).reset_index()
    pp.to_csv(OUT/'PATIENT_DISTANCE_DECAY.csv',index=False)
    ss=[]
    for (config,label,lo),g in pp.groupby(['config','distance_bin','lo']):
        ci=boot(g.reduction);ss.append(dict(config=config,distance_bin=label,lo=lo,n_patients=len(g),mean_reduction=g.reduction.mean(),median_reduction=g.reduction.median(),ci_low=ci[0],ci_high=ci[1],fraction_reduction_positive=(g.reduction>0).mean(),mean_absolute_difference=g.mean_absolute_difference.mean(),mean_random_expected=g.random_expected.mean()))
    pd.DataFrame(ss).to_csv(OUT/'DISTANCE_DECAY_SUMMARY.csv',index=False)
    r=pd.DataFrame(regions);mask=r.auc_p.notna();r.loc[mask,'auc_q']=r[mask].groupby('config').auc_p.transform(bh)
    r.to_csv(OUT/'ROI_GEOMETRIC_STRUCTURE.csv',index=False)
    # Each patient's mean across available ROIs prevents multiple ROIs dominating summaries.
    rp=r.groupby(['patient_id','config']).agg(spatial_auc=('spatial_auc','mean'),gradient_cv_r2=('gradient_cv_r2','mean'),n_roi=('roi_id','size')).reset_index()
    rp.to_csv(OUT/'PATIENT_GEOMETRIC_STRUCTURE.csv',index=False)
    rss=[]
    for k in [30,100]:
        a=r[r.config.eq(f'RAMP_k{k}')];b=rp[rp.config.eq(f'RAMP_k{k}')]
        aa=a[a.spatial_auc.notna()];bb=b[b.spatial_auc.notna()]
        rss.append(dict(config=f'RAMP_k{k}',gradient_n_roi=len(a),gradient_n_patients=len(b),gradient_median_roi_r2=a.gradient_cv_r2.median(),gradient_roi_r2_ge025=int(a.gradient_cv_r2.ge(.25).sum()),
            segregation_n_roi=len(aa),segregation_n_patients=len(bb),median_roi_auc=aa.spatial_auc.median(),median_patient_auc=bb.spatial_auc.median(),mean_patient_auc=bb.spatial_auc.mean(),
            roi_auc_ge07=int(aa.spatial_auc.ge(.7).sum()),roi_auc_ge08=int(aa.spatial_auc.ge(.8).sum()),roi_auc_q_lt05=int(aa.auc_q.lt(.05).sum()),patients_mean_auc_ge07=int(bb.spatial_auc.ge(.7).sum())))
    pd.DataFrame(rss).to_csv(OUT/'GEOMETRIC_SUMMARY.csv',index=False)
    regional=[];regionmeans=[]
    for pid,g in d.groupby('patient_id',sort=True):
        counts=g.groupby('roi_id').size();good=counts.index[counts>=15];a=g[g.roi_id.isin(good)]
        if len(good)<2:continue
        labs=pd.Categorical(a.roi_id,categories=good).codes;count=np.bincount(labs);n=len(a)
        W=(labs[None,:]==np.arange(len(good))[:,None]).astype(float)/count[:,None]
        for k in [30,100]:
            z=a[f'RAMP_k{k}'].to_numpy();means=W@z;grand=z.mean();den=np.sum((z-grand)**2)
            if den<=1e-12:continue
            eta=float(np.sum(count*(means-grand)**2)/den);span=float(means.max()-means.min())
            null_span=[];null_eta=[]
            for start in range(0,499,50):
                zz=np.vstack([rng.permutation(z) for _ in range(min(50,499-start))]);mm=zz@W.T
                null_span.extend(mm.max(axis=1)-mm.min(axis=1));null_eta.extend(np.sum(count[None,:]*(mm-grand)**2,axis=1)/den)
            regional.append(dict(patient_id=pid,config=f'RAMP_k{k}',n_roi=len(good),n_vcaf=n,mean_range=span,null_range_mean=np.mean(null_span),eta2=eta,null_eta2_mean=np.mean(null_eta),range_p=(1+np.sum(np.asarray(null_span)>=span))/500))
            for roi,nn,mn in zip(good,count,means):regionmeans.append(dict(patient_id=pid,roi_id=roi,config=f'RAMP_k{k}',n_vcaf=int(nn),mean_score=float(mn)))
    reg=pd.DataFrame(regional);reg['range_q']=reg.groupby('config').range_p.transform(bh);reg.to_csv(OUT/'PATIENT_BETWEEN_ROI_STRUCTURE.csv',index=False)
    pd.DataFrame(regionmeans).to_csv(OUT/'ROI_MEANS.csv',index=False)
    regs=[]
    for config,g in reg.groupby('config'):
        regs.append(dict(config=config,n_patients=len(g),median_mean_range=g.mean_range.median(),median_null_range=g.null_range_mean.median(),median_eta2=g.eta2.median(),median_null_eta2=g.null_eta2_mean.median(),n_range_ge01=int(g.mean_range.ge(.1).sum()),n_range_q_lt05=int(g.range_q.lt(.05).sum())))
    pd.DataFrame(regs).to_csv(OUT/'BETWEEN_ROI_SUMMARY.csv',index=False)
    assert rt.sha(SOURCE)==source_hash
    rt.save_json(OUT/'VERIFICATION.json',dict(status='pass',finished_at_utc=rt.now(),source_unchanged=True,n_vcaf=len(d),self_excluded_from_spatial_neighbors=True,no_new_HLA_prediction=True,script_sha256=rt.sha(__file__)))
    print('DISTANCE\n',pd.DataFrame(ss).to_string(index=False),flush=True)
    print('GEOMETRY\n',pd.DataFrame(rss).to_string(index=False),flush=True)
    print('REGIONS\n',pd.DataFrame(regs).to_string(index=False),flush=True)
if __name__=='__main__':main()
