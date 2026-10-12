"""Extend the displayed ROI's neighborhood composition to every Cords patient."""
import runtime as rt
from runtime import BASE
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree, distance
from scipy.stats import spearmanr, wilcoxon

OUT=BASE/'results/neighborhood_composition';OUT.mkdir(exist_ok=True)
RAW=['Other CAF','Tumour','Vessel','T cell','Immune','Other']
PRIMARY=['Other CAF','Vessel','Immune total','Tumour','Other']
TARGETS=PRIMARY+['T cell','Immune']
SEED=20260909
PATHS={'metadata':BASE.parent/'data_cords/data/CELL_METADATA.csv.gz',
       'vcaf_predictions':BASE/'results/VCAF_CURRENT_PREDICTIONS_AND_DISTANCES.csv.gz',
       'cutoffs':BASE.parent/'data_cords/results/HIGH_STATE_THRESHOLDS.csv',
       'dfs_ids':BASE.parent/'survival/results/PATIENT_INPUTS.csv',
       'example':BASE/'results/EXAMPLE_175_10_NEIGHBOR_COMPOSITION.csv'}

def bh(p):
    p=np.asarray(p,float);idx=np.argsort(p);n=len(p);q=np.empty(n)
    q[idx]=np.minimum(1,np.minimum.accumulate((p[idx]*n/np.arange(1,n+1))[::-1])[::-1]);return q

def boot(x):
    x=np.asarray(x,float);rng=np.random.default_rng(SEED)
    if not len(x):return (np.nan,np.nan)
    return tuple(np.quantile(x[rng.integers(len(x),size=(2000,len(x)))].mean(axis=1),[.025,.975]))

def pops(df,dfs):
    a=df.copy();a['population']='all_available'
    b=df[df.patient_id.isin(dfs)].copy();b['population']='DFS_280_members'
    return pd.concat([a,b],ignore_index=True)

def summarize(p,keys,col='difference'):
    rows=[]
    for key,g in p.groupby(keys,sort=False):
        a=g[col].to_numpy();lo,hi=boot(a)
        row=dict(zip(keys,key));row.update(n_patients=len(g),mean=float(a.mean()),median=float(np.median(a)),
            ci_low=lo,ci_high=hi,n_positive=int(np.sum(a>1e-12)),n_negative=int(np.sum(a< -1e-12)),
            n_tied=int(np.sum(np.abs(a)<=1e-12)),fraction_positive=float(np.mean(a>1e-12)),fraction_negative=float(np.mean(a< -1e-12)),
            wilcoxon_p=float(wilcoxon(a).pvalue) if np.any(a!=0) else 1.)
        for c in ['low','high']:
            if c in g:row[c]=float(g[c].mean())
        rows.append(row)
    s=pd.DataFrame(rows);s['family']=np.where(s.target.isin(PRIMARY),'primary_composition','immune_components')
    group=[x for x in keys if x!='target']+['family']
    s['q_bh']=s.groupby(group).wilcoxon_p.transform(bh)
    return s

def main():
    hashes={k:rt.sha(p) for k,p in PATHS.items()}
    rt.save_json(OUT/'DESIGN.json',dict(started_at_utc=rt.now(),input_paths={k:str(p) for k,p in PATHS.items()},input_sha256=hashes,
        prediction='Existing current three-cohort RAMP k30 primary and k100 companion; no new fitting or threshold selection.',
        query='All author-annotated vCAFs in all 527 available Cords patients; three patients without vCAFs recorded in coverage.',
        neighbors='All non-vCAF cells in the SAME patient and ROI. Strict centroid distance <r. Exclude all vCAFs from the denominator.',
        composition='For each vCAF, count each neighbor category / all non-vCAF neighbors. Mean cell fractions within each state and ROI. Empty neighborhoods have NaN fractions, not zero.',
        primary_radius=40,companion_radii=[20,80],coordinate_units='Unchanged provider units.',
        high_low='Use existing all-CAF frozen cell-state thresholds, not the patient DFS grouping.',
        primary_contrast='Patient >=15 vCAFs and >=5 cells of each state; included ROI >=3 cells WITH NONEMPTY neighborhoods of each state. Pair states within ROI, average eligible ROIs equally within patient, then patients equally.',
        relaxed_contrast='All same-ROI pairs with at least one nonempty cell of each state. No patient minimum; retain all available within-ROI mixed-state patients.',
        pooled_companion='Within each patient, pool its cells across ROIs, compare high and low state mean fractions if both exist. This does not control ROI composition.',
        continuous_companion='Within-ROI Spearman between continuous score and composition fraction; >=15 nonempty vCAFs; constant fractions yield rho=0 and are retained as no within-ROI association. Average ROIs then patients.',
        summaries='All available patients and fixed 280-patient DFS membership; read IDs only, no outcomes. 2000 patient bootstrap replicates, paired Wilcoxon, BH within config/radius/population/comparison and target family.',
        primary_categories=PRIMARY,raw_categories=RAW,immune_total='Provider Immune plus T cell; the five primary fractions sum to 1.',seed=SEED))
    m=pd.read_csv(PATHS['metadata'],usecols=['source_row_id','patient_id','roi_id','x','y','cell_type','cell_category'],dtype={'patient_id':str,'roi_id':str},float_precision='round_trip')
    v=pd.read_csv(PATHS['vcaf_predictions'],usecols=['source_row_id','patient_id','roi_id','x','y','RAMP_k30','RAMP_k100','high_k30','high_k100'],dtype={'patient_id':str,'roi_id':str},float_precision='round_trip')
    assert len(m)==2849385 and len(v)==52328 and m.source_row_id.is_unique and v.source_row_id.is_unique
    original=m[m.cell_type.eq('vCAF')].set_index('source_row_id').sort_index()
    query=v.set_index('source_row_id').sort_index()
    assert original.index.equals(query.index)
    for key in ['patient_id','roi_id','x','y']:assert np.array_equal(original[key],query[key]),key
    cuts=pd.read_csv(PATHS['cutoffs'],float_precision='round_trip').set_index('config').threshold
    for k in [30,100]:assert np.array_equal(v[f'high_k{k}'],v[f'RAMP_k{k}'].ge(cuts[f'RAMP_k{k}']))
    dfs=set(pd.read_csv(PATHS['dfs_ids'],usecols=['patient_id'],dtype=str).patient_id);assert len(dfs)==280
    vg={key:g for key,g in v.groupby(['patient_id','roi_id'],sort=False)}
    rows=[];max_brute_err=0.;roicount=0
    for i,((pid,roi),c) in enumerate(m.groupby(['patient_id','roi_id'],sort=True)):
        if (pid,roi) not in vg:continue
        g=vg[(pid,roi)];qxy=g[['x','y']].to_numpy()
        ref=c[~c.cell_type.eq('vCAF')];rxy=ref[['x','y']].to_numpy()
        labels=ref.cell_category.replace({'Fibroblast':'Other CAF','vessel':'Vessel'}).to_numpy()
        assert set(labels)<=set(RAW)
        codes=pd.Categorical(labels,categories=RAW).codes
        nn=cKDTree(rxy).query_ball_point(qxy,80)
        counts={radius:np.zeros((len(g),len(RAW)),dtype=np.int32) for radius in [20,40,80]}
        for j,ix in enumerate(nn):
            ix=np.asarray(ix,dtype=np.int64)
            d=np.sqrt(np.sum((rxy[ix]-qxy[j])**2,axis=1))
            for radius in counts:counts[radius][j]=np.bincount(codes[ix[d<radius]],minlength=len(RAW))
        if i in [0,100,500,900]:
            brute=distance.cdist(qxy[:8],rxy)
            for radius,z in counts.items():
                for code in range(len(RAW)):
                    expected=((brute<radius)&(codes==code)[None,:]).sum(axis=1)
                    max_brute_err=max(max_brute_err,float(np.abs(expected-z[:8,code]).max(initial=0)))
            assert max_brute_err==0
        for radius,z in counts.items():
            a=g.copy();a['radius']=radius;a['n_neighbors']=z.sum(axis=1)
            for j,target in enumerate(RAW):
                a['count__'+target]=z[:,j]
                a['fraction__'+target]=np.divide(z[:,j],a.n_neighbors,out=np.full(len(a),np.nan),where=a.n_neighbors.gt(0))
            a['fraction__Immune total']=a['fraction__Immune']+a['fraction__T cell']
            assert np.allclose(a.loc[a.n_neighbors.gt(0),['fraction__'+t for t in PRIMARY]].sum(axis=1),1)
            rows.append(a)
        roicount+=1
        if i%200==0:print(f'Neighborhood counts: ROI {i+1}/1000',flush=True)
    cells=pd.concat(rows,ignore_index=True);assert len(cells)==len(v)*3
    cells.to_csv(OUT/'VCAF_NEIGHBOR_COMPOSITION.csv.gz',index=False)
    rows=[]
    for k in [30,100]:
        n=v.groupby('patient_id').agg(n_vcaf=('source_row_id','size'),n_high=(f'high_k{k}','sum'))
        n=n.reindex(sorted(m.patient_id.unique()),fill_value=0);n['n_low']=n.n_vcaf-n.n_high
        n['eligible_primary']=n.n_vcaf.ge(15)&n.n_high.ge(5)&n.n_low.ge(5)
        n['has_both_states']=n.n_high.gt(0)&n.n_low.gt(0);n['in_DFS_280']=n.index.isin(dfs);n['config']=f'RAMP_k{k}'
        rows.append(n.reset_index(names='patient_id'))
    coverage=pd.concat(rows,ignore_index=True);coverage.to_csv(OUT/'PATIENT_COVERAGE.csv',index=False)
    roirows=[];staterows=[];controws=[]
    for k in [30,100]:
        eligible=set(coverage.loc[coverage.config.eq(f'RAMP_k{k}')&coverage.eligible_primary,'patient_id'])
        for (pid,roi,radius),full in cells.groupby(['patient_id','roi_id','radius'],sort=True):
            g=full[full.n_neighbors.gt(0)];h=g[f'high_k{k}'].to_numpy();nhi=int(h.sum());nlo=len(h)-nhi
            if not len(g):continue
            values=g[['fraction__'+t for t in TARGETS]].to_numpy()
            base=dict(patient_id=pid,roi_id=roi,radius=radius,config=f'RAMP_k{k}',n_high=nhi,n_low=nlo)
            for j,target in enumerate(TARGETS):
                for state,mask in [('high',h),('low',~h)]:
                    if np.any(mask):staterows.append(dict(base,target=target,state=state,mean_fraction=values[mask,j].mean(),n_cells=int(mask.sum())))
                if min(nhi,nlo)>=1:
                    low=float(values[~h,j].mean());high=float(values[h,j].mean())
                    roirows.append(dict(base,target=target,low=low,high=high,difference=high-low,primary_eligible=pid in eligible and min(nhi,nlo)>=3))
                if len(g)>=15 and np.ptp(g[f'RAMP_k{k}'])>0:
                    rho=float(spearmanr(g[f'RAMP_k{k}'],values[:,j]).statistic) if np.ptp(values[:,j])>0 else 0.
                    controws.append(dict(base,target=target,rho=rho,constant_fraction=bool(np.ptp(values[:,j])==0),n_vcaf=len(g)))
        print(f'State and continuous comparisons: k={k}',flush=True)
    r=pd.DataFrame(roirows);r.to_csv(OUT/'ROI_PAIRED_CONTRASTS.csv',index=False)
    states=pd.DataFrame(staterows);states.to_csv(OUT/'ALL_ROI_STATE_COMPOSITION.csv',index=False)
    pat=[]
    for label,a in [('within_ROI_primary',r[r.primary_eligible]),('within_ROI_all_mixed',r)]:
        p=a.groupby(['patient_id','config','radius','target']).agg(low=('low','mean'),high=('high','mean'),difference=('difference','mean'),n_roi=('roi_id','nunique')).reset_index()
        p['comparison']=label;pat.append(p)
    # Cell-weighted patient means preserve the original per-cell definition while pooling ROIs.
    states['weighted_sum']=states.mean_fraction*states.n_cells
    sp=states.groupby(['patient_id','config','radius','target','state']).agg(total=('weighted_sum','sum'),n_cells=('n_cells','sum'),n_roi=('roi_id','nunique')).reset_index()
    sp['mean_fraction']=sp.total/sp.n_cells;sp.to_csv(OUT/'ALL_PATIENT_STATE_COMPOSITION.csv',index=False)
    pooled=sp.pivot(index=['patient_id','config','radius','target'],columns='state',values='mean_fraction').dropna().reset_index()
    pooled['difference']=pooled.high-pooled.low;pooled['comparison']='within_patient_pooled';pooled['n_roi']=np.nan;pat.append(pooled)
    p=pops(pd.concat(pat,ignore_index=True),dfs);p.to_csv(OUT/'PATIENT_PAIRED_CONTRASTS.csv',index=False)
    summary=summarize(p,['population','config','radius','comparison','target']);summary.to_csv(OUT/'COMPOSITION_SUMMARY.csv',index=False)
    c=pd.DataFrame(controws);c.to_csv(OUT/'ROI_CONTINUOUS_ASSOCIATIONS.csv',index=False)
    cp=c.groupby(['patient_id','config','radius','target']).agg(difference=('rho','mean'),n_roi=('roi_id','nunique')).reset_index()
    cp=pops(cp,dfs);cp.to_csv(OUT/'PATIENT_CONTINUOUS_ASSOCIATIONS.csv',index=False)
    cs=summarize(cp,['population','config','radius','target']);cs.to_csv(OUT/'CONTINUOUS_SUMMARY.csv',index=False)
    ex=states[states.patient_id.eq('175_10')&states.roi_id.eq('175_A3,3')&states.target.isin(RAW)]
    old=pd.read_csv(PATHS['example'],float_precision='round_trip')
    joined=ex.merge(old,on=['config','radius','state','target'],suffixes=('_new','_old'),validate='one_to_one')
    err=float(np.max(np.abs(joined.mean_fraction_new-joined.mean_fraction_old)));assert len(joined)==72 and err<1e-12
    assert all(rt.sha(PATHS[k])==h for k,h in hashes.items())
    rt.save_json(OUT/'VERIFICATION.json',dict(status='pass',finished_at_utc=rt.now(),n_input_cells=len(m),n_patients=m.patient_id.nunique(),n_roi=m.groupby(['patient_id','roi_id']).ngroups,
        n_vcaf=len(v),n_vcaf_patients=v.patient_id.nunique(),n_vcaf_roi=roicount,metadata_and_prediction_identity='exact',high_state_labels='match frozen cutoffs',
        primary_fractions_sum_to_one=True,brute_force_neighbor_count_max_error=max_brute_err,previous_example_max_error=err,previous_example_rows_checked=len(joined),
        source_hashes_unchanged=True,empty_neighborhoods_by_radius={str(r):int((a.n_neighbors==0).sum()) for r,a in cells.groupby('radius')},script_sha256=rt.sha(__file__)))
    print(summary[summary.population.eq('all_available')&summary.config.eq('RAMP_k30')&summary.radius.eq(40)&summary.comparison.eq('within_ROI_primary')].to_string(index=False),flush=True)
if __name__=='__main__':main()
