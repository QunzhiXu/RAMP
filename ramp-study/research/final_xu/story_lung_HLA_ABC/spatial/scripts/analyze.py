"""Spatial characterization of CURRENT three-cohort frozen HLA transfer in Cords."""
import runtime as rt
from runtime import BASE
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree, distance
from scipy.stats import spearmanr, wilcoxon

OUT=BASE/'results'; OUT.mkdir(exist_ok=True)
PRIMARY=['T cell','B cell','Myeloid','Neutrophil','Tumour','Vessel','Other CAF']
TARGETS={
    'T cell':('cell_category','T cell'), 'B cell':('cell_type','Bcell'),
    'Myeloid':('cell_type','Myeloid'), 'Neutrophil':('cell_type','Neutrophil'),
    'Tumour':('cell_category','Tumour'), 'Vessel':('cell_category','vessel'),
    'Other CAF':('other_caf',True), 'CD4 T cell':('cell_type','CD4'),
    'CD8 T cell':('cell_type','CD8'), 'Blood vessel':('cell_type','Blood'),
    'Lymphatic vessel':('cell_type','Lymphatic'), 'HEV':('cell_type','HEV'),
    'Hypoxic tumour':('cell_type','hypoxic'), 'Nonhypoxic tumour':('cell_type','normal'),
    **{t:('cell_type',t) for t in ['Collagen_CAF','IDO_CAF','PDPN_CAF','SMA_CAF','dCAF','hypoxic_CAF','hypoxic_tpCAF','iCAF','mCAF','tpCAF']}}
SEED=20260909
PATHS={'metadata':BASE.parent/'data_cords/data/CELL_METADATA.csv.gz',
       'predictions':BASE.parent/'data_cords/results/ALL_CORDS_PREDICTIONS.csv.gz',
       'cutoffs':BASE.parent/'data_cords/results/HIGH_STATE_THRESHOLDS.csv',
       'dfs_membership':BASE.parent/'survival/results/PATIENT_INPUTS.csv'}

def bh(p):
    p=np.asarray(p,float); order=np.argsort(p); n=len(p); q=np.empty(n)
    q[order]=np.minimum(1,np.minimum.accumulate((p[order]*n/np.arange(1,n+1))[::-1])[::-1]);return q

def boot(x):
    x=np.asarray(x,float);x=x[np.isfinite(x)]
    if not len(x):return (np.nan,np.nan)
    rng=np.random.default_rng(SEED)
    return tuple(np.quantile(x[rng.integers(len(x),size=(2000,len(x)))].mean(axis=1),[.025,.975]))

def summarize_patient(p, keys, value='difference'):
    rows=[]
    for key,g in p.groupby(keys,dropna=False,sort=False):
        x=g[value].to_numpy(); lo,hi=boot(x)
        row=dict(zip(keys,key if isinstance(key,tuple) else (key,)))
        row.update(n_patients=len(g),mean=float(x.mean()),median=float(np.median(x)),ci_low=lo,ci_high=hi,
                   fraction_positive=float(np.mean(x>0)),p=float(wilcoxon(x).pvalue) if np.any(x!=0) else 1.)
        for col in ['low','high']:
            if col in g: row[col]=g[col].mean()
        rows.append(row)
    return pd.DataFrame(rows)

def population_copy(df, dfs):
    a=df.copy();a['population']='all_available'
    b=df[df.patient_id.isin(dfs)].copy();b['population']='DFS_280_members'
    return pd.concat([a,b],ignore_index=True)

def prepare():
    hashes={k:rt.sha(p) for k,p in PATHS.items()}
    rt.save_json(OUT/'DESIGN.json',dict(started_at_utc=rt.now(),input_paths={k:str(p) for k,p in PATHS.items()},input_sha256=hashes,
        primary_model='Frozen three-dataset RAMP_k30; RAMP_k100 companion only; no refitting or baseline comparison.',
        population='All 527 Cords patients available; query cells are author-annotated vCAFs. Repeat summaries in fixed DFS cohort using IDs only.',
        high_state='Existing fixed original all-CAF reference cutoff, unchanged. Not the patient survival split.',
        primary_radius=40,companion_radii=[20,80],coordinate_units='Unmodified provider coordinate units; centroid distance within the same patient and ROI.',
        binary_contrast='Patient >=15 vCAFs, >=5 high and >=5 low; ROI >=3 high and >=3 low. Compare proportion with each neighbor type within radius. Average ROIs equally within patient, then patients equally.',
        continuous_contrast='ROI >=15 vCAFs; within-ROI Spearman between continuous prediction and nearest-target distance; mean valid ROIs within each patient. Negative means higher score is closer.',
        spatial_clustering='ROI >=15 vCAFs and >=10 edges. Undirected union of six-nearest-vCAF edges, distance <40. Moran I for continuous score, no binary-state eligibility required. 499 within-ROI random label permutations. Same-state mixing tested when >=3 cells of each state.',
        uncertainty='2000 patient bootstrap samples; paired Wilcoxon. BH by population/config/metric/radius/primary-secondary target family. Moran BH over all eligible ROIs per config.',
        targets=TARGETS,primary_targets=PRIMARY,seed=SEED,no_outcomes_used=True))
    d=pd.read_csv(PATHS['metadata'],dtype={'patient_id':str,'roi_id':str},float_precision='round_trip')
    pred=pd.read_csv(PATHS['predictions'],usecols=['source_row_id','matrix_row_index','patient_id','roi_id','RAMP_k30','RAMP_k100'],
                     dtype={'patient_id':str,'roi_id':str},float_precision='round_trip')
    assert len(d)==len(pred)==2849385 and d.source_row_id.is_unique and pred.source_row_id.is_unique
    assert np.array_equal(d.matrix_row_index,np.arange(len(d)))
    pred=pred.sort_values('matrix_row_index')
    for c in ['source_row_id','matrix_row_index','patient_id','roi_id']:assert np.array_equal(d[c],pred[c])
    for c in ['RAMP_k30','RAMP_k100']:d[c]=pred[c].to_numpy()
    assert np.isfinite(d[['x','y','RAMP_k30','RAMP_k100']]).all().all()
    d['other_caf']=d.cell_category.eq('Fibroblast')&~d.cell_type.eq('vCAF')
    cut=pd.read_csv(PATHS['cutoffs'],float_precision='round_trip').set_index('config').threshold.to_dict()
    dfs=set(pd.read_csv(PATHS['dfs_membership'],usecols=['patient_id'],dtype=str).patient_id)
    assert len(dfs)==280
    parts=[]; maxerr=0.; missing=[]
    for i,((pid,roi),g) in enumerate(d.groupby(['patient_id','roi_id'],sort=True)):
        v=g[g.cell_type.eq('vCAF')].copy()
        if not len(v):continue
        xy=v[['x','y']].to_numpy()
        for target,(col,label) in TARGETS.items():
            t=g.loc[g[col].eq(label),['x','y']].to_numpy()
            z=cKDTree(t).query(xy,workers=1)[0] if len(t) else np.full(len(v),np.inf)
            v['dist__'+target]=z
            missing.append(dict(patient_id=pid,roi_id=roi,target=target,n_target=len(t),n_vcaf=len(v)))
            if i in [0,100,500,900] and len(t):
                err=float(np.max(np.abs(distance.cdist(xy[:10],t).min(axis=1)-z[:10])))
                maxerr=max(maxerr,err);assert err<1e-9
        for k in [30,100]:v[f'high_k{k}']=v[f'RAMP_k{k}'].ge(cut[f'RAMP_k{k}'])
        v['in_DFS_280']=pid in dfs;parts.append(v.drop(columns='other_caf'))
        if i%200==0:print(f'Coordinates/neighbors: ROI {i+1}/1000',flush=True)
    v=pd.concat(parts,ignore_index=True)
    assert len(v)==52328
    v.to_csv(OUT/'VCAF_CURRENT_PREDICTIONS_AND_DISTANCES.csv.gz',index=False)
    pd.DataFrame(missing).to_csv(OUT/'NEIGHBOR_AVAILABILITY.csv',index=False)
    rt.save_json(OUT/'INPUT_VERIFICATION.json',dict(n_cells=len(d),n_patients=d.patient_id.nunique(),n_roi=d.groupby(['patient_id','roi_id']).ngroups,
        n_vcaf=len(v),n_vcaf_patients=v.patient_id.nunique(),n_vcaf_roi=v.groupby(['patient_id','roi_id']).ngroups,
        exact_prediction_metadata_identity=True,brute_force_distance_max_error=maxerr,cutoffs={f'RAMP_k{k}':cut[f'RAMP_k{k}'] for k in [30,100]}))
    return d,v,dfs,hashes

def neighborhoods(v,dfs):
    binary=[];continuous=[];eligibility=[]
    for k in [30,100]:
        n=v.groupby('patient_id').agg(n_vcaf=('source_row_id','size'),n_high=(f'high_k{k}','sum'))
        n['n_low']=n.n_vcaf-n.n_high;n['eligible']=n.n_vcaf.ge(15)&n.n_high.ge(5)&n.n_low.ge(5)
        n['config']=f'RAMP_k{k}';n['in_DFS_280']=n.index.isin(dfs);eligibility.append(n.reset_index())
        good=set(n.index[n.eligible])
        for (pid,roi),g in v.groupby(['patient_id','roi_id'],sort=True):
            score=g[f'RAMP_k{k}'].to_numpy();high=g[f'high_k{k}'].to_numpy();nh=int(high.sum());nl=len(g)-nh
            base=dict(patient_id=pid,roi_id=roi,config=f'RAMP_k{k}',n_vcaf=len(g),n_high=nh,n_low=nl)
            for target in TARGETS:
                dist=g['dist__'+target].to_numpy()
                assert np.isfinite(dist).all() or np.isinf(dist).all()
                if len(g)>=15 and np.isfinite(dist).all() and np.ptp(dist)>0 and np.ptp(score)>0:
                    rho=float(spearmanr(score,dist).statistic)
                    continuous.append(dict(base,target=target,rho=rho))
                if pid not in good or min(nh,nl)<3:continue
                for radius in [20,40,80]:
                    x=(dist<radius).astype(float)
                    binary.append(dict(base,target=target,metric='proximity_fraction',radius=radius,low=x[~high].mean(),high=x[high].mean()))
                if np.isfinite(dist).all():
                    for metric,x in [('mean_log1p_distance',np.log1p(dist)),('mean_distance',dist)]:
                        binary.append(dict(base,target=target,metric=metric,radius=0,low=x[~high].mean(),high=x[high].mean()))
        print(f'Neighborhood contrasts k={k}: {len(good)} patients eligible for fixed high/low comparison',flush=True)
    pd.concat(eligibility).to_csv(OUT/'PATIENT_STATE_ELIGIBILITY.csv',index=False)
    b=pd.DataFrame(binary);b['difference']=b.high-b.low;b.to_csv(OUT/'ROI_HIGH_LOW_CONTRASTS.csv',index=False)
    keys=['patient_id','config','target','metric','radius']
    bp=b.groupby(keys).agg(low=('low','mean'),high=('high','mean'),difference=('difference','mean'),n_roi=('roi_id','nunique')).reset_index()
    bp=population_copy(bp,dfs);bp.to_csv(OUT/'PATIENT_HIGH_LOW_CONTRASTS.csv',index=False)
    bs=summarize_patient(bp,['population','config','target','metric','radius'])
    bs['family']=np.where(bs.target.isin(PRIMARY),'primary','secondary')
    bs['q_bh']=bs.groupby(['population','config','metric','radius','family']).p.transform(bh)
    bs.to_csv(OUT/'HIGH_LOW_SUMMARY.csv',index=False)
    c=pd.DataFrame(continuous);c.to_csv(OUT/'ROI_CONTINUOUS_ASSOCIATIONS.csv',index=False)
    cp=c.groupby(['patient_id','config','target']).agg(difference=('rho','mean'),n_roi=('roi_id','nunique')).reset_index()
    cp=population_copy(cp,dfs);cp.to_csv(OUT/'PATIENT_CONTINUOUS_ASSOCIATIONS.csv',index=False)
    cs=summarize_patient(cp,['population','config','target'])
    cs['family']=np.where(cs.target.isin(PRIMARY),'primary','secondary')
    cs['q_bh']=cs.groupby(['population','config','family']).p.transform(bh)
    cs.to_csv(OUT/'CONTINUOUS_SUMMARY.csv',index=False)

def clustering(v,dfs):
    rows=[];rng=np.random.default_rng(SEED)
    for k in [30,100]:
        for i,((pid,roi),g) in enumerate(v.groupby(['patient_id','roi_id'],sort=True)):
            n=len(g)
            if n<15:continue
            xy=g[['x','y']].to_numpy();score=g[f'RAMP_k{k}'].to_numpy();h=g[f'high_k{k}'].to_numpy()
            dist,idx=cKDTree(xy).query(xy,k=min(7,n),workers=1)
            a=np.repeat(np.arange(n),idx.shape[1]);b=idx.ravel();keep=(a!=b)&(dist.ravel()<40)
            e=np.unique(np.sort(np.column_stack([a[keep],b[keep]]),axis=1),axis=0)
            if len(e)<10:continue
            a,b=e.T;z=score-score.mean();var=np.mean(z*z)
            if var<=1e-14:continue
            I=float(np.mean(z[a]*z[b])/var)
            mixed=min(int(h.sum()),int((~h).sum()))>=3
            same=float(np.mean(h[a]==h[b])); hh=float(np.mean(h[a]&h[b]))
            expected_same=float((h.sum()*(h.sum()-1)+(~h).sum()*((~h).sum()-1))/(n*(n-1)))
            expected_hh=float(h.sum()*(h.sum()-1)/(n*(n-1)))
            null=[];nullsame=[];nullhh=[]
            for start in range(0,499,50):
                ix=np.vstack([rng.permutation(n) for _ in range(min(50,499-start))]);zz=z[ix]
                null.extend(np.mean(zz[:,a]*zz[:,b],axis=1)/var)
                if mixed:
                    hp=h[ix];nullsame.extend(np.mean(hp[:,a]==hp[:,b],axis=1));nullhh.extend(np.mean(hp[:,a]&hp[:,b],axis=1))
            rows.append(dict(patient_id=pid,roi_id=roi,config=f'RAMP_k{k}',n_vcaf=n,n_high=int(h.sum()),n_low=int((~h).sum()),n_edges=len(e),
                moran_I=I,random_expectation=-1/(n-1),moran_excess=I+1/(n-1),moran_p=(1+np.sum(np.asarray(null)>=I))/500,
                mixed_states=mixed,same_state_fraction=same,expected_same_state=expected_same,
                same_state_excess=same-expected_same if mixed else np.nan,
                mixing_p=(1+np.sum(np.asarray(nullsame)>=same))/500 if mixed else np.nan,
                high_high_fraction=hh,expected_high_high=expected_hh,high_high_excess=hh-expected_hh if mixed else np.nan,
                high_high_p=(1+np.sum(np.asarray(nullhh)>=hh))/500 if mixed else np.nan))
            if i%200==0:print(f'Spatial permutation k={k}: ROI {i+1}/975',flush=True)
        print(f'Spatial permutations k={k} complete',flush=True)
    r=pd.DataFrame(rows);r['moran_q']=r.groupby('config').moran_p.transform(bh)
    for metric in ['mixing','high_high']:
        mask=r[f'{metric}_p'].notna();r.loc[mask,f'{metric}_q']=r.loc[mask].groupby('config')[f'{metric}_p'].transform(bh)
    r.to_csv(OUT/'ROI_CLUSTERING.csv',index=False)
    p=r.groupby(['patient_id','config']).agg(moran_I=('moran_I','mean'),moran_excess=('moran_excess','mean'),same_state_excess=('same_state_excess','mean'),high_high_excess=('high_high_excess','mean'),n_roi=('roi_id','nunique')).reset_index()
    p=population_copy(p,dfs);p.to_csv(OUT/'PATIENT_CLUSTERING.csv',index=False)
    s=[]
    for metric in ['moran_I','moran_excess','same_state_excess','high_high_excess']:
        a=summarize_patient(p[p[metric].notna()],['population','config'],metric);a['metric']=metric;s.append(a)
    pd.concat(s,ignore_index=True).to_csv(OUT/'CLUSTERING_SUMMARY.csv',index=False)

def select_maps(d,v):
    r=pd.read_csv(OUT/'ROI_CLUSTERING.csv',dtype={'patient_id':str,'roi_id':str})
    r=r[r.config.eq('RAMP_k30')].copy()
    r['balanced_count']=r[['n_high','n_low']].min(axis=1)
    rich=r[r.balanced_count.ge(10)].sort_values(['balanced_count','n_vcaf'],ascending=False).drop_duplicates('patient_id').head(3).copy()
    rich['selection']='highest balanced cell coverage'
    strong=r[r.n_vcaf.ge(50)&r.balanced_count.ge(5)].sort_values('moran_I',ascending=False).drop_duplicates('patient_id').head(6).copy()
    strong['selection']='largest Moran I among >=50 vCAFs and >=5 per state; exploratory examples'
    selected=pd.concat([rich,strong],ignore_index=True);selected.to_csv(OUT/'MAP_SELECTION.csv',index=False)
    keys=pd.MultiIndex.from_frame(selected[['patient_id','roi_id']].drop_duplicates())
    mask=pd.MultiIndex.from_frame(d[['patient_id','roi_id']]).isin(keys)
    d.loc[mask].drop(columns='other_caf').to_csv(OUT/'MAP_CONTEXT.csv.gz',index=False)

def main():
    d,v,dfs,hashes=prepare();neighborhoods(v,dfs);clustering(v,dfs);select_maps(d,v)
    assert all(rt.sha(PATHS[k])==h for k,h in hashes.items())
    rt.save_json(OUT/'RUN_STATUS.json',dict(status='complete',finished_at_utc=rt.now(),all_source_inputs_unchanged=True,
        script_sha256=rt.sha(__file__),current_model_only=True,no_refitting=True))
    s=pd.read_csv(OUT/'HIGH_LOW_SUMMARY.csv')
    print(s[s.population.eq('all_available')&s.config.eq('RAMP_k30')&s.metric.eq('proximity_fraction')&s.radius.eq(40)&s.family.eq('primary')].to_string(index=False),flush=True)
    print(pd.read_csv(OUT/'CLUSTERING_SUMMARY.csv').to_string(index=False),flush=True)
if __name__=='__main__':main()
