"""Define coordinate-only states and freeze reverse predictions before reading outcomes."""
import runtime as rt
from runtime import BASE
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree, distance
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

OUT=BASE/'results'; S=BASE.parent; SEED=20260909
CATS=['CAF','Tumour','Vessel','T_cell','Other_immune','Other']
PRIMARY=['CAF_fraction','Tumour_fraction','Immune_fraction','Vessel_fraction','Mixing_entropy','Vascular_proximity']
INPUTS={'metadata':S/'data_cords/data/CELL_METADATA.csv.gz','anchors':S/'data_umap/data/CELL_METADATA.csv.gz',
        'aligned':S/'data_umap/reference/RAMP_SHARED9.npy','umap':S/'umap/results/RAMP_UMAP.npy'}

def weights(d,k):
    w=np.exp(-(d[:,:k]/np.maximum(d[:,k-1:k],1e-8))**2)
    return w/w.sum(1,keepdims=True)

def blend(train,ix,w):
    out=np.empty((len(ix),train.shape[1]))
    for start in range(0,len(ix),2000):
        sl=slice(start,start+2000);z=train[ix[sl]];ok=np.isfinite(z)
        den=np.einsum('ij,ijh->ih',w[sl],ok)
        out[sl]=np.divide(np.einsum('ij,ijh->ih',w[sl],np.nan_to_num(z)),den,out=np.full(den.shape,np.nan),where=den>0)
    return out

def main():
    if (OUT/'PREDICTION_FREEZE.json').exists():raise SystemExit('Predictions already frozen.')
    hashes={key:rt.sha(p) for key,p in INPUTS.items()}
    rt.save(OUT/'DESIGN.json',dict(started=rt.now(),input_paths={k:str(v) for k,v in INPUTS.items()},input_sha256=hashes,
        purpose='Transfer independent spatial neighborhood states from Cords to Senosain; test held-out measured HLA-ABC and link to existing vCAF HLA/DFS story.',
        state_definition='SAME ROI, all cell types, exclude query cell itself; full ROI cells used for context. Never use any HLA value, prediction, or clinical outcome to define states.',
        radius_primary=40,radii_companion=[20,80],units='Original Cords coordinate units; not assumed micrometres.',
        continuous_states=PRIMARY,vascular_proximity='exp(-distance to nearest OTHER vessel cell / radius), 0 if none in ROI',
        mixing_entropy='Shannon entropy of six neighborhood fractions divided by log(6). Empty neighborhoods missing.',
        niches='Six KMeans clusters of square-root 40-radius six-category proportions; fitted on 28,000 balanced Cords anchor cells only; seed fixed, no outcome-based choice of K.',
        transfer='All 28,000 Cords anchors to all 28,000 Senosain anchors using frozen three-cohort RAMP shared9, Cords-source StandardScaler, Euclidean KNN, exp(-(d/d_k)^2) weights.',
        k_primary=30,k_companion=100,celltype_restriction_for_transfer=False,
        source_validation='Leave-one-Cords-donor-out, scaler fit to training donors only; report within-donor and within-fibroblast performance.',
        senosain_analysis='Primary original paper mesenchymal gate; companion broad stromal group and all cells. At least 15 cells per donor, donor paired top/bottom spatial-score tertiles; HLA patient ranks from all 1000 cells. Marker panel excludes shared9 inputs.',
        cords_analysis='All 52,328 vCAF spatial states; within-ROI and patient-level HLA associations; existing fixed 280-patient DFS cohort, continuous states per SD with original covariates. No optimized survival cutoffs.',
        reporting='Keep all states and both favorable and unfavorable results; BH within primary HLA family and exploratory protein panel; no guarantee of stronger association.',seed=SEED))
    meta=pd.read_csv(INPUTS['metadata'],dtype={'patient_id':str,'roi_id':str},low_memory=False,float_precision='round_trip')
    a=pd.read_csv(INPUTS['anchors'],dtype={'patient_id':str,'roi_id':str},low_memory=False,float_precision='round_trip')
    ci=np.flatnonzero(a.dataset.eq('Cords IMC'));si=np.flatnonzero(a.dataset.eq('Senosain CyTOF'))
    assert len(ci)==len(si)==28000
    am=a.iloc[ci];sm=a.iloc[si].copy();sm['reference_index']=si
    ids=set(am.source_row_id);q=meta[meta.source_row_id.isin(ids)|meta.cell_type.eq('vCAF')].copy().reset_index(drop=True)
    q['anchor']=q.source_row_id.isin(ids);assert q.anchor.sum()==28000 and q.cell_type.eq('vCAF').sum()==52328
    q['query_index']=np.arange(len(q)); q.to_csv(OUT/'CORDS_QUERY_METADATA.csv.gz',index=False)
    labels=meta.cell_category.replace({'Fibroblast':'CAF','vessel':'Vessel','T cell':'T_cell','Immune':'Other_immune'})
    assert set(labels)<=set(CATS)
    meta['code']=pd.Categorical(labels,categories=CATS).codes
    query_groups=q.groupby(['patient_id','roi_id'],sort=False).indices
    counts={r:np.zeros((len(q),6),np.int32) for r in [20,40,80]};vd=np.full(len(q),np.inf)
    brute_error=0;checked=0
    for no,(key,g) in enumerate(meta.groupby(['patient_id','roi_id'],sort=True)):
        if key not in query_groups:continue
        qi=query_groups[key];qg=q.iloc[qi];xy=g[['x','y']].to_numpy();qx=qg[['x','y']].to_numpy();codes=g.code.to_numpy()
        gi=g.source_row_id.to_numpy();neighbors=cKDTree(xy).query_ball_point(qx,80)
        for j,ix in enumerate(neighbors):
            ix=np.asarray(ix,dtype=int);ix=ix[gi[ix]!=qg.source_row_id.iloc[j]]
            d2=((xy[ix]-qx[j])**2).sum(1)
            for radius in counts:counts[radius][qi[j]]=np.bincount(codes[ix[d2<radius**2]],minlength=6)
        vessels=g[g.cell_category.eq('vessel')]
        if len(vessels):
            nnear=min(2,len(vessels));d,j=cKDTree(vessels[['x','y']].to_numpy()).query(qx,k=nnear)
            if nnear==1:d=d[:,None];j=j[:,None]
            same=vessels.source_row_id.to_numpy()[j]==qg.source_row_id.to_numpy()[:,None];d[same]=np.inf;vd[qi]=d.min(1)
        if no%100==0:
            d=distance.cdist(qx[:5],xy);valid=gi[None,:]!=qg.source_row_id.to_numpy()[:5,None]
            for radius in counts:
                expected=np.stack([((d<radius)&valid&(codes==c)[None,:]).sum(1) for c in range(6)],1)
                brute_error=max(brute_error,int(np.abs(expected-counts[radius][qi[:5]]).max()));checked+=len(expected)
        if no%200==0:print(f'Coordinate-only neighborhoods: {no+1}/1000 ROIs',flush=True)
    assert brute_error==0
    features=[];names=[];fractions={}
    for r,z in counts.items():
        n=z.sum(1);p=np.divide(z,n[:,None],out=np.full(z.shape,np.nan),where=n[:,None]>0);fractions[r]=p
        values=np.column_stack([p[:,0],p[:,1],p[:,3]+p[:,4],p[:,2],-np.nansum(p*np.log(np.maximum(p,1e-12)),axis=1)/np.log(6),np.exp(-vd/r)])
        values[n==0,:5]=np.nan
        features.append(values);names += [f'r{r}__{c}' for c in PRIMARY]
        np.savez_compressed(OUT/f'NEIGHBOR_COUNTS_r{r}.npz',source_row_id=q.source_row_id.to_numpy(),counts=z,categories=np.array(CATS))
    loc=pd.Index(q.source_row_id).get_indexer(am.source_row_id);assert (loc>=0).all()
    fitloc=loc[np.isfinite(fractions[40][loc]).all(1)]
    km=KMeans(n_clusters=6,n_init=30,random_state=SEED).fit(np.sqrt(fractions[40][fitloc]))
    niches=np.full(len(q),-1);valid=np.isfinite(fractions[40]).all(1);niches[valid]=km.predict(np.sqrt(fractions[40][valid]))
    # IDs remain fixed; descriptive names are derived from centroid composition before any outcomes.
    centroid=[]
    for j in range(6):
        ix=fitloc[km.labels_==j];p=fractions[40][ix].mean(0);order=np.argsort(-p)
        name=f'N{j+1}_{CATS[order[0]]}_{CATS[order[1]]}'
        centroid.append(dict(niche=j,niche_id=f'N{j+1}',name=name,n_anchors=len(ix),**dict(zip(CATS,p))))
        one=(niches==j).astype(float);one[~valid]=np.nan;features.append(one[:,None]);names.append(f'niche__N{j+1}')
    pd.DataFrame(centroid).to_csv(OUT/'NICHE_DEFINITIONS.csv',index=False)
    np.savez_compressed(OUT/'SPATIAL_MODEL.npz',cluster_centers=km.cluster_centers_,categories=np.array(CATS))
    f=np.column_stack(features);q['niche']=niches;q['nearest_other_vessel_distance']=vd
    for j,n in enumerate(names):q[n]=f[:,j]
    q.to_csv(OUT/'CORDS_SPATIAL_STATES.csv.gz',index=False)
    train=f[loc]
    x=np.load(INPUTS['aligned']);scale=StandardScaler().fit(x[ci]);cx=scale.transform(x[ci]);sx=scale.transform(x[si])
    nn=NearestNeighbors(n_neighbors=100,n_jobs=1).fit(cx);d,ix=nn.kneighbors(sx)
    result=sm.copy()
    for k in [30,100]:
        w=weights(d,k);pred=blend(train,ix[:,:k],w)
        for j,n in enumerate(names):result[f'k{k}__{n}']=pred[:,j]
        assert np.allclose(pred[:,-6:].sum(1),1)
    result['distance_30']=d[:,29];result['distance_100']=d[:,99]
    result.to_csv(OUT/'SENOSAIN_REVERSE_PREDICTIONS.csv.gz',index=False)
    np.savez_compressed(OUT/'REVERSE_NEIGHBORS.npz',source_reference_index=ci,target_reference_index=si,distances=d,indices=ix,scale_mean=scale.mean_,scale_scale=scale.scale_)
    # Held-out-donor predictability without HLA or survival responses.
    cv=np.full_like(train,np.nan);metrics=[];donors=am.patient_id.to_numpy()
    primary=[n for n in names if n.startswith('r40__') or n.startswith('niche__')]
    for no,pid in enumerate(np.unique(donors)):
        te=donors==pid;tr=~te;sc=StandardScaler().fit(x[ci[tr]])
        nd,ni=NearestNeighbors(n_neighbors=30,n_jobs=1).fit(sc.transform(x[ci[tr]])).kneighbors(sc.transform(x[ci[te]]))
        cv[te]=blend(train[tr],ni,weights(nd,30))
        for pop,mask in [('all',te),('fibroblast',te&am.cell_category.eq('Fibroblast').to_numpy()),('vCAF',te&am.provider_cell_type.eq('vCAF').to_numpy())]:
            if mask.sum()<15:continue
            for n in primary:
                j=names.index(n);valid=mask&np.isfinite(train[:,j])&np.isfinite(cv[:,j]);truth=train[valid,j];pred=cv[valid,j]
                rho=float(spearmanr(truth,pred).statistic) if np.ptp(truth)>0 and np.ptp(pred)>0 else np.nan
                r2=1-np.sum((truth-pred)**2)/np.sum((truth-truth.mean())**2) if np.ptp(truth)>0 else np.nan
                metrics.append(dict(patient_id=pid,population=pop,state=n,n_cells=int(mask.sum()),rho=rho,R2=r2,MAE=np.mean(np.abs(truth-pred))))
        if no%7==0:print(f'Leave-one-donor-out spatial transfer: {no+1}/28',flush=True)
    np.savez_compressed(OUT/'CORDS_LODO_PREDICTIONS.npz',source_row_id=am.source_row_id.to_numpy(),truth=train,prediction=cv,states=np.array(names))
    pd.DataFrame(metrics).to_csv(OUT/'CORDS_LODO_METRICS.csv',index=False)
    assert all(rt.sha(INPUTS[key])==v for key,v in hashes.items())
    rt.save(OUT/'PREDICTION_FREEZE.json',dict(finished=rt.now(),new_HLA_or_outcome_values_read=False,source_hashes_unchanged=True,
        output_sha256={n:rt.sha(OUT/n) for n in ['CORDS_SPATIAL_STATES.csv.gz','SENOSAIN_REVERSE_PREDICTIONS.csv.gz','NICHE_DEFINITIONS.csv']},
        n_queries=len(q),n_source=28000,n_target=28000,brute_force_count_error=brute_error,brute_force_queries=checked,
        n_empty_neighborhoods={str(r):int((z.sum(1)==0).sum()) for r,z in counts.items()}))
    print(pd.DataFrame(centroid).to_string(index=False),flush=True)

if __name__=='__main__':main()
