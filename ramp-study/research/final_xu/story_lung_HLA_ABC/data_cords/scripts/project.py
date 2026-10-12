"""Project all Cords cells into the frozen 28+28+28 reference and rebuild vCAF scores."""
import package_runtime as rt
from package_runtime import BASE
ROOT=BASE.parents[1]
STORY=ROOT/'story_test_lung'
import sys,json,time,fcntl,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata,spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
sys.path.insert(0,str(BASE.parent/'data_umap/vendor'))
from ramp import ramp_ver2 as ramp
from frozen_maps import apply_maps,fit_maps

ALIGN=BASE.parent/'data_umap/reference'
DATA=BASE.parent/'data_umap/data'
PROJ=BASE/'reference'
OUT=BASE/'results'
FULL=ROOT/'overnight0909/results/fullcell_transfer/cords_full527_allcells_joint10.csv.gz'
MARKERS=['CD3','CD4','CD8','CD31','KI67','VIMENTIN','EPI','CD45','HLA_DR']
CONFIGS=[f'{method}_k{k}' for method in ['RAMP','baseline'] for k in [30,100]]

def random_rows(pid,n):
    key=f'20260909|cells|Cords IMC|{pid}'.encode()
    rng=np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8],'little'))
    return np.sort(rng.choice(n,1000,replace=False))

def main():
    OUT.mkdir(exist_ok=True);(OUT/'patients').mkdir(exist_ok=True)
    lock=(OUT/'RUNNING.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('Projection already running.')
    if (OUT/'PREPARATION_VERIFICATION.json').exists():raise SystemExit('Completed projection exists; do not overwrite.')
    start=time.time()
    paths=[FULL,DATA/'RAW_SHARED9.npy',DATA/'CELL_METADATA.csv.gz',ALIGN/'baseline_SHARED9.npy',ALIGN/'RAMP_SHARED9.npy',
           DATA/'DESIGN.json',ALIGN/'REFERENCE_PROFILES.npz',PROJ/'FROZEN_PLATFORM_MAPS.npz',PROJ/'FIDELITY_SUMMARY.csv',
           PROJ/'THRESHOLD_CAF_CELL_IDS.csv.gz',BASE.parent/'validation/data/SOURCE_RESPONSE.csv.gz',
           STORY/'shared_inputs/senosain/senosain_exact70k_raw35.npy',STORY/'shared_inputs/senosain/senosain_raw35_markers.csv',
           BASE.parent/'data_umap/vendor/ramp/ramp.py',BASE.parent/'data_umap/vendor/ramp/ramp_ver2.py',Path(__file__)]
    hashes={str(p):rt.sha(p) for p in paths}
    signature=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    md=pd.read_csv(DATA/'CELL_METADATA.csv.gz',dtype={'patient_id':str},low_memory=False)
    raw=np.load(DATA/'RAW_SHARED9.npy');base=np.load(ALIGN/'baseline_SHARED9.npy');aligned=np.load(ALIGN/'RAMP_SHARED9.npy')
    si=np.flatnonzero(md.dataset.eq('Senosain CyTOF'));ci=np.flatnonzero(md.dataset.eq('Cords IMC'))
    reference_ids=set(md.iloc[ci].patient_id)
    ref_indices={pid:np.flatnonzero(md.dataset.eq('Cords IMC')&md.patient_id.eq(pid)) for pid in reference_ids}
    design=json.loads((DATA/'DESIGN.json').read_text())
    rt.save_json(OUT/'DESIGN.json',{'time_utc':rt.now(),'reference':'Frozen 28 Senosain + 28 Cords + 28 Zheng, common9 RAMP 2.3.0; no additional donors enter alignment.',
        'source_cells':'All 28,000 Senosain cells; no CAF gate','target_cells':'All 2,849,385 Cords cells across 527 patients',
        'reference_Cords':'Read existing profiles from data_umap; apply each reference patient profile to all remaining cells. No RAMP refit.',
        'new_Cords':'Fixed platform-level monotone baseline-to-RAMP projection distilled only from the 28 Cords anchors; patient-specific KMEANSLOW uses 1000 fixed random cells.',
        'projection_distinction':'This is a frozen-reference extension, not a joint RAMP refit including 527 Cords donors. New-donor maps approximate the original donor-specific family.',
        'source_and_Zheng_space':'Unchanged; prior all-source Zheng HLA-A validation remains attached to the same predictor and reference.',
        'primary':'RAMP k30; k100 sensitivity; exact same-input KMEANSLOW baseline retained',
        'survival':'Original 280 patients / 96 DFS events; original author vCAF; no T-neighborhood requirement.',
        'states':'Mean over all vCAFs; fraction above recomputed median among the original 150,024 sampled author CAFs. Same definition, new score scale, no optimized threshold.',
        'adjustment':'Primary matched model retains all previous covariates. A fixed companion substitutes the new global-CAF median for the old global-CAF score.',
        'scope_changes_vs_old':'New shared9 panel (HLA-DR, no Histone H3), all-source pool rather than CAF-source, 28 rather than 70 source donors, Zheng bridge, frozen projection. Changes cannot be attributed only to alignment.',
        'input_sha256':hashes})

    profile_path=ALIGN/'REFERENCE_PROFILES.npz'
    with np.load(profile_path) as z:profiles={key:z[key] for key in z.files}
    with np.load(PROJ/'FROZEN_PLATFORM_MAPS.npz') as z:maps=[(z[f'x{j}'],z[f'y{j}']) for j in range(9)]
    refit_maps=fit_maps(base[ci],aligned[ci])
    map_error=max(float(np.abs(v-w).max()) for pair,other in zip(maps,refit_maps) for v,w in zip(pair,other))
    assert map_error==0
    rt.save_json(OUT/'REFERENCE_REPRODUCTION.json',{'platform_map_max_error':map_error,'new_donors_in_alignment':0,
        'reference_cells':84000,'existing_profiles_read_from':str(profile_path),'RAMP_refitted':False,'time_utc':rt.now()})

    source=STORY/'shared_inputs/senosain'
    names=pd.read_csv(source/'senosain_raw35_markers.csv').marker.tolist()
    sr=np.load(source/'senosain_exact70k_raw35.npy',mmap_mode='r')
    sm=md.iloc[si];yraw=sr[sm.source_row_id.to_numpy(int),names.index('HLA_ABC')];y=np.empty(len(si))
    for pid in sm.patient_id.unique():
        ix=np.flatnonzero(sm.patient_id.eq(pid));y[ix]=rankdata(yraw[ix])/(len(ix)+1)
    response=sm[['cell_key','source_row_id','patient_id']].copy()
    response['observed_HLA_ABC']=yraw;response['source_patient_rank']=y
    check=pd.read_csv(BASE.parent/'validation/data/SOURCE_RESPONSE.csv.gz',float_precision='round_trip')
    assert np.array_equal(check.cell_key,response.cell_key) and np.array_equal(check.source_patient_rank,y)
    response.to_csv(BASE/'data/SOURCE_RESPONSE.csv.gz',index=False,compression={'method':'gzip','compresslevel':3,'mtime':0})
    models={}
    for method,x in [('RAMP',aligned[si]),('baseline',base[si])]:
        scale=StandardScaler().fit(x);nn=NearestNeighbors(n_neighbors=100,n_jobs=1).fit(scale.transform(x));models[method]=(scale,nn)
    cols=['patient_id','source_row_id','cell_category','cell_type','cell_subtype','roi_id','x','y',*MARKERS]
    full=pd.read_csv(FULL,usecols=cols,dtype={'patient_id':str},low_memory=False)
    assert len(full)==2849385 and full.patient_id.nunique()==527 and full.source_row_id.is_unique
    full['matrix_row_index']=np.arange(len(full))
    meta=full[[c for c in full.columns if c not in MARKERS]].copy()
    meta['reference_donor']=meta.patient_id.isin(reference_ids)
    meta.to_csv(BASE/'data/CELL_METADATA.csv.gz',index=False,compression={'method':'gzip','compresslevel':3,'mtime':0})
    np.save(BASE/'data/RAW_SHARED9.npy',full[MARKERS].to_numpy(float))
    pd.DataFrame({'marker':MARKERS}).to_csv(BASE/'data/MARKERS.csv',index=False)
    all_b=np.lib.format.open_memmap(BASE/'data/baseline_SHARED9.npy',mode='w+',dtype=np.float64,shape=(len(full),9))
    all_a=np.lib.format.open_memmap(BASE/'data/RAMP_SHARED9.npy',mode='w+',dtype=np.float64,shape=(len(full),9))
    audits=[];parts=[];sample_ids=[];parameter_rows=[]
    for no,(pid,ix) in enumerate(full.groupby('patient_id',sort=True).indices.items(),1):
        g=full.iloc[ix].reset_index(drop=True);vals=g[MARKERS].to_numpy(float);unit=f'Cords IMC|{pid}'
        saved=OUT/'patients'/f'{pid}.npz';audit_path=OUT/'patients'/f'{pid}.json'
        if saved.exists() and audit_path.exists():
            with np.load(saved) as z:
                assert np.array_equal(z['source_row_id'],g.source_row_id.to_numpy())
                pp=pd.DataFrame({c:z[c] for c in CONFIGS});train_ids=z['sampled_source_row_id']
                bq=z['baseline'];aq=z['RAMP'];parameters=z['parameters']
            audit=json.loads(audit_path.read_text());assert audit['run_signature']==signature
        else:
            if pid in reference_ids:
                ri=ref_indices[pid]
                anchor_pos=pd.Index(g.source_row_id).get_indexer(md.iloc[ri].source_row_id)
                assert (anchor_pos>=0).all() and np.max(np.abs(vals[anchor_pos]-raw[ri]))<1e-9
                tr=vals[anchor_pos];tb=base[ri];ta=aligned[ri]
            else:
                anchor_pos=random_rows(pid,len(g));tr=vals[anchor_pos]
                tb=ramp.kmeanslow_global(tr,np.repeat(unit,1000),np.repeat('Cords IMC',1000),MARKERS)
            train_ids=g.iloc[anchor_pos].source_row_id.to_numpy()
            bq=np.empty_like(vals);aq=np.empty_like(vals);recon_error=0.;outside=0;parameters=[]
            for j in range(9):
                v=tr[:,j];shift=-float(v.min())+.01 if v.min()<0 else 0.
                shifted=v+shift;median=float(np.median(shifted));cofactor=max(float(shifted.mean()) if median==0 else median,1e-6)
                trans=np.arcsinh(shifted/cofactor);scale=max(float(trans.std()),1e-9)
                offset=float(np.median(trans-scale*tb[:,j]))
                parameters.append([shift,cofactor,scale,offset])
                recon_error=max(recon_error,float(np.abs((trans-offset)/scale-tb[:,j]).max()))
                bq[:,j]=(np.arcsinh((vals[:,j]+shift)/cofactor)-offset)/scale
                outside+=int(((vals[:,j]<v.min())|(vals[:,j]>v.max())).sum())
                if pid in reference_ids:
                    key=f'{unit}::{j}'
                    if key not in profiles:aq[:,j]=bq[:,j];continue
                    u=ramp.within_unit_percentiles(tb[:,j]);order=np.argsort(v,kind='stable')
                    unique,first=np.unique(v[order],return_index=True);pu=u[order][first]
                    qu=np.interp(vals[:,j],unique,pu,left=pu[0],right=pu[-1])
                    aq[:,j]=ramp.bounded_profile_mapping(bq[:,j],qu,profiles[key])
            assert recon_error<1e-9
            anchor_error=np.nan
            if pid in reference_ids:
                anchor_error=max(float(np.abs(bq[anchor_pos]-tb).max()),float(np.abs(aq[anchor_pos]-ta).max()))
                assert anchor_error<1e-8
                bq[anchor_pos]=tb;aq[anchor_pos]=ta
            else:aq=apply_maps(bq,maps)
            for j in range(9):
                order=np.argsort(vals[:,j],kind='stable')
                assert not np.any((np.diff(vals[:,j][order])>0)&(np.diff(aq[:,j][order]) < -1e-10))
            assert np.isfinite(aq).all() and np.max(np.abs(aq-bq))<=.75+1e-9
            pp={}
            for method,x in [('RAMP',aq),('baseline',bq)]:
                scale,nn=models[method];dist,inds=nn.kneighbors(scale.transform(x))
                for k in [30,100]:
                    dd=dist[:,:k];w=np.exp(-(dd/np.maximum(dd[:,-1:],1e-8))**2);w/=w.sum(axis=1,keepdims=True)
                    pp[f'{method}_k{k}']=np.sum(w*y[inds[:,:k]],axis=1)
            pp=pd.DataFrame(pp)
            assert np.isfinite(pp.to_numpy()).all()
            np.savez_compressed(saved,source_row_id=g.source_row_id.to_numpy(),sampled_source_row_id=train_ids,baseline=bq,RAMP=aq,parameters=np.asarray(parameters),**{c:pp[c].to_numpy() for c in CONFIGS})
            audit={'run_signature':signature,'patient_id':pid,'n_cells':len(g),'mapping':'existing_donor_frozen_profile' if pid in reference_ids else 'new_donor_platform_projection',
                   'baseline_reconstruction_error':recon_error,'anchor_coordinate_error':None if not np.isfinite(anchor_error) else anchor_error,
                   'out_of_sample_marker_support_values':outside,'strict_rank_inversions':0}
            rt.save_json(audit_path,audit)
        all_b[ix]=bq;all_a[ix]=aq
        for marker,par in zip(MARKERS,parameters):
            parameter_rows.append({'patient_id':pid,'marker':marker,'shift':par[0],'cofactor':par[1],'scale':par[2],'offset':par[3],'reference_donor':pid in reference_ids})
        pp=pd.concat([g[['source_row_id','patient_id','cell_type','cell_category','roi_id','matrix_row_index']],pp],axis=1)
        pp['mapping']=audit['mapping'];parts.append(pp);audits.append(audit)
        sample_ids.extend(train_ids.tolist())
        if no%10==0 or no==1:
            print(no,'/527 patients;',pid,'elapsed',round(time.time()-start,1),flush=True)
            rt.save_json(OUT/'RUN_STATUS.json',{'phase':'projecting and predicting','patients_done':no,'patients_total':527,'updated_utc':rt.now()})
    all_b.flush();all_a.flush();del all_b,all_a
    pd.DataFrame(parameter_rows).to_csv(OUT/'BASELINE_PARAMETERS.csv',index=False)
    pred=pd.concat(parts,ignore_index=True)
    assert len(pred)==len(full) and pred.source_row_id.is_unique
    pred.to_csv(OUT/'ALL_CORDS_PREDICTIONS.csv.gz',index=False,compression={'method':'gzip','compresslevel':3,'mtime':0})
    pd.DataFrame(audits).to_csv(OUT/'PROJECTION_AUDIT.csv',index=False)
    pd.DataFrame({'source_row_id':sample_ids}).to_csv(OUT/'BASELINE_FIT_CELL_IDS.csv.gz',index=False)
    sample=pd.read_csv(PROJ/'THRESHOLD_CAF_CELL_IDS.csv.gz')
    sample_set=set(sample.source_row_id)
    sf=pred[pred.source_row_id.isin(sample_set)]
    assert len(sf)==150024 and sf.cell_category.eq('Fibroblast').all()
    cuts={c:float(sf[c].median()) for c in CONFIGS}
    pd.DataFrame([{'config':c,'threshold':v,'reference_sample_CAFs':len(sf)} for c,v in cuts.items()]).to_csv(OUT/'HIGH_STATE_THRESHOLDS.csv',index=False)
    # Freeze molecular scores and thresholds before opening survival input.
    rt.save_json(OUT/'PREDICTION_FREEZE.json',{'time_utc':rt.now(),'all_cell_predictions_sha256':rt.sha(OUT/'ALL_CORDS_PREDICTIONS.csv.gz'),
        'thresholds':cuts,'clinical_outcomes_used_in_prediction':False,'original_reference_hashes_unchanged':True})
    cf=pred[pred.cell_category.eq('Fibroblast')];vf=cf[cf.cell_type.eq('vCAF')]
    assert len(cf)==530184 and len(vf)==52328
    counts=cf.groupby('patient_id').size().rename('n_full_caf')
    frames=[]
    for config in CONFIGS:
        method,kk=config.split('_k')
        v=vf[['patient_id',config]].copy();v['high']=v[config]>=cuts[config]
        pv=v.groupby('patient_id').agg(n_vcaf=(config,'size'),all_mean=(config,'mean'),all_high=('high','mean'))
        pv=counts.to_frame().join(pv);pv['n_vcaf']=pv.n_vcaf.fillna(0).astype(int)
        pv['vcaf_fraction']=pv.n_vcaf/pv.n_full_caf
        pv['current_global_CAF_median']=sf.groupby('patient_id')[config].median()
        pv['reference_donor']=pv.index.isin(reference_ids);pv['config']=config;pv['method']=method;pv['k']=int(kk)
        frames.append(pv.reset_index())
    scores=pd.concat(frames,ignore_index=True);scores.to_csv(OUT/'ALL_PATIENT_SCORES.csv',index=False)
    coverage=pred.groupby('patient_id').agg(n_cells=('source_row_id','size'),mapping=('mapping','first'))
    coverage['n_CAF']=counts;coverage['n_vCAF']=vf.groupby('patient_id').size().reindex(coverage.index,fill_value=0)
    coverage['reference_donor']=coverage.index.isin(reference_ids)
    coverage.to_csv(OUT/'PATIENT_COVERAGE.csv')
    unchanged={str(p):rt.sha(p)==hashes[str(p)] for p in paths};assert all(unchanged.values())
    rt.save_json(OUT/'PREPARATION_VERIFICATION.json',{'status':'PASS','time_utc':rt.now(),'inputs_unchanged':unchanged,
        'original_reference_patients':28,'new_projected_patients':499,'Cords_patients':527,'all_Cords_cells':2849385,
        'all_CAF_cells':530184,'all_vCAF_cells':52328,'source_response_exactly_matches_Zheng_validation':True,
        'strict_rank_inversions':0,'thresholds':cuts,'RAMP_refitted':False,'clinical_outcomes_used':False,
        'saved_full_expression_matrices':['RAW_SHARED9.npy','baseline_SHARED9.npy','RAMP_SHARED9.npy'],
        'script_sha256':rt.sha(__file__),'seconds':time.time()-start})
    print('Full projection and molecular aggregation complete',round(time.time()-start,1),'seconds',flush=True)

if __name__=='__main__':main()
