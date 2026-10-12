"""Apply existing donor maps to full Zheng tissue cores, preserving the frozen reference."""
import package_runtime as rt
from package_runtime import BASE
ROOT=BASE.parents[1]
STORY=ROOT/'story_test_lung'
import sys,json,time,fcntl
import numpy as np
import pandas as pd
import h5py
from scipy.stats import rankdata,spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
sys.path.insert(0,str(BASE.parent/'data_umap/vendor'))
from ramp import ramp_ver2 as ramp

OUT=BASE/'results'
ALIGN=BASE.parent/'data_umap/reference'
DATA=BASE.parent/'data_umap/data'
VALID=BASE/'reference'
SOURCE=STORY/'shared_inputs/senosain'
ZFILE=ROOT/'raw_data/lung_cancer/Zheng2025/mIF_data.h5ad'
PROFILES=ALIGN/'REFERENCE_PROFILES.npz'
MARKERS=['CD3','CD4','CD8','CD31','KI67','VIMENTIN','EPI','CD45','HLA_DR']
ZMARKERS=['CD3e','CD4','CD8','CD31','Ki67','Vimentin','Pan-Cytokeratin','CD45','HLA-DR']
CONFIGS=[f'{m}_k{k}' for m in ['RAMP','baseline'] for k in [30,100]]

def obs_read(f,key):
    d=f['obs'][key]
    if isinstance(d,h5py.Group):
        cats=d['categories'].asstr()[:];codes=d['codes'][:]
        return np.where(codes>=0,cats[np.maximum(codes,0)],None)
    return d.asstr()[:] if d.dtype.kind in 'OSU' else d[:]

def main():
    OUT.mkdir(exist_ok=True);(OUT/'patients').mkdir(exist_ok=True)
    lock=(OUT/'RUNNING.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('Full Zheng prediction already running.')
    if (OUT/'PREPARATION_VERIFICATION.json').exists():raise SystemExit('Completed full Zheng prediction exists; do not overwrite.')
    start=time.time()
    inputs=[ZFILE,PROFILES,DATA/'CELL_METADATA.csv.gz',DATA/'RAW_SHARED9.npy',ALIGN/'baseline_SHARED9.npy',
        ALIGN/'RAMP_SHARED9.npy',DATA/'DESIGN.json',VALID/'ANCHOR_PREDICTIONS_AND_TRUTH.csv.gz',
        SOURCE/'senosain_exact70k_raw35.npy',SOURCE/'senosain_raw35_markers.csv',BASE.parent/'data_umap/vendor/ramp/ramp.py',BASE.parent/'data_umap/vendor/ramp/ramp_ver2.py']
    inputs += [BASE.parent/'umap/results/RAMP_UMAP.npy', __file__]
    hashes={str(p):rt.sha(p) for p in inputs}
    import hashlib
    run_signature=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    md=pd.read_csv(DATA/'CELL_METADATA.csv.gz',dtype={'patient_id':str},low_memory=False)
    si=np.flatnonzero(md.dataset.eq('Senosain CyTOF'));zi=np.flatnonzero(md.dataset.eq('Zheng mIF'))
    sm=md.iloc[si];zm=md.iloc[zi]
    raw=np.load(DATA/'RAW_SHARED9.npy');baseline=np.load(ALIGN/'baseline_SHARED9.npy');aligned=np.load(ALIGN/'RAMP_SHARED9.npy')
    with np.load(PROFILES) as z:profiles={k:z[k] for k in z.files if k.startswith('Zheng mIF|')}
    assert len(profiles)==252
    with h5py.File(ZFILE,'r') as f:
        meta=pd.DataFrame({'source_row_id':np.arange(f['X'].shape[0]),'CellID':obs_read(f,'CellID'),
            'patient_id':obs_read(f,'patient_id'),'core_id':obs_read(f,'core_id'),'cell_type':obs_read(f,'phenotype'),
            'x_um':obs_read(f,'Centroid X µm'),'y_um':obs_read(f,'Centroid Y µm')})
        full=meta[meta.patient_id.isin(set(zm.patient_id))].copy().reset_index(drop=True)
        names=f['var/_index'].asstr()[:].tolist();cols=[names.index(n) for n in ZMARKERS]
        sorted_cols=sorted(cols)
        # Read only the nine alignment inputs, explicitly excluding the target HLA-A channel.
        xx=f['X'][:,sorted_cols]
        xx=xx[full.source_row_id.to_numpy()][:,[sorted_cols.index(c) for c in cols]].astype(float)
    (BASE/'data').mkdir(exist_ok=True)
    np.save(BASE/'data/ZHENG_SHARED9.npy',xx)
    full.to_csv(BASE/'data/ZHENG_METADATA.csv.gz',index=False,compression={'method':'gzip','compresslevel':3,'mtime':0})
    assert len(full)==693693 and full.patient_id.nunique()==28 and full.core_id.nunique()==119
    assert full.groupby('core_id').patient_id.nunique().max()==1
    assert np.isfinite(full[['x_um','y_um']]).all().all() and full.source_row_id.is_unique
    counts=full.groupby(['patient_id','core_id']).size().rename('n_full').reset_index()
    sc=zm.groupby(['patient_id','roi_id']).size().rename('n_original_sample').reset_index().rename(columns={'roi_id':'core_id'})
    counts=counts.merge(sc,on=['patient_id','core_id'],how='left',validate='one_to_one')
    counts['original_sample_coverage']=counts.n_original_sample/counts.n_full
    counts.to_csv(OUT/'CORE_COVERAGE.csv',index=False)
    show=counts.sort_values(['n_full','patient_id','core_id'],ascending=[False,True,True]).drop_duplicates('patient_id').head(3).copy()
    show['selection']='Largest cell count, one core per patient; selected before new predictions or HLA-A truth were read'
    show.to_csv(OUT/'MAIN_EXAMPLE_SELECTION.csv',index=False)
    rt.save_json(OUT/'DESIGN.json',{'time_utc':rt.now(),'task':'Measured-versus-predicted paired plots on real Zheng tissue coordinates',
        'reference':'Existing 28+28+28 three-platform reference; no alignment refit or new donor calibration',
        'target':'All 693,693 cells in all 119 cores of the same 28 Zheng patients',
        'transfer':'All 28,000 Senosain source cells -> all Zheng target cells; no fibroblast gate',
        'application':'Freeze original per-patient KMEANSLOW parameters and RAMP profiles; interpolate the original sample percentile map to additional cells',
        'predictor':'Identical source-only StandardScaler and exp(-(d/d_k)^2) KNN weights; k30 primary, k100 sensitivity; baseline retained',
        'input_markers':MARKERS,'target_excluded_from_inputs':'HLA-A / HLA-ABC',
        'source_response':'Within-Senosain-patient HLA-ABC rank/(1000+1), unchanged',
        'display':'Left: measured HLA-A within full-patient rank. Right: original transferred HLA-ABC rank, described as an HLA-A proxy. Shared 0..1 viridis; never rerank/rescale predictions.',
        'spatial_plot':'Identical cell IDs, raw centroid coordinates, drawing order, limits and marker size; equal aspect; image-coordinate Y direction downward; no smoothing or UMAP',
        'main_examples':show[['patient_id','core_id','n_full']].to_dict('records'),'all_cores_gallery':True,
        'clinical_outcomes_used':False,'HLA_A_used_for_prediction_training':False,'historical_validation_already_seen':True,'presentation_core':'P_26 / C_30 retained from prior exploratory selection; no new core or color search','input_sha256':hashes})
    source_names=pd.read_csv(SOURCE/'senosain_raw35_markers.csv').marker.tolist()
    source_raw=np.load(SOURCE/'senosain_exact70k_raw35.npy',mmap_mode='r')
    yr=source_raw[sm.source_row_id.to_numpy(int),source_names.index('HLA_ABC')];target=np.empty(len(si))
    for pid in sm.patient_id.unique():
        ix=np.flatnonzero(sm.patient_id.eq(pid));target[ix]=rankdata(yr[ix])/(len(ix)+1)
    response=sm[['cell_key','source_row_id','patient_id']].copy()
    response['observed_HLA_ABC']=yr;response['source_patient_rank']=target
    response.to_csv(BASE/'data/SOURCE_RESPONSE.csv.gz',index=False,compression={'method':'gzip','compresslevel':3,'mtime':0})
    models={}
    for method,x in [('RAMP',aligned[si]),('baseline',baseline[si])]:
        scale=StandardScaler().fit(x);models[method]=(scale,NearestNeighbors(n_neighbors=100,n_jobs=1).fit(scale.transform(x)))
    output=np.empty((len(full),4));audit=[]
    for no,(pid,ix) in enumerate(full.groupby('patient_id',sort=True).indices.items(),1):
        g=full.iloc[ix];vals=xx[ix];unit=f'Zheng mIF|{pid}'
        checkpoint=OUT/'patients'/f'{pid}.npz';audit_path=OUT/'patients'/f'{pid}.json'
        if checkpoint.exists() and audit_path.exists():
            with np.load(checkpoint) as z:
                assert np.array_equal(z['source_row_id'],g.source_row_id.to_numpy())
                output[ix]=z['prediction']
            rec=json.loads(audit_path.read_text())
            assert rec['run_signature']==run_signature,'Checkpoint inputs or code changed' 
        else:
            ri=np.flatnonzero(md.dataset.eq('Zheng mIF')&md.patient_id.eq(pid))
            loc=pd.Index(g.source_row_id).get_indexer(md.iloc[ri].source_row_id)
            assert len(loc)==1000 and (loc>=0).all()
            assert np.max(np.abs(vals[loc]-raw[ri]))<1e-9
            tr=vals[loc];tb=baseline[ri];ta=aligned[ri];bq=np.empty_like(vals);aq=np.empty_like(vals)
            reconstruction_error=0.;outside=0
            for j in range(9):
                v=tr[:,j];shift=-float(v.min())+.01 if v.min()<0 else 0.
                shifted=v+shift;med=float(np.median(shifted));cofactor=max(float(shifted.mean()) if med==0 else med,1e-6)
                t=np.arcsinh(shifted/cofactor);scale=max(float(t.std()),1e-9);offset=float(np.median(t-scale*tb[:,j]))
                reconstruction_error=max(reconstruction_error,float(np.max(np.abs((t-offset)/scale-tb[:,j]))))
                bq[:,j]=(np.arcsinh((vals[:,j]+shift)/cofactor)-offset)/scale
                outside+=int(((vals[:,j]<v.min())|(vals[:,j]>v.max())).sum())
                u=ramp.within_unit_percentiles(tb[:,j]);order=np.argsort(v,kind='stable')
                uv,first=np.unique(v[order],return_index=True);pu=u[order][first]
                qu=np.interp(vals[:,j],uv,pu,left=pu[0],right=pu[-1])
                aq[:,j]=ramp.bounded_profile_mapping(bq[:,j],qu,profiles[f'{unit}::{j}'])
            err=max(float(np.abs(bq[loc]-tb).max()),float(np.abs(aq[loc]-ta).max()))
            assert reconstruction_error<1e-9 and err<1e-8
            bq[loc]=tb;aq[loc]=ta
            inversions=0
            for j in range(9):
                order=np.argsort(vals[:,j],kind='stable')
                inversions+=int(((np.diff(vals[:,j][order])>0)&(np.diff(aq[:,j][order]) < -1e-10)).sum())
            displacement=float(np.abs(aq-bq).max());assert inversions==0 and displacement<=.75+1e-9
            for method,x in [('RAMP',aq),('baseline',bq)]:
                scale,nn=models[method];dist,inds=nn.kneighbors(scale.transform(x))
                for k in [30,100]:
                    dd=dist[:,:k];w=np.exp(-(dd/np.maximum(dd[:,-1:],1e-8))**2);w/=w.sum(axis=1,keepdims=True)
                    output[ix,CONFIGS.index(f'{method}_k{k}')]=np.sum(w*target[inds[:,:k]],axis=1)
            np.savez_compressed(checkpoint,source_row_id=g.source_row_id.to_numpy(),prediction=output[ix])
            rec={'run_signature':run_signature,'patient_id':pid,'n_full_cells':len(g),'n_anchor_cells':1000,'n_cores':g.core_id.nunique(),
                'anchor_coordinate_max_error':err,'baseline_reconstruction_error':reconstruction_error,
                'strict_rank_inversions':inversions,'max_displacement':displacement,'outside_anchor_marker_support':outside}
            rt.save_json(audit_path,rec)
        audit.append(rec)
        print(f'{no}/28 Zheng patients: {pid}, {len(g):,} cells; {time.time()-start:.1f}s',flush=True)
        rt.save_json(OUT/'RUN_STATUS.json',{'phase':'full-cell prediction','patients_done':no,'patients_total':28,'updated_utc':rt.now()})
    assert np.isfinite(output).all() and output.min()>=0 and output.max()<=1
    for j,col in enumerate(CONFIGS):full[col]=output[:,j]
    full.to_csv(OUT/'PREDICTIONS_BEFORE_TRUTH.csv.gz',index=False,compression={'method':'gzip','compresslevel':3,'mtime':0})
    freeze_hash=rt.sha(OUT/'PREDICTIONS_BEFORE_TRUTH.csv.gz')
    rt.save_json(OUT/'PREDICTION_FREEZE.json',{'time_utc':rt.now(),'sha256':freeze_hash,'new_HLA_A_values_read':False,'source_response':'HLA_ABC'})
    # Open the held-out channel only after freezing all full-cell predictions.
    with h5py.File(ZFILE,'r') as f:
        measured=f['X'][:,names.index('HLA-A')][full.source_row_id.to_numpy()]
    assert np.isfinite(measured).all() and (measured>=0).all()
    full['observed_HLA_A']=measured;rank=np.empty(len(full))
    for pid,ix in full.groupby('patient_id',sort=False).indices.items():rank[ix]=rankdata(measured[ix])/(len(ix)+1)
    full['observed_HLA_A_patient_rank']=rank
    full['original_reference_cell']=full.source_row_id.isin(set(zm.source_row_id))
    old=pd.read_csv(VALID/'ANCHOR_PREDICTIONS_AND_TRUTH.csv.gz',dtype={'patient_id':str})
    check=full.set_index('source_row_id').loc[old.source_row_id];anchor_pred_errors={}
    assert np.array_equal(check.patient_id.to_numpy(),old.patient_id.to_numpy())
    assert np.array_equal(check.core_id.to_numpy(),old.roi_id.to_numpy())
    assert np.allclose(check.observed_HLA_A,old.observed_HLA_A,rtol=0,atol=1e-12)
    for conf in CONFIGS:
        method,kk=conf.split('_k');err=float(np.max(np.abs(check[conf].to_numpy()-old[f'all_source__{method}__k{kk}'].to_numpy())))
        anchor_pred_errors[conf]=err;assert err<1e-12
    coords=zm.set_index('source_row_id').loc[old.source_row_id]
    coordinate_error=max(float(np.abs(check.x_um.to_numpy()-coords.x.to_numpy()).max()),float(np.abs(check.y_um.to_numpy()-coords.y.to_numpy()).max()))
    assert coordinate_error==0
    full.to_csv(OUT/'PREDICTIONS_AND_TRUTH.csv.gz',index=False,compression={'method':'gzip','compresslevel':3,'mtime':0})
    pd.DataFrame(audit).to_csv(OUT/'PROJECTION_AUDIT.csv',index=False)
    metrics=[]
    for level,keys in [('core',['patient_id','core_id']),('patient',['patient_id'])]:
        for key,g in full.groupby(keys,sort=True):
            pid=key[0];cid=key[1] if level=='core' else ''
            for conf in CONFIGS:
                metrics.append({'level':level,'patient_id':pid,'core_id':cid,'config':conf,'n_cells':len(g),
                    'cell_spearman':float(spearmanr(g.observed_HLA_A,g[conf]).statistic),
                    'rank_MAE':float(np.abs(g.observed_HLA_A_patient_rank-g[conf]).mean()),
                    'prediction_mean':float(g[conf].mean()),'prediction_sd':float(g[conf].std()),
                    'observed_rank_mean':float(g.observed_HLA_A_patient_rank.mean())})
    pd.DataFrame(metrics).to_csv(OUT/'METRICS.csv',index=False)
    unchanged={p:rt.sha(p)==h for p,h in hashes.items()};assert all(unchanged.values())
    assert rt.sha(OUT/'PREDICTIONS_BEFORE_TRUTH.csv.gz')==freeze_hash
    rt.save_json(OUT/'PREPARATION_VERIFICATION.json',{'status':'PASS','time_utc':rt.now(),'seconds':time.time()-start,
        'n_cells':len(full),'n_patients':28,'n_cores':119,'unchanged_input_files':unchanged,
        'original_anchor_prediction_errors':anchor_pred_errors,'original_anchor_coordinate_error':coordinate_error,
        'strict_rank_inversions':0,'source_cells':28000,'source_and_target_all_cell_types':True,
        'alignment_refitted':False,'clinical_outcomes_used':False,'script_sha256':rt.sha(__file__)})
    rt.save_json(OUT/'RUN_STATUS.json',{'phase':'predictions_complete','updated_utc':rt.now(),'patients_done':28,'patients_total':28})
    print('Full Zheng prediction and verification complete',round(time.time()-start,1),flush=True)

if __name__=='__main__':main()
