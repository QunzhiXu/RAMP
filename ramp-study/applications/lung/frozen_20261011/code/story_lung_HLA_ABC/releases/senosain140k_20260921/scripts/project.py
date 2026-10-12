import runtime as rt
from runtime import BASE,STORY,MARKERS,CONFIGS
import sys,json,time,shutil
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from ramp import ramp_ver2 as ramp
from frozen_maps import apply_maps

def anchor_map(vals,tr,tb,ta,unit,profiles,known_baseline=None):
    b=np.empty_like(vals);a=np.empty_like(vals)
    for j in range(9):
        v=tr[:,j];shift=-float(v.min())+.01 if v.min()<0 else 0.
        shifted=v+shift;med=float(np.median(shifted));cofactor=max(med if med else float(shifted.mean()),1e-6)
        z=np.arcsinh(shifted/cofactor);scale=max(float(z.std()),1e-9);offset=float(np.median(z-scale*tb[:,j]))
        assert np.max(np.abs((z-offset)/scale-tb[:,j]))<1e-9
        b[:,j]=(np.arcsinh((vals[:,j]+shift)/cofactor)-offset)/scale
        key=f'{unit}::{j}'
        if key not in profiles:a[:,j]=b[:,j];continue
        u=ramp.within_unit_percentiles(tb[:,j]);order=np.argsort(v,kind='stable')
        unique,first=np.unique(v[order],return_index=True)
        pu=u[order][first];qu=np.interp(vals[:,j],unique,pu,left=pu[0],right=pu[-1])
        a[:,j]=ramp.bounded_profile_mapping(b[:,j],qu,profiles[key])
    if known_baseline is not None:assert np.max(np.abs(b-known_baseline))<1e-9
    return b,a

def predict(models,y,b,a):
    p=np.empty((len(a),4))
    for method,x in [('RAMP',a),('baseline',b)]:
        scale,nn=models[method]
        for start in range(0,len(x),5000):
            sl=slice(start,min(len(x),start+5000));dist,ind=nn.kneighbors(scale.transform(x[sl]))
            for k in [30,100]:
                d=dist[:,:k];w=np.exp(-(d/np.maximum(d[:,-1:],1e-8))**2);w/=w.sum(1,keepdims=True)
                p[sl,CONFIGS.index(f'{method}_k{k}')]=np.sum(w*y[ind[:,:k]],axis=1)
    assert np.isfinite(p).all() and p.min()>=0 and p.max()<=1
    return p

def main(target):
    assert target in ['validation','cords'];out=BASE/target
    if (out/'COMPLETE.json').exists():print(target,'already complete',flush=True);return
    assert (BASE/'alignment/COMPLETE.json').exists()
    start=time.time();(out/'patients').mkdir(exist_ok=True)
    md=pd.read_csv(BASE/'inputs/CELL_METADATA.csv.gz',dtype={'patient_id':str})
    raw=np.load(BASE/'inputs/RAW_SHARED9.npy');bref=np.load(BASE/'alignment/baseline_SHARED9.npy');aref=np.load(BASE/'alignment/RAMP_SHARED9.npy')
    y=pd.read_csv(BASE/'inputs/SOURCE_RESPONSE.csv.gz',float_precision='round_trip').source_patient_rank.to_numpy()
    si=np.flatnonzero(md.dataset.eq('Senosain CyTOF'));assert len(y)==len(si)==28000
    models={}
    for name,x in [('RAMP',aref[si]),('baseline',bref[si])]:
        scale=StandardScaler().fit(x);models[name]=(scale,NearestNeighbors(n_neighbors=100,n_jobs=1).fit(scale.transform(x)))
    with np.load(BASE/'alignment/REFERENCE_PROFILES.npz') as z:profiles={k:z[k] for k in z.files}
    with np.load(BASE/'alignment/FROZEN_PLATFORM_MAPS.npz') as z:maps=[(z[f'x{j}'],z[f'y{j}']) for j in range(9)]
    if target=='cords':
        meta_path=STORY/'data_cords/data/CELL_METADATA.csv.gz';raw_path=STORY/'data_cords/data/RAW_SHARED9.npy'
        bp=STORY/'data_cords/data/baseline_SHARED9.npy';known=np.load(bp,mmap_mode='r');dataset='Cords IMC';expected=2849385;total=527
    else:
        meta_path=STORY/'validation/data/ZHENG_METADATA.csv.gz';raw_path=STORY/'validation/data/ZHENG_SHARED9.npy'
        known=None;dataset='Zheng mIF';expected=693693;total=28
    paths=[meta_path,raw_path,BASE/'inputs/CELL_METADATA.csv.gz',BASE/'inputs/SOURCE_RESPONSE.csv.gz',BASE/'alignment/baseline_SHARED9.npy',BASE/'alignment/RAMP_SHARED9.npy',BASE/'alignment/REFERENCE_PROFILES.npz',BASE/'alignment/FROZEN_PLATFORM_MAPS.npz']
    if known is not None:paths.append(bp)
    h=rt.hashes(paths);signature=rt.sha(__file__)+'|'+rt.sha(BASE/'scripts/frozen_maps.py')+'|'+rt.sha(BASE/'scripts/runtime.py')+'|'+json.dumps(h,sort_keys=True)
    full=pd.read_csv(meta_path,dtype={'patient_id':str});xx=np.load(raw_path,mmap_mode='r')
    assert len(full)==expected and full.source_row_id.is_unique and full.patient_id.nunique()==total
    shutil.copy2(meta_path,out/'CELL_METADATA.csv.gz')
    output=np.lib.format.open_memmap(out/'PREDICTIONS.npy',mode='w+',dtype='float64',shape=(len(full),4))
    aligned=np.lib.format.open_memmap(out/'RAMP_SHARED9.npy',mode='w+',dtype='float64',shape=(len(full),9))
    allb=np.lib.format.open_memmap(out/'baseline_SHARED9.npy',mode='w+',dtype='float64',shape=(len(full),9)) if target=='validation' else None
    refs={p:np.flatnonzero(md.dataset.eq(dataset)&md.patient_id.eq(p)) for p in md.loc[md.dataset.eq(dataset),'patient_id'].unique()}
    audits=[]
    for no,(pid,ix) in enumerate(full.groupby('patient_id',sort=True).indices.items(),1):
        vals=np.asarray(xx[ix]);g=full.iloc[ix];unit=f'{dataset}|{pid}';path=out/'patients'/f'{pid}.npz';jp=path.with_suffix('.json')
        if path.exists() and jp.exists():
            rec=json.loads(jp.read_text());assert rec['signature']==signature
            with np.load(path) as z:
                assert np.array_equal(z['source_row_id'],g.source_row_id)
                b=z['baseline'] if target=='validation' else np.asarray(known[ix]);a=z['aligned'];pred=z['prediction']
        else:
            err=0.
            if pid in refs:
                ri=refs[pid];pos=pd.Index(g.source_row_id).get_indexer(md.iloc[ri].source_row_id)
                assert len(pos)==1000 and (pos>=0).all() and np.max(np.abs(vals[pos]-raw[ri]))<1e-9
                b,a=anchor_map(vals,raw[ri],bref[ri],aref[ri],unit,profiles,None if known is None else known[ix])
                err=max(float(np.abs(b[pos]-bref[ri]).max()),float(np.abs(a[pos]-aref[ri]).max()));assert err<1e-8
                b[pos]=bref[ri];a[pos]=aref[ri]
            else:
                assert target=='cords';b=np.array(known[ix]);a=apply_maps(b,maps)
            inversions=0
            for j in range(9):
                order=np.argsort(vals[:,j],kind='stable')
                inversions+=int(((np.diff(vals[order,j])>0)&(np.diff(a[order,j]) < -1e-10)).sum())
            disp=float(np.abs(a-b).max());assert inversions==0 and disp<=.75+1e-9 and np.isfinite(a).all()
            pred=predict(models,y,b,a)
            extra={'baseline':b} if target=='validation' else {}
            np.savez_compressed(path,source_row_id=g.source_row_id.to_numpy(),prediction=pred,aligned=a,**extra)
            rec=dict(patient_id=pid,n_cells=len(g),reference_patient=pid in refs,anchor_error=err,rank_inversions=inversions,max_displacement=disp,signature=signature)
            rt.save(jp,rec)
        output[ix]=pred;aligned[ix]=a
        if allb is not None:allb[ix]=b
        audits.append({k:v for k,v in rec.items() if k!='signature'})
        if no%10==0 or no==1 or no==total:
            output.flush();aligned.flush();print(target,no,'/',total,'elapsed',round(time.time()-start,1),flush=True)
            rt.status(f'{target} prediction',patients_done=no,patients_total=total,elapsed_seconds=time.time()-start)
    output.flush();aligned.flush()
    if allb is not None:allb.flush()
    pd.DataFrame(audits).to_csv(out/'PROJECTION_AUDIT.csv',index=False)
    rt.unchanged(h)
    rt.save(out/'COMPLETE.json',dict(seconds=time.time()-start,configs=CONFIGS,n_cells=expected,n_patients=total,input_hashes=h,source_files_unchanged=True,prediction_sha256=rt.sha(out/'PREDICTIONS.npy'),no_clinical_outcomes_or_HLA_A_used=True,code_sha256=rt.sha(__file__)))
    print(target,'complete',flush=True)
if __name__=='__main__':main(sys.argv[1])
