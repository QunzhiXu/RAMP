"""Apply the existing CD19 model and frozen preprocessing to all H14 cells."""
from pathlib import Path
import os, sys, json, hashlib
BASE = Path(__file__).resolve().parents[1]
STORY = BASE.parents[1]
ROOT = STORY.parent
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','LOKY_MAX_CPU_COUNT']:
    os.environ[k] = '1'
sys.dont_write_bytecode = True
sys.path.insert(0, str(STORY/'prediction'))
sys.path.insert(0, str(ROOT/'algorithm'))
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
import masked_marker_model as modeldef
from ramp import ramp_ver2 as ramp

def sha(p):
    with open(p,'rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def metrics(y, score):
    y=np.asarray(y,bool); p=np.asarray(score)>1.4
    tp=int((y&p).sum()); fp=int((~y&p).sum()); fn=int((y&~p).sum()); tn=int((~y&~p).sum())
    return dict(n_cells=len(y),n_b=int(y.sum()),n_high=int(p.sum()),tp=tp,fp=fp,fn=fn,tn=tn,
                precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),f1=2*tp/max(1,2*tp+fp+fn),
                iou=tp/max(1,tp+fp+fn),auroc=float(roc_auc_score(y,score)),ap=float(average_precision_score(y,score)))

def main():
    sources=[STORY/'sampled_data'/x/'source_input_15.csv.gz' for x in ['bm_codex','triana_abseq','bm_cytof']]
    sources += [STORY/'prediction/output/masked_marker_model.pt', STORY/'prediction/output/codex_cd19_prediction.npz',STORY/'prediction/masked_marker_model.py', ROOT/'algorithm/ramp/ramp.py']
    hashes={str(p):sha(p) for p in sources}
    markers=list(modeldef.NORMAL_MARKERS)
    frames=[pd.read_csv(p) for p in sources[:3]]
    frame=pd.concat(frames,ignore_index=True)
    raw=frame[markers].to_numpy(float); baseline=np.empty_like(raw)
    units=frame.unit_key.to_numpy(str); params=[]
    for unit in pd.unique(units):
        ix=np.flatnonzero(units==unit); dataset=frame.dataset.iloc[ix[0]]
        print('Recovering frozen parameters:',unit,flush=True)
        for j,marker in enumerate(markers):
            v=raw[ix,j]; shift=-float(v.min())+.01 if v.min()<0 else 0.
            shifted=v+shift
            median,mean=float(np.median(shifted)),float(np.mean(shifted))
            cofactor=max(mean if median==0 else median,1e-6)
            z=np.arcsinh(shifted/cofactor); scale=max(float(np.std(z)),1e-9)
            fit=KMeans(n_clusters=2,init='k-means++',n_init=20,max_iter=500,tol=1e-6,algorithm='lloyd',
                       random_state=ramp.stable_seed('KMEANSLOW',dataset,unit,marker)).fit(z[:,None])
            centers=fit.cluster_centers_.ravel(); lower=int(np.argmin(centers))
            support=np.bincount(fit.labels_,minlength=2); minimum=max(20,int(np.ceil(.005*len(ix))))
            valid=(np.isfinite(centers).all() and support[lower]>=minimum and support[1-lower]>=minimum
                   and fit.n_iter_<500 and abs(np.diff(np.sort(centers))[0])>1e-10)
            low=float(z[fit.labels_==lower].mean()) if valid else float(z.mean())
            baseline[ix,j]=(z-low)/scale
            params.append(dict(unit=unit,marker=marker,shift=shift,cofactor=cofactor,scale=scale,low_mean=low))
    scaler=StandardScaler().fit(baseline)
    features=scaler.transform(baseline).astype(np.float32)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    model=modeldef.MaskedMarkerModel(14,15,3)
    model.load_state_dict(torch.load(sources[3],map_location='cpu',weights_only=True)); model.eval()
    codex_rows=np.flatnonzero(frame.dataset.eq('BM CODEX'))
    with torch.no_grad(): reproduced,_,_=modeldef.predict_cd19(model,features,codex_rows)
    frozen=np.load(sources[4])['predicted_kmeanslow']
    err=float(np.max(np.abs(reproduced-frozen)))
    assert err<2e-5, f'Frozen model mismatch {err}'
    full=pd.read_csv(BASE/'data/H14_full_source15.csv.gz')
    md=pd.read_csv(BASE/'data/H14_full_metadata.csv.gz')
    assert full.cell_id.equals(md.cell_id) and full.cell_id.is_unique
    unit='BM CODEX|SB67_NBM28_H14_CODEX_Mesmer'
    hparams={p['marker']:p for p in params if p['unit']==unit}
    z=np.empty((len(full),14))
    for j,marker in enumerate(markers):
        p=hparams[marker]
        z[:,j]=(np.arcsinh((full[marker].to_numpy()+p['shift'])/p['cofactor'])-p['low_mean'])/p['scale']
    newfeatures=scaler.transform(z).astype(np.float32)
    with torch.no_grad(): prediction,triana,cytof=modeldef.predict_cd19(model,newfeatures,np.arange(len(full)))
    out=md.copy(); out['predicted_cd19']=prediction;out['triana_component']=triana;out['cytof_component']=cytof
    out['observed_cd19_source']=full.CD19.to_numpy()
    out['annotated_b']=md.cluster_anno_l1.eq('B-Cells');out['predicted_high']=prediction>1.4
    sample=frame.iloc[codex_rows].copy();sample['frozen_pred']=frozen
    overlap=out[['cell_id','predicted_cd19']].merge(sample[['cell_id','frozen_pred']],on='cell_id')
    assert len(overlap)==2500
    fullerr=float(np.max(abs(overlap.predicted_cd19-overlap.frozen_pred)))
    assert fullerr<2e-5,f'Full projection differs on sampled cells {fullerr}'
    out['in_frozen_sample']=out.cell_id.isin(sample.cell_id)
    out.to_csv(BASE/'data/H14_full_predictions.csv.gz',index=False)
    audit=dict(donor='H14',model_retrained=False,normalization_refitted_on_full_donor=False,
               frozen_reproduction_n=30000,frozen_reproduction_max_abs_error=err,
               full_projection_overlap_n=2500,full_projection_overlap_max_abs_error=fullerr,
               full_donor=metrics(out.annotated_b,out.predicted_cd19),
               frozen_h14=metrics(out.loc[out.in_frozen_sample,'annotated_b'],out.loc[out.in_frozen_sample,'predicted_cd19']),
               threshold=1.4,source_sha256=hashes,
               selection_plan_sha256=sha(BASE/'SELECTION_PLAN.md'),markers=markers)
    (BASE/'results/projection_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    (BASE/'data/frozen_preprocessing.json').write_text(json.dumps(dict(parameters=params,scaler_mean=scaler.mean_.tolist(),scaler_scale=scaler.scale_.tolist()),indent=2)+'\n')
    print(json.dumps(audit,indent=2),flush=True)
    assert hashes=={str(p):sha(p) for p in sources}

if __name__=='__main__': main()
