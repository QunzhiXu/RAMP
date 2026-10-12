"""Exhaustive finite-grid spatial-window and shared color-bound search."""
from pathlib import Path
import os,json,hashlib,time
BASE=Path(__file__).resolve().parents[1];SOURCE=BASE.parent
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import numpy as np,pandas as pd,h5py
from scipy.stats import spearmanr

def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    start=time.time()
    plan_hash=sha(BASE/'SEARCH_PLAN.md')
    geo_path=SOURCE/'data/C30_synthetic_cell_lookup.csv.gz'
    val_path=SOURCE/'data/C30_HLA_values_unchanged.csv.gz'
    g=pd.read_csv(geo_path);v=pd.read_csv(val_path)
    assert g.CellID.equals(v.CellID)
    xy=g[['render_x_px','render_y_px']].to_numpy()*.5
    x=v.observed_HLA_A_patient_rank.to_numpy();y=v.RAMP_k30.to_numpy()
    syn=json.loads((SOURCE/'results/synthesis_audit.json').read_text())
    height,width=np.array(syn['image_shape_yx'])*.5
    stride=16;nx=int(np.ceil(width/stride));ny=int(np.ceil(height/stride))
    binx=np.floor(xy[:,0]/stride).astype(int);biny=np.floor(xy[:,1]/stride).astype(int)
    binid=biny*nx+binx
    def integral(weights):
        hist=np.bincount(binid,weights=weights,minlength=nx*ny).reshape(ny,nx)
        return np.pad(hist,((1,0),(1,0))).cumsum(0).cumsum(1)
    count_integral=integral(np.ones(len(x)))
    entries=[];n_scanned=0
    for side in [192,256,320]:
        k=side//stride
        for by in range(int(np.floor((height-side)/stride))+1):
            for bx in range(int(np.floor((width-side)/stride))+1):
                n_scanned+=1
                n=int(round(count_integral[by+k,bx+k]-count_integral[by,bx+k]-count_integral[by+k,bx]+count_integral[by,bx]))
                if n<300:continue
                entries.append(dict(roi_id=f'S{side}_X{bx*stride}_Y{by*stride}',side_um=side,x0_um=bx*stride,y0_um=by*stride,
                                    x1_um=bx*stride+side,y1_um=by*stride+side,bx=bx,by=by,k=k,n_cells=n,kind='grid'))
    old=syn['detail_crop']
    oldbox=np.array([old[k] for k in ['x0','y0','x1','y1']])*.5
    original_mask=(xy[:,0]>=oldbox[0])&(xy[:,0]<oldbox[2])&(xy[:,1]>=oldbox[1])&(xy[:,1]<oldbox[3])
    assert int(original_mask.sum())==old['n_centroids']
    rois=pd.DataFrame(entries)
    bx=rois.bx.to_numpy(int);by=rois.by.to_numpy(int);kk=rois.k.to_numpy(int)
    def query_all(weights):
        z=integral(weights)
        out=z[by+kk,bx+kk]-z[by,bx+kk]-z[by+kk,bx]+z[by,bx]
        return np.r_[out,np.sum(weights[original_mask])]
    entries.append(dict(roi_id='PRIOR_CENTRAL',side_um=256,x0_um=oldbox[0],y0_um=oldbox[1],x1_um=oldbox[2],y1_um=oldbox[3],
                        bx=-1,by=-1,k=16,n_cells=int(original_mask.sum()),kind='prior_central'))
    rois=pd.DataFrame(entries);n=rois.n_cells.to_numpy(float)
    rhos=[]
    for row in rois.itertuples(index=False):
        local=(xy[:,0]>=row.x0_um)&(xy[:,0]<row.x1_um)&(xy[:,1]>=row.y0_um)&(xy[:,1]<row.y1_um)
        assert int(local.sum())==row.n_cells
        rhos.append(float(spearmanr(x[local],y[local]).statistic))
    rois['raw_spearman']=rhos
    assert np.isfinite(rhos).all()
    rois.to_csv(BASE/'data/roi_candidates.csv',index=False)
    pairs=[(l/100,u/100) for l in range(0,81,5) for u in range(l+20,101,5)]
    assert len(pairs)==153
    assert all(round(v*100)%5==0 for pair in pairs for v in pair)
    bounds=pd.DataFrame(pairs,columns=['lower','upper']);bounds.insert(0,'threshold_id',range(len(bounds)))
    bounds.to_csv(BASE/'data/threshold_grid.csv',index=False)
    N=len(rois);T=len(pairs)
    best=np.full(N,-np.inf);best_idx=np.full(N,-1,int);best_width=np.zeros(N);admissible_total=0
    metric_names=['display_ccc','display_mae','min_panel_sd','min_interior_fraction','max_endpoint_fraction','presentation_score']
    f=h5py.File(BASE/'results/all_roi_threshold_results.h5','w')
    f.attrs['interpretation']='Presentation-only exhaustive search; no pixel similarity, no DAPI contribution to score'
    f.attrs['search_plan_sha256']=plan_hash
    datasets={name:f.create_dataset(name,(T,N),dtype='f4',chunks=(1,N),compression='gzip',compression_opts=1,shuffle=True) for name in metric_names}
    valid_data=f.create_dataset('admissible',(T,N),dtype='?',chunks=(1,N),compression='gzip',compression_opts=1)
    f.create_dataset('roi_id',data=rois.roi_id.to_numpy(),dtype=h5py.string_dtype())
    f.create_dataset('bounds',data=np.array(pairs));f.create_dataset('raw_spearman',data=np.array(rhos))
    threshold_best=[];original_setting=None
    for ti,(lo,hi) in enumerate(pairs):
        a=np.clip((x-lo)/(hi-lo),0,1);b=np.clip((y-lo)/(hi-lo),0,1)
        ma=query_all(a)/n;mb=query_all(b)/n
        va=np.maximum(query_all(a*a)/n-ma*ma,0);vb=np.maximum(query_all(b*b)/n-mb*mb,0)
        cov=query_all(a*b)/n-ma*mb
        denom=va+vb+(ma-mb)**2
        ccc=np.divide(2*cov,denom,out=np.full(N,np.nan),where=denom>1e-12)
        mae=query_all(abs(a-b))/n
        lowa=query_all((x<=lo).astype(float))/n;higha=query_all((x>=hi).astype(float))/n
        lowb=query_all((y<=lo).astype(float))/n;highb=query_all((y>=hi).astype(float))/n
        min_sd=np.sqrt(np.minimum(va,vb));interior=np.minimum(1-lowa-higha,1-lowb-highb)
        endpoint=np.maximum.reduce([lowa,higha,lowb,highb])
        valid=(min_sd>=.22-1e-12)&(interior>=.30-1e-12)&(endpoint<=.35+1e-12)&np.isfinite(ccc)
        score=(np.array(rhos)+ccc)/2
        for key,array in zip(metric_names,[ccc,mae,min_sd,interior,endpoint,score]):datasets[key][ti]=array.astype('f4')
        valid_data[ti]=valid
        admissible_total+=int(valid.sum())
        improve=valid&((ccc>best+1e-12)|((abs(ccc-best)<=1e-12)&(hi-lo>best_width+1e-12)))
        best[improve]=ccc[improve];best_idx[improve]=ti;best_width[improve]=hi-lo
        if valid.any():
            winner=int(np.argmax(np.where(valid,score,-np.inf)))
            threshold_best.append(dict(threshold_id=ti,lower=lo,upper=hi,roi_id=rois.roi_id.iloc[winner],score=float(score[winner]),n_admissible_rois=int(valid.sum())))
        if lo==.25 and hi==.60:
            original_setting=dict(roi_id='PRIOR_CENTRAL',n_cells=int(n[-1]),lower=lo,upper=hi,raw_spearman=rhos[-1],
                display_ccc=float(ccc[-1]),display_mae=float(mae[-1]),min_panel_sd=float(min_sd[-1]),
                min_interior_fraction=float(interior[-1]),max_endpoint_fraction=float(endpoint[-1]),
                admissible=bool(valid[-1]),presentation_score=float(score[-1]))
        if (ti+1)%300==0:print(f'Thresholds {ti+1}/{T}; {N} ROIs; elapsed {time.time()-start:.1f}s',flush=True)
    result=rois.copy();result['best_threshold_id']=best_idx;result['display_ccc']=np.where(best_idx>=0,best,np.nan)
    result['lower']=np.nan;result['upper']=np.nan
    for name in metric_names[1:]:result[name]=np.nan
    for i,ti in enumerate(best_idx):
        if ti<0:continue
        result.loc[i,['lower','upper']]=pairs[ti]
        for name in metric_names[1:]:result.loc[i,name]=float(datasets[name][ti,i])
    f.close()
    result['presentation_score']=(result['raw_spearman']+result['display_ccc'])/2
    result=result.sort_values(['presentation_score','n_cells','y0_um','x0_um'],ascending=[False,False,True,True],na_position='last').reset_index(drop=True)
    result.insert(0,'presentation_rank',np.arange(1,len(result)+1))
    result.to_csv(BASE/'results/best_bounds_per_ROI.csv',index=False)
    pd.DataFrame(threshold_best).to_csv(BASE/'results/best_ROI_per_bounds.csv',index=False)
    def iou(a,b):
        inter=max(0,min(a.x1_um,b.x1_um)-max(a.x0_um,b.x0_um))*max(0,min(a.y1_um,b.y1_um)-max(a.y0_um,b.y0_um))
        return inter/(a.side_um**2+b.side_um**2-inter)
    distinct=[]
    for row in result.itertuples(index=False):
        if row.best_threshold_id<0:continue
        if all(iou(row,r)<.25 for r in distinct):distinct.append(row)
        if len(distinct)==6:break
    pd.DataFrame(distinct).to_csv(BASE/'results/top_distinct_ROIs.csv',index=False)
    best_by_size=result[result.best_threshold_id>=0].groupby('side_um',sort=True).head(1)
    best_by_size.to_csv(BASE/'results/best_by_window_size.csv',index=False)
    selected=result.iloc[0].to_dict();assert selected['best_threshold_id']>=0
    selected['origin_um']=syn['origin_um'];selected['pixel_size_um']=syn['pixel_size_um']
    report=dict(scope='Zheng C30 only; finite exhaustive grid',n_cells_full=10719,whole_C30_raw_spearman=float(spearmanr(x,y).statistic),
        side_um=[192,256,320],stride_um=16,n_spatial_grid_windows_scanned=n_scanned,
        n_grid_windows_with_at_least_300_cells=len(rois)-1,n_extra_prior_central_windows=1,n_ROIs_evaluated=N,
        color_bound_step=0.05,n_threshold_pairs=T,n_ROI_threshold_pairs_evaluated=int(N*T),n_admissible_pairs=admissible_total,
        n_ROIs_with_admissible_bounds=int((best_idx>=0).sum()),original_central_original_bounds=original_setting,
        original_central_optimized_bounds=result[result.roi_id=='PRIOR_CENTRAL'].iloc[0].to_dict(),best=selected,
        elapsed_seconds=time.time()-start,source_sha256={str(p):sha(p) for p in [geo_path,val_path,SOURCE/'data/C30_synthetic_DAPI.ome.tif',SOURCE/'data/C30_synthetic_cell_labels.tif']},
        search_plan_sha256=plan_hash,expression_or_prediction_modified=False,synthetic_DAPI_modified=False,
        display_thresholds_are_not_biological_classification_thresholds=True)
    (BASE/'results/search_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2),flush=True)

if __name__=='__main__':main()
