"""Measured vCAF markers and unique nearest-vCAF T-cell neighborhoods."""
import runtime as rt
from runtime import BASE, INPUTS
import json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import rankdata, spearmanr, wilcoxon

OUT=BASE/'results'
CONFIGS=['RAMP_k30','RAMP_k100']
PRIMARY=['vCAF_CD73','vCAF_IDO','nearby_CD8_PD1','nearby_CD8_TCF1']
SEED=20260909

def rho(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    ok=np.isfinite(a)&np.isfinite(b);a=a[ok];b=b[ok]
    return float(spearmanr(a,b).statistic) if len(a)>=3 and np.ptp(a)>1e-12 and np.ptp(b)>1e-12 else np.nan

def bh(p):
    p=np.asarray(p,float);out=np.full(len(p),np.nan);ok=np.isfinite(p);a=p[ok]
    order=np.argsort(a);q=np.empty(len(a));q[order]=np.minimum(1,np.minimum.accumulate((a[order]*len(a)/np.arange(1,len(a)+1))[::-1])[::-1]);out[ok]=q
    return out

def estimate(x):
    x=np.asarray(x,float);x=x[np.isfinite(x)]
    if not len(x):return {'n_patients':0,'mean':np.nan,'median':np.nan,'ci_low':np.nan,'ci_high':np.nan,'p':np.nan,'fraction_positive':np.nan}
    rng=np.random.default_rng(SEED)
    boots=np.mean(x[rng.integers(len(x),size=(3000,len(x)))],axis=1)
    lo,hi=np.quantile(boots,[.025,.975])
    return dict(n_patients=len(x),mean=float(x.mean()),median=float(np.median(x)),ci_low=float(lo),ci_high=float(hi),p=float(wilcoxon(x).pvalue) if np.any(x!=0) else 1.,fraction_positive=float((x>0).mean()))

def prepare():
    assert json.loads((OUT/'EXPORT_VERIFICATION.json').read_text())['status']=='PASS'
    req=pd.read_csv(BASE/'data/MEASUREMENT_REQUEST.csv.gz',float_precision='round_trip')
    meas=pd.read_csv(BASE/'data/MEASURED_MARKERS.csv.gz',float_precision='round_trip')
    for col in ['source_row_id','patient_id','roi_id','cell_type','cell_subtype']:
        assert np.array_equal(req[col],meas[col]),col
    names=pd.read_csv(INPUTS['shared_markers']).marker.tolist()
    raw=np.load(INPUTS['raw_shared9'],mmap_mode='r')
    expected=raw[req.matrix_row_index.to_numpy(int)]
    error=float(np.max(np.abs(meas[names].to_numpy()-expected)))
    assert error<1e-9,error
    d=req.copy()
    for col in ['CD73','IDO','PD1','TCF1','FOXP3']:d[col]=meas[col].to_numpy()
    v=d[d.cell_type.eq('vCAF')].copy()
    ids=set(v.source_row_id)
    cols=['source_row_id','patient_id','roi_id',*CONFIGS]
    pieces=[]
    for chunk in pd.read_csv(INPUTS['predictions'],usecols=cols,chunksize=300000,float_precision='round_trip'):
        pieces.append(chunk[chunk.source_row_id.isin(ids)])
    pred=pd.concat(pieces).set_index('source_row_id').loc[v.source_row_id]
    assert len(pred)==len(v)==52328
    for col in ['patient_id','roi_id']:assert np.array_equal(v[col],pred[col])
    for c in CONFIGS:v[c]=pred[c].to_numpy()
    for marker in ['CD73','IDO']:
        v[marker+'_rank']=v.groupby(['patient_id','roi_id'])[marker].transform(lambda a:rankdata(a)/(len(a)+1))
    groups={key:g for key,g in v.groupby(['patient_id','roi_id'],sort=False)}
    ts=[]
    for typ in ['CD8','CD4']:
        t=d[d.cell_category.eq('T cell')&d.cell_type.eq(typ)].copy()
        for marker in ['PD1','TCF1','FOXP3']:
            t[marker+'_rank']=t.groupby(['patient_id','roi_id'])[marker].transform(lambda a:rankdata(a)/(len(a)+1))
        t['PD1_minus_TCF1']=t.PD1_rank-t.TCF1_rank
        t['author_Treg']=t.cell_subtype.eq('CD4_Treg').astype(float)
        t['nearest_vcaf_id']=-1;t['distance_vcaf']=np.nan
        for c in CONFIGS:t[c]=np.nan
        for key,ix in t.groupby(['patient_id','roi_id'],sort=False).groups.items():
            vg=groups.get(key)
            if vg is None:continue
            dist,idx=cKDTree(vg[['x','y']].to_numpy()).query(t.loc[ix,['x','y']].to_numpy(),k=1)
            t.loc[ix,'nearest_vcaf_id']=vg.source_row_id.to_numpy()[idx]
            t.loc[ix,'distance_vcaf']=dist
            for c in CONFIGS:t.loc[ix,c]=vg[c].to_numpy()[idx]
        ts.append(t)
    t=pd.concat(ts,ignore_index=True)
    v.to_csv(OUT/'VCAF_MEASURED_AND_PREDICTED.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    t.to_csv(OUT/'T_CELL_NEAREST_VCAF.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    qc=[]
    for typ,g in d.groupby('cell_type'):
        for marker in ['CD73','IDO','PD1','TCF1','FOXP3']:
            x=g[marker].to_numpy()
            qc.append(dict(cell_type=typ,marker=marker,n=len(x),n_unique=len(np.unique(x)),fraction_zero=float((x==0).mean()),min=float(x.min()),q25=float(np.quantile(x,.25)),median=float(np.median(x)),q75=float(np.quantile(x,.75)),max=float(x.max())))
    pd.DataFrame(qc).to_csv(OUT/'MARKER_SIGNAL_QC.csv',index=False)
    qsub=d.groupby(['cell_type','cell_subtype'])[['PD1','TCF1','FOXP3','IDO']].agg(['size','median','mean'])
    qsub.to_csv(OUT/'PROVIDER_SUBTYPE_MARKER_QC.csv')
    rt.save(OUT/'JOIN_VERIFICATION.json',{'status':'PASS','n_measured':len(d),'n_vcaf':len(v),'n_cd8':int(t.cell_type.eq('CD8').sum()),'n_cd4':int(t.cell_type.eq('CD4').sum()),'raw_shared9_max_error':error,'prediction_join_exact':True,'unique_t_cell_assignment':t.source_row_id.is_unique})
    print('Measured marker and prediction joins verified; nearest-vCAF assignments saved.',flush=True)
    return v,t

def roi_analysis(v,t):
    rows=[];coverage=[];vgroups={key:g for key,g in v.groupby(['patient_id','roi_id'],sort=False)}
    for key,g in vgroups.items():
        pid,roi=key
        for c in CONFIGS:
            lo,hi=np.quantile(g[c],[1/3,2/3])
            low=g[c]<=lo;high=g[c]>=hi
            for marker in ['CD73','IDO']:
                eligible=len(g)>=15
                paired=eligible and hi>lo and low.sum()>=5 and high.sum()>=5
                a=float(g.loc[low,marker+'_rank'].mean()) if paired else np.nan
                b=float(g.loc[high,marker+'_rank'].mean()) if paired else np.nan
                rows.append(dict(patient_id=pid,roi_id=roi,config=c,radius=0,endpoint='vCAF_'+marker,n_cells=len(g),n_vcaf=len(g),n_low=int(low.sum()),n_high=int(high.sum()),rho=rho(g[c],g[marker]) if eligible else np.nan,low=a,high=b,difference=b-a))
    for (pid,roi,typ),g in t.groupby(['patient_id','roi_id','cell_type'],sort=False):
        vg=vgroups.get((pid,roi))
        for radius in [20,40,80]:
            a=g[g.distance_vcaf.le(radius)].copy()
            nu=a.nearest_vcaf_id.nunique()
            eligible=len(a)>=15 and nu>=5
            coverage.append(dict(patient_id=pid,roi_id=roi,cell_type=typ,radius=radius,n_total=len(g),n_within=len(a),n_distinct_vcaf=nu,eligible_correlation=eligible))
            if vg is None:continue
            endpoints=[('nearby_CD8_PD1','PD1_rank'),('nearby_CD8_TCF1','TCF1_rank'),('nearby_CD8_PD1_minus_TCF1','PD1_minus_TCF1')] if typ=='CD8' else [('nearby_CD4_FOXP3','FOXP3_rank'),('nearby_CD4_author_Treg','author_Treg')]
            for c in CONFIGS:
                lo,hi=np.quantile(vg[c],[1/3,2/3]);low=a[c]<=lo;high=a[c]>=hi
                paired=eligible and len(vg)>=15 and hi>lo and low.sum()>=5 and high.sum()>=5
                for endpoint,col in endpoints:
                    aval=float(a.loc[low,col].mean()) if paired else np.nan
                    bval=float(a.loc[high,col].mean()) if paired else np.nan
                    rows.append(dict(patient_id=pid,roi_id=roi,config=c,radius=radius,endpoint=endpoint,n_cells=len(a),n_vcaf=nu,n_low=int(low.sum()),n_high=int(high.sum()),rho=rho(a[c],a[col]) if eligible else np.nan,low=aval,high=bval,difference=bval-aval))
    r=pd.DataFrame(rows);r.to_csv(OUT/'ROI_ASSOCIATIONS.csv',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'NEIGHBOR_COVERAGE.csv',index=False)
    print('ROI associations complete:',len(r),flush=True)
    return r

def summarize(r):
    group=['patient_id','config','radius','endpoint']
    p=r.groupby(group,as_index=False).agg(rho=('rho','mean'),n_rois_rho=('rho','count'),low=('low','mean'),high=('high','mean'),difference=('difference','mean'),n_rois_contrast=('difference','count'))
    p.to_csv(OUT/'PATIENT_ASSOCIATIONS.csv',index=False)
    dfs=set(pd.read_csv(BASE/'data/DFS_280.csv').patient_id)
    results=[]
    for pop,q in [('all_available',p),('DFS_280_members',p[p.patient_id.isin(dfs)])]:
        for key,g in q.groupby(['config','radius','endpoint'],sort=False):
            c,radius,endpoint=key
            for metric in ['rho','difference']:
                result={'population':pop,'config':c,'radius':radius,'endpoint':endpoint,'metric':metric,**estimate(g[metric])}
                result['primary_endpoint']=endpoint in PRIMARY and radius in [0,40]
                if metric=='difference':result.update(low=float(g.low.mean()),high=float(g.high.mean()))
                result['n_rois']=int(g['n_rois_rho' if metric=='rho' else 'n_rois_contrast'].sum())
                results.append(result)
    s=pd.DataFrame(results)
    s['q_primary']=np.nan;s['q_family']=np.nan
    for _,ix in s[s.primary_endpoint].groupby(['population','config','metric']).groups.items():
        assert len(ix)==4
        s.loc[ix,'q_primary']=bh(s.loc[ix,'p'])
    for _,ix in s.groupby(['population','config','metric','primary_endpoint']).groups.items():s.loc[ix,'q_family']=bh(s.loc[ix,'p'])
    s.to_csv(OUT/'ASSOCIATION_SUMMARY.csv',index=False)
    return p,s

def patient_and_clinical(v,t):
    a=v[['patient_id','roi_id','RAMP_k30','RAMP_k100','CD73','IDO']].copy()
    for marker in ['CD73','IDO']:a[marker+'_asinh']=np.arcsinh(a[marker])
    means=a.groupby('patient_id',as_index=False).agg(n_vcaf=('RAMP_k30','size'),HLA_mean=('RAMP_k30','mean'),HLA_k100_mean=('RAMP_k100','mean'),CD73_mean_asinh=('CD73_asinh','mean'),IDO_mean_asinh=('IDO_asinh','mean'))
    means['TMA']=means.patient_id.str.split('_').str[0]
    means.to_csv(OUT/'PATIENT_MARKER_MEANS.csv',index=False)
    clinical=pd.read_csv(BASE/'data/DFS_280.csv',float_precision='round_trip')
    both=clinical.merge(means,on='patient_id',validate='one_to_one',suffixes=('','_measured'))
    assert len(both)==280 and both.event.sum()==96
    assert np.max(np.abs(both.all_mean-both.HLA_mean))<1e-12
    both.to_csv(OUT/'SURVIVAL_INPUTS.csv',index=False)
    corr=[]
    for pop,g in [('all_available',means[means.n_vcaf>=15]),('DFS_280_members',both)]:
        for marker in ['CD73','IDO']:
            stat=spearmanr(g.HLA_mean,g[marker+'_mean_asinh'])
            corr.append(dict(population=pop,marker=marker,n=len(g),rho=float(stat.statistic),p=float(stat.pvalue)))
    pd.DataFrame(corr).to_csv(OUT/'BETWEEN_PATIENT_ASSOCIATIONS.csv',index=False)

def verify_neighbors(t,v):
    tests=[]
    selected=t[t.nearest_vcaf_id.ge(0)].sample(n=30,random_state=42)
    for _,row in selected.iterrows():
        vg=v[v.patient_id.eq(row.patient_id)&v.roi_id.eq(row.roi_id)]
        dist=np.sqrt((vg.x-row.x)**2+(vg.y-row.y)**2)
        assert abs(dist.min()-row.distance_vcaf)<1e-10
        loc=vg[vg.source_row_id.eq(row.nearest_vcaf_id)].iloc[0]
        assert abs(np.hypot(loc.x-row.x,loc.y-row.y)-row.distance_vcaf)<1e-10
        for c in CONFIGS:assert loc[c]==row[c]
        tests.append(int(row.source_row_id))
    rt.save(OUT/'NEIGHBOR_VERIFICATION.json',{'status':'PASS','brute_force_cells':tests,'exact_distances_and_assigned_predictions':True})

def main():
    v,t=prepare();r=roi_analysis(v,t);p,s=summarize(r);patient_and_clinical(v,t);verify_neighbors(t,v)
    primary=s[(s.population=='all_available')&(s.config=='RAMP_k30')&s.primary_endpoint&(s.metric=='rho')]
    print(primary[['endpoint','n_patients','mean','ci_low','ci_high','p','q_primary']].to_string(index=False),flush=True)
    rt.save(OUT/'ANALYSIS_STATUS.json',{'status':'complete','time_utc':rt.now(),'primary':primary.to_dict('records')})

if __name__=='__main__':main()
