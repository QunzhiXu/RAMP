from runtime import *

def estimate(values):
    x=np.asarray(values,float);x=x[np.isfinite(x)]
    if not len(x): return dict(n_patients=0,mean=np.nan,median=np.nan,ci_low=np.nan,ci_high=np.nan,p=np.nan,fraction_positive=np.nan)
    boot=np.mean(x[np.random.default_rng(SEED).integers(len(x),size=(3000,len(x)))],axis=1)
    lo,hi=np.quantile(boot,[.025,.975])
    return dict(n_patients=len(x),mean=float(x.mean()),median=float(np.median(x)),ci_low=float(lo),ci_high=float(hi),p=float(stats.wilcoxon(x).pvalue) if np.any(x!=0) else 1.,fraction_positive=float(np.mean(x>0)))

def load_cells():
    assert json.loads((OUT/'EXPORT_VERIFICATION.json').read_text())['status']=='PASS'
    req=pd.read_csv(DATA/'MEASUREMENT_REQUEST.csv.gz',float_precision='round_trip')
    m=pd.read_csv(DATA/'MEASURED_MARKERS.csv.gz',float_precision='round_trip')
    assert req.source_row_id.is_unique and len(req)==len(m)==2849385
    for col in ['source_row_id','patient_id','roi_id','cell_type']:assert np.array_equal(req[col],m[col])
    raw=np.load(INPUTS['raw_shared9'],mmap_mode='r');names=pd.read_csv(INPUTS['shared_markers']).marker.tolist()
    err=0.
    for marker in ['CD3','HLA_DR']:
        err=max(err,float(np.abs(m[marker].to_numpy()-raw[req.matrix_row_index.to_numpy(int),names.index(marker)]).max()))
    assert err<1e-9
    d=req[['source_row_id','patient_id','roi_id','cell_category','cell_type']].copy()
    for marker in MARKERS:d[marker]=m[marker].to_numpy()
    del req,m,raw
    pred=pd.read_csv(INPUTS['predictions'],usecols=['source_row_id','patient_id','roi_id',*CONFIGS],float_precision='round_trip').set_index('source_row_id')
    assert pred.index.is_unique
    pred=pred.loc[d.source_row_id]
    for col in ['patient_id','roi_id']:assert np.array_equal(d[col],pred[col])
    for config in CONFIGS:d[config]=pred[config].to_numpy()
    del pred
    assert np.isfinite(d[MARKERS+CONFIGS]).all().all()
    previous_errors={}
    for key in ['old_vcaf_measurements','old_immune_measurements']:
        old=pd.read_csv(INPUTS[key],usecols=['source_row_id',*MARKERS],float_precision='round_trip').set_index('source_row_id')
        current=d.set_index('source_row_id').loc[old.index,MARKERS]
        previous_errors[key]=float(np.abs(current.to_numpy()-old[MARKERS].to_numpy()).max())
        assert previous_errors[key]<1e-12
    save(OUT/'JOIN_VERIFICATION.json',{'status':'PASS','n_cells':len(d),'predictions_joined_by_unique_source_ID':True,'all_patient_ROI_labels_exact':True,'raw_shared_marker_max_error':err,'previous_measured_max_errors':previous_errors})
    d.to_csv(DATA/'CELL_VALUES.csv.gz',index=False,compression={'method':'gzip','mtime':0,'compresslevel':5})
    return d

def roi_analysis(d):
    rows=[]
    groups=d.groupby(['cell_category','cell_type','patient_id','roi_id'],sort=False)
    for (category,typ,pid,roi),g in groups:
        n=len(g)
        if n<15:continue
        ranks={m:stats.rankdata(g[m])/(n+1) for m in MARKERS}
        for config in CONFIGS:
            pred=g[config].to_numpy();lo,hi=np.quantile(pred,[1/3,2/3]);low=pred<=lo;high=pred>=hi
            paired=hi>lo and low.sum()>=5 and high.sum()>=5
            for marker in MARKERS:
                value=g[marker].to_numpy()
                rho=float(stats.spearmanr(pred,value).statistic) if np.ptp(pred)>1e-12 and np.ptp(value)>1e-12 else np.nan
                a=float(ranks[marker][low].mean()) if paired else np.nan
                b=float(ranks[marker][high].mean()) if paired else np.nan
                rows.append(dict(cell_category=category,cell_type=typ,patient_id=pid,roi_id=roi,config=config,marker=marker,n_cells=n,n_low=int(low.sum()),n_high=int(high.sum()),rho=rho,low=a,high=b,difference=b-a))
    r=pd.DataFrame(rows);r.to_csv(OUT/'ROI_ASSOCIATIONS.csv',index=False)
    p=r.groupby(['cell_category','cell_type','patient_id','config','marker'],as_index=False).agg(rho=('rho','mean'),n_rois_rho=('rho','count'),low=('low','mean'),high=('high','mean'),difference=('difference','mean'),n_rois_contrast=('difference','count'))
    p.to_csv(OUT/'PATIENT_ASSOCIATIONS.csv',index=False)
    return r,p

def summaries(p):
    ids=set(pd.read_csv(INPUTS['immune'],usecols=['patient_id']).patient_id)
    rows=[]
    for pop,d in [('all_available',p),('DFS_280_members',p[p.patient_id.isin(ids)])]:
        for (cat,typ,config,marker),g in d.groupby(['cell_category','cell_type','config','marker']):
            for metric in ['rho','difference']:
                a=dict(population=pop,cell_category=cat,cell_type=typ,config=config,marker=marker,metric=metric,**estimate(g[metric]))
                if metric=='difference':a.update(low=float(g.low.mean()),high=float(g.high.mean()))
                rows.append(a)
    s=pd.DataFrame(rows);s['q_bh']=np.nan
    for _,ix in s.groupby(['population','config','metric']).groups.items():s.loc[ix,'q_bh']=bh(s.loc[ix,'p'])
    s.to_csv(OUT/'MOLECULAR_SUMMARY.csv',index=False)
    return s

def matched_specificity(r,p):
    ids=set(pd.read_csv(INPUTS['immune'],usecols=['patient_id']).patient_id)
    out=[]
    for pop,patients in [('all_available',None),('DFS_280_members',ids)]:
        for config in CONFIGS:
            for marker in MARKERS:
                q=p[p.config.eq(config)&p.marker.eq(marker)]
                z=r[r.config.eq(config)&r.marker.eq(marker)]
                if patients is not None:q=q[q.patient_id.isin(patients)];z=z[z.patient_id.isin(patients)]
                ref=q[q.cell_type.eq('vCAF')][['patient_id','rho']].rename(columns={'rho':'vcaf_rho'})
                refroi=z[z.cell_type.eq('vCAF')][['patient_id','roi_id','rho']].rename(columns={'rho':'vcaf_rho'})
                for typ,g in q[q.cell_type.ne('vCAF')].groupby('cell_type'):
                    both=g[['patient_id','rho']].merge(ref,on='patient_id',validate='one_to_one').dropna()
                    out.append(dict(population=pop,config=config,marker=marker,cell_type=typ,matching='same_patient',candidate_mean=float(both.rho.mean()),vcaf_mean=float(both.vcaf_rho.mean()),**estimate(both.rho-both.vcaf_rho)))
                    a=z[z.cell_type.eq(typ)][['patient_id','roi_id','rho']].merge(refroi,on=['patient_id','roi_id'],validate='one_to_one').dropna()
                    a=a.groupby('patient_id')[['rho','vcaf_rho']].mean()
                    out.append(dict(population=pop,config=config,marker=marker,cell_type=typ,matching='same_patient_ROI',candidate_mean=float(a.rho.mean()),vcaf_mean=float(a.vcaf_rho.mean()),**estimate(a.rho-a.vcaf_rho)))
    out=pd.DataFrame(out);out['q_bh']=np.nan
    for _,ix in out.groupby(['population','config','matching']).groups.items():out.loc[ix,'q_bh']=bh(out.loc[ix,'p'])
    out.to_csv(OUT/'MATCHED_VCAF_COMPARISONS.csv',index=False)

def patient_scores(d):
    thresholds=pd.read_csv(INPUTS['thresholds']).set_index('config').threshold
    frames=[]
    for config in CONFIGS:
        a=d[['patient_id','cell_category','cell_type',config,*MARKERS]].copy()
        a['high_state']=a[config]>=thresholds[config]
        for marker in MARKERS:a[marker+'_asinh']=np.arcsinh(a[marker])
        g=a.groupby(['patient_id','cell_category','cell_type'],as_index=False).agg(n_cells=(config,'size'),HLA_mean=(config,'mean'),HLA_median=(config,'median'),high_fraction=('high_state','mean'),CD73_mean=('CD73_asinh','mean'),IDO_mean=('IDO_asinh','mean'))
        g['config']=config;frames.append(g)
    g=pd.concat(frames,ignore_index=True)
    total=d.groupby('patient_id').size().rename('total_cells');g=g.merge(total,on='patient_id',validate='many_to_one');g['cell_fraction']=g.n_cells/g.total_cells
    g.to_csv(OUT/'PATIENT_CELLTYPE_SCORES.csv',index=False)
    g[g.n_cells>=15].groupby(['cell_category','cell_type','config']).agg(n_patients=('patient_id','nunique'),mean_patient_HLA=('HLA_mean','mean'),median_patient_HLA=('HLA_mean','median'),mean_high_fraction=('high_fraction','mean'),total_cells=('n_cells','sum')).to_csv(OUT/'CELLTYPE_SCORE_OVERVIEW.csv')

def verify(r,p,d):
    old=pd.read_csv(INPUTS['old_vcaf_associations'],float_precision='round_trip')
    old=old[old.radius.eq(0)&old.endpoint.isin(['vCAF_CD73','vCAF_IDO'])].copy();old['marker']=old.endpoint.str.replace('vCAF_','',regex=False)
    new=p[p.cell_type.eq('vCAF')]
    a=new.merge(old,on=['patient_id','config','marker'],validate='one_to_one',suffixes=('_new','_old'))
    errors={col:float(np.nanmax(np.abs(a[col+'_new']-a[col+'_old']))) for col in ['rho','low','high','difference']}
    assert len(a)==len(new) and all(x<1e-10 for x in errors.values())
    checks=[]
    for _,row in r.sample(n=12,random_state=SEED).iterrows():
        g=d[d.patient_id.eq(row.patient_id)&d.roi_id.eq(row.roi_id)&d.cell_type.eq(row.cell_type)]
        x=stats.rankdata(g[row.config]);y=stats.rankdata(g[row.marker]);value=float(np.corrcoef(x,y)[0,1])
        assert abs(value-row.rho)<1e-12
        checks.append(dict(patient_id=row.patient_id,roi_id=row.roi_id,cell_type=row.cell_type,marker=row.marker))
    save(OUT/'MOLECULAR_VERIFICATION.json',{'status':'PASS','previous_vcaf_patients_reproduced':len(a),'previous_vcaf_max_errors':errors,'independent_rank_Pearson_checks':checks})

if __name__=='__main__':
    d=load_cells();print('All raw-marker and prediction joins verified.',flush=True)
    r,p=roi_analysis(d);s=summaries(p);matched_specificity(r,p);patient_scores(d);verify(r,p,d)
    print(s[s.population.eq('all_available')&s.config.eq('RAMP_k30')&s.metric.eq('rho')][['cell_type','marker','n_patients','mean','ci_low','ci_high','q_bh']].to_string(index=False),flush=True)
    print('Molecular comparisons complete.',flush=True)
