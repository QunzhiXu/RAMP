import runtime as rt
from runtime import BASE,STORY
import numpy as np
import pandas as pd
from scipy import stats
from statistics_helpers import bh,ols_hc3,design_matrix,compare
OUT=BASE/'analysis'
MECH=STORY/'mechanism_support'
LINEAGES=['CD8','CD4','Bcell','Myeloid','Neutrophil']
PRIMARY=[('CD8','PD1'),('CD8','TCF1'),('CD4','PD1'),('CD4','TCF1'),('CD4','FOXP3'),('Myeloid','IDO'),('Myeloid','CD73'),('Bcell','CD73'),('Neutrophil','IDO'),('Neutrophil','CD73')]
SECONDARY=[('CD4_nonTreg','PD1'),('CD4_nonTreg','TCF1'),('Treg','PD1'),('Treg','TCF1'),('Treg','CD73'),('Treg','IDO'),('Bcell','HLA_DR'),('Myeloid','HLA_DR'),('Neutrophil','MPO'),('Neutrophil','MMP9')]

def estimate(x):
    x=np.asarray(x,float);x=x[np.isfinite(x)];rng=np.random.default_rng(20260909)
    boots=x[rng.integers(len(x),size=(3000,len(x)))].mean(1);lo,hi=np.quantile(boots,[.025,.975])
    return dict(n_patients=len(x),mean=float(x.mean()),median=float(np.median(x)),ci_low=float(lo),ci_high=float(hi),p=float(stats.wilcoxon(x).pvalue) if np.any(x!=0) else 1.)

def main():
    rt.status('molecular and immune interpretation recheck')
    paths=[MECH/'results/VCAF_MEASURED_AND_PREDICTED.csv.gz',MECH/'patient_immune/results/PATIENT_IMMUNE_COMPOSITION.csv',MECH/'patient_immune/results/PATIENT_LINEAGE_MARKERS.csv',MECH/'results/PATIENT_ASSOCIATIONS.csv',MECH/'patient_immune/results/COMPOSITION_ADJUSTED.csv',MECH/'patient_immune/results/ADJUSTED_COMPARISONS.csv']
    h=rt.hashes(paths)
    clinical=pd.read_csv(OUT/'SURVIVAL_INPUTS.csv',dtype={'patient_id':str},float_precision='round_trip');clinical=clinical[clinical.config.eq('RAMP_k30')].copy()
    v=pd.read_csv(paths[0],float_precision='round_trip',dtype={'patient_id':str})
    meta=pd.read_csv(BASE/'cords/CELL_METADATA.csv.gz',usecols=['source_row_id','patient_id','cell_type'],dtype={'patient_id':str})
    pos=pd.Index(meta.source_row_id).get_indexer(v.source_row_id);assert (pos>=0).all() and len(v)==52328
    assert np.array_equal(meta.iloc[pos].patient_id,v.patient_id) and meta.iloc[pos].cell_type.eq('vCAF').all()
    new=np.load(BASE/'cords/PREDICTIONS.npy',mmap_mode='r')[pos,:2]
    # Recompute both versions using the same ROI rules, and reproduce the old summary.
    rows=[]
    for version in ['old','new']:
        for conf,j in [('RAMP_k30',0),('RAMP_k100',1)]:
            pred=v[conf].to_numpy() if version=='old' else new[:,j]
            for (pid,roi),ix in v.groupby(['patient_id','roi_id']).indices.items():
                g=v.iloc[ix];s=pred[ix];lo,hi=np.quantile(s,[1/3,2/3]);lower=s<=lo;higher=s>=hi
                eligible=len(g)>=15;paired=eligible and hi>lo and lower.sum()>=5 and higher.sum()>=5
                for marker in ['CD73','IDO']:
                    ranks=stats.rankdata(g[marker])/(len(g)+1)
                    a=float(ranks[lower].mean()) if paired else np.nan;b=float(ranks[higher].mean()) if paired else np.nan
                    rho=float(stats.spearmanr(s,g[marker]).statistic) if eligible and np.ptp(s)>1e-12 and np.ptp(g[marker])>1e-12 else np.nan
                    rows.append(dict(version=version,config=conf,patient_id=pid,roi_id=roi,endpoint='vCAF_'+marker,rho=rho,low=a,high=b,difference=b-a))
    roi=pd.DataFrame(rows);roi.to_csv(OUT/'VCAF_ROI_MARKER_ASSOCIATIONS.csv',index=False)
    patient=roi.groupby(['version','config','patient_id','endpoint'],as_index=False)[['rho','low','high','difference']].mean()
    patient.to_csv(OUT/'VCAF_PATIENT_MARKER_ASSOCIATIONS.csv',index=False)
    prior=pd.read_csv(paths[3],dtype={'patient_id':str});prior=prior[(prior.radius==0)&prior.endpoint.isin(['vCAF_CD73','vCAF_IDO'])]
    check=patient[patient.version.eq('old')].merge(prior,on=['config','patient_id','endpoint'],suffixes=('','_prior'),validate='one_to_one')
    assert np.nanmax(np.abs(check.rho-check.rho_prior))<1e-10
    assert np.nanmax(np.abs(check.difference-check.difference_prior))<1e-10
    summary=[]
    for key,g in patient.groupby(['version','config','endpoint']):
        for metric in ['rho','difference']:
            d=dict(version=key[0],config=key[1],endpoint=key[2],metric=metric,**estimate(g[metric]));d['low']=g.low.mean();d['high']=g.high.mean();summary.append(d)
    s=pd.DataFrame(summary);s['q_two_vcaf_markers']=np.nan
    for _,ix in s.groupby(['version','config','metric']).groups.items():s.loc[ix,'q_two_vcaf_markers']=bh(s.loc[ix,'p'])
    s.to_csv(OUT/'VCAF_MARKER_SUMMARY.csv',index=False)
    comp=pd.read_csv(paths[1],dtype={'patient_id':str});expr=pd.read_csv(paths[2],dtype={'patient_id':str})
    endpoints=[l+'_'+s for l in LINEAGES for s in ['of_immune','of_all']]+['immune_of_all','Treg_of_CD4']
    cc=comp[['patient_id',*endpoints]];ee=expr[['patient_id','lineage','marker','mean','median','roi_equal','n_cells','n_rois']]
    composition=[];expression=[];groups=[]
    for version,g in clinical.groupby('version',sort=False):
        g=g.copy();cut=g.all_mean.median();g['high_HLA']=(g.all_mean>=cut).astype(int);g['group']=np.where(g.high_HLA,'Higher','Lower');g['TMA']=g.patient_id.str.split('_').str[0]
        groups.append(g)
        merged=cc.merge(g,on='patient_id',validate='one_to_one')
        for endpoint in endpoints:
            d=merged.dropna(subset=[endpoint]).copy();val=d[endpoint];y=(val-val.mean())/val.std(ddof=1)
            x=design_matrix(d);assert np.linalg.matrix_rank(x)==x.shape[1]
            composition.append({'version':version,'endpoint':endpoint,**compare(d,endpoint),**ols_hc3(y,x)})
        merged=ee.merge(g,on='patient_id',validate='many_to_one')
        for family,pairs in [('primary',PRIMARY),('secondary',SECONDARY)]:
            for lineage,marker in pairs:
                for case,minimum,col in [('all_available_mean',1,'mean'),('min20_mean',20,'mean'),('min50_mean',50,'mean'),('roi_equal',1,'roi_equal')]:
                    d=merged[merged.lineage.eq(lineage)&merged.marker.eq(marker)&merged.n_cells.ge(minimum)].copy()
                    if len(d)<30 or d.group.value_counts().min()<10:continue
                    val=d[col];y=(val-val.mean())/val.std(ddof=1);x=design_matrix(d);assert np.linalg.matrix_rank(x)==x.shape[1]
                    expression.append({'version':version,'family':family,'lineage':lineage,'marker':marker,'analysis':case,**compare(d,col),**ols_hc3(y,x)})
    pd.concat(groups,ignore_index=True).to_csv(OUT/'PATIENT_GROUPS.csv',index=False)
    c=pd.DataFrame(composition);e=pd.DataFrame(expression)
    for table,keys,file in [(c,['version'],'IMMUNE_COMPOSITION.csv'),(e,['version','family','analysis'],'IMMUNE_EXPRESSION.csv')]:
        table['q_bh']=np.nan
        for _,ix in table.groupby(keys).groups.items():table.loc[ix,'q_bh']=bh(table.loc[ix,'p'])
        table.to_csv(OUT/file,index=False)
    oldcomp=pd.read_csv(paths[4]);oldcomp=oldcomp[~oldcomp.full_CAF_covariates]
    check=c[c.version.eq('old')].merge(oldcomp,on='endpoint',suffixes=('','_prior'))
    err_comp=max(float(np.abs(check[k]-check[k+'_prior']).max()) for k in ['coef','se','p','q_bh']);assert err_comp<1e-10
    oldexpr=pd.read_csv(paths[5]);oldexpr=oldexpr[~oldexpr.full_CAF_covariates]
    check=e[e.version.eq('old')].merge(oldexpr,on=['family','lineage','marker','analysis'],suffixes=('','_prior'))
    err_expr=max(float(np.abs(check[k]-check[k+'_prior']).max()) for k in ['coef','se','p','q_bh']);assert err_expr<1e-10
    rt.unchanged(h);rt.save(OUT/'MECHANISM_CHECKS.json',dict(source_files_unchanged=True,old_vcaf_associations_reproduced=True,old_immune_composition_max_error=err_comp,old_immune_expression_max_error=err_expr,input_hashes=h,vcaf_multiple_testing_family='two vCAF markers only, explicitly different from earlier four-endpoint primary family'))
    print(c[c.version.eq('new')][['endpoint','lower_mean','higher_mean','coef','p','q_bh']].to_string(index=False),flush=True)
if __name__=='__main__':main()
