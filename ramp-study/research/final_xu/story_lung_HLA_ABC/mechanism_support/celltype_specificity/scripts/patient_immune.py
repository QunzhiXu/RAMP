from runtime import *

def matrix(g,exposure='candidate_HLA',extra=None):
    x=pd.DataFrame({'intercept':np.ones(len(g))},index=g.index)
    x['exposure']=(g[exposure]-g[exposure].mean())/g[exposure].std(ddof=1)
    x['age']=(g.age-g.age.mean())/g.age.std(ddof=1)
    x=pd.concat([x,pd.get_dummies(g[['TMA','sex','Stage']].astype(str),drop_first=True,dtype=float)],axis=1)
    if extra:
        for col in extra:x[col]=(g[col]-g[col].mean())/g[col].std(ddof=1)
    return x

def run():
    immune=pd.read_csv(INPUTS['immune'],float_precision='round_trip').rename(columns={'config':'survival_config'})
    assert len(immune)==280 and immune.patient_id.is_unique
    immune['vcaf_HLA']=immune.all_mean;immune['CD8_TCF1']=immune.TCF1_mean
    t=pd.read_csv(INPUTS['immune_markers'],usecols=['patient_id','lineage','marker','mean','n_cells'],float_precision='round_trip')
    cd4=t[t.lineage.eq('CD4')&t.marker.eq('TCF1')][['patient_id','mean']].rename(columns={'mean':'CD4_TCF1'})
    immune=immune.merge(cd4,on='patient_id',how='left',validate='one_to_one')
    scores=pd.read_csv(OUT/'PATIENT_CELLTYPE_SCORES.csv',float_precision='round_trip')
    scores=scores[['patient_id','cell_category','cell_type','config','n_cells','HLA_mean','cell_fraction','CD73_mean','IDO_mean']].rename(columns={'HLA_mean':'candidate_HLA','CD73_mean':'candidate_CD73','IDO_mean':'candidate_IDO'})
    both=scores.merge(immune,on='patient_id',validate='many_to_one')
    both.to_csv(OUT/'PATIENT_IMMUNE_INPUTS.csv',index=False)
    rows=[];matched=[];skipped=[]
    for (cat,typ,config),d in both.groupby(['cell_category','cell_type','config']):
        for endpoint in IMMUNE_ENDPOINTS:
            for case,mincells,extra in [('main',15,None),('min50',50,None),('type_fraction',15,['cell_fraction']),('with_vCAF',15,['vcaf_HLA'])]:
                if case=='with_vCAF' and typ=='vCAF':continue
                g=d[d.n_cells.ge(mincells)].dropna(subset=[endpoint,'candidate_HLA']).copy()
                if len(g)<40:
                    skipped.append(dict(cell_type=typ,config=config,endpoint=endpoint,analysis=case,n=len(g),reason='fewer than 40 eligible patients'));continue
                x=matrix(g,extra=extra);y=(g[endpoint]-g[endpoint].mean())/g[endpoint].std(ddof=1)
                if np.linalg.matrix_rank(x)!=x.shape[1]:
                    skipped.append(dict(cell_type=typ,config=config,endpoint=endpoint,analysis=case,n=len(g),reason='rank-deficient covariate matrix'));continue
                row=dict(cell_category=cat,cell_type=typ,config=config,endpoint=endpoint,analysis=case,**ols_hc3(y,x))
                row['candidate_vcaf_rho']=float(stats.spearmanr(g.candidate_HLA,g.vcaf_HLA).statistic)
                rows.append(row)
                if case=='main':
                    matched.append(dict(cell_category=cat,cell_type=typ,config=config,endpoint=endpoint,**ols_hc3(y,matrix(g,exposure='vcaf_HLA'))))
    result=pd.DataFrame(rows);result['q_bh']=np.nan
    for _,ix in result.groupby(['config','analysis']).groups.items():result.loc[ix,'q_bh']=bh(result.loc[ix,'p'])
    result.to_csv(OUT/'PATIENT_IMMUNE_ASSOCIATIONS.csv',index=False)
    paired=pd.DataFrame(matched);paired['q_bh']=np.nan
    for _,ix in paired.groupby(['config']).groups.items():paired.loc[ix,'q_bh']=bh(paired.loc[ix,'p'])
    paired.to_csv(OUT/'VCAF_ON_MATCHED_PATIENTS.csv',index=False)
    pd.DataFrame(skipped).to_csv(OUT/'SKIPPED_PATIENT_MODELS.csv',index=False)
    corr=[]
    for (cat,typ,config),g in scores[scores.n_cells>=15].groupby(['cell_category','cell_type','config']):
        for marker in MARKERS:
            r,p=stats.spearmanr(g.candidate_HLA,g['candidate_'+marker]);corr.append(dict(cell_category=cat,cell_type=typ,config=config,marker=marker,n=len(g),rho=float(r),p=float(p)))
    corr=pd.DataFrame(corr);corr['q_bh']=np.nan
    for _,ix in corr.groupby('config').groups.items():corr.loc[ix,'q_bh']=bh(corr.loc[ix,'p'])
    corr.to_csv(OUT/'BETWEEN_PATIENT_MARKER_ASSOCIATIONS.csv',index=False)
    # Check a fitted effect against the normal-equation HC3 implementation used earlier.
    tests=[]
    for typ in ['vCAF','SMA_CAF','CD8','normal']:
        g=both[both.cell_type.eq(typ)&both.config.eq('RAMP_k30')&both.n_cells.ge(15)].dropna(subset=['immune_of_all'])
        x=matrix(g).to_numpy();y=((g.immune_of_all-g.immune_of_all.mean())/g.immune_of_all.std(ddof=1)).to_numpy()
        inv=np.linalg.pinv(x.T@x);b=inv@x.T@y;res=y-x@b;h=np.einsum('ij,jk,ik->i',x,inv,x);cov=inv@(x.T@((res**2/(1-h)**2)[:,None]*x))@inv
        row=result[result.cell_type.eq(typ)&result.config.eq('RAMP_k30')&result.endpoint.eq('immune_of_all')&result.analysis.eq('main')].iloc[0]
        assert abs(row.coef-b[1])<1e-10 and abs(row.se-np.sqrt(cov[1,1]))<1e-10
        tests.append(typ)
    vcaf=both[both.cell_type.eq('vCAF')&both.config.eq('RAMP_k30')]
    assert len(vcaf)==280 and np.max(np.abs(vcaf.candidate_HLA-vcaf.vcaf_HLA))<1e-12
    save(OUT/'PATIENT_VERIFICATION.json',{'status':'PASS','original_280_patient_membership':True,'vcaf_mean_exactly_reproduced':True,'HC3_independent_checks':tests,'patient_models':len(result),'skipped_models':len(skipped),'inference_uses_continuous_own_celltype_HLA_no_cutoff_search':True})
    primary=result[result.config.eq('RAMP_k30')&result.analysis.eq('main')]
    print('Patient-level standardized coefficients:',flush=True)
    print(primary.pivot(index='cell_type',columns='endpoint',values='coef').round(3).to_string(),flush=True)
    print('Corresponding BH q:',flush=True)
    print(primary.pivot(index='cell_type',columns='endpoint',values='q_bh').round(5).to_string(),flush=True)

if __name__=='__main__':run()
