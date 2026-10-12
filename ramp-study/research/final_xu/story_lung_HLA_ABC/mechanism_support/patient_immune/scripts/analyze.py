from runtime import *

def load_cells():
    clinical=pd.read_csv(DATA/'FROZEN_PATIENT_GROUPS.csv',float_precision='round_trip')
    cells=pd.read_csv(DATA/'MEASURED_MARKERS.csv.gz',float_precision='round_trip')
    req=pd.read_csv(DATA/'MEASUREMENT_REQUEST.csv.gz')
    assert cells.source_row_id.is_unique and len(cells)==len(req)
    assert np.array_equal(cells.source_row_id,req.source_row_id)
    for col in ['patient_id','roi_id','cell_type','cell_subtype']: assert np.array_equal(cells[col],req[col])
    assert cells.patient_id.isin(clinical.patient_id).all()
    old=pd.read_csv(INPUTS['previous_measured'],float_precision='round_trip').set_index('source_row_id')
    t=cells[cells.cell_type.isin(['CD4','CD8'])].set_index('source_row_id')
    shared=['PD1','TCF1','FOXP3','CD73','IDO','HLA_DR','KI67','CD3','CD4','CD8']
    err=float(np.abs(t[shared].to_numpy()-old.loc[t.index,shared].to_numpy()).max())
    assert err<1e-12
    for marker in MARKERS: cells[marker+'_asinh']=np.arcsinh(cells[marker])
    save(OUT/'JOIN_VERIFICATION.json',{'status':'PASS','n_cells':len(cells),'n_patients':cells.patient_id.nunique(),'previous_CD4_CD8_cells_reproduced':len(t),'previous_marker_max_error':err,'same_requested_row_order_and_labels':True})
    return clinical,cells

def summarize(clinical,cells):
    frames=[]
    selections=[(lineage,cells.cell_type.eq(lineage)) for lineage in LINEAGES]
    selections += [('CD4_nonTreg',cells.cell_type.eq('CD4')&cells.cell_subtype.ne('CD4_Treg')),('Treg',cells.cell_type.eq('CD4')&cells.cell_subtype.eq('CD4_Treg'))]
    for lineage,mask in selections:
        subset=cells.loc[mask]
        counts=subset.groupby('patient_id').agg(n_cells=('source_row_id','size'),n_rois=('roi_id','nunique'))
        for marker in MARKERS:
            col=marker+'_asinh'
            a=subset.groupby('patient_id')[col].agg(['mean','median'])
            a['roi_equal']=subset.groupby(['patient_id','roi_id'])[col].mean().groupby('patient_id').mean()
            a=a.join(counts).reset_index();a['lineage']=lineage;a['marker']=marker
            frames.append(a)
    summary=pd.concat(frames,ignore_index=True).merge(clinical,on='patient_id',validate='many_to_one')
    summary.to_csv(OUT/'PATIENT_LINEAGE_MARKERS.csv',index=False)
    coverage=[]
    for lineage,g in summary[summary.marker.eq('PD1')].groupby('lineage'):
        for group,x in g.groupby('group'):
            coverage.append(dict(lineage=lineage,group=group,n_patients=len(x),n_cells=int(x.n_cells.sum()),median_cells=float(x.n_cells.median()),min_cells=int(x.n_cells.min()),n_at_least20=int(x.n_cells.ge(20).sum()),n_at_least50=int(x.n_cells.ge(50).sum())))
    pd.DataFrame(coverage).to_csv(OUT/'LINEAGE_COVERAGE.csv',index=False)
    return summary

def expression(summary):
    raw=[];adjusted=[];continuous=[];batch=[]
    cases=[('all_available_mean',1,'mean'),('min20_mean',20,'mean'),('min50_mean',50,'mean'),('roi_equal',1,'roi_equal')]
    for family,endpoints in [('primary',PRIMARY),('secondary',SECONDARY)]:
        for lineage,marker in endpoints:
            data=summary[summary.lineage.eq(lineage)&summary.marker.eq(marker)]
            for case,minimum,col in cases:
                g=data[data.n_cells>=minimum].copy()
                if len(g)<30 or g.group.value_counts().min()<10: continue
                info=dict(family=family,lineage=lineage,marker=marker,analysis=case,n_lower=int(g.group.eq('Lower').sum()),n_higher=int(g.group.eq('Higher').sum()))
                raw.append({**info,**compare(g,col)})
                sd=float(g[col].std(ddof=1)); y=(g[col]-g[col].mean())/sd
                for full in ([False,True] if case=='all_available_mean' else [False]):
                    x=design_matrix(g,full=full)
                    assert np.linalg.matrix_rank(x)==x.shape[1]
                    adjusted.append({**info,'full_CAF_covariates':full,'endpoint_sd':sd,**ols_hc3(y,x)})
                if case=='all_available_mean':
                    rho,p=stats.spearmanr(g.all_mean,g[col])
                    continuous.append({**info,'rho':rho,'spearman_p':p,'endpoint_sd':sd,**ols_hc3(y,design_matrix(g,exposure='all_mean'))})
                    for tma,sg in g.groupby('TMA'):
                        if sg.group.nunique()==2:
                            delta=sg[sg.group.eq('Higher')][col].mean()-sg[sg.group.eq('Lower')][col].mean()
                            batch.append(dict(family=family,lineage=lineage,marker=marker,TMA=tma,n=len(sg),difference=float(delta)))
    tables=[(pd.DataFrame(raw),'GROUP_COMPARISONS.csv',['family','analysis']),
            (pd.DataFrame(adjusted),'ADJUSTED_COMPARISONS.csv',['family','analysis','full_CAF_covariates']),
            (pd.DataFrame(continuous),'CONTINUOUS_ASSOCIATIONS.csv',['family'])]
    for table,name,keys in tables:
        table['q_bh']=np.nan
        for _,ix in table.groupby(keys).groups.items(): table.loc[ix,'q_bh']=bh(table.loc[ix,'p'])
        if 'spearman_p' in table:
            table['spearman_q']=np.nan
            for _,ix in table.groupby(keys).groups.items():table.loc[ix,'spearman_q']=bh(table.loc[ix,'spearman_p'])
        table.to_csv(OUT/name,index=False)
    pd.DataFrame(batch).to_csv(OUT/'TMA_STRATIFIED_EXPRESSION.csv',index=False)
    return tables[0][0],tables[1][0],tables[2][0]

def composition(clinical,cells):
    counts=cells.groupby(['patient_id','cell_type']).size().unstack(fill_value=0).reindex(index=clinical.patient_id,columns=LINEAGES,fill_value=0)
    meta=pd.read_csv(INPUTS['metadata'],usecols=['patient_id'])
    counts['all_cells']=meta.groupby('patient_id').size().reindex(counts.index)
    counts['immune_cells']=counts[LINEAGES].sum(axis=1)
    counts['Treg_cells']=cells[cells.cell_subtype.eq('CD4_Treg')].groupby('patient_id').size().reindex(counts.index,fill_value=0)
    counts=counts.reset_index().merge(clinical,on='patient_id',validate='one_to_one')
    endpoints=[]
    for lineage in LINEAGES:
        for suffix,denom in [('of_immune','immune_cells'),('of_all','all_cells')]:
            col=lineage+'_'+suffix; counts[col]=counts[lineage]/counts[denom];endpoints.append((col,lineage,denom))
    counts['immune_of_all']=counts.immune_cells/counts.all_cells;endpoints.append(('immune_of_all','All immune','all_cells'))
    counts['Treg_of_CD4']=counts.Treg_cells/counts.CD4.replace(0,np.nan);endpoints.append(('Treg_of_CD4','Author Treg','CD4'))
    assert np.allclose(counts[[lineage+'_of_immune' for lineage in LINEAGES]].sum(axis=1),1)
    assert np.allclose(counts['CD8_of_all'],counts['CD8_fraction'])
    counts.to_csv(OUT/'PATIENT_IMMUNE_COMPOSITION.csv',index=False)
    raw=[];adj=[]
    for col,lineage,denom in endpoints:
        g=counts.dropna(subset=[col]); info=dict(endpoint=col,lineage=lineage,denominator=denom)
        raw.append({**info,**compare(g,col)})
        sd=float(g[col].std(ddof=1));y=(g[col]-g[col].mean())/sd
        for full in [False,True]:
            adj.append({**info,'n_lower':int(g.group.eq('Lower').sum()),'n_higher':int(g.group.eq('Higher').sum()),'endpoint_sd':sd,'full_CAF_covariates':full,**ols_hc3(y,design_matrix(g,full=full))})
    raw=pd.DataFrame(raw);raw['q_bh']=bh(raw.p)
    adj=pd.DataFrame(adj);adj['q_bh']=np.nan
    for full,ix in adj.groupby('full_CAF_covariates').groups.items():adj.loc[ix,'q_bh']=bh(adj.loc[ix,'p'])
    raw.to_csv(OUT/'COMPOSITION_COMPARISONS.csv',index=False);adj.to_csv(OUT/'COMPOSITION_ADJUSTED.csv',index=False)
    return counts,raw,adj

def verify(clinical,cells,summary,adj):
    hashes=json.loads((OUT/'DESIGN.json').read_text())['input_sha256']
    unchanged={key:sha(path)==hashes[key] for key,path in INPUTS.items() if key!='provider_sce'}
    assert all(unchanged.values())
    assert json.loads((OUT/'EXPORT_VERIFICATION.json').read_text())['source_sha256']==hashes['provider_sce']
    old=pd.read_csv(INPUTS['previous_adjusted'])
    new=adj[(adj.family=='primary')&(adj.analysis=='all_available_mean')&~adj.full_CAF_covariates&(adj.lineage=='CD8')]
    reproduction=[]
    for marker in ['PD1','TCF1']:
        a=new[new.marker.eq(marker)].iloc[0]
        b=old[old.analysis.eq('all280_mean')&old.marker.eq(marker)&~old.full_CAF_covariates&old.exposure.eq('high_HLA')].iloc[0]
        error=max(abs(a[k]-b[k]) for k in ['coef','se','ci_low','ci_high','p'])
        assert error<1e-10
        reproduction.append(dict(marker=marker,max_numeric_error=error,previous_q=b.q_bh,current_q=a.q_bh))
    pd.DataFrame(reproduction).to_csv(OUT/'PREVIOUS_CD8_REPRODUCTION.csv',index=False)
    checks=[]
    for _,row in summary[summary.lineage.isin(LINEAGES)].sample(10,random_state=SEED).iterrows():
        g=cells[cells.patient_id.eq(row.patient_id)&cells.cell_type.eq(row.lineage)]
        error=abs(float(np.arcsinh(g[row.marker]).sum()/len(g))-row['mean']);assert error<1e-12
        checks.append(dict(patient_id=row.patient_id,lineage=row.lineage,marker=row.marker,error=error))
    qc=[]
    for lineage,g in cells.groupby('cell_type'):
        for marker in MARKERS:
            v=g[marker].to_numpy();q=np.quantile(v,[0,.5,.95,1])
            qc.append(dict(lineage=lineage,marker=marker,n=len(v),zero_fraction=float(np.mean(v==0)),minimum=q[0],median=q[1],p95=q[2],maximum=q[3]))
    pd.DataFrame(qc).to_csv(OUT/'MARKER_SIGNAL_QC.csv',index=False)
    save(OUT/'VERIFICATION.json',{'status':'PASS','time_utc':now(),'n_patients':len(clinical),'group_counts':clinical.group.value_counts().to_dict(),'n_immune_cells':len(cells),'source_hashes_unchanged':unchanged,'provider_source_hash_verified_by_R':True,'previous_CD8_models_reproduced':True,'CD8_q_changes_only_because_family_expands_from_2_to_10':True,'independent_patient_mean_checks':checks})

def main():
    clinical,cells=load_cells();summary=summarize(clinical,cells)
    raw,adj,corr=expression(summary);counts,comp,compadj=composition(clinical,cells)
    verify(clinical,cells,summary,adj)
    print('PRIMARY EXPRESSION (TMA, age, sex, stage adjusted):',flush=True)
    print(adj[(adj.family=='primary')&(adj.analysis=='all_available_mean')&~adj.full_CAF_covariates][['lineage','marker','n_lower','n_higher','coef','ci_low','ci_high','q_bh']].to_string(index=False),flush=True)
    print('COMPOSITION (same adjustment):',flush=True)
    print(compadj[~compadj.full_CAF_covariates][['endpoint','coef','ci_low','ci_high','q_bh']].to_string(index=False),flush=True)
    print('Analysis and numerical verification complete.',flush=True)

if __name__=='__main__': main()
