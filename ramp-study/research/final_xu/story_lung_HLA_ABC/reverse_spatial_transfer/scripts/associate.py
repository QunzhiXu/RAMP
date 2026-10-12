"""Read held-out proteins and outcomes only after spatial definitions and reverse transfer freeze."""
import runtime as rt
from runtime import BASE
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata,spearmanr,wilcoxon

OUT=BASE/'results';S=BASE.parent;ROOT=S.parent;SEED=20260909

def bh(p):
    p=np.asarray(p,float);ok=np.isfinite(p);res=np.full(len(p),np.nan);v=p[ok];i=np.argsort(v)
    ans=np.empty(len(v));ans[i]=np.minimum(1,np.minimum.accumulate((v[i]*len(v)/np.arange(1,len(v)+1))[::-1])[::-1]);res[ok]=ans;return res

def rho(x,y):
    x=np.asarray(x);y=np.asarray(y);ok=np.isfinite(x)&np.isfinite(y);x=x[ok];y=y[ok]
    return float(spearmanr(x,y).statistic) if len(x)>=3 and np.ptp(x)>0 and np.ptp(y)>0 else np.nan

def summarize(df,keys,col):
    rows=[]
    for key,g in df.groupby(keys,sort=False,dropna=False):
        x=g[col].dropna().to_numpy();rng=np.random.default_rng(SEED)
        if not len(x):continue
        lo,hi=np.quantile(x[rng.integers(len(x),size=(2000,len(x)))].mean(1),[.025,.975])
        rows.append(dict(zip(keys,key),n_patients=len(x),mean=x.mean(),median=np.median(x),ci_low=lo,ci_high=hi,
            fraction_positive=np.mean(x>0),wilcoxon_p=wilcoxon(x).pvalue if np.any(x!=0) else 1.))
    return pd.DataFrame(rows)

def main():
    freeze=json.loads((OUT/'PREDICTION_FREEZE.json').read_text())
    for n,h in freeze['output_sha256'].items():assert rt.sha(OUT/n)==h
    paths={'raw_proteins':ROOT/'story_test_lung/shared_inputs/senosain/senosain_exact70k_raw35.npy',
        'marker_names':ROOT/'story_test_lung/shared_inputs/senosain/senosain_raw35_markers.csv',
        'hla_predictions':S/'spatial/results/VCAF_CURRENT_PREDICTIONS_AND_DISTANCES.csv.gz',
        'clinical':S/'survival/results/PATIENT_INPUTS.csv'}
    hashes={k:rt.sha(p) for k,p in paths.items()}
    rt.save(OUT/'ASSOCIATION_INPUTS.json',dict(started=rt.now(),spatial_prediction_freeze=freeze,input_sha256=hashes,input_paths={k:str(p) for k,p in paths.items()}))
    sm=pd.read_csv(OUT/'SENOSAIN_REVERSE_PREDICTIONS.csv.gz',dtype={'patient_id':str},low_memory=False,float_precision='round_trip')
    allmarkers=pd.read_csv(paths['marker_names']).marker.tolist()
    shared=['CD3','CD4','CD8','CD31','KI67','VIMENTIN','CYTOKERATIN','CD45','HLA_DR']
    markers=[n for n in allmarkers if n not in shared];assert 'HLA_ABC' in markers and 'PD_L1' in markers and len(markers)==26
    raw=np.load(paths['raw_proteins'],mmap_mode='r')[sm.source_row_id.to_numpy(int)][:,[allmarkers.index(n) for n in markers]]
    ranks=np.empty_like(raw,dtype=float)
    for pid,ix in sm.groupby('patient_id',sort=False).indices.items():ranks[ix]=rankdata(raw[ix],axis=0)/(len(ix)+1)
    response=sm[['source_row_id','patient_id','provider_cell_type','broad_type','paper_mesenchymal_gate','reference_index']].copy()
    for j,m in enumerate(markers):response[f'measured__{m}']=raw[:,j];response[f'rank__{m}']=ranks[:,j]
    response.to_csv(OUT/'SENOSAIN_MEASURED_HELDOUT_PROTEINS.csv.gz',index=False)
    states=[c.removeprefix('k30__') for c in sm if c.startswith('k30__')]
    primary=[c for c in states if c.startswith('r40__') or c.startswith('niche__')]
    records=[];corrs=[];coverage=[]
    pops={'paper_mesenchymal':sm.paper_mesenchymal_gate.eq(True).to_numpy(),
          'broad_stromal':sm.broad_type.eq('Fibroblast / stroma').to_numpy(),'all_cells':np.ones(len(sm),bool)}
    for pop,pm in pops.items():
        for pid in sorted(sm.patient_id.unique()):
            ix=np.flatnonzero(pm&sm.patient_id.eq(pid).to_numpy());coverage.append(dict(population=pop,patient_id=pid,n_cells=len(ix),eligible=len(ix)>=15))
            if len(ix)<15:continue
            y=ranks[ix]
            # Matrix Spearman, retaining constant-response markers as NaN.
            yr=rankdata(y,axis=0);yr-=yr.mean(0);ynorm=np.sqrt((yr**2).sum(0))
            for k in [30,100]:
                for state in states:
                    x=sm[f'k{k}__{state}'].to_numpy()[ix];ok=np.isfinite(x)
                    if not ok.all():raise ValueError('Missing target spatial score needs explicit handling')
                    if np.ptp(x)==0:continue
                    xr=rankdata(x);xr-=xr.mean();den=np.sqrt((xr**2).sum())*ynorm
                    rs=np.divide(xr@yr,den,out=np.full(len(markers),np.nan),where=den>0)
                    q1,q2=np.quantile(x,[1/3,2/3]);lo=x<=q1;hi=x>=q2
                    for j,m in enumerate(markers):
                        base=dict(population=pop,patient_id=pid,k=k,state=state,marker=m,n_cells=len(ix),primary_state=state in primary)
                        corrs.append(dict(base,rho=rs[j]))
                        if q1>=q2 or lo.sum()<5 or hi.sum()<5:continue
                        records.append(dict(base,n_low=int(lo.sum()),n_high=int(hi.sum()),low_spatial=x[lo].mean(),high_spatial=x[hi].mean(),
                            low=y[lo,j].mean(),high=y[hi,j].mean(),difference=y[hi,j].mean()-y[lo,j].mean()))
        print(f'Measured proteins: {pop} complete',flush=True)
    pp=pd.DataFrame(records);cp=pd.DataFrame(corrs)
    pp.to_csv(OUT/'SENOSAIN_DONOR_PROTEIN_CONTRASTS.csv.gz',index=False);cp.to_csv(OUT/'SENOSAIN_DONOR_PROTEIN_CORRELATIONS.csv.gz',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'SENOSAIN_COVERAGE.csv',index=False)
    keys=['population','k','state','marker','primary_state']
    for filename,df,col in [('SENOSAIN_PROTEIN_CONTRASTS',pp,'difference'),('SENOSAIN_PROTEIN_CORRELATIONS',cp,'rho')]:
        summ=summarize(df,keys,col)
        summ['q_all_heldout']=summ.groupby(['population','k','primary_state']).wilcoxon_p.transform(bh)
        summ['q_HLA_states']=np.nan;h=summ.marker.eq('HLA_ABC')
        summ.loc[h,'q_HLA_states']=summ[h].groupby(['population','k','primary_state']).wilcoxon_p.transform(bh)
        if col=='difference':summ=summ.merge(df.groupby(keys)[['low','high']].mean().reset_index(),on=keys,validate='one_to_one')
        summ.to_csv(OUT/(filename+'.csv'),index=False)
    # Cords linkage uses author vCAF annotations and the original frozen HLA predictions.
    q=pd.read_csv(OUT/'CORDS_SPATIAL_STATES.csv.gz',dtype={'patient_id':str,'roi_id':str},float_precision='round_trip')
    v=q[q.cell_type.eq('vCAF')].copy()
    h=pd.read_csv(paths['hla_predictions'],usecols=['source_row_id','patient_id','roi_id','RAMP_k30','RAMP_k100'],dtype={'patient_id':str,'roi_id':str},float_precision='round_trip')
    v=v.merge(h,on=['source_row_id','patient_id','roi_id'],validate='one_to_one');assert len(v)==52328
    v.to_csv(OUT/'VCAF_SPATIAL_STATES_AND_HLA.csv.gz',index=False)
    agg=v.groupby('patient_id')[states+['RAMP_k30','RAMP_k100']].mean().reset_index()
    agg=agg.merge(v.groupby('patient_id').size().rename('n_vcaf'),on='patient_id');agg.to_csv(OUT/'CORDS_PATIENT_SPATIAL_SCORES.csv',index=False)
    clinical=pd.read_csv(paths['clinical'],dtype={'patient_id':str},float_precision='round_trip')
    clinical=clinical[clinical.config.eq('RAMP_k30')].copy();assert len(clinical)==280
    clinical=clinical.merge(agg,on='patient_id',validate='one_to_one');assert len(clinical)==280
    meanerr=float(np.max(np.abs(clinical.all_mean-clinical.RAMP_k30)));assert meanerr<1e-12
    clinical.to_csv(OUT/'SURVIVAL_INPUTS.csv',index=False)
    rows=[];roirows=[]
    for pop,df in [('all_vCAF_patients',agg[agg.n_vcaf>=15]),('DFS_280',clinical)]:
        for k in [30,100]:
            for state in states:
                rr=rho(df[state],df[f'RAMP_k{k}']);test=spearmanr(df[state],df[f'RAMP_k{k}'],nan_policy='omit')
                rows.append(dict(population=pop,k=k,state=state,n_patients=len(df),rho=rr,p=test.pvalue,primary_state=state in primary))
    a=pd.DataFrame(rows);a['q_bh']=a.groupby(['population','k','primary_state']).p.transform(bh);a.to_csv(OUT/'CORDS_PATIENT_HLA_ASSOCIATIONS.csv',index=False)
    for (pid,roi),g in v.groupby(['patient_id','roi_id']):
        if len(g)<15:continue
        for k in [30,100]:
            for state in states:
                rr=rho(g[state],g[f'RAMP_k{k}'])
                roirows.append(dict(patient_id=pid,roi_id=roi,k=k,state=state,n_cells=len(g),rho=rr,primary_state=state in primary))
    roi=pd.DataFrame(roirows);roi.to_csv(OUT/'CORDS_ROI_HLA_ASSOCIATIONS.csv',index=False)
    pat=roi.groupby(['patient_id','k','state','primary_state'],as_index=False).rho.mean();pat.to_csv(OUT/'CORDS_WITHIN_ROI_PATIENT_HLA_ASSOCIATIONS.csv',index=False)
    cs=summarize(pat,['k','state','primary_state'],'rho');cs['q_bh']=cs.groupby(['k','primary_state']).wilcoxon_p.transform(bh);cs.to_csv(OUT/'CORDS_WITHIN_ROI_HLA_SUMMARY.csv',index=False)
    # Preserve niche-specific HLA means as descriptive data, without choosing a survival threshold.
    niche=v.groupby(['patient_id','niche']).agg(n_vcaf=('source_row_id','size'),HLA_k30=('RAMP_k30','mean'),HLA_k100=('RAMP_k100','mean')).reset_index()
    niche.to_csv(OUT/'VCAF_NICHE_SPECIFIC_HLA.csv',index=False)
    assert all(rt.sha(paths[k])==h for k,h in hashes.items())
    for n,h in freeze['output_sha256'].items():assert rt.sha(OUT/n)==h
    rt.save(OUT/'ASSOCIATION_VERIFICATION.json',dict(finished=rt.now(),source_hashes_unchanged=True,spatial_prediction_hashes_unchanged=True,
        measured_proteins=markers,alignment_input_markers=shared,target_HLA_excluded=True,n_senosain_cells=len(sm),n_paper_mesenchymal=int(pops['paper_mesenchymal'].sum()),
        n_cords_vcaf=len(v),n_survival=280,original_vcaf_HLA_mean_max_error=meanerr))
    print(pd.read_csv(OUT/'SENOSAIN_PROTEIN_CONTRASTS.csv').query("population=='paper_mesenchymal' and k==30 and marker=='HLA_ABC' and primary_state").sort_values('mean',ascending=False).to_string(index=False),flush=True)
    print(a.query("population=='DFS_280' and k==30 and primary_state").sort_values('rho',ascending=False).to_string(index=False),flush=True)

if __name__=='__main__':main()
