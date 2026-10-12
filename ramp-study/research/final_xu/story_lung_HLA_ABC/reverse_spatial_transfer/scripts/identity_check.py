"""Check inherited gates and remove broad-label composition effects in a companion analysis."""
import runtime as rt
from runtime import BASE
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from associate import summarize,bh
OUT=BASE/'results'

def main():
    sm=pd.read_csv(OUT/'SENOSAIN_REVERSE_PREDICTIONS.csv.gz',dtype={'patient_id':str})
    raw=pd.read_csv(OUT/'SENOSAIN_MEASURED_HELDOUT_PROTEINS.csv.gz',dtype={'patient_id':str})
    assert np.array_equal(sm.source_row_id,raw.source_row_id)
    gatepath=BASE.parents[1]/'story_test_lung/shared_inputs/senosain/senosain_exact70k_paper_mesenchymal_gate.npy'
    gate=np.load(gatepath)[sm.source_row_id.to_numpy(int)].astype(bool)
    assert np.array_equal(gate,sm.paper_mesenchymal_gate)
    table=pd.crosstab(sm.broad_type,gate);table.to_csv(OUT/'GATE_VS_BROAD_LABEL.csv')
    states=[c.removeprefix('k30__') for c in sm if c.startswith('k30__r40__') or c.startswith('k30__niche__')]
    markers=['HLA_ABC','PD_L1'];rows=[]
    for pop,mask in [('paper_mesenchymal',gate),('broad_stromal',sm.broad_type.eq('Fibroblast / stroma').to_numpy()),('all_cells',np.ones(len(sm),bool))]:
        for pid in sorted(sm.patient_id.unique()):
            ix=np.flatnonzero(mask&sm.patient_id.eq(pid).to_numpy())
            if len(ix)<15:continue
            y=raw.iloc[ix][['rank__'+m for m in markers]].to_numpy()
            types=sm.iloc[ix].broad_type.to_numpy()
            # Residualize within patient and existing broad label, without defining new cell types.
            yr=rankdata(y,axis=0)
            for k in [30,100]:
                for state in states:
                    x=sm[f'k{k}__{state}'].to_numpy()[ix];xr=rankdata(x);xx=xr.copy();yy=yr.copy();ym=y.copy()
                    for t in np.unique(types):
                        ii=types==t;xx[ii]-=xx[ii].mean();yy[ii]-=yy[ii].mean(0);ym[ii]-=ym[ii].mean(0)
                    lo,hi=np.quantile(x,[1/3,2/3]);lower=x<=lo;upper=x>=hi
                    if lo>=hi:continue
                    den=np.sqrt((xx**2).sum()*(yy**2).sum(0));rs=np.divide(xx@yy,den,out=np.full(2,np.nan),where=den>0)
                    for j,m in enumerate(markers):rows.append(dict(population=pop,patient_id=pid,k=k,state=state,marker=m,n_cells=len(ix),partial_rho=rs[j],
                        residual_difference=ym[upper,j].mean()-ym[lower,j].mean()))
    a=pd.DataFrame(rows);a.to_csv(OUT/'BROAD_LABEL_ADJUSTED_DONOR_ASSOCIATIONS.csv',index=False)
    for col in ['partial_rho','residual_difference']:
        z=summarize(a,['population','k','state','marker'],col);z['q_bh']=z.groupby(['population','k','marker']).wilcoxon_p.transform(bh)
        z.to_csv(OUT/f'BROAD_LABEL_ADJUSTED_{col.upper()}.csv',index=False)
    rt.save(OUT/'IDENTITY_VERIFICATION.json',dict(original_gate_membership_exact=True,original_gate_sha256=rt.sha(gatepath),n_gate=int(gate.sum()),
        n_broad_stromal=int(sm.broad_type.eq('Fibroblast / stroma').sum()),n_overlap=int((gate&sm.broad_type.eq('Fibroblast / stroma')).sum()),
        interpretation='Two inherited annotations differ substantially. Historical gate must not be described as an author-confirmed Cords vCAF identity. Broad-label residualization is companion analysis, not a new gate.'))
    print(pd.read_csv(OUT/'BROAD_LABEL_ADJUSTED_RESIDUAL_DIFFERENCE.csv').query("population=='paper_mesenchymal' and k==30 and state=='r40__CAF_fraction'").to_string(index=False))
if __name__=='__main__':main()
