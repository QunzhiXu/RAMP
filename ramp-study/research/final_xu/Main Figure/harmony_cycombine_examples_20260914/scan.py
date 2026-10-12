"""Read-only rescanning of frozen Harmony-direct/cyCombine marker examples."""
from pathlib import Path
import os,sys,json,itertools
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'Main Figure/pair_inversion_boxplot_20260914'))
import compute_pair_inversions as frozen
os.environ['NUMBA_CACHE_DIR']=str(HERE/'.cache/numba')
import numpy as np
import pandas as pd
from scipy.stats import rankdata

CELLS=['CD4 T','CD8 T','B cells','Monocytes']
BMETHOD={'Harmony-direct[source_scale]':'Harmony-direct','cyCombine':'cyCombine'}
def auc(x,y):
    ranks=rankdata(np.r_[x,y])
    return float((ranks[:len(x)].sum()-len(x)*(len(x)+1)/2)/(len(x)*len(y)))

def masks_for_display(case,meta):
    b={'bone_marrow':'Mature / CODEX B','tnbc':'B lineage','lung_cancer':'B-like'}.get(case,'NO_B_LABEL')
    masks={c:meta.celltype.eq(c).to_numpy() for c in ['CD4 T','CD8 T']}
    masks['B cells']=meta.celltype.eq(b).to_numpy()
    masks['Monocytes']=meta.celltype.eq('Monocyte').to_numpy()
    if case=='tnbc':
        # Narrow provider labels; do not relabel broad Myeloid or Mac as Monocytes.
        masks['Monocytes']=meta.provider_label_l1.eq('Monocyte').to_numpy()
        assert np.array_equal(masks['Monocytes'],meta.provider_label_l2.eq('Monocyte').to_numpy())
    return masks

def main():
    rules=pd.read_csv(frozen.track(ROOT/'benchmark_results/exact_value_artifacts/config/canonical_expectations.csv'))
    rows=[];coverage=[];stats=[];pairs=[];skips=[]
    for case in frozen.CASES:
        base=ROOT/'benchmark_results'/case/'ramp/alignment/input'
        meta=pd.read_csv(frozen.track(base/'cell_metadata.csv.gz'),low_memory=False,dtype={'cell_id':str})
        markers=frozen.track(base/'markers.txt').read_text().splitlines()
        matrices={'Raw':np.load(frozen.track(base/'raw.npy'))}
        for method,_,z in frozen.load_methods(case,meta.cell_id.to_numpy(),markers):
            if method in BMETHOD:matrices[BMETHOD[method]]=z
            if method=='cyCombine':break
        assert all(z.shape==matrices['Raw'].shape and np.isfinite(z).all() for z in matrices.values())
        for unit,ix in meta.groupby('unit_key',sort=True).indices.items():
            local=meta.iloc[ix];dataset=local.dataset.iloc[0]
            masks=masks_for_display(case,local)
            ns={k:int(v.sum()) for k,v in masks.items()}
            four_ok=min(ns.values())>=50
            coverage.append(dict(case=case,unit_key=unit,dataset=dataset,n_cells=len(ix),
                **{k.replace(' ','_'):v for k,v in ns.items()},four_groups_eligible=four_ok,
                markers=';'.join(markers)))
            ranks={m:rankdata(z[ix,:],axis=0)/(len(ix)+1) for m,z in matrices.items()}
            for rule in rules[rules.cancer.eq(case)].itertuples():
                if rule.dataset_allowlist!='*' and dataset not in rule.dataset_allowlist.split(';'):continue
                if rule.marker not in markers:continue
                j=markers.index(rule.marker);pos=local.celltype.isin(rule.positive_labels.split(';')).to_numpy()
                for neglabel in rule.negative_labels.split(';'):
                    neg=local.celltype.eq(neglabel).to_numpy()
                    key=dict(case=case,unit_key=unit,dataset=dataset,marker=rule.marker,rule_id=rule.rule_id,
                        target=rule.positive_labels,non_target=neglabel,n_target=int(pos.sum()),n_non_target=int(neg.sum()),four_groups_eligible=four_ok)
                    if min(pos.sum(),neg.sum())<50:
                        skips.append(key);continue
                    for method in ['Harmony-direct','cyCombine']:
                        row=dict(key,method=method)
                        for name,r in [('raw',ranks['Raw'][:,j]),('corrected',ranks[method][:,j])]:
                            row[name+'_auc']=auc(r[pos],r[neg])
                            for label,mask in [('target',pos),('non_target',neg)]:
                                for q,n in zip([.25,.5,.75],['q25','median','q75']):row[name+'_'+label+'_'+n]=float(np.quantile(r[mask],q))
                        row['auc_loss']=row['raw_auc']-row['corrected_auc']
                        row['raw_median_gap']=row['raw_target_median']-row['raw_non_target_median']
                        row['corrected_median_gap']=row['corrected_target_median']-row['corrected_non_target_median']
                        row['median_gap_loss']=row['raw_median_gap']-row['corrected_median_gap']
                        row['nontarget_median_shift']=row['corrected_non_target_median']-row['raw_non_target_median']
                        rows.append(row)
            # Every shared marker and every eligible pair among the desired four types.
            # These descriptive tables are not automatically labelled artifacts.
            for j,marker in enumerate(markers):
                eligible=[c for c in CELLS if ns[c]>=50]
                for method,r in ranks.items():
                    for c in eligible:
                        v=r[masks[c],j]
                        stats.append(dict(case=case,unit_key=unit,marker=marker,method=method,celltype=c,n_cells=len(v),
                            q25=float(np.quantile(v,.25)),median=float(np.median(v)),q75=float(np.quantile(v,.75)),four_groups_eligible=four_ok))
                    for left,right in itertools.combinations(eligible,2):
                        pairs.append(dict(case=case,unit_key=unit,marker=marker,method=method,left=left,right=right,
                            auc=auc(r[masks[left],j],r[masks[right],j]),four_groups_eligible=four_ok))
        print(case,'complete',flush=True)
    d=pd.DataFrame(rows)
    d.to_csv(HERE/'all_canonical_comparisons.csv.gz',index=False)
    pd.DataFrame(coverage).to_csv(HERE/'sample_coverage.csv',index=False)
    pd.DataFrame(skips).to_csv(HERE/'canonical_low_count_skips.csv.gz',index=False)
    pd.DataFrame(stats).to_csv(HERE/'four_type_marker_statistics.csv.gz',index=False)
    pd.DataFrame(pairs).to_csv(HERE/'four_type_pair_auc.csv.gz',index=False)
    for method in ['Harmony-direct','cyCombine']:
        q=d[d.method.eq(method)&d.raw_auc.ge(.8)].sort_values('auc_loss',ascending=False)
        q.to_csv(HERE/f'{method}_ranked_candidates.csv',index=False)
        cols=['unit_key','marker','target','non_target','raw_auc','corrected_auc','auc_loss','raw_median_gap','corrected_median_gap','four_groups_eligible']
        print('\n'+method+' top candidates',flush=True)
        print(q[cols].head(12).to_string(index=False),flush=True)
        print('\n'+method+' four-group candidates',flush=True)
        print(q[q.four_groups_eligible][cols].head(18).to_string(index=False),flush=True)
    old=pd.read_csv(frozen.track(ROOT/'Main Figure/Panel D/cycombine_review/all_unit_canonical_comparisons.csv.gz'))
    keys=['case','unit_key','marker','rule_id','target','non_target']
    joined=d[d.method.eq('cyCombine')].merge(old,on=keys,suffixes=('_new','_old'),validate='one_to_one')
    assert len(joined)==len(old)==1542
    for c in ['raw_auc','corrected_auc','auc_loss']:
        assert np.allclose(joined[c+'_new'],joined[c+'_old'],rtol=0,atol=1e-12),c
    assert all(frozen.sha(p)==v for p,v in frozen.READS.items())
    (HERE/'scan_manifest.json').write_text(json.dumps(dict(cases=frozen.CASES,sample_units=len(coverage),
        canonical_comparisons_per_method=int(len(d)/2),minimum_cells_per_pair_group=50,
        descriptive_scope='All shared markers and all >=50-cell pairs among CD4 T, CD8 T, B cells, explicitly labelled Monocytes.',
        canonical_scope='Existing frozen marker/lineage rules, no rules invented to fit the outputs.',
        candidate_filter='Raw AUROC >=0.80; no statistical significance or population prevalence inferred from example selection.',
        percentile_reference='All cells of the sample for every marker/method; no subset re-ranking.',
        cycombine_previous_scan_exact_regression=True,inputs_unchanged=True,input_sha256=frozen.READS),indent=2)+'\n')

if __name__=='__main__':main()
