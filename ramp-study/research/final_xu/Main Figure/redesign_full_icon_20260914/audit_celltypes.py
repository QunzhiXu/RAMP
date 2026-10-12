"""Inspect every annotated cell type in the two existing artifact examples."""
from pathlib import Path
import json
import rendering_helpers as h
import numpy as np
import pandas as pd
from scipy.stats import rankdata, mannwhitneyu

HERE=Path(__file__).resolve().parent
EXAMPLES=[('tnbc','Meyer IMC|B12.64192','CD8','CD8 T'),
          ('bone_marrow','BM CyTOF|H1','CD4','CD4 T')]

def main():
    values=[];stats=[];aucs=[]
    for case,unit,marker,target in EXAMPLES:
        meta,outputs=h.load_artifact(case,unit,marker)
        assert meta.cell_id.is_unique
        positive=meta.celltype.eq(target).to_numpy()
        for method,v in outputs.items():
            percentile=rankdata(v,method='average')/(len(v)+1)
            values.append(pd.DataFrame({'benchmark':case,'unit_key':unit,'marker':marker,
                'method':method,'cell_id':meta.cell_id,'celltype':meta.celltype,
                'value':v,'percentile':percentile}))
            for label in sorted(meta.celltype.unique()):
                mask=meta.celltype.eq(label).to_numpy()
                q1,med,q3=np.quantile(percentile[mask],[.25,.5,.75])
                stats.append(dict(benchmark=case,unit_key=unit,marker=marker,method=method,
                    celltype=label,n_cells=int(mask.sum()),scope_cells=len(v),
                    q1=q1,median=med,q3=q3,mean_percentile=float(percentile[mask].mean()),
                    mean_value=float(v[mask].mean())))
                if label!=target:
                    auc=mannwhitneyu(v[positive],v[mask],alternative='greater').statistic/(positive.sum()*mask.sum())
                    aucs.append(dict(case=case,scope=unit,marker=marker,method=method,
                        target=target,other=label,target_vs_other_auc=float(auc),n_other=int(mask.sum())))
    full=pd.concat(values,ignore_index=True)
    full.to_csv(HERE/'artifact_cells.csv.gz',index=False)
    pd.DataFrame(stats).to_csv(HERE/'all_celltype_statistics.csv',index=False)
    a=pd.DataFrame(aucs)
    a.to_csv(HERE/'all_celltype_auc.csv',index=False)
    # Confirm the existing two-cell-type contrasts are unchanged.
    old=pd.read_csv(h.track(h.MAIN/'Panel D/artifact_auc.csv'))
    for row in old.itertuples():
        other='CD4 T' if row.case=='tnbc' else 'Mature / CODEX B'
        new=a[a.case.eq(row.case)&a.method.eq(row.method)&a.other.eq(other)].iloc[0]
        assert np.isclose(new.target_vs_other_auc,row.target_vs_other_auc,rtol=0,atol=1e-12)
    for case,_,_,_ in EXAMPLES:
        raw=full[full.benchmark.eq(case)&full.method.eq('Raw')].set_index('cell_id').percentile
        ramp=full[full.benchmark.eq(case)&full.method.eq('RAMP')].set_index('cell_id').percentile
        assert np.array_equal(raw.to_numpy(),ramp.loc[raw.index].to_numpy())
    assert all(h.sha(p)==s for p,s in h.READS.items())
    (HERE/'audit_manifest.json').write_text(json.dumps({'source_sha256':h.READS,
        'source_unchanged':True,'old_ten_auc_values_unchanged':True,
        'ramp_raw_percentiles_identical':True,'all_celltypes_scanned':True,
        'scope':'Same two previously selected sample examples; no donor or marker search.',
        'rank_scope':'All cells in each stated sample separately within each method; not just plotted cell types.'},indent=2)+'\n')
    print(pd.DataFrame(stats).pivot_table(index=['benchmark','celltype','n_cells'],columns='method',values='median').round(3).to_string())
    print(a.pivot_table(index=['case','other'],columns='method',values='target_vs_other_auc').round(3).to_string())

if __name__=='__main__':main()
