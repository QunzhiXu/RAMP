"""Read-only audit of frozen metric summaries; does not refit or exclude metrics."""
from pathlib import Path
import hashlib
import json
import pandas as pd

OUT=Path(__file__).resolve().parent
FIG2=OUT.parents[1]
ROOT=FIG2.parents[1]
SOURCE=FIG2/'curve_review_20260914/source_values_and_weights.csv'
REGISTRY=ROOT/'benchmark_results/comparison_evaluation/summary/metric_registry.csv'
CODE=ROOT/'benchmark_results/comparison_evaluation/scripts/evaluate_benchmark.py'

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def rationale(m):
    if m in ['umap_unit_abs_asw','umap_dataset_abs_asw']:
        return ('Secondary display diagnostic', 'Global UMAP silhouette is not conditioned on cell type; composition and donor biology can confound a pure technical mixing interpretation. Absolute value is taken after macro-averaging, unlike the cell-type-conditioned mean of absolute silhouettes. Apply the same rule to both metrics.')
    if m=='umap_celltype_asw_macro':
        return ('Secondary display diagnostic', 'Useful for the displayed UMAP, but projection and cluster geometry affect it. The same geometric limitations apply even though RAMP performs well.')
    if m.startswith(('leiden_','louvain_')):
        if '_dataset_' in m or '_unit_' in m:
            return ('Review redundant/global mixing readouts', 'Global cluster-versus-batch association can reflect unequal cell-type composition; compare with cell-type-conditioned mixing. Leiden/Louvain and ARI/NMI are related readouts; deduplicate by definition across all methods, not by RAMP rank.')
        return ('Retain representative biology metrics', 'Cluster agreement with cell-type annotations is relevant. Leiden/Louvain and ARI/NMI overlap; retain a declared representative set and preserve the complete table.')
    if m=='celltype_knn_purity':
        return ('Retain', 'Same-label fraction in 30-NN measures local cell-type mixing. RAMP has a real lower score on some comparisons; cell-frequency weighting is a limitation, not grounds for removal.')
    if m=='celltype_lisi_purity':
        return ('Retain or transparently deduplicate local-purity group', 'A related local label-diversity measure, implemented here as uniform 30-NN inverse Simpson rather than canonical graph-LISI. Redundancy with kNN purity is measurable but does not make it meaningless.')
    if m=='celltype_graph_connectivity':
        return ('Retain', 'Largest same-type connected component quantifies fragmentation. Small score differences can become whole-rank differences, but lung RAMP is appreciably below MaxFuse.')
    if m=='highdim_celltype_asw_macro':
        return ('Retain with geometric caveat', 'Cell-type separation in the evaluated representation; complements local purity and clustering, but silhouette favors compact separated shapes. Not grounds to remove only a weak result.')
    if m in ['highdim_dataset_asw_conditional','highdim_unit_asw_conditional','conditional_kbet_acceptance','conditional_cross_dataset_30nn_fraction']:
        return ('Retain conditional mixing evidence', 'Conditions mixing on cell identity, reducing composition confounding. Silhouette and custom kBET-like/neighbor estimators still have their documented assumptions.')
    if m.startswith('crossplatform_knn_'):
        return ('Retain', 'Cell-type transfer across platforms is a relevant downstream check. Uses frozen transductively aligned representations, not an independently held-out alignment fit; macro F1 and balanced accuracy are complementary but correlated.')
    if m=='crossplatform_mnn_coverage':
        return ('Secondary matching-availability diagnostic', 'Cross-platform mutual-neighbor coverage measures availability of matches, not true same-cell pairing accuracy; ground-truth paired cells are absent.')
    return ('Secondary global mixing / redundancy review', 'Global local-neighborhood mixing is relevant but may favor mixing different cell types and is sensitive to category composition. Read alongside conditional mixing and biology preservation, not as an isolated objective.')

def main():
    before={str(p):sha(p) for p in [SOURCE,REGISTRY,CODE]}
    d=pd.read_csv(SOURCE,float_precision='round_trip')
    assert len(d)==720 and d.groupby('method_family').size().eq(120).all()
    ramp=d[d.method_family.eq('RAMP')]
    summary=ramp.groupby(['metric_family','metric','direction']).agg(
        ramp_mean_rank=('rank','mean'),ramp_worst_rank=('rank','max'),
        ramp_first_places=('rank',lambda s:int(s.eq(1).sum())),
        ramp_mean_range_normalized_gap=('range_normalized_gap','mean')).reset_index()
    registry=pd.read_csv(REGISTRY)
    summary=summary.merge(registry[['metric','definition']],on='metric',validate='one_to_one')
    summary['recommendation']=summary.metric.map(lambda m:rationale(m)[0])
    summary['reason']=summary.metric.map(lambda m:rationale(m)[1])
    summary.sort_values(['ramp_mean_rank','metric'],ascending=[False,True]).to_csv(OUT/'all_30_metrics_audit.csv',index=False)
    d.to_csv(OUT/'all_methods_metric_values.csv',index=False)
    ranks=d.pivot(index=['benchmark','method_family'],columns='metric',values='rank')
    corr=ranks.corr(method='spearman')
    corr.to_csv(OUT/'rank_profile_correlations.csv')
    pairs=[]
    for i,a in enumerate(corr.index):
        for b in corr.index[i+1:]:
            pairs.append({'metric_a':a,'metric_b':b,'spearman_across_24_benchmark_method_entries':corr.loc[a,b]})
    pd.DataFrame(pairs).sort_values('spearman_across_24_benchmark_method_entries',ascending=False).to_csv(OUT/'redundancy_pairs.csv',index=False)
    families=d.groupby(['metric_family','method_family'])['rank'].mean().unstack()
    families.to_csv(OUT/'category_mean_ranks.csv')
    assert all(sha(Path(p))==v for p,v in before.items())
    (OUT/'manifest.json').write_text(json.dumps({'input_sha256':before,
        'inputs_unchanged':True,'metric_scope_changed':False,'alignment_refitted':False,
        'n_metrics':30,'n_benchmarks':4,'n_methods':6,
        'correlation_scope':'Descriptive Spearman correlation of rank profiles over 24 method-by-benchmark combinations, not 24 independent experimental replicates.',
        'review_scope':'All 30 metrics reviewed, including RAMP-favorable metrics. No score-based subset search and no post-exclusion ranking recomputation.'},indent=2)+'\n')

if __name__=='__main__':main()
