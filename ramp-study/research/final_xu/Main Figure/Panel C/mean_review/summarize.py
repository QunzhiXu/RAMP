"""Equal-benchmark means for the four currently displayed metrics."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from figure_style import ROOT, track, READS, sha
import json
import pandas as pd

source = ROOT/'benchmark_results/comparison_evaluation/summary/primary_method_family_metric_summary.csv'
data = pd.read_csv(track(source))
metrics = ['leiden_celltype_ari', 'umap_celltype_asw_macro', 'dataset_ilisi_norm', 'crossplatform_knn_macro_f1']
data = data[data.metric.isin(metrics)].copy()
assert data.n_resamples.eq(20).all() and len(data) == 96
assert data.groupby(['metric', 'method_family']).size().eq(4).all()
summary = data.groupby(['metric', 'method_family'])['mean'].mean().rename('equal_benchmark_mean').reset_index()
summary['rank'] = summary.groupby('metric').equal_benchmark_mean.rank(ascending=False, method='min').astype(int)
summary.to_csv(HERE/'metric_method_means.csv', index=False)
data.to_csv(HERE/'source_benchmark_means.csv', index=False)
assert all(sha(p) == digest for p, digest in READS.items())
(HERE/'manifest.json').write_text(json.dumps({'input_sha256': READS, 'inputs_unchanged': True,
    'aggregation': 'First retain the frozen mean of 20 resamples per method and benchmark, then average the four benchmarks with equal weight.',
    'ramp_ranks': summary[summary.method_family.eq('RAMP')].set_index('metric')['rank'].to_dict()}, indent=2)+'\n')
print(summary.sort_values(['metric','rank']).to_string(index=False))
