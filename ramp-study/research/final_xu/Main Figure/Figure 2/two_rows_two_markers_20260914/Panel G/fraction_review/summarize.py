"""Read-only reuse of frozen gates to evaluate sample-level fraction errors."""
from pathlib import Path
import json,hashlib
import pandas as pd

HERE=Path(__file__).resolve().parent
FIG2=HERE.parents[2]
SOURCE=FIG2/'value_evaluation_20260914/results'
CASES=['bone_marrow','tnbc','lung_cancer','melanoma']
hashes={};frames=[]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
for case in CASES:
    path=SOURCE/case/'gate_target_unit_scores.csv'
    hashes[str(path)]=sha(path)
    frames.append(pd.read_csv(path,float_precision='round_trip'))
d=pd.concat(frames,ignore_index=True)
d=d[d.primary_eligible].copy()
d['absolute_fraction_error']=(d.predicted_positive_fraction-d.reference_positive_fraction).abs()
direction=d.groupby(['benchmark','marker','method','source_dataset','target_dataset'],as_index=False).absolute_fraction_error.mean()
task=direction.groupby(['benchmark','marker','method'],as_index=False).absolute_fraction_error.mean()
case=task.groupby(['benchmark','method'],as_index=False).absolute_fraction_error.mean()
overall=case.groupby('method',as_index=False).absolute_fraction_error.mean().sort_values('absolute_fraction_error')
for df,name in [(d,'sample_direction_errors.csv'),(direction,'direction_means.csv'),(task,'task_means.csv'),
                (case,'benchmark_means.csv'),(overall,'overall_equal_benchmark.csv')]:
    df['error_percentage_points']=100*df.absolute_fraction_error
    df.to_csv(HERE/name,index=False)
assert d.groupby('method').size().eq(458).all()
assert task.groupby('method').size().eq(11).all()
assert direction.groupby(['benchmark','marker','method']).size().eq(6).all()
assert all(sha(path)==h for path,h in hashes.items())
(HERE/'manifest.json').write_text(json.dumps({'input_sha256':hashes,'all_inputs_unchanged':True,
    'definition':'Absolute difference between gate-positive fraction and annotation-reference-positive fraction within the frozen task parent population.',
    'aggregation':'Equal target units within each direction; six directions within each task; tasks within benchmark; four benchmarks equally.',
    'scope':'All 11 original tasks and all 458 eligible unit-task-direction records per method.',
    'exploratory_selection':'Sample-fraction endpoint examined after viewing existing gating performance; not a new independent validation.',
    'refits':False},indent=2)+'\n')
print(overall.to_string(index=False))
