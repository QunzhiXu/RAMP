"""Replace only lung_story using the completed new all-method evaluation."""
from pathlib import Path
import json,hashlib
import pandas as pd
import numpy as np
B=Path(__file__).resolve().parent;F=B.parent;ROOT=B.parents[2];C=ROOT/'story_lung_HLA_ABC/Comparison_senosain140k_20260921';S=B/'selection'
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return pd.read_csv(p,float_precision='round_trip')
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def markdown(frame):
 return '\n'.join(['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |',*['| '+' | '.join(f'{v:.4f}' if isinstance(v,float) else str(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)]])
assert json.loads((C/'COMPLETE.json').read_text())['status']=='PASS'
policy=json.loads((C/'METRIC_POLICY.json').read_text());assert len(policy['included_metrics'])==23
sources=[F/'metric_selection_main23/metric_ranks_7_benchmarks_main23.csv',C/'evaluation/metric_method_ranks_all31.csv',ROOT/'benchmark_results/comparison_evaluation/summary/primary_method_family_metric_ranks.csv',ROOT/'figure_melanoma/Comparison/maxfuse_full_panel/evaluation/metric_method_ranks_all31.csv',ROOT/'figure_HNSCC/Comparison/maxfuse_full_panel/evaluation/metric_method_ranks_all31.csv']
hashes={str(p):sha(p) for p in sources}
old=read(sources[0]);new=read(sources[1]);cols=old.columns.tolist();new=new[cols];assert len(new)==186 and new.benchmark.eq('lung_story').all()
rows=pd.concat([old[old.benchmark.ne('lung_story')],new[new.metric.isin(policy['included_metrics'])]],ignore_index=True)
assert len(rows)==966 and not rows.duplicated(['benchmark','method','metric']).any()
keys=['benchmark','method','metric'];a=rows[rows.benchmark.ne('lung_story')].sort_values(keys).reset_index(drop=True);b=old[old.benchmark.ne('lung_story')].sort_values(keys).reset_index(drop=True);pd.testing.assert_frame_equal(a,b,check_exact=True)
assert rows.groupby(['benchmark','method']).size().eq(23).all()
registry=read(F/'metric_selection_main23/metric_registry_main23.csv');assert registry.groupby('family').size().to_dict()=={'biology':5,'mixing':14,'transfer':2,'visual':2}
rows.to_csv(S/'metric_ranks_7_benchmarks_main23.csv',index=False);registry.to_csv(S/'metric_registry_main23.csv',index=False)
# No re-ranking category means: raw per-metric ranks -> category means -> task means -> 7-task mean.
def summarize(frame):
 family=frame.groupby(['benchmark','method','family'])['rank'].mean()
 cat=family.groupby(['benchmark','method']).mean().rename('family_balanced_mean_rank')
 metric=frame.groupby(['benchmark','method'])['rank'].mean().rename('metric_equal_mean_rank')
 return pd.concat([cat,metric],axis=1).reset_index()
tasks=summarize(rows);tasks['n_metrics']=23;tasks.to_csv(S/'method_ranking_by_benchmark_main23.csv',index=False)
primary=tasks[['benchmark','method','family_balanced_mean_rank']].rename(columns={'family_balanced_mean_rank':'mean_rank'});primary['weighting']='family_equal';primary['position']=primary.groupby('benchmark').mean_rank.rank(method='min').astype(int);primary.to_csv(S/'primary_method_ranking_by_benchmark.csv',index=False)
over=[]
for w,col in [('family_equal','family_balanced_mean_rank'),('metric_equal','metric_equal_mean_rank')]:
 d=tasks.groupby('method')[col].mean().rename('mean_rank').reset_index();d['weighting']=w;d['overall_position']=d.mean_rank.rank(method='min').astype(int);over.append(d.sort_values('mean_rank'))
over=pd.concat(over,ignore_index=True);over.to_csv(S/'overall_method_ranking_main23.csv',index=False);over[over.weighting.eq('family_equal')].to_csv(S/'primary_overall_method_ranking.csv',index=False)
# All31 sensitivity table mirrors existing manuscript Table S1, now with the same refreshed task.
all4=read(sources[2]).rename(columns={'metric_family':'family','method_family':'method'})[cols]
all31=pd.concat([all4,new,read(sources[3])[cols],read(sources[4])[cols]],ignore_index=True);assert len(all31)==1302
pd.testing.assert_frame_equal(all31[all31.metric.isin(policy['included_metrics'])].sort_values(keys).reset_index(drop=True),rows.sort_values(keys).reset_index(drop=True),check_exact=False,rtol=0,atol=1e-14)
all31.to_csv(S/'metric_ranks_7_benchmarks_all31.csv',index=False)
sens31=summarize(all31).groupby('method')[['family_balanced_mean_rank','metric_equal_mean_rank']].mean();sens23=tasks.groupby('method')[['family_balanced_mean_rank','metric_equal_mean_rank']].mean()
sens23=sens23.add_prefix('main23_');sens31=sens31.add_prefix('all31_');sensitivity=sens23.join(sens31);sensitivity.to_csv(S/'SENSITIVITY_ALL23_ALL31.csv')
policy.update(version='lung_reference_update_20260921',source_sha256=hashes,models_and_umaps_refit=True,refit_scope='lung_story only; all 5 comparator methods; RAMP new application alignment reused',new_score_reporting_policy='Existing list/weights fixed before new scores; no metric additions or exclusions this update',original_metric_policy_sha256=sha(C/'METRIC_POLICY.json'))
save(S/'metric_selection.json',policy)
oldmean=read(F/'metric_selection_main23/primary_overall_method_ranking.csv');newmean=over[over.weighting.eq('family_equal')];comparison=newmean.merge(oldmean,on=['method','weighting'],suffixes=('_new','_old'));comparison.to_csv(S/'OVERALL_OLD_NEW.csv',index=False)
new.merge(old[old.benchmark.eq('lung_story')],on=['benchmark','method','metric','family','direction'],suffixes=('_new','_old')).to_csv(S/'LUNG_MAIN23_OLD_NEW.csv',index=False)
save(S/'VERIFICATION.json',{'status':'PASS','same_23_metrics_and_four_equal_weights':True,'other_six_tasks_numerically_unchanged':True,'new_task_from_complete_all31_evaluation':True,'n_rows':966,'all31_sensitivity_rows':1302,'source_hashes_unchanged':all(sha(p)==h for p,h in hashes.items())})
(S/'README.md').write_text('# Figure 2 current 23-metric summary\n\nOnly lung_story was replaced using the new all-method comparison. Existing 23 metrics, four equal category weights and seven equal task weights are unchanged. Other six tasks are numerically identical to the previous figure. Complete 31 metrics and both metric-equal and category-equal sensitivity summaries are retained.\n\n'+markdown(newmean[['method','mean_rank','overall_position']])+'\n')
print(newmean.to_string(index=False));print(primary[primary.benchmark.eq('lung_story')].sort_values('mean_rank').to_string(index=False))
