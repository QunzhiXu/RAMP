"""Stage Figure 2, captions and all dependent benchmark tables after new results pass QA."""
from pathlib import Path
import json,hashlib,shutil,re,difflib
import pandas as pd
import numpy as np
B=Path(__file__).resolve().parent;F=B.parent;ROOT=B.parents[2];C=ROOT/'story_lung_HLA_ABC/Comparison_senosain140k_20260921';LIVE=ROOT.parent/'bioRxiv_Ver1';M=B/'manuscript_stage';S=B/'selection'
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return pd.read_csv(p,float_precision='round_trip')
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
assert json.loads((S/'VERIFICATION.json').read_text())['status']=='PASS'
assert (B/'Combined/figure_2_lung_reference_update.pdf').exists()
assert not M.exists();M.mkdir()
for p in LIVE.iterdir():
 if p.is_file() and p.suffix in ['.tex','.bib','.bst','.cls','.sty','.bbl','.md']:shutil.copy2(p,M/p.name)
shutil.copytree(LIVE/'figure',M/'figure');shutil.copytree(LIVE/'supplementary_data',M/'supplementary_data')
save(B/'LIVE_BEFORE.json',{str(p.relative_to(LIVE)):sha(p) for p in LIVE.rglob('*') if p.is_file() and '_backups' not in p.relative_to(LIVE).parts})
shutil.copy2(B/'Combined/figure_2_lung_reference_update.pdf',M/'figure/figure2_benchmark.pdf')
all31=read(S/'metric_ranks_7_benchmarks_all31.csv');main=read(S/'metric_ranks_7_benchmarks_main23.csv');all31.to_csv(M/'supplementary_data/all31_metric_scores_and_ranks.csv',index=False)
summary=[]
for selection,frame in [('Main23',main),('Full31',all31)]:
 for weighting in ['category_equal','metric_equal']:
  if weighting=='category_equal':d=frame.groupby(['benchmark','method','family'])['rank'].mean().groupby(['benchmark','method']).mean()
  else:d=frame.groupby(['benchmark','method'])['rank'].mean()
  d=d.rename('mean_rank').reset_index();d.to_csv(M/f'supplementary_data/{selection}_{weighting}_by_task.csv',index=False)
  v=d.groupby('method').mean_rank.mean().reset_index();v['selection']=selection;v['weighting']=weighting;summary.append(v)
sensitivity=pd.concat(summary,ignore_index=True)[['selection','weighting','method','mean_rank']];sensitivity.to_csv(M/'supplementary_data/rank_sensitivity.csv',index=False)
# Current lung configuration records supersede the old lung records only.
p=M/'supplementary_data/comparison_preprocessing.json';j=json.loads(p.read_text())
for r in j['records']:
 if 'story_lung_HLA_ABC/Comparison/' in r['source']:r['superseded_by']='story_lung_HLA_ABC/Comparison_senosain140k_20260921'
newpaths=[C/'input/manifest.json',C/'run_config.json',C/'maxfuse_full_panel/data/manifest.json',*[C/f'methods/{k}/R_COMPLETE.json' for k in ['harmony','fastmnn','cycombine','cytonorm']]]
for source in newpaths:
 record=json.loads(source.read_text());record=json.loads(json.dumps(record).replace(str(ROOT),'final_xu'))
 j['records'].append({'source':str(source.relative_to(ROOT)),'source_sha256':sha(source),'record':record})
j['version_note']='Lung story uses the Senosain140k reference update, with all comparator models refitted and matched metrics recomputed. Marked old lung records are historical. The other six task configurations are retained; full-active MaxFuse supersedes shared-only MaxFuse configurations.';save(p,j)
p=M/'supplementary_data/README.md';t=p.read_text();t=t.replace('These tables summarize existing, frozen analysis outputs. No alignment was refitted and no cell-level metric was recomputed during manuscript revision.','These tables include the 2026-09-21 Lung story reference update. Its five comparator methods were refitted on the current RAMP application cohort and all six methods were re-evaluated with the same 31-metric protocol. The other six tasks retain their previous values. The 23-metric reporting list and weights are unchanged.');t=t.replace('36 archived preprocessing/configuration records',f"{len(j['records'])} preprocessing/configuration records");p.write_text(t)
# Derive every ranking-dependent claim from the refreshed tables.
over=read(S/'primary_overall_method_ranking.csv').sort_values('mean_rank');ranks=over.set_index('method').mean_rank;positions=read(S/'primary_method_ranking_by_benchmark.csv');rp=positions[positions.method.eq('RAMP')];counts=rp.position.value_counts().sort_index()
words={1:'one',2:'two',3:'three',4:'four',5:'five',6:'six',7:'seven'};ordinals={1:'first',2:'second',3:'third',4:'fourth',5:'fifth',6:'sixth'}
dist=' and '.join(f'{ordinals[int(k)]} in {words[int(v)]}'+(' task' if int(v)==1 else ' tasks') for k,v in counts.items())
chdist='、'.join(f'{int(v)} 组第{int(k)}' for k,v in counts.items())
metrics=main.groupby(['method','metric'])['mean'].mean();a=metrics.xs('louvain_celltype_ari',level='metric');b=metrics.xs('umap_celltype_asw_macro',level='metric')
profile=read(B/'Panel C/plotted_values.csv');winners=profile.loc[profile.groupby('family').relative_performance.idxmax()].set_index('family').method.to_dict()
allwins=sensitivity.loc[sensitivity.groupby(['selection','weighting']).mean_rank.idxmin()].method.eq('RAMP').all()
claims={'mean_rank':ranks.to_dict(),'RAMP_task_positions':counts.to_dict(),'RAMP_rank_distribution':dist,'A_winner':a.idxmax(),'B_winner':b.idxmax(),'profile_winners':winners,'RAMP_first_all_four_sensitivities':bool(allwins)};save(B/'CHECKED_CLAIMS.json',claims)
p=M/'template.tex';old=p.read_text();t=old
# Exact numeric statements; no task is removed, and all sensitivity values are retained.
t=re.sub(r'five tasks\s+and second in\s+(?:the other )?two',dist.replace('first in ', '',1),t)
t=t.replace('first in five of seven',f"first in {words[int(counts.get(1,0))]} of seven")
oldnums={'RAMP':'1.96','MaxFuse':'2.67','fastMNN':'2.90','cyCombine':'3.68','CytoNorm':'4.59','Harmony-direct':'5.21'}
start=t.index('\\subsection{RAMP achieves the best overall rank');end=t.index('\\subsection{Aligned marker values',start);part=t[start:end]
part=part.replace('overall mean rank of 1.96',f"overall mean rank of {ranks['RAMP']:.2f}")
a0='''The resulting overall mean rank was 1.96 for RAMP, followed by MaxFuse (2.67),
fastMNN (2.90), cyCombine (3.68), CytoNorm (4.59) and Harmony-direct (5.21;'''
newnum='The resulting overall mean rank was '+f"{ranks['RAMP']:.2f} for RAMP, compared with "+',\n'.join(f'{m} ({ranks[m]:.2f})' for m in ['MaxFuse','fastMNN','cyCombine','CytoNorm'])+f" and Harmony-direct ({ranks['Harmony-direct']:.2f};"
assert a0 in part;part=part.replace(a0,newnum)
if a.idxmax()!='RAMP' or b.idxmax()!='RAMP':
 a0='''RAMP had the highest seven-task mean for Louvain adjusted Rand index against
harmonized cell types and for macro-averaged cell-type silhouette in UMAP space''';assert a0 in part;part=part.replace(a0,f"RAMP's seven-task mean Louvain adjusted Rand index was {a['RAMP']:.3f},\nand its macro-averaged cell-type silhouette in UMAP space was {b['RAMP']:.3f}")
if set(winners.values())!={'RAMP'}:
 a0='''Across the main 23-metric summary, it also
had the highest mean relative performance in each of four categories:
cell-type structure, platform and donor mixing, cross-platform label transfer,
and UMAP-based separation'''
 replacement='Across the main 23-metric summary, relative performance was assessed in\nfour categories: cell-type structure, platform and donor mixing,\ncross-platform label transfer, and UMAP-based separation'
 if winners=={'biology':'RAMP','mixing':'RAMP','transfer':'fastMNN','visual':'RAMP'}:
  replacement='Across the main 23-metric summary, RAMP had the highest mean relative\nperformance in cell-type structure, platform and donor mixing, and UMAP-based\nseparation; fastMNN had the highest mean relative performance in\ncross-platform label transfer'
 assert a0 in part;part=part.replace(a0,replacement)
if not allwins:
 part=part.replace('''RAMP
retained the best overall rank under both choices, although the ordering of
other methods changed (Methods).''','The alternative rankings are reported in Supplementary\nTable~\\ref{tab:rank-sensitivity} (Methods).')
t=t[:start]+part+t[end:]
assert over.iloc[0].method=='RAMP','Overall winner changed: review the Abstract, heading and Discussion before installing.'
if not allwins:
 t=t.replace('and the complete registry; RAMP remains first in all four summaries.','and the complete registry; the four rankings are reported in full.')
 t=t.replace('''best overall rank under the main category-balanced summary and under
alternative summaries using all evaluated metrics.''','''best overall rank under the main category-balanced summary. Alternative
summaries using all evaluated metrics are reported in Supplementary
Table~\\ref{tab:rank-sensitivity}.''')
# Clarify which current dataset version supplies the integration task and which still supplies trainability.
a0='''refresh applies to the HLA-ABC application; the integration and trainability
benchmarks retain their separately frozen reference datasets.'''
b0='''reference also supplies the lung-story integration benchmark in
Figure~\\ref{fig:benchmark}; the original four benchmarks and cross-platform
learning experiments retain their separately frozen reference datasets.'''
assert a0 in t;t=t.replace(a0,b0)
a0='''sampling is described below. Each comparison method uses the corresponding
frozen RAMP cell cohort.''';b0='''sampling is described below. The lung-story task uses the updated Senosain
reference described in the HLA-ABC application; all five comparator methods
were refitted and all method outputs evaluated with the unchanged protocol.
Each comparison method uses the corresponding frozen RAMP cell cohort.''';assert a0 in t;t=t.replace(a0,b0)
# Supplementary Table S1 entries from the exact refreshed aggregation.
t=t.replace('No alignment or metric was recomputed for this summary.',
 'The lung-story entries use the updated Senosain reference; the other six tasks retain their prior scores.')
for method in ['RAMP','MaxFuse','fastMNN','cyCombine','CytoNorm','Harmony-direct']:
 vals=[sensitivity[(sensitivity.method==method)&(sensitivity.selection==sel)&(sensitivity.weighting==w)].mean_rank.item() for sel,w in [('Main23','category_equal'),('Main23','metric_equal'),('Full31','category_equal'),('Full31','metric_equal')]]
 replacement=method+' & '+' & '.join(f'{v:.3f}' for v in vals)+r' \\'
 pattern=re.compile('^'+re.escape(method)+r' & [0-9.]+ & [0-9.]+ & [0-9.]+ & [0-9.]+ \\\\$',re.M);assert len(pattern.findall(t))==1,method;t=pattern.sub(lambda m:replacement,t)
p.write_text(t);(B/'MANUSCRIPT_DIFF.txt').write_text(''.join(difflib.unified_diff(old.splitlines(True),t.splitlines(True),fromfile='before/template.tex',tofile='after/template.tex')))
# Detailed bilingual caption: retain all panel definitions, refresh only dependent results/version.
caption=(F/'FIGURE_CAPTION.md').read_text();caption=caption.replace('RAMP ranks first in five tasks and second in two.',f'RAMP ranks {dist}.')
caption=caption.replace('RAMP 五组第一、两组第二。',f'RAMP：{chdist}。')
for m,v in oldnums.items():
 # Replace values only within each ranking paragraph, not unrelated metric descriptions.
 caption=caption.replace(f'{m} ({v})',f'{m} ({ranks[m]:.2f})').replace(f'{m} {v}',f'{m} {ranks[m]:.2f}')
caption=caption.replace('lowest aggregate mean rank (1.96)',f"lowest aggregate mean rank ({ranks['RAMP']:.2f})").replace('RAMP 综合平均名次为 1.96',f"RAMP 综合平均名次为 {ranks['RAMP']:.2f}")
caption=caption.replace('These are repeated evaluations of fixed alignments, rather than 20 separately fitted models.','For the lung story, all methods use the refreshed Senosain reference cells from the current HLA-ABC application; comparator models and evaluation UMAPs were refitted using the existing configurations. The other six tasks retain their previous results. These are repeated evaluations of fixed alignments, rather than 20 separately fitted models.')
caption=caption.replace('七组任务可能共享来源队列','Lung story 使用当前 HLA-ABC 故事的新版 Senosain 参考细胞，重跑全部比较方法及评价 UMAP，沿用原参数；其他六组结果不变。七组任务可能共享来源队列')
caption=caption.replace('louvain_ari_fastmnn_20260914/','lung_reference_update_20260921/').replace('(metric_selection_main23/README.md)','(lung_reference_update_20260921/selection/README.md)')
caption=caption.replace('(lung_reference_update_20260921/','(').replace('(value_evaluation_20260914/','(../value_evaluation_20260914/').replace('(../fastmnn_reconstructed_audit_20260914/','(../../fastmnn_reconstructed_audit_20260914/').replace('(fastmnn_gating_metric_review_20260914/','(../fastmnn_gating_metric_review_20260914/')
(B/'FIGURE_CAPTION.md').write_text(caption)
start=t.index('\\paperfig{figure2_benchmark}{%');end=t.index('}{fig:benchmark}',start)
(M/'FIGURE2_CAPTION.md').write_text('# Figure 2 caption\n\nUpdated lung reference, 2026-09-21. Caption is identical to the current TeX.\n\n```latex\n'+t[start:end]+'}{fig:benchmark}\n```\n')
p=M/'README.md';u=p.read_text().replace('| 2 | [Integration and gating benchmark](figure/figure2_benchmark.pdf) | Copied unchanged from Biorxiv |','| 2 | [Integration and gating benchmark](figure/figure2_benchmark.pdf) | Updated Lung story comparisons; `../final_xu/Main Figure/Figure 2/lung_reference_update_20260921/` |');p.write_text(u)
print(json.dumps(claims,indent=2))
