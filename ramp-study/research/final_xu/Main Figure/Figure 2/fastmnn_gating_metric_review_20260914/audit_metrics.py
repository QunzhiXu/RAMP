"""Audit every existing cell-type/transfer metric on all seven frozen tasks."""
from pathlib import Path
import sys,json,hashlib
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import common as h
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
OUT=HERE/'metric_review';OUT.mkdir(exist_ok=True)
ROOT=h.ROOT
READS={}
def read(p):
    p=p.resolve();READS[str(p)]=h.sha(p)
    return pd.read_csv(p,float_precision='round_trip')

base=read(ROOT/'benchmark_results/comparison_evaluation/summary/primary_method_family_metric_ranks.csv')
base=base.rename(columns={'method_family':'method','metric_family':'family'})
cols=['benchmark','method','metric','family','direction','mean','rank']
frames=[base[cols]]
for path in ['story_lung_HLA_ABC','figure_melanoma','figure_HNSCC']:
    frames.append(read(ROOT/path/'Comparison/maxfuse_full_panel/evaluation/metric_method_ranks_all31.csv')[cols])
all_values=pd.concat(frames,ignore_index=True)
assert len(all_values)==7*6*31
assert not all_values.duplicated(['benchmark','method','metric']).any()
selected=read(h.SELECTION/'metric_ranks_7_benchmarks_main23.csv')
keys=['benchmark','method','metric']
a=all_values[all_values.metric.isin(selected.metric)].sort_values(keys).reset_index(drop=True)
b=selected.sort_values(keys).reset_index(drop=True)
pd.testing.assert_frame_equal(a[cols],b[cols],check_exact=True)
registry=read(ROOT/'benchmark_results/comparison_evaluation/summary/metric_registry.csv')
definitions=registry.set_index('metric').definition.to_dict()
relevant=all_values[all_values.family.isin(['biology','transfer'])|all_values.metric.eq('umap_celltype_asw_macro')].copy()
relevant.to_csv(OUT/'all_celltype_and_transfer_values.csv',index=False)
all_values.to_csv(OUT/'all_31_metric_values.csv',index=False)
means=relevant.groupby(['metric','method'])['mean'].mean().unstack('method')
means.to_csv(OUT/'metric_method_means.csv')
summary=[];paired=[];drivers=[]
for metric,g in relevant.groupby('metric',sort=False):
    wide=g.pivot(index='benchmark',columns='method',values='mean').reindex(h.CASES)
    competitors=[m for m in h.METHODS if m!='RAMP']
    ramp=wide.RAMP;best_other=wide[competitors].max(axis=1)
    method_means=wide.mean(axis=0)
    best_mean_other=method_means.drop('RAMP').idxmax()
    summary.append(dict(metric=metric,definition=definitions[metric],
        included_in_current_23=metric in set(selected.metric),RAMP_mean=method_means.RAMP,
        strongest_competitor=best_mean_other,strongest_competitor_mean=method_means[best_mean_other],
        RAMP_mean_margin=method_means.RAMP-method_means[best_mean_other],
        RAMP_mean_score_rank=method_means.rank(ascending=False).RAMP,
        RAMP_first_place_tasks=int((ramp>=best_other-1e-12).sum()),
        RAMP_mean_task_rank=g[g.method.eq('RAMP')]['rank'].mean(),
        RAMP_wins_vs_fastMNN=int((ramp>wide.fastMNN+1e-12).sum()),
        RAMP_wins_vs_MaxFuse=int((ramp>wide.MaxFuse+1e-12).sum())))
    for task,row in wide.iterrows():
        best=row[competitors].idxmax()
        drivers.append(dict(metric=metric,benchmark=task,RAMP=row.RAMP,
            RAMP_rank=float(g[g.benchmark.eq(task)&g.method.eq('RAMP')]['rank'].iloc[0]),
            strongest_other_method=best,strongest_other_value=row[best],
            RAMP_minus_best_other=row.RAMP-row[best],
            RAMP_minus_fastMNN=row.RAMP-row.fastMNN,RAMP_minus_MaxFuse=row.RAMP-row.MaxFuse))
        for method in competitors:
            paired.append(dict(metric=metric,benchmark=task,method=method,
                RAMP_value=row.RAMP,comparison_value=row[method],delta=row.RAMP-row[method]))
summary=pd.DataFrame(summary);drivers=pd.DataFrame(drivers);paired=pd.DataFrame(paired)
summary.to_csv(OUT/'metric_summary.csv',index=False)
drivers.to_csv(OUT/'all_metric_task_drivers.csv',index=False)
ab=['leiden_celltype_ari','umap_celltype_asw_macro']
drivers[drivers.metric.isin(ab)].to_csv(OUT/'current_A_B_task_drivers.csv',index=False)
paired[paired.metric.isin(ab)].to_csv(OUT/'paired_A_B_values.csv',index=False)

h.setup()
fig,axes=plt.subplots(1,2,figsize=(170/25.4,76/25.4))
fig.subplots_adjust(left=.085,right=.99,top=.86,bottom=.29,wspace=.36)
competitors=[m for m in h.METHODS if m!='RAMP']
for ax,metric,title,label in zip(axes,ab,['Cell-type agreement','Cell-type separation'],
                               ['ARI difference (RAMP - method)','UMAP silhouette difference (RAMP - method)']):
    d=paired[paired.metric.eq(metric)]
    arrays=[d[d.method.eq(m)].delta.to_numpy() for m in competitors]
    bp=ax.boxplot(arrays,positions=range(5),widths=.5,patch_artist=True,showfliers=False,
        medianprops={'color':'#4E5960','lw':.8},whiskerprops={'color':'#7B858A','lw':.55},
        capprops={'color':'#7B858A','lw':.55},boxprops={'edgecolor':'#7B858A','lw':.55})
    for patch in bp['boxes']:patch.set_facecolor('#EDF0F1')
    for i,method in enumerate(competitors):
        for j,task in enumerate(h.CASES):
            row=d[d.method.eq(method)&d.benchmark.eq(task)].iloc[0]
            ax.scatter(i+np.linspace(-.2,.2,7)[j],row.delta,s=18,marker=h.CASE_SHAPES[task],
                color=h.CASE_COLORS[task],edgecolor='white',linewidth=.3,zorder=4)
        ax.scatter(i,np.mean(arrays[i]),marker='D',s=18,c='#182C33',edgecolor='white',linewidth=.35,zorder=5)
    ax.axhline(0,color='#4D5B63',lw=.75,ls='--',zorder=2)
    ax.set_xticks(range(5),[h.DISPLAY[m] for m in competitors],rotation=28,ha='right')
    ax.set_ylabel(label,fontsize=7.1)
    ax.set_title(title,fontsize=9,weight='bold',loc='left',pad=8)
    lo,hi=d.delta.min(),d.delta.max();span=hi-lo
    ax.set_ylim(min(lo-.12*span,-.025),hi+.12*span)
    ax.set_xlim(-.55,4.45);h.style_axis(ax)
handles=[Line2D([],[],ls='none',marker=h.CASE_SHAPES[c],color=h.CASE_COLORS[c],ms=4,label=h.CASE_NAMES[c]) for c in h.CASES]
handles.append(Line2D([],[],ls='none',marker='D',color='#182C33',ms=3.5,label='Mean'))
fig.legend(handles=handles,ncol=4,loc='lower center',bbox_to_anchor=(.5,.00),frameon=False,fontsize=7,
           handlelength=.8,handletextpad=.4,columnspacing=1.2)
for ext in ['png','svg']:fig.savefig(OUT/f'paired_A_B_preview.{ext}',dpi=360)
plt.close(fig)

assert all(h.sha(p)==v for p,v in READS.items())
(OUT/'manifest.json').write_text(json.dumps(dict(
    metric_count=11,tasks=7,methods=6,source_records=1302,
    matches_current_23=True,all_metrics_and_tasks_preserved=True,
    metrics_or_alignments_recomputed=False,paired_delta_definition='RAMP minus comparator within each task; all seven tasks shown; positive favors RAMP.',
    any_relevant_metric_RAMP_first_all_seven=bool(summary.RAMP_first_place_tasks.eq(7).any()),
    input_sha256=READS),indent=2)+'\n')
print(summary[['metric','RAMP_mean','strongest_competitor','strongest_competitor_mean','RAMP_first_place_tasks','RAMP_wins_vs_fastMNN','RAMP_wins_vs_MaxFuse']].to_string(index=False))
print('\nA/B task drivers\n'+drivers[drivers.metric.isin(ab)].to_string(index=False))
