from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *

def draw(fig,box):
    title(fig,box,'D','Mean rank')
    source=csv(FIG2/'curve_review_20260914/source_values_and_weights.csv')
    summary=csv(FIG2/'curve_review_20260914/overall_ranking_before_after.csv')
    # Each point is a benchmark mean of five equally weighted categories.
    categories=source.groupby(['benchmark','metric_family','method_family'])['rank'].mean()
    d=categories.groupby(['benchmark','method_family']).mean().rename('mean_rank').reset_index()
    d=d.rename(columns={'method_family':'method'})
    means=d.groupby('method')['mean_rank'].mean().sort_values()
    methods=means.index.tolist()
    expected=summary.set_index('method_family').mean_rank_30_metrics
    assert source.metric.nunique()==30 and len(d)==24
    assert d.groupby('method').size().eq(4).all()
    assert np.allclose(means,expected.loc[methods],atol=1e-12)
    save_data('D',d)
    save_data('D',means.rename('mean_rank').reset_index(),'equal_benchmark_means.csv')
    ax=mm_axes(fig,box,(14,10,30,33))
    arrays=[d.loc[d.method.eq(m),'mean_rank'].to_numpy() for m in methods]
    boxes=ax.boxplot(arrays,positions=np.arange(6),vert=False,widths=.50,
        patch_artist=True,showfliers=False,manage_ticks=False,
        medianprops={'color':'#4E5960','lw':.8},
        whiskerprops={'color':'#7B858A','lw':.55},
        capprops={'color':'#7B858A','lw':.55},
        boxprops={'edgecolor':'#7B858A','lw':.55})
    for i,(m,patch) in enumerate(zip(methods,boxes['boxes'])):
        patch.set_facecolor('#C5E0D7' if m=='RAMP' else '#EDF0F1')
        for j,c in enumerate(CASES):
            value=d.loc[d.method.eq(m)&d.benchmark.eq(c),'mean_rank'].item()
            ax.scatter(value,i+[-.165,-.055,.055,.165][j],s=13,
                marker=CASE_SHAPES[c],c=CASE_COLORS[c],edgecolors='white',
                linewidths=.25,zorder=4,alpha=.88)
    ax.scatter(means.to_numpy(),np.arange(6),marker='D',s=15,color='#182C33',
        edgecolors='white',linewidths=.35,zorder=6)
    ax.set_yticks(np.arange(6),[DISPLAY[m] for m in methods],fontsize=6.6)
    ax.get_yticklabels()[0].set_color('#008866');ax.get_yticklabels()[0].set_weight('bold')
    ax.set_ylim(5.6,-.6);ax.set_xlim(.8,6.15);ax.set_xticks([1,2,3,4,5,6])
    ax.set_xlabel('Mean rank ↓',fontsize=7,labelpad=2)
    ax.spines[['top','right','left']].set_visible(False)
    ax.spines['bottom'].set_color('#748087')
    ax.tick_params(axis='y',length=0,pad=3)
    ax.tick_params(axis='x',length=2,pad=2,labelsize=6.3)
    ax.grid(axis='x',color='#E5E9EB',lw=.4);ax.set_axisbelow(True)
    mm_text(fig,box,27.5,7.5,'30 metrics · 4 benchmarks',fontsize=6.3,ha='center')
    return {'n_metrics':30,'n_categories':5,'n_benchmarks':4,'box_n':4,
        'aggregation':'Average ranks within benchmark/category; average five categories equally within each benchmark; black diamond averages four benchmarks equally.',
        'box':'Median, IQR and 1.5 IQR whiskers across four benchmark means; all four means overlaid using benchmark colors and shapes.',
        'rank_values':means.rename('mean_rank').reset_index().to_dict('records'),
        'orientation':'Horizontal, methods sorted by overall mean rank, lower is better.',
        'p_values':'None; four benchmark summaries are descriptive, not patient-level replicates.',
        'scope':'Current 30-metric scope retained; original 31-metric and old curve results preserved.'}

if __name__=='__main__':run_panel('D',draw)
