from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *

def draw(fig,box):
    title(fig,box,'D','Rank distribution')
    source=csv(FIG2/'curve_review_20260914/source_values_and_weights.csv')
    summary=csv(FIG2/'curve_review_20260914/overall_ranking_before_after.csv')
    # Show all metric/benchmark ranks. Preserve the previous weighted summary.
    categories=source.groupby(['benchmark','metric_family','method_family'])['rank'].mean()
    bycase=categories.groupby(['benchmark','method_family']).mean().rename('mean_rank').reset_index()
    bycase=bycase.rename(columns={'method_family':'method'})
    means=bycase.groupby('method')['mean_rank'].mean().sort_values()
    methods=means.index.tolist()
    d=source.rename(columns={'method_family':'method'}).copy()
    expected=summary.set_index('method_family').mean_rank_30_metrics
    assert source.metric.nunique()==30 and len(d)==720
    assert d.groupby('method').size().eq(120).all()
    assert not d.duplicated(['benchmark','metric','method']).any()
    assert np.allclose(means,expected.loc[methods],atol=1e-12)
    save_data('D',d)
    save_data('D',bycase,'benchmark_mean_ranks.csv')
    save_data('D',means.rename('mean_rank').reset_index(),'equal_benchmark_means.csv')
    ax=mm_axes(fig,box,(14,10,30,33))
    arrays=[d.loc[d.method.eq(m),'rank'].to_numpy() for m in methods]
    boxes=ax.boxplot(arrays,positions=np.arange(6),vert=False,widths=.50,
        patch_artist=True,showfliers=False,manage_ticks=False,
        medianprops={'color':'#4E5960','lw':.8},
        whiskerprops={'color':'#7B858A','lw':.55},
        capprops={'color':'#7B858A','lw':.55},
        boxprops={'edgecolor':'#7B858A','lw':.55})
    rng=np.random.default_rng(20260914)
    display=[]
    for i,(m,patch) in enumerate(zip(methods,boxes['boxes'])):
        patch.set_facecolor('#C5E0D7' if m=='RAMP' else '#EDF0F1')
        for j,c in enumerate(CASES):
            subset=d.loc[d.method.eq(m)&d.benchmark.eq(c)].sort_values('metric')
            assert len(subset)==30
            xx=subset['rank'].to_numpy()+rng.uniform(-.105,.105,len(subset))
            yy=i+[-.255,-.085,.085,.255][j]+rng.uniform(-.072,.072,len(subset))
            ax.scatter(xx,yy,s=3.3,
                marker=CASE_SHAPES[c],c=CASE_COLORS[c],edgecolors='white',
                linewidths=.08,zorder=4,alpha=.68)
            display.extend(subset.assign(display_x=xx,display_y=yy).to_dict('records'))
    save_data('D',pd.DataFrame(display),'display_point_coordinates.csv')
    ax.scatter(means.to_numpy(),np.arange(6),marker='D',s=15,color='#182C33',
        edgecolors='white',linewidths=.45,zorder=6)
    ax.set_yticks(np.arange(6),[DISPLAY[m] for m in methods],fontsize=6.6)
    ax.get_yticklabels()[0].set_color('#008866');ax.get_yticklabels()[0].set_weight('bold')
    ax.set_ylim(5.6,-.6);ax.set_xlim(.8,6.15);ax.set_xticks([1,2,3,4,5,6])
    ax.set_xlabel('Rank ↓',fontsize=7,labelpad=2)
    ax.spines[['top','right','left']].set_visible(False)
    ax.spines['bottom'].set_color('#748087')
    ax.tick_params(axis='y',length=0,pad=3)
    ax.tick_params(axis='x',length=2,pad=2,labelsize=6.3)
    ax.grid(axis='x',color='#E5E9EB',lw=.4);ax.set_axisbelow(True)
    mm_text(fig,box,27.5,7.5,'30 metrics · 4 benchmarks',fontsize=6.3,ha='center')
    return {'n_metrics':30,'n_categories':5,'n_benchmarks':4,'box_n':120,
        'aggregation':'Average ranks within benchmark/category; average five categories equally within each benchmark; black diamond averages four benchmarks equally.',
        'box':'Unweighted median, IQR and 1.5 IQR whiskers of 120 metric-by-benchmark ranks per method, without pre-averaging; all 120 points overlaid.',
        'points':'Every metric/benchmark combination, colored and shaped by benchmark; 720 points across six methods, none excluded.',
        'jitter':'Small display-only jitter with seed 20260914; true ranks retained in plotted_values.csv and box statistics.',
        'rank_values':means.rename('mean_rank').reset_index().to_dict('records'),
        'orientation':'Horizontal, methods sorted by overall mean rank, lower is better.',
        'p_values':'None; 120 metric/benchmark summaries are correlated descriptive results, not 120 independent biological replicates.',
        'scope':'Current 30-metric scope retained; original 31-metric and old curve results preserved.'}

if __name__=='__main__':run_panel('D',draw)
