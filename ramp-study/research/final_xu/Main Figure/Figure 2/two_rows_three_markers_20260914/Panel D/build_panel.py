from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *

def draw(fig,box):
    title(fig,box,'D','Mean rank')
    source=csv(FIG2/'curve_review_20260914/source_values_and_weights.csv')
    summary=csv(FIG2/'curve_review_20260914/overall_ranking_before_after.csv')
    d=summary.sort_values('mean_rank_30_metrics').reset_index(drop=True)
    check=source.groupby(['benchmark','metric_family','method_family'])['rank'].mean().groupby('method_family').mean()
    assert source.metric.nunique()==30
    assert np.allclose(d.mean_rank_30_metrics,check.loc[d.method_family],atol=1e-12)
    save_data('D',d)
    ax=mm_axes(fig,box,(14,10,30,33))
    for i,row in d.iterrows():
        color='#008866' if row.method_family=='RAMP' else '#A5B2B8'
        if row.method_family=='RAMP':ax.axhspan(i-.46,i+.46,color='#E9F4EE',zorder=0)
        ax.barh(i,row.mean_rank_30_metrics,height=.56,color=color,lw=0)
        ax.text(row.mean_rank_30_metrics+.12,i,f'{row.mean_rank_30_metrics:.2f}',
            fontsize=6.3,ha='left',va='center',color=color if i==0 else INK,
            weight='bold' if i==0 else 'normal')
    ax.set_yticks(range(len(d)),[DISPLAY[m] for m in d.method_family],fontsize=6.6)
    ax.get_yticklabels()[0].set_color('#008866');ax.get_yticklabels()[0].set_weight('bold')
    ax.set_ylim(5.6,-.6);ax.set_xlim(0,6.6);ax.set_xticks([0,2,4,6])
    ax.set_xlabel('Mean rank ↓',fontsize=7,labelpad=2)
    ax.spines[['top','right','left']].set_visible(False)
    ax.spines['bottom'].set_color('#748087')
    ax.tick_params(axis='y',length=0,pad=3)
    ax.tick_params(axis='x',length=2,pad=2,labelsize=6.3)
    mm_text(fig,box,27.5,7.5,'30 metrics · 4 benchmarks',fontsize=6.3,ha='center')
    return {'n_metrics':30,'n_categories':5,'n_benchmarks':4,
        'aggregation':'Average original metric ranks within each benchmark/category, then average all 20 units equally.',
        'rank_values':d[['method_family','mean_rank_30_metrics']].to_dict('records'),
        'scope':'Current 30-metric scope retained; original 31-metric and old curve results preserved.'}

if __name__=='__main__':run_panel('D',draw)
