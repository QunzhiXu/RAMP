from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
def draw(fig,box):
    title(fig,box,'D','Mean rank')
    d=csv(FIG2/'curve_review_20260914/overall_ranking_before_after.csv')
    d=d.sort_values('mean_rank_30_metrics').reset_index(drop=True)
    source=csv(FIG2/'curve_review_20260914/source_values_and_weights.csv')
    check=source.groupby(['benchmark','metric_family','method_family'])['rank'].mean().groupby('method_family').mean()
    assert np.allclose(d.mean_rank_30_metrics,check.loc[d.method_family])
    save_data('D',d)
    ax=mm_axes(fig,box,(1,11,27,39))
    for i,row in d.iterrows():
        m=row.method_family;v=row.mean_rank_30_metrics
        if m=='RAMP':ax.axhspan(i-.48,i+.49,color='#E9F4EE',zorder=0)
        ax.barh(i+.12,v,height=.29,color='#008866' if m=='RAMP' else '#A5B2B8',edgecolor='none')
        ax.text(.08,i-.20,f'{int(row.position_30_metrics)}  {DISPLAY[m]}',ha='left',va='bottom',fontsize=6.0,
            color='#008866' if m=='RAMP' else INK,weight='bold' if m=='RAMP' else 'normal')
        ax.text(6.85,i+.12,f'{v:.2f}',ha='right',va='center',fontsize=5.8,
            color='#008866' if m=='RAMP' else INK,weight='bold' if m=='RAMP' else 'normal')
    ax.set_ylim(5.62,-.62);ax.set_xlim(0,7.1);ax.set_yticks([]);ax.set_xticks([0,2,4,6])
    ax.set_xlabel('Mean rank ↓',fontsize=6.2,labelpad=2)
    ax.spines[['left','top','right']].set_visible(False);ax.spines['bottom'].set_color('#748087')
    ax.tick_params(axis='x',labelsize=5.9,length=2,pad=2)
    mm_text(fig,box,15,8.1,'30 metrics · 5 categories',ha='center',fontsize=5.8)
    return {'display':'Horizontal mean-rank bars, sorted from best to worst; lower is better.',
        'n_metrics':30,'n_categories':5,'n_benchmarks':4,
        'weights':'Original metric ranks averaged within each benchmark/category, then 20 benchmark/category units equally averaged.',
        'values':d[['method_family','mean_rank_30_metrics','position_30_metrics']].to_dict('records'),
        'not_restoring_31_metric_scope':True,'old_curve_and_full_31_metric_results_preserved':True}
if __name__=='__main__':run_panel('D',draw)
