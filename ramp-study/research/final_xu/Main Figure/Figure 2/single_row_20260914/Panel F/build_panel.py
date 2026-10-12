from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
def draw(fig,box):
    title(fig,box,'F','Gate transfer')
    d=csv(PRIOR/'Panel F/plotted_values.csv')
    means=csv(PRIOR/'Panel F/equal_benchmark_means.csv').set_index('method').balanced_accuracy
    assert len(d)==55 and d.balanced_accuracy.between(.45,1.005).all()
    save_data('F',d);save_data('F',means.reset_index(),'equal_benchmark_means.csv')
    ax=mm_axes(fig,box,(6.5,11,19.7,39))
    boxpoints(ax,d,VALUE_METHODS,'balanced_accuracy',means)
    ax.set_ylim(.45,1.005);ax.set_yticks([.5,.6,.7,.8,.9,1])
    ax.axhline(.5,ls='--',lw=.5,color='#96A0A6',zorder=1)
    mm_text(fig,box,15.5,8.1,'Balanced accuracy',ha='center',fontsize=5.9)
    return {'source':'All unchanged values from prior Panel F.','points_per_method':11,'n_benchmarks':4,
        'markers':'CD3/CD8 in all four, CD4 in the three benchmarks with that shared marker.',
        'aggregation':'Each point averages six source/target directions; black diamond averages four benchmark means equally.',
        'methods':VALUE_METHODS,'embedding_only_methods':'Not applicable to marker-value gate transfer.'}
if __name__=='__main__':run_panel('F',draw)
