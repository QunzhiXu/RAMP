from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
def draw(fig,box):
    title(fig,box,'F','Cross-platform gating')
    d=csv(FIG2/'value_evaluation_20260914/summary/gate_task_means.csv')
    means=csv(FIG2/'value_evaluation_20260914/summary/gate_overall_equal_case.csv').set_index('method').balanced_accuracy
    assert len(d)==55 and d.n_directions.eq(6).all()
    save_data('F',d);save_data('F',means.reset_index(),'equal_benchmark_means.csv')
    ax=mm_axes(fig,box,(10,14,34,45))
    boxpoints(ax,d,VALUE_METHODS,'balanced_accuracy',means)
    ax.set_ylim(.45,1.005);ax.set_yticks([.5,.6,.7,.8,.9,1])
    ax.axhline(.5,ls='--',lw=.55,color='#96A0A6',zorder=1)
    ax.set_ylabel('Balanced accuracy',labelpad=3)
    mm_text(fig,box,27,10,'CD3 · CD4 · CD8',ha='center',fontsize=6.6)
    mm_text(fig,box,24,73,'11 marker tasks · 4 benchmarks',ha='center',fontsize=6.2)
    return {'metric':'Fixed-source-gate cross-platform balanced accuracy','points_per_method':11,
        'each_point':'A benchmark/marker task, averaged equally over six transfer directions, each direction averaged over eligible target units.',
        'black_diamond':'Four benchmarks weighted equally; not the unweighted mean of 11 tasks.',
        'box':'Median, IQR, 1.5 IQR whiskers of all 11 task values; all values overlaid.',
        'methods':VALUE_METHODS,'embedding_only_methods':'fastMNN/MaxFuse NOT_APPLICABLE, not zero or Raw proxies.',
        'p_values':'None; tasks sharing donors are not treated as independent patient replicates.'}
if __name__=='__main__':run_panel('F',draw)
