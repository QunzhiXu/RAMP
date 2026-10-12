from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
from matplotlib.ticker import FixedLocator,FixedFormatter,NullLocator
def draw(fig,box):
    title(fig,box,'G','Expression agreement')
    d=csv(FIG2/'value_evaluation_20260914/summary/emd_marker_means.csv')
    means=csv(FIG2/'value_evaluation_20260914/summary/emd_overall_equal_case.csv').set_index('method').emd_normalized
    assert len(d)==225 and d.emd_normalized.gt(0).all()
    save_data('G',d);save_data('G',means.reset_index(),'equal_benchmark_means.csv')
    ax=mm_axes(fig,box,(10,11,34,33))
    boxpoints(ax,d,VALUE_METHODS,'emd_normalized',means)
    ax.set_yscale('log');ax.set_ylim(.06,5)
    ticks=[.1,.2,.5,1,2,5]
    ax.yaxis.set_major_locator(FixedLocator(ticks));ax.yaxis.set_major_formatter(FixedFormatter(['0.1','0.2','0.5','1','2','5']))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylabel('Normalized EMD (log scale)',labelpad=2)
    mm_text(fig,box,25.5,8.3,'Lower = closer distributions',ha='center',fontsize=6.4)
    mm_text(fig,box,25.5,58.8,'Same cell type · all shared markers',ha='center',fontsize=6.0)
    return {'metric':'Exact 1D Wasserstein-1 distance divided by one global method/benchmark/marker 95th-5th percentile span.',
        'points_per_method':45,'each_point':'One benchmark/shared-marker, averaging same-cell-type cross-platform distances under the frozen hierarchy.',
        'raw_points_by_benchmark':d[d.method.eq('Raw')].groupby('benchmark').size().to_dict(),
        'black_diamond':'Four benchmark means equally weighted; not the unweighted mean of 45 markers.',
        'box':'Median, IQR, 1.5 IQR whiskers in original normalized EMD units; all 45 points shown on a logarithmic axis.',
        'no_clipping':bool(d.emd_normalized.between(.06,5).all()),
        'interpretation':'Measures same-type distribution proximity across platforms. Lower is closer; not ground-truth concentration reconstruction error or a measure of gating.',
        'limitations':'May also reflect biological cohort differences; compare alongside gate transfer and retained AUROC. Normalization denominator depends on each method output.',
        'methods':VALUE_METHODS,'embedding_only_methods':'NOT_APPLICABLE'}
if __name__=='__main__':run_panel('G',draw)
