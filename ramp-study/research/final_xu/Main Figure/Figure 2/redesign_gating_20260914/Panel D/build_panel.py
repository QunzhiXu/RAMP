from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
from matplotlib.ticker import PercentFormatter
def draw(fig,box):
    title(fig,box,'D','Overall performance')
    d=csv(FIG2/'curve_review_20260914/curve_coordinates.csv')
    d=d[d.curve.eq('normalized_gap')];save_data('D',d)
    ax=mm_axes(fig,box,(10.5,9,32.3,34))
    for m in METHODS:
        g=d[d.method.eq(m)]
        ax.step(g.threshold,g.coverage,where='post',color=METHOD_COLORS[m],
            lw=1.25 if m=='RAMP' else .75,zorder=6 if m=='RAMP' else 3,label=DISPLAY[m])
    ax.set_xlim(0,1);ax.set_ylim(0,1.025);ax.set_xticks([0,.5,1]);ax.set_yticks([0,.25,.5,.75,1])
    ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    ax.set_ylabel('Weighted coverage',labelpad=2);ax.set_xlabel('Gap from best / method range',fontsize=6.0,labelpad=2)
    style_axis(ax)
    ax.legend(loc='center',bbox_to_anchor=(.5,-.30),ncol=2,fontsize=6.0,frameon=False,
        borderpad=0,labelspacing=.3,handlelength=1.4,handletextpad=.4,columnspacing=.7)
    mm_text(fig,box,26,7.6,'30 metrics · 4 benchmarks',ha='center',fontsize=6.3)
    return {'n_metrics':30,'curve':'Exact weighted empirical CDF of normalized performance gap; no smoothing.',
        'weights':'Equal benchmark/category weights across five retained categories; metrics equal within category.',
        'source':'Previously computed 30-metric curve, unchanged. Geometry exclusion retained and disclosed.'}
if __name__=='__main__':run_panel('D',draw)
