from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
from matplotlib.patches import Rectangle
def draw(fig,box):
    title(fig,box,'C','Performance profile')
    d=csv(FIG2/'curve_review_20260914/source_values_and_weights.csv')
    assert len(d)==720 and d.metric.nunique()==30
    d['relative_performance']=1-d.range_normalized_gap
    cats=['biology','mixing','transfer','matching','visual']
    bycase=d.groupby(['benchmark','metric_family','method_family']).relative_performance.mean().reset_index()
    f=bycase.groupby(['metric_family','method_family']).relative_performance.mean().reset_index()
    save_data('C',f);save_data('C',bycase,'benchmark_category_values.csv');save_data('C',d,'all_metric_values.csv')
    mat=f.pivot(index='method_family',columns='metric_family',values='relative_performance').reindex(index=METHODS,columns=cats)
    ax=mm_axes(fig,box,(13,10,29,30))
    im=ax.imshow(mat,cmap='YlGnBu',vmin=0,vmax=1,aspect='auto')
    ax.set_yticks(range(6),[DISPLAY[m] for m in METHODS],fontsize=6.1)
    ax.get_yticklabels()[-1].set_weight('bold');ax.get_yticklabels()[-1].set_color('#008866')
    ax.set_xticks(range(5),['Cell type','Mixing','Transfer','Matching','UMAP'],rotation=58,ha='right',rotation_mode='anchor',fontsize=6.0)
    ax.tick_params(length=0,pad=2)
    ax.set_xticks(np.arange(-.5,5),minor=True);ax.set_yticks(np.arange(-.5,6),minor=True)
    ax.grid(which='minor',color='white',lw=.8);ax.tick_params(which='minor',length=0)
    for sp in ax.spines.values():sp.set_visible(False)
    ax.add_patch(Rectangle((-.5,4.5),5,1,fill=False,lw=.8,color='#008866',clip_on=False))
    ca=mm_axes(fig,box,(15,51,25,1.7))
    cb=fig.colorbar(im,cax=ca,orientation='horizontal',ticks=[0,.5,1]);cb.outline.set_visible(False)
    ca.tick_params(length=1.5,pad=1,labelsize=5.8)
    mm_text(fig,box,27.5,7.6,'Relative performance ↑',ha='center',fontsize=6.3)
    return {'n_metrics':30,'categories':cats,'metric_definition':'1 - (best - oriented score)/(best-worst), then metrics within category and benchmarks equally averaged.',
        'scale':'0=worst, 1=best within each observed six-method comparison. Category mean, not ordinal rank.',
        'geometry_exclusion':'Inherited user-requested post-inspection exclusion; original 31-metric results preserved in earlier curve folder.'}
if __name__=='__main__':run_panel('C',draw)
