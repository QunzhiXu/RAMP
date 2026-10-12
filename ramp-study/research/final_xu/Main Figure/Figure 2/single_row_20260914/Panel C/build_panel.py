from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
from matplotlib.patches import Rectangle
def draw(fig,box):
    title(fig,box,'C','Metric profile')
    d=csv(PRIOR/'Panel C/plotted_values.csv');save_data('C',d)
    cats=['biology','mixing','transfer','matching','visual']
    mat=d.pivot(index='method_family',columns='metric_family',values='relative_performance').reindex(index=METHODS,columns=cats)
    assert mat.shape==(6,5) and mat.notna().all().all()
    ax=mm_axes(fig,box,(10.6,11,22.5,35))
    im=ax.imshow(mat,cmap='YlGnBu',vmin=0,vmax=1,aspect='auto')
    ax.set_yticks(range(6),[DISPLAY[m] for m in METHODS],fontsize=5.6)
    ax.get_yticklabels()[-1].set_weight('bold');ax.get_yticklabels()[-1].set_color('#008866')
    ax.set_xticks(range(5),['Cell type','Mixing','Transfer','Matching','UMAP'],rotation=66,ha='right',rotation_mode='anchor',fontsize=5.8)
    ax.tick_params(length=0,pad=1.5)
    ax.set_xticks(np.arange(-.5,5),minor=True);ax.set_yticks(np.arange(-.5,6),minor=True)
    ax.grid(which='minor',color='white',lw=.7);ax.tick_params(which='minor',length=0)
    for sp in ax.spines.values():sp.set_visible(False)
    ax.add_patch(Rectangle((-.5,4.5),5,1,fill=False,lw=.8,color='#008866',clip_on=False))
    ca=mm_axes(fig,box,(12,58.6,19.5,1.4))
    cb=fig.colorbar(im,cax=ca,orientation='horizontal',ticks=[0,.5,1]);cb.outline.set_visible(False)
    ca.tick_params(length=1.5,pad=1,labelsize=5.6)
    mm_text(fig,box,21.8,8.1,'Relative performance ↑',ha='center',fontsize=5.6)
    return {'source':'Exact matrix from prior two-row Panel C, unchanged.','n_metrics':30,'n_categories':5,
        'scale':'Direction-adjusted min-max relative performance averaged per metric category then equally across four benchmarks.',
        'geometry_exclusion':'Existing post-inspection scope change retained; full original results remain available.'}
if __name__=='__main__':run_panel('C',draw)
