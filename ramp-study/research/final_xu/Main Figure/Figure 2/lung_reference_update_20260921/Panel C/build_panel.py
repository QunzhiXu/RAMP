from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
from matplotlib.patches import Rectangle
def draw(fig,box):
    title(fig,box,'C','Performance profile')
    d=selected_values()
    quality=np.where(d.direction.eq('higher'),d['mean'],-d['mean'])
    d['oriented_score']=quality
    groups=d.groupby(['benchmark','metric']).oriented_score
    best,worst=groups.transform('max'),groups.transform('min')
    d['relative_performance']=1-np.divide(best-quality,best-worst,out=np.zeros(len(d)),where=(best-worst).ne(0))
    cats=['biology','mixing','transfer','visual']
    d['primary_weight']=1/(7*4*d.groupby(['benchmark','family','method']).metric.transform('size'))
    assert d.relative_performance.between(0,1).all()
    assert np.allclose(d.groupby('method').primary_weight.sum(),1)
    bycase=d.groupby(['benchmark','family','method']).relative_performance.mean().reset_index()
    f=bycase.groupby(['family','method']).relative_performance.mean().reset_index()
    save_data('C',f);save_data('C',bycase,'benchmark_category_values.csv');save_data('C',d,'all_metric_values.csv')
    mat=f.pivot(index='method',columns='family',values='relative_performance').reindex(index=METHODS,columns=cats)
    ax=mm_axes(fig,box,(13,10,29,30))
    im=ax.imshow(mat,cmap='YlGnBu',vmin=0,vmax=1,aspect='auto')
    ax.set_yticks(range(6),[DISPLAY[m] for m in METHODS],fontsize=6.1)
    ax.get_yticklabels()[-1].set_weight('bold');ax.get_yticklabels()[-1].set_color('#008866')
    ax.set_xticks(range(4),['Cell type','Mixing','Transfer','UMAP'],rotation=58,ha='right',rotation_mode='anchor',fontsize=6.0)
    ax.tick_params(length=0,pad=2)
    ax.set_xticks(np.arange(-.5,4),minor=True);ax.set_yticks(np.arange(-.5,6),minor=True)
    ax.grid(which='minor',color='white',lw=.8);ax.tick_params(which='minor',length=0)
    for sp in ax.spines.values():sp.set_visible(False)
    ax.add_patch(Rectangle((-.5,4.5),4,1,fill=False,lw=.8,color='#008866',clip_on=False))
    ca=mm_axes(fig,box,(15,51,25,1.7))
    cb=fig.colorbar(im,cax=ca,orientation='horizontal',ticks=[0,.5,1]);cb.outline.set_visible(False)
    ca.tick_params(length=1.5,pad=1,labelsize=5.8)
    mm_text(fig,box,27.5,7.6,'Relative performance ↑',ha='center',fontsize=6.3)
    return {'n_metrics':23,'n_benchmarks':7,'categories':cats,'metric_definition':'1 - (best - oriented score)/(best-worst), then metrics within category and seven benchmarks equally averaged; an all-equal comparison receives 1 for every method.',
        'scale':'0=worst, 1=best within each observed six-method comparison. Category mean, not ordinal rank.',
        'selection':str(SELECTION/'metric_selection.json'),
        'aggregation_note':'Category profile retains the prior normalized-score definition; Panel D separately reports the primary mean-rank aggregation. No ranking of these heatmap cells enters D.'}
if __name__=='__main__':run_panel('C',draw)
