"""Threefold ridge bandwidth and TNBC UMAP display refinements; no embedding refit."""
from pathlib import Path
import os
import sys
import json

HERE=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(HERE.parent/'bandwidth_review/.cache/matplotlib'))
sys.path.insert(0,str(HERE.parent/'benchmark_review'))
from build_ridge_review import load_case,load_matrices,combined,sha
from build_review import scatter,legend,METHODS,COLORS,DATASET_COLORS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PALETTE={
    'Tumor / epithelial':'#D56D79',
    'Myeloid':'#DDB130',
    'Stroma':'#9B76B5',
    'Endothelial':'#079EAA',
    'B lineage':'#32906C',
    'NK':'#6154A3',
    'CD4 T':'#2878B5',
    'CD8 T':'#E46A2E',
    'Non-joint':'#BFC3C8',
}
DRAW_ORDER=['Non-joint','Tumor / epithelial','Stroma','Myeloid',
            'Endothelial','B lineage','NK','CD4 T','CD8 T']


def finish_axis(ax,xy):
    lo,hi=xy.min(axis=0),xy.max(axis=0)
    margin=(hi-lo)*.025
    ax.set_xlim(lo[0]-margin[0],hi[0]+margin[0])
    ax.set_ylim(lo[1]-margin[1],hi[1]+margin[1])
    ax.set_aspect('equal',adjustable='box')
    ax.set_axis_off()


def refined_scatter(ax,meta,xy,color_by,layered=True,draw_order=None):
    if color_by=='Dataset':
        palette=dict(zip(meta.dataset.unique(),DATASET_COLORS))
        colors=np.array([palette[d] for d in meta.dataset])
        order=np.random.default_rng(42).permutation(len(meta))
        ax.scatter(xy[order,0],xy[order,1],c=colors[order],s=.28,alpha=.72,
                   edgecolors='none',rasterized=True)
    elif not layered:
        colors=np.array([PALETTE[c] for c in meta.celltype])
        order=np.random.default_rng(42).permutation(len(meta))
        ax.scatter(xy[order,0],xy[order,1],c=colors[order],s=.32,alpha=.8,
                   edgecolors='none',rasterized=True)
    else:
        values=meta.celltype.to_numpy()
        display_order = DRAW_ORDER if draw_order is None else draw_order
        assert len(display_order)==len(PALETTE) and set(display_order)==set(PALETTE)
        for z,label in enumerate(display_order):
            selected=np.flatnonzero(values==label)
            # Keep every cell. Cell-type layering is identical in every method.
            selected=np.random.default_rng(42).permutation(selected)
            size,alpha=(.28,.5) if label=='Tumor / epithelial' else (.34,.85)
            if label=='Non-joint':
                size,alpha=.28,.65
            ax.scatter(xy[selected,0],xy[selected,1],c=PALETTE[label],s=size,alpha=alpha,
                       edgecolors='none',rasterized=True,zorder=z+1)
    finish_axis(ax,xy)


def style_comparison(meta,coords):
    fig=plt.figure(figsize=(12,12),facecolor='white')
    grid=fig.add_gridspec(4,3,left=.06,right=.995,top=.94,bottom=.17,
                          wspace=.025,hspace=.13)
    titles=['Previous colors','New colors | random order','New colors | layered']
    for row,(method,name) in enumerate(METHODS.items()):
        for col in range(3):
            ax=fig.add_subplot(grid[row,col])
            if col==0:
                scatter(ax,meta,coords[method],'Cell type')
            else:
                refined_scatter(ax,meta,coords[method],'Cell type',layered=col==2)
            if row==0:
                ax.set_title(titles[col],fontsize=11,weight='bold',pad=10)
        box=grid[row,0].get_position(fig)
        fig.text(.025,(box.y0+box.y1)/2,name,rotation=90,ha='center',va='center',
                 fontsize=12,weight='bold')
    fig.text(.06,.989,'TNBC | Frozen UMAP, display alternatives',fontsize=17,
             weight='bold',va='top')
    labels=meta.celltype.value_counts().index.tolist()
    for left,palette,title in [(.03,COLORS,'Previous palette'),(.52,PALETTE,'New palette')]:
        fig.text(left+.225,.134,title,ha='center',fontsize=10,weight='bold')
        ax=fig.add_axes([left,.023,.45,.103])
        legend(ax,labels,[palette[v] for v in labels],3,8.3)
    fig.savefig(HERE/'tnbc_umap_style_comparison.png',dpi=185)
    plt.close(fig)


def main():
    sources,audit={},[]
    meta,coords=load_case('tnbc',sources)
    markers,matrices=load_matrices('tnbc',meta,sources)
    assert set(meta.celltype)==set(PALETTE)
    plt.rcParams.update({'font.family':'DejaVu Sans'})
    combined('tnbc',meta,coords,markers,matrices,audit,sigma_bins=6,
             raw_display='arcsinh',compact=True,
             output_path=HERE/'tnbc_ridge_umap_sigma6_compact.png')
    layout=combined('tnbc',meta,coords,markers,matrices,audit,sigma_bins=6,
                    raw_display='arcsinh',compact=True,scatter_func=refined_scatter,
                    cell_palette=PALETTE,
                    output_path=HERE/'tnbc_ridge_umap_sigma6_refined.png')
    style_comparison(meta,coords)
    assert all(sha(Path(p))==v for p,v in sources.items())
    (HERE/'refined_display_manifest.json').write_text(json.dumps({
        'input_sha256':sources,'inputs_unchanged':True,
        'cells':len(meta),'all_cells_retained':True,'original_labels_retained':True,
        'umap_refit':False,'coordinate_transform':None,'axis_limits_changed':False,
        'equal_aspect_preserved':True,'ridge_sigma_bins':6,'ridge_histogram_bins':160,
        'raw_ridge_display':'arcsinh(x/1)','layout':layout,
        'celltype_palette':PALETTE,'celltype_draw_order_bottom_to_top':DRAW_ORDER,
        'celltype_default_size':.34,'celltype_default_alpha':.85,
        'tumor_size':.28,'tumor_alpha':.5,'non_joint_size':.28,'non_joint_alpha':.65,
        'dataset_size':.28,'dataset_alpha':.72,'dataset_palette_changed':False,
        'dataset_random_draw_seed':42,'same_display_rules_for_all_methods':True,
        'random_order_alternative':{'size':.32,'alpha':.8,'seed':42},
    },indent=2)+'\n')


if __name__=='__main__':
    main()
