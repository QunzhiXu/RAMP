"""Plot saved UMAP coordinates with tumor below, myeloid above fibroblast, endothelial on top."""
import package_runtime as rt
from package_runtime import BASE
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DATASETS=['Senosain CyTOF','Cords IMC','Zheng mIF']
DCOLORS={'Senosain CyTOF':'#cb7641','Cords IMC':'#277b9b','Zheng mIF':'#8859a5'}
TYPE_COLORS={'Tumor / epithelial':'#cf5748','T cell':'#258671','B cell':'#6686cd',
             'Myeloid / APC':'#d0a02e','Fibroblast / stroma':'#c35199','Endothelial':'#46aab6',
             'NK cell':'#8ba249','Benign epithelium':'#e6b1a1','Other / uncertain':'#bcbfc5'}
COMMON=['Tumor / epithelial','T cell','Myeloid / APC','Fibroblast / stroma','Endothelial']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})

def axis_clean(ax,u,title):
    lo=u.min(axis=0); hi=u.max(axis=0); pad=(hi-lo)*.035
    ax.set(xlim=(lo[0]-pad[0],hi[0]+pad[0]),ylim=(lo[1]-pad[1],hi[1]+pad[1]),title=title,xlabel='UMAP 1',ylabel='UMAP 2')
    ax.set_xticks([]);ax.set_yticks([]);ax.set_aspect('equal',adjustable='box')

def scatter_category(ax,u,values,colors,size=1.3,alpha=.6,mask=None,bottom_label=None,top_labels=()):
    ix=np.arange(len(u)) if mask is None else np.flatnonzero(mask)
    ix=np.random.default_rng(42).permutation(ix)
    if bottom_label is not None or top_labels:
        labels=np.asarray(values)[ix]
        bottom=labels==bottom_label if bottom_label is not None else np.zeros(len(ix),dtype=bool)
        assert len(set(top_labels))==len(top_labels) and bottom_label not in top_labels
        middle=~(bottom|np.isin(labels,top_labels))
        masks=[bottom,middle]+[labels==label for label in top_labels]
        layer_names=[bottom_label,'Other cell identities',*top_labels]
        ix=np.concatenate([ix[part] for part in masks])
        layer_index=np.concatenate([np.full(int(part.sum()),i) for i,part in enumerate(masks)])
        assert len(ix)==len(labels) and np.all(np.diff(layer_index)>=0)
        DRAW_ORDER_AUDIT.append({'drawing_order_verified':True,
                                 'layers_bottom_to_top':[{'label':label,'n_cells':int(part.sum())}
                                                        for label,part in zip(layer_names,masks)]})
    color=pd.Series(values).map(colors).to_numpy()[ix]
    ax.scatter(u[ix,0],u[ix,1],c=color,s=size,alpha=alpha,linewidths=0,rasterized=True)

def legend(colors):
    return [Line2D([0],[0],marker='o',linestyle='',color=v,label=k,markersize=6) for k,v in colors.items()]

def figures(md):
    us={m:np.load(OUT/f'{m}_UMAP.npy') for m in ['baseline','RAMP']}
    focus=md.focus_fibroblast.to_numpy(bool)
    fig,axes=plt.subplots(2,3,figsize=(16,10.6),layout='constrained')
    for row,(method,u) in enumerate(us.items()):
        scatter_category(axes[row,0],u,md.dataset,DCOLORS)
        scatter_category(axes[row,1],u,md.broad_type,TYPE_COLORS,bottom_label='Tumor / epithelial',top_labels=('Fibroblast / stroma','Myeloid / APC','Endothelial'))
        axes[row,2].scatter(u[:,0],u[:,1],s=.8,c='#dadde1',alpha=.18,linewidths=0,rasterized=True)
        scatter_category(axes[row,2],u,md.dataset,DCOLORS,size=3.3,alpha=.78,mask=focus)
        for col,title in enumerate(['Dataset','Cell identity','Source gate / annotated fibroblasts']):
            prefix='Before RAMP (KMEANSLOW)' if method=='baseline' else 'After RAMP 2.3.0'
            axis_clean(axes[row,col],u,prefix+'\n'+title)
    axes[0,0].legend(handles=legend(DCOLORS),loc='lower left',frameon=True,fontsize=8)
    axes[0,1].legend(handles=legend(TYPE_COLORS),loc='lower left',frameon=True,fontsize=7,ncol=2)
    fig.suptitle('Stage-matched lung cohorts | 28 donors per dataset, 1,000 cells per donor\nShared 9-marker panel; HLA-A / HLA-ABC excluded',fontsize=16)
    fig.savefig(FIG/'alignment_overview.png',dpi=190)
    plt.close(fig)

    u=us['RAMP']
    fig,axes=plt.subplots(2,3,figsize=(16,10.4),layout='constrained')
    for col,ds in enumerate(DATASETS):
        mask=md.dataset.eq(ds).to_numpy()
        axes[0,col].scatter(u[:,0],u[:,1],s=.6,c='#dadddf',alpha=.16,linewidths=0,rasterized=True)
        scatter_category(axes[0,col],u,md.broad_type,TYPE_COLORS,size=2,alpha=.8,mask=mask,bottom_label='Tumor / epithelial',top_labels=('Fibroblast / stroma','Myeloid / APC','Endothelial'))
        axes[1,col].scatter(u[:,0],u[:,1],s=.6,c='#dadddf',alpha=.16,linewidths=0,rasterized=True)
        scatter_category(axes[1,col],u,md.dataset,DCOLORS,size=4.2,alpha=.83,mask=mask&focus)
        axis_clean(axes[0,col],u,ds+' | cell identities')
        axis_clean(axes[1,col],u,ds+f' | fibroblast focus (n={int((mask&focus).sum()):,})')
    fig.legend(handles=legend(TYPE_COLORS),loc='outside lower center',ncol=5,frameon=False,fontsize=9)
    fig.suptitle('RAMP alignment | same coordinates in every panel\nSenosain: project annotations / paper mesenchymal gate; Cords and Zheng: author annotations',fontsize=14)
    fig.savefig(FIG/'aligned_dataset_and_fibroblast_facets.png',dpi=190);plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(16,5.6),layout='constrained')
    donor_rows=[]
    cmap=plt.get_cmap('turbo',28)
    for ax,ds in zip(axes,DATASETS):
        mask=md.dataset.eq(ds).to_numpy()
        ids=sorted(md.loc[mask,'patient_id'].unique())
        code={pid:i+1 for i,pid in enumerate(ids)}
        donor_rows.extend({'dataset':ds,'donor_color_index':v,'patient_id':k} for k,v in code.items())
        ax.scatter(u[:,0],u[:,1],s=.6,c='#dadde1',alpha=.16,linewidths=0,rasterized=True)
        ix=np.random.default_rng(42).permutation(np.flatnonzero(mask))
        im=ax.scatter(u[ix,0],u[ix,1],c=md.iloc[ix].patient_id.map(code),cmap=cmap,vmin=.5,vmax=28.5,s=2,alpha=.8,linewidths=0,rasterized=True)
        axis_clean(ax,u,ds+' | 28 donors')
    fig.colorbar(im,ax=axes,label='Donor index within each dataset (see DONOR_COLORS.csv)',ticks=[1,7,14,21,28],shrink=.75)
    fig.suptitle('RAMP alignment | donor distribution, identical UMAP coordinates',fontsize=15)
    fig.savefig(FIG/'aligned_donor_facets.png',dpi=190);plt.close(fig)
    pd.DataFrame(donor_rows).to_csv(OUT/'DONOR_COLORS.csv',index=False)

    # HLA target values are deliberately absent from all figures at this stage.
    result=md[['cell_key','dataset','patient_id','roi_id','provider_cell_type','broad_type','focus_fibroblast']].copy()
    for method,u in us.items(): result[[method+'_UMAP1',method+'_UMAP2']]=u
    result.to_csv(OUT/'UMAP_COORDINATES.csv.gz',index=False,compression={'method':'gzip','compresslevel':3,'mtime':0})
    print('Three UMAP figures saved',flush=True)


def render(directory, md):
    global OUT, FIG, DRAW_ORDER_AUDIT
    OUT = directory
    FIG = OUT / 'figures'
    DRAW_ORDER_AUDIT=[]
    FIG.mkdir(parents=True, exist_ok=True)
    figures(md)
    rt.save_json(OUT/'PLOT_MANIFEST.json',{'time_utc':rt.now(),
        'drawing_order':'Tumor / epithelial first, other cell identities next, Fibroblast / stroma next, Myeloid / APC next, Endothelial last; seeded order preserved within all five groups',
        'cell_type_panels':DRAW_ORDER_AUDIT,'n_cell_type_panels':len(DRAW_ORDER_AUDIT),
        'plot_script_sha256':rt.sha(__file__),
        'coordinate_sha256':{m:rt.sha(OUT/f'{m}_UMAP.npy') for m in ['baseline','RAMP']},
        'figure_sha256':{p.name:rt.sha(p) for p in FIG.glob('*.png')}})
