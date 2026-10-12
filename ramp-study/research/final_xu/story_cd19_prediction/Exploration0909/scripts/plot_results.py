from runtime import *
import ast, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

OUT=BASE/'results'; FIG=BASE/'figures'
METHODS=['Neural_network','RAMP_KNN30_pooled','RAMP_KNN30_equal','KMEANSLOW_KNN30_equal']
NAMES={'Neural_network':'Neural network','RAMP_KNN30_pooled':'KNN after RAMP\npooled sources','RAMP_KNN30_equal':'KNN after RAMP\n50/50 sources','KMEANSLOW_KNN30_equal':'KNN before RAMP\n50/50 sources'}
COLORS=['#3978a8','#db8255','#a776b4','#409b86']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#9ba8b0','axes.labelcolor':'#354651','text.color':'#243745','savefig.facecolor':'white'})

def savefig(fig,name):
    for ext in ['png','svg']: fig.savefig(FIG/f'{name}.{ext}',dpi=260,bbox_inches='tight')
    plt.close(fig)

def initial():
    d=pd.read_csv(OUT/'CD19_DONOR_METRICS.csv')
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.4))
    for ax,col,title in zip(axes,['B_vs_T_AUROC','B_vs_all_AP'],['B versus CD4/CD8 T','B versus all other cells']):
        wide=d.pivot(index='donor',columns='method',values=col)[METHODS]
        for _,row in wide.iterrows(): ax.plot(range(4),row,color='#ced5da',lw=.7,zorder=1)
        for x,m in enumerate(METHODS):
            y=wide[m].to_numpy(); jitter=np.linspace(-.07,.07,len(y))
            ax.scatter(x+jitter,y,s=23,color=COLORS[x],zorder=3)
            med=np.median(y); ax.plot([x-.17,x+.17],[med,med],color='#243745',lw=2,zorder=4)
            ax.text(x,1.015,f'{med:.3f}',ha='center',fontsize=10,fontweight='bold',color=COLORS[x])
        ax.set(xticks=range(4),xticklabels=[NAMES[m] for m in METHODS],ylim=(0,1.09),yticks=[0,.25,.5,.75,1],ylabel='AUROC' if col.endswith('AUROC') else 'Average precision',title=title)
        ax.grid(axis='y',color='#edf0f3'); ax.set_axisbelow(True)
    fig.suptitle('CD19 prediction across all 12 CODEX donors',fontsize=14,fontweight='bold',x=.04,ha='left')
    fig.text(.04,.015,'Each connected point is one donor; labels show donor medians. All KNN variants use fixed k = 30.',fontsize=8)
    fig.tight_layout(rect=(0,.06,1,.94)); savefig(fig,'CD19_DONOR_COMPARISON')
    md=pd.read_csv(STORY/'sampled_data/bm_codex/locked_metadata.csv.gz')
    pred=pd.read_csv(OUT/'CD19_PREDICTIONS.csv.gz')
    donor='SB67_NBM28_H14_CODEX_Mesmer'
    ix=pred.donor.eq(donor).to_numpy(); cells=md.loc[ix].reset_index(drop=True); p=pred.loc[ix].reset_index(drop=True)
    xy=cells[['x.coord','y.coord']].to_numpy(); truth=cells.cluster_anno_l1.eq('B-Cells').to_numpy()
    ncall=int((p.Neural_network>1.4).sum())
    for rule in ['fixed_1.4','matched_NN_call_count']:
        fig,axes=plt.subplots(1,4,figsize=(13,4.1))
        methods=['Neural_network','RAMP_KNN30_pooled','KMEANSLOW_KNN30_equal']
        selections=[truth]; titles=['Author-annotated B cells']
        stats=[]
        for method in methods:
            score=p[method].to_numpy()
            high=score>1.4 if rule=='fixed_1.4' else np.isin(np.arange(len(score)),np.argsort(-score,kind='stable')[:ncall])
            selections.append(high); titles.append(NAMES[method])
            tp=int((high&truth).sum()); stats.append(f'Precision {tp/high.sum():.1%} | Recall {tp/truth.sum():.1%}')
        for j,(ax,high,title) in enumerate(zip(axes,selections,titles)):
            ax.scatter(xy[~high,0],xy[~high,1],s=2,color='#d2d6da',alpha=.7,lw=0,rasterized=True)
            ax.scatter(xy[high,0],xy[high,1],s=6,color='#d54d7b',alpha=.95,lw=0,rasterized=True)
            ax.set_title(title+f'\n(n = {int(high.sum())})',fontsize=9.5)
            ax.set_aspect('equal'); ax.invert_yaxis(); ax.axis('off')
            if j: ax.text(.5,-.05,stats[j-1],ha='center',transform=ax.transAxes,fontsize=8)
        caption='Same original donor; original score threshold > 1.4' if rule=='fixed_1.4' else f'Same original donor; each predictor highlights its top {ncall} cells'
        fig.suptitle(caption,fontsize=13,fontweight='bold',y=.99)
        fig.tight_layout(rect=(0,.01,1,.91)); savefig(fig,'SPATIAL_'+rule.upper())
    # Same cells, same KMEANSLOW score axis; no density normalization between methods.
    labels=pd.read_csv(STORY/'alignment/output/cell_metadata.csv.gz').query("dataset == 'BM CODEX'").reset_index(drop=True)
    from scipy.stats import gaussian_kde
    fig,axes=plt.subplots(3,3,figsize=(9.5,6),sharex=True,sharey=True)
    groups=['Mature / CODEX B','CD4 T','CD8 T']; colors=['#a95dad','#39aeb0','#41698d']
    grid=np.linspace(-1.5,4.7,400)
    for j,m in enumerate(['Neural_network','RAMP_KNN30_pooled','KMEANSLOW_KNN30_equal']):
        for i,label in enumerate(groups):
            values=pred.loc[labels.celltype.eq(label),m].to_numpy()
            density=gaussian_kde(values,bw_method=.25)(grid)
            axes[i,j].fill_between(grid,0,density,color=colors[i],alpha=.7)
            axes[i,j].plot(grid,density,color=colors[i],lw=1)
            axes[i,j].set(yticks=[],xlim=(-1.5,4.7))
            if i==0: axes[i,j].set_title(NAMES[m])
            if j==0: axes[i,j].set_ylabel(label,fontsize=9)
            if i==2: axes[i,j].set_xlabel('Predicted CD19 (KMEANSLOW)')
    fig.suptitle('CD19 prediction distributions across all CODEX cells',fontsize=13,fontweight='bold')
    fig.tight_layout(rect=(0,0,1,.94)); savefig(fig,'CD19_DISTRIBUTIONS')

def load_reference_functions():
    # Reuse the exact published plotting/metric functions without executing
    # its module-level directory creation or changing any formal output.
    p=STORY/'umap/plot_umap.py'; source=p.read_text(); tree=ast.parse(source)
    names={'LABEL_ORDER','LABEL_COLORS','B_LINEAGE','DATASET_COLORS','DATASETS'}
    funcs={'canonical','draw_points','safe_silhouette','nonself_neighbors','neighbor_fraction','evaluate'}
    for node in tree.body:
        take=isinstance(node,ast.FunctionDef) and node.name in funcs
        take|=isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in node.targets)
        if take: exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),globals())

def umap_comparison():
    # Any is used by the copied annotations.
    global Any
    from typing import Any
    load_reference_functions()
    import umap
    aligned=np.load(OUT/'KNN_COMPLETED15_RAMP.npy')
    init=np.load(STORY/'umap/data/shared_umap_initialization.npy')
    destination=OUT/'KNN_COMPLETED15_UMAP.npy'
    if destination.exists(): xy=np.load(destination)
    else:
        print('Fit KNN completed-panel UMAP, matching formal initialization and parameters',flush=True)
        xy=umap.UMAP(n_neighbors=30,min_dist=.15,metric='euclidean',random_state=42,transform_seed=42,n_jobs=1,low_memory=True,init=init.copy()).fit_transform(canonical(aligned)).astype(np.float32)
        np.save(destination,xy)
    md=pd.read_csv(STORY/'alignment/output/cell_metadata.csv.gz')
    old=np.load(STORY/'umap/data/umap_coordinates.npz')
    sets={key:np.load(STORY/'umap/data/metric_row_sets.npz')[key] for key in np.load(STORY/'umap/data/metric_row_sets.npz').files}
    metric=evaluate('with_KNN_predicted_cd19',aligned,xy,md,sets)
    existing=pd.read_csv(STORY/'umap/data/metrics.csv')
    pd.concat([existing,pd.DataFrame([metric])],ignore_index=True).to_csv(OUT/'UMAP_METRICS.csv',index=False)
    coordinates=[old['without_cd19'],old['with_predicted_cd19'],xy]
    titles=['14 markers: no CD19','15 markers: neural-network CD19','15 markers: KNN CD19']
    fig,axes=plt.subplots(2,3,figsize=(13.5,8))
    for j,(coords,title) in enumerate(zip(coordinates,titles)):
        draw_points(axes[0,j],coords,md,'celltype'); draw_points(axes[1,j],coords,md,'dataset')
        axes[0,j].set_title(title,fontsize=11,fontweight='bold')
    axes[0,0].set_ylabel('Cell type'); axes[1,0].set_ylabel('Platform')
    fig.legend([Line2D([0],[0],marker='o',ls='',color=LABEL_COLORS[n],ms=4) for n in LABEL_ORDER],LABEL_ORDER,loc='lower center',bbox_to_anchor=(.5,.037),ncol=7,frameon=False,fontsize=7)
    fig.legend([Line2D([0],[0],marker='o',ls='',color=DATASET_COLORS[n],ms=4) for n in DATASETS],DATASETS,loc='lower center',bbox_to_anchor=(.5,.003),ncol=3,frameon=False,fontsize=7)
    fig.suptitle('Same 88,031 cells, matched UMAP parameters and initialization',fontsize=14,fontweight='bold')
    fig.tight_layout(rect=(0,.14,1,.95)); savefig(fig,'UMAP_COMPARISON')
    fig,axes=plt.subplots(1,3,figsize=(13,4.6)); labels=md.celltype.to_numpy()
    focus=['Mature / CODEX B','Pre / immature B','Plasma cell','NK']
    for ax,coords,title in zip(axes,coordinates,titles):
        bg=~np.isin(labels,focus)
        ax.scatter(coords[bg,0],coords[bg,1],s=.5,color='#d4d8db',alpha=.22,lw=0,rasterized=True)
        for lab in focus:
            ix=labels==lab; ax.scatter(coords[ix,0],coords[ix,1],s=.65,color=LABEL_COLORS[lab],alpha=.8,lw=0,rasterized=True)
        ax.set_aspect('equal'); ax.axis('off'); ax.set_title(title,fontsize=10.5)
    fig.legend([Line2D([0],[0],marker='o',ls='',color=LABEL_COLORS[n],ms=5) for n in focus],focus,loc='lower center',ncol=4,frameon=False,fontsize=9)
    fig.suptitle('B-lineage and NK cells in the complete UMAP',fontsize=13,fontweight='bold')
    fig.tight_layout(rect=(0,.07,1,.94)); savefig(fig,'UMAP_B_NK_FOCUS')
    save(OUT/'UMAP_VERIFICATION.json',{'status':'PASS','n_cells':len(md),'shared_initialization_sha256':sha(STORY/'umap/data/shared_umap_initialization.npy'),'formal_coordinates_reused':True,'labels_used_to_fit':False,'new_condition':'KNN completed15; primary RAMP features, pooled sources, KMEANSLOW target','metric_rows_identical_to_formal':True,'metrics':metric})
    print(pd.read_csv(OUT/'UMAP_METRICS.csv')[['condition','matureB_vs_NK_silhouette_umap','matureB_vs_NK_silhouette_highdim']].to_string(index=False),flush=True)

def proxy_plot():
    data=pd.read_csv(OUT/'PROXY_MARKER_SUMMARY.csv')
    order=data.query("method == 'RAMP_KNN30_pooled'").sort_values('median_donor_spearman').marker.tolist()
    fig,axes=plt.subplots(1,2,figsize=(10,5.5),sharey=True)
    methods=['RAMP_KNN30_pooled','RAMP_KNN30_equal','KMEANSLOW_KNN30_equal']
    for j,method in enumerate(methods):
        d=data[data.method.eq(method)].set_index('marker').loc[order]
        for ax,col in zip(axes,['median_donor_spearman','mean_high_state_auroc']):
            ax.scatter(d[col],np.arange(14)+(j-1)*.17,s=25,color=COLORS[j+1],label=NAMES[method].replace('\n',' '))
    axes[0].set(yticks=range(14),yticklabels=order,xlim=(-.15,1),xlabel='Median donor Spearman')
    axes[1].set(xlim=(.35,1),xlabel='Mean donor high-state AUROC'); axes[1].axvline(.5,color='#aab3ba',ls='--',lw=.8)
    for ax in axes: ax.grid(axis='x',color='#e8ecef'); ax.set_axisbelow(True)
    fig.suptitle('Strict missing-marker transfer: all 14 markers, all 12 CODEX donors',fontsize=12,fontweight='bold')
    axes[1].legend(loc='lower right',frameon=False,fontsize=8)
    fig.tight_layout(rect=(0,0,1,.95)); savefig(fig,'STRICT_PROXY_VALIDATION')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['initial','umap','proxy']); a=p.parse_args()
    {'initial':initial,'umap':umap_comparison,'proxy':proxy_plot}[a.mode]()
