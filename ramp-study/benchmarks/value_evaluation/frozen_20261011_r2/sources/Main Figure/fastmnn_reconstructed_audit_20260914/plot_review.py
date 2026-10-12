"""Show frozen common-protocol UMAPs, reconstructed ridges, and full diagnostics."""
from pathlib import Path
import os,sys,json
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
os.environ['MPLCONFIGDIR']=str(HERE/'.cache/matplotlib')
os.environ['MPLBACKEND']='Agg'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator,FuncFormatter
from scipy.ndimage import gaussian_filter1d
from scipy.stats import rankdata
import run_audit as a
sys.path.insert(0,str(ROOT/'Main Figure/Panel B/benchmark_review'))
import build_review as oldplot

NAMES=dict(zip(a.CASES,['Bone marrow','TNBC','Melanoma','Lung cancer']))
SELECTED={'bone_marrow':['CD3','CD4','CD8'],'tnbc':['CD3','CD8','PANCK'],
          'melanoma':['CD3','CD8','SOX10'],'lung_cancer':['CD3','VIMENTIN','EPI']}
METHODS=['Raw','fastMNN reconstructed','RAMP']
PAL={'Raw':'#88929A','fastMNN reconstructed':'#6D5CA7','RAMP':'#008866',
     'Harmony-direct':'#858B91','cyCombine':'#C88D3D','CytoNorm':'#479DAD'}
DATASET_COLORS=['#2775B6','#E58A2B','#38A169']
RIDGE_AUDIT=[]
plt.rcParams.update({'font.family':'Arial','font.size':9,'axes.titlesize':10,
    'axes.titleweight':'bold','axes.linewidth':.65,'svg.fonttype':'none','text.color':'#203037',
    'axes.labelcolor':'#203037','xtick.color':'#203037','ytick.color':'#203037'})

def save(fig,stem,svg=True):
    fig.savefig(HERE/(stem+'.png'),dpi=190,facecolor='white')
    if svg:fig.savefig(HERE/(stem+'.svg'),dpi=190,facecolor='white')
    plt.close(fig)
    print(HERE/(stem+'.png'),flush=True)

def clean(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color('#899297')
    ax.tick_params(length=2.5,pad=3)

def ridge(ax,v,meta,case,method,marker):
    # Same accepted 3x bandwidth for all methods (sigma 6 / 160 histogram bins).
    lo,hi=np.quantile(v,[.005,.995]);hi=max(hi,lo+1e-10)
    palette=dict(zip(meta.dataset.unique(),DATASET_COLORS))
    groups=[(ds,u,ix) for (ds,u),ix in meta.groupby(['dataset','unit_key'],sort=False).indices.items()]
    groups.sort(key=lambda x:(list(palette).index(x[0]),x[1]))
    level=0;previous=None
    for ds,u,ix in reversed(groups):
        if previous is not None and ds!=previous:level+=.38
        vals=v[ix];counts,edges=np.histogram(vals,bins=160,range=(lo,hi))
        smooth=gaussian_filter1d(counts.astype(float),sigma=6,mode='constant')
        if smooth.max()>0:smooth/=smooth.max()
        x=(edges[1:]+edges[:-1])/2;y=level+.48*smooth
        ax.fill_between(x,level,y,color=palette[ds],alpha=.34,lw=0)
        ax.plot(x,y,color=palette[ds],lw=.65)
        RIDGE_AUDIT.append(dict(case=case,method=method,marker=marker,unit_key=u,n_cells=len(ix),
            display_low=lo,display_high=hi,n_below=int((vals<lo).sum()),n_above=int((vals>hi).sum()),
            histogram_bins=160,sigma_bins=6,peak_height_normalized=True))
        previous=ds;level+=.56
    ax.set_xlim(lo,hi);ax.set_ylim(-.1,level+.1);ax.set_yticks([])
    ax.spines[['left','right','top']].set_visible(False)
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x/1000:g}k' if abs(x)>=1000 else f'{x:.2g}'))
    ax.tick_params(axis='x',labelsize=8,length=2,pad=2)

def scatter(ax,meta,xy,by):
    palette=dict(zip(meta.dataset.unique(),DATASET_COLORS)) if by=='Dataset' else oldplot.COLORS
    labels=meta.dataset.astype(str) if by=='Dataset' else meta.celltype.fillna('Non-joint').astype(str)
    assert set(labels)<=set(palette)
    colors=np.array([palette[v] for v in labels])
    order=np.random.default_rng(42).permutation(len(meta))
    ax.scatter(xy[order,0],xy[order,1],c=colors[order],s=.45,alpha=.74,lw=0,rasterized=True)
    low,high=xy.min(axis=0),xy.max(axis=0);pad=(high-low)*.025
    ax.set_xlim(low[0]-pad[0],high[0]+pad[0]);ax.set_ylim(low[1]-pad[1],high[1]+pad[1])
    ax.set_aspect('equal',adjustable='box');ax.axis('off')

def load(case):
    meta=a.read_csv(HERE/case/'cell_metadata.csv.gz',dtype={'cell_id':str})
    markers=a.track(HERE/case/'markers.txt').read_text().splitlines()
    base=ROOT/'benchmark_results'/case
    rp=base/'ramp/alignment/data/ramp_aligned_float64.npy'
    if not rp.exists():rp=base/'ramp/alignment/data/ramp_aligned.npy'
    arrays={'Raw':np.load(a.track(base/'ramp/alignment/input/raw.npy')),
            'fastMNN reconstructed':np.load(a.track(HERE/case/'reconstructed.npy')),
            'RAMP':np.load(a.track(rp))}
    coords={}
    for m,stem in [('Raw','Unaligned'),('fastMNN reconstructed','fastMNN'),('RAMP','RAMP')]:
        p=ROOT/f'benchmark_results/comparison_evaluation/{case}/umap/coordinates/{stem}.csv.gz'
        frame=a.read_csv(p).sort_values('row_id')
        assert np.array_equal(frame.row_id.to_numpy(),np.arange(len(meta)))
        if 'cell_id' in frame:assert np.array_equal(frame.cell_id.to_numpy(),meta.cell_id.to_numpy())
        coords[m]=frame[['UMAP1','UMAP2']].to_numpy()
        a.track(p.parents[1]/'manifests'/f'{stem}.json')
    return meta,markers,arrays,coords

def review_sheets(loaded,raw_arcsinh=True):
    for case,(meta,markers,arrays,coords) in loaded.items():
        fig=plt.figure(figsize=(12.2,7.7))
        fig.text(.065,.985,NAMES[case],fontsize=16,weight='bold',va='top')
        fig.text(.99,.983,f'{len(meta):,} cells · {meta.unit_key.nunique()} samples',ha='right',va='top',fontsize=10)
        handles=[Line2D([],[],color=c,lw=3,label=d) for d,c in zip(meta.dataset.unique(),DATASET_COLORS)]
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,.952),frameon=False,ncol=3,fontsize=8.5)
        grid=fig.add_gridspec(3,5,left=.065,right=.993,bottom=.135,top=.865,
            width_ratios=[1,1,1,1.1,1.1],hspace=.25,wspace=.15)
        for i,m in enumerate(METHODS):
            for j,marker in enumerate(SELECTED[case]):
                ax=fig.add_subplot(grid[i,j])
                v=arrays[m][:,markers.index(marker)]
                display_method=m
                if m=='Raw' and raw_arcsinh:
                    v=np.arcsinh(v);display_method='Raw (arcsinh display)'
                ridge(ax,v,meta,case,display_method,marker)
                if i==0:ax.set_title(marker,pad=10)
                if j==0:ax.set_ylabel('fastMNN' if m.startswith('fastMNN') else 'Raw\n(arcsinh)' if m=='Raw' and raw_arcsinh else m,
                    fontsize=10,weight='bold',color=PAL[m],labelpad=11)
            for j,by in enumerate(['Dataset','Cell type'],start=3):
                ax=fig.add_subplot(grid[i,j]);scatter(ax,meta,coords[m],by)
                if i==0:ax.set_title(by,pad=10)
        labels=meta.celltype.astype(str).value_counts().index.tolist()
        handles=[Line2D([],[],marker='o',color=oldplot.COLORS[l],lw=0,ms=4,label=l) for l in labels]
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.52,.021),frameon=False,ncol=5,fontsize=7.5,columnspacing=1,handletextpad=.3)
        fig.text(.52,.007,'fastMNN ridges: reconstructed markers; UMAP: corrected PCs. Raw arcsinh is display-only.' if raw_arcsinh else
            'fastMNN ridges: reconstructed markers; UMAP: corrected PCs. Raw ridges retain native values.',ha='center',fontsize=8)
        save(fig,f'{case}/ridge_umap_review' if raw_arcsinh else f'{case}/ridge_umap_native_raw',svg=raw_arcsinh)
        if not raw_arcsinh:continue
        # Every shared marker, no selection, including markers absent from the main display.
        nr=int(np.ceil(len(markers)/4));fig=plt.figure(figsize=(12,2.8*nr+.8))
        grid=fig.add_gridspec(nr,4,left=.04,right=.99,top=1-.75/fig.get_figheight(),bottom=.04,hspace=.28,wspace=.25)
        for j,marker in enumerate(markers):
            ax=fig.add_subplot(grid[j//4,j%4]);ridge(ax,arrays['fastMNN reconstructed'][:,j],meta,case,'fastMNN reconstructed',marker)
            ax.set_title(marker,pad=6)
        fig.suptitle(NAMES[case]+' | fastMNN reconstructed | all shared markers',fontsize=14,weight='bold',y=.99)
        fig.legend(handles=[Line2D([],[],color=c,lw=3,label=d) for d,c in zip(meta.dataset.unique(),DATASET_COLORS)],
            loc='upper center',bbox_to_anchor=(.5,1-.28/fig.get_figheight()),frameon=False,ncol=3,fontsize=8)
        save(fig,f'{case}/all_markers_ridge',svg=False)

def examples(loaded):
    specifications=[('bone_marrow','BM CyTOF|H2','CD14',['Monocyte','Mature / CODEX B','CD4 T'],['Monocytes','B cells','CD4 T']),
        ('tnbc','Shiao CODEX|h05','CD4',['CD4 T','B lineage','CD8 T'],['CD4 T','B cells','CD8 T']),
        ('melanoma','MEL_Hoch2021_IMC|patient_93','CD68',['Myeloid','CD8 T','Tumor'],['Myeloid','CD8 T','Tumor']),
        ('lung_cancer','Cords|175_191','CD31',['Endothelial','Tumor/Epithelial','Myeloid'],['Endothelial','Tumor','Myeloid'])]
    fig,axes=plt.subplots(2,2,figsize=(11.8,7.2),layout='constrained')
    stats=[]
    colors=['#2775B6','#D9799D','#BBC2C6']
    for ax,(case,unit,marker,cells,labels) in zip(axes.ravel(),specifications):
        meta,markers,arrays,_=loaded[case];rows=np.flatnonzero(meta.unit_key.eq(unit));j=markers.index(marker)
        values={'Raw':arrays['Raw'],'Cosine':np.load(a.track(HERE/case/'cosine_normalized.npy')),
                'PCA only':np.load(a.track(HERE/case/'pca_reconstructed.npy')),
                'fastMNN':arrays['fastMNN reconstructed'],'RAMP':arrays['RAMP']}
        local_labels=meta.iloc[rows].celltype.to_numpy()
        for i,(stage,matrix) in enumerate(values.items()):
            rr=a.ranks(matrix[rows,j])
            vv=[rr[local_labels==c] for c in cells]
            b=ax.boxplot(vv,positions=i+np.array([-.26,0,.26]),widths=.22,showfliers=False,patch_artist=True,manage_ticks=False,
                medianprops={'color':'#2C363B','lw':.8},boxprops={'lw':.65},whiskerprops={'lw':.65},capprops={'lw':.65})
            for box,color in zip(b['boxes'],colors):box.set_facecolor(color)
            for c,v in zip(cells,vv):stats.append(dict(case=case,unit_key=unit,marker=marker,celltype=c,stage=stage,n=len(v),median_percentile=float(np.median(v))))
            au=a.auc(matrix[rows,j][local_labels==cells[0]],matrix[rows,j][local_labels==cells[1]])
            ax.text(i,104,f'{au:.2f}',ha='center',fontsize=9,color='#5B4A94' if stage=='fastMNN' else '#344047')
        ax.set_xticks(range(5),['Raw','Cosine','PCA only','fastMNN','RAMP'])
        ax.set_ylim(0,110);ax.set_yticks([0,50,100]);ax.set_ylabel(marker+' percentile');clean(ax)
        ax.set_title(NAMES[case]+' | '+marker,loc='left',pad=34)
        ax.legend(handles=[Line2D([],[],marker='s',ls='',color=c,label=l,ms=6) for c,l in zip(colors,labels)],
                  frameon=False,ncol=3,loc='lower left',bbox_to_anchor=(0,1.05),fontsize=8,handletextpad=.3,columnspacing=.8)
    fig.suptitle('Marker reconstruction diagnostics\nNumbers: AUROC of blue versus pink cells; boxes: all cells in the selected sample',fontsize=12,weight='bold')
    save(fig,'marker_examples_four_benchmarks')
    pd.DataFrame(stats).to_csv(HERE/'example_plot_statistics.csv',index=False)
    # Check the two already published examples rather than replacing them silently.
    old=a.read_csv(ROOT/'Main Figure/redesign_full_icon_20260914/artifact_cells.csv.gz')
    fig,axes=plt.subplots(1,2,figsize=(12,3.7),layout='constrained')
    for ax,case,unit,marker,cells,labels in [
        (axes[0],'tnbc','Meyer IMC|B12.64192','CD8',['CD8 T','CD4 T','Tumor / epithelial'],['CD8 T','CD4 T','Tumor']),
        (axes[1],'bone_marrow','BM CyTOF|H1','CD4',['CD4 T','Mature / CODEX B','CD8 T'],['CD4 T','B cells','CD8 T'])]:
        meta,markers,arrays,_=loaded[case];rows=np.flatnonzero(meta.unit_key.eq(unit));j=markers.index(marker)
        for i,m in enumerate(['Raw','Harmony-direct','cyCombine','CytoNorm','fastMNN reconstructed','RAMP']):
            if m.startswith('fastMNN'):
                pr=a.ranks(arrays[m][rows,j]);ll=meta.iloc[rows].celltype.to_numpy()
                vv=[pr[ll==c] for c in cells]
            else:
                sub=old[old.benchmark.eq(case)&old.unit_key.eq(unit)&old.marker.eq(marker)&old.method.eq(m)]
                vv=[100*sub[sub.celltype.eq(c)].percentile.to_numpy() for c in cells]
            b=ax.boxplot(vv,positions=i+np.array([-.26,0,.26]),widths=.22,showfliers=False,patch_artist=True,manage_ticks=False)
            for box,color in zip(b['boxes'],colors):box.set_facecolor(color)
        ax.set_ylim(0,101);ax.set_yticks([0,50,100]);ax.set_ylabel(marker+' percentile');clean(ax)
        ax.set_xticks(range(6),['Raw','Harmony','cyCombine','CytoNorm','fastMNN','RAMP'],fontsize=8)
        ax.set_title(NAMES[case]+' | existing '+marker+' example',loc='left',pad=28)
        ax.legend(handles=[Line2D([],[],marker='s',ls='',color=c,label=l,ms=6) for c,l in zip(colors,labels)],frameon=False,
                  ncol=3,loc='lower left',bbox_to_anchor=(0,1.02),fontsize=8)
    save(fig,'existing_artifact_examples_with_fastmnn')

def metrics():
    base=HERE/'value_evaluation/summary'
    d=a.read_csv(base/'gate_task_means.csv');emd=a.read_csv(base/'emd_case_means.csv')
    overall=a.read_csv(base/'RESULTS_OVERVIEW.csv').set_index('method')
    shift=a.read_csv(HERE/'comparison_celltype_shifts.csv.gz')
    order=['Raw','Harmony-direct','cyCombine','CytoNorm','fastMNN reconstructed','RAMP']
    fig,axs=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    for ax,table,value,title,means in [(axs[0],d,'balanced_accuracy','Fixed-gate balanced accuracy',overall.balanced_accuracy),
            (axs[1],emd,'emd_normalized','Same-cell-type normalized EMD',overall.emd_normalized)]:
        for i,m in enumerate(order):
            v=table[table.method.eq(m)][value].to_numpy()
            b=ax.boxplot(v,positions=[i],widths=.48,showfliers=False,patch_artist=True,manage_ticks=False)
            b['boxes'][0].set_facecolor(PAL[m]);b['boxes'][0].set_alpha(.25)
            ax.scatter(i+np.linspace(-.12,.12,len(v)),v,s=17,color=PAL[m],zorder=3)
            ax.scatter(i,means[m],marker='D',s=23,c='#172B32',zorder=4)
        ax.set_xticks(range(6),['Raw','Harmony','cyCombine','CytoNorm','fastMNN','RAMP'],rotation=43,ha='right',fontsize=8)
        ax.set_title(title,fontsize=10,loc='left');clean(ax)
    axs[0].set_ylim(.4,1.01);axs[0].axhline(.5,ls='--',lw=.6,color='#929B9F')
    names=['Harmony-direct[source_scale]','Harmony-direct[fixed_arcsinh]','cyCombine','CytoNorm','fastMNN reconstructed','RAMP']
    for i,m in enumerate(names):
        v=shift[shift.method.eq(m)].abs_delta_median_pp.to_numpy()
        b=axs[2].boxplot(v,positions=[i],vert=False,widths=.48,showfliers=False,patch_artist=True,manage_ticks=False)
        col='#6D5CA7' if m.startswith('fastMNN') else '#008866' if m=='RAMP' else '#B9C1C5'
        b['boxes'][0].set_facecolor(col)
        yy=i+np.random.default_rng(42).uniform(-.16,.16,len(v))
        axs[2].scatter(v,yy,s=.5,alpha=.10,color=col,rasterized=True)
    axs[2].set_yticks(range(6),['Harmony','Harmony-arcsinh','cyCombine','CytoNorm','fastMNN','RAMP'],fontsize=8)
    axs[2].invert_yaxis();axs[2].set_xlim(0,90);axs[2].set_xlabel('Absolute median shift (pp)')
    axs[2].set_title('Cell-type percentile shifts',fontsize=10,loc='left');clean(axs[2])
    fig.suptitle('Four-benchmark diagnostic extension | fastMNN reconstructed',fontsize=13,weight='bold')
    save(fig,'value_metrics_with_fastmnn')

def integration_inventory():
    p=ROOT/'Main Figure/Figure 2/metric_selection_main23/metric_ranks_7_benchmarks_main23.csv'
    d=a.read_csv(p)
    fast=d[d.method.eq('fastMNN')]
    assert len(fast)==7*23
    fast.to_csv(HERE/'existing_fastmnn_integration_metrics.csv',index=False)
    selected=d[d.benchmark.isin(a.CASES)&d.method.isin(['fastMNN','RAMP'])]
    selected.to_csv(HERE/'existing_fastmnn_ramp_integration_comparison.csv',index=False)

def main():
    a.track(__file__)
    loaded={case:load(case) for case in a.CASES}
    review_sheets(loaded,False);review_sheets(loaded,True);examples(loaded);metrics();integration_inventory()
    pd.DataFrame(RIDGE_AUDIT).to_csv(HERE/'ridge_display_audit.csv',index=False)
    assert all(a.sha(p)==h for p,h in a.READS.items())
    (HERE/'plot_manifest.json').write_text(json.dumps(dict(umap_refit=False,alignment_refit=False,
        umap_source='Existing common-protocol comparison_evaluation coordinates: pooled dimension-wise StandardScaler of each frozen representation, spectral initialization, seed 42.',
        umap_fastmnn_representation='Frozen corrected PCs, not a newly refitted standardized reconstructed-marker UMAP.',
        display_subsampling=False,random_draw_order_seed=42,shared_color_palettes=True,
        ridge_smoothing='160 bins, sigma=6 bins for every method; per-ridge peak height normalized',
        ridge_range='0.5%-99.5% of pooled cells, separately per method-marker; no clipping into edge bins; all omitted-tail counts saved',
        raw_display='Main review uses arcsinh(raw) for readable native dynamic range; native-raw review also saved. Metrics always use original unmodified values.',
        examples='New four-panel examples are exploratory, selected from existing canonical-rule scan; original two examples shown separately.',
        inputs_unchanged=True,input_sha256=a.READS),indent=2)+'\n')

if __name__=='__main__':main()
