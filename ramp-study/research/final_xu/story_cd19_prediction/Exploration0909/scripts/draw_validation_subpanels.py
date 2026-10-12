"""Separate measured/predicted CD3 maps and a compact all-marker strip."""
from runtime import BASE,save,sha
import argparse,json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,Normalize

OUT=BASE/'validation_subpanels'
for folder in ['measured_CD3','predicted_CD3','all_markers','preview']:
    (OUT/folder).mkdir(parents=True,exist_ok=True)
SOURCE=BASE/'CD3_NN_UMAP_validation/cell_predictions_and_umap.csv.gz'
CELLS=pd.read_csv(SOURCE)
XY=CELLS[['UMAP1','UMAP2']].to_numpy()
assert len(CELLS)==30000 and CELLS.dataset.eq('BM CODEX').all()
prior=json.loads((BASE/'CD3_NN_UMAP_validation/PLOT_AUDIT.json').read_text())
NORM=Normalize(prior['vmin'],prior['vmax'],clip=True)
EXPRESSION_CMAP=LinearSegmentedColormap.from_list('CD3_values',
    ['#DCE1E6','#F5E7B2','#EDB34E','#CF572D','#861E41'])
ACCURACY_CMAP=LinearSegmentedColormap.from_list('accuracy',
    ['#BA6035','#F5F7F9','#245B8A'])
ACCURACY_NORM=Normalize(-1,1)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none',
    'pdf.fonttype':42,'text.color':'#29343C','xtick.color':'#53636D','ytick.color':'#53636D'})
LO=XY.min(axis=0); HI=XY.max(axis=0); PAD=.035*(HI-LO)

def export(fig,folder,name):
    for ext in ['png','svg']:
        fig.savefig(OUT/folder/f'{name}.{ext}',dpi=300,facecolor='white')
    plt.close(fig)

def expression(ax,column,annotate=True):
    values=CELLS[column].to_numpy(); order=np.argsort(values,kind='stable')
    sc=ax.scatter(XY[order,0],XY[order,1],c=values[order],s=1.05,cmap=EXPRESSION_CMAP,
        norm=NORM,alpha=1,lw=0,rasterized=True)
    ax.set(xlim=(LO[0]-PAD[0],HI[0]+PAD[0]),ylim=(LO[1]-PAD[1],HI[1]+PAD[1]))
    ax.set_aspect('equal'); ax.axis('off')
    if annotate:
        # These labels point to the median coordinates of existing annotations.
        # They do not change cell labels, points or expression values.
        for celltype,offset in [('CD8 T',(-37,12)),('CD4 T',(13,14))]:
            center=np.median(XY[CELLS.celltype.eq(celltype)],axis=0)
            ax.annotate(celltype,xy=center,xytext=offset,textcoords='offset points',
                fontsize=8,color='#3A4852',ha='left',va='center',
                arrowprops={'arrowstyle':'-','lw':.7,'color':'#77848B',
                            'shrinkA':3,'shrinkB':5})
    return sc

def add_expression_bar(fig,sc,position,label_above=False):
    cb=fig.colorbar(sc,cax=fig.add_axes(position))
    if label_above: cb.ax.set_title('CD3',fontsize=9,pad=6)
    else: cb.set_label('CD3',fontsize=9,labelpad=4)
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=8,length=2,width=.6)

def umaps():
    for title,column,folder in [('Measured CD3','measured_CD3','measured_CD3'),
                                ('Predicted CD3','predicted_CD3','predicted_CD3')]:
        fig=plt.figure(figsize=(4.0,4.05))
        ax=fig.add_axes([.025,.075,.765,.795]); sc=expression(ax,column)
        fig.text(.06,.97,title,ha='left',va='top',fontsize=14,fontweight='bold')
        add_expression_bar(fig,sc,[.84,.245,.025,.42])
        fig.text(.06,.018,'CODEX · 12 donors · Joint UMAP',fontsize=8.5,color='#64717A')
        export(fig,folder,folder)
    save(OUT/'UMAP_subpanels_audit.json',{'source_sha256':sha(SOURCE),'n_cells':len(CELLS),
        'n_donors':CELLS.donor.nunique(),'coordinate_refit':False,'prediction_changed':False,
        'shared_color_limits':[prior['vmin'],prior['vmax']],'value_scale':'KMEANSLOW',
        'all_cells_retained':True,'label_method':'median coordinates of frozen CD4 T/CD8 T annotations',
        'expression_order':'ascending score, identical rule for both maps'})

def load_summary():
    p=OUT/'all_marker_validation/all_14_markers_summary.csv'
    summary=pd.read_csv(p)
    assert len(summary)==14 and summary.n_donors.eq(12).all()
    assert summary.median_donor_spearman.between(-1,1).all()
    return summary

def strip(ax,summary,fontsize=8.3):
    values=summary.median_donor_spearman.to_numpy()
    im=ax.imshow(values[None,:],cmap=ACCURACY_CMAP,norm=ACCURACY_NORM,aspect='auto',interpolation='nearest')
    ax.set_xticks(np.arange(14),summary.marker.to_list(),fontsize=fontsize)
    ax.set_yticks([]); ax.tick_params(axis='x',length=0,pad=5)
    ax.set_xticks(np.arange(-.5,14,1),minor=True)
    ax.grid(which='minor',axis='x',color='white',lw=1.4)
    ax.tick_params(which='minor',length=0)
    ax.spines[:].set_visible(False)
    return im

def accuracy_bar(fig,im,position):
    cb=fig.colorbar(im,cax=fig.add_axes(position),orientation='horizontal',ticks=[-1,0,1])
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7.5,length=2,width=.6)
    cb.ax.set_title('Spearman ρ',fontsize=8,pad=3)

def all_markers():
    summary=load_summary()
    fig=plt.figure(figsize=(8.0,1.42))
    fig.text(.035,.93,'Prediction accuracy across all markers',va='top',fontsize=11.5,fontweight='bold')
    ax=fig.add_axes([.035,.36,.925,.23]); im=strip(ax,summary)
    accuracy_bar(fig,im,[.755,.80,.20,.055])
    fig.text(.035,.065,'Median within-donor Spearman · 14 markers · 12 CODEX donors',fontsize=8.5,color='#64717A')
    export(fig,'all_markers','all_marker_accuracy_strip')
    save(OUT/'all_markers/heatmap_audit.json',{'source_sha256':sha(OUT/'all_marker_validation/all_14_markers_summary.csv'),
        'n_markers':14,'n_donors_per_marker':12,'statistic':'median within-donor Spearman',
        'marker_order':summary.marker.to_list(),'fixed_color_limits':[-1,1],
        'individual_numbers_printed':False,'all_markers_included':True})

def preview():
    summary=load_summary()
    fig=plt.figure(figsize=(8.0,5.35))
    fig.text(.035,.975,'Held-out marker validation',va='top',fontsize=14,fontweight='bold')
    for left,title,column in [(.025,'Measured CD3','measured_CD3'),(.505,'Predicted CD3','predicted_CD3')]:
        ax=fig.add_axes([left,.32,.41,.56]); sc=expression(ax,column)
        fig.text(left+.205,.888,title,ha='center',fontsize=11.5,fontweight='bold')
    add_expression_bar(fig,sc,[.937,.44,.013,.29],label_above=True)
    fig.text(.04,.238,'All-marker prediction accuracy',fontsize=10.5,fontweight='bold')
    im=strip(fig.add_axes([.04,.126,.88,.052]),summary,fontsize=7.5)
    accuracy_bar(fig,im,[.747,.232,.17,.018])
    fig.text(.04,.025,'30,000 CODEX cells · 12 donors · Permanent marker holdout',fontsize=8.5,color='#64717A')
    export(fig,'preview','validation_subpanels_preview')

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['umap','all'])
    args=parser.parse_args(); umaps()
    if args.mode=='all': all_markers(); preview()
