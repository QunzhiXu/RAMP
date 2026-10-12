import runtime as rt
from runtime import BASE,STORY
import json,time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    md=pd.read_csv(BASE/'inputs/CELL_METADATA.csv.gz',low_memory=False)
    oldmd=pd.read_csv(STORY/'data_umap/data/CELL_METADATA.csv.gz',low_memory=False)
    config=json.loads((STORY/'data_umap/reference/RAMP_UMAP.json').read_text())['parameters']
    coords={}
    for m in ['baseline','RAMP']:
        out=BASE/'alignment'/f'{m}_UMAP.npy'
        if not out.exists():
            start=time.time();x=np.load(BASE/'alignment'/f'{m}_SHARED9.npy');x=StandardScaler().fit_transform(x)
            u=umap.UMAP(**config).fit_transform(x);np.save(out,u);print(m,'UMAP',round(time.time()-start,1),flush=True)
        coords[m]=np.load(out)
    old=np.load(STORY/'data_umap/reference/RAMP_UMAP.npy')
    rows=[('Old RAMP',old,oldmd),('New RAMP',coords['RAMP'],md)]
    colors={'Senosain CyTOF':'#3281BD','Cords IMC':'#D4574F','Zheng mIF':'#47A078'}
    ct_colors={'Tumor / epithelial':'#C85054','Other / mixed':'#BDBDBD','B cell':'#AD8A3E','Myeloid / APC':'#DAA73B','Fibroblast / stroma':'#CD79AC','Endothelial':'#5BAF82','T cell':'#5686C0'}
    # Harmonize only display strings; labels never enter any fitting.
    def types(df):
        v=df.broad_type.fillna('Other / mixed').astype(str)
        return v.where(v.isin(ct_colors),'Other / mixed')
    plt.rcParams.update({'font.family':'Arial','font.size':11,'axes.titlesize':13,'axes.titleweight':'bold'})
    fig,axs=plt.subplots(2,3,figsize=(13,8))
    for ri,(name,u,d) in enumerate(rows):
        for ci,kind in enumerate(['Platform','Cell type','Senosain annotations']):
            ax=axs[ri,ci]
            if ci==0:v=d.dataset;pal=colors
            elif ci==1:v=types(d);pal=ct_colors
            else:
                mask=d.dataset.eq('Senosain CyTOF');ax.scatter(u[~mask,0],u[~mask,1],s=.25,c='#ECECEC',rasterized=True)
                v=types(d).where(mask,'background');pal=ct_colors
            for label,c in pal.items():
                z=np.asarray(v.eq(label));ax.scatter(u[z,0],u[z,1],s=.65,c=c,alpha=.75,label=label,rasterized=True,linewidths=0)
            ax.set_title(f'{name}: {kind}');ax.set_xticks([]);ax.set_yticks([]);ax.set_aspect('equal',adjustable='datalim')
            for s in ax.spines.values():s.set_visible(False)
    handles,labels=axs[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,fontsize=9,markerscale=7,loc='lower left',bbox_to_anchor=(.02,.004),frameon=False,ncol=1)
    handles,labels=axs[0,1].get_legend_handles_labels()
    fig.legend(handles,labels,fontsize=9,markerscale=7,loc='lower right',bbox_to_anchor=(.985,.009),frameon=False,ncol=4)
    fig.tight_layout(pad=.7,rect=(0,.085,1,1));fig.savefig(BASE/'figures/OLD_NEW_UMAP.png',dpi=220);plt.close(fig)
    rt.save(BASE/'alignment/UMAP_COMPLETE.json',dict(parameters=config,labels_display_only=True,old_new_separate_embeddings=True))
if __name__=='__main__':main()
