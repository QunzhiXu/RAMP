"""Clearly labelled synthetic-DAPI previews and unchanged cell-level HLA overlays."""
from pathlib import Path
import os,sys,json,hashlib
BASE=Path(__file__).resolve().parents[1];ROOT=BASE.parents[1]
sys.path.insert(0,str(ROOT/'story_cd19_prediction/Exploration0909/real_spatial0913/deps'))
os.environ['MPLCONFIGDIR']=str(BASE/'cache/matplotlib');sys.dont_write_bytecode=True
import numpy as np,pandas as pd,tifffile
from PIL import Image
from scipy.stats import spearmanr
from scipy import ndimage as ndi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def save_png(a,name):Image.fromarray(np.round(np.clip(a,0,1)*255).astype(np.uint8)).save(BASE/'figures'/name)

def bar(ax,shape,pixel_um,um):
    h,w=shape[:2];length=um/pixel_um;x=w*.93-length;y=h*.94
    ax.plot([x,x+length],[y,y],color='white',lw=3,solid_capstyle='butt')
    ax.text(x+length/2,y-h*.027,f'{um:g} µm',ha='center',va='bottom',color='white',fontsize=10,fontweight='bold')

def export(fig,name):
    fig.savefig(BASE/'figures'/f'{name}.png',dpi=240,facecolor='white')
    fig.savefig(BASE/'figures'/f'{name}.pdf',dpi=300,facecolor='white')
    plt.close(fig)

def main():
    audit=json.loads((BASE/'results/synthesis_audit.json').read_text());crop=audit['detail_crop'];p=audit['pixel_size_um']
    dapipath=BASE/'data/C30_synthetic_DAPI.ome.tif';dapi_hash=sha(dapipath)
    dapi=tifffile.imread(dapipath)
    z=np.clip(dapi.astype(np.float32)/45000,0,1)**.85
    rgb=z[:,:,None]*np.array([.43,.62,1.],np.float32)
    sl=np.s_[crop['y0']:crop['y1'],crop['x0']:crop['x1']]
    save_png(rgb,'C30_synthetic_DAPI_blue_full.png')
    save_png(rgb[sl],'C30_synthetic_DAPI_blue_detail.png')
    fig,axs=plt.subplots(1,2,figsize=(10.2,5.4))
    for ax,img,title,scale in zip(axs,[rgb,rgb[sl]],['C30 · full core','Central detail · 577 cells'],[100,50]):
        ax.imshow(img,interpolation='nearest');ax.axis('off');ax.set_title(title,fontsize=14,fontweight='bold',pad=8);bar(ax,img.shape,p,scale)
    axs[0].add_patch(Rectangle((crop['x0'],crop['y0']),crop['x1']-crop['x0'],crop['y1']-crop['y0'],fill=False,ec='#FFC247',lw=1.6))
    fig.suptitle('Synthetic DAPI · real Zheng cell coordinates',fontsize=16,fontweight='bold',y=.985)
    fig.subplots_adjust(left=.01,right=.99,top=.875,bottom=.02,wspace=.04)
    export(fig,'C30_synthetic_DAPI_preview')
    # Only now load the fixed HLA values. The already-generated DAPI file is not modified.
    lookup=pd.read_csv(BASE/'data/C30_synthetic_cell_lookup.csv.gz')
    values=pd.read_csv(BASE/'data/C30_HLA_values_unchanged.csv.gz')
    full=lookup.merge(values,on='CellID',how='left',validate='one_to_one',sort=False)
    assert len(full)==10719 and full.synthetic_label.tolist()==list(range(1,10720))
    original_path=ROOT/'story_lung_HLA_ABC/validation/results/C30_CELLS.csv.gz'
    original=pd.read_csv(original_path)
    assert np.array_equal(full.CellID.to_numpy(),original.CellID.to_numpy())
    columns=['observed_HLA_A_patient_rank','RAMP_k30']
    for col in columns:np.testing.assert_array_equal(full[col].to_numpy(),original[col].to_numpy())
    rho=float(spearmanr(full[columns[0]],full[columns[1]]).statistic)
    labels=tifffile.imread(BASE/'data/C30_synthetic_cell_labels.tif')
    norm=Normalize(.25,.6,clip=True);cmap=plt.get_cmap('viridis')
    boundary=(ndi.maximum_filter(labels,3)!=ndi.minimum_filter(labels,3))&(labels>0)
    overlays=[]
    for col in columns:
        colors=np.zeros((len(full)+1,3),np.float32);colors[full.synthetic_label]=cmap(norm(full[col].to_numpy()))[:,:3]
        overlay=rgb*.65+colors[labels]*.66
        overlay[boundary]*=.88
        overlays.append(np.clip(overlay,0,1))
    for name,array in zip(['measured_HLA_A','predicted_HLA_ABC'],overlays):save_png(array,f'C30_{name}_synthetic_DAPI_overlay.png')
    for detail,name in [(False,'C30_HLA_overlay_synthetic_DAPI_full'),(True,'C30_HLA_overlay_synthetic_DAPI_detail')]:
        images=[a[sl] for a in overlays] if detail else overlays
        fig=plt.figure(figsize=(10.8,5.55))
        axes=[fig.add_axes([.02,.07,.425,.79]),fig.add_axes([.47,.07,.425,.79])]
        for ax,img,title in zip(axes,images,['Measured HLA-A','Predicted HLA-ABC']):
            ax.imshow(img,interpolation='nearest');ax.axis('off');ax.set_title(title,fontsize=15,fontweight='bold',pad=10)
            bar(ax,img.shape,p,50 if detail else 100)
        cbax=fig.add_axes([.916,.21,.018,.51]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cbax,ticks=[.25,.4,.6])
        cb.ax.set_yticklabels(['≤0.25','0.4','≥0.6']);cb.set_label('Relative rank',fontsize=11);cb.ax.tick_params(labelsize=10)
        fig.suptitle('Cell-level HLA overlay · synthetic DAPI background',fontsize=15,fontweight='bold',y=.98)
        export(fig,name)
    report={'synthetic_background_explicitly_labelled':True,'DAPI_display_max':45000,'DAPI_display_gamma':.85,
        'DAPI_display_RGB':[.43,.62,1.],'HLA_color_map':'viridis','HLA_color_range':[.25,.6],
        'HLA_overlay_alpha_like_weight':.66,'DAPI_weight_in_HLA_overlay':.65,'boundary_darkening_factor':.88,
        'n_cells':len(full),'HLA_values_unchanged':True,'same_DAPI_background_both_panels':True,
        'same_synthetic_cell_shapes_both_panels':True,'cell_spearman_unchanged':rho,
        'source_C30_sha256':sha(original_path),'DAPI_sha256_before':dapi_hash,'DAPI_sha256_after':sha(dapipath),
        'image_similarity_not_used_for_validation':True,'crop':crop,
        'interpretation':'Uniform per-cell color annotations on synthetic cell shapes; not measured or predicted HLA subcellular fluorescence'}
    assert report['DAPI_sha256_before']==report['DAPI_sha256_after']
    (BASE/'results/preview_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
