"""Show the approved C30 local HLA pair on the frozen synthetic DAPI background."""
from runtime import BASE, STORY
from pathlib import Path
import sys, hashlib, json
sys.path.insert(0,str(STORY.parent/'story_cd19_prediction/Exploration0909/real_spatial0913/deps'))
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

SOURCE = STORY/'synthetic_DAPI0913'
SELECTED = SOURCE/'roi_threshold_search0050913'
NEW = STORY/'releases/senosain140k_20260921/analysis/C30_SPATIAL.csv.gz'

def sha(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def save_json(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')

def main():
    files=[SELECTED/'results/search_summary.json',SELECTED/'results/render_verification.json',
           SELECTED/'data/selected_ROI_cells.csv.gz',SOURCE/'data/C30_synthetic_cell_lookup.csv.gz',
           SOURCE/'data/C30_HLA_values_unchanged.csv.gz',SOURCE/'data/C30_synthetic_DAPI.ome.tif',
           SOURCE/'data/C30_synthetic_cell_labels.tif',SOURCE/'results/synthesis_audit.json',NEW]
    hashes={str(p):sha(p) for p in files}
    search=json.loads(files[0].read_text());prior=json.loads(files[1].read_text())
    for p,h in search['source_sha256'].items():assert sha(p)==h,p
    chosen=search['best'];assert chosen['roi_id']=='S192_X416_Y624'
    assert [chosen['lower'],chosen['upper']]==[.25,.55]
    assert chosen['n_cells']==389 and chosen['side_um']==192
    geom=pd.read_csv(SOURCE/'data/C30_synthetic_cell_lookup.csv.gz')
    values=pd.read_csv(SOURCE/'data/C30_HLA_values_unchanged.csv.gz')
    assert values.CellID.equals(geom.CellID)
    new=pd.read_csv(NEW).set_index('CellID')
    assert new.index.is_unique and len(new)==len(values)==10719
    np.testing.assert_allclose(values.observed_HLA_A_patient_rank, new.loc[values.CellID,'measured_relative_score'],rtol=0,atol=1e-14)
    values['RAMP_k30']=new.loc[values.CellID,'new_prediction'].to_numpy()
    full_rho=float(spearmanr(values.observed_HLA_A_patient_rank,values.RAMP_k30).statistic)
    assert np.array_equal(geom.synthetic_label.to_numpy(),np.arange(1,len(geom)+1))
    syn=json.loads((SOURCE/'results/synthesis_audit.json').read_text())
    umpp=syn['pixel_size_um'];assert umpp==.5
    dapi=tifffile.imread(SOURCE/'data/C30_synthetic_DAPI.ome.tif')
    cellmap=tifffile.imread(SOURCE/'data/C30_synthetic_cell_labels.tif')
    blue=(np.clip(dapi.astype(np.float32)/45000,0,1)**.85)[:,:,None]*np.array([.43,.62,1.],np.float32)
    boundary=(ndi.maximum_filter(cellmap,3)!=ndi.minimum_filter(cellmap,3))&(cellmap>0)
    x0,y0,x1,y1=[int(round(chosen[k]/umpp)) for k in ['x0_um','y0_um','x1_um','y1_um']]
    sl=np.s_[y0:y1,x0:x1];lab=cellmap[sl]
    norm=Normalize(chosen['lower'],chosen['upper'],clip=True)
    cols=['observed_HLA_A_patient_rank','RAMP_k30']
    pair=[]
    for col in cols:
        c=np.zeros((len(values)+1,3),np.float32)
        c[1:]=plt.get_cmap('viridis')(norm(values[col]))[:,:3]
        img=blue[sl]*.65+c[lab]*.66
        img[boundary[sl]]*=.88
        pair.append(np.clip(img,0,1))
    # The region, lookup, normalization, and blending are exactly the approved renderer's.
    xy=geom[['render_x_px','render_y_px']].to_numpy()*umpp
    local=(xy[:,0]>=chosen['x0_um'])&(xy[:,0]<chosen['x1_um'])&(xy[:,1]>=chosen['y0_um'])&(xy[:,1]<chosen['y1_um'])
    selected=geom.loc[local,['synthetic_label','CellID','x_um','y_um']].merge(values,on='CellID',validate='one_to_one')
    frozen=pd.read_csv(SELECTED/'data/selected_ROI_cells.csv.gz')
    pd.testing.assert_frame_equal(selected.drop(columns='RAMP_k30').reset_index(drop=True),frozen.drop(columns='RAMP_k30'),check_exact=False,rtol=1e-14,atol=1e-12)
    assert len(selected)==389
    rho=float(spearmanr(selected[cols[0]],selected[cols[1]]).statistic)
    assert np.isfinite(rho)
    selected.to_csv(BASE/'data/PLOT_DATA.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    save_json(BASE/'provenance/SELECTED_ROI.json',chosen)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'svg.fonttype':'none',
                         'savefig.facecolor':'white','text.color':'#243745','axes.labelcolor':'#43515a'})
    width,height=4.90,2.75;side=1.95;bottom=.23
    fig=plt.figure(figsize=(width,height))
    fig.text(.017,.960,'D',fontsize=22,fontweight='bold',va='top')
    fig.text(.50/width,.957,'Validation in mIF',fontsize=12.3,fontweight='bold',va='top')
    labels=[];axes=[]
    for left,img,title in zip([.12,2.19],pair,['Measured HLA-A','Predicted HLA-ABC']):
        ax=fig.add_axes([left/width,bottom/height,side/width,side/height])
        ax.imshow(img,interpolation='nearest');ax.axis('off')
        ax.set_title(title,fontsize=10,fontweight='bold',pad=7,color='#243745')
        ph,pw=img.shape[:2];length=50/umpp;bx=pw*.93-length;by=ph*.94
        ax.plot([bx,bx+length],[by,by],color='white',lw=1.6,solid_capstyle='butt')
        labels.append(ax.text(bx+length/2,by-ph*.035,'50 µm',ha='center',va='bottom',
                              color='white',fontsize=7.5,fontweight='bold'))
        axes.append(ax)
    cax=fig.add_axes([4.23/width,.62/height,.085/width,1.12/height])
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap='viridis'),cax=cax,ticks=[.25,.4,.55])
    cb.ax.set_yticklabels(['≤0.25','0.40','≥0.55'])
    cb.set_label('Relative score',fontsize=8,labelpad=5)
    cb.ax.tick_params(labelsize=7.2,length=2,pad=2);cb.outline.set_linewidth(.45)
    fig.text(2.13/width,.046/height,'Synthetic DAPI background',ha='center',va='bottom',fontsize=8.1,color='#243745')
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for t in [*fig.texts,*labels,*(a.title for a in axes),*cb.ax.get_yticklabels(),cb.ax.yaxis.label]:
        b=t.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        assert b.x0>=0 and b.x1<=1 and b.y0>=0 and b.y1<=1,(t.get_text(),b)
    for ext in ['png','svg']:fig.savefig(BASE/f'panel_D_validation.{ext}',dpi=300)
    plt.close(fig)
    assert all(sha(Path(p))==h for p,h in hashes.items())
    report={'status':'PASS','version':'senosain140k_20260921_same_ROI','patient_id':'P_26','core_id':'C_30',
            'n_cells_per_plot':389,'roi_id':chosen['roi_id'],'roi_size_um':[192,192],
            'roi_original_coordinate_bounds_um':prior['selected_original_coordinate_bounds_um'],
            'cell_spearman':rho,'full_C30_cell_spearman':full_rho,
            'measured_column':cols[0],'predicted_column':cols[1],
            'figure_size_inches':[width,height],'plot_size_inches':[side,side],'plot_bottom_inches':bottom,
            'color_range':[.25,.55],'cmap':'viridis','scale_bar_um':50,
            'rendered_pair_shape':list(pair[0].shape),'shared_DAPI_and_synthetic_shapes':True,
            'synthetic_background_label_on_figure':True,
            'background_description':'Frozen morphology-constrained synthetic DAPI, not acquired Zheng DAPI',
            'same_ROI_geometry_color_limits_and_blending_as_approved_preview':True,'new_ROI_or_threshold_search':False,
            'new_prediction_or_refit':True,'spatial_smoothing':False,'numeric_values_changed':True, 'new_prediction_source':'senosain140k_20260921',
            'all_text_inside_canvas':True,'source_files_unchanged':True,'source_sha256':hashes,
            'HLA_overlay_description':'Cell-level numerical annotation, not measured subcellular fluorescence'}
    save_json(BASE/'provenance/VERIFICATION.json',report)
    save_json(BASE/'provenance/SOURCES.json',[{'source':p,'sha256':h} for p,h in hashes.items()])
    save_json(BASE/'provenance/MANIFEST.json',[{'file':str(p.relative_to(BASE)),'sha256':sha(p)}
        for p in sorted(BASE.rglob('*')) if p.is_file() and p.relative_to(BASE).parts[0] not in {'cache','tmp','versions'} and p.name!='MANIFEST.json'])
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
