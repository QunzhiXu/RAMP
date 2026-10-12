"""Render selected settings with the frozen synthetic background and HLA values."""
from pathlib import Path
import os,sys,json,hashlib
BASE=Path(__file__).resolve().parents[1];SOURCE=BASE.parent;ROOT=BASE.parents[2]
sys.path.insert(0,str(ROOT/'story_cd19_prediction/Exploration0909/real_spatial0913/deps'))
os.environ['MPLCONFIGDIR']=str(BASE/'cache/matplotlib');sys.dont_write_bytecode=True
import numpy as np,pandas as pd,tifffile,h5py
from scipy import ndimage as ndi
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from PIL import Image

def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    report=json.loads((BASE/'results/search_summary.json').read_text())
    geom=pd.read_csv(SOURCE/'data/C30_synthetic_cell_lookup.csv.gz')
    values=pd.read_csv(SOURCE/'data/C30_HLA_values_unchanged.csv.gz');assert values.CellID.equals(geom.CellID)
    syn=json.loads((SOURCE/'results/synthesis_audit.json').read_text());umpp=syn['pixel_size_um']
    dapi=tifffile.imread(SOURCE/'data/C30_synthetic_DAPI.ome.tif')
    cellmap=tifffile.imread(SOURCE/'data/C30_synthetic_cell_labels.tif')
    blue=(np.clip(dapi.astype(np.float32)/45000,0,1)**.85)[:,:,None]*np.array([.43,.62,1.],np.float32)
    boundary=(ndi.maximum_filter(cellmap,3)!=ndi.minimum_filter(cellmap,3))&(cellmap>0)
    def images(row):
        x0,y0,x1,y1=[int(round(float(row[k])/umpp)) for k in ['x0_um','y0_um','x1_um','y1_um']]
        sl=np.s_[y0:y1,x0:x1];lab=cellmap[sl];norm=Normalize(row['lower'],row['upper'],clip=True)
        pair=[]
        for col in ['observed_HLA_A_patient_rank','RAMP_k30']:
            c=np.zeros((len(values)+1,3),np.float32);c[1:]=plt.get_cmap('viridis')(norm(values[col]))[:,:3]
            img=blue[sl]*.65+c[lab]*.66;img[boundary[sl]]*=.88
            pair.append(np.clip(img,0,1))
        return pair,norm
    def axisimage(ax,img):
        ax.imshow(img,interpolation='nearest');ax.axis('off');h,w=img.shape[:2]
        length=50/umpp;x=w*.93-length;y=h*.94
        ax.plot([x,x+length],[y,y],color='white',lw=2.6,solid_capstyle='butt')
        ax.text(x+length/2,y-h*.035,'50 µm',ha='center',va='bottom',color='white',fontsize=10,fontweight='bold')
    def colorbar(fig,norm,rect):
        mid=round((norm.vmin+norm.vmax)*10)/20
        cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap='viridis'),cax=fig.add_axes(rect),ticks=[norm.vmin,mid,norm.vmax])
        cb.ax.set_yticklabels([f'≤{norm.vmin:.2f}',f'{mid:.2f}',f'≥{norm.vmax:.2f}'])
        cb.set_label('Relative rank',fontsize=10);cb.ax.tick_params(labelsize=10)
    def export(fig,name):
        for ext in ['png','pdf']:fig.savefig(BASE/'figures'/f'{name}.{ext}',dpi=240,facecolor='white')
        plt.close(fig)
    def single(row,name,heading='Selected ROI · synthetic DAPI background'):
        pair,norm=images(row);fig=plt.figure(figsize=(10.8,5.55))
        for rect,img,title in zip([[.02,.10,.425,.77],[.47,.10,.425,.77]],pair,['Measured HLA-A','Predicted HLA-ABC']):
            ax=fig.add_axes(rect);axisimage(ax,img);ax.set_title(title,fontsize=15,fontweight='bold',pad=10)
        colorbar(fig,norm,[.916,.25,.018,.48])
        fig.suptitle(heading,fontsize=15,fontweight='bold',y=.987)
        fig.text(.46,.018,f'{int(row["n_cells"])} cells  ·  {int(row["side_um"])} × {int(row["side_um"])} µm  ·  raw cell ρ = {row["raw_spearman"]:.3f}',ha='center',fontsize=12)
        export(fig,name)
    best=report['best'];single(best,'best_ROI_HLA_pair')
    best_size=pd.read_csv(BASE/'results/best_by_window_size.csv')
    for row in best_size.to_dict('records'):single(row,f'best_{int(row["side_um"])}um_HLA_pair')
    originals=pd.read_csv(BASE/'data/roi_candidates.csv')
    old=originals[originals.roi_id=='PRIOR_CENTRAL'].iloc[0].to_dict();old.update(report['original_central_original_bounds'])
    rows=[old,best]
    fig=plt.figure(figsize=(10.8,10.7))
    for j,(row,heading) in enumerate(zip(rows,['Previous central ROI','Selected ROI'])):
        pair,norm=images(row);bottom=.535-j*.49
        for x,img,title in zip([.02,.47],pair,['Measured HLA-A','Predicted HLA-ABC']):
            ax=fig.add_axes([x,bottom,.425,.385]);axisimage(ax,img);ax.set_title(title,fontsize=13,fontweight='bold',pad=6)
        colorbar(fig,norm,[.916,bottom+.055,.018,.28])
        fig.text(.46,bottom+.42,f'{heading} · {int(row["n_cells"])} cells · raw ρ {row["raw_spearman"]:.3f} · display CCC {row["display_ccc"]:.3f}',ha='center',fontsize=13,fontweight='bold')
    fig.suptitle('Same synthetic DAPI · ROI and color-range comparison',fontsize=15,fontweight='bold',y=.996)
    export(fig,'previous_vs_selected')
    distinct=pd.read_csv(BASE/'results/top_distinct_ROIs.csv').head(4)
    fig=plt.figure(figsize=(11,18.0))
    for j,row in enumerate(distinct.to_dict('records')):
        pair,norm=images(row);bottom=.755-j*.245
        for x,img in zip([.025,.465],pair):axisimage(fig.add_axes([x,bottom,.415,.21]),img)
        colorbar(fig,norm,[.915,bottom+.02,.016,.16])
        fig.text(.45,bottom+.218,f'#{j+1} · {int(row["side_um"])} µm · {int(row["n_cells"])} cells · raw ρ {row["raw_spearman"]:.3f} · CCC {row["display_ccc"]:.3f}',ha='center',fontsize=13,fontweight='bold')
    fig.suptitle('Distinct ROI candidates · synthetic DAPI\nMeasured HLA-A (left) / Predicted HLA-ABC (right)',fontsize=15,fontweight='bold',y=.999)
    export(fig,'top_distinct_ROI_comparisons')
    fig,ax=plt.subplots(figsize=(6.6,6.7));ax.imshow(blue,interpolation='nearest');ax.axis('off')
    for row,color,label in [(old,'#C0C5CB','Previous'),(best,'#FFBE35','Selected')]:
        xx=row['x0_um']/umpp;yy=row['y0_um']/umpp;side=row['side_um']/umpp
        ax.add_patch(Rectangle((xx,yy),side,side,fill=False,ec=color,lw=2))
        ax.text(xx,yy-20,label,color=color,fontsize=11,fontweight='bold',va='bottom')
    ax.set_title('C30 · synthetic DAPI · ROI locations',fontsize=14,fontweight='bold')
    fig.subplots_adjust(left=.01,right=.99,bottom=.01,top=.94);export(fig,'ROI_locations')
    with h5py.File(BASE/'results/all_roi_threshold_results.h5','r') as f:
        ids=f['roi_id'].asstr()[:];i=int(np.where(ids==best['roi_id'])[0][0])
        ccc=f['display_ccc'][:,i];valid=f['admissible'][:,i];bounds=f['bounds'][:]
    heat=np.full((21,21),np.nan)
    for (lo,hi),val,ok in zip(bounds,ccc,valid):
        if ok:heat[int(round(hi*20)),int(round(lo*20))]=val
    fig,ax=plt.subplots(figsize=(6,5))
    artist=ax.imshow(heat,origin='lower',extent=[-.025,1.025,-.025,1.025],cmap='magma',aspect='equal')
    ax.plot(best['lower'],best['upper'],marker='o',mfc='none',mec='#13E9D0',ms=10,mew=2)
    ax.set(xlabel='Color lower limit',ylabel='Color upper limit',title='Selected ROI · display CCC')
    cb=fig.colorbar(artist,ax=ax);cb.set_label('Cell-value display CCC')
    fig.tight_layout();export(fig,'selected_ROI_threshold_landscape')
    # Save exact original values for selected centroids and independently recompute all key metrics.
    xy=geom[['render_x_px','render_y_px']].to_numpy()*umpp
    local=(xy[:,0]>=best['x0_um'])&(xy[:,0]<best['x1_um'])&(xy[:,1]>=best['y0_um'])&(xy[:,1]<best['y1_um'])
    selected=geom.loc[local,['synthetic_label','CellID','x_um','y_um']].merge(values,on='CellID',validate='one_to_one')
    selected.to_csv(BASE/'data/selected_ROI_cells.csv.gz',index=False)
    x=selected.observed_HLA_A_patient_rank.to_numpy();y=selected.RAMP_k30.to_numpy()
    a=np.clip((x-best['lower'])/(best['upper']-best['lower']),0,1);b=np.clip((y-best['lower'])/(best['upper']-best['lower']),0,1)
    ccc=2*np.mean((a-a.mean())*(b-b.mean()))/(a.var()+b.var()+(a.mean()-b.mean())**2)
    assert len(selected)==best['n_cells']
    assert abs(float(spearmanr(x,y).statistic)-best['raw_spearman'])<1e-12
    assert abs(ccc-best['display_ccc'])<1e-10
    panel_stats={name:{'lower_saturated_fraction':float(np.mean(t<=0)),'upper_saturated_fraction':float(np.mean(t>=1)),
                            'interior_fraction':float(np.mean((t>0)&(t<1))),'display_sd':float(t.std())} for name,t in [('measured',a),('predicted',b)]}
    verified={'status':'PASS','selected_ROI_n_cells':len(selected),'raw_spearman':best['raw_spearman'],'display_ccc':float(ccc),
              'display_MAE':float(np.mean(abs(a-b))),'panel_statistics':panel_stats,'shared_DAPI_and_synthetic_shapes':True,
              'values_not_spatially_smoothed':True,'all_input_hashes_unchanged':True,
              'selected_original_coordinate_bounds_um':[best['x0_um']+syn['origin_um'][0],best['y0_um']+syn['origin_um'][1],best['x1_um']+syn['origin_um'][0],best['y1_um']+syn['origin_um'][1]]}
    for path,h in report['source_sha256'].items():assert sha(path)==h,path
    (BASE/'results/render_verification.json').write_text(json.dumps(verified,indent=2)+'\n')
    print(json.dumps(verified,indent=2),flush=True)

if __name__=='__main__':main()
