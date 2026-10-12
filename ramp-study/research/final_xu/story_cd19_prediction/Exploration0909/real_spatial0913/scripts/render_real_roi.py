"""Render original TIFF pixels and predictions on genuine Mesmer cell masks."""
from pathlib import Path
import os,sys,json,xml.etree.ElementTree as ET
BASE=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(BASE/'cache/matplotlib')
sys.path.insert(0,str(BASE/'deps'));sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import maximum_filter,minimum_filter
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def read_crop(t,page,roi):
    x0,y0,x1,y1=[int(roi[k]) for k in ['x0','y0','x1','y1']]
    out=np.zeros((y1-y0,x1-x0),dtype=page.dtype)
    th,tw=page.tilelength,page.tilewidth; nx=(page.shape[1]+tw-1)//tw
    for row in range(y0//th,(y1-1)//th+1):
        for col in range(x0//tw,(x1-1)//tw+1):
            index=row*nx+col;off=page.dataoffsets[index];nb=page.databytecounts[index]
            assert off+nb <= t.filehandle.size, 'Source file still downloading requested tile'
            t.filehandle.seek(off);segment,_,_=page.decode(t.filehandle.read(nb),index)
            tile=segment[0,:,:,0]
            top,left=row*th,col*tw
            ya,yb=max(y0,top),min(y1,top+th);xa,xb=max(x0,left),min(x1,left+tw)
            out[ya-y0:yb-y0,xa-x0:xb-x0]=tile[ya-top:yb-top,xa-left:xb-left]
    return out

def save_rgb(rgb,name):
    Image.fromarray(np.round(np.clip(rgb,0,1)*255).astype(np.uint8)).save(BASE/'figures'/name)

def normalized(a,hi=None,gamma=1):
    if hi is None: hi=max(1,float(np.percentile(a[a>0],99.5))) if np.any(a>0) else 1
    return np.clip(a.astype(float)/hi,0,1)**gamma,hi

def axes_image(ax,rgb,title,umpp,subtitle=None):
    ax.imshow(rgb,interpolation='nearest');ax.set_axis_off()
    ax.set_title(title,fontsize=14,fontweight='bold',pad=10)
    pixels=50/umpp;y=rgb.shape[0]-55;x=rgb.shape[1]-pixels-45
    ax.plot([x,x+pixels],[y,y],color='white',lw=3,solid_capstyle='butt')
    ax.text(x+pixels/2,y-17,'50 µm',color='white',ha='center',va='bottom',fontsize=10,fontweight='bold')
    if subtitle: ax.text(.5,-.048,subtitle,ha='center',va='top',transform=ax.transAxes,fontsize=11)

def figure_images(images,titles,filename,roi,subtitles=None):
    n=len(images);fig,axes=plt.subplots(1,n,figsize=(4.1*n,4.6),squeeze=False)
    for i,(a,rgb,title) in enumerate(zip(axes[0],images,titles)):
        axes_image(a,rgb,title,roi['pixel_size_um'],None if subtitles is None else subtitles[i])
    fig.subplots_adjust(left=.01,right=.99,bottom=.09,top=.90,wspace=.035)
    fig.savefig(BASE/'figures'/f'{filename}.png',dpi=250,facecolor='white')
    fig.savefig(BASE/'figures'/f'{filename}.pdf',dpi=300,facecolor='white')
    plt.close(fig)

def main():
    roi=json.loads((BASE/'results/selected_roi.json').read_text())
    full=pd.read_csv(BASE/'data/H14_full_predictions.csv.gz')
    chosen=pd.read_csv(BASE/'data/selected_roi_cells.csv.gz')
    source_mask=tifffile.memmap(BASE/'downloads/H14_Scan1_segmentation_cell.tif',mode='r')
    x0,y0,x1,y1=[int(roi[k]) for k in ['x0','y0','x1','y1']]
    mask=np.array(source_mask[y0:y1,x0:x1])
    np.save(BASE/'data/selected_roi_mask.npy',mask)
    raw={};pages={};sourcepath=BASE/'downloads/H14_Scan1.qptiff'
    with tifffile.TiffFile(sourcepath) as t:
        for i,p in enumerate(t.pages):
            try:
                el=ET.fromstring(p.description)
                if el.findtext('ImageType')=='FullResolution': pages[el.findtext('Biomarker')]=i
            except ET.ParseError:pass
        for marker in ['DAPI','CD19','CD79A','PAX5']:
            raw[marker]=read_crop(t,t.pages[pages[marker]],roi)
    np.savez_compressed(BASE/'data/selected_roi_measured_channels.npz',**raw)
    maxid=int(max(mask.max(),full.CellID.max()))
    lookup_b=np.zeros(maxid+1,bool);lookup_pred=np.zeros(maxid+1,bool);lookup_known=np.zeros(maxid+1,bool)
    ids=full.CellID.to_numpy(int);assert len(np.unique(ids))==len(ids)
    lookup_b[ids]=full.annotated_b;lookup_pred[ids]=full.predicted_high;lookup_known[ids]=True
    # Centroid-outside cells intersecting the crop remain colored by their actual labels/predictions.
    bmap=lookup_b[mask];pmap=lookup_pred[mask]
    boundary=(maximum_filter(mask,3)!=minimum_filter(mask,3))&(mask>0)
    dapi,dapi_hi=normalized(raw['DAPI'],gamma=.8)
    background=dapi[:,:,None]*np.array([.32,.45,.82])
    positive=np.array([.10,.85,.48])
    pred=np.clip(background+pmap[:,:,None]*positive,0,1)
    truth=np.clip(background+bmap[:,:,None]*positive,0,1)
    pred[pmap&boundary]*=.78;truth[bmap&boundary]*=.78
    # Measured-channel images use original pixels, with separately documented linear display ranges.
    b79,b79_hi=normalized(raw['CD79A'],gamma=1)
    pax,pax_hi=normalized(raw['PAX5'],gamma=1)
    observed,cd19_hi=normalized(raw['CD19'],gamma=1)
    actual_b=np.clip(background+b79[:,:,None]*np.array([.10,.90,.45]),0,1)
    actual_pax=np.clip(background+pax[:,:,None]*np.array([.90,.25,.45]),0,1)
    actual_cd19=np.clip(background+observed[:,:,None]*positive,0,1)
    for marker,a in raw.items():
        tifffile.imwrite(BASE/'data'/f'H14_ROI_measured_{marker}.tif',a,metadata={'axes':'YX','source':str(sourcepath),'roi':roi})
    for name,rgb in [('DAPI',background),('predicted_CD19_masks',pred),('annotated_B_masks',truth),('measured_CD19',actual_cd19),('measured_CD79A',actual_b),('measured_PAX5',actual_pax)]:
        save_rgb(rgb,'H14_ROI_'+name+'.png')
    figure_images([pred,truth],['Predicted CD19 > 1.4','Annotated B cells'],'H14_best_ROI_real_cells_pair',roi,
                  [f'Precision {roi["precision"]:.1%}',f'Recall {roi["recall"]:.1%}'])
    figure_images([actual_b,pred,truth],['Measured CD79A + DAPI','Predicted CD19 > 1.4','Annotated B cells'],
                  'H14_best_ROI_true_image_and_predictions',roi,
                  ['Original CODEX pixels',f'Precision {roi["precision"]:.1%}',f'Recall {roi["recall"]:.1%}'])
    figure_images([actual_cd19,actual_b,actual_pax],['Measured CD19 + DAPI','Measured CD79A + DAPI','Measured PAX5 + DAPI'],
                  'H14_best_ROI_measured_channels',roi,['Original CODEX pixels']*3)
    # Geometry QC: compare source metadata with independent pixel label at each nucleus/cell centroid.
    m=tifffile.memmap(BASE/'downloads/H14_Scan1_segmentation_cell.tif',mode='r')
    xx=np.rint(full['x.coord']).to_numpy(int);yy=np.rint(full['y.coord']).to_numpy(int)
    at=m[yy,xx];mismatch=at!=ids
    records=[]
    for i in np.flatnonzero(mismatch):
        patch=m[max(0,yy[i]-64):yy[i]+65,max(0,xx[i]-64):xx[i]+65]
        records.append(dict(cell_id=str(full.cell_id.iloc[i]),CellID=int(ids[i]),centroid_label=int(at[i]),own_label_near_centroid=bool(np.any(patch==ids[i]))))
    areas=np.zeros(maxid+1,np.int64)
    for y0 in range(0,m.shape[0],512):
        h=np.bincount(np.asarray(m[y0:y0+512]).reshape(-1),minlength=maxid+1)
        if len(h)>len(areas): areas=np.pad(areas,(0,len(h)-len(areas)))
        areas[:len(h)]+=h
    sizes=full.Size.to_numpy(int)
    chosen_ids=chosen.CellID.to_numpy(int)
    unknown=np.setdiff1d(np.unique(mask),np.r_[0,ids]);outside=np.setdiff1d(np.intersect1d(np.unique(mask),ids),chosen_ids)
    audit=dict(roi=roi,source_channel_pages=pages,raw_channel_stats={k:dict(min=int(a.min()),max=int(a.max()),mean=float(a.mean()),nonzero_fraction=float(np.mean(a>0)),quantiles=np.percentile(a,[50,90,99,99.5,100]).tolist()) for k,a in raw.items()},
        display=dict(dapi_hi=dapi_hi,dapi_gamma=.8,cd19_hi=cd19_hi,cd79a_hi=b79_hi,pax5_hi=pax_hi,
                     cell_mask_positive_color=positive.tolist(),same_background_both_comparison_panels=True,spatial_smoothing=False),
        correspondence=dict(n_atlas_cells=len(full),centroid_label_matches=int((~mismatch).sum()),centroid_exceptions=records,
                            all_atlas_labels_present=bool(np.all(areas[ids]>0)),size_exact_match_fraction=float(np.mean(areas[ids]==sizes)),
                            max_area_difference=int(np.max(abs(areas[ids]-sizes))),roi_centroid_cells_all_in_mask=bool(np.isin(chosen_ids,np.unique(mask)).all()),
                            segmentation_labels_without_atlas_record_in_crop=unknown.tolist(),atlas_cell_ids_with_centroid_outside_crop=outside.tolist(),
                            edge_cell_rule='Color every overlapping atlas cell; metric inclusion uses centroids in ROI. Unknown objects show only real DAPI.'),
        source_pixels_unaltered_in_tif_exports=True,predicted_fluorescence_texture_generated=False)
    (BASE/'results/render_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k!='source_channel_pages'},indent=2),flush=True)
    assert np.isin(chosen_ids,np.unique(mask)).all()
    assert np.all(areas[ids]==sizes),'Mask area differs from source metadata; stop for investigation'
    print(BASE/'figures/H14_best_ROI_true_image_and_predictions.png',flush=True)

if __name__=='__main__':main()
