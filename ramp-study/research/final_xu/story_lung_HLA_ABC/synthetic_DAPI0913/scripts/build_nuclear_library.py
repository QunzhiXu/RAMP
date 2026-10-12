"""Extract real DAPI nuclear textures; no HLA, cell-type or outcome inputs."""
from pathlib import Path
import sys,json,os,hashlib
BASE=Path(__file__).resolve().parents[1]
ROOT=BASE.parents[1]
BM=ROOT/'story_cd19_prediction/Exploration0909/real_spatial0913'
sys.dont_write_bytecode=True
sys.path.insert(0,str(BM/'deps'))
import numpy as np,pandas as pd,tifffile
from scipy import ndimage as ndi

def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def otsu(values):
    h=np.bincount(values.astype(np.uint8),minlength=256).astype(float)
    p=h/h.sum();w=p.cumsum();u=(p*np.arange(256)).cumsum()
    score=(u[-1]*w-u)**2/np.maximum(w*(1-w),1e-12)
    return int(np.argmax(score))

def main():
    config=json.loads((BASE/'config.json').read_text());rng=np.random.default_rng(config['seed'])
    mdpath=BM/'data/H14_full_metadata.csv.gz'
    md=pd.read_csv(mdpath,usecols=['cell_id','CellID','x.coord','y.coord','Size'])
    with tifffile.TiffFile(BM/'downloads/H14_Scan1.qptiff') as t:
        p=t.pages[0];assert '<Biomarker>DAPI</Biomarker>' in p.description
        im=p.asarray(maxworkers=4)
        res=p.tags['XResolution'].value;umpp=10000*res[1]/res[0]
    masks=tifffile.memmap(BM/'downloads/H14_Scan1_segmentation_cell.tif',mode='r')
    n=config['n_prototypes'];sz=config['prototype_grid_size'];lim=1.25
    v=np.linspace(-lim,lim,sz);gx,gy=np.meshgrid(v,v)
    textures=[];shapes=[];records=[];attempts=0
    for ix in rng.permutation(len(md)):
        row=md.iloc[ix];attempts+=1
        if not 90<=row.Size<=2200:continue
        cx,cy=int(round(row['x.coord'])),int(round(row['y.coord']));r=38
        if cx-r<0 or cy-r<0 or cx+r>=im.shape[1] or cy+r>=im.shape[0]:continue
        patch=im[cy-r:cy+r+1,cx-r:cx+r+1]
        own=masks[cy-r:cy+r+1,cx-r:cx+r+1]==int(row.CellID)
        vals=patch[own]
        if len(vals)<80 or np.percentile(vals,98)<45:continue
        bg=float(np.percentile(patch[~own],20)) if np.any(~own) else 5.
        threshold=max(otsu(vals),bg+10)
        lab,nlab=ndi.label((patch>threshold)&own)
        if not nlab:continue
        counts=np.bincount(lab.ravel());counts[0]=0;label=int(counts.argmax())
        size=int(counts[label])
        if size<25 or size>1200:continue
        if nlab>1 and np.sort(counts[1:])[-2]>.3*size:continue
        mask=ndi.binary_fill_holes(lab==label)
        yy,xx=np.where(mask);yc,xc=yy.mean(),xx.mean()
        if abs(xc-r)>14 or abs(yc-r)>14:continue
        coords=np.column_stack([xx-xc,yy-yc]);evals,evecs=np.linalg.eigh(np.cov(coords.T,bias=True))
        major=evecs[:,1];minor=evecs[:,0]
        xp=coords@major;yp=coords@minor
        a=max(abs(xp).max()+.5,1);b=max(abs(yp).max()+.5,1)
        if a/b>3.5 or a<3 or b<2.5:continue
        high=float(np.percentile(patch[mask],98))
        if high-bg<30:continue
        norm=np.clip((patch.astype(float)-bg)/(high-bg),0,1)
        alpha=ndi.gaussian_filter(mask.astype(float),.50)
        source=norm*alpha
        mx=xc+major[0]*gx*a+minor[0]*gy*b
        my=yc+major[1]*gx*a+minor[1]*gy*b
        texture=ndi.map_coordinates(source,[my,mx],order=1,mode='constant',cval=0).astype(np.float32)
        shape=ndi.map_coordinates(mask.astype(float),[my,mx],order=1,mode='constant',cval=0)>.5
        area_norm=float(shape.sum()*(2*lim/(sz-1))**2)
        fill=float(mask.sum()/(np.pi*a*b))
        if not .55<fill<1.25:continue
        textures.append(texture);shapes.append(shape)
        records.append(dict(prototype_id=len(records),source_cell_id=str(row.cell_id),source_label=int(row.CellID),
                            source_x_px=cx,source_y_px=cy,nuclear_area_um2=float(mask.sum()*umpp**2),
                            aspect_ratio=float(a/b),shape_fill_fraction=fill,canonical_area=area_norm,
                            raw_dapi_background=bg,raw_dapi_high=high,threshold=threshold))
        if len(records)%256==0:print('Accepted nuclear prototypes',len(records),'from',attempts,'candidates',flush=True)
        if len(records)==n:break
    assert len(records)==n,'Not enough acceptable real nuclei'
    np.savez_compressed(BASE/'data/real_DAPI_nuclear_library.npz',texture=np.stack(textures),mask=np.stack(shapes),
                        canonical_limit=lim,source_pixel_um=umpp)
    pd.DataFrame(records).to_csv(BASE/'data/nuclear_library_metadata.csv',index=False)
    audit=dict(source='H14 original CODEX DAPI',n_prototypes=n,candidates_examined=attempts,
        source_whole_cell_mask_used_as_extraction_bound=True,nuclear_segmentation='per-cell Otsu; dominant connected bright component; fill holes; image-quality filters',
        nuclear_masks_are_new_estimates_not_author_nuclear_masks=True,
        source_metadata_sha256=sha(mdpath),input_markers=['DAPI'],HLA_used=False,cell_labels_used=False,
        source_dapi_MD5='dd857566152f4d40e7bc7838e79a3100',source_masks_MD5='a8f10d65805ec64f2130016e22520c34',
        output_library_sha256=sha(BASE/'data/real_DAPI_nuclear_library.npz'))
    (BASE/'results/library_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2),flush=True)

if __name__=='__main__':main()
