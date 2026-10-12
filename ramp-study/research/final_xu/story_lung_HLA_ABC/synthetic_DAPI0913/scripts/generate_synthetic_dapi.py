"""Morphology-constrained DAPI synthesis. This program never reads HLA values."""
from pathlib import Path
import sys,json,hashlib
BASE=Path(__file__).resolve().parents[1]
ROOT=BASE.parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'story_cd19_prediction/Exploration0909/real_spatial0913/deps'))
import numpy as np,pandas as pd,tifffile
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    cfg=json.loads((BASE/'config.json').read_text());rng=np.random.default_rng(cfg['seed'])
    geometry_path=BASE/'data/C30_geometry_only.csv.gz'
    g=pd.read_csv(geometry_path)
    assert not any('HLA' in c or 'RAMP' in c for c in g.columns)
    lm=pd.read_csv(BASE/'data/nuclear_library_metadata.csv')
    lib=np.load(BASE/'data/real_DAPI_nuclear_library.npz')
    textures=lib['texture'];shapes=lib['mask'];lim=float(lib['canonical_limit']);sz=textures.shape[1]
    p=cfg['pixel_size_um'];pad=cfg['padding_um']
    xy=g[['x_um','y_um']].to_numpy();origin=np.floor((xy.min(0)-pad)/p)*p
    extent=np.ceil((xy.max(0)+pad-origin)/p).astype(int)+1
    width,height=extent
    fluorescence=np.zeros((height,width),np.float32)
    nuclear_labels=np.zeros((height,width),np.uint32)
    cell_labels=np.zeros((height,width),np.uint32)
    nuclear_distance=np.full((height,width),np.inf,np.float32)
    cell_distance=np.full((height,width),np.inf,np.float32)
    # Match measured area/aspect ratio and approximate shape compactness, then diversify among 16 nearest candidates.
    reference=np.column_stack([np.log(lm.nuclear_area_um2),2*np.log(lm.aspect_ratio),2*lm.shape_fill_fraction])
    tree=cKDTree(reference)
    ratios=g.nucleus_max_diameter_um.to_numpy()/g.nucleus_min_diameter_um.to_numpy()
    fill=g.nucleus_area_um2.to_numpy()/(np.pi*g.nucleus_max_diameter_um*g.nucleus_min_diameter_um/4)
    query=np.column_stack([np.log(g.nucleus_area_um2),2*np.log(ratios),2*fill])
    dist,knn=tree.query(query,k=16)
    chosen=[]
    for ds,ids in zip(dist,knn):
        weights=np.exp(-(ds-ds.min())/.15);weights/=weights.sum()
        chosen.append(int(rng.choice(ids,p=weights)))
    chosen=np.array(chosen)
    angles=rng.uniform(0,2*np.pi,len(g))
    brightness=np.clip(rng.lognormal(np.log(.42),.32,len(g)),.16,.90)
    geometry_area_before=[]
    for i,row in enumerate(g.itertuples(index=False)):
        proto=chosen[i];canonical_area=float(lm.canonical_area.iloc[proto]);ratio=ratios[i]
        a=np.sqrt(row.nucleus_area_um2*ratio/canonical_area);b=a/ratio
        cellscale=np.sqrt(row.cell_area_um2/row.nucleus_area_um2)
        ac,bc=a*cellscale,b*cellscale
        cx,cy=(np.array([row.x_um,row.y_um])-origin)/p
        radius=int(np.ceil(lim*max(ac,bc)/p))+3
        x0=max(0,int(np.floor(cx))-radius);x1=min(width,int(np.ceil(cx))+radius+1)
        y0=max(0,int(np.floor(cy))-radius);y1=min(height,int(np.ceil(cy))+radius+1)
        yy,xx=np.mgrid[y0:y1,x0:x1];dx=(xx-cx)*p;dy=(yy-cy)*p
        c,s=np.cos(angles[i]),np.sin(angles[i]);u=c*dx+s*dy;v=-s*dx+c*dy
        mx=(u/a+lim)/(2*lim)*(sz-1);my=(v/b+lim)/(2*lim)*(sz-1)
        tex=ndi.map_coordinates(textures[proto],[my,mx],order=1,mode='constant',cval=0)
        nuc=ndi.map_coordinates(shapes[proto].astype(np.float32),[my,mx],order=1,mode='constant',cval=0)>.5
        mcx=(u/ac+lim)/(2*lim)*(sz-1);mcy=(v/bc+lim)/(2*lim)*(sz-1)
        cell=ndi.map_coordinates(shapes[proto].astype(np.float32),[mcy,mcx],order=1,mode='constant',cval=0)>.5
        geometry_area_before.append(float(nuc.sum()*p*p))
        # Original-coordinate nuclei remain fixed. Competing synthetic masks use nearest-centroid ownership.
        d2=(dx*dx+dy*dy).astype(np.float32)
        for region,labelmap,dmap in [(nuc,nuclear_labels,nuclear_distance),(cell,cell_labels,cell_distance)]:
            local=dmap[y0:y1,x0:x1];win=region&(d2<local)
            local[win]=d2[win];labelmap[y0:y1,x0:x1][win]=i+1
        np.maximum(fluorescence[y0:y1,x0:x1],tex*brightness[i],out=fluorescence[y0:y1,x0:x1])
        if (i+1)%2000==0:print('Rendered',i+1,'nuclei',flush=True)
    assert len(np.unique(nuclear_labels))-1==len(g),'At least one nucleus was lost'
    assert len(np.unique(cell_labels))-1==len(g),'At least one cell was lost'
    # Prototypes already contain optical texture; only mild resampling blur and an independent detector-noise floor.
    fluorescence=ndi.gaussian_filter(fluorescence,.30)
    fluorescence=np.maximum(fluorescence+rng.normal(.0004,.0015,fluorescence.shape).astype(np.float32),0)
    dapi=np.round(np.clip(fluorescence,0,1)*65535).astype(np.uint16)
    description='SYNTHETIC DAPI. Real Zheng C30 nuclear coordinates and measured morphology; texture prototypes from bone marrow H14. Not an acquired Zheng DAPI image. HLA values were not used.'
    tifffile.imwrite(BASE/'data/C30_synthetic_DAPI.ome.tif',dapi,ome=True,compression='zlib',
        metadata={'axes':'YX','PhysicalSizeX':p,'PhysicalSizeXUnit':'µm','PhysicalSizeY':p,'PhysicalSizeYUnit':'µm',
                  'Channel':{'Name':['Synthetic DAPI']},'Description':description})
    for name,array in [('nuclear',nuclear_labels),('cell',cell_labels)]:
        tifffile.imwrite(BASE/'data'/f'C30_synthetic_{name}_labels.tif',array,compression='zlib',
            metadata={'axes':'YX','description':'Synthetic instance labels, not author segmentation','pixel_size_um':p,'origin_um':origin.tolist()})
    out=g.copy();out.insert(0,'synthetic_label',np.arange(1,len(g)+1))
    out['prototype_id']=chosen;out['synthetic_angle_rad']=angles;out['synthetic_brightness']=brightness
    out['render_x_px']=(xy[:,0]-origin[0])/p;out['render_y_px']=(xy[:,1]-origin[1])/p
    out['nuclear_area_before_overlap_um2']=geometry_area_before
    area=np.bincount(nuclear_labels.ravel(),minlength=len(g)+1)[1:]*p*p
    out['nuclear_area_after_overlap_um2']=area
    out.to_csv(BASE/'data/C30_synthetic_cell_lookup.csv.gz',index=False)
    # A fixed central 256-µm crop is chosen from coordinates alone, without expression or prediction performance.
    center=(xy.min(0)+xy.max(0))/2;side=256
    lo=np.rint((center-side/2-origin)/p).astype(int);hi=lo+int(round(side/p))
    in_crop=(xy[:,0]>=origin[0]+lo[0]*p)&(xy[:,0]<origin[0]+hi[0]*p)&(xy[:,1]>=origin[1]+lo[1]*p)&(xy[:,1]<origin[1]+hi[1]*p)
    crop={'x0':int(lo[0]),'y0':int(lo[1]),'x1':int(hi[0]),'y1':int(hi[1]),'side_um':side,'n_centroids':int(in_crop.sum()),
          'selection':'Fixed 256-µm window at the midpoint of the C30 coordinate bounds; no HLA-based selection'}
    rel_before=np.array(geometry_area_before)/g.nucleus_area_um2.to_numpy()
    rel_after=area/g.nucleus_area_um2.to_numpy()
    audit={'synthetic':True,'geometry_file_sha256':sha(geometry_path),'library_sha256':sha(BASE/'data/real_DAPI_nuclear_library.npz'),
        'image_shape_yx':[int(height),int(width)],'origin_um':origin.tolist(),'pixel_size_um':p,'n_cells':len(g),
        'n_nuclear_instance_labels':len(np.unique(nuclear_labels))-1,'n_cell_instance_labels':len(np.unique(cell_labels))-1,
        'n_distinct_prototypes_used':len(np.unique(chosen)),'HLA_values_read':False,'cell_type_labels_read':False,
        'coordinates_unchanged':bool(np.allclose(out[['render_x_px','render_y_px']].to_numpy()*p+origin,xy,rtol=0,atol=1e-10)),
        'orientation_observed':False,'brightness_observed':False,'seed':cfg['seed'],'detail_crop':crop,
        'nuclear_area_ratio_before_overlap_quantiles':np.quantile(rel_before,[0,.01,.5,.99,1]).tolist(),
        'nuclear_area_ratio_after_overlap_quantiles':np.quantile(rel_after,[0,.01,.5,.99,1]).tolist(),
        'overlap_rule':'nearest original centroid wins in conflicting synthetic label pixels; centers are not moved',
        'emission_overlap_rule':'pixelwise maximum, followed by mild resampling blur; no HLA inputs',
        'synthetic_dapi_sha256':sha(BASE/'data/C30_synthetic_DAPI.ome.tif')}
    (BASE/'results/synthesis_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2),flush=True)

if __name__=='__main__':main()
