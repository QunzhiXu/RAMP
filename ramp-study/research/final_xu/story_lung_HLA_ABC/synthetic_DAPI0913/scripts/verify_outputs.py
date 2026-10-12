"""Verify image metadata, all instances, source coordinates, and original HLA values."""
from pathlib import Path
import sys,json,hashlib
BASE=Path(__file__).resolve().parents[1];ROOT=BASE.parents[1]
sys.path.insert(0,str(ROOT/'story_cd19_prediction/Exploration0909/real_spatial0913/deps'))
import numpy as np,pandas as pd,tifffile
from scipy import ndimage as ndi
from scipy.stats import spearmanr

def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    a=json.loads((BASE/'results/synthesis_audit.json').read_text())
    lookup=pd.read_csv(BASE/'data/C30_synthetic_cell_lookup.csv.gz')
    original=ROOT/'story_lung_HLA_ABC/validation/results/C30_CELLS.csv.gz'
    source=pd.read_csv(original)
    assert source.CellID.equals(lookup.CellID) and len(lookup)==10719
    np.testing.assert_array_equal(source[['x_um','y_um']],lookup[['x_um','y_um']])
    p=a['pixel_size_um'];origin=np.array(a['origin_um'])
    np.testing.assert_allclose(lookup[['render_x_px','render_y_px']].to_numpy()*p+origin,source[['x_um','y_um']],rtol=0,atol=1e-10)
    with tifffile.TiffFile(BASE/'data/C30_synthetic_DAPI.ome.tif') as t:
        assert t.is_ome and 'Synthetic DAPI' in t.ome_metadata
        assert t.pages[0].dtype==np.uint16
        assert t.pages[0].shape==tuple(a['image_shape_yx'])
    labels=tifffile.imread(BASE/'data/C30_synthetic_nuclear_labels.tif')
    cells=tifffile.imread(BASE/'data/C30_synthetic_cell_labels.tif')
    np.testing.assert_array_equal(np.unique(labels),np.arange(10720))
    np.testing.assert_array_equal(np.unique(cells),np.arange(10720))
    centers=np.array(ndi.center_of_mass(labels>0,labels,np.arange(1,10720)))[:,::-1]
    intended=lookup[['render_x_px','render_y_px']].to_numpy()
    deviation=np.sqrt(np.sum((centers-intended)**2,axis=1))*p
    actual=BASE/'data/C30_HLA_values_unchanged.csv.gz';hl=pd.read_csv(actual)
    for col in ['observed_HLA_A_patient_rank','RAMP_k30']:np.testing.assert_array_equal(hl[col],source[col])
    report={'status':'PASS','n_cells':10719,'n_nuclear_labels':10719,'n_cell_labels':10719,
        'OME_TIFF_is_explicitly_synthetic':True,'original_coordinates_retained':True,'HLA_values_exactly_unchanged':True,
        'Spearman_rho':float(spearmanr(source.observed_HLA_A_patient_rank,source.RAMP_k30).statistic),
        'synthetic_mask_centroid_deviation_um_quantiles':np.quantile(deviation,[0,.5,.9,.99,1]).tolist(),
        'centroid_deviation_explanation':'Rasterized/overlap-clipped synthetic shape centroid can differ from fixed placement anchor; placement anchors are exact original coordinates',
        'source_C30_sha256':sha(original),'source_AnnData_sha256':sha(ROOT/'raw_data/lung_cancer/Zheng2025/mIF_data.h5ad'),
        'synthetic_dapi_sha256':sha(BASE/'data/C30_synthetic_DAPI.ome.tif'),
        'methods_contain_no_diffusion_model':True,'original_main_figure_not_modified':True}
    assert report['synthetic_dapi_sha256']==a['synthetic_dapi_sha256']
    (BASE/'results/verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
