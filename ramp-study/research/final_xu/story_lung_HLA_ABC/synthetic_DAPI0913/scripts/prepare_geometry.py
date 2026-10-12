"""Read original Zheng geometry and keep HLA values in a separate file."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd,h5py
BASE=Path(__file__).resolve().parents[1];ROOT=BASE.parents[1]

def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    c30=ROOT/'story_lung_HLA_ABC/validation/results/C30_CELLS.csv.gz'
    source=ROOT/'raw_data/lung_cancer/Zheng2025/mIF_data.h5ad'
    d=pd.read_csv(c30)
    fields={'Nucleus: Area µm^2':'nucleus_area_um2','Nucleus: Max diameter µm':'nucleus_max_diameter_um',
            'Nucleus: Min diameter µm':'nucleus_min_diameter_um','Nucleus: Circularity':'nucleus_circularity',
            'Nucleus: Solidity':'nucleus_solidity','Cell: Area µm^2':'cell_area_um2',
            'Cell: Max diameter µm':'cell_max_diameter_um','Cell: Min diameter µm':'cell_min_diameter_um'}
    g=d[['source_row_id','CellID','x_um','y_um']].copy()
    with h5py.File(source,'r') as f:
        ix=d.source_row_id.to_numpy(int)
        np.testing.assert_array_equal(f['obs/CellID'][ix].astype(str),d.CellID.to_numpy(str))
        np.testing.assert_array_equal(f['obs/Centroid X µm'][ix],d.x_um.to_numpy())
        np.testing.assert_array_equal(f['obs/Centroid Y µm'][ix],d.y_um.to_numpy())
        for src,dest in fields.items():g[dest]=f['obs'][src][ix]
    assert len(g)==10719 and g.CellID.is_unique
    assert np.isfinite(g.select_dtypes('number').to_numpy()).all()
    g.to_csv(BASE/'data/C30_geometry_only.csv.gz',index=False)
    d[['CellID','observed_HLA_A_patient_rank','RAMP_k30']].to_csv(BASE/'data/C30_HLA_values_unchanged.csv.gz',index=False)
    (BASE/'results/geometry_source_audit.json').write_text(json.dumps({'files':{str(p):sha(p) for p in [c30,source]},
        'n_cells':len(g),'CellID_exact_match':True,'coordinates_exact_match':True,'all_morphology_fields_finite':True,
        'HLA_values_excluded_from_geometry':True},indent=2)+'\n')

if __name__=='__main__':main()
