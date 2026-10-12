"""Compute UMAPs from read-only matrices in the sibling data_umap package."""
import package_runtime as rt
from package_runtime import BASE
import argparse
import fcntl
import json
import sys
import time
import importlib.metadata as im
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

SOURCE=BASE.parent/'data_umap'
OUT=BASE/'results'
METHODS=['baseline','RAMP']


def inputs():
    design=json.loads((SOURCE/'data/DESIGN.json').read_text())
    md=pd.read_csv(SOURCE/'data/CELL_METADATA.csv.gz',dtype={'patient_id':str},low_memory=False)
    files=[SOURCE/'data/DESIGN.json',SOURCE/'data/CELL_METADATA.csv.gz',SOURCE/'data/MARKER_MAPPING.csv',
           SOURCE/'data/SELECTED_DONORS.csv']
    arrays={};config={}
    for method in METHODS:
        path=SOURCE/'reference'/f'{method}_SHARED9.npy'
        arrays[method]=np.load(path)
        settings=SOURCE/'reference'/f'{method}_UMAP.json'
        config[method]=json.loads(settings.read_text())
        assert arrays[method].shape==(84000,9) and np.isfinite(arrays[method]).all()
        assert rt.sha(path)==config[method]['input_sha256']
        files.extend([path,settings,SOURCE/'reference'/f'{method}_UMAP.npy',SOURCE/'reference'/f'{method}_STANDARDIZED.npy'])
    assert len(md)==84000 and md.cell_key.is_unique
    assert md.groupby('unit_key').size().eq(1000).all() and md.unit_key.nunique()==84
    assert md.groupby('dataset').patient_id.nunique().eq(28).all()
    assert np.array_equal(md.dataset.to_numpy(),np.repeat(design['datasets'],28000))
    assert len(design['markers'])==9 and not {'HLA-A','HLA_A','HLA_ABC','HLA-ABC'}&set(design['markers'])
    return md,arrays,config,{str(p):rt.sha(p) for p in files}


def verify():
    md,arrays,config,hashes=inputs()
    manifest=json.loads((OUT/'RUN_MANIFEST.json').read_text())
    assert hashes==manifest['source_sha256'],'Source data/settings changed since this run.'
    checks={}
    for method in METHODS:
        x=np.load(OUT/f'{method}_STANDARDIZED.npy')
        u=np.load(OUT/f'{method}_UMAP.npy')
        assert x.shape==(84000,9) and u.shape==(84000,2) and np.isfinite(u).all()
        regenerated=StandardScaler().fit(arrays[method]).transform(arrays[method])
        assert np.max(np.abs(regenerated-x))<1e-12
        assert rt.sha(OUT/f'{method}_UMAP.npy')==manifest['output_sha256'][f'{method}_UMAP.npy']
        target=np.load(SOURCE/'reference'/f'{method}_UMAP.npy')
        checks[method]={'coordinate_max_error_vs_reference':float(np.max(np.abs(u-target))),
                        'coordinates_exactly_equal':bool(np.array_equal(u,target))}
    status='PASS' if all(v['coordinates_exactly_equal'] for v in checks.values()) else 'VALID_UMAP_DIFFERENT_FROM_REFERENCE'
    rt.save_json(OUT/'VERIFICATION.json',{'status':status,'time_utc':rt.now(),
        'n_cells':len(md),'same_cells_and_row_order':True,'source_files_unchanged':True,
        'source_sha256':hashes,'methods':checks})
    print(json.dumps({'status':status,**checks},indent=2),flush=True)


def run(recompute=False):
    OUT.mkdir(exist_ok=True)
    lock=(OUT/'RUNNING.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'RUN_MANIFEST.json').exists() and not recompute:
        print('Existing completed run found. Verifying it; use --recompute to calculate again.',flush=True)
        return verify()
    md,arrays,config,hashes=inputs()
    import umap
    t=time.time();records={}
    rt.save_json(OUT/'RUN_STATUS.json',{'phase':'running','started_utc':rt.now()})
    for method in METHODS:
        start=time.time()
        print(f'{method}: pooled StandardScaler + fresh UMAP for 84,000 cells...',flush=True)
        scaler=StandardScaler().fit(arrays[method])
        x=scaler.transform(arrays[method])
        expected=np.load(SOURCE/'reference'/f'{method}_STANDARDIZED.npy')
        assert np.max(np.abs(x-expected))<1e-12
        np.save(OUT/f'{method}_STANDARDIZED.npy',x)
        params=config[method]['parameters']
        u=umap.UMAP(**params).fit_transform(x)
        assert u.shape==(84000,2) and np.isfinite(u).all()
        np.save(OUT/f'{method}_UMAP.npy',u)
        records[method]={'parameters':params,'scale_mean':scaler.mean_.tolist(),'scale_sd':scaler.scale_.tolist(),
                         'preprocessing':'pooled StandardScaler fitted separately for each method; no PCA',
                         'source_expression_matrix':str(SOURCE/'reference'/f'{method}_SHARED9.npy'),
                         'seconds':time.time()-start}
        rt.save_json(OUT/f'{method}_UMAP.json',records[method])
        rt.save_json(OUT/'RUN_STATUS.json',{'phase':f'{method} UMAP complete','updated_utc':rt.now()})
        print(f'{method}: done in {time.time()-start:.1f}s.',flush=True)
    from plot_umap import render
    render(OUT,md)
    assert all(rt.sha(p)==h for p,h in hashes.items())
    rt.save_json(OUT/'RUN_MANIFEST.json',{'status':'complete','time_utc':rt.now(),'seconds':time.time()-t,
        'source_directory':str(SOURCE),'source_sha256':hashes,'n_cells':len(md),'n_units':84,'n_shared_markers':9,
        'calculation':'Fresh UMAP from existing expression matrices; no RAMP refit or donor resampling',
        'reference_coordinates_used_for':'Verification only, never as UMAP input or initialization',
        'input_metadata_used_for_embedding':False,'methods':records,
        'output_sha256':{f'{m}_UMAP.npy':rt.sha(OUT/f'{m}_UMAP.npy') for m in METHODS},
        'environment':{'python':sys.version,'packages':{name:im.version(name) for name in ['numpy','pandas','scipy','scikit-learn','umap-learn','numba','pynndescent','matplotlib']}},
        'script_sha256':{p.name:rt.sha(p) for p in (BASE/'scripts').glob('*.py')}})
    verify()
    rt.save_json(OUT/'RUN_STATUS.json',{'phase':'complete','updated_utc':rt.now()})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['run','plot','verify'],default='run',nargs='?')
    parser.add_argument('--recompute',action='store_true',help='Explicitly replace this folder\'s existing UMAP outputs with a fresh run.')
    args=parser.parse_args()
    if args.command=='run':run(args.recompute)
    elif args.command=='verify':verify()
    else:
        verify()
        md,_,_,_=inputs()
        from plot_umap import render
        render(OUT,md)


if __name__=='__main__':main()
