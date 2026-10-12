"""Verify the packaged data, or reproduce RAMP, frozen profiles and both UMAPs."""
import package_runtime as rt
from package_runtime import BASE
import argparse
import json
import sys
import time
import fcntl
import numpy as np
import pandas as pd

DATA=BASE/'data'
REFERENCE=BASE/'reference'
OUT=BASE/'reproduced'


def load_inputs():
    design=json.loads((DATA/'DESIGN.json').read_text())
    md=pd.read_csv(DATA/'CELL_METADATA.csv.gz',dtype={'patient_id':str},low_memory=False)
    raw=np.load(DATA/'RAW_SHARED9.npy')
    return design,md,raw


def verify(write=True):
    manifest=json.loads((BASE/'MANIFEST.json').read_text())
    for item in manifest['copied_files']:
        assert rt.sha(BASE/item['file'])==item['sha256'],item['file']
    design,md,raw=load_inputs()
    markers=design['markers'];datasets=design['datasets']
    assert raw.shape==(84000,9) and np.isfinite(raw).all() and (raw>=0).all()
    assert len(md)==84000 and md.cell_key.is_unique
    assert md.groupby('unit_key').size().eq(1000).all() and md.unit_key.nunique()==84
    assert md.groupby('dataset').size().eq(28000).all()
    assert md.groupby('dataset').patient_id.nunique().eq(28).all()
    assert np.array_equal(md.dataset.to_numpy(),np.repeat(datasets,28000))
    initial=pd.read_csv(BASE/'provenance/CELL_METADATA.csv.gz',dtype={'patient_id':str},low_memory=False)
    identifiers=['cell_key','dataset','patient_id','source_row_id','source_cell_id','unit_key','roi_id','provider_cell_type']
    assert md[identifiers].equals(initial[identifiers])
    donors=pd.read_csv(DATA/'SELECTED_DONORS.csv',dtype={'patient_id':str})
    assert donors.groupby(['dataset','stage_group']).size().to_dict()=={(d,s):n for d in datasets for s,n in [('I',19),('II',9)]}
    mapping=pd.read_csv(DATA/'MARKER_MAPPING.csv')
    assert mapping.marker.tolist()==markers
    assert not {'HLA-A','HLA_A','HLA_ABC','HLA-ABC','HISTONE_H3'}&set(markers)
    expected_units=set(md.unit_key)
    assert expected_units==set(donors.dataset+'|'+donors.patient_id)
    result={'status':'PASS','verified_utc':rt.now(),'n_cells':84000,'n_markers':9,'n_units':84,
            'per_dataset':{'patients':28,'cells':28000,'stage_I':19,'stage_II':9},
            'all_41_imported_hashes_match':True,'matrix_row_order_and_identity_verified':True,
            'corrected_display_metadata_fit_identity_unchanged':True,'HLA_targets_excluded':True}
    from sklearn.preprocessing import StandardScaler
    for method in ['baseline','RAMP']:
        a=np.load(REFERENCE/f'{method}_SHARED9.npy')
        x=np.load(REFERENCE/f'{method}_STANDARDIZED.npy')
        u=np.load(REFERENCE/f'{method}_UMAP.npy')
        assert a.shape==x.shape==(84000,9) and u.shape==(84000,2)
        assert np.isfinite(a).all() and np.isfinite(x).all() and np.isfinite(u).all()
        scale=StandardScaler().fit(a)
        err=float(np.abs(scale.transform(a)-x).max())
        assert err<1e-12
        result[f'{method}_standardization_error']=err
    if write:rt.save_json(BASE/'DATA_VERIFICATION.json',result)
    print('Data verification PASS: 84,000 cells, 84 patients, 9 markers, intact row order.',flush=True)
    return design,md,raw


def reproduce():
    design,md,raw=verify()
    OUT.mkdir(exist_ok=True)
    lock=(OUT/'RUNNING.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'REPRODUCTION.json').exists():
        raise SystemExit('Completed reproduction exists; use verify or plot to inspect it. Existing results are preserved.')
    sys.path.insert(0,str(BASE/'vendor'))
    from ramp import ramp_ver2 as ramp
    from sklearn.preprocessing import StandardScaler
    import umap
    assert ramp.__version__=='2.3.0'
    begin=time.time();captured={};original=ramp.apply_monotone_profiles
    def capture(b,m,p):
        captured.update({key:np.asarray(value).copy() for key,value in p.items()})
        return original(b,m,p)
    rt.save_json(OUT/'RUN_STATUS.json',{'phase':'fitting RAMP','updated_utc':rt.now()})
    print('Reproducing canonical RAMP 2.3.0 from packaged RAW_SHARED9...',flush=True)
    ramp.apply_monotone_profiles=capture
    try:
        r=ramp.fit_transform(raw,md[['dataset','unit_key']],design['markers'],datasets=design['datasets'],family=design['family'])
    finally:ramp.apply_monotone_profiles=original
    assert r.strict_rank_inversions==0
    checks={}
    for method,a in [('baseline',r.baseline),('RAMP',r.aligned)]:
        expected=np.load(REFERENCE/f'{method}_SHARED9.npy')
        err=float(np.abs(a-expected).max());assert err<1e-10
        checks[f'{method}_matrix_max_error']=err
        np.save(OUT/f'{method}_SHARED9.npy',a)
    for name in ['coupling_audit','strength_audit','evidence_audit','reference_weight_audit','profile_audit','algorithm_audit']:
        getattr(r,name).to_csv(OUT/f'{name.upper()}.csv',index=False)
    profiles={f'{unit}::{j}':p for (unit,j),p in captured.items()}
    with np.load(REFERENCE/'REFERENCE_PROFILES.npz') as expected:
        assert set(profiles)==set(expected.files)
        profile_error=max(float(np.abs(profiles[k]-expected[k]).max()) for k in profiles)
    assert profile_error<1e-10
    checks['profile_max_error']=profile_error
    checks['n_profiles']=len(profiles)
    np.savez_compressed(OUT/'REFERENCE_PROFILES.npz',**profiles)
    # Recover and save the exact per-reference-patient baseline transform for future projection.
    parameters=[]
    for unit,idx in md.groupby('unit_key',sort=True).indices.items():
        for j,marker in enumerate(design['markers']):
            v=raw[idx,j]
            shift=-float(v.min())+.01 if v.min()<0 else 0.
            shifted=v+shift;median=float(np.median(shifted))
            cofactor=max(float(shifted.mean()) if median==0 else median,1e-6)
            t=np.arcsinh(shifted/cofactor);scale=max(float(t.std()),1e-9)
            offset=float(np.median(t-scale*r.baseline[idx,j]))
            err=float(np.abs((t-offset)/scale-r.baseline[idx,j]).max());assert err<1e-9
            parameters.append({'unit_key':unit,'marker_index':j,'marker':marker,'shift':shift,
                               'cofactor':cofactor,'scale':scale,'offset':offset,'reconstruction_error':err})
    pd.DataFrame(parameters).to_csv(OUT/'BASELINE_PARAMETERS.csv',index=False)
    print(f'Alignment and {len(profiles)} profiles match frozen reference; {time.time()-begin:.1f}s.',flush=True)
    for method in ['baseline','RAMP']:
        a=np.load(OUT/f'{method}_SHARED9.npy')
        scale=StandardScaler().fit(a);x=scale.transform(a)
        np.save(OUT/f'{method}_STANDARDIZED.npy',x)
        config=json.loads((REFERENCE/f'{method}_UMAP.json').read_text())
        rt.save_json(OUT/'RUN_STATUS.json',{'phase':f'{method} UMAP','updated_utc':rt.now()})
        print(f'Reproducing {method} UMAP...',flush=True)
        u=umap.UMAP(**config['parameters']).fit_transform(x)
        assert u.shape==(84000,2) and np.isfinite(u).all()
        error=float(np.abs(u-np.load(REFERENCE/f'{method}_UMAP.npy')).max())
        checks[f'{method}_UMAP_max_error']=error
        checks[f'{method}_UMAP_exact_match']=bool(np.array_equal(u,np.load(REFERENCE/f'{method}_UMAP.npy')))
        np.save(OUT/f'{method}_UMAP.npy',u)
        rt.save_json(OUT/f'{method}_UMAP.json',{'parameters':config['parameters'],
            'scale_mean':scale.mean_.tolist(),'scale_sd':scale.scale_.tolist(),
            'preprocessing':config['preprocessing'],'max_coordinate_error_vs_reference':error})
        print(f'{method} UMAP max coordinate difference: {error:.6g}',flush=True)
    from plot_umap import render
    render(OUT,md)
    verify(write=False)
    umap_match=all(checks[f'{m}_UMAP_max_error']<1e-5 for m in ['baseline','RAMP'])
    rt.save_json(OUT/'REPRODUCTION.json',{'status':'PASS' if umap_match else 'ALIGNMENT_PASS_UMAP_DIFFERS',
        'time_utc':rt.now(),'seconds':time.time()-begin,'strict_rank_inversions':0,**checks,
        'baseline_parameters_max_error':max(p['reconstruction_error'] for p in parameters),
        'matrix_row_order':'identical to data/CELL_METADATA.csv.gz','canonical_input_metadata':['dataset','unit_key'],
        'reference_files_unchanged':True,'visual_review':'pending',
        'script_sha256':{p.name:rt.sha(p) for p in (BASE/'scripts').glob('*.py')}})
    rt.save_json(OUT/'RUN_STATUS.json',{'phase':'complete','updated_utc':rt.now()})
    print(json.dumps(checks,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['verify','reproduce','plot'],nargs='?',default='verify')
    args=parser.parse_args()
    if args.command=='verify':verify()
    elif args.command=='reproduce':reproduce()
    else:
        _,md,_=verify()
        if not (OUT/'RAMP_UMAP.npy').exists():raise SystemExit('Run reproduce first; the canonical plots are already in reference/figures/.')
        from plot_umap import render
        render(OUT,md)


if __name__=='__main__':main()
