"""One isolated canonical fit. No historical output or evaluation labels are read."""
import argparse
import importlib
from importlib.metadata import version
import os
from pathlib import Path
import platform
import sys

import safe_io as io

VERSIONS={'numpy':'2.4.0','pandas':'2.3.3','scipy':'1.16.3','scikit-learn':'1.8.0','joblib':'1.5.3','threadpoolctl':'3.6.0'}
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS','LOKY_MAX_CPU_COUNT')


def environment():
    import numpy, pandas, scipy, sklearn
    from threadpoolctl import threadpool_info
    found={n:version(n) for n in VERSIONS}
    if found!=VERSIONS or platform.python_version()!='3.14.1':raise ValueError('Historical Python environment differs')
    if any(os.environ.get(k)!='1' for k in THREADS):raise ValueError('Thread policy differs')
    if os.environ.get('PYTHONDONTWRITEBYTECODE')!='1':raise ValueError('Bytecode writes not disabled')
    return {'python':platform.python_version(),'versions':found,'platform':platform.platform(),
            'executable':str(Path(sys.executable).resolve()),'thread_limits':{k:os.environ[k] for k in THREADS},
            'threadpools':threadpool_info()}


def validate_input(raw, metadata, markers, spec):
    import numpy as np
    import pandas as pd
    if list(raw.shape)!=spec['shape'] or str(raw.dtype)!=spec['input_dtype'] or not np.isfinite(raw).all():raise ValueError('Input matrix differs')
    if list(metadata.columns)!=spec['fit_columns'] or len(metadata)!=len(raw):raise ValueError('Fit metadata columns/length differ')
    if markers!=spec['markers']:raise ValueError('Marker order differs')
    if metadata[['dataset','unit_key']].isna().any().any():raise ValueError('Missing fit identity')
    if list(pd.unique(metadata.dataset.astype(str)))!=spec['datasets']:raise ValueError('Dataset order differs')
    if metadata.groupby('dataset',sort=False).size().to_dict()!=spec['dataset_cells']:raise ValueError('Dataset counts differ')
    if metadata.groupby('dataset',sort=False).unit_key.nunique().to_dict()!=spec['dataset_units']:raise ValueError('Unit counts differ')
    if metadata.groupby('unit_key',sort=False).dataset.nunique().max()!=1:raise ValueError('Unit belongs to multiple datasets')
    if 'cell_id' in metadata and (metadata.cell_id.isna().any() or metadata.cell_id.duplicated().any()):raise ValueError('Cell identity differs')
    return metadata[['dataset','unit_key']].copy()


def independent_invariants(raw, baseline, aligned, metadata):
    import numpy as np
    inversions=tie_splits=0
    for rows in metadata.groupby('unit_key',sort=False).indices.values():
        for j in range(raw.shape[1]):
            order=np.argsort(raw[rows,j],kind='stable');x=raw[rows,j][order];y=aligned[rows,j][order]
            # The running maximum detects non-adjacent inversions as well.
            starts=np.r_[0,np.flatnonzero(np.diff(x)!=0)+1]
            mins=np.minimum.reduceat(y,starts);maxs=np.maximum.reduceat(y,starts)
            tie_splits+=int(np.count_nonzero(maxs-mins>1e-10))
            if len(starts)>1:inversions+=int(np.count_nonzero(mins[1:]<np.maximum.accumulate(maxs)[:-1]-1e-10))
    displacement=float(np.max(np.abs(aligned-baseline)))
    if inversions or tie_splits or displacement>0.75+1e-10:raise ValueError('Independent rank/tie/displacement check failed')
    return {'strict_inversions':inversions,'split_raw_ties':tie_splits,'maximum_displacement_float64':displacement,'tolerance':1e-10}


def calculate(raw, metadata, spec, ramp):
    """Only lung technical changes the three pair-fit constants; always restore."""
    import numpy as np
    names=('PAIR_FIT_MINIMUM','PAIR_FIT_PER_UNIT','PAIR_FIT_MAXIMUM')
    old=[getattr(ramp,n) for n in names];original=ramp.apply_monotone_profiles;captured={}
    if old!=[4000,100,8000]:raise ValueError('Canonical defaults changed')
    def capture(b,m,p):
        captured.update({k:np.asarray(v).copy() for k,v in p.items()})
        return original(b,m,p)
    try:
        for name,key in zip(names,('minimum','per_unit','maximum')):setattr(ramp,name,spec['pair_fit'][key])
        if spec['case']=='lung_story':ramp.apply_monotone_profiles=capture
        result=ramp.fit_transform(raw,metadata,spec['markers'],datasets=spec['datasets'],family=spec['family'])
    finally:
        for n,v in zip(names,old):setattr(ramp,n,v)
        ramp.apply_monotone_profiles=original
    return result,captured


def save_result(root, result, profiles, spec, metadata):
    import numpy as np
    output=Path(root)/'result';output.mkdir(exist_ok=False)
    for name,(attr,dtype) in spec['matrices'].items():
        with (output/name).open('xb') as f:np.save(f,np.asarray(getattr(result,attr),dtype=dtype),allow_pickle=False)
    for name,attr in spec['audits'].items():
        with (output/name).open('x') as f:getattr(result,attr).to_csv(f,index=False)
    if spec['case']=='lung_story':
        from frozen_maps import fit_maps
        with (output/'REFERENCE_PROFILES.npz').open('xb') as f:np.savez_compressed(f,**{f'{u}::{j}':p for (u,j),p in profiles.items()})
        rows=np.flatnonzero(metadata.dataset.eq('Cords IMC'))
        maps=fit_maps(result.baseline[rows],result.aligned[rows])
        with (output/'FROZEN_PLATFORM_MAPS.npz').open('xb') as f:np.savez_compressed(f,**{f'{axis}{j}':v for j,pair in enumerate(maps) for axis,v in zip(['x','y'],pair)})
    return io.files(output)


def execute(root):
    import numpy as np
    import pandas as pd
    import study
    root=Path(root).resolve(strict=True);req=io.read(root/'REQUEST.json')
    if req['run']!=str(root):raise ValueError('Run path differs from request')
    spec=study.verify_bundle(Path(req['bundle']))
    if spec['case']!=req['case'] or spec['reference_only']:raise ValueError('Invalid requested case')
    study.check_code(req['code_records']);io.check(req['bundle']+'/MANIFEST.json',req['bundle_manifest'])
    io.check(sys.executable,req['python'])
    study.check_runtime(root,spec)
    io.write(root/'STARTED.json',{'case':spec['case'],'pid':os.getpid()})
    runtime=root/'runtime';sys.path.insert(0,str(runtime/'vendor'));sys.path.insert(0,str(runtime/'helpers'))
    ramp=importlib.import_module('ramp.ramp_ver2')
    if Path(ramp.__file__).resolve()!=runtime/'vendor/ramp/ramp_ver2.py' or ramp.__version__!='2.3.0' or ramp.SEED!=42:raise ValueError('Wrong algorithm loaded')
    before=environment();io.write(root/'ENVIRONMENT_BEFORE.json',before)
    raw=np.load(runtime/'input/raw.npy',allow_pickle=False)
    md=pd.read_csv(runtime/'input/fit_metadata.csv.gz',low_memory=False)
    markers=(runtime/'input/markers.txt').read_text().splitlines()
    fitmd=validate_input(raw,md,markers,spec)
    result,profiles=calculate(raw,fitmd,spec,ramp)
    if result.strict_rank_inversions!=0:raise ValueError('Canonical rank inversion')
    invariant=independent_invariants(raw,result.baseline,result.aligned,fitmd)
    outputs=save_result(root,result,profiles,spec,fitmd)
    after=environment();io.write(root/'ENVIRONMENT_AFTER.json',after)
    if before!=after:raise ValueError('Runtime changed during fit')
    study.verify_bundle(Path(req['bundle']));study.check_runtime(root,spec);study.check_code(req['code_records'])
    io.check(sys.executable,req['python']);io.check(req['bundle']+'/MANIFEST.json',req['bundle_manifest'])
    io.write(root/'WORKER_RESULT.json',{'case':spec['case'],'family':spec['family'],'pair_fit':spec['pair_fit'],
             'metadata_columns_read_for_fitting':['dataset','unit_key'],'historical_results_read':False,
             'canonical_rank_inversions':int(result.strict_rank_inversions),'independent_invariants':invariant,
             'captured_profiles':len(profiles),'outputs':outputs})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path);p.add_argument('--environment',action='store_true');a=p.parse_args()
    if a.environment:
        import json
        print(json.dumps(environment(),sort_keys=True));return
    if a.run is None:p.error('--run required')
    execute(a.run)


if __name__=='__main__':main()
