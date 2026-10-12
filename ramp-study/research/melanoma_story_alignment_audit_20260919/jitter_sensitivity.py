import sys; sys.path.insert(0,'/Users/qxu6/Documents/New project/cross_platform_trainability/src/lomo')
import numpy as np, pandas as pd, ramp_run
FX=ramp_run.FINAL_XU
sys.path.insert(0,str(FX/'algorithm'))
from ramp import ramp_ver2 as impl
out={}
for task in ['melanoma_story','lung_cancer']:
    s=ramp_run.spec(task)
    raw0=np.load(s['raw']).astype(np.float64); md=pd.read_csv(s['meta'],low_memory=False)[['dataset','unit_key']]
    res=[]
    for raw in [raw0, raw0*(1+1e-12)]:
        saved=(impl.PAIR_FIT_MINIMUM,impl.PAIR_FIT_PER_UNIT,impl.PAIR_FIT_MAXIMUM)
        try:
            if s['pair_fit']:
                impl.PAIR_FIT_MINIMUM=s['pair_fit']['minimum']; impl.PAIR_FIT_PER_UNIT=s['pair_fit']['per_unit']; impl.PAIR_FIT_MAXIMUM=s['pair_fit']['maximum']
            r=impl.fit_transform(raw,md.copy(),s['markers'],datasets=s['datasets'],family=s['family'])
        finally:
            impl.PAIR_FIT_MINIMUM,impl.PAIR_FIT_PER_UNIT,impl.PAIR_FIT_MAXIMUM=saved
        res.append(np.asarray(r.aligned,float))
    d=np.abs(res[0]-res[1])
    print(f'{task:16s} run-to-run under 1e-12 jitter: max={d.max():.6f} mean={d.mean():.3e} frac>1e-5={(d>1e-5).mean():.4f}',flush=True)
