"""Subprocess boundary pins BLAS threads before RAMP/scikit-learn imports."""
import sys
sys.dont_write_bytecode=True
import json
from pathlib import Path
import numpy as np
import pandas as pd
work=Path(sys.argv[1]).resolve()
request=json.loads((work/'request.json').read_text())
sys.path.insert(0,request['algorithm'])
from ramp import ramp_ver2 as impl
assert impl.__version__=='2.3.0' and impl.SEED==42
result=impl.fit_transform(np.load(work/'input.npy'),pd.read_csv(work/'identity.csv',dtype=str)[['dataset','unit_key']],
    request['markers'],datasets=request['datasets'],family=request['family'])
assert result.strict_rank_inversions==0
np.save(work/'aligned.npy',np.asarray(result.aligned,dtype=np.float64))
np.save(work/'kmeanslow.npy',np.asarray(result.baseline,dtype=np.float64))
(work/'runtime.json').write_text(json.dumps(dict(version=impl.__version__,seed=impl.SEED,
    family=request['family'],strict_rank_inversions=int(result.strict_rank_inversions),
    pair_fit_minimum=impl.PAIR_FIT_MINIMUM,pair_fit_per_unit=impl.PAIR_FIT_PER_UNIT,
    pair_fit_maximum=impl.PAIR_FIT_MAXIMUM),indent=2)+'\n')
