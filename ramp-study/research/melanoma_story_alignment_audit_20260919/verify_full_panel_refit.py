"""Refit RAMP at the full panel and compare with the published alignment.

If the recovered family / dataset order / pair-fit policy are right, the refit
reproduces the recorded matrix; anything else means the LOMO runs would be
measuring a different algorithm than the one the manuscript reports.
"""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, ramp_run

FX = ramp_run.FINAL_XU
REF = {
 'bone_marrow':    FX/'benchmark_results/bone_marrow/ramp/alignment/data/ramp_aligned.npy',
 'tnbc':           FX/'benchmark_results/tnbc/ramp/alignment/data/ramp_aligned.npy',
 'melanoma':       FX/'benchmark_results/melanoma/ramp/alignment/data/ramp_aligned.npy',
 'lung_cancer':    FX/'benchmark_results/lung_cancer/ramp/alignment/data/ramp_aligned.npy',
 'lung_story':     FX/'story_lung_HLA_ABC/Comparison/input/ramp.npy',
 'melanoma_story': FX/'figure_melanoma/analysis/alignment/output/ramp_aligned.npy',
 'hnscc_story':    FX/'figure_HNSCC/alignment/output/aligned_ramp_ver2.float32.npy',
}


def check(task):
    s = ramp_run.spec(task)
    raw = np.load(s['raw'], allow_pickle=False).astype(np.float64)
    import pandas as pd
    meta = pd.read_csv(s['meta'], low_memory=False)[['dataset', 'unit_key']]
    sys.path.insert(0, str(FX / 'algorithm'))
    from ramp import ramp_ver2 as impl
    saved = (impl.PAIR_FIT_MINIMUM, impl.PAIR_FIT_PER_UNIT, impl.PAIR_FIT_MAXIMUM)
    t0 = time.time()
    try:
        if s['pair_fit']:
            impl.PAIR_FIT_MINIMUM = s['pair_fit']['minimum']
            impl.PAIR_FIT_PER_UNIT = s['pair_fit']['per_unit']
            impl.PAIR_FIT_MAXIMUM = s['pair_fit']['maximum']
        r = impl.fit_transform(raw, meta.copy(), s['markers'],
                               datasets=s['datasets'], family=s['family'])
    finally:
        impl.PAIR_FIT_MINIMUM, impl.PAIR_FIT_PER_UNIT, impl.PAIR_FIT_MAXIMUM = saved
    elapsed = time.time() - t0
    ref = np.load(REF[task], allow_pickle=False)
    got = np.asarray(r.aligned, dtype=np.float64)
    err = float(np.max(np.abs(ref.astype(np.float64) - got)))
    return dict(task=task, seconds=round(elapsed, 1), shape=list(got.shape),
                ref_dtype=str(ref.dtype), max_abs_diff=err,
                inversions=int(r.strict_rank_inversions))


if __name__ == '__main__':
    out = check(sys.argv[1])
    print(json.dumps(out), flush=True)
