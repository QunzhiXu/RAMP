"""Refit canonical RAMP 2.3.0 with one marker removed.

Each task is refitted exactly as the benchmark recorded it: same source-scale
matrix, same two fitting columns, same dataset order, same `family` string
(which seeds RAMP's role assignment, so it is not interchangeable), and the
same pair-fit policy.  The only change is the marker that is held out.
"""
import json, os, sys
from pathlib import Path

os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '1')
import numpy as np, pandas as pd

FINAL_XU = Path('/Users/qxu6/Documents/New project/final_xu')
STORY_ROOT = {'lung_story': 'story_lung_HLA_ABC', 'melanoma_story': 'figure_melanoma',
              'hnscc_story': 'figure_HNSCC'}
# Recovered from each task's ramp_manifest.json / DESIGN.json / coupling_audit.csv.
# `pair_fit` is None wherever the run used the module defaults (4000/100/8000).
FAMILY = {
    'bone_marrow': 'bm_prediction_story_observed14',
    'tnbc': 'tnbc_tietscher1_meyer5_coverage_guided_seed20260960_shiao5_uniform4k_seed20260827',
    'melanoma': 'melanoma_metadata_stage3_20k_each_20260827',
    'lung_cancer': 'lung_alignment_nested_10x3_20k_joint8_current_ramp',
    'lung_story': 'lung_zheng_senosain_cords_28each_common9_seed20260909',
    'melanoma_story': 'melanoma_full_cohort_survival_projection',
    'hnscc_story': 'hnscc_complete_os_pt74_locked_tigit_cd39_stability',
}
PAIR_FIT = {'lung_cancer': dict(minimum=7_000, per_unit=0, maximum=7_000)}


def spec(task):
    if task in STORY_ROOT:
        base = FINAL_XU / STORY_ROOT[task] / 'Comparison' / 'input'
        datasets = json.loads((base.parent.parent / 'Comparison' / 'run_config.json')
                              .read_text())['datasets']
    else:
        root = FINAL_XU / 'benchmark_results' / task / 'ramp' / 'alignment'
        base = root / 'input'
        datasets = json.loads((root / 'data' / 'ramp_manifest.json').read_text())['datasets']
    markers = [m for m in (base / 'markers.txt').read_text().split() if m]
    return dict(raw=base / 'raw.npy', meta=base / 'fit_metadata.csv.gz',
                markers=markers, datasets=datasets, family=FAMILY[task],
                pair_fit=PAIR_FIT.get(task))


def fit(task, drop):
    s = spec(task)
    keep = [m for m in s['markers'] if m != drop]
    if len(keep) != len(s['markers']) - 1:
        raise ValueError(f'{drop} not among {task} markers')
    raw = np.load(s['raw'], allow_pickle=False).astype(np.float64)
    raw = raw[:, [s['markers'].index(m) for m in keep]]
    meta = pd.read_csv(s['meta'], low_memory=False)[['dataset', 'unit_key']]

    if str(FINAL_XU / 'algorithm') not in sys.path:
        sys.path.insert(0, str(FINAL_XU / 'algorithm'))
    from ramp import ramp_ver2 as impl
    assert impl.__version__ == '2.3.0' and int(impl.SEED) == 42

    saved = (impl.PAIR_FIT_MINIMUM, impl.PAIR_FIT_PER_UNIT, impl.PAIR_FIT_MAXIMUM)
    try:
        if s['pair_fit']:
            impl.PAIR_FIT_MINIMUM = s['pair_fit']['minimum']
            impl.PAIR_FIT_PER_UNIT = s['pair_fit']['per_unit']
            impl.PAIR_FIT_MAXIMUM = s['pair_fit']['maximum']
        r = impl.fit_transform(raw, meta.copy(), keep,
                               datasets=s['datasets'], family=s['family'])
    finally:
        impl.PAIR_FIT_MINIMUM, impl.PAIR_FIT_PER_UNIT, impl.PAIR_FIT_MAXIMUM = saved

    aligned = np.asarray(r.aligned, dtype=np.float64)
    baseline = np.asarray(r.baseline, dtype=np.float64)
    if int(r.strict_rank_inversions) != 0:
        raise AssertionError('RAMP reported a strict within-unit rank inversion')
    return keep, aligned, baseline
