"""Evaluate saved marker matrices without touching source data or fitted outputs."""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import os
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '1'
os.environ['MPLCONFIGDIR'] = str(HERE/'.cache')
os.environ['MPLBACKEND'] = 'Agg'
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance, rankdata
from sklearn.metrics import roc_curve, roc_auc_score

CASES = ['bone_marrow', 'tnbc', 'lung_cancer', 'melanoma']
METHODS = ['Raw', 'Harmony-direct', 'cyCombine', 'CytoNorm', 'RAMP']
AMBIGUOUS = {'Non-joint', 'Mix', 'Unclassified / technical', 'Unannotated', 'nan'}
T_LABELS = {'CD4 T', 'CD8 T', 'CD8− T', 'Other T / CD8−', 'Other T / NKT'}
NEG_CD8 = {'CD4 T', 'CD8− T', 'Other T / CD8−'}
PATHS = {
    'Raw': 'ramp/alignment/input/raw.npy',
    'Harmony-direct': 'harmony/source_scale/alignment/data/harmony_adjusted_marker_coordinates.csv.gz',
    'cyCombine': 'cycombine/alignment/data/corrected_expression.csv.gz',
    'CytoNorm': 'cytonorm/alignment/data/corrected_expression.csv.gz',
    'RAMP': 'ramp/alignment/data/ramp_aligned.npy',
}


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n')


def save(frame, path):
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    frame.to_csv(path, index=False)


def load_case(case):
    base = ROOT/'benchmark_results'/case
    sources = {}
    def track(rel):
        p = base/rel
        sources[str(p)] = sha(p)
        return p
    meta = pd.read_csv(track('ramp/alignment/input/cell_metadata.csv.gz'), low_memory=False)
    assert meta.cell_id.is_unique
    meta['unit_key'] = meta.unit_key.astype(str)
    meta['dataset'] = meta.dataset.astype(str)
    if 'row_id' in meta:
        assert np.array_equal(meta.row_id.to_numpy(), np.arange(len(meta)))
    label_col = 'celltype_joint_evaluation' if 'celltype_joint_evaluation' in meta else 'celltype'
    meta['eval_label'] = meta[label_col].fillna('Unannotated').astype(str)
    markers = track('ramp/alignment/input/markers.txt').read_text().splitlines()
    ids = meta.cell_id.astype(str).to_numpy()
    matrices = {}
    for method, rel in PATHS.items():
        p = track(rel)
        if p.suffix == '.npy':
            arr = np.asarray(np.load(p, mmap_mode='r'), dtype=np.float64)
        else:
            f = pd.read_csv(p, usecols=['cell_id']+markers, dtype={'cell_id': str}, low_memory=False)
            assert f.cell_id.is_unique and len(f) == len(meta)
            assert set(f.cell_id) == set(ids)
            arr = f.set_index('cell_id').loc[ids, markers].to_numpy(dtype=np.float64)
        assert arr.shape == (len(meta), len(markers)) and np.isfinite(arr).all(), (case, method)
        matrices[method] = arr
    return meta, markers, matrices, sources, label_col


def tasks_for(meta, markers):
    labels = meta.eval_label.to_numpy()
    clear = ~np.isin(labels, list(AMBIGUOUS))
    out = {}
    if 'CD3' in markers:
        out['CD3'] = (clear, np.isin(labels, list(T_LABELS)))
    if 'CD8' in markers:
        out['CD8'] = (np.isin(labels, ['CD8 T']+list(NEG_CD8)), labels == 'CD8 T')
    if 'CD4' in markers:
        out['CD4'] = (np.isin(labels, ['CD4 T', 'CD8 T']), labels == 'CD4 T')
    return out


def fit_gate(values, positive, unit):
    weights = np.zeros(len(values))
    for label in [False, True]:
        selected = positive == label
        levels = np.unique(unit[selected])
        for u in levels:
            mask = selected & (unit == u)
            weights[mask] = .5/(len(levels)*mask.sum())
    assert np.isclose(weights.sum(), 1)
    fpr, tpr, thresholds = roc_curve(positive, values, sample_weight=weights, drop_intermediate=False)
    ba = (tpr+1-fpr)/2
    opts = np.flatnonzero(np.isclose(ba, ba.max(), rtol=0, atol=1e-12))
    # ROC thresholds descend: first tied optimum has the highest specificity.
    i = int(opts[0])
    return dict(threshold=float(thresholds[i]), source_balanced_accuracy=float(ba[i]),
                source_sensitivity=float(tpr[i]), source_specificity=float(1-fpr[i]),
                source_auc=float(roc_auc_score(positive, values, sample_weight=weights)))


def score(values, positive, threshold):
    pred = values >= threshold
    tp, tn = int((pred & positive).sum()), int((~pred & ~positive).sum())
    fp, fn = int((pred & ~positive).sum()), int((~pred & positive).sum())
    sensitivity = tp/(tp+fn) if tp+fn else np.nan
    specificity = tn/(tn+fp) if tn+fp else np.nan
    f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
    auc = roc_auc_score(positive, values) if positive.any() and (~positive).any() else np.nan
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, sensitivity=sensitivity, specificity=specificity,
                balanced_accuracy=(sensitivity+specificity)/2, f1=f1,
                precision=tp/(tp+fp) if tp+fp else 0., auc=auc,
                predicted_positive_fraction=float(pred.mean()), reference_positive_fraction=float(positive.mean()))


def evaluate_gates(case, meta, markers, matrices, output):
    eligibility, gates, scores, aucs = [], [], [], []
    datasets, units = meta.dataset.to_numpy(), meta.unit_key.to_numpy()
    tasks = tasks_for(meta, markers)
    for marker, (included, positive) in tasks.items():
        col = markers.index(marker)
        unit_rows, eligible_units = {}, {}
        for dataset in sorted(meta.dataset.unique()):
            eligible_units[dataset] = []
            for unit in sorted(meta.loc[meta.dataset.eq(dataset), 'unit_key'].unique()):
                rows = np.flatnonzero(included & (datasets == dataset) & (units == unit))
                pos = int(positive[rows].sum()); neg = len(rows)-pos
                valid = pos >= 10 and neg >= 10
                eligibility.append(dict(benchmark=case, marker=marker, dataset=dataset, unit_key=unit,
                    n_positive=pos, n_negative=neg, primary_eligible=valid))
                unit_rows[dataset, unit] = rows
                if valid:
                    eligible_units[dataset].append(unit)
                if pos and neg:
                    raw_auc = float(roc_auc_score(positive[rows], matrices['Raw'][rows, col]))
                    for method, matrix in matrices.items():
                        auc = float(roc_auc_score(positive[rows], matrix[rows, col]))
                        aucs.append(dict(benchmark=case, marker=marker, dataset=dataset, unit_key=unit,
                            method=method, n_positive=pos, n_negative=neg, primary_eligible=valid,
                            auc=auc, raw_auc=raw_auc, auc_change=auc-raw_auc,
                            unique_values=len(np.unique(matrix[rows, col]))))
        for source in sorted(meta.dataset.unique()):
            train = np.concatenate([unit_rows[source, u] for u in eligible_units[source]]) if eligible_units[source] else np.array([], dtype=int)
            npos = int(positive[train].sum()); nneg = len(train)-npos
            if min(npos, nneg) < 50:
                for method in METHODS:
                    gates.append(dict(benchmark=case, marker=marker, source_dataset=source, method=method,
                        status='INSUFFICIENT_SOURCE', n_positive=npos, n_negative=nneg))
                continue
            for method, matrix in matrices.items():
                fitted = fit_gate(matrix[train, col], positive[train], units[train])
                gates.append(dict(benchmark=case, marker=marker, source_dataset=source, method=method,
                    status='OK', n_positive=npos, n_negative=nneg, n_units=len(eligible_units[source]), **fitted))
                for target in sorted(set(datasets)-{source}):
                    for unit in sorted(meta.loc[meta.dataset.eq(target), 'unit_key'].unique()):
                        rows = unit_rows[target, unit]
                        if not len(rows):
                            continue
                        pos = int(positive[rows].sum()); neg = len(rows)-pos
                        scores.append(dict(benchmark=case, marker=marker, method=method,
                            source_dataset=source, target_dataset=target, target_unit=unit,
                            n_positive=pos, n_negative=neg, primary_eligible=pos >= 10 and neg >= 10,
                            threshold=fitted['threshold'], **score(matrix[rows, col], positive[rows], fitted['threshold'])))
    save(eligibility, output/'gate_eligibility.csv')
    save(gates, output/'source_gates.csv')
    save(scores, output/'gate_target_unit_scores.csv')
    save(aucs, output/'within_unit_auc_preservation.csv')
    return pd.DataFrame(scores)


def weighted_quantile(values, weights, q):
    order = np.argsort(values, kind='stable')
    v, w = values[order], weights[order]
    cdf = (w.cumsum()-.5*w)/w.sum()
    return np.interp(q, cdf, v)


def balanced_allcell_weights(meta):
    weights = np.zeros(len(meta))
    platforms = meta.dataset.unique()
    for d in platforms:
        ds = meta[meta.dataset.eq(d)]
        for _, g in ds.groupby('unit_key'):
            weights[g.index] = 1/(len(platforms)*ds.unit_key.nunique()*len(g))
    assert np.isclose(weights.sum(), 1)
    return weights


def distribution_groups(meta):
    eligibility, groups = [], {}
    clear = meta[~meta.eval_label.isin(AMBIGUOUS)]
    present = clear.groupby('eval_label').dataset.nunique()
    for (label, dataset, unit), g in clear.groupby(['eval_label', 'dataset', 'unit_key']):
        valid = len(g) >= 30 and present[label] >= 2
        eligibility.append(dict(celltype=label, dataset=dataset, unit_key=unit,
            n_cells=len(g), primary_eligible=valid, label_n_platforms=int(present[label])))
        if valid:
            groups[label, dataset, unit] = g.index.to_numpy()
    pairs = []
    for label in sorted({k[0] for k in groups}):
        platforms = sorted({k[1] for k in groups if k[0] == label})
        for left, right in itertools.combinations(platforms, 2):
            ll = [k for k in groups if k[:2] == (label, left)]
            rr = [k for k in groups if k[:2] == (label, right)]
            for l, r in itertools.product(ll, rr):
                pairs.append((l, r))
    return eligibility, groups, pairs


def evaluate_distributions(case, meta, markers, matrices, output):
    eligibility, groups, pairs = distribution_groups(meta)
    save(eligibility, output/'distribution_eligibility.csv')
    weights = balanced_allcell_weights(meta)
    distances, scales, dispersions = [], [], []
    for method, matrix in matrices.items():
        print(f'{case}: EMD {method}, {len(pairs)} unit/type pairs x {len(markers)} markers', flush=True)
        for col, marker in enumerate(markers):
            low, high = weighted_quantile(matrix[:, col], weights, [.05, .95])
            span = float(high-low)
            valid_scale = span > 1e-12
            scales.append(dict(benchmark=case, method=method, marker=marker,
                q05=low, q95=high, span=span, valid_scale=valid_scale))
            values = {k: np.sort(matrix[rows, col]) for k, rows in groups.items()}
            for (label, dataset, unit), v in values.items():
                median = np.median(v); mad = float(np.median(np.abs(v-median)))
                dispersions.append(dict(benchmark=case, method=method, marker=marker, celltype=label,
                    dataset=dataset, unit_key=unit, n_cells=len(v), median=median, mad_native=mad,
                    mad_normalized=mad/span if valid_scale else np.nan))
            for left, right in pairs:
                a, b = values[left], values[right]
                distance = float(wasserstein_distance(a, b))
                distances.append(dict(benchmark=case, method=method, marker=marker, celltype=left[0],
                    platform_left=left[1], platform_right=right[1], unit_left=left[2], unit_right=right[2],
                    n_left=len(a), n_right=len(b), emd_native=distance,
                    emd_normalized=distance/span if valid_scale else np.nan, scale_span=span))
    save(scales, output/'marker_scales.csv')
    save(distances, output/'emd_unit_pairs.csv.gz')
    save(dispersions, output/'within_type_dispersion.csv.gz')


def run_case(case):
    start = time.time()
    output = HERE/'results'/case
    output.mkdir(parents=True, exist_ok=False)
    protocol_hash = sha(HERE/'PROTOCOL.md')
    meta, markers, matrices, sources, label_col = load_case(case)
    print(f'{case}: {len(meta)} cells, {meta.unit_key.nunique()} units, {len(markers)} markers', flush=True)
    labels = meta.groupby(['dataset', 'unit_key', 'eval_label']).size().rename('n_cells').reset_index()
    save(labels, output/'label_counts.csv')
    evaluate_gates(case, meta, markers, matrices, output)
    evaluate_distributions(case, meta, markers, matrices, output)
    assert all(sha(p) == h for p, h in sources.items())
    assert sha(HERE/'PROTOCOL.md') == protocol_hash
    dump(output/'COMPLETE.json', dict(status='PASS', benchmark=case, elapsed_seconds=time.time()-start,
        n_cells=len(meta), n_units=meta.unit_key.nunique(), markers=markers, label_column=label_col,
        protocol_sha256=protocol_hash, input_hashes=sources, inputs_unchanged=True,
        embedding_only_methods={'fastMNN': 'NOT_APPLICABLE', 'MaxFuse': 'NOT_APPLICABLE'},
        refitted_alignment=False, target_labels_used_to_fit_gate=False))
    print(f'{case}: complete in {time.time()-start:.1f}s', flush=True)


def checks():
    # Confirm threshold semantics, source class/unit weighting, and scale invariance.
    v = np.array([0., 0.1, .8, 1., .2, .3, .9, 1.1])
    y = np.array([False, False, True, True]*2)
    u = np.array(['a']*4+['b']*4)
    a = fit_gate(v, y, u)
    assert score(v, y, a['threshold'])['balanced_accuracy'] == 1
    b = fit_gate(v*7+3, y, u)
    assert np.array_equal(v >= a['threshold'], v*7+3 >= b['threshold'])
    d = wasserstein_distance(v[:4], v[4:])
    assert np.isclose(wasserstein_distance(v[:4]*7+3, v[4:]*7+3), d*7)
    print('Threshold and distance checks passed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=CASES)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.check:
        checks()
    if args.case:
        run_case(args.case)
