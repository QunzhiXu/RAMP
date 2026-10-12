"""Exact unequal-input pair reversal fractions on the four frozen benchmarks.

Read-only inputs; all new output and compilation caches stay beside this script.
No cell-pair subsampling. Fenwick counting costs O(n log n), not O(n^2).
"""
from pathlib import Path
import os
import json
import hashlib
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
os.environ['NUMBA_CACHE_DIR'] = str(HERE / '.cache/numba')
import numpy as np
import pandas as pd
from numba import njit

CASES = ['bone_marrow', 'tnbc', 'melanoma', 'lung_cancer']
METHODS = ['Harmony-direct[source_scale]', 'Harmony-direct[fixed_arcsinh]',
           'cyCombine', 'CytoNorm', 'RAMP']
TOL = 1e-12
READS = {}


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def track(p, expected=None):
    p = Path(p).resolve()
    digest = sha(p)
    if expected is not None:
        assert digest == expected, (p, digest, expected)
    READS[str(p)] = digest
    return p


def read_json(p):
    return json.loads(track(p).read_text())


@njit(cache=True)
def count_sorted(x, y, unique_y, ranks, tol):
    """Only insert an input-tie group after all its queries are complete."""
    tree = np.zeros(len(unique_y) + 1, dtype=np.int64)
    seen = np.int64(0)
    inverted = np.int64(0)
    comparable = np.int64(0)
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and x[end] == x[start]:
            end += 1
        comparable += seen * (end - start)
        for i in range(start, end):
            # Match the old audit's subtraction > tol exactly, including at
            # the floating-point boundary (not an approximation y + tol).
            lo, hi = 0, len(unique_y)
            while lo < hi:
                mid = (lo + hi) // 2
                if unique_y[mid] - y[i] <= tol:
                    lo = mid + 1
                else:
                    hi = mid
            k = lo
            prefix = np.int64(0)
            while k > 0:
                prefix += tree[k]
                k -= k & -k
            inverted += seen - prefix
        for i in range(start, end):
            k = ranks[i] + 1
            while k < len(tree):
                tree[k] += 1
                k += k & -k
        seen += end - start
        start = end
    return inverted, comparable


def count_pairs(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    assert len(x) == len(y) and np.isfinite(x).all() and np.isfinite(y).all()
    order = np.argsort(x, kind='stable')
    unique, rank = np.unique(y, return_inverse=True)
    inv, total = count_sorted(x[order], y[order], unique, rank[order], TOL)
    return int(inv), int(total)


def brute_force(x, y):
    x, y = np.asarray(x), np.asarray(y)
    comparable = x[:, None] < x[None, :]
    return (int(np.count_nonzero(comparable & ((y[:, None] - y[None, :]) > TOL))),
            int(np.count_nonzero(comparable)))


def verify_counter():
    rng = np.random.default_rng(20260914)
    tests = [([1, 2, 3], [3, 2, 1]), ([1, 1, 1], [3, 2, 1]),
             ([1, 2, 3], [1, 1, 1]), ([1, 1, 2], [3, 1, 2]),
             ([1, 2], [1e-12, 0]), ([1, 2], [np.nextafter(1e-12, np.inf), 0]),
             ([1, 2], [np.nextafter(1e-12, 0), 0])]
    for _ in range(120):
        n = int(rng.integers(2, 100))
        tests.append((rng.integers(0, 10, n), rng.integers(0, 10, n).astype(float)))
        tests.append((rng.normal(size=n), rng.normal(size=n)))
        tests.append((rng.integers(0, 10, n), rng.normal(size=n) * TOL))
    for x, y in tests:
        assert count_pairs(x, y) == brute_force(x, y), (x, y)
    return len(tests)


def read_matrix(p, ids, markers, expected, round_trip=False):
    # Match the prior audit's parser for each method, preserving its numerical
    # tie definition. CytoNorm used round_trip; other old audits used default.
    opts = {'float_precision': 'round_trip'} if round_trip else {}
    d = pd.read_csv(track(p, expected), low_memory=False, dtype={'cell_id': str}, **opts)
    assert np.array_equal(d.cell_id.to_numpy(), ids), p
    z = d[markers].to_numpy(np.float64)
    assert z.shape == (len(ids), len(markers)) and np.isfinite(z).all(), p
    return z


def load_methods(case, ids, markers):
    base = ROOT / 'benchmark_results' / case
    for variant, name in [('source_scale', METHODS[0]), ('adtnorm_fixed_arcsinh', METHODS[1])]:
        root = base / 'harmony' / variant
        manifest = read_json(root / 'alignment/data/manifest.json')
        assert manifest['status'] == 'complete'
        op = root / 'alignment/data/harmony_adjusted_marker_coordinates.csv.gz'
        out_hash = {r['file']: r['sha256'] for r in manifest['output_files']}[op.name]
        yield name, read_matrix(root / 'preprocessing/data/combined_analysis_ready.csv.gz',
                               ids, markers, manifest['input']['sha256']), read_matrix(op, ids, markers, out_hash)
    root = base / 'cycombine'
    m = read_json(root / 'alignment/data/manifest.json')
    yield METHODS[2], read_matrix(root / 'preprocessing/data/combined_analysis_ready.csv.gz',
                                 ids, markers, m['inputs']['expression']['sha256']), read_matrix(
        root / 'alignment/data/corrected_expression.csv.gz', ids, markers,
        m['outputs']['corrected_expression_csv']['sha256'])
    root = base / 'cytonorm'
    m = read_json(root / 'COMPLETE.json')['outputs']
    a, b = 'preprocessing/analysis_ready.csv.gz', 'alignment/data/corrected_expression.csv.gz'
    yield METHODS[3], read_matrix(root / a, ids, markers, m[a], True), read_matrix(root / b, ids, markers, m[b], True)
    root = base / 'ramp/alignment'
    m = read_json(root / 'alignment_manifest.json')
    assert m['strict_rank_inversions'] == 0
    def ramp_array(stem):
        p = root / 'data' / f'{stem}_float64.npy'
        if not p.exists():
            p = root / 'data' / f'{stem}.npy'
        z = np.asarray(np.load(track(p, m['outputs'][p.name]), mmap_mode='r'), dtype=np.float64)
        assert z.shape == (len(ids), len(markers)) and np.isfinite(z).all()
        return z
    yield METHODS[4], ramp_array('kmeanslow_global'), ramp_array('ramp_aligned')


def main():
    started = time.time()
    tests = verify_counter()
    old = pd.read_csv(track(ROOT / 'Main Figure/Panel E/rank_integrity_unit_marker.csv.gz'))
    keys = ['case', 'dataset', 'unit_key', 'marker', 'method']
    old_flags = old.set_index(keys).inversion.to_dict()
    records = []
    subset_tests = 0
    for case in CASES:
        inp = ROOT / 'benchmark_results' / case / 'ramp/alignment/input'
        meta = pd.read_csv(track(inp / 'cell_metadata.csv.gz'), low_memory=False, dtype={'cell_id': str})
        ids = meta.cell_id.to_numpy()
        assert not meta.cell_id.duplicated().any()
        markers = track(inp / 'markers.txt').read_text().strip().splitlines()
        groups = list(meta.groupby('unit_key', sort=True).indices.items())
        for method, before, after in load_methods(case, ids, markers):
            for unit, rows in groups:
                datasets = meta.iloc[rows].dataset.unique()
                assert len(datasets) == 1
                for j, marker in enumerate(markers):
                    x, y = before[rows, j], after[rows, j]
                    inverted, comparable = count_pairs(x, y)
                    assert comparable > 0 and 0 <= inverted <= comparable
                    key = (case, datasets[0], unit, marker, method)
                    assert (inverted > 0) == old_flags[key], key
                    records.append(dict(zip(keys, key), n_cells=len(rows),
                        input_tied_pairs=len(rows) * (len(rows) - 1) // 2 - comparable,
                        comparable_pairs=comparable, reversed_pairs=inverted,
                        fraction=inverted / comparable, percent=100 * inverted / comparable))
            rr = groups[0][1][:100]
            assert count_pairs(before[rr, 0], after[rr, 0]) == brute_force(before[rr, 0], after[rr, 0])
            subset_tests += 1
            print(f'{case}: {method} complete ({len(records)} combinations; {time.time()-started:.1f}s)', flush=True)
        pd.DataFrame(records).to_csv(HERE / 'pair_inversion_fractions.csv.gz', index=False)
    result = pd.DataFrame(records)
    assert set(map(tuple, result[keys].to_numpy())) == set(old_flags)
    assert len(result) == 5755
    summaries = []
    for method in METHODS:
        d = result[result.method.eq(method)]
        assert len(d) == 1151
        q = d.percent.quantile([.25, .5, .75])
        summaries.append(dict(method=method, n_combinations=len(d),
            n_with_any_reversal=int((d.reversed_pairs > 0).sum()),
            mean_percent=d.percent.mean(), q1_percent=q.loc[.25], median_percent=q.loc[.5],
            q3_percent=q.loc[.75], max_percent=d.percent.max(),
            pooled_pair_percent=100 * d.reversed_pairs.sum() / d.comparable_pairs.sum()))
    pd.DataFrame(summaries).to_csv(HERE / 'pair_inversion_summary.csv', index=False)
    result.groupby(['case', 'method']).percent.agg(['count', 'mean', 'median', 'min', 'max']).to_csv(
        HERE / 'pair_inversion_by_benchmark.csv')
    assert all(sha(p) == digest for p, digest in READS.items())
    manifest = dict(methods=METHODS, cases=CASES, combinations_per_method=1151,
        n_units=result[['case','unit_key']].drop_duplicates().shape[0],
        definition='100 * # unordered cell pairs with x_i < x_j and y_i - y_j > 1e-12 / # unordered cell pairs with x_i < x_j',
        input='Each method actual frozen preprocessed input compared with its frozen corrected marker values.',
        ties='Exact input ties excluded; output ties and reversals <= 1e-12 count as non-reversed comparable pairs.',
        aggregation='One observation per sample x marker; pooled combination distribution across four benchmarks. Not pair-weighted or benchmark-equal.',
        algorithm='Exact Fenwick tree O(n log n), int64 pair counts, no pair sampling.',
        verification=dict(synthetic_brute_force_tests=tests, real_subset_brute_force_tests=subset_tests,
                          all_5755_any_reversal_flags_match_previous_audit=True, all_inputs_unchanged=True),
        excluded_methods='Embedding-only fastMNN and MaxFuse have no comparable single-marker output axis.',
        runtime_seconds=time.time()-started, input_sha256=READS, script_sha256=sha(__file__))
    (HERE / 'pair_inversion_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
