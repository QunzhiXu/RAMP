"""Read-only checks of the frozen value-evaluation lineage and saved summaries.

This is not a gate refit or an end-to-end rerun. Reconstruction is checked in
memory from frozen fastMNN PCs/rotations; saved per-unit scores are reaggregated.
No legacy module is imported (several legacy files have import-time writes).
"""
import argparse
import ast
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_curve, roc_auc_score
import inventory as i

CATALOG_SHA = '0801f1e58fe063cfad29449ae41ecbd4705fabf32278ee536875893add97bfd9'
KEYS = ['case', 'dataset', 'unit_key', 'celltype', 'marker']
GATE_METRICS = ['balanced_accuracy', 'f1', 'sensitivity', 'specificity', 'precision', 'auc']
INITIAL = {'Harmony-direct[source_scale]', 'Harmony-direct[fixed_arcsinh]', 'cyCombine', 'CytoNorm', 'RAMP'}
FINAL_D = INITIAL - {'Harmony-direct[fixed_arcsinh]'} | {'fastMNN reconstructed'}
FINAL_F = {'Raw', 'Harmony-direct', 'cyCombine', 'CytoNorm', 'RAMP', 'fastMNN reconstructed'}


class Sources:
    def __init__(self, snapshot, source_root=None):
        self.snapshot = Path(snapshot).resolve(strict=True)
        i.verify(self.snapshot, CATALOG_SHA, source_root)
        self.catalog = i.read(self.snapshot / 'catalog.json')
        self.source_root = Path(source_root).resolve(strict=True) if source_root else None

    def path(self, name):
        if name not in self.catalog['sources']: raise ValueError('Unregistered source: ' + str(name))
        root = self.snapshot / 'sources' if name in self.catalog['copies'] else self.source_root
        i.require(root is not None, 'External source root required for array checks')
        path = i.bind(root, name)
        i.require(i.record(path) == self.catalog['sources'][name], 'Source changed')
        return path

    def csv(self, name, default_parser=False):
        return pd.read_csv(self.path(name), low_memory=False,
                           **({} if default_parser else {'float_precision': 'round_trip'}))

    def json(self, name): return i.read(self.path(name))


def pure_functions(path, names, namespace):
    """Compile only explicitly named, undecorated function bodies, not top-level code."""
    tree = ast.parse(Path(path).read_text())
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    i.require({n.name for n in selected} == set(names) and len(selected) == len(names), 'Function set differs')
    i.require(all(not n.decorator_list for n in selected), 'Unexpected decorator')
    ns = dict(namespace)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), ns)
    return {name: ns[name] for name in names}


def percentile_contract(s):
    original = s.csv(i.PERCENTILE + '/celltype_quartile_shifts_display.csv.gz')
    combined = s.csv(i.RECON + '/comparison_celltype_shifts.csv.gz')
    i.require(set(original.method) == INITIAL and set(combined.method) == INITIAL | {'fastMNN reconstructed'}, 'Wrong method set')
    key = KEYS + ['method']
    i.require(not combined.duplicated(key).any() and not original.duplicated(key).any(), 'Duplicate percentile row')
    common = set(map(tuple, original[original.method.eq('RAMP')][KEYS].to_numpy()))
    i.require(len(common) == 6173 and len(combined) == 37038, 'Percentile grid size differs')
    for method in set(combined.method):
        rows = combined[combined.method.eq(method)]
        i.require(set(map(tuple, rows[KEYS].to_numpy())) == common and rows.n_cells.ge(50).all()
                  and rows.included_in_plot.eq(True).all(), 'Percentile eligibility differs')
        if method in INITIAL:
            pd.testing.assert_frame_equal(rows[original.columns].sort_values(KEYS).reset_index(drop=True),
                original[original.method.eq(method)].sort_values(KEYS).reset_index(drop=True), check_exact=True)
        prefix = 'value_' if method == 'fastMNN reconstructed' else 'corrected_'
        for q in ('q25', 'median', 'q75'):
            np.testing.assert_allclose(rows['delta_' + q + '_pp'], rows[prefix + q] - rows['raw_' + q], rtol=0, atol=3e-14)
            np.testing.assert_array_equal(rows['abs_delta_' + q + '_pp'], abs(rows['delta_' + q + '_pp']))
    stats = s.csv('Main Figure/ridge_compact_20260924/median_shift_box_statistics.csv')
    i.require(set(stats.method) == FINAL_D and len(stats) == 5, 'Final Figure 1D methods differ')
    errors = []
    for row in stats.itertuples():
        vals = combined[combined.method.eq(row.method)].abs_delta_median_pp.to_numpy()
        actual = np.quantile(vals, [.25, .5, .75], method='linear')
        expected = np.array([row.q1, row.median, row.q3])
        errors.append(float(np.max(abs(actual - expected))))
    i.require(max(errors) < 1e-12, 'Final percentile boxes differ')
    n_units = len(combined[['case', 'unit_key']].drop_duplicates())
    i.require(n_units == 103, 'Eligible unit count differs')
    return {'initial_methods': sorted(INITIAL), 'final_methods': sorted(FINAL_D), 'combinations_per_method': len(common),
            'eligible_units': n_units, 'final_box_max_difference': max(errors), 'new_percentiles_computed': False}


def table_difference(new, old, keys, columns):
    i.require(not new.duplicated(keys).any() and not old.duplicated(keys).any(), 'Duplicate summary keys')
    a = new.sort_values(keys).reset_index(drop=True); b = old.sort_values(keys).reset_index(drop=True)
    i.require(a[keys].equals(b[keys]), 'Summary key grid differs')
    av = a[columns].to_numpy(float); bv = b[columns].to_numpy(float)
    i.require(np.isfinite(av).all() and np.isfinite(bv).all(), 'Nonfinite summary')
    error = float(np.max(abs(av - bv)))
    i.require(error <= 1e-12, 'Saved summary differs')
    return error


def gate_contract(s):
    frames = []; gates = []
    for case in i.CASES:
        old = s.csv(i.VALUE + '/results/' + case + '/gate_target_unit_scores.csv')
        ext = s.csv(i.RECON + '/value_evaluation/' + case + '/gate_target_unit_scores.csv')
        frames.extend([old, ext[ext.method.eq('fastMNN reconstructed')]])
        gates.append(s.csv(i.VALUE + '/results/' + case + '/source_gates.csv'))
    scores = pd.concat(frames, ignore_index=True); primary = scores[scores.primary_eligible].copy()
    keys = ['benchmark', 'marker', 'source_dataset', 'target_dataset', 'target_unit']
    i.require(set(primary.method) == FINAL_F and not primary.duplicated(keys + ['method']).any(), 'Gate method/row grid differs')
    i.require(primary.groupby(keys).method.nunique().eq(6).all(), 'Unmatched method rows')
    i.require(len(primary) == 458 * 6 and primary[GATE_METRICS].notna().all().all(), 'Gate coverage differs')
    np.testing.assert_allclose(primary.balanced_accuracy, (primary.sensitivity + primary.specificity) / 2, rtol=0, atol=2e-16)
    i.require((primary.tp + primary.fn).eq(primary.n_positive).all() and
              (primary.tn + primary.fp).eq(primary.n_negative).all(), 'Confusion counts differ')
    dk = ['benchmark', 'marker', 'method', 'source_dataset', 'target_dataset']; tk = ['benchmark', 'marker', 'method']
    directions = primary.groupby(dk, observed=True, dropna=False)[GATE_METRICS].mean().reset_index()
    tasks = directions.groupby(tk)[GATE_METRICS].mean().reset_index()
    counts = directions.groupby(tk).size().rename('n_directions').reset_index()
    i.require(len(tasks) == 66 and counts.n_directions.eq(6).all(), 'Not 11 tasks x six methods x six directions')
    cases = tasks.groupby(['benchmark', 'method'])[GATE_METRICS].mean().reset_index()
    overall = cases.groupby('method')[GATE_METRICS].mean().reset_index()
    summary = i.RECON + '/value_evaluation/summary/'
    errors = {name: table_difference(new, s.csv(summary + name + '.csv'), key, GATE_METRICS)
              for name, new, key in [('gate_direction_means', directions, dk), ('gate_task_means', tasks, tk),
                                    ('gate_case_means', cases, ['benchmark', 'method']), ('gate_overall_equal_case', overall, ['method'])]}
    # Final figure deliberately retains the original five-method tables after
    # checking the diagnostic extension differs only by parsing/aggregation ULPs.
    historical = 'Main Figure/Figure 2/seven_benchmarks_main23_20260914/Panel F/'
    original_plot = s.csv(historical + 'plotted_values.csv')
    extended = s.csv(summary + 'gate_task_means.csv').merge(counts, on=tk, validate='one_to_one')
    retained = extended[~extended.method.eq('fastMNN reconstructed')]
    errors['five_method_extension'] = table_difference(retained, original_plot, tk, GATE_METRICS)
    expected = pd.concat([original_plot, extended[extended.method.eq('fastMNN reconstructed')]], ignore_index=True)
    final = s.csv(i.FIGURE2 + '/Panel F/plotted_values.csv')
    errors['final_plot'] = table_difference(expected, final, tk, GATE_METRICS + ['n_directions'])
    i.require(errors['final_plot'] == 0, 'Final task values not preserved exactly')
    old_means = s.csv(historical + 'equal_benchmark_means.csv')
    new_means = s.csv(summary + 'gate_overall_equal_case.csv')
    expected_mean = pd.concat([old_means, new_means[new_means.method.eq('fastMNN reconstructed')][old_means.columns]])
    errors['final_equal_case_means'] = table_difference(expected_mean, s.csv(i.FIGURE2 + '/Panel F/equal_benchmark_means.csv'),
                                                       ['method'], ['balanced_accuracy'])
    allgates = pd.concat(gates)
    i.require(len(allgates) == 165 and allgates.status.eq('OK').all(), 'Source gate coverage differs')
    return {'methods': sorted(FINAL_F), 'tasks': 11, 'directions_per_task': 6,
            'target_unit_direction_tasks_per_method': 458, 'source_gates_per_initial_method': 33,
            'summary_differences': errors, 'new_gates_fitted': False, 'new_target_scores_computed': False}


def reconstruction_contract(s):
    records = []
    for case in i.CASES:
        base = 'benchmark_results/' + case + '/fastmnn/alignment/data/'
        z = s.csv(base + 'fastmnn_corrected_embedding.csv.gz')
        rotation = s.csv(base + 'fastmnn_pca_rotation.csv.gz')
        meta = s.csv(i.RECON + '/' + case + '/cell_metadata.csv.gz')
        markers = s.path(i.RECON + '/' + case + '/markers.txt').read_text().splitlines()
        i.require(z.cell_id.equals(meta.cell_id) and rotation.marker.tolist() == markers, 'Reconstruction identity differs')
        values = z.drop(columns='cell_id').to_numpy(float) @ rotation.drop(columns='marker').to_numpy(float).T
        old = np.load(s.path(i.RECON + '/' + case + '/reconstructed.npy'), allow_pickle=False)
        i.require(values.shape == old.shape and np.isfinite(values).all(), 'Reconstruction dimensions differ')
        error = float(np.max(abs(values - old)))
        i.require(error < 1e-12, 'Reconstructed expression differs')
        records.append({'case': case, 'cells': len(meta), 'markers': len(markers), 'max_absolute_difference': error,
                        'array_equal': bool(np.array_equal(values, old)), 'formula': 'corrected PCs @ rotation.T; no center added'})
    return records


def inspect(snapshot, source_root=None):
    s = Sources(snapshot, source_root)
    result = {'scope': 'frozen provenance, saved percentile/gate tables, in-memory reconstruction and summary checks',
              'catalog_sha256': CATALOG_SHA, 'percentiles': percentile_contract(s), 'gates': gate_contract(s)}
    if source_root is not None: result['reconstruction'] = reconstruction_contract(s)
    i.verify(snapshot, CATALOG_SHA, source_root)
    result['status'] = 'passed'
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', type=Path, required=True); p.add_argument('--source-root', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); out = i.reserve(args.output, [args.snapshot, *([args.source_root] if args.source_root else [])])
    try: i.write(out / 'report.json', inspect(args.snapshot, args.source_root))
    except BaseException as error:
        i.write(out / 'FAILED.json', {'error': repr(error)}); raise
