"""Cell-type marker quartile shifts, using the same frozen cells for all methods.

Percentile baseline: raw marker ranks among ALL cells in each sample. All cell
types are computed; the primary display uses resolved groups with >=50 cells,
matching the existing exhaustive artifact audit's minimum group size.
"""
from pathlib import Path
import os
import sys
import json
import re
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE.parent / 'pair_inversion_boxplot_20260914'))
import compute_pair_inversions as frozen
os.environ['NUMBA_CACHE_DIR'] = str(HERE / '.cache/numba')
import numpy as np
import pandas as pd
from scipy.stats import rankdata

MIN_CELLS = 50
INVALID = re.compile(r'non-joint|unclassified|technical|unannotated|unknown|ambiguous|missing|\bmix\b', re.I)
QUANTILES = ['q25', 'median', 'q75']
KEYS = ['case', 'dataset', 'unit_key', 'celltype', 'marker']


def percentile_matrix(values):
    return 100 * rankdata(values, method='average', axis=0) / (len(values) + 1.)


def quartiles(values):
    return np.quantile(values, [.25, .5, .75], axis=0, method='linear')


def main():
    started = time.time()
    frozen.track(__file__)
    frozen.track(frozen.__file__)
    records, coverage, rank_audit = [], [], []
    for case in frozen.CASES:
        source = ROOT / 'benchmark_results' / case / 'ramp/alignment/input'
        meta = pd.read_csv(frozen.track(source / 'cell_metadata.csv.gz'),
                           low_memory=False, dtype={'cell_id': str})
        ids = meta.cell_id.to_numpy()
        assert not meta.cell_id.duplicated().any()
        markers = frozen.track(source / 'markers.txt').read_text().strip().splitlines()
        raw = np.asarray(np.load(frozen.track(source / 'raw.npy'), mmap_mode='r'), dtype=np.float64)
        assert raw.shape == (len(meta), len(markers)) and np.isfinite(raw).all()
        labels = meta.celltype.fillna('Missing').astype(str).to_numpy()
        unit_groups = []
        for unit, rows in meta.groupby('unit_key', sort=True).indices.items():
            ds = meta.iloc[rows].dataset.unique()
            assert len(ds) == 1
            raw_rank = percentile_matrix(raw[rows])
            cell_groups = []
            for label in sorted(set(labels[rows])):
                local = np.flatnonzero(labels[rows] == label)
                reason = 'unresolved/non-joint label' if INVALID.search(label) else \
                         'fewer than 50 cells' if len(local) < MIN_CELLS else 'included'
                eligible = reason == 'included'
                coverage.append(dict(case=case, dataset=ds[0], unit_key=unit, celltype=label,
                    n_cells=len(local), n_sample_cells=len(rows), n_markers=len(markers),
                    included_in_plot=eligible, inclusion_reason=reason))
                cell_groups.append((label, local, eligible, reason, quartiles(raw_rank[local])))
            unit_groups.append((unit, ds[0], rows, cell_groups, raw_rank))
        for method, before, after in frozen.load_methods(case, ids, markers):
            for unit, dataset, rows, groups, raw_rank in unit_groups:
                input_rank, output_rank = percentile_matrix(before[rows]), percentile_matrix(after[rows])
                rank_audit.append(dict(case=case, unit_key=unit, method=method,
                    max_cell_percentile_raw_to_input=float(np.max(np.abs(input_rank-raw_rank))),
                    max_cell_percentile_input_to_output=float(np.max(np.abs(output_rank-input_rank))),
                    max_cell_percentile_raw_to_output=float(np.max(np.abs(output_rank-raw_rank)))))
                for label, local, eligible, reason, raw_q in groups:
                    input_q, output_q = quartiles(input_rank[local]), quartiles(output_rank[local])
                    for j, marker in enumerate(markers):
                        r = dict(case=case, dataset=dataset, unit_key=unit, celltype=label,
                                 marker=marker, method=method, n_cells=len(local), n_sample_cells=len(rows),
                                 included_in_plot=eligible, inclusion_reason=reason)
                        for k, q in enumerate(QUANTILES):
                            r['raw_'+q] = raw_q[k,j]
                            r['method_input_'+q] = input_q[k,j]
                            r['corrected_'+q] = output_q[k,j]
                            r['delta_'+q+'_pp'] = output_q[k,j] - raw_q[k,j]
                            r['abs_delta_'+q+'_pp'] = abs(r['delta_'+q+'_pp'])
                            r['input_to_output_delta_'+q+'_pp'] = output_q[k,j] - input_q[k,j]
                        records.append(r)
            print(case, method, f'{time.time()-started:.1f}s', flush=True)
    d = pd.DataFrame(records)
    d.to_csv(HERE / 'celltype_quartile_shifts_all.csv.gz', index=False)
    pd.DataFrame(coverage).to_csv(HERE / 'celltype_coverage.csv', index=False)
    pd.DataFrame(rank_audit).to_csv(HERE / 'preprocessing_percentile_audit.csv', index=False)
    primary = d[d.included_in_plot].copy()
    primary.to_csv(HERE / 'celltype_quartile_shifts_display.csv.gz', index=False)
    assert not d.duplicated([*KEYS, 'method']).any()
    common = set(map(tuple, primary[primary.method.eq('RAMP')][KEYS].to_numpy()))
    for method in frozen.METHODS:
        assert set(map(tuple, primary[primary.method.eq(method)][KEYS].to_numpy())) == common
    summaries = []
    for scope, table in [('display', primary), ('all_groups', d)]:
        for method in frozen.METHODS:
            sub = table[table.method.eq(method)]
            for q in QUANTILES:
                v = sub['abs_delta_'+q+'_pp']
                summaries.append(dict(scope=scope, method=method, quartile=q, n=len(v),
                    mean_pp=v.mean(), q25_pp=v.quantile(.25), median_pp=v.median(),
                    q75_pp=v.quantile(.75), q90_pp=v.quantile(.90), max_pp=v.max(),
                    n_above_10_pp=int((v>10).sum()), n_above_20_pp=int((v>20).sum())))
    summary = pd.DataFrame(summaries)
    summary.to_csv(HERE / 'quartile_shift_summary.csv', index=False)
    primary.groupby(['case','method']).abs_delta_median_pp.agg(['size','mean','median','min','max']).to_csv(
        HERE / 'median_shift_by_benchmark.csv')
    # Independently produced Panel C quartiles provide a real-data regression
    # check of all the groups (not just the six shown in the final figure).
    old = pd.read_csv(frozen.track(HERE.parent / 'redesign_full_icon_20260914/all_celltype_statistics.csv'))
    old_map = {'Raw':'Raw', 'Harmony-direct':frozen.METHODS[0],
               'cyCombine':'cyCombine', 'CytoNorm':'CytoNorm', 'RAMP':'RAMP'}
    checked = 0
    max_error = 0.
    for r in old.itertuples():
        sub = d[d.case.eq(r.benchmark)&d.unit_key.eq(r.unit_key)&d.celltype.eq(r.celltype)&d.marker.eq(r.marker)]
        method = old_map[r.method]
        row = sub[sub.method.eq('RAMP' if method=='Raw' else method)].iloc[0]
        prefix = 'raw_' if method=='Raw' else 'corrected_'
        for old_q, new_q in [('q1','q25'),('median','median'),('q3','q75')]:
            error = abs(row[prefix+new_q] - 100 * getattr(r,old_q))
            max_error = max(max_error,error)
            assert error < 1e-9, (r, new_q, error)
            checked += 1
    assert all(frozen.sha(p)==s for p,s in frozen.READS.items())
    manifest = dict(cases=frozen.CASES, methods=frozen.METHODS, baseline='Common frozen raw marker values',
        cell_percentile='100 * average_rank / (n_sample_cells + 1), separately per sample and marker, before any cell-type filtering',
        primary_metric='Absolute change in the median cell percentile within each sample x cell type x marker, in percentile points',
        auxiliary_metrics='Signed/absolute Q25 and Q75 shifts; signed median shift; method-input-to-output shifts retained separately',
        quantile_interpolation='numpy linear (type 7)', minimum_group_cells=MIN_CELLS,
        excluded_group_labels=INVALID.pattern, excluded_cells_still_in_percentile_reference=True,
        display_groups_per_method=len(common), all_groups_per_method=len(d)//5,
        sample_units=primary[['case','unit_key']].drop_duplicates().shape[0],
        aggregation='One equally weighted observation per eligible sample x cell type x marker; no cell-count, sample-equal or benchmark-equal weighting',
        tied_values='Average ranks; exact output ties retained. Never force a method to have zero change.',
        common_combinations_identical_across_methods=True, all_frozen_sources_unchanged=True,
        panel_c_quartile_regression_checks=checked, max_panel_c_quartile_error_pp=max_error,
        no_significance_testing=True, celltype_annotations_used_only_for_evaluation=True,
        interpretation='Magnitude of cell-type distribution displacement, not automatic evidence that every displacement is a biological artifact',
        runtime_seconds=time.time()-started, input_sha256=frozen.READS)
    (HERE / 'computation_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(summary[summary.scope.eq('display')&summary.quartile.eq('median')].to_string(index=False),flush=True)
    print('Display groups per method:',len(common),'All groups:',len(d)//5,flush=True)


if __name__ == '__main__':
    main()
