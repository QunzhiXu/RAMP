"""Read-only numerical comparison; explicit output is a NEW evidence directory."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

import contract as c
import replay as r

ATOL = 1e-12  # CSV decimal round-trips only; discrepancies above this fail.
DIAGNOSTICS = ['target_cluster_count', 'n_leiden_clusters', 'n_louvain_clusters',
               'leiden_resolution', 'louvain_resolution', 'leiden_cluster_count_error',
               'louvain_cluster_count_error', 'leiden_cluster_count_valid', 'louvain_cluster_count_valid']


def compare_values(actual, expected):
    actual = np.asarray(actual, dtype=float); expected = np.asarray(expected, dtype=float)
    r.require(actual.shape == expected.shape, 'Comparison shape differs')
    r.require(np.array_equal(np.isnan(actual), np.isnan(expected)), 'Missing-value pattern differs')
    finite = np.isfinite(actual) & np.isfinite(expected)
    r.require(np.all(finite | (np.isnan(actual) & np.isnan(expected))), 'Infinite metric')
    delta = np.abs(actual[finite] - expected[finite])
    r.require(np.all(delta <= ATOL), f'Metric discrepancy exceeds {ATOL}: max={delta.max(initial=0)}')
    return {'compared_values': int(actual.size), 'exact_values': int(np.sum(delta == 0)),
            'maximum_absolute_difference': float(delta.max(initial=0)), 'missing_values': int(np.isnan(actual).sum())}


def historical(catalog, case, variant):
    base = r.SNAPSHOT / 'sources' / case['evaluation_directory']
    file = (base / 'metric_battery_replicates.csv' if not case['case'].endswith('_story')
            else base / 'replicates' / (variant['key'] + '.csv'))
    rel = file.relative_to(r.SNAPSHOT / 'sources').as_posix()
    r.check(file, catalog['sources'][rel])
    frame = pd.read_csv(file, float_precision='round_trip')
    return frame[frame.method.eq(variant['method'])].set_index('sampling_seed')


def branch_records(manifest, batch_manifest):
    expected = {(case['case'], v['id']) for case in manifest['cases'] for v in case['variants']}
    entries = batch_manifest['branches']
    keys = [(x['case'], x['variant']) for x in entries]
    r.require(len(keys) == len(set(keys)) and set(keys) == expected, 'Batch branch grid differs')
    r.require(batch_manifest['code_records'] == r.code_records(), 'Batch used another controller')
    return dict(zip(keys, entries))


def check_run_record(batch, run, entry, parent):
    r.check(run / 'RUN_MANIFEST.json', entry['manifest'])
    request = r.read(run / 'REQUEST.json')
    r.require((request['case'], request['variant']) == (entry['case'], entry['variant']), 'Branch identity differs')
    r.require(all(parent.get(k) == v for k, v in request.items() if k not in ('case', 'variant')),
              'Worker request differs from batch')
    expected = [request['python'], '-B', str(batch / 'controller/replay_worker.py'), '--run', str(run)]
    r.require(r.read(run / 'COMMAND.json')['argv'] == expected, 'Worker command differs')
    return request


def compare_batch(batch, bundle, output=None, verify=True):
    manifest = r.verify_bundle(bundle)
    catalog, prior = r.anchored()
    done = r.read(batch / 'COMPLETE.json')
    r.require(done.get('status') == 'complete', 'Batch is not complete')
    r.check(batch / 'BATCH_MANIFEST.json', done['manifest'])
    bmanifest = r.read(batch / 'BATCH_MANIFEST.json')
    entries = branch_records(manifest, bmanifest)
    parent = r.read(batch / 'REQUEST.json')
    r.require(parent['code_records'] == r.code_records(), 'Batch request controller differs')
    for name, rec in parent['code_records'].items(): r.check(batch / 'controller' / name, rec)
    r.check(Path(parent['python']), parent['python_record'])
    names = [x[0] for x in manifest['metric_registry']]
    results = []; records = []; controller_sets = []
    helpers = c.pure_helpers(bundle / 'evaluator.py')
    for case in manifest['cases']:
        md = pd.read_csv(bundle / case['case'] / 'metadata.csv.gz', low_memory=False)
        expected_rows = {seed: helpers['sha256_array'](helpers['sample_rows'](md, seed, 5000)) for seed in manifest['seeds']}
        for v in case['variants']:
            run = batch / case['case'] / v['id']
            if verify: r.verify_branch(run, bundle)
            request = check_run_record(batch, run, entries[(case['case'], v['id'])], parent)
            controller_sets.append(request['code_records'])
            historic = historical(catalog, case, v)
            checks = []; diagnostics = []
            for seed in manifest['seeds']:
                row = r.read(run / 'replicates' / (str(seed) + '.json'))
                values = row['metrics']
                r.require(set(values) == set(names + DIAGNOSTICS), 'Unexpected metric fields')
                r.require(row['n_cells'] == int(historic.loc[seed, 'n_cells']) and
                          row['n_joint_celltypes'] == int(historic.loc[seed, 'n_joint_celltypes']), 'Evaluated label scope differs')
                r.require(row['sampled_rows_sha256'] == expected_rows[seed], 'Sampled rows differ')
                if 'sampled_rows_sha256' in historic:
                    r.require(row['sampled_rows_sha256'] == historic.loc[seed, 'sampled_rows_sha256'], 'Historical sampled rows differ')
                checks.append(compare_values([values[n] for n in names], historic.loc[seed, names].to_numpy(float)))
                diagnostics.append(compare_values([values[n] for n in DIAGNOSTICS], historic.loc[seed, DIAGNOSTICS].to_numpy(float)))
                records.append({'benchmark': case['case'], 'variant': v['id'], 'method': v['method'],
                                'sampling_seed': seed, 'n_cells': row['n_cells'],
                                'n_joint_celltypes': row['n_joint_celltypes'],
                                'sampled_rows_sha256': row['sampled_rows_sha256'], **values})
            before = r.read(run / 'ENVIRONMENT_BEFORE.json'); after = r.read(run / 'ENVIRONMENT_AFTER.json')
            r.require(before == after and before['packages'] == r.PACKAGES, 'Environment differed')
            results.append({'case': case['case'], 'method': v['method'], 'resamples': len(checks),
                            'metric_values': sum(x['compared_values'] for x in checks),
                            'exact_metric_values': sum(x['exact_values'] for x in checks),
                            'missing_metric_values': sum(x['missing_values'] for x in checks),
                            'diagnostic_values': sum(x['compared_values'] for x in diagnostics),
                            'exact_diagnostic_values': sum(x['exact_values'] for x in diagnostics),
                            'max_metric_difference': max(x['maximum_absolute_difference'] for x in checks),
                            'max_diagnostic_difference': max(x['maximum_absolute_difference'] for x in diagnostics),
                            'source_route': v['binding']['route'], 'runtime': before,
                            'stdout': r.record(run / 'stdout.log'), 'stderr': r.record(run / 'stderr.log')})
    r.require(all(x == r.code_records() for x in controller_sets), 'REQUEST code records differ')
    frames = pd.DataFrame(records)
    long_rows = []
    for case in manifest['cases']:
        group = frames[frames.benchmark.eq(case['case'])].copy()
        group = group[~group.method.eq('Unaligned') & ~group.method.str.contains('fixed_arcsinh', regex=False)].copy()
        group['family_method'] = group.method.map(lambda x: 'MaxFuse' if x.startswith('MaxFuse') else
                                                  'Harmony-direct' if x.startswith('Harmony-direct') else x)
        # All historical scores are finite. Refuse missing values rather than
        # silently changing the old matched-resample policy.
        r.require(np.isfinite(group[names].to_numpy(float)).all(), 'Missing primary metric requires explicit matched-policy review')
        collapsed = group.groupby(['sampling_seed', 'family_method'])[names].median()
        means = collapsed.groupby('family_method').mean()
        for metric, family, direction, _ in manifest['metric_registry']:
            ranks = means[metric].rank(method='average', ascending=(direction == 'lower'))
            for method in means.index:
                long_rows.append({'benchmark': case['case'], 'method': method, 'metric': metric, 'family': family,
                                  'direction': direction, 'mean': float(means.loc[method, metric]), 'rank': float(ranks[method])})
    all31 = pd.DataFrame(long_rows)
    selection = r.SNAPSHOT / 'sources' / c.SELECTION
    old = pd.read_csv(selection / 'metric_ranks_7_benchmarks_all31.csv', float_precision='round_trip')
    sort = ['benchmark', 'method', 'metric']
    new = all31.sort_values(sort).reset_index(drop=True); old = old.sort_values(sort).reset_index(drop=True)
    r.require(new[sort].equals(old[sort]), 'Aggregate task grid differs')
    means_check = compare_values(new['mean'], old['mean'])
    r.require(np.array_equal(new['rank'].to_numpy(), old['rank'].to_numpy()), 'Metric ranks changed')
    policy = r.read(selection / 'metric_selection.json')
    main23 = all31[all31.metric.isin(policy['included_metrics'])].copy()
    reference = pd.read_csv(selection / 'primary_overall_method_ranking.csv', float_precision='round_trip')
    summary = c.check_rank_tables(all31, main23, policy, reference)
    report = {'scope': '63 variants x 20 seeds; frozen coordinates, fresh metrics', 'status': 'passed',
              'absolute_tolerance': ATOL, 'branch_results': results, 'aggregate_means': means_check,
              'rank_discrepancies': 0, 'selection': summary,
              'identical_63_request_code_records': all(x == controller_sets[0] for x in controller_sets),
              'max_metric_difference': max(x['max_metric_difference'] for x in results),
              'metric_values': sum(x['metric_values'] for x in results),
              'exact_metric_values': sum(x['exact_metric_values'] for x in results),
              'missing_metric_values': sum(x['missing_metric_values'] for x in results),
              'diagnostic_values': sum(x['diagnostic_values'] for x in results),
              'exact_diagnostic_values': sum(x['exact_diagnostic_values'] for x in results),
              'batch_manifest': r.record(batch / 'BATCH_MANIFEST.json'), 'input_manifest': r.record(bundle / 'MANIFEST.json')}
    if output is not None:
        output = r.reserve(r.local_output(output, 'validation'), [batch, bundle])
        for name, frame in [('metric_replicates.csv', frames), ('all31_ranks.csv', all31), ('main23_ranks.csv', main23)]:
            with (output / name).open('x') as handle: frame.to_csv(handle, index=False)
        r.write(output / 'report.json', report)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch', type=Path, required=True); p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    result = compare_batch(args.batch.resolve(), args.bundle.resolve(), args.output)
    print(json.dumps({k: result[k] for k in ('status', 'metric_values', 'max_metric_difference', 'rank_discrepancies')}))
