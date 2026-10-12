"""Read-only fresh-coordinate and metric comparison, without requiring favorable equality."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

import study as s
from bootstrap import r
import contract
from compare_replay import DIAGNOSTICS


def compare(batch, bundle, snapshot, output):
    s.verify_run(batch, bundle)
    manifest = s.inputs(bundle)
    catalog = r.read(Path(bundle) / 'trust/catalog.json')
    r.require(r.record(Path(snapshot) / 'catalog.json')['sha256'] == r.CATALOG_SHA, 'Reference snapshot anchor differs')
    request = r.read(batch / 'REQUEST.json')
    r.require(request['scope'] == 'all', 'Pilot is not a complete evaluation')
    rows = []; branch_rows = []; metric_rows = []
    names = [x[0] for x in manifest['metric_registry']]
    families = {x[0]: x[1] for x in manifest['metric_registry']}
    helper = contract.pure_helpers(bundle / 'evaluator.py')
    for case, variant in s.branches(manifest, 'all'):
        run = batch / case['case'] / variant['id']
        path = Path(case['evaluation_directory']) / ('replicates/' + variant['key'] + '.csv' if case['case'].endswith('_story')
                                                    else 'metric_battery_replicates.csv')
        file = snapshot / 'sources' / path; r.check(file, catalog['sources'][path.as_posix()])
        history = pd.read_csv(file, float_precision='round_trip')
        history = history[history.method.eq(variant['method'])].set_index('sampling_seed')
        md = pd.read_csv(bundle / case['case'] / 'metadata.csv.gz', low_memory=False)
        coordinate = r.read(run / 'COORDINATE_COMPARISON.json')
        new_rows = []
        for seed in manifest['seeds']:
            record = r.read(run / 'replicates' / (str(seed) + '.json'))
            sample = helper['sample_rows'](md, seed, 5000)
            r.require(record['sampled_rows_sha256'] == helper['sha256_array'](sample) and
                      record['n_cells'] == int(history.loc[seed, 'n_cells']) and
                      record['n_joint_celltypes'] == int(history.loc[seed, 'n_joint_celltypes']), 'Sample/label identity differs')
            r.require(set(record['metrics']) == set(names + DIAGNOSTICS), 'Metric grid differs')
            row = {k: record[k] for k in ('benchmark', 'method', 'sampling_seed', 'n_cells', 'n_joint_celltypes', 'sampled_rows_sha256')}
            row.update(variant=variant['id'], **record['metrics']); rows.append(row); new_rows.append(row)
        frame = pd.DataFrame(new_rows).set_index('sampling_seed')
        for metric in names + DIAGNOSTICS:
            current = frame.loc[manifest['seeds'], metric].to_numpy(float)
            old = history.loc[manifest['seeds'], metric].to_numpy(float)
            r.require(np.array_equal(np.isnan(current), np.isnan(old)), 'Missing metric pattern changed')
            r.require(np.isfinite(current).all(), 'Nonfinite metric requires investigation')
            delta = np.abs(current - old)
            metric_rows.append({'case': case['case'], 'variant': variant['id'], 'method': variant['method'],
                'metric': metric, 'family': families.get(metric, 'diagnostic'), 'values': len(current),
                'exact_values': int(np.count_nonzero(current == old)), 'max_absolute_difference': float(delta.max()),
                'mean_signed_difference': float(np.mean(current - old))})
        branch_rows.append({'case': case['case'], 'variant': variant['id'], 'method': variant['method'], **coordinate,
                            'fit_seconds': r.read(run / 'FIT.json')['seconds']})
    replicates = pd.DataFrame(rows); differences = pd.DataFrame(metric_rows); coordinates = pd.DataFrame(branch_rows)
    r.require(len(replicates) == 1260 and not replicates.duplicated(['benchmark', 'variant', 'sampling_seed']).any(), 'Incomplete resamples')
    long_rows = []
    for case in manifest['cases']:
        data = replicates[replicates.benchmark.eq(case['case'])].copy()
        data = data[~data.method.eq('Unaligned') & ~data.method.str.contains('fixed_arcsinh', regex=False)].copy()
        data['family_method'] = data.method.map(lambda x: 'MaxFuse' if x.startswith('MaxFuse') else
                                                'Harmony-direct' if x.startswith('Harmony-direct') else x)
        means = data.groupby(['sampling_seed', 'family_method'])[names].median().groupby('family_method').mean()
        for metric, family, direction, _ in manifest['metric_registry']:
            ranks = means[metric].rank(method='average', ascending=direction == 'lower')
            for method in means.index:
                long_rows.append({'benchmark': case['case'], 'method': method, 'metric': metric, 'family': family,
                                  'direction': direction, 'mean': float(means.loc[method, metric]), 'rank': float(ranks[method])})
    all31 = pd.DataFrame(long_rows)
    selection = snapshot / 'sources' / contract.SELECTION
    for name in ('metric_selection.json', 'metric_ranks_7_benchmarks_all31.csv', 'primary_overall_method_ranking.csv'):
        r.check(selection / name, catalog['sources'][(Path(contract.SELECTION) / name).as_posix()])
    policy = r.read(selection / 'metric_selection.json')
    main23 = all31[all31.metric.isin(policy['included_metrics'])].copy()
    r.require(len(all31) == 1302 and len(main23) == 966 and
              policy['primary_family_weights'] == dict(biology=.25, mixing=.25, transfer=.25, visual=.25), 'Ranking protocol changed')
    overall = (main23.groupby(['benchmark', 'method', 'family'])['rank'].mean().groupby(['benchmark', 'method']).mean()
               .groupby('method').mean().rename('mean_rank').reset_index())
    old_overall = pd.read_csv(selection / 'primary_overall_method_ranking.csv', float_precision='round_trip')
    overall = overall.merge(old_overall[['method', 'mean_rank']], on='method', suffixes=('', '_historical'), validate='one_to_one')
    overall['difference'] = overall.mean_rank - overall.mean_rank_historical
    old_ranks = pd.read_csv(selection / 'metric_ranks_7_benchmarks_all31.csv', float_precision='round_trip')
    joined = all31.merge(old_ranks[['benchmark', 'method', 'metric', 'mean', 'rank']], on=['benchmark', 'method', 'metric'],
                         suffixes=('', '_historical'), validate='one_to_one')
    nonvisual = differences[~differences.family.isin(['visual', 'diagnostic'])]
    scientific_equal = bool(coordinates.array_equal.all() and differences.max_absolute_difference.eq(0).all())
    report = {'status': 'completed', 'scientific_equality': scientific_equal,
        'scope': '63 fresh full-cohort UMAP fits and 1260 newly evaluated resamples; historical MaxFuse embeddings unchanged',
        'coordinate_exact_branches': int(coordinates.array_equal.sum()), 'branches': 63,
        'metric_values': int(differences[differences.family.ne('diagnostic')]['values'].sum()),
        'exact_metric_values': int(differences[differences.family.ne('diagnostic')].exact_values.sum()),
        'exact_diagnostic_values': int(differences[differences.family.eq('diagnostic')].exact_values.sum()),
        'max_metric_difference': float(differences.max_absolute_difference.max()),
        'max_nonvisual_difference': float(nonvisual.max_absolute_difference.max()),
        'max_aggregate_mean_difference': float((joined['mean'] - joined.mean_historical).abs().max()),
        'rank_discrepancies': int(joined['rank'].ne(joined.rank_historical).sum()),
        'max_overall_score_difference': float(overall.difference.abs().max()),
        'batch_manifest': r.record(batch / 'BATCH_MANIFEST.json'), 'input_manifest': r.record(bundle / 'MANIFEST.json'),
        'main23_posthoc_policy_unchanged': True, 'new_parameter_search': False}
    output = r.reserve(output, [batch, bundle, snapshot])
    for name, data in [('metric_replicates', replicates), ('coordinate_comparison', coordinates),
                       ('metric_differences', differences), ('all31_ranks', all31), ('main23_ranks', main23),
                       ('overall_scores', overall), ('rank_comparison', joined)]:
        with (output / (name + '.csv')).open('x') as handle: data.to_csv(handle, index=False)
    r.write(output / 'report.json', report)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('batch', 'bundle', 'snapshot', 'output'): p.add_argument('--' + key, type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(compare(a.batch.resolve(), a.bundle.resolve(), a.snapshot.resolve(), a.output.resolve()), indent=2))
