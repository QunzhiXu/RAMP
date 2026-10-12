"""Read-only discovery of the published seven-task integration evaluation.

Never imports the legacy runners (several write at import time). Only five
pure helper functions are compiled from their ASTs, without module statements.
No fitting, UMAP fitting, metric selection, or metric recomputation occurs here.
"""
import ast
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
STUDY = HERE.parents[1]
REGISTRY = STUDY / 'benchmarks/integration_inputs/registry_20261010.json'
REGISTRY_SHA = '4f477c04dc32e98480d2d74ef29927c3e324813b64ca3c832d1a31f4c9f07b22'
EVALUATOR = 'benchmark_results/comparison_evaluation/scripts/evaluate_benchmark.py'
OLD_EVALUATOR = 'benchmark_results/comparison_evaluation/archive/pre_cytonorm_20260909/scripts/evaluate_benchmark.py'
SELECTION = 'Main Figure/Figure 2/lung_reference_update_20260921/selection'
STORIES = {
    'lung_story': 'story_lung_HLA_ABC/Comparison_senosain140k_20260921',
    'melanoma_story': 'figure_melanoma/Comparison',
    'hnscc_story': 'figure_HNSCC/Comparison',
}
LABELS = dict(raw='Unaligned', ramp='RAMP', harmony='Harmony-direct',
              fastmnn='fastMNN', cycombine='cyCombine', cytonorm='CytoNorm',
              maxfuse='MaxFuse', maxfuse_hub0='MaxFuse hub 1',
              maxfuse_hub1='MaxFuse hub 2', maxfuse_hub2='MaxFuse hub 3')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def record(path):
    with Path(path).open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    return {'bytes': Path(path).stat().st_size, 'sha256': digest}


def read(path):
    return json.loads(Path(path).read_text())


def pure_helpers(path):
    tree = ast.parse(path.read_text())
    names = {'safe_name', 'sha256_array', 'standardized', 'sample_rows', 'sha256_json'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    require({n.name for n in nodes} == names, 'Missing pure evaluator helper')
    namespace = dict(np=np, pd=pd, hashlib=hashlib, re=re, StandardScaler=StandardScaler)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    for name in ('REGISTRY', 'SEEDS', 'UMAP_PROTOCOL'):
        matches = [n.value for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
        require(len(matches) == 1, 'Ambiguous literal: ' + name)
        namespace[name] = ast.literal_eval(matches[0])
    return namespace


def check_grid(frame, methods, seeds, metrics):
    keys = ['method', 'sampling_seed']
    require(not frame.duplicated(keys).any(), 'Duplicate method/seed rows')
    require(set(frame[keys].itertuples(index=False, name=None)) ==
            {(m, s) for m in methods for s in seeds}, 'Incomplete method/seed grid')
    require(set(metrics).issubset(frame.columns), 'Missing metric columns')


def check_rank_tables(all31, main23, policy, reference):
    keys = ['benchmark', 'method', 'metric']
    methods = {'RAMP', 'MaxFuse', 'cyCombine', 'Harmony-direct', 'fastMNN', 'CytoNorm'}
    require(set(all31.method) == methods and all31.benchmark.nunique() == 7,
            'Wrong seven-task/six-method scope')
    for data, count in ((all31, 31), (main23, 23)):
        require(len(data) == 7 * 6 * count and not data.duplicated(keys).any(), 'Invalid rank grid')
        require(data.groupby(['benchmark', 'method']).size().eq(count).all(), 'Metric coverage')
    included = policy['included_metrics']
    require(len(included) == len(set(included)) == 23, 'Main23 changed')
    subset = all31[all31.metric.isin(included)]
    cols = main23.columns.tolist()
    pd.testing.assert_frame_equal(subset[cols].sort_values(keys).reset_index(drop=True),
                                  main23.sort_values(keys).reset_index(drop=True),
                                  check_exact=False, rtol=0, atol=1e-14)
    require(policy['primary_family_weights'] == dict(biology=.25, mixing=.25, transfer=.25, visual=.25),
            'Four-category weights changed')
    require(main23[['metric', 'family']].drop_duplicates().groupby('family').size().to_dict()
            == dict(biology=5, mixing=14, transfer=2, visual=2), 'Main23 category counts')
    # Same three means as the published builder: no re-ranking category means.
    overall = (main23.groupby(['benchmark', 'method', 'family'])['rank'].mean()
               .groupby(['benchmark', 'method']).mean().groupby('method').mean())
    expected = reference.set_index('method').mean_rank.reindex(overall.index)
    require(np.allclose(overall, expected, rtol=0, atol=1e-14), 'Published aggregate differs')
    return {'all31_rows': len(all31), 'main23_rows': len(main23),
            'recomputed_family_equal_mean_ranks': overall.to_dict(),
            'max_aggregate_difference': float(np.max(np.abs(overall - expected)))}


class Sources:
    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)
        self.records = {}
        self.copies = set()
        self.links = []

    def add(self, path, copy=False, expected=None):
        path = Path(path)
        if not path.is_absolute():
            path = self.root / path
        require(path.resolve().is_relative_to(self.root), 'Outside source root: ' + str(path))
        key = path.relative_to(self.root).as_posix()
        current = record(path)
        if key in self.records:
            require(current == self.records[key], 'Source changed while collecting: ' + key)
        if expected is not None:
            require(current['sha256'] == expected, 'Historical hash mismatch: ' + key)
        self.records[key] = current
        if copy:
            self.copies.add(key)
        return path

    def bind(self, json_path, field, target):
        owner = self.add(json_path, copy=True)
        value = read(owner)
        for part in field:
            value = value[part]
        target = self.add(target, expected=value)
        self.links.append({'owner': owner.relative_to(self.root).as_posix(), 'field': field,
                           'target': target.relative_to(self.root).as_posix(), 'sha256': value})


def load_matrix(path, method, metadata, markers):
    if path.suffix == '.npy':
        values = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    else:
        kwargs = {'float_precision': 'round_trip'} if method == 'CytoNorm' else {}
        table = pd.read_csv(path, low_memory=False, **kwargs)
        require(np.array_equal(table.cell_id.astype(str), metadata.cell_id.astype(str)), 'Matrix row IDs differ')
        columns = [c for c in table if c != 'cell_id']
        if method != 'fastMNN':
            require(columns == markers, 'Marker order differs')
        values = table[columns].to_numpy(dtype=np.float64)
    require(values.ndim == 2 and len(values) == len(metadata) and np.isfinite(values).all(),
            'Invalid evaluation representation')
    return values


def collect(root, registry=REGISTRY):
    registry_bytes = Path(registry).read_bytes()
    require(hashlib.sha256(registry_bytes).hexdigest() == REGISTRY_SHA, 'Public registry anchor differs')
    registry_doc = json.loads(registry_bytes)
    src = Sources(root)
    helper = pure_helpers(src.add(EVALUATOR, copy=True))
    old = ast.parse(src.add(OLD_EVALUATOR, copy=True).read_text())
    current = ast.parse((src.root / EVALUATOR).read_text())
    functions = lambda tree: {n.name: ast.dump(n, include_attributes=False) for n in tree.body
                              if isinstance(n, ast.FunctionDef) and n.name != 'load_case'}
    require(functions(old) == functions(current), 'Historical numerical evaluator changed')
    metrics = [r[0] for r in helper['REGISTRY']]
    require(len(metrics) == 31 and len(helper['SEEDS']) == 20, 'Metric/seed contract differs')
    for relative in ('benchmark_results/comparison_evaluation/scripts/summarize_all.py',
                     'benchmark_results/cytonorm_benchmark_runner.py',
                     'algorithm_test/vendor/evaluation_reference.py',
                     'algorithm/CytoNorm/benchmark.py',
                     'algorithm/CytoNorm/evaluate.py',
                     'Main Figure/Figure 2/lung_reference_update_20260921/build_selection.py'):
        src.add(relative, copy=True)
    cases = []
    for spec in registry_doc['cases']:
        case = spec['case']
        metadata_path = src.add(spec['files']['evaluation_metadata']['path'],
                                expected=spec['files']['evaluation_metadata']['sha256'])
        md = pd.read_csv(metadata_path, low_memory=False)
        marker_path = src.add(spec['files']['markers']['path'], expected=spec['files']['markers']['sha256'])
        markers = marker_path.read_text().splitlines()
        require(len(markers) == spec['expected_shape'][1] and len(md) == spec['expected_shape'][0] and md.unit_key.nunique() == spec['expected_units'],
                'Cohort size differs: ' + case)
        require(not md.cell_id.isna().any() and not md.cell_id.duplicated().any(), 'Invalid cell IDs')
        branches = []
        if spec['kind'] == 'technical':
            out = src.root / 'benchmark_results/comparison_evaluation' / case
            checkpoint_path = src.add(out / 'checkpoint_manifest.json', copy=True)
            config = read(checkpoint_path)['configuration']
            for listname in ('source_files', 'incremental_source_files'):
                for i, item in enumerate(config[listname]):
                    # Resolve recorded original-tree paths without trusting arbitrary absolute paths.
                    rel = item['path'].split('/final_xu/', 1)[1]
                    src.bind(checkpoint_path, ['configuration', listname, i, 'sha256'], rel)
                    if listname == 'incremental_source_files': src.add(rel, copy=True)
            inventory = pd.read_csv(src.add(out / 'method_inventory.csv', copy=True))
            reps = pd.read_csv(src.add(out / 'metric_battery_replicates.csv', copy=True))
            require(config['sampling_seeds'] == list(helper['SEEDS']), 'Technical seeds differ')
            check_grid(reps, inventory.method, helper['SEEDS'], metrics)
            require(reps.run_fingerprint.astype(str).equals(reps.method.map(config['method_run_fingerprints']).astype(str)),
                    'Technical per-method fingerprint differs')
            provenance = src.root / 'benchmark_results' / case / 'cytonorm/evaluation/provenance.json'
            src.bind(provenance, ['evaluator_sha256'], 'algorithm/CytoNorm/evaluate.py')
            src.bind(provenance, ['reference_sha256'], 'algorithm_test/vendor/evaluation_reference.py')
            for row in inventory.itertuples():
                source_path = row.source_path.split('/final_xu/', 1)[1]
                branches.append((row.method, src.add(source_path), out, row.method))
        else:
            base = src.root / STORIES[case]
            out = base / ('evaluation' if case == 'lung_story' else 'maxfuse_full_panel/evaluation')
            for directory in (base / 'scripts', base / 'maxfuse_full_panel/scripts'):
                for code in sorted(directory.glob('*.py')):
                    src.add(code, copy=True)
            keys = ['raw', 'ramp', 'harmony', 'fastmnn', 'cycombine', 'cytonorm']
            keys += ['maxfuse_hub0', 'maxfuse_hub1', 'maxfuse_hub2'] if case == 'lung_story' else ['maxfuse']
            frames = []
            for key in keys:
                path = (base / 'input' / (key + '.npy') if key in ('raw', 'ramp') else
                        base / ('maxfuse_full_panel' if key.startswith('maxfuse') else '') / 'methods' / key / 'values.npy')
                prefix = out / 'replicates' / key
                config_path = src.add(str(prefix) + '_config.json', copy=True)
                config = read(config_path)
                for field, target in [('source_values_sha256', path), ('metadata_sha256', metadata_path),
                                      ('evaluator_source_sha256', EVALUATOR),
                                      ('adapter_sha256', base / 'scripts/evaluate.py'),
                                      ('input_manifest_sha256', base / 'input/manifest.json')]:
                    src.bind(config_path, ['configuration', field], target)
                require(config['configuration']['seeds'] == list(helper['SEEDS']) and
                        config['configuration']['cells_per_dataset'] == 5000 and
                        config['configuration']['umap_protocol'] == helper['UMAP_PROTOCOL'], 'Story protocol differs')
                require(config['fingerprint'] == hashlib.sha256(json.dumps(config['configuration'], sort_keys=True).encode()).hexdigest(), 'Story fingerprint differs')
                completion = src.add(str(prefix) + '_COMPLETE.json', copy=True)
                src.bind(completion, ['replicates_sha256'], str(prefix) + '.csv')
                done = read(completion)
                require(done['fingerprint'] == config['fingerprint'] and done['n_resamples'] == 20, 'Incomplete story evaluation')
                frame = pd.read_csv(src.add(str(prefix) + '.csv', copy=True))
                check_grid(frame, [LABELS[key]], helper['SEEDS'], metrics)
                require(frame.fingerprint.eq(config['fingerprint']).all(), 'Replicate fingerprint differs')
                for seed in helper['SEEDS']:
                    digest = helper['sha256_array'](helper['sample_rows'](md, seed, 5000))
                    require(frame.loc[frame.sampling_seed.eq(seed), 'sampled_rows_sha256'].eq(digest).all(), 'Story resampling rows differ')
                frames.append(frame)
                branches.append((LABELS[key], src.add(path), out, key))
            reps = pd.concat(frames, ignore_index=True)
        for filename in ('metric_registry.csv', 'metric_battery_summary.csv', 'metric_battery_replicates.csv'):
            src.add(out / filename, copy=True)
        if (out / 'runtime_notes.md').exists(): src.add(out / 'runtime_notes.md', copy=True)
        branch_records = []
        for method, path, directory, key in branches:
            values = load_matrix(path, method, md, markers)
            name = helper['safe_name'](method)
            umap_manifest = src.add(directory / 'umap/manifests' / (name + '.json'), copy=True)
            umap_record = read(umap_manifest)
            coordinates = src.add(directory / 'umap/coordinates' / (name + '.npy'))
            coords = np.load(coordinates, allow_pickle=False)
            scaled = helper['standardized'](values, method)
            require(umap_record['protocol'] == helper['UMAP_PROTOCOL'] and
                    umap_record['labels_used_for_fit'] is False and umap_record['method'] == method,
                    'UMAP protocol or identity differs')
            require(umap_record['representation_sha256'] == helper['sha256_array'](scaled),
                    'UMAP representation differs: ' + case + '/' + method)
            require(coords.shape == (len(md), 2) and np.isfinite(coords).all() and
                    umap_record['coordinate_sha256'] == helper['sha256_array'](coords), 'UMAP coordinate digest differs')
            require(umap_record['n_cells'] == len(md) and umap_record['n_dimensions'] == scaled.shape[1],
                    'UMAP dimensions differ')
            branch_records.append({'method': method, 'key': key, 'matrix': path.relative_to(src.root).as_posix(),
                                   'umap_manifest': umap_manifest.relative_to(src.root).as_posix(),
                                   'coordinates': coordinates.relative_to(src.root).as_posix(),
                                   'dimensions': scaled.shape[1], 'representation_hash_verified': True})
        cases.append({'case': case, 'cells': len(md), 'units': md.unit_key.nunique(),
                      'markers': markers, 'variants': branch_records, 'replicate_rows': len(reps),
                      'evaluation_directory': out.relative_to(src.root).as_posix()})
    # Freeze the published selection and verify its exact five source-table hash links.
    selection = src.root / SELECTION
    for path in sorted(selection.iterdir()):
        if path.is_file(): src.add(path, copy=True)
    policy_path = selection / 'metric_selection.json'
    policy = read(policy_path)
    for path, digest in policy['source_sha256'].items():
        rel = path.split('/final_xu/', 1)[1]
        src.bind(policy_path, ['source_sha256', path], rel)
        src.add(rel, copy=True)
    csv = lambda p: pd.read_csv(p, float_precision='round_trip')
    all31 = csv(selection / 'metric_ranks_7_benchmarks_all31.csv')
    main23 = csv(selection / 'metric_ranks_7_benchmarks_main23.csv')
    sources = [csv(src.root / 'benchmark_results/comparison_evaluation/summary/primary_method_family_metric_ranks.csv')
               .rename(columns={'metric_family': 'family', 'method_family': 'method'})]
    for case in [c for c in cases if c['case'] in STORIES]:
        sources.append(csv(src.root / case['evaluation_directory'] / 'metric_method_ranks_all31.csv'))
    cols = all31.columns.tolist(); keys = ['benchmark', 'method', 'metric']
    pd.testing.assert_frame_equal(pd.concat([s[cols] for s in sources], ignore_index=True).sort_values(keys).reset_index(drop=True),
                                  all31.sort_values(keys).reset_index(drop=True), check_exact=False, rtol=0, atol=1e-14)
    summary = check_rank_tables(all31, main23, policy, csv(selection / 'primary_overall_method_ranking.csv'))
    require(sum(len(c['variants']) for c in cases) == 63, 'Expected 63 evaluated variants')
    require(record(registry)['sha256'] == REGISTRY_SHA, 'Registry changed during collection')
    return {'schema': 'integration-evaluation-inventory-v1', 'scope': 'source contracts and saved evaluation artifacts; no fitting',
            'registry_sha256': REGISTRY_SHA, 'metric_registry': helper['REGISTRY'], 'seeds': helper['SEEDS'],
            'umap_protocol': helper['UMAP_PROTOCOL'], 'cases': cases, 'selection_check': summary,
            'sources': src.records, 'copies': sorted(src.copies), 'historical_hash_links': src.links,
            'numerical_evaluator_functions_equal_excluding_load_case': True,
            'not_verified': ['fresh UMAP/metric execution', 'per-replicate metric numerical recomputation',
                             'new-run result routing', 'clean environment installation', 'provider/clinical chains']}
