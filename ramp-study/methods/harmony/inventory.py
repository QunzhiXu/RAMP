"""Freeze Harmony-direct adapter and eleven historical branches; never fit."""
import argparse
from pathlib import Path

import harmony_io as io

HERE = Path(__file__).resolve().parent
REGISTRY = HERE.parents[1] / 'benchmarks/integration_inputs/registry_20261010.json'
REGISTRY_SHA = '4f477c04dc32e98480d2d74ef29927c3e324813b64ca3c832d1a31f4c9f07b22'
SNAPSHOT = HERE / 'frozen_20261010'
CATALOG_SHA = '35022c47d166473d63604e062f0561b372b98f788f53bdadaa923a90c8a47fee'
TECHNICAL = ('bone_marrow', 'tnbc', 'melanoma', 'lung_cancer')
STORIES = {'hnscc_story': 'figure_HNSCC/Comparison', 'melanoma_story': 'figure_melanoma/Comparison',
           'lung_story': 'story_lung_HLA_ABC/Comparison_senosain140k_20260921'}
PROFILES = ('source_scale', 'adtnorm_fixed_arcsinh')
PARAMETERS = dict(seed=42, rng_kind='Mersenne-Twister', normal_kind='Inversion', sample_kind='Rejection',
    theta=2, sigma=0.1, **{'lambda': 1}, nclust_policy='min(round(n_cells / 30), 100)',
    max_iter=10, early_stop=True, ncores=1, plot_convergence=False, return_object=False,
    alpha=0.2, tau=0, block_size=0.05, max_iter_cluster=20, epsilon_cluster=0.001, epsilon_harmony=0.01)
ADAPTER_FILES = ('harmony_direct_repro.R', 'run_harmony_direct.R', 'bootstrap_library.R',
    'upstream_lock.json', 'runtime_lock.csv', 'config.example.json', 'README.md',
    'PREPROCESSING_PROTOCOL.md', 'PARAMETER_PROVENANCE.md', 'vendor/LICENSE_NOTICE.md',
    'vendor/harmony_1.2.3.tar.gz', 'vendor/RcppProgress_0.4.2.tar.gz', 'vendor/RhpcBLASctl_0.23-42.tar.gz',
    'tests/smoke_test.R', 'tests/integrity_contract_test.R', 'tests/cli_contract_test.R')


class Builder:
    def __init__(self, source):
        self.source = Path(source).resolve(strict=True)
        self.files, self.links = {}, []

    def rel(self, path):
        path = Path(path)
        return path.relative_to(self.source).as_posix() if path.is_absolute() else path.as_posix()

    def add(self, path, copy=False):
        name = self.rel(path)
        value = {'source': name, **io.record(io.within(self.source, name))}
        if copy:
            value['copy'] = ('adapter/' + str(Path(name).relative_to('algorithm/harmony'))
                             if name.startswith('algorithm/harmony/') else 'study_sources/' + name)
        old = self.files.get(name)
        if old:
            io.check(io.within(self.source, name), old)
            if 'copy' in old: value['copy'] = old['copy']
        self.files[name] = value
        return name

    def json(self, path):
        return io.read_json(self.source / self.add(path, copy=True))

    def link(self, manifest, fields, target):
        name, dest = self.add(manifest, copy=True), self.add(target)
        value = io.read_json(self.source / name)
        for field in fields: value = value[field]
        if value != self.files[dest]['sha256']:
            raise ValueError(f'Historical hash differs: {name} {fields} -> {dest}')
        self.links.append({'manifest': name, 'fields': fields, 'target': dest})


def identity(source, spec):
    import numpy as np
    source = Path(source)
    inp = io.table(source / spec['input'])
    meta = io.table(source / spec['canonical']['evaluation_metadata'])
    markers = (source / spec['marker_file']).read_text().splitlines()
    if markers != spec['markers'] or list(inp.columns) != ['cell_id', 'dataset', 'unit_key', *markers]:
        raise ValueError('Input marker/column contract differs')
    if len(inp) != spec['cells'] or inp.cell_id.duplicated().any() or (inp.iloc[:, :3] == '').any().any():
        raise ValueError('Input cell identity differs')
    if not inp.iloc[:, :3].equals(meta[['cell_id', 'dataset', 'unit_key']]):
        raise ValueError('Input differs from canonical cell/dataset/unit order')
    if list(dict.fromkeys(inp.dataset)) != spec['datasets'] or inp.unit_key.nunique() != spec['units']:
        raise ValueError('Dataset order or unit count differs')
    if (inp.groupby('unit_key').dataset.nunique() != 1).any() or not np.isfinite(inp[markers].to_numpy(float)).all():
        raise ValueError('Invalid units or non-finite values')
    if spec['kind'] == 'story':
        ids = np.load(source / spec['ids'], allow_pickle=False)
        if not np.array_equal(ids, inp.cell_id) or not io.table(source / spec['metadata']).equals(inp.iloc[:, :3]):
            raise ValueError('Story IDs/fit metadata differ')
    out = io.table(source / spec['output'])
    if list(out.columns) != ['cell_id', *markers] or not np.array_equal(out.cell_id, inp.cell_id):
        raise ValueError('Historical output cell/marker order differs')
    if not np.isfinite(out[markers].to_numpy(float)).all(): raise ValueError('Non-finite historical output')
    if spec['kind'] == 'story':
        # Keep the historical pandas-default conversion, not the round-trip parser.
        import pandas as pd
        values = pd.read_csv(source / spec['output'], low_memory=False).iloc[:, 1:].to_numpy(float)
        old = np.load(source / spec['values'], allow_pickle=False)
        if old.dtype != np.float64 or not np.array_equal(old, values):
            raise ValueError('Historical NPY is not the documented CSV conversion')
    return {'cells': len(inp), 'markers': len(markers), 'units': int(inp.unit_key.nunique()),
            'ordered_identity_matches': True, 'finite_input_and_historical_output': True}


def build(source):
    if io.sha(REGISTRY) != REGISTRY_SHA: raise ValueError('Frozen common registry changed')
    registry = io.read_json(REGISTRY)
    cases = registry['cases']
    if isinstance(cases, dict): cases = list(cases.values())
    b, branches = Builder(source), []
    for name in ADAPTER_FILES: b.add('algorithm/harmony/' + name, copy=True)
    b.add('benchmark_results/harmony_benchmark_runner.py', copy=True)
    for case in cases:
        name = case['case']
        canonical = {}
        for role, row in case['files'].items():
            io.check(io.within(b.source, row['path']), row)
            canonical[role] = b.add(row['path'])
        common = dict(case=name, cells=case['expected_shape'][0], units=case['expected_units'],
                      markers=case['observed']['markers'], datasets=case['observed']['dataset_order'],
                      canonical=canonical, parameters={**PARAMETERS, 'resolved_nclust': 100})
        if name in TECHNICAL:
            base = Path('benchmark_results') / name / 'harmony'
            for f in ('config.json', 'run_all.py'): b.add(base / f, copy=True)
            for profile in PROFILES:
                pp, al = base / profile / 'preprocessing', base / profile / 'alignment'
                pm, am, cfg = b.json(pp / 'manifest.json'), b.json(al / 'data/manifest.json'), b.json(al / 'config.json')
                for directory in (pp, al / 'data'):
                    if b.json(directory / 'COMPLETE.json').get('status') != 'complete': raise ValueError('Partial history')
                    b.link(directory / 'COMPLETE.json', ['manifest_sha256'], directory / 'manifest.json')
                if pm.get('status') != 'complete' or am.get('status') != 'complete': raise ValueError('Partial history')
                if cfg['algorithm_parameters'] != PARAMETERS or am['algorithm']['resolved_parameters'] != common['parameters']:
                    raise ValueError('Historical Harmony parameters differ')
                if cfg['batch_levels'] != common['datasets'] or am['input']['markers'] != common['markers']:
                    raise ValueError('Historical Harmony dataset/marker order differs')
                input_file, marker = pp / 'data/combined_analysis_ready.csv.gz', pp / 'markers.txt'
                for filename in pm['outputs']:
                    target = pp / ('data/' + filename if filename == 'combined_analysis_ready.csv.gz' else filename)
                    b.link(pp / 'manifest.json', ['outputs', filename], target)
                    if filename != 'combined_analysis_ready.csv.gz': b.add(target, copy=True)
                for index, row in enumerate(pm['source_files']): b.link(pp / 'manifest.json', ['source_files', index, 'sha256'], row['path'])
                for key, target in [('input_sha256', input_file), ('marker_file_sha256', marker),
                                    ('preprocessing_record_sha256', pp / 'preprocessing_record.txt')]:
                    b.link(al / 'config.json', ['input_scale', key], target)
                for key, target in [('sha256', input_file), ('marker_file_sha256', marker),
                                    ('preprocessing_record_sha256', pp / 'preprocessing_record.txt')]:
                    b.link(al / 'data/manifest.json', ['input', key], target)
                b.link(al / 'data/manifest.json', ['config_file', 'sha256'], al / 'config.json')
                for key, row in am['implementation'].items(): b.link(al / 'data/manifest.json', ['implementation', key, 'sha256'], row['path'])
                for index, row in enumerate(am['output_files']): b.link(al / 'data/manifest.json', ['output_files', index, 'sha256'], al / 'data' / row['file'])
                spec = {**common, 'kind': 'technical', 'profile': profile, 'config': str(al / 'config.json'),
                    'input': str(input_file), 'marker_file': str(marker), 'record': str(pp / 'preprocessing_record.txt'),
                    'input_manifest': str(pp / 'manifest.json'), 'output_manifest': str(al / 'data/manifest.json'),
                    'output': str(al / 'data/harmony_adjusted_marker_coordinates.csv.gz'), 'formula': pm['formula']}
                branches.append(spec)
        else:
            base = Path(STORIES[name]); im = b.json(base / 'input/manifest.json'); cfg = b.json(base / 'run_config.json')
            rc = b.json(base / 'methods/harmony/R_COMPLETE.json'); done = b.json(base / 'methods/harmony/COMPLETE.json')
            if (cfg['datasets'] != common['datasets'] or cfg['markers'] != common['markers'] or
                rc['method'] != 'harmony' or rc['parameters'] != common['parameters'] or done['runtime'] != rc):
                raise ValueError('Historical story configuration differs')
            for f in ('run_r.R', 'run_r_methods.py', 'prepare.py'): b.add(base / 'scripts' / f, copy=True)
            for filename in im['output_sha256']: b.link(base / 'input/manifest.json', ['output_sha256', filename], base / 'input' / filename)
            for path in im['source_sha256']: b.link(base / 'input/manifest.json', ['source_sha256', path], path)
            b.link(base / 'run_config.json', ['input_manifest_sha256'], base / 'input/manifest.json')
            for fields, target in [(['input_sha256'], base / 'input/raw.csv.gz'),
                                  (['helper_sha256'], Path('algorithm/harmony/harmony_direct_repro.R')),
                                  (['output_sha256'], base / 'methods/harmony/corrected_expression.csv.gz')]:
                b.link(base / 'methods/harmony/R_COMPLETE.json', fields, target)
            for key, target in [('values_sha256', base / 'methods/harmony/values.npy'), ('cell_ids_sha256', base / 'input/cell_ids.npy'),
                                ('input_manifest_sha256', base / 'input/manifest.json')]:
                b.link(base / 'methods/harmony/COMPLETE.json', [key], target)
            for path in sorted((b.source / base / 'methods/harmony').iterdir()):
                if path.is_file(): b.add(path, copy=path.suffix in ('.json', '.txt'))
            if 'data.table_1.17.8' not in (b.source / base / 'methods/harmony/sessionInfo.txt').read_text():
                raise ValueError('Historical story data.table version not established')
            branches.append({**common, 'kind': 'story', 'profile': 'source_scale', 'base': str(base),
                'config': str(base / 'run_config.json'), 'input': str(base / 'input/raw.csv.gz'),
                'marker_file': str(base / 'input/markers.txt'), 'metadata': str(base / 'input/fit_metadata.csv.gz'),
                'ids': str(base / 'input/cell_ids.npy'), 'input_manifest': str(base / 'input/manifest.json'),
                'output_manifest': str(base / 'methods/harmony/R_COMPLETE.json'),
                'output': str(base / 'methods/harmony/corrected_expression.csv.gz'), 'values': str(base / 'methods/harmony/values.npy'),
                'formula': 'Identity on frozen raw.csv.gz source-scale inputs; no additional transform in Harmony.'})
    for spec in branches: spec['identity_check'] = identity(b.source, spec)
    # Catch mutation during the longer table checks, before publishing any snapshot.
    for name, row in b.files.items(): io.check(io.within(b.source, name), row)
    if io.sha(REGISTRY) != REGISTRY_SHA: raise ValueError('Registry changed during scan')
    return {'schema': 1, 'status': 'complete', 'registry_sha256': REGISTRY_SHA,
            'scope': 'Frozen method-ready inputs, not provider reconstruction or real rerun',
            'branches': branches, 'files': sorted(b.files.values(), key=lambda r: r['source']), 'links': b.links}


def freeze(source, output):
    catalog = build(source)
    output = io.reserve(output, (source, REGISTRY, HERE / 'inventory.py', HERE / 'harmony_io.py'))
    try:
        for row in catalog['files']:
            if 'copy' in row: io.copy_new(io.within(source, row['source']), output / row['copy'], row)
        io.copy_new(REGISTRY, output / 'registry.json', io.record(REGISTRY))
        for row in catalog['files']: io.check(io.within(source, row['source']), row)
        if io.sha(REGISTRY) != REGISTRY_SHA or io.sha(output / 'registry.json') != REGISTRY_SHA:
            raise ValueError('Registry changed during freeze')
        io.create_json(output / 'catalog.json', catalog)
        io.payload(output, {r['copy']: r for r in catalog['files'] if 'copy' in r}, ('registry.json', 'catalog.json'))
        io.create_json(output / 'COMPLETE.json', {'status': 'complete', 'manifest_sha256': io.sha(output / 'catalog.json')})
    except BaseException as error:
        io.failure(output, error); raise
    return output


def verify_local(snapshot=None):
    root = SNAPSHOT if snapshot is None else Path(snapshot)
    if io.sha(root / 'catalog.json') != CATALOG_SHA: raise ValueError('Frozen catalog anchor changed')
    catalog = io.complete(root, 'catalog.json')
    if catalog.get('status') != 'complete' or catalog['registry_sha256'] != REGISTRY_SHA or io.sha(root / 'registry.json') != REGISTRY_SHA:
        raise ValueError('Snapshot registry or status differs')
    io.payload(root, {r['copy']: r for r in catalog['files'] if 'copy' in r}, ('registry.json', 'catalog.json', 'COMPLETE.json'))
    return catalog


def verify_source(source, snapshot=None):
    catalog = verify_local(snapshot)
    for row in catalog['files']: io.check(io.within(source, row['source']), row)
    for link in catalog['links']:
        value = io.read_json(io.within(source, link['manifest']))
        for key in link['fields']: value = value[key]
        if value != io.sha(io.within(source, link['target'])): raise ValueError('Historical hash relation differs')
    for spec in catalog['branches']: identity(source, spec)
    return {'files': len(catalog['files']), 'copies': sum('copy' in r for r in catalog['files']),
            'hash_relations': len(catalog['links']), 'branches': len(catalog['branches'])}


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('command', choices=('freeze', 'verify'))
    p.add_argument('--source-root', type=Path, required=True); p.add_argument('--output', type=Path, default=SNAPSHOT)
    a = p.parse_args()
    print(freeze(a.source_root, a.output) if a.command == 'freeze' else verify_source(a.source_root, a.output))


if __name__ == '__main__': main()
