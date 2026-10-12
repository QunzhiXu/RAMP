"""Freeze/verify cyCombine code and seven historical correction contracts, never fit.

The catalog covers the correction-stage dependencies explicitly enumerated below.
Provider extraction, UMAP/metric execution and a clean R installation are separate.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parents[1]
REGISTRY = STUDY / 'benchmarks/integration_inputs/registry_20261010.json'
REGISTRY_SHA = '4f477c04dc32e98480d2d74ef29927c3e324813b64ca3c832d1a31f4c9f07b22'
SNAPSHOT = HERE / 'frozen_20261010'
TECHNICAL = ('bone_marrow', 'tnbc', 'melanoma', 'lung_cancer')
STORIES = {
    'lung_story': 'story_lung_HLA_ABC/Comparison_senosain140k_20260921',
    'melanoma_story': 'figure_melanoma/Comparison',
    'hnscc_story': 'figure_HNSCC/Comparison',
}
PARAMETERS = dict(xdim=8, ydim=8, rlen=10, seed=840, norm_method='rank',
                  ties_method='average', parametric=True, covariate=None, anchor=None)
ADAPTER = Path('algorithm/cycombine')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def create_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def within(root, relative):
    root = Path(root).resolve(strict=True)
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Not a safe relative path: ' + str(relative))
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Not an internal file: ' + str(path))
    return path


def reserve(output, protected=()):
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    target = output.resolve()
    for item in protected:
        item = Path(item).resolve()
        if target == item or target.is_relative_to(item) or item.is_relative_to(target):
            raise ValueError('Output overlaps protected input: ' + str(item))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)  # Atomic ownership, including preflight races.
    return output.resolve(strict=True)


def record(path):
    return {'sha256': sha(path), 'bytes': Path(path).stat().st_size}


def check(path, expected):
    if record(path) != {k: expected[k] for k in ('sha256', 'bytes')}:
        raise ValueError('File changed: ' + str(path))


def copy_new(source, target, expected):
    target.parent.mkdir(parents=True, exist_ok=True)
    with Path(source).open('rb') as src, target.open('xb') as dst:
        shutil.copyfileobj(src, dst)
    check(target, expected)


def field(document, keys):
    value = document
    for key in keys:
        value = value[key]
    return value


def original_relative(value, base):
    """Historical absolute paths may only be re-rooted at their final_xu segment."""
    path = Path(value)
    if path.is_absolute():
        if str(path).count('/final_xu/') != 1:
            raise ValueError('External historical path: ' + str(path))
        path = Path(str(path).split('/final_xu/', 1)[1])
    else:
        path = Path(base) / path
    # Relative ../ components are resolved by caller and then confined to source root.
    return path


class CatalogBuilder:
    def __init__(self, source):
        self.source = Path(source).resolve(strict=True)
        self.files = {}
        self.links = []

    def add(self, relative, copy_to=None):
        path = (self.source / relative).resolve(strict=True)
        if not path.is_relative_to(self.source) or not path.is_file():
            raise ValueError('Source escapes tree')
        relative = path.relative_to(self.source).as_posix()
        info = self.files.setdefault(relative, {'source': relative, **record(path)})
        if copy_to is not None:
            if 'copy' in info and info['copy'] != copy_to:
                raise ValueError('Ambiguous snapshot destination')
            info['copy'] = copy_to
        return relative

    def link(self, manifest, keys, target):
        manifest = self.add(manifest)
        target = self.add(target)
        digest = field(read_json(self.source / manifest), keys)
        if digest != self.files[target]['sha256']:
            raise ValueError('Historical JSON hash mismatch: ' + manifest + ' ' + repr(keys))
        self.links.append({'manifest': manifest, 'field': keys, 'target': target, 'sha256': digest})

    def document(self, relative):
        relative = self.add(relative, 'study_sources/' + str(relative))
        return read_json(self.source / relative)

    def pairs(self, manifest, entries, prefix, base):
        for name, info in entries.items():
            self.link(manifest, [*prefix, name, 'sha256'], original_relative(info['path'], base))


def collect(source, registry=REGISTRY):
    registry_bytes = Path(registry).read_bytes()
    if hashlib.sha256(registry_bytes).hexdigest() != REGISTRY_SHA:
        raise ValueError('Accepted input registry changed')
    builder = CatalogBuilder(source)
    rootspec = json.loads(registry_bytes)
    contracts = []
    for path in sorted((builder.source / ADAPTER).rglob('*')):
        rel = path.relative_to(builder.source / ADAPTER)
        if path.is_file() and not any(p.startswith('.') for p in rel.parts):
            builder.add(ADAPTER / rel, 'adapter/' + rel.as_posix())
    for name in ('validated_direct_runtime_packages.csv', 'bootstrap_sessionInfo.txt', '.cycombine_source_sha256'):
        builder.add(ADAPTER / '.Rlib' / name, 'environment/' + name)
    for spec in rootspec['cases']:
        name = spec['case']
        canonical = {}
        for role in ('fit_metadata', 'evaluation_metadata', 'markers'):
            info = spec['files'][role]
            check(builder.source / info['path'], info)
            canonical[role] = builder.add(info['path'])
        if name in TECHNICAL:
            base = Path('benchmark_results') / name / 'cycombine'
            # Source snapshots only: no arrays, CSV matrices, figures or fit RDS copies.
            for path in sorted((builder.source / base).rglob('*')):
                if path.is_file() and path.suffix in ('.R', '.py', '.json', '.md', '.txt', '.diff'):
                    builder.add(path.relative_to(builder.source), 'study_sources/' + str(path.relative_to(builder.source)))
            mp = base / 'alignment/data/manifest.json'
            m = builder.document(mp)
            cp = base / 'alignment/data/COMPLETE.json'
            complete = builder.document(cp)
            if complete.get('status') != 'complete':
                raise ValueError('Incomplete correction')
            builder.link(cp, ['manifest_sha256'], mp)
            builder.pairs(mp, m['inputs'], ['inputs'], mp.parent)
            builder.pairs(mp, m['outputs'], ['outputs'], mp.parent)
            for key in ('runner', 'wrapper', 'upstream_lock'):
                leaf = {'runner': 'run_cycombine.R', 'wrapper': 'cycombine_repro.R', 'upstream_lock': 'upstream_lock.json'}[key]
                builder.link(mp, ['local_source_sha256', key], ADAPTER / leaf)
            pp = base / 'preprocessing/preprocessing_manifest.json'
            pm = builder.document(pp)
            ready = base / 'preprocessing/data/combined_analysis_ready.csv.gz'
            builder.link(pp, ['combined_input_sha256'], ready)
            for prefix in ('correction_marker', 'marker_harmonization_manifest', 'cell_selection_manifest'):
                path_key = prefix + '_file'
                builder.link(pp, [prefix + '_sha256'], original_relative(pm[path_key], pp.parent))
            for index, dataset in enumerate(pm['datasets']):
                for key in ('input_matrix', 'processing_code'):
                    builder.link(pp, ['datasets', index, key + '_sha256'], original_relative(dataset[key + '_file'], pp.parent))
            markers = base / 'preprocessing/markers.txt'
            output = base / 'alignment/data/corrected_expression.csv.gz'
            metadata = base / 'alignment/data/cell_metadata_label_free.csv.gz'
            config = base / 'alignment/config.json'
            datasets = m['batch_levels']
            parameters = m['algorithm_parameters']
            protocol = pm['datasets']
            helper = str(ADAPTER / 'cycombine_repro.R')
            value_npy = None
        else:
            base = Path(STORIES[name])
            for relative in ('scripts/prepare.py', 'scripts/run_r.R', 'scripts/run_r_methods.py',
                             'scripts/cycombine_repro.R', 'run_config.json', 'input/manifest.json',
                             'methods/cycombine/COMPLETE.json', 'methods/cycombine/R_COMPLETE.json',
                             'methods/cycombine/sessionInfo.txt'):
                p = base / relative
                builder.add(p, 'study_sources/' + str(p))
            for leaf in ('scripts/cycombine_repro.R.diff', 'scripts/adapter_manifest.json'):
                if (builder.source / base / leaf).is_file():
                    builder.add(base / leaf, 'study_sources/' + str(base / leaf))
            config = base / 'run_config.json'
            cfg = read_json(builder.source / config)
            mp = base / 'input/manifest.json'
            m = read_json(builder.source / mp)
            cp = base / 'methods/cycombine/COMPLETE.json'
            done = read_json(builder.source / cp)
            rp = base / 'methods/cycombine/R_COMPLETE.json'
            runtime = read_json(builder.source / rp)
            if runtime != done['runtime'] or done['method'] != 'cycombine':
                raise ValueError('Story completion records disagree')
            helper = str(base / 'scripts/cycombine_repro.R')
            ready = base / 'input/analysis_ready.csv.gz'
            output = base / 'methods/cycombine/corrected_expression.csv.gz'
            markers = base / 'input/markers.txt'
            value_npy = base / 'methods/cycombine/values.npy'
            metadata = base / 'input/fit_metadata.csv.gz'
            for leaf in ('analysis_ready.csv.gz', 'analysis_ready.npy', 'cell_ids.npy', 'markers.txt', 'fit_metadata.csv.gz'):
                builder.link(mp, ['output_sha256', leaf], base / 'input' / leaf)
            for keys, target in ((['input_sha256'], ready), (['output_sha256'], output), (['helper_sha256'], helper)):
                builder.link(rp, keys, target)
            for keys, target in ((['values_sha256'], value_npy), (['cell_ids_sha256'], base / 'input/cell_ids.npy'), (['input_manifest_sha256'], mp)):
                builder.link(cp, keys, target)
            builder.link(config, ['input_manifest_sha256'], mp)
            for p in sorted((builder.source / base / 'methods/cycombine').iterdir()):
                if p.is_file():
                    builder.add(p.relative_to(builder.source))
            parameters = runtime['parameters']
            datasets = cfg['datasets']
            protocol = m['preprocessing']
        if parameters != PARAMETERS:
            raise ValueError('Unexpected cyCombine scientific parameters: ' + name)
        contracts.append({'case': name, 'kind': spec['kind'], 'cells': spec['expected_shape'][0],
                          'markers': spec['observed']['markers'], 'datasets': datasets,
                          'input': builder.add(ready), 'output': builder.add(output),
                          'metadata': builder.add(metadata), 'marker_file': builder.add(markers),
                          'config': builder.add(config), 'helper': helper,
                          'values_npy': builder.add(value_npy) if value_npy else None,
                          'canonical': canonical, 'parameters': parameters, 'preprocessing': protocol})
    return {'schema': 1, 'scope': 'Correction-stage code/configuration and existing result provenance; not a provider dependency closure',
            'registry_sha256': REGISTRY_SHA, 'cases': contracts,
            'files': sorted(builder.files.values(), key=lambda x: x['source']), 'links': builder.links}


def _verify_payload(snapshot, *, completed):
    """Validate all payloads before publishing, or within a completed snapshot."""
    snapshot = Path(snapshot).resolve(strict=True)
    catalog = read_json(snapshot / 'catalog.json')
    expected = {'catalog.json', 'accepted_registry.json'}
    if completed:
        expected.add('COMPLETE.json')
    if sha(snapshot / 'accepted_registry.json') != catalog['registry_sha256'] or catalog['registry_sha256'] != REGISTRY_SHA:
        raise ValueError('Wrong registry anchor')
    for row in catalog['files']:
        if 'copy' in row:
            check(within(snapshot, row['copy']), row)
            expected.add(row['copy'])
    actual = {p.relative_to(snapshot).as_posix() for p in snapshot.rglob('*') if p.is_file() or p.is_symlink()}
    if actual != expected:
        raise ValueError('Unexpected/missing snapshot files')
    return catalog


def verify_local(snapshot=SNAPSHOT):
    snapshot = Path(snapshot).resolve(strict=True)
    complete = read_json(snapshot / 'COMPLETE.json')
    if complete.get('status') != 'complete' or sha(snapshot / 'catalog.json') != complete.get('catalog_sha256'):
        raise ValueError('Incomplete or changed snapshot')
    return _verify_payload(snapshot, completed=True)


def verify_sources(source, snapshot=SNAPSHOT):
    source = Path(source).resolve(strict=True)
    catalog = verify_local(snapshot)
    for row in catalog['files']:
        check(within(source, row['source']), row)
    for link in catalog['links']:
        if field(read_json(within(source, link['manifest'])), link['field']) != link['sha256'] or sha(within(source, link['target'])) != link['sha256']:
            raise ValueError('JSON field relation changed: ' + str(link))
    return {'source_files': len(catalog['files']), 'json_hash_links': len(catalog['links']),
            'snapshots': sum('copy' in r for r in catalog['files'])}


def verify_wrappers(snapshot=SNAPSHOT):
    snapshot = Path(snapshot).resolve(strict=True)
    verify_local(snapshot)
    adapter = snapshot / 'adapter'
    lock = read_json(adapter / 'upstream_lock.json')
    if sha(adapter / lock['source_archive']['path']) != lock['source_archive']['sha256']:
        raise ValueError('Wrong vendored upstream archive')
    base = (adapter / 'cycombine_repro.R').read_text()
    old_guard = 'length(batch_levels) != 3L'
    old_message = 'requires exactly three ordered, unique dataset batches.'
    if base.count(old_guard) != 1 or base.count(old_message) != 1:
        raise ValueError('Unexpected original batch guard')
    expected = base.replace(old_guard, '!(length(batch_levels) %in% c(2L, 3L))').replace(
        old_message, 'requires two or three ordered, unique dataset batches.')
    for relative in STORIES.values():
        if (snapshot / 'study_sources' / relative / 'scripts/cycombine_repro.R').read_text() != expected:
            raise ValueError('Story adapter changes more than the batch guard')
    return {'version': lock['package_version'], 'commit': lock['publication_commit'],
            'story_copies': 3, 'only_batch_count_guard_and_message_differ': True,
            'archive_sha256': lock['source_archive']['sha256']}


def table(path):
    import pandas as pd
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def verify_results(source, snapshot=SNAPSHOT):
    import numpy as np
    source = Path(source).resolve(strict=True)
    catalog = verify_local(snapshot)
    reports = []
    for spec in catalog['cases']:
        md = table(within(source, spec['canonical']['evaluation_metadata']))
        fitmd = table(within(source, spec['canonical']['fit_metadata']))
        inp = table(within(source, spec['input']))
        out = table(within(source, spec['output']))
        meta = table(within(source, spec['metadata']))
        ids = md.cell_id.to_numpy()
        n = spec['cells']
        if len(ids) != n or len(set(ids)) != n or len(inp) != n or len(out) != n or len(meta) != n:
            raise ValueError('Wrong cell count/IDs: ' + spec['case'])
        for current in (inp, out, meta):
            if not np.array_equal(current.cell_id.to_numpy(), ids):
                raise ValueError('Cell order mismatch: ' + spec['case'])
        for key in ('dataset', 'unit_key'):
            if not np.array_equal(inp[key], fitmd[key]) or not np.array_equal(md[key], fitmd[key]):
                raise ValueError('Canonical dataset/unit mismatch: ' + spec['case'])
        columns = ('batch', 'sample') if spec['kind'] == 'technical' else ('dataset', 'unit_key')
        if not np.array_equal(meta[columns[0]], inp.dataset) or not np.array_equal(meta[columns[1]], inp.unit_key):
            raise ValueError('Result metadata mismatch')
        markers = within(source, spec['marker_file']).read_text().splitlines()
        if markers != spec['markers'] or markers != within(source, spec['canonical']['markers']).read_text().splitlines():
            raise ValueError('Marker order mismatch')
        if list(out.columns) != ['cell_id', *markers]:
            raise ValueError('Output marker order mismatch')
        x = inp[markers].to_numpy(float)
        y = out[markers].to_numpy(float)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Non-finite correction matrix')
        if list(dict.fromkeys(inp.dataset)) != spec['datasets']:
            raise ValueError('Dataset order mismatch')
        difference = None
        if spec['values_npy']:
            values = np.load(within(source, spec['values_npy']), allow_pickle=False)
            difference = float(np.max(np.abs(values - y)))
            # Historical conversion uses pandas' default parser; CSV round-trip can
            # differ at the last bit. Require strict scale-relative round-trip error.
            if values.shape != y.shape or not np.allclose(values, y, rtol=1e-14, atol=1e-14):
                raise ValueError('Historical NPY differs from corrected CSV')
        reports.append({'case': spec['case'], 'cells': n, 'markers': len(markers),
                        'units': int(inp.unit_key.nunique()), 'ordered_ids_verified': True,
                        'datasets_and_units_verified': True, 'finite': True,
                        'csv_npy_max_abs_difference': difference})
    return reports


def freeze(source, output):
    source = Path(source).resolve(strict=True)
    output = reserve(output, (source, REGISTRY))
    try:
        registry_record = record(REGISTRY)
        if registry_record['sha256'] != REGISTRY_SHA:
            raise ValueError('Accepted input registry changed')
        catalog = collect(source, registry=REGISTRY)
        for row in catalog['files']:
            if 'copy' in row:
                copy_new(within(source, row['source']), output / row['copy'], row)
        copy_new(REGISTRY, output / 'accepted_registry.json', registry_record)
        for row in catalog['files']:
            check(within(source, row['source']), row)
        create_json(output / 'catalog.json', catalog)
        _verify_payload(output, completed=False)
        check(REGISTRY, registry_record)
        # Publication is the last operation: no fallible validation follows it.
        create_json(output / 'COMPLETE.json', {'status': 'complete', 'catalog_sha256': sha(output / 'catalog.json')})
    except BaseException as error:
        create_json(output / 'FAILED.json', {'error': repr(error), 'partial_retained': True})
        raise
    return output


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['freeze', 'verify', 'list'])
    p.add_argument('--source-root', type=Path)
    p.add_argument('--snapshot', type=Path, default=SNAPSHOT)
    p.add_argument('--output', type=Path)
    args = p.parse_args(argv)
    if args.action == 'freeze':
        if args.source_root is None or args.output is None:
            p.error('freeze requires --source-root and a new --output')
        print(freeze(args.source_root, args.output))
    else:
        catalog = verify_local(args.snapshot)
        if args.action == 'list':
            print(json.dumps([{k: c[k] for k in ('case', 'cells', 'datasets', 'markers')} for c in catalog['cases']], indent=2))
        elif args.source_root:
            print(json.dumps({'hash_checks': verify_sources(args.source_root, args.snapshot),
                              'results': verify_results(args.source_root, args.snapshot)}, indent=2))
        else:
            print('Verified local source snapshot (external inputs/results not checked).')


if __name__ == '__main__':
    main()
