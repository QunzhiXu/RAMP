"""Create-only source/contract inventory for marker-value evaluation; never imports legacy runners."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

HERE = Path(__file__).resolve().parent
CASES = ('bone_marrow', 'tnbc', 'melanoma', 'lung_cancer')
PERCENTILE = 'Main Figure/celltype_percentile_shift_20260914'
RECON = 'Main Figure/fastmnn_reconstructed_audit_20260914'
VALUE = 'Main Figure/Figure 2/value_evaluation_20260914'
FIGURE2 = 'Main Figure/Figure 2/lung_reference_update_20260921'
FIELDS = {'input_sha256', 'all_input_sha256', 'source_sha256', 'input_hashes'}
HEX = re.compile(r'^[0-9a-f]{64}$')


def require(condition, message):
    if not condition: raise ValueError(message)


def record(path):
    with Path(path).open('rb') as f: digest = hashlib.file_digest(f, 'sha256').hexdigest()
    return {'sha256': digest, 'bytes': Path(path).stat().st_size}


def read(path): return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as f: json.dump(value, f, indent=2, allow_nan=False); f.write('\n')


def reserve(path, protected=()):
    path = Path(path).absolute()
    if path.exists() or path.is_symlink(): raise FileExistsError(path)
    resolved = path.resolve()
    for source in protected:
        source = Path(source).resolve()
        require(not (resolved == source or resolved.is_relative_to(source) or source.is_relative_to(resolved)), 'Overlapping output')
    path.parent.mkdir(parents=True, exist_ok=True); path.mkdir(exist_ok=False)
    return path.resolve()


def bind(root, historical):
    name = str(historical)
    if '/final_xu/' in name: name = name.split('/final_xu/', 1)[1]
    path = Path(name)
    require(not path.is_absolute() and '..' not in path.parts, 'Unsafe historical path')
    target = Path(root) / path
    require(target.resolve().is_relative_to(Path(root).resolve()) and not target.is_symlink(), 'Source outside root')
    return target


def collect(root):
    root = Path(root).resolve(strict=True)
    sources = {}; copies = set(); links = []; owners = []
    def add(path, copy=False, expected=None):
        path = Path(path); path = path if path.is_absolute() else root / path
        require(path.resolve().is_relative_to(root) and not path.is_symlink(), 'Unsafe source')
        key = path.relative_to(root).as_posix(); rec = record(path)
        require(key not in sources or sources[key] == rec, 'Source changed during collection')
        if expected is not None: require(rec['sha256'] == expected, 'Historical hash mismatch: ' + key)
        sources[key] = rec
        if copy: copies.add(key)
        return key
    # Scripts, small tabular results and declarations, not arrays, figures, caches or provider data.
    for stem in (PERCENTILE, RECON, VALUE):
        for path in sorted((root / stem).rglob('*')):
            if not path.is_file() or any(p.startswith('.') or p == '__pycache__' for p in path.relative_to(root / stem).parts): continue
            rel = path.relative_to(root / stem)
            is_script = path.suffix in ('.py', '.R', '.md', '.json')
            is_table = (path.name.endswith(('.csv', '.csv.gz')) and
                        not path.name.startswith(('cell_metadata', 'display_cells')) and
                        (len(rel.parts) == 1 or rel.parts[0] in ('results', 'summary', 'value_evaluation')))
            if is_script or is_table:
                key = add(path, copy=True)
                if path.suffix == '.json': owners.append(key)
    for name in ('Main Figure/pair_inversion_boxplot_20260914/compute_pair_inversions.py',
                 'Main Figure/redesign_full_icon_20260914/all_celltype_statistics.csv',
                 FIGURE2 + '/Panel E/build_panel.py', FIGURE2 + '/Panel F/build_panel.py', FIGURE2 + '/common.py',
                 'Main Figure/Figure 2/restored_means_metric_review_20260914/Panel E/prepare_data.py',
                 'Main Figure/Figure 2/restored_means_metric_review_20260914/common.py',
                 'Main Figure/ridge_compact_20260924/build_figure.py',
                 'Main Figure/ridge_compact_20260924/median_shift_box_statistics.csv',
                 'Main Figure/Figure 2/seven_benchmarks_main23_20260914/Panel F/plotted_values.csv',
                 'Main Figure/Figure 2/seven_benchmarks_main23_20260914/Panel F/equal_benchmark_means.csv',
                 FIGURE2 + '/Panel F/plotted_values.csv', FIGURE2 + '/Panel F/equal_benchmark_means.csv',
                 FIGURE2 + '/Panel E/data_manifest.json', FIGURE2 + '/Panel E/density_coordinates.csv',
                 FIGURE2 + '/Panel E/gate_statistics.csv',
                 'figure_raw_labels_20260924/MAIN_README_NEXT.md'):
        key = add(name, copy=True)
        if Path(name).suffix == '.json': owners.append(key)
    for case in CASES:
        for name in ('reconstructed.npy', 'cosine_normalized.npy', 'pca_reconstructed.npy', 'cell_metadata.csv.gz', 'markers.txt'):
            add(Path(RECON) / case / name)
    def scan(node, owner, fields=()):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in FIELDS and isinstance(value, dict):
                    for historical, sha in value.items():
                        require(isinstance(sha, str) and HEX.fullmatch(sha), 'Malformed historical digest')
                        target = add(bind(root, historical), expected=sha)
                        links.append({'owner': owner, 'fields': [*fields, key, historical], 'target': target, 'sha256': sha})
                else: scan(value, owner, (*fields, key))
        elif isinstance(node, list):
            for index, value in enumerate(node): scan(value, owner, (*fields, index))
    for owner in owners: scan(read(root / owner), owner)
    return {'sources': sources, 'copies': sorted(copies), 'hash_links': links,
            'scope': 'existing value-evaluation sources and results, not full figure rendering or a new analysis run',
            'cases': list(CASES), 'source_root': str(root), 'schema': 1}


def verify_sources(catalog, root):
    for name, rec in catalog['sources'].items(): require(record(bind(root, name)) == rec, 'Source changed: ' + name)
    for link in catalog['hash_links']:
        value = read(bind(root, link['owner']))
        for field in link['fields']: value = value[field]
        require(value == link['sha256'] == catalog['sources'][link['target']]['sha256'], 'Historical hash field differs')
    return {'sources': len(catalog['sources']), 'hash_links': len(catalog['hash_links'])}


def verify(snapshot, catalog_sha, source_root=None, complete=True):
    snapshot = Path(snapshot).resolve(strict=True)
    require(record(snapshot / 'catalog.json')['sha256'] == catalog_sha, 'Catalog anchor differs')
    catalog = read(snapshot / 'catalog.json')
    actual = set()
    for path in snapshot.rglob('*'):
        require(not path.is_symlink(), 'Unexpected snapshot symlink')
        if path.is_file(): actual.add(path.relative_to(snapshot).as_posix())
    expected = {'sources/' + p for p in catalog['copies']} | {'catalog.json'} | ({'COMPLETE.json'} if complete else set())
    require(actual == expected, 'Snapshot inventory differs')
    for name in catalog['copies']:
        require(record(bind(snapshot / 'sources', name)) == catalog['sources'][name], 'Snapshot payload differs')
    if complete:
        done = read(snapshot / 'COMPLETE.json')
        require(done.get('status') == 'complete' and done['catalog'] == record(snapshot / 'catalog.json'), 'Snapshot incomplete')
    if source_root is not None: verify_sources(catalog, source_root)
    return {'copies': len(catalog['copies']), 'sources': len(catalog['sources']), 'hash_links': len(catalog['hash_links']),
            'catalog_sha256': catalog_sha}


def freeze(root, output):
    root = Path(root).resolve(strict=True); catalog = collect(root)
    output = reserve(output, [root])
    try:
        for name in catalog['copies']:
            source = bind(root, name); target = bind(output / 'sources', name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open('rb') as src, target.open('xb') as dst: shutil.copyfileobj(src, dst)
            require(record(target) == catalog['sources'][name], 'Source changed during copying')
        verify_sources(catalog, root)
        write(output / 'catalog.json', catalog)
        anchor = record(output / 'catalog.json')['sha256']
        summary = verify(output, anchor, complete=False)
        write(output / 'COMPLETE.json', {'status': 'complete', 'catalog': record(output / 'catalog.json')})
        # Completion is the final operation: no subsequent validation can turn a
        # completed snapshot into a contradictory COMPLETE + FAILED directory.
        return summary
    except BaseException as error:
        write(output / 'FAILED.json', {'error': repr(error)}); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['freeze', 'verify'])
    parser.add_argument('--source-root', type=Path); parser.add_argument('--output', type=Path)
    parser.add_argument('--snapshot', type=Path); parser.add_argument('--catalog-sha')
    a = parser.parse_args()
    if a.action == 'freeze': result = freeze(a.source_root, a.output)
    else:
        require(a.catalog_sha is not None, 'Supply the independently recorded catalog hash')
        result = verify(a.snapshot, a.catalog_sha, a.source_root)
    print(json.dumps(result, indent=2))
