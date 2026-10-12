"""Create-only snapshots; verification never writes or launches legacy code."""
import argparse
import json
from pathlib import Path
import shutil

import contract as c


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')


def reserve(path, protected):
    path = Path(path).absolute()
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path = path.resolve()
    for source in protected:
        source = Path(source).resolve()
        c.require(not (path == source or path.is_relative_to(source) or source.is_relative_to(path)),
                  'Output overlaps protected source')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(exist_ok=False)
    return path


def child(root, name):
    rel = Path(name)
    c.require(bool(rel.parts) and not rel.is_absolute() and '..' not in rel.parts, 'Unsafe relative path')
    root = Path(root).resolve(strict=True)
    path = root / rel
    for ancestor in (path, *path.parents):
        if ancestor == root: break
        c.require(not ancestor.is_symlink(), 'Symlink in snapshot path')
    c.require(path.resolve().is_relative_to(root), 'Escaping snapshot path')
    return path


def check(path, expected):
    c.require(c.record(path) == expected, 'File fingerprint differs: ' + str(path))


def verify_sources(doc, source):
    for name, expected in doc['sources'].items():
        path = Path(source) / name
        c.require(path.resolve().is_relative_to(Path(source).resolve()), 'Escaping source path')
        check(path, expected)
    for link in doc['historical_hash_links']:
        value = c.read(Path(source) / link['owner'])
        for part in link['field']: value = value[part]
        c.require(value == link['sha256'] == doc['sources'][link['target']]['sha256'], 'Hash link differs')


def verify_payload(output, doc, complete=False):
    output = Path(output)
    expected = {'catalog.json', 'registry.json'} | {'sources/' + n for n in doc['copies']}
    if complete: expected.add('COMPLETE.json')
    observed = set()
    for p in output.rglob('*'):
        c.require(not p.is_symlink(), 'Snapshot contains symlink')
        if p.is_file(): observed.add(p.relative_to(output).as_posix())
    c.require(expected == observed, 'Snapshot inventory differs')
    c.require(c.record(output / 'registry.json')['sha256'] == c.REGISTRY_SHA == doc['registry_sha256'],
              'Registry anchor differs')
    for name in doc['copies']:
        check(child(output, 'sources/' + name), doc['sources'][name])


def freeze(source, output):
    # CLI only writes beneath this NEW module, never inside accepted modules.
    output = Path(output).absolute()
    c.require(output.resolve().is_relative_to(c.HERE), 'Output must be under integration_evaluation')
    output = reserve(output, [source, c.REGISTRY, c.HERE / 'contract.py', c.HERE / 'inventory.py'])
    try:
        registry_bytes = c.REGISTRY.read_bytes()
        c.require(c.record(c.REGISTRY)['sha256'] == c.REGISTRY_SHA, 'Registry anchor differs')
        doc = c.collect(source)
        for name in doc['copies']:
            dest = child(output, 'sources/' + name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with (Path(source) / name).open('rb') as src, dest.open('xb') as dst:
                shutil.copyfileobj(src, dst)
            check(dest, doc['sources'][name])
        with (output / 'registry.json').open('xb') as handle:
            handle.write(registry_bytes)
        verify_sources(doc, source)
        c.require(c.REGISTRY.read_bytes() == registry_bytes, 'Registry changed during freeze')
        write(output / 'catalog.json', doc)
        verify_payload(output, doc)
        write(output / 'COMPLETE.json', {'status': 'complete', 'catalog_sha256': c.record(output / 'catalog.json')['sha256']})
        return doc
    except BaseException as error:
        write(output / 'FAILED.json', {'error': repr(error)})
        raise


def verify(output, source=None):
    output = Path(output)
    done = c.read(output / 'COMPLETE.json')
    c.require(done.get('status') == 'complete', 'Snapshot not complete')
    c.require(done['catalog_sha256'] == c.record(output / 'catalog.json')['sha256'], 'Catalog digest differs')
    doc = c.read(output / 'catalog.json')
    verify_payload(output, doc, complete=True)
    if source is not None: verify_sources(doc, source)
    return {'cases': len(doc['cases']), 'variants': sum(len(x['variants']) for x in doc['cases']),
            'sources': len(doc['sources']), 'copies': len(doc['copies']), 'hash_links': len(doc['historical_hash_links'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['freeze', 'verify'])
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--source-root', type=Path)
    args = parser.parse_args()
    if args.action == 'freeze':
        if args.source_root is None: parser.error('freeze requires --source-root')
        freeze(args.source_root.resolve(strict=True), args.snapshot)
    print(json.dumps(verify(args.snapshot, args.source_root), indent=2))


if __name__ == '__main__':
    main()
