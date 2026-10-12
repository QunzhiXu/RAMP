"""Figure 3 source/summary freeze, NOT a training runner or dependency closure."""
import argparse
from pathlib import Path
from safe_io import require, record, read, write, member, reserve, copy_checked, files

HERE = Path(__file__).resolve().parent
ROOT = 'cross_platform_trainability_v2_20260919'
FINAL = 'releases/figure3_bc_update_20260922'
J1 = 'releases/j1_scope_update_20260921_r3'
BASE = 'results/formal_v0_5/summary_1790002465230065000'
J2 = 'exploration_lung_new140k_j2_20260922'
UPDATES = {
    'tnbc': 'exploration_tnbc_five_classes_20260921',
    'melanoma': 'exploration_melanoma_three_classes_20260921',
    'lung_cancer': 'exploration_lung_original_four_classes_20260921',
    'lung_story': 'exploration_lung_new140k_comparison_20260921',
    'melanoma_story': 'exploration_melanoma_story_lymphocytes_20260921',
    'hnscc_story': 'exploration_hnscc_cd4_cd8_b_20260921',
}
PILOT = 'exploration_lung_new140k_ramp_c02_20260921'
CASES = ('bone_marrow', *UPDATES)
POINTER_SHA = '3f794292e4c25f2cb324599641efe200278dd21060c78563b696b551a1f3eb39'


def bind(name):
    """Read historical final_xu paths under an explicit, relocatable source root."""
    text = str(name)
    require('/final_xu/' in text, 'Unrecognized historical root: ' + text)
    tail = text.split('/final_xu/', 1)[1]
    from safe_io import relative
    return relative(tail).as_posix()


def collect(source):
    source = Path(source).resolve(strict=True)
    rows, copies, links = {}, set(), []

    def add(name, copy=False):
        p = member(source, name)
        rec = record(p)
        require(name not in rows or rows[name] == rec, 'Source changed during collection')
        rows[name] = rec
        if copy:
            require(p.suffix in {'.py', '.R', '.json', '.csv', '.md', '.txt'}, 'Disallowed copy format')
            copies.add(name)
        return rec

    def link(owner, fields, target, copy=False):
        expected = read(member(source, owner))
        for field in fields:
            expected = expected[field]
        rec = add(target, copy)
        require(expected == rec['sha256'], 'Historical hash mismatch: ' + owner + ' -> ' + target)
        links.append({'owner': owner, 'fields': fields, 'target': target})

    def local(rel, copy=True):
        return add(ROOT + '/' + rel, copy)

    local('CURRENT_FIGURE3_RESULTS.json')
    require(rows[ROOT + '/CURRENT_FIGURE3_RESULTS.json']['sha256'] == POINTER_SHA, 'Current Figure 3 pointer changed')
    pointer = read(source / ROOT / 'CURRENT_FIGURE3_RESULTS.json')
    require(bind(pointer['directory']) == ROOT + '/' + FINAL, 'Wrong final release')
    link(ROOT + '/CURRENT_FIGURE3_RESULTS.json', ['primary_means_sha256'], ROOT + '/' + FINAL + '/primary_means.json', True)
    link(ROOT + '/CURRENT_FIGURE3_RESULTS.json', ['report_sha256'], ROOT + '/' + FINAL + '/REPORT.json', True)
    # Copy source, profiles/shards and top-level release tables. No deployment is executed.
    for dirname in ('src', 'config/formal_v0_5'):
        for p in sorted((source / ROOT / dirname).rglob('*')):
            if p.is_file() and p.suffix in {'.py', '.R', '.json'} and '__pycache__' not in p.parts:
                add(p.relative_to(source).as_posix(), True)
    for dirname in (FINAL, J1, *UPDATES.values(), PILOT, J2, J2 + '/summary'):
        for p in sorted((source / ROOT / dirname).iterdir()):
            if p.is_file() and p.suffix in {'.py', '.R', '.json', '.csv', '.md'}:
                add(p.relative_to(source).as_posix(), True)
    for filename in ('REPORT.json', 'primary_means.json', 'metrics.csv', 'coverage.csv'):
        local(BASE + '/' + filename)
    local('validation/formal_preflight_v0_5/EXCLUSIONS.json')
    local('validation/formal_preflight_v0_5/REPORT.json')
    suite_owner = ROOT + '/config/formal_v0_5/suite.json'
    suite = read(source / suite_owner)
    link(suite_owner, ['profile_sha256'], bind(suite['profile']), True)
    link(suite_owner, ['release_report_sha256'], bind(suite['release_report']), True)
    for n, shard in enumerate(suite['shards']):
        link(suite_owner, ['shards', n, 'sha256'], bind(shard['config']), True)
    for release in (FINAL, J1):
        owner = ROOT + '/' + release + '/REPORT.json'
        report = read(source / owner)
        # Releases use source_files (final) and source_summaries (J1); inspect exact maps only.
        for field in ('source_files', 'source_summaries'):
            for path, value in report.get(field, {}).items():
                if isinstance(value, str) and len(value) == 64:
                    link(owner, [field, path], bind(path), True)
        if release == J1:
            link(owner, ['original_summary_sha256'], ROOT + '/' + BASE + '/primary_means.json', True)
            link(owner, ['original_report_sha256'], ROOT + '/' + BASE + '/REPORT.json', True)
            for case, item in report['sources'].items():
                for path in item['sources']:
                    link(owner, ['sources', case, 'sources', path], bind(path), True)
    # Registered frozen input files are hashed, not distributed/copied in this package.
    for case in CASES:
        owner = ROOT + '/data/real_inputs_v0_3_r2/' + case + '/manifest.json'
        add(owner, True)
        for filename in read(source / owner)['outputs']:
            link(owner, ['outputs', filename], str(Path(owner).parent / filename))
    # Freeze split plans by hash only; manifests themselves include grids/candidates.
    for dirname in (*UPDATES.values(), PILOT, J2):
        owner = ROOT + '/' + dirname + '/MANIFEST.json'
        manifest = read(source / owner)
        for path in manifest.get('experiment_hashes', {}):
            target = bind(path) if Path(path).is_absolute() else str(Path(owner).parent / path)
            link(owner, ['experiment_hashes', path], target, Path(target).suffix in {'.py', '.R'})
    # Current Lung source-scale matrix + metadata (not the older formal Lung cohort).
    owner = ROOT + '/' + J2 + '/INPUT_CONTRACT.json'
    for path in read(source / owner)['source_files']:
        # Only the actually consumed release matrix, metadata and design in this phase.
        if '/releases/senosain140k_20260921/inputs/' in path:
            link(owner, ['source_files', path], bind(path), path.endswith('.json'))
    # Freeze J2 representation receipts (not full embeddings or their native provider trees).
    # Full matrix/prediction hash validation belongs to execution preparation, not this inventory.
    rep_dirs = set()
    for shard in suite['shards']:
        conf = read(source / bind(shard['config']))
        for job in conf['jobs']:
            if 'primary' in job['analysis_scopes'] and job['task_type'] == 'J2' and job['task'] != 'lung_story':
                rep_dirs.add(ROOT + '/representations/formal_v0_5/' + job['task'] + '/' + job['marker'] + '/' + job['method'])
    new = read(source / ROOT / J2 / 'MANIFEST.json')
    rep_dirs.update(ROOT + '/' + J2 + '/representations/' + j['marker'] + '/' + j['method'] for j in new['jobs'])
    for dirname in sorted(rep_dirs):
        for name in ('START.json', 'COMPLETE.json'):
            add(dirname + '/' + name, True)
        if dirname.startswith(ROOT + '/' + J2 + '/'):
            add(dirname + '/MASK_BOUNDARY_AUDIT.json', True)
    return {'schema': 1, 'scope': 'source contracts and existing score-table aggregation; no fitting',
            'files': dict(sorted(rows.items())), 'copies': sorted(copies), 'links': links,
            'j2_representation_receipts': sorted(rep_dirs),
            'not_verified': ['representation matrices/predictions/models', 'all per-job ledgers',
                             'provider extraction closure', 'clean runtime or retraining']}


def verify_sources(catalog, source):
    for name, expected in catalog['files'].items():
        require(record(member(source, name)) == expected, 'Source changed: ' + name)
    for rel in catalog['links']:
        value = read(member(source, rel['owner']))
        for field in rel['fields']:
            value = value[field]
        require(value == catalog['files'][rel['target']]['sha256'], 'Broken historical hash relation')
    return {'source_files': len(catalog['files']), 'hash_relations': len(catalog['links'])}


def payload(snapshot, catalog):
    require(len(catalog['copies']) == len(set(catalog['copies'])), 'Duplicate copy paths')
    expected = {'tree/' + name for name in catalog['copies']}
    tree = member(snapshot, 'tree')
    require({'tree/' + n for n in files(tree)} == expected, 'Unexpected/missing snapshot payload')
    for name in catalog['copies']:
        require(record(member(tree, name)) == catalog['files'][name], 'Snapshot file changed: ' + name)


def verify(snapshot, anchor, source=None):
    snapshot = Path(snapshot).resolve(strict=True)
    require(record(snapshot / 'catalog.json')['sha256'] == anchor, 'Wrong catalog trust anchor')
    catalog = read(snapshot / 'catalog.json')
    complete = read(snapshot / 'COMPLETE.json')
    require(complete == {'status': 'complete', 'catalog_sha256': anchor}, 'Snapshot incomplete')
    payload(snapshot, catalog)
    require(files(snapshot) == {'catalog.json', 'COMPLETE.json'} | {'tree/' + n for n in catalog['copies']},
            'Unexpected snapshot files')
    result = {'copied_files': len(catalog['copies']), 'catalog_sha256': anchor}
    if source is not None:
        result.update(verify_sources(catalog, source))
    return result


def freeze(source, output):
    source = Path(source).resolve(strict=True)
    out = reserve(output, [source])
    try:
        catalog = collect(source)
        (out / 'tree').mkdir()
        for name in catalog['copies']:
            copy_checked(member(source, name), out / 'tree' / name, catalog['files'][name])
        verify_sources(catalog, source)
        write(out / 'catalog.json', catalog)
        payload(out, catalog)
        require(files(out) == {'catalog.json'} | {'tree/' + n for n in catalog['copies']}, 'Unexpected partial files')
        anchor = record(out / 'catalog.json')['sha256']
        write(out / 'COMPLETE.json', {'status': 'complete', 'catalog_sha256': anchor})
        return {'catalog_sha256': anchor, 'copies': len(catalog['copies']), 'sources': len(catalog['files'])}
    except BaseException as error:
        write(out / 'FAILED.json', {'error': repr(error)})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    f = sub.add_parser('freeze'); f.add_argument('--source-root', type=Path, required=True); f.add_argument('--output', type=Path, required=True)
    v = sub.add_parser('verify'); v.add_argument('--snapshot', type=Path, required=True); v.add_argument('--anchor', required=True); v.add_argument('--source-root', type=Path)
    a = p.parse_args()
    print(freeze(a.source_root, a.output) if a.command == 'freeze' else verify(a.snapshot, a.anchor, a.source_root))


if __name__ == '__main__':
    main()
