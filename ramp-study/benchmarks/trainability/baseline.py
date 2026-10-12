"""Existing accepted artifacts are read-only. Explicitly exclude run payloads from the NEW census."""
import argparse
import os
from pathlib import Path
from safe_io import require, record, read, write, reserve

HERE = Path(__file__).resolve().parent
STUDY = HERE.parents[1]
PRIOR = STUDY / 'benchmarks/umap_reproduction/validation/preflight_20261011/report.json'
PRIOR_SHA = '8360f5c47389ce2528d3e710dd8690fbf205fad344e3a2e305d63094c0f9b2f2'


def prior():
    require(record(PRIOR)['sha256'] == PRIOR_SHA, 'Overnight baseline changed')
    d = read(PRIOR)
    for name, rec in d['outside_files'].items():
        p = STUDY / name
        if 'symlink' in rec:
            require(p.is_symlink() and str(p.resolve()) == rec['symlink'], 'Prior symlink changed')
        else:
            require(record(p) == rec, 'Prior file changed: ' + name)
    # P2 fix is newer than this prior integration snapshot: use current capture for it.
    counts = {}
    for name in ('source_census.json', 'curated_sources.json'):
        n = 0
        for row in read(STUDY / 'inventory_20261010/evidence' / name)['files']:
            if row.get('sha256'):
                require(record(STUDY.parent / row['path']) == {k: row[k] for k in ('bytes', 'sha256')}, 'Original source changed')
                n += 1
        counts[name] = n
    return {'legacy_outside_files': len(d['outside_files']), 'original_indexes': counts}


def census():
    result = {}
    for root, dirs, names in os.walk(STUDY):
        dirs[:] = sorted(d for d in dirs if not (d.startswith('.') or d in
            {'runs', '__pycache__', 'node_modules', 'trainability'}))
        for name in sorted(names):
            if name == '.DS_Store' or name.endswith('.pyc'):
                continue
            p = Path(root) / name
            result[p.relative_to(STUDY).as_posix()] = ({'symlink': os.readlink(p)} if p.is_symlink() else record(p))
    return result


def verify(snapshot):
    d = read(snapshot)
    require(prior() == d['prior'], 'Prior baseline changed')
    require(census() == d['study_non_run_files'], 'Previously accepted study tree changed')
    return {'prior': d['prior'], 'current_non_run_records': len(d['study_non_run_files'])}


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('mode', choices=['capture', 'verify']); p.add_argument('--path', type=Path, required=True)
    a = p.parse_args()
    if a.mode == 'verify':
        print(verify(a.path))
    else:
        out = reserve(a.path, [STUDY.parent / 'final_xu', STUDY.parent / 'ramp-proteomics'])
        try:
            data = {'prior': prior(), 'study_non_run_files': census(),
                    'scope': 'Legacy indexed files + current non-run, non-environment study tree; NOT all run payloads'}
            write(out / 'report.json', data)
            print({'files': len(data['study_non_run_files']), 'sha256': record(out / 'report.json')['sha256']})
        except BaseException as error:
            write(out / 'FAILED.json', {'error': repr(error)}); raise


if __name__ == '__main__':
    main()
