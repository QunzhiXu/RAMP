"""Bind in-place inputs and references; copy only small, original code files."""
import argparse
from pathlib import Path
from common import HERE, STORY, EXP, MARKERS, record, read, write, reserve, sha, files
from specs import CODES, INPUTS, REF_DIRS, STAGES

LOCK_SHA = '15f8d366fae994cc8e1bad0d7cdfb9f2c4b130eed6195de3959ff5012e12601f'


def build(source, output):
    source = Path(source).resolve()
    records = {rel: {'role': 'code' if rel in CODES else 'input', **record(source / rel)}
               for rel in CODES + INPUTS}
    for directory in REF_DIRS:
        for path in sorted((source / directory).rglob('*')):
            if path.is_file() and path.name != '.DS_Store' and path.suffix in ('.json', '.csv', '.gz', '.npy', '.npz', '.pt'):
                rel = str(path.relative_to(source))
                if rel not in records:
                    records[rel] = {'role': 'reference', **record(path)}
    # Bind the final release pointer and the frozen cohort provenance, not old KNN explorations.
    for rel in (f'{STORY}/sampled_data/manifest.json', f'{STORY}/sampled_data/relabel/manifest.json',
                f'{EXP}/figure_layout_v31_DAPI_pair/build_figure.py',
                f'{EXP}/figure_layout_v31_DAPI_pair/cd19_story_three_row_v31_DAPI_pair.pdf'):
        records[rel] = {'role': 'provenance', **record(source / rel)}
    output = reserve(output, [source, HERE / 'runs'])
    try:
        for rel in CODES:
            dest = output / 'code' / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open('xb') as target, (source / rel).open('rb') as origin:
                target.write(origin.read())
        lock = {'schema': 1, 'source_root': str(source), 'stages': STAGES, 'markers': MARKERS,
                'files': records, 'input_copies': 0, 'historical_result_copies': 0}
        write(output / 'LOCK.json', lock)
        verify(output, source=True, require_complete=False)
        write(output / 'COMPLETE.json', {'status': 'complete', 'lock_sha256': sha(output / 'LOCK.json')})
    except BaseException as exc:
        write(output / 'FAILED.json', {'error': repr(exc)})
        raise
    return {'files': len(records), 'copied_code_files': len(CODES), 'lock_sha256': sha(output / 'LOCK.json')}


def verify(lockdir, source=False, require_complete=True, expected_sha=None):
    lockdir = Path(lockdir).resolve()
    digest = sha(lockdir / 'LOCK.json')
    if digest != LOCK_SHA:
        raise ValueError('Unrecognized CD19 source lock')
    if expected_sha is not None and digest != expected_sha:
        raise ValueError('Lock changed')
    if (lockdir / 'FAILED.json').exists():
        raise ValueError('Failed lock')
    if require_complete and read(lockdir / 'COMPLETE.json') != {'status': 'complete', 'lock_sha256': digest}:
        raise ValueError('Lock incomplete')
    lock = read(lockdir / 'LOCK.json')
    if lock['stages'] != STAGES or lock['markers'] != MARKERS:
        raise ValueError('Scientific grid differs')
    if {p for p, r in lock['files'].items() if r['role'] == 'code'} != set(CODES):
        raise ValueError('Code grid differs')
    if {p for p, r in lock['files'].items() if r['role'] == 'input'} != set(INPUTS):
        raise ValueError('Input grid differs')
    expected = {rel: {k: lock['files'][rel][k] for k in ('bytes', 'sha256')} for rel in CODES}
    if files(lockdir / 'code') != expected or any(p.is_symlink() for p in lockdir.rglob('*')):
        raise ValueError('Frozen code differs')
    for rel, rec in lock['files'].items():
        if Path(rel).is_absolute() or '..' in Path(rel).parts:
            raise ValueError('Unsafe source path')
        if source and record(Path(lock['source_root']) / rel) != {k: rec[k] for k in ('bytes', 'sha256')}:
            raise ValueError(f'Source changed: {rel}')
    return lock


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='mode', required=True)
    b = sub.add_parser('freeze'); b.add_argument('--source', required=True); b.add_argument('--out', required=True)
    v = sub.add_parser('verify'); v.add_argument('--lockdir', required=True); v.add_argument('--source', action='store_true')
    args = parser.parse_args()
    print(build(args.source, args.out) if args.mode == 'freeze' else {'files': len(verify(args.lockdir, args.source)['files'])})

