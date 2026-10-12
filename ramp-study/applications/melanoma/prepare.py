"""Pin source files in place. Only nine small source scripts are copied."""
import argparse
from pathlib import Path
from common import HERE, STUDY, MARKERS, read, write, sha, record, reserve, files
from specs import CODES, INPUTS, STAGES, ALIGN, STORY, references

LOCK_SHA = 'cd1a12aaf396ed618f5dec01ea2dc36574542e235bfcb8e212a55fd0261c0bd1'
ACCEPTED_REL = 'benchmarks/ramp_alignment/runs/six_fit_20261011_r2/melanoma_story'
ALIGN_SHA = '06371c47943e0700a19f79538cd2b289b8d9d67119a3203344a5b34d8906cd9b'
R_EXEC = '/Library/Frameworks/R.framework/Resources/bin/Rscript'

def build(source, output):
    source = Path(source).resolve()
    accepted = STUDY / ACCEPTED_REL
    alignment = accepted / 'result/ramp_aligned.npy'
    if sha(alignment) != ALIGN_SHA or sha(source / ALIGN / 'ramp_aligned.npy') != ALIGN_SHA:
        raise ValueError('Accepted alignment binding differs')
    recs = {rel: {'role': 'code' if rel in CODES else 'input', **record(source / rel)} for rel in CODES + INPUTS}
    for rel in references(source):
        recs[rel] = {'role': 'reference', **record(source / rel)}
    for rel in (STORY + '/README.md', STORY + '/figure_layout_v8/build_figure.py'):
        recs[rel] = {'role': 'provenance', **record(source / rel)}
    input_paths = {rel: str(source / rel) for rel in INPUTS}
    input_paths[ALIGN + '/ramp_aligned.npy'] = str(alignment)
    lock = {'schema': 1, 'source_root': str(source), 'files': recs, 'input_paths': input_paths,
            'accepted_alignment': str(alignment), 'stages': STAGES, 'markers': MARKERS,
            'accepted_manifest': {'path': str(accepted / 'RUN_MANIFEST.json'), **record(accepted / 'RUN_MANIFEST.json')},
            'accepted_complete': {'path': str(accepted / 'COMPLETE.json'), **record(accepted / 'COMPLETE.json')},
            'r_executable': {'path': R_EXEC, **record(Path(R_EXEC).resolve())}}
    out = reserve(output, [source, accepted, HERE / 'runs'])
    try:
        for rel in CODES:
            dest = out / 'code' / rel; dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open('xb') as f:
                f.write((source / rel).read_bytes())
        write(out / 'LOCK.json', lock)
        check_sources(lock)
        write(out / 'COMPLETE.json', {'status': 'complete', 'lock_sha256': sha(out / 'LOCK.json')})
    except BaseException as e:
        write(out / 'FAILED.json', {'error': repr(e)}); raise
    return {'lock_sha256': sha(out / 'LOCK.json'), 'sources': len(recs), 'copies': len(CODES)}

def check_sources(lock):
    for rel, rec in lock['files'].items():
        if record(Path(lock['source_root']) / rel) != {k: rec[k] for k in ('bytes', 'sha256')}:
            raise ValueError('Source changed: ' + rel)
    if sha(lock['accepted_alignment']) != ALIGN_SHA:
        raise ValueError('Accepted alignment changed')
    for key in ('accepted_manifest', 'accepted_complete', 'r_executable'):
        rec = lock[key]
        if record(Path(rec['path']).resolve()) != {k: rec[k] for k in ('bytes', 'sha256')}:
            raise ValueError('External provenance changed: ' + key)

def verify(lockdir, source=False, expected_sha=None):
    folder = Path(lockdir).resolve(); digest = sha(folder / 'LOCK.json')
    if digest != LOCK_SHA or (expected_sha is not None and digest != expected_sha):
        raise ValueError('Unrecognized or changed melanoma lock')
    if (folder / 'FAILED.json').exists() or read(folder / 'COMPLETE.json') != {'status': 'complete', 'lock_sha256': digest}:
        raise ValueError('Lock incomplete')
    lock = read(folder / 'LOCK.json')
    if lock['stages'] != STAGES or lock['markers'] != MARKERS or set(lock['input_paths']) != set(INPUTS):
        raise ValueError('Grid differs')
    if {k for k, v in lock['files'].items() if v['role'] == 'code'} != set(CODES):
        raise ValueError('Code grid differs')
    if {k for k, v in lock['files'].items() if v['role'] == 'input'} != set(INPUTS):
        raise ValueError('Input grid differs')
    expected = {rel: {k: lock['files'][rel][k] for k in ('bytes', 'sha256')} for rel in CODES}
    if files(folder / 'code') != expected or any(p.is_symlink() for p in folder.rglob('*')):
        raise ValueError('Frozen source differs')
    if source:
        check_sources(lock)
    return lock

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['freeze', 'verify'])
    p.add_argument('--source'); p.add_argument('--out'); p.add_argument('--lockdir'); p.add_argument('--check-sources', action='store_true')
    a = p.parse_args()
    print(build(a.source, a.out) if a.mode == 'freeze' else {'files': len(verify(a.lockdir, a.check_sources)['files'])})
