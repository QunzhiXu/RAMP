"""Fixed source binding; copy only twelve small scientific scripts."""
import argparse
from pathlib import Path
from common import HERE, STUDY, MARKERS, read, write, sha, record, reserve, files, EXP
from specs import CODES, INPUTS, STAGES, ALIGN_FILES, PROVENANCE, references

LOCK_SHA = 'cc5f3e4a4fe13fce08a0b4ca29388a871174597cc596c8c86cc95d9e7496d1e9'
ACCEPTED_REL = 'benchmarks/ramp_alignment/runs/six_fit_20261011_r2/lung_story'
ALIGN_SHA = dict(zip(ALIGN_FILES, [
    'b91b1774b5588c6968ad7e9dd6f594a9be4620f01146d7797693aac9a329fd52',
    'c8a41dff073249f6ad4daf66f6ee4b9312b55c8dc9407b7aaf757675065840ee',
    'e10347bf5264135f4c93e32592e1deee853259854133166764ea2d8ccc0e0636',
    'a809573cea410f6d2c6e961a4b7ec1c5ab5c464ad6460f9d0c96bf4f07ead421']))
R_EXEC = '/Library/Frameworks/R.framework/Resources/bin/Rscript'

def build(source, output):
    source = Path(source).resolve(); accepted = STUDY / ACCEPTED_REL
    input_paths = {rel: str(source / rel) for rel in INPUTS}
    for name, digest in ALIGN_SHA.items():
        rel = EXP + '/alignment/' + name
        dest = accepted / 'result' / name
        if sha(dest) != digest or sha(source / rel) != digest:
            raise ValueError('Accepted alignment differs: ' + name)
        input_paths[rel] = str(dest)
    recs = {}
    for role, paths in [('code', CODES), ('input', INPUTS), ('reference', references(source)), ('provenance', PROVENANCE)]:
        for rel in paths:
            if rel in recs: raise ValueError('Overlapping source roles: ' + rel)
            recs[rel] = {'role': role, **record(source / rel)}
    lock = {'schema': 1, 'source_root': str(source), 'files': recs, 'input_paths': input_paths,
        'stages': STAGES, 'markers': MARKERS, 'accepted_alignment_sha256': ALIGN_SHA,
        'accepted_manifest': {'path': str(accepted / 'RUN_MANIFEST.json'), **record(accepted / 'RUN_MANIFEST.json')},
        'accepted_complete': {'path': str(accepted / 'COMPLETE.json'), **record(accepted / 'COMPLETE.json')},
        'r_executable': {'path': R_EXEC, **record(Path(R_EXEC).resolve())}}
    out = reserve(output, [source, accepted, HERE / 'runs'])
    try:
        for rel in CODES:
            dest = out / 'code' / rel; dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open('xb') as stream: stream.write((source / rel).read_bytes())
        write(out / 'LOCK.json', lock); check_sources(lock)
        write(out / 'COMPLETE.json', {'status': 'complete', 'lock_sha256': sha(out / 'LOCK.json')})
    except BaseException as exc:
        write(out / 'FAILED.json', {'error': repr(exc)}); raise
    return {'lock_sha256': sha(out / 'LOCK.json'), 'sources': len(recs), 'copies': len(CODES)}

def check_sources(lock):
    for rel, rec in lock['files'].items():
        if record(Path(lock['source_root']) / rel) != {k: rec[k] for k in ('bytes', 'sha256')}:
            raise ValueError('Source changed: ' + rel)
    for name, digest in ALIGN_SHA.items():
        if sha(lock['input_paths'][EXP + '/alignment/' + name]) != digest:
            raise ValueError('Accepted alignment changed')
    for key in ('accepted_manifest', 'accepted_complete', 'r_executable'):
        rec = lock[key]
        if record(Path(rec['path']).resolve()) != {k: rec[k] for k in ('bytes', 'sha256')}:
            raise ValueError('External provenance changed: ' + key)

def verify(lockdir, source=False, expected_sha=None):
    folder = Path(lockdir).resolve(); digest = sha(folder / 'LOCK.json')
    if digest != LOCK_SHA or (expected_sha is not None and digest != expected_sha):
        raise ValueError('Unrecognized or changed lung lock')
    if (folder / 'FAILED.json').exists() or read(folder / 'COMPLETE.json') != {'status': 'complete', 'lock_sha256': digest}:
        raise ValueError('Lock incomplete')
    lock = read(folder / 'LOCK.json')
    if lock['stages'] != STAGES or lock['markers'] != MARKERS or set(lock['input_paths']) != set(INPUTS):
        raise ValueError('Grid differs')
    for role, expected in [('input', INPUTS), ('code', CODES), ('provenance', PROVENANCE)]:
        if {k for k,v in lock['files'].items() if v['role'] == role} != set(expected):
            raise ValueError('Source role grid differs: ' + role)
    expected = {rel: {k: lock['files'][rel][k] for k in ('bytes', 'sha256')} for rel in CODES}
    if files(folder / 'code') != expected or any(p.is_symlink() for p in folder.rglob('*')):
        raise ValueError('Frozen source differs')
    if source: check_sources(lock)
    return lock

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('mode', choices=['freeze','verify'])
    p.add_argument('--source'); p.add_argument('--out'); p.add_argument('--lockdir'); p.add_argument('--check-sources',action='store_true')
    a=p.parse_args(); print(build(a.source,a.out) if a.mode=='freeze' else {'files':len(verify(a.lockdir,a.check_sources)['files'])})
