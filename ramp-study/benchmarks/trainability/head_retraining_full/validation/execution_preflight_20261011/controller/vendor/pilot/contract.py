"""Portable, create-only pilot contracts. No imports perform data reads or fits."""
import hashlib
import json
import os
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
STUDY = HERE.parents[2]
SOURCE = STUDY.parent/'final_xu'
INDEX_SHA = '82eafb7e4e41652b23844b53247081e3242163cc606004771734d2ba51ea7c4d'
ACCEPTED_SHA = 'ac8352daa4dbc6129894d64b759938dbb7925c83f0ced3f88de657348620991b'
PLAN_SHA = '7508952ccb43b62882cd94fc23a722d01e222e931d5c0bed597e54d7338a86b2'
CODE = ('contract.py', 'study.py', 'worker.py')
THREADS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS','LOKY_MAX_CPU_COUNT')
VERSIONS = dict(python='3.14.1', numpy='2.4.0', pandas='2.3.3', scipy='1.16.3', sklearn='1.8.0')

def require(ok, message):
    if not ok: raise ValueError(message)

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()).hexdigest()

def record(path):
    p = Path(path); require(p.is_file() and not p.is_symlink(), 'Not a regular file: '+str(p))
    with p.open('rb') as stream: sha = hashlib.file_digest(stream,'sha256').hexdigest()
    return dict(bytes=p.stat().st_size, sha256=sha)

def read(path): return json.loads(Path(path).read_text())

def write(path, value):
    with Path(path).open('x') as f: json.dump(value,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n')

def member(root, name):
    root = Path(root).resolve(strict=True); rel = Path(name)
    require(not rel.is_absolute() and rel.parts and '..' not in rel.parts and str(rel) != '.', 'Unsafe member')
    path = root/rel
    require(not any(p.is_symlink() for p in (path,*path.parents) if p.is_relative_to(root) and p != root), 'Symlink member')
    require(path.resolve().is_relative_to(root), 'Escaping member')
    return path

def files(root):
    result = {}
    for p in sorted(Path(root).rglob('*')):
        require(not p.is_symlink(), 'Symlink in artifact')
        if p.is_file(): result[p.relative_to(root).as_posix()] = record(p)
    return result

def reserve(path, protected=()):
    path = Path(os.path.abspath(path)); require(not path.exists() and not path.is_symlink(), 'Output already exists')
    dest = path.resolve()
    if dest.is_relative_to(STUDY):
        require(any(dest.is_relative_to(HERE/n) and dest != HERE/n for n in ('runs','validation','bundles')), 'Output outside new evidence roots')
    for base in (SOURCE, STUDY.parent/'ramp-proteomics', HERE.parent/'prediction_replay', HERE.parent/'frozen_20261011_r2', *protected):
        base = Path(base).resolve()
        require(not(dest.is_relative_to(base) or base.is_relative_to(dest)), 'Output overlaps protected input')
    dest.parent.mkdir(parents=True,exist_ok=True); dest.mkdir(exist_ok=False)
    return dest

def copy(source, dest, expected):
    require(record(source) == expected, 'Source changed: '+str(source))
    dest.parent.mkdir(parents=True,exist_ok=True)
    with Path(source).open('rb') as src, dest.open('xb') as dst: shutil.copyfileobj(src,dst,1024*1024)
    require(record(dest) == expected, 'Copied file differs')

def codes(): return {n:record(HERE/n) for n in CODE}

def load_plan():
    p = HERE/'locks/pilot_20261011.json'
    require(record(p)['sha256'] == PLAN_SHA, 'Pilot plan changed')
    return read(p)

def seal(root, details):
    require(not(root/'FAILED.json').exists(), 'Failed output cannot complete')
    manifest = dict(details=details, files=files(root))
    write(root/'RUN_MANIFEST.json',manifest)
    require(files(root) == manifest['files']|{'RUN_MANIFEST.json':record(root/'RUN_MANIFEST.json')}, 'Output changed before completion')
    write(root/'COMPLETE.json',dict(status='complete',manifest=record(root/'RUN_MANIFEST.json')))

def verify_seal(root):
    root = Path(root).resolve(); complete = read(root/'COMPLETE.json')
    require(not(root/'FAILED.json').exists() and complete.get('status') == 'complete', 'Incomplete artifact')
    require(complete.get('manifest') == record(root/'RUN_MANIFEST.json'), 'Manifest changed')
    m = read(root/'RUN_MANIFEST.json')
    require(files(root) == m['files']|{n:record(root/n) for n in ('RUN_MANIFEST.json','COMPLETE.json')}, 'Artifact files differ')
    return m

def environment():
    import platform, numpy, pandas, scipy, sklearn
    from threadpoolctl import threadpool_info
    versions = dict(python=platform.python_version(),numpy=numpy.__version__,pandas=pandas.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__)
    require(versions == VERSIONS, 'Runtime versions differ')
    require(all(os.environ.get(n) == '1' for n in THREADS), 'Thread environment differs')
    pools = threadpool_info(); require(bool(pools) and all(p['num_threads'] == 1 for p in pools), 'Thread pool differs')
    return dict(versions=versions, threads={n:os.environ[n] for n in THREADS},pools=pools)

def child_environment():
    env = dict(os.environ)
    for n in THREADS: env[n] = '1'
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0', LC_ALL='C.UTF-8', LANG='C.UTF-8')
    env.pop('PYTHONPATH',None)
    return env

def verify_bundle(bundle):
    bundle = Path(bundle).resolve(); plan = load_plan()
    require(read(bundle/'MANIFEST.json') == dict(plan_sha256=PLAN_SHA,files=plan['training_files']), 'Bundle contract differs')
    require(read(bundle/'COMPLETE.json') == dict(status='complete',manifest=record(bundle/'MANIFEST.json')), 'Incomplete bundle')
    expected = {n:v['record'] for n,v in plan['training_files'].items()}
    expected.update({n:record(bundle/n) for n in ('MANIFEST.json','COMPLETE.json')})
    require(files(bundle) == expected, 'Bundle payload differs')
    return plan

def check_grid(actual, expected):
    require(isinstance(actual,list) and len(actual)==len(set(actual)) and set(actual)==set(expected), 'Pilot task grid differs')

def checked_json(bundle, name, plan):
    p = member(bundle,name); require(record(p)==plan['training_files'][name]['record'], 'Input changed')
    return read(p)
