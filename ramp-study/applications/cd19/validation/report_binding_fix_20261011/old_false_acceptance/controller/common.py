"""Create-only records and bounded, read-only-source CD19 execution."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys

HERE = Path(__file__).resolve().parent
STUDY = HERE.parents[1]
STORY = 'story_cd19_prediction'
EXP = STORY + '/Exploration0909'
MARKERS = 'CD117 CD11B CD11C CD123 CD14 CD3 CD33 CD34 CD38 CD4 CD45 CD45RA CD61 CD8'.split()
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'LOKY_MAX_CPU_COUNT', 'NUMBA_NUM_THREADS')
VERSIONS = {'numpy': '2.4.0', 'pandas': '2.3.3', 'scipy': '1.16.3',
            'scikit-learn': '1.8.0', 'torch': '2.10.0', 'umap-learn': '0.5.11'}
CONTROLLERS = ('common.py', 'specs.py', 'prepare.py', 'worker.py', 'study.py')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def regular(path):
    path = Path(path)
    if not stat.S_ISREG(path.lstat().st_mode) or path.stat().st_nlink != 1:
        raise ValueError(f'Not a single-link regular file: {path}')
    return path


def record(path):
    path = regular(path)
    return {'bytes': path.stat().st_size, 'sha256': sha(path)}


def codes():
    return {name: record(HERE / name) for name in CONTROLLERS}


def stage_root(run):
    # The unchanged Exploration runtime hook permits writes only below EXP.
    return Path(run) / 'tree' / EXP / '.execution' / 'stages'


def no_alias(path):
    path = Path(os.path.abspath(path))
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError(f'Symlink in output path: {part}')
    return path


def reserve(path, protected):
    path = no_alias(path)
    for other in protected:
        other = Path(other).resolve()
        if path == other or path.is_relative_to(other) or other.is_relative_to(path):
            raise ValueError(f'Output overlaps protected path: {other}')
    path.mkdir(exist_ok=False)  # Parent must already exist; no preliminary writes.
    return path


def resource_state():
    allocated = int(subprocess.check_output(['du', '-sk', str(STUDY)], text=True).split()[0]) * 1024
    free = shutil.disk_usage(STUDY).free
    return {'study_allocated_bytes': allocated, 'filesystem_free_bytes': free}


def check_resources(state=None):
    state = resource_state() if state is None else state
    if state['study_allocated_bytes'] >= 60_000_000_000 or state['filesystem_free_bytes'] < 20_000_000_000:
        raise RuntimeError(f'Resource limit: {state}')
    return state


def environment(run):
    env = os.environ.copy()
    for key in ('PYTHONPATH', 'PYTHONHOME', 'R_LIBS', 'R_LIBS_USER', 'R_LIBS_SITE'):
        env.pop(key, None)
    env.update({key: '1' for key in THREADS})
    env.update(PYTHONDONTWRITEBYTECODE='1', LC_ALL='C.UTF-8', LANG='C.UTF-8')
    base = Path(run) / 'tree' / EXP
    for key, rel in {'TMPDIR': 'tmp', 'MPLCONFIGDIR': 'cache/matplotlib',
                     'NUMBA_CACHE_DIR': 'cache/numba', 'XDG_CACHE_HOME': 'cache/xdg',
                     'JOBLIB_TEMP_FOLDER': 'cache/joblib'}.items():
        env[key] = str(base / rel)
    return env


def files(root):
    return {str(p.relative_to(root)): record(p) for p in sorted(Path(root).rglob('*'))
            if p.is_file() and not p.is_symlink()}


def links(root):
    return {str(p.relative_to(root)): str(p.resolve()) for p in sorted(Path(root).rglob('*')) if p.is_symlink()}


def write_guard(root, immutable):
    root = Path(root).resolve()
    immutable = {Path(p).resolve() for p in immutable}

    def allowed(path):
        if path is None or isinstance(path, int):
            return
        path = Path(os.fsdecode(path)).resolve()
        if path == Path('/dev/null'):
            return
        if not path.is_relative_to(root) or path in immutable:
            raise PermissionError(f'Protected write: {path}')

    def audit(event, args):
        if event == 'open':
            path, mode, flags = args
            if (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (
                    isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)):
                allowed(path)
        elif event in {'os.remove', 'os.rmdir', 'os.mkdir', 'os.chmod', 'os.utime', 'os.truncate'}:
            allowed(args[0])
        elif event in {'os.rename', 'os.replace', 'os.link', 'os.symlink'}:
            allowed(args[0]); allowed(args[1])
    sys.addaudithook(audit)

