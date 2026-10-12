"""Portable runtime; all writes stay inside this umap folder."""
from pathlib import Path
import os
import sys
import json
import hashlib
import datetime

BASE = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
for key, value in {
    'TMPDIR': BASE/'tmp', 'TMP': BASE/'tmp', 'TEMP': BASE/'tmp',
    'MPLCONFIGDIR': BASE/'cache/matplotlib', 'NUMBA_CACHE_DIR': BASE/'cache/numba',
    'XDG_CACHE_HOME': BASE/'cache/xdg', 'JOBLIB_TEMP_FOLDER': BASE/'tmp',
}.items():
    value.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(value)
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','LOKY_MAX_CPU_COUNT','NUMBA_NUM_THREADS']:
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'


def allowed(path):
    if isinstance(path, int) or path is None:
        return
    p = Path(os.fsdecode(path)).resolve()
    if p == Path('/dev/null'):
        return
    if p != BASE and BASE not in p.parents:
        raise PermissionError(f'Write outside umap denied: {p}')


def audit(event, args):
    if event == 'open':
        name, mode, flags = args
        if (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (
            isinstance(flags, int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)
        ):
            allowed(name)
    elif event in {'os.remove','os.rmdir','os.mkdir','os.chmod','os.utime','os.truncate'}:
        allowed(args[0])
    elif event in {'os.rename','os.replace'}:
        allowed(args[0]); allowed(args[1])
    elif event in {'os.link','os.symlink'}:
        allowed(args[1])


sys.addaudithook(audit)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
