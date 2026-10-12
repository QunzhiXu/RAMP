"""Confine plotting outputs and caches to the Combined ABCDEF package."""
from pathlib import Path
import os
import sys

BASE = Path(__file__).resolve().parents[1]
STORY = BASE.parents[1]
sys.dont_write_bytecode = True


def allowed(path):
    if isinstance(path, int) or path is None:
        return
    path = Path(os.fsdecode(path)).resolve()
    if path == Path('/dev/null'):
        return
    if not path.is_relative_to(BASE):
        raise PermissionError(f'Write outside Combined ABCDEF: {path}')


def audit(event, args):
    if event == 'open':
        path, mode, flags = args
        if (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (
            isinstance(flags, int) and flags & (
                os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
            )
        ):
            allowed(path)
    elif event in {'os.remove', 'os.rmdir', 'os.mkdir', 'os.chmod', 'os.utime', 'os.truncate'}:
        allowed(args[0])
    elif event in {'os.rename', 'os.replace'}:
        allowed(args[0])
        allowed(args[1])
    elif event in {'os.link', 'os.symlink'}:
        allowed(args[1])


sys.addaudithook(audit)
for key, relative in {
    'TMPDIR': 'tmp', 'MPLCONFIGDIR': 'cache/matplotlib', 'XDG_CACHE_HOME': 'cache/xdg'
}.items():
    path = BASE / relative
    path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(path)
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[key] = '1'
