from pathlib import Path
import os, sys, json, hashlib, datetime
BASE = Path(__file__).resolve().parents[1]
STORY = BASE.parent
ROOT = STORY.parent
sys.dont_write_bytecode = True

def allowed(p):
    if isinstance(p, int) or p is None: return
    p = Path(os.fsdecode(p)).resolve()
    if p == Path('/dev/null'): return
    if not p.is_relative_to(BASE): raise PermissionError(f'Write outside summary_steps3_6: {p}')

def audit(event, args):
    if event == 'open':
        p, mode, flags = args
        if (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)): allowed(p)
    elif event in {'os.remove','os.rmdir','os.mkdir','os.chmod','os.utime','os.truncate'}: allowed(args[0])
    elif event in {'os.rename','os.replace'}: allowed(args[0]); allowed(args[1])
    elif event in {'os.link','os.symlink'}: allowed(args[1])

sys.addaudithook(audit)
for key, path in {'TMPDIR':BASE/'tmp','MPLCONFIGDIR':BASE/'cache/matplotlib','XDG_CACHE_HOME':BASE/'cache/xdg'}.items():
    path.mkdir(parents=True, exist_ok=True); os.environ[key] = str(path)
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']: os.environ[key] = '1'
