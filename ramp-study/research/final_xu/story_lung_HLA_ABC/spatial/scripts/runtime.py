"""Keep every output and cache inside this spatial analysis folder."""
from pathlib import Path
import os, sys, json, hashlib, datetime
BASE = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
for k, v in {'TMPDIR':BASE/'tmp', 'TMP':BASE/'tmp', 'TEMP':BASE/'tmp',
             'MPLCONFIGDIR':BASE/'cache/matplotlib', 'XDG_CACHE_HOME':BASE/'cache/xdg'}.items():
    v.mkdir(parents=True, exist_ok=True)
    os.environ[k] = str(v)
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[k] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
def allowed(p):
    if isinstance(p, int) or p is None: return
    p = Path(os.fsdecode(p)).resolve()
    if p == Path('/dev/null'): return
    if p != BASE and BASE not in p.parents: raise PermissionError(f'Write outside spatial denied: {p}')
def audit(event, args):
    if event == 'open':
        name, mode, flags = args
        if (isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)): allowed(name)
    elif event in {'os.remove','os.rmdir','os.mkdir','os.chmod','os.utime','os.truncate'}: allowed(args[0])
    elif event in {'os.rename','os.replace'}: allowed(args[0]); allowed(args[1])
    elif event in {'os.link','os.symlink'}: allowed(args[1])
sys.addaudithook(audit)
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save_json(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
