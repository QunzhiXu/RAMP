from pathlib import Path
import os, sys, json, hashlib, datetime
BASE = Path(__file__).resolve().parents[1]
STORY = BASE.parent
ROOT = STORY.parent
sys.dont_write_bytecode = True

def allowed(path):
    if isinstance(path, int) or path is None: return
    p = Path(os.fsdecode(path)).resolve()
    if p == Path('/dev/null'): return
    if not p.is_relative_to(BASE): raise PermissionError(f'Write outside Exploration0909: {p}')

def audit(event, args):
    if event == 'open':
        p, mode, flags = args
        if (isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)): allowed(p)
    elif event in {'os.remove','os.rmdir','os.mkdir','os.chmod','os.utime','os.truncate'}: allowed(args[0])
    elif event in {'os.rename','os.replace','os.link','os.symlink'}: allowed(args[0]); allowed(args[1])
sys.addaudithook(audit)
for key,path in {'TMPDIR':BASE/'tmp','MPLCONFIGDIR':BASE/'cache/matplotlib','NUMBA_CACHE_DIR':BASE/'cache/numba','XDG_CACHE_HOME':BASE/'cache/xdg','JOBLIB_TEMP_FOLDER':BASE/'cache/joblib'}.items():
    path.mkdir(parents=True,exist_ok=True); os.environ[key]=str(path)
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','LOKY_MAX_CPU_COUNT']: os.environ[key]='1'
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,obj): Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
