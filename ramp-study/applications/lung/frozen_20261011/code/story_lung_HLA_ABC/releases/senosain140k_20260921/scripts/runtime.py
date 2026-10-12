from pathlib import Path
import os, sys, json, hashlib, datetime
BASE=Path(__file__).resolve().parents[1]
STORY=BASE.parents[1]
ROOT=STORY.parent
sys.dont_write_bytecode=True
for k,p in {'TMPDIR':BASE/'tmp','TMP':BASE/'tmp','TEMP':BASE/'tmp','MPLCONFIGDIR':BASE/'cache/mpl','NUMBA_CACHE_DIR':BASE/'cache/numba','XDG_CACHE_HOME':BASE/'cache/xdg','JOBLIB_TEMP_FOLDER':BASE/'tmp'}.items():
    p.mkdir(parents=True,exist_ok=True);os.environ[k]=str(p)
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS','LOKY_MAX_CPU_COUNT']:os.environ[k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
def allowed(p):
    if isinstance(p,int) or p is None:return
    p=Path(os.fsdecode(p)).resolve()
    if p==Path('/dev/null'):return
    if not p.is_relative_to(BASE):raise PermissionError(f'Write outside test denied: {p}')
def audit(event,args):
    if event=='open':
        p,m,f=args
        if (isinstance(m,str) and any(c in m for c in 'wax+')) or (isinstance(f,int) and f&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)):allowed(p)
    elif event in {'os.remove','os.rmdir','os.mkdir','os.chmod','os.utime','os.truncate'}:allowed(args[0])
    elif event in {'os.rename','os.replace'}:allowed(args[0]);allowed(args[1])
    elif event in {'os.link','os.symlink'}:allowed(args[1])
sys.addaudithook(audit)
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
def status(stage,**kwargs):save(BASE/'STATUS.json',dict(stage=stage,time_utc=now(),**kwargs))
def hashes(paths):return {str(p):sha(p) for p in paths}
def unchanged(h):
    for p,s in h.items():assert sha(p)==s,p
sys.path.insert(0,str(BASE/'vendor'))
MARKERS=['CD3','CD4','CD8','CD31','KI67','VIMENTIN','EPI','CD45','HLA_DR']
CONFIGS=['RAMP_k30','RAMP_k100','baseline_k30','baseline_k100']
