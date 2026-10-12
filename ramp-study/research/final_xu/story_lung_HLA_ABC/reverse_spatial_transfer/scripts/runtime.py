"""All new analysis and caches stay inside reverse_spatial_transfer."""
from pathlib import Path
import os,sys,json,hashlib,datetime
BASE=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
for key,p in {'TMPDIR':BASE/'tmp','TMP':BASE/'tmp','TEMP':BASE/'tmp','MPLCONFIGDIR':BASE/'cache/matplotlib','XDG_CACHE_HOME':BASE/'cache/xdg'}.items():
    p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
def allowed(p):
    if isinstance(p,int) or p is None:return
    p=Path(os.fsdecode(p)).resolve()
    if p==Path('/dev/null'):return
    if p!=BASE and BASE not in p.parents:raise PermissionError(f'Write outside reverse_spatial_transfer: {p}')
def audit(e,a):
    if e=='open':
        p,m,f=a
        if (isinstance(m,str) and any(c in m for c in 'wax+')) or (isinstance(f,int) and f&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)):allowed(p)
    elif e in {'os.remove','os.rmdir','os.mkdir','os.chmod','os.utime','os.truncate'}:allowed(a[0])
    elif e in {'os.rename','os.replace'}:allowed(a[0]);allowed(a[1])
    elif e in {'os.link','os.symlink'}:allowed(a[1])
sys.addaudithook(audit)
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
