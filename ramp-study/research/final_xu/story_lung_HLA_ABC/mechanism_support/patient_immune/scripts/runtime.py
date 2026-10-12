from pathlib import Path
import os, sys, json, hashlib, datetime
BASE = Path(__file__).resolve().parents[1]
MECH = BASE.parent
STORY = MECH.parent
ROOT = STORY.parent
sys.dont_write_bytecode = True

def allowed(p):
    if isinstance(p, int) or p is None: return
    p = Path(os.fsdecode(p)).resolve()
    if p == Path('/dev/null'): return
    if not p.is_relative_to(BASE): raise PermissionError(f'Write outside patient_immune: {p}')

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
import numpy as np
import pandas as pd
from scipy import stats
DATA = BASE/'data'
OUT = BASE/'results'
FIG = OUT/'figures'
SEED = 20260909
LINEAGES = ['CD8','CD4','Bcell','Myeloid','Neutrophil']
PRIMARY = [('CD8','PD1'),('CD8','TCF1'),('CD4','PD1'),('CD4','TCF1'),('CD4','FOXP3'),('Myeloid','IDO'),('Myeloid','CD73'),('Bcell','CD73'),('Neutrophil','IDO'),('Neutrophil','CD73')]
SECONDARY = [('CD4_nonTreg','PD1'),('CD4_nonTreg','TCF1'),('Treg','PD1'),('Treg','TCF1'),('Treg','CD73'),('Treg','IDO'),('Bcell','HLA_DR'),('Myeloid','HLA_DR'),('Neutrophil','MPO'),('Neutrophil','MMP9')]
MARKERS = ['PD1','TCF1','FOXP3','CD73','IDO','HLA_DR','KI67','MPO','MMP9','CD3','CD4','CD8','CD68','CD20','CD15']
INPUTS = {'clinical':MECH/'patient_cd8/results/PATIENT_CD8.csv', 'metadata':STORY/'data_cords/data/CELL_METADATA.csv.gz', 'previous_measured':MECH/'data/MEASURED_MARKERS.csv.gz', 'previous_adjusted':MECH/'patient_cd8/results/ADJUSTED_COMPARISONS.csv', 'provider_sce':ROOT/'raw_data/lung_cancer/Cords2024/SingleCellExperiment Objects/sce_all_annotated.rds'}
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def save(p, obj): Path(p).write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n')
def bh(p):
    p=np.asarray(p,float); good=np.isfinite(p); out=np.full(len(p),np.nan); a=p[good]; ix=np.argsort(a); q=np.empty(len(a))
    q[ix]=np.minimum(1,np.minimum.accumulate((a[ix]*len(a)/np.arange(1,len(a)+1))[::-1])[::-1]); out[good]=q
    return out
def ols_hc3(y, x, term=1):
    x=np.asarray(x,float); y=np.asarray(y,float)
    pinv=np.linalg.pinv(x); beta=pinv@y; res=y-x@beta; hat=np.sum(x*pinv.T,axis=1)
    weighted=pinv*(res/np.maximum(1-hat,1e-8))[None,:]
    cov=weighted@weighted.T; se=np.sqrt(max(cov[term,term],0)); df=len(y)-np.linalg.matrix_rank(x)
    critical=stats.t.ppf(.975,df); p=float(2*stats.t.sf(abs(beta[term]/se),df))
    return dict(coef=float(beta[term]),se=float(se),ci_low=float(beta[term]-critical*se),ci_high=float(beta[term]+critical*se),p=p,n=len(y),df=int(df),max_leverage=float(hat.max()))
def design_matrix(g, exposure='high_HLA', full=False):
    x=pd.DataFrame({'intercept':np.ones(len(g))},index=g.index)
    e=g[exposure].astype(float)
    if exposure=='all_mean': e=(e-e.mean())/e.std(ddof=1)
    x['exposure']=e; x['age']=(g.age-g.age.mean())/g.age.std(ddof=1)
    x=pd.concat([x,pd.get_dummies(g[['TMA','sex','Stage']].astype(str),drop_first=True,dtype=float)],axis=1)
    if full:
        for col in ['caf_fraction','old_global_CAF_score','vcaf_fraction']: x[col]=(g[col]-g[col].mean())/g[col].std(ddof=1)
    return x
def compare(g, col, bootstrap=3000):
    a=g.loc[g.group.eq('Lower'),col].to_numpy(); b=g.loc[g.group.eq('Higher'),col].to_numpy()
    test=stats.mannwhitneyu(b,a,alternative='two-sided',method='asymptotic')
    rng=np.random.default_rng(SEED)
    differences=b[rng.integers(len(b),size=(bootstrap,len(b)))].mean(1)-a[rng.integers(len(a),size=(bootstrap,len(a)))].mean(1)
    lo,hi=np.quantile(differences,[.025,.975])
    return dict(n_lower=len(a),n_higher=len(b),lower_mean=float(a.mean()),higher_mean=float(b.mean()),lower_median=float(np.median(a)),higher_median=float(np.median(b)),difference=float(b.mean()-a.mean()),ci_low=float(lo),ci_high=float(hi),p=float(test.pvalue),rank_biserial=float(2*test.statistic/(len(a)*len(b))-1))
