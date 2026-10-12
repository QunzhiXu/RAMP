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
    if not p.is_relative_to(BASE): raise PermissionError(f'Write outside celltype_specificity: {p}')

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
MARKERS = ['CD73','IDO']
CONFIGS = ['RAMP_k30','RAMP_k100']
IMMUNE_ENDPOINTS = ['immune_of_all','CD4_of_immune','Bcell_of_immune','Neutrophil_of_immune','CD8_TCF1','CD4_TCF1']
INPUTS = {'metadata':STORY/'data_cords/data/CELL_METADATA.csv.gz',
          'predictions':STORY/'data_cords/results/ALL_CORDS_PREDICTIONS.csv.gz',
          'raw_shared9':STORY/'data_cords/data/RAW_SHARED9.npy',
          'shared_markers':STORY/'data_cords/data/MARKERS.csv',
          'immune':MECH/'patient_immune/results/PATIENT_IMMUNE_COMPOSITION.csv',
          'immune_markers':MECH/'patient_immune/results/PATIENT_LINEAGE_MARKERS.csv',
          'old_vcaf_associations':MECH/'results/PATIENT_ASSOCIATIONS.csv',
          'old_vcaf_measurements':MECH/'results/VCAF_MEASURED_AND_PREDICTED.csv.gz',
          'old_immune_measurements':MECH/'patient_immune/data/MEASURED_MARKERS.csv.gz',
          'thresholds':STORY/'data_cords/results/HIGH_STATE_THRESHOLDS.csv',
          'provider_sce':ROOT/'raw_data/lung_cancer/Cords2024/SingleCellExperiment Objects/sce_all_annotated.rds'}
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
