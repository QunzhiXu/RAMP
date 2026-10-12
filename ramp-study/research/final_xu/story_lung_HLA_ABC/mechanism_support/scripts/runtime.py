"""Restrict all new writes to this analysis package."""
from pathlib import Path
import os, sys, json, hashlib, datetime
BASE = Path(__file__).resolve().parents[1]
STORY = BASE.parent
ROOT = STORY.parent
sys.dont_write_bytecode = True
def allowed(path):
    if isinstance(path, int) or path is None: return
    path = Path(os.fsdecode(path)).resolve()
    if path == Path('/dev/null'): return
    if path != BASE and BASE not in path.parents:
        raise PermissionError(f'Write outside mechanism_support: {path}')
def audit(event, args):
    if event == 'open':
        path, mode, flags = args
        if (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (isinstance(flags, int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)):
            allowed(path)
    elif event in {'os.remove','os.rmdir','os.mkdir','os.chmod','os.utime','os.truncate'}: allowed(args[0])
    elif event in {'os.rename','os.replace'}: allowed(args[0]); allowed(args[1])
    elif event in {'os.link','os.symlink'}: allowed(args[1])
sys.addaudithook(audit)
for key, path in {'TMPDIR':BASE/'tmp','TMP':BASE/'tmp','TEMP':BASE/'tmp','MPLCONFIGDIR':BASE/'cache/matplotlib','XDG_CACHE_HOME':BASE/'cache/xdg'}.items():
    path.mkdir(parents=True,exist_ok=True); os.environ[key]=str(path)
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']: os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
INPUTS={
    'metadata':STORY/'data_cords/data/CELL_METADATA.csv.gz',
    'predictions':STORY/'data_cords/results/ALL_CORDS_PREDICTIONS.csv.gz',
    'raw_shared9':STORY/'data_cords/data/RAW_SHARED9.npy',
    'shared_markers':STORY/'data_cords/data/MARKERS.csv',
    'clinical':STORY/'survival/results/PATIENT_INPUTS.csv',
    'provider_sce':ROOT/'raw_data/lung_cancer/Cords2024/SingleCellExperiment Objects/sce_all_annotated.rds',
    'provider_panel':ROOT/'raw_data/lung_cancer/Cords2024/cp_csv/panel.csv',
}
