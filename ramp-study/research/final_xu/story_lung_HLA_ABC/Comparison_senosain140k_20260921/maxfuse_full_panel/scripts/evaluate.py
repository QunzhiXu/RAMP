"""Run the unchanged evaluator, routing new MaxFuse outputs to this directory."""
import os,sys,json,hashlib,importlib.util
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parents[1];COMP=HERE.parent
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
for var,leaf in [('MPLCONFIGDIR','matplotlib'),('NUMBA_CACHE_DIR','numba'),('XDG_CACHE_HOME','xdg')]:os.environ[var]=str(HERE/'.cache/evaluation'/leaf)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
key=sys.argv[1];model=HERE/'methods'/key/'COMPLETE.json';record=json.loads(model.read_text());assert sha(HERE/'methods'/key/'values.npy')==record['values_sha256']
source=COMP/'scripts/evaluate.py';spec=importlib.util.spec_from_file_location('unchanged_evaluation_adapter',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.HERE=HERE;m.OUT=HERE/'evaluation';os.environ['RAMP_EVALUATION_CACHE']=str(HERE/'.cache/evaluation')
m.OUT.mkdir(exist_ok=True)
m.run(key)
(HERE/'evaluation/replicates'/(key+'_full_panel_provenance.json')).write_text(json.dumps({'unchanged_adapter':str(source),'adapter_sha256':sha(source),'full_panel_manifest_sha256':sha(HERE/'data/manifest.json'),'model_complete_sha256':sha(model),'routing_wrapper_sha256':sha(__file__),'numerical_evaluator_unchanged':True},indent=2)+'\n')
