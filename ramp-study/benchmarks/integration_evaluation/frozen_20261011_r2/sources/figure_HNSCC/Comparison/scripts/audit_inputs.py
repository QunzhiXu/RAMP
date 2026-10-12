import sys,json,hashlib
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parent.parent
manifest=json.loads((HERE/'input/manifest.json').read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
checks={}
for key in ['raw','analysis_ready']:
 x=np.load(HERE/'input'/(key+'.npy'))
 d=pd.read_csv(HERE/'input'/(key+'.csv.gz'),float_precision='round_trip',low_memory=False)
 assert np.array_equal(x,d.iloc[:,3:].to_numpy(float))
 assert np.array_equal(d.cell_id.astype(str),np.load(HERE/'input/cell_ids.npy'))
 checks[key+'_csv_float64_roundtrip_exact']=True
for p,h in manifest['source_sha256'].items():assert sha(p)==h
checks['all_original_source_sha256_unchanged']=True
if HERE.parent.name=='figure_melanoma':
 src=json.loads((HERE.parent/'analysis/alignment/output/manifest.json').read_text())
 md=pd.read_csv(HERE/'input/fit_metadata.csv.gz');x=np.load(HERE/'input/raw.npy');markers=manifest['markers']
 detail=[]
 for ds in src['dataset_order']:
  for m,v in src['positive_fraction_new_markers'][ds].items():
   fraction=float((x[md.dataset.eq(ds),markers.index(m)]>0).mean());assert fraction==v
   detail.append({'dataset':ds,'marker':m,'source_record':v,'reassembled':fraction,'exact_match':True})
 checks['new_marker_source_recorded_positive_fractions']=detail
(HERE/'input_audit.json').write_text(json.dumps(checks,indent=2)+'\n')
print(HERE.parent.name,'INPUT AUDIT COMPLETE')
