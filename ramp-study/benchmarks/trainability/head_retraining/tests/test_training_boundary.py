"""Synthetic boundary check, separate from real-cohort numerical evidence."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import contract as c

SCRIPT=r'''
import json,sys
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(sys.argv[1])/'src'))
from trainability.inputs import FittingPool,PreparedRepresentation
from trainability.real_data import RealTask
from trainability.sampling import Supervision
from trainability.heads import Candidate
import trainability.runner as runner
from trainability.core import ROOT
rng=np.random.default_rng(17);x=rng.normal(size=(80,4));ids=tuple('cell'+str(i) for i in range(80))
md=pd.DataFrame(dict(cell_id=ids,dataset=['source']*60+['target']*20,unit_key=['s0']*30+['s1']*30+['t0']*20,
 patient_id=['s0']*30+['s1']*30+['t0']*20,identity_level=['patient']*80,celltype=['A' if v>0 else 'B' for v in x[:,0]]))
pool=FittingPool('synthetic',ids,tuple(md.dataset),tuple(md.unit_key));rep=PreparedRepresentation(x,ids,pool.sha256,'unaligned')
plan=Supervision(ids[:24],ids[30:54],());results=[]
for family in ('knn','linear','mlp'):
 predictions=[]; selections=[]
 for variant in range(2):
  metadata=md.copy()
  if variant:metadata.loc[60:,'celltype']=metadata.loc[60:,'celltype'].map({'A':'B','B':'A'})
  task=RealTask('synthetic',x,metadata,tuple('abcd'),{},dict(classes=['A','B'],canonical_markers={},pilot=True),pool)
  output=ROOT/(family+str(variant));original=runner.classification_score
  def guarded(*a,**kw):
   assert (output/'predictions.npz').is_file() and (output/'predictions.json').is_file()
   return original(*a,**kw)
  runner.classification_score=guarded
  try:r=runner.run_supervised(task,rep,plan,sources=['source'],target='target',marker=None,candidates=[Candidate(family=family,max_epochs=5,patience=3)],seed=42,out=output,split_mode='patient_disjoint_validation')
  finally:runner.classification_score=original
  with np.load(output/'predictions.npz',allow_pickle=False) as f:predictions.append(f['predictions'].copy())
  s=r['selection'].copy();s.pop('seconds');selections.append(s)
 assert np.array_equal(*predictions) and selections[0]==selections[1]
 results.append(family)
print(json.dumps(dict(target_label_independence=results,predictions_frozen_before_scoring=True)))
'''

class TrainingBoundary(unittest.TestCase):
    def test_target_answers_do_not_select_heads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();plan=c.load_plan()
            for n,v in plan['training_files'].items():
                if n.startswith('code/'):c.copy(c.SOURCE/v['source'],root/Path(n).relative_to('code'),v['record'])
            p=subprocess.run([sys.executable,'-B','-c',SCRIPT,str(root)],env=c.child_environment(),capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stdout+p.stderr)
            result=json.loads(p.stdout);self.assertEqual(result['target_label_independence'],['knn','linear','mlp'])
            self.assertTrue(result['predictions_frozen_before_scoring'])

if __name__=='__main__':unittest.main()
