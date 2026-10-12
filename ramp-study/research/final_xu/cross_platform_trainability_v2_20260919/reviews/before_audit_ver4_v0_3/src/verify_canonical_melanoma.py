"""Reproducibility check only: two clean full-cohort RAMP fits, no head scoring."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np
from trainability.real_data import load_task
from trainability.adapters import fit_representation
from trainability.core import ROOT,require,write_json,file_sha256

task=load_task('melanoma_story',ROOT/'data/real_inputs_v0_3_r2')
out=ROOT/'validation/canonical_melanoma_v0_3'
reps=[]
for repeat in [1,2]:
    print('Canonical melanoma repeat',repeat,flush=True)
    reps.append(fit_representation(task.input('ramp'),'ramp',fitting_pool=task.pool,
        out=out/f'fit{repeat}',input_contract=task.receipt,family='trainability_v0.3:melanoma_story'))
delta=float(np.max(np.abs(reps[0].x-reps[1].x)))
require(delta==0,'Canonical RAMP refits are not bitwise reproducible under pinned threads')
old=ROOT.parent/'figure_melanoma/Comparison/input/ramp.npy'
historical=np.load(old,allow_pickle=False)
write_json(out/'REPORT.json',dict(status='passed',cells=len(task.metadata),markers=len(task.markers),
    bitwise_equal=True,max_repeat_absolute_difference=delta,
    historical_reference_sha256=file_sha256(old),
    max_difference_from_historical=float(np.abs(reps[0].x-historical).max()),
    mean_difference_from_historical=float(np.abs(reps[0].x-historical).mean()),
    interpretation='New canonical v0.3 family/thread-pinned fit; not a retroactive repair or reproduction claim for the historical figure.',
    target_labels_used=False,downstream_models_trained=False))
print('Canonical reproducibility passed',flush=True)
