import json
from pathlib import Path
import subprocess
import sys

WORK=Path('/private/tmp/cytonorm_phase2_safety_zJXCnl54')
MOD=WORK/'module_final'
ORIG=Path('/Users/qxu6/Documents/New project/ramp-study/methods/cytonorm')
SOURCE=ORIG.parents[2]/'final_xu'
sys.path.insert(0,str(MOD))
import compare_execution as comp
import execution as ex
import inventory as inv
import cytonorm_io as io

assert io.sha(MOD/'compare_models.R')=='9a66fc7be49e586466f616351a32631ef1f6dceccf4d35d87302d8ef3e10b0b5'
results={}
for s in inv.verify_local()['branches']:
    run=ORIG/'runs/seven_fit_20261011'/s['case'];bundle=ORIG/'runs'/(s['case']+'_inputs_20261011')
    value=comp.compare(run,bundle,SOURCE)
    with (WORK/('semantic_'+s['case']+'.json')).open('x') as f:json.dump(value,f,indent=2)
    results[s['case']]={'csv_difference':value['csv_maximum_absolute_difference'],
       'npy_difference':value.get('npy_maximum_absolute_difference'),
       'model':value['model_semantic_comparison'],'differing_files':list(value['differing_payload_hash_pairs'])}
    print('PASS exact semantic',s['case'],flush=True)
bm=next(s for s in inv.verify_local()['branches'] if s['case']=='bone_marrow')
old=SOURCE/bm['result_dir']/'model.rds'
new=ORIG/'runs/seven_fit_20261011/bone_marrow'/ex.result_path(bm)/'model.rds'
library=SOURCE/'algorithm/CytoNorm/.Rlib'
command=['/usr/local/bin/Rscript','--vanilla',str(MOD/'tests/check_model_comparison.R'),str(MOD/'compare_models.R'),str(old),str(new),str(library)]
with (WORK/'model_negative.stdout.log').open('x') as out,(WORK/'model_negative.stderr.log').open('x') as err:
    p=subprocess.run(command,env=ex.environment(),cwd=WORK,stdout=out,stderr=err)
assert p.returncode==0
negative=io.read_json(WORK/'model_negative.stdout.log');assert negative['status']=='passed' and negative['tests']==15
with (WORK/'semantic_summary.json').open('x') as f:json.dump({'cases':results,'mutation_tests':negative},f,indent=2)
print('PASS 15 semantic mutation tests',flush=True)
