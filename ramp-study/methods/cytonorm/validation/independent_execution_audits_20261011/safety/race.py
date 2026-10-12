import json
from pathlib import Path
import subprocess
import sys
import time
WORK=Path('/private/tmp/cytonorm_phase2_safety_zJXCnl54')
MOD=WORK/'module';race=WORK/'fit_contention';race.mkdir()
script='''from pathlib import Path
import sys,time
sys.path.insert(0,sys.argv[1]);import execution as ex
root=Path(sys.argv[2]);tag=sys.argv[3]
def stop(*a,**k):raise RuntimeError('AUDIT STOP BEFORE R')
ex.run_command=stop
(root/('ready'+tag)).touch()
while not (root/'go').exists():time.sleep(.01)
try:ex.fit(Path(sys.argv[1])/'runs/bone_marrow_inputs_20261011',Path('/Users/qxu6/Documents/New project/final_xu/algorithm/CytoNorm/.Rlib'),Path('/usr/local/bin/Rscript'),root/'one_output',root)
except FileExistsError:sys.exit(17)
except RuntimeError as e:
 if str(e)=='AUDIT STOP BEFORE R':sys.exit(33)
 raise
'''
ps=[subprocess.Popen([sys.executable,'-B','-c',script,str(MOD),str(race),str(i)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for i in range(2)]
deadline=time.monotonic()+20
while not all((race/('ready'+str(i))).exists() for i in range(2)):
    assert time.monotonic()<deadline
    time.sleep(.01)
(race/'go').touch()
codes=[]
for p in ps:
    out,err=p.communicate(timeout=40);assert not err,(out,err);codes.append(p.returncode)
assert sorted(codes)==[17,33],codes
assert (race/'one_output/FAILED.json').exists()
assert not (race/'one_output/COMPLETE.json').exists()
assert not (race/'one_output/correction.stdout.log').exists()
with (WORK/'race.json').open('x') as f:json.dump({'exit_codes':codes,'one_owner_failed_directory_retained':True,'R_executed':False},f,indent=2)
print('controller contention passed')
