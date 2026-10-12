import sys, subprocess, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import common
here = Path(__file__).parent
logs = here.parent / 'results' / 'logs_m5'; logs.mkdir(parents=True, exist_ok=True)
procs = {}
for t in [x['name'] for x in common.D['tasks']]:
    f = open(logs / f'{t}.log', 'w')
    procs[t] = (subprocess.Popen([sys.executable, '-u', str(here / 'm5_fewshot.py'), t],
                                 cwd=here, stdout=f, stderr=subprocess.STDOUT), f)
    print('launched', t, flush=True)
t0 = time.time()
while any(p.poll() is None for p, _ in procs.values()):
    time.sleep(30)
for t, (p, f) in procs.items():
    f.close(); print(f'{t}: exit {p.returncode}')
print(f'done in {(time.time()-t0)/60:.0f} min')
