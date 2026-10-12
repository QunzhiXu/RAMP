"""Run all seven tasks, one process per task."""
import sys, subprocess, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import common

TASKS = [t['name'] for t in common.D['tasks']]
here = Path(__file__).parent
logs = here.parent / 'results' / 'logs'
logs.mkdir(parents=True, exist_ok=True)

procs = {}
for t in TASKS:
    f = open(logs / f'{t}.log', 'w')
    procs[t] = (subprocess.Popen([sys.executable, '-u', str(here / 'run_task.py'), t],
                                 cwd=here, stdout=f, stderr=subprocess.STDOUT), f)
    print(f'launched {t}', flush=True)

t0 = time.time()
while any(p.poll() is None for p, _ in procs.values()):
    time.sleep(30)
for t, (p, f) in procs.items():
    f.close()
    print(f'{t}: exit {p.returncode}')
print(f'all done in {(time.time()-t0)/60:.0f} min')
