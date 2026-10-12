from pathlib import Path
import os,json,subprocess,sys
from concurrent.futures import ThreadPoolExecutor,as_completed
HERE=Path(__file__).resolve().parent
def run(item):
    p=Path(item['output']);env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with (p/'render.log').open('w') as log:
        result=subprocess.run([sys.executable,str(p/'build_figure.py')],stdout=log,stderr=subprocess.STDOUT,env=env)
    if result.returncode:raise RuntimeError(item['number']+'\n'+(p/'render.log').read_text()[-3500:])
    return item['number']+' '+item['case']
with ThreadPoolExecutor(max_workers=2) as pool:
    for f in as_completed([pool.submit(run,p) for p in json.loads((HERE/'PLAN.json').read_text())]):
        print('Rendered:',f.result(),flush=True)
