from pathlib import Path
import os,json,subprocess,sys
from concurrent.futures import ThreadPoolExecutor,as_completed
HERE=Path(__file__).resolve().parent
plan=json.loads((HERE/'REVISION_PLAN.json').read_text())
def run(item):
    folder=Path(item['new']);env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
    with (folder/'render.log').open('w') as log:
        result=subprocess.run([sys.executable,str(folder/'build_figure.py')],stdout=log,stderr=subprocess.STDOUT,env=env)
    if result.returncode:raise RuntimeError(item['name']+'\n'+(folder/'render.log').read_text()[-3500:])
    return item['name']
with ThreadPoolExecutor(max_workers=2) as pool:
    tasks=[pool.submit(run,item) for item in plan]
    for task in as_completed(tasks):print('Rendered:',task.result(),flush=True)
