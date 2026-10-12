import time,os,sys,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
while not (HERE/'methods/maxfuse_hub0/manifest.json').exists():time.sleep(5)
env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
subprocess.run([sys.executable,str(HERE/'scripts/run_maxfuse.py'),'0'],check=True,env=env,cwd=HERE)
