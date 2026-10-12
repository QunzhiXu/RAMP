import runtime as rt
from runtime import BASE
import sys,subprocess,os
mode=sys.argv[1] if len(sys.argv)>1 else 'all'
def py(name):subprocess.run([sys.executable,'-B',str(BASE/'scripts'/name)],cwd=BASE,check=True)
if mode=='all':
    if not (BASE/'results/PREDICTION_FREEZE.json').exists():py('build_states.py')
    if not (BASE/'results/ASSOCIATION_VERIFICATION.json').exists():py('associate.py')
    if not (BASE/'results/SPATIAL_DFS_COX.csv').exists():
        env=os.environ.copy();env.update(TMPDIR='tmp',TMP='tmp',TEMP='tmp');subprocess.run(['Rscript','--vanilla','scripts/survival.R','.'],cwd=BASE,env=env,check=True)
    if not (BASE/'results/IDENTITY_VERIFICATION.json').exists():py('identity_check.py')
    py('report.py');py('verify.py')
elif mode in ['report','verify']:py(mode+'.py')
else:raise SystemExit('Usage: run.sh [all|report|verify]')
