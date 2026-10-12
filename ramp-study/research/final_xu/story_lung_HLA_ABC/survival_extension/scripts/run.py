import runtime as rt
from runtime import BASE
import os,subprocess,shutil
import prepare
prepare.main()
env=os.environ.copy()
for k in ['TMPDIR','TMP','TEMP']:env[k]='tmp'
env['R_PROFILE_USER']='/dev/null';env['R_ENVIRON_USER']='/dev/null'
subprocess.run([shutil.which('Rscript'),'--vanilla',str(BASE/'scripts/fit.R'),str(BASE)],cwd=BASE,env=env,check=True)
import report
report.main()
