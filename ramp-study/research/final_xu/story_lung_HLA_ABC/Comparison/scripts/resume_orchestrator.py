import os,signal,json,time,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
record=HERE/'parallel_dispatch.json';data=json.loads(record.read_text());pid=data['paused_orchestrator_pid']
while not all((HERE/'methods'/f'maxfuse_hub{i}'/'COMPLETE.json').exists() for i in range(3)):time.sleep(5)
try:
 command=subprocess.check_output(['ps','-p',str(pid),'-o','command='],text=True).strip()
 assert command.endswith('story_lung_HLA_ABC/Comparison/scripts/run_maxfuse.py'),'PID no longer belongs to original orchestrator'
 os.kill(pid,signal.SIGCONT)
 data['status']='resumed after all three official hub fits were completed'
except ProcessLookupError:
 data['status']='orchestrator already exited; all hub fits completed'
record.write_text(json.dumps(data,indent=2)+'\n')
print(data['status'],flush=True)
