"""Finish this running experiment with scoring/audits; no scheduled recurrence."""
import json
import subprocess
import sys
import time
from pathlib import Path

OUT=Path(__file__).resolve().parent
deadline=time.monotonic()+3*3600
while not (OUT/'RUN_COMPLETE.json').exists():
    if time.monotonic()>deadline:
        raise TimeoutError('Experiment did not finish within three hours; inspect run.log')
    time.sleep(10)
run=json.loads((OUT/'RUN_COMPLETE.json').read_text())
assert run['all_passed'],'Method failure; do not publish an incomplete ranking'
with (OUT/'summary.log').open('w') as log:
    subprocess.run([sys.executable,'-B',str(OUT/'summarize.py')],stdout=log,
                   stderr=subprocess.STDOUT,check=True,cwd=OUT)
checks=json.loads((OUT/'CHECKS.json').read_text())
assert checks['jobs']==1080 and checks['original_files_unchanged']
(OUT/'FINALIZATION_COMPLETE.json').write_text(json.dumps(dict(
    finished_unix=time.time(),jobs=1080,scoring_and_checks_passed=True,
    report=str(OUT/'README.md')),indent=2)+'\n')
print('Final report and all paired audits complete.',flush=True)
