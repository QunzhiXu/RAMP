"""Read-only prior-module checks in isolated child interpreters; no fitting."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import fastmnn_io as io

HERE = Path(__file__).resolve().parent
HARMONY_REPORT_SHA = '94d1cca60d91e0f7572b05fd1ac575e5574366adfcf8ec3cf4e6c7ec194ba155'
SCRIPT = '''
import json,sys,subprocess
from pathlib import Path
root,source=map(Path,sys.argv[1:3]); mod=root/'methods/harmony';sys.path.insert(0,str(mod))
import harmony_io as io
import inventory as inv
from validate_lung_execution import accepted_previous,check_case
from validate import BASELINE_SCRIPT
p=mod/'validation/lung_execution_20261010/report.json'
if io.sha(p)!=sys.argv[3]:raise ValueError('Accepted Harmony report changed')
r=io.read_json(p)
for n,row in r['current_files'].items():io.check(mod/n,row)
previous=accepted_previous(source,mod/'runs');lung=check_case(source,mod/'runs')
if previous!=r['accepted_previous'] or lung!=r['case']:raise ValueError('Accepted Harmony evidence changed')
old=subprocess.run([sys.executable,'-B','-c',BASELINE_SCRIPT,str(source),str(root/'methods/cycombine')],capture_output=True,text=True,check=True)
old=json.loads(old.stdout);old['BM_RAMP'].pop('checked_at')
print(json.dumps({'harmony_report_sha256':io.sha(p),'harmony_source':inv.verify_source(source),'previous':previous,'lung':lung,'earlier_modules':old}))
'''


def existing(study):
    rows = {}
    for p in sorted(Path(study).rglob('*')):
        rel = p.relative_to(study)
        if rel.parts[:2] == ('methods', 'fastmnn') or any(
            part.startswith('.') or part in ('runs', '__pycache__') for part in rel.parts
        ): continue
        if p.is_symlink(): rows[rel.as_posix()] = {'symlink': str(p.resolve(strict=True))}
        elif p.is_file(): rows[rel.as_posix()] = io.record(p)
    return rows


def check_existing(study, rows):
    if existing(study) != rows: raise ValueError('An existing study file/link was added, removed or changed')


def capture(study, source, output):
    env = os.environ.copy(); env.pop('PYTHONPATH', None); env['PYTHONDONTWRITEBYTECODE'] = '1'
    with (output/'prior.stdout.log').open('x') as stdout, (output/'prior.stderr.log').open('x') as stderr:
        result = subprocess.run([sys.executable, '-B', '-c', SCRIPT, str(study), str(source), HARMONY_REPORT_SHA],
                                stdout=stdout, stderr=stderr, env=env, cwd=output)
    if result.returncode: raise RuntimeError('Prior verification failed; see logs')
    return io.read_json(output/'prior.stdout.log')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('study-root', 'source-root', 'output'): p.add_argument('--'+name, type=Path, required=True)
    a=p.parse_args(); study=a.study_root.resolve(strict=True);source=a.source_root.resolve(strict=True)
    output=io.reserve(a.output,(source,study/'methods/harmony',study/'methods/cycombine',study/'methods/maxfuse'))
    try:
        rows=existing(study); prior=capture(study,source,output);check_existing(study,rows)
        io.create_json(output/'report.json',{'status':'passed','utc':datetime.now(timezone.utc).isoformat(),
            'scope':'read-only preflight; no scientific computation','existing_files':rows,'prior':prior,
            'tool':io.record(HERE/'baseline.py')})
    except BaseException as error:io.failure(output,error);raise
    print(output/'report.json')


if __name__=='__main__': main()
