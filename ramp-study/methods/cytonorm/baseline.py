"""Read-only previous-module checks, isolated from this module's imports."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

import cytonorm_io as io

HERE = Path(__file__).resolve().parent
FAST_REPORT = '364e164ebeb602c66359636b8266b2638197b59d15053b0c0f29c63781cefce1'
SCRIPT = '''
import copy,json,sys
from pathlib import Path
root,source,out=map(Path,sys.argv[1:4]); mod=root/'methods/fastmnn';sys.path.insert(0,str(mod))
import baseline,compare_execution as comp,fastmnn_io as io,inventory as inv
p=mod/'validation/batch_boundary_fix_20261010/report.json'
if io.sha(p)!=sys.argv[4]:raise ValueError('Accepted fastMNN report changed')
r=io.read_json(p)
for n,row in r['current_files'].items():io.check(mod/n,row)
prior=baseline.capture(root,source,out)
cases={}
for spec in inv.verify_local()['branches']:
 c=spec['case']; result=comp.compare(mod/'runs/seven_fit_20261010_r2'/c,mod/'runs'/(c+'_inputs_20261010'),source)
 if result['csv_maximum_absolute_difference']!=0:raise ValueError('fastMNN result differs')
 cases[c]=result
print(json.dumps({'fastmnn_report_sha256':io.sha(p),'fastmnn_source':inv.verify_source(source),'cases':cases,'earlier_modules':prior}))
'''


def existing(study):
    rows = {}
    for folder, dirs, files in os.walk(study):
        rel = Path(folder).relative_to(study)
        dirs[:] = sorted(d for d in dirs if not d.startswith('.') and d not in ('runs','__pycache__')
                         and (rel/d).as_posix() != 'methods/cytonorm')
        for n in sorted(files):
            if n.startswith('.'): continue
            p = Path(folder)/n; name = p.relative_to(study).as_posix()
            rows[name] = {'symlink':str(p.resolve(strict=True))} if p.is_symlink() else io.record(p)
    return rows


def check_existing(study, rows):
    if existing(study) != rows: raise ValueError('Existing study files changed')


def capture(study, source, output):
    env = dict(os.environ);env.pop('PYTHONPATH',None);env['PYTHONDONTWRITEBYTECODE']='1'
    with (output/'baseline.stdout.log').open('x') as stdout, (output/'baseline.stderr.log').open('x') as stderr:
        p=subprocess.run([sys.executable,'-B','-c',SCRIPT,str(study),str(source),str(output),FAST_REPORT],
                         env=env,stdout=stdout,stderr=stderr,cwd=output)
    if p.returncode: raise RuntimeError('Previous-module verification failed; see logs')
    return io.read_json(output/'baseline.stdout.log')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('study-root','source-root','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();study=a.study_root.resolve(strict=True);source=a.source_root.resolve(strict=True)
    if not a.output.resolve().is_relative_to(HERE/'validation'):p.error('Use a new module validation directory')
    out=io.reserve(a.output,(source,study/'methods/fastmnn',study/'methods/harmony',study/'methods/cycombine',study/'methods/maxfuse'))
    try:
        rows=existing(study);prior=capture(study,source,out);check_existing(study,rows)
        io.create_json(out/'report.json',{'status':'passed','existing_files':rows,'prior':prior,'tool':io.record(HERE/'baseline.py')})
    except BaseException as e:io.failure(out,e);raise
    print(out/'report.json')


if __name__=='__main__':main()
