"""Verify phase-1 migration and prepared inputs; never fit real cohorts."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

import cytonorm_io as io
import inventory as inv
import study
import baseline
import probe_runtime

PREFLIGHT_SHA='3099bf86316519d18dbc7a4e618baebe42839a47679d0ddbd68ee0d4976298aa'


def code_files():
    return {p.relative_to(inv.HERE).as_posix():io.record(p)for p in sorted(inv.HERE.rglob('*'))
        if p.is_file() and not p.is_symlink() and not any(n in ('validation','runs','frozen_20261011','__pycache__') or n.startswith('.') for n in p.relative_to(inv.HERE).parts)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source-root','study-root','runs-root','preflight','runtime','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();source=a.source_root.resolve(strict=True);root=a.study_root.resolve(strict=True);runs=a.runs_root.resolve(strict=True)
    before_path=a.preflight.resolve(strict=True)/'report.json'
    if io.sha(before_path)!=PREFLIGHT_SHA:raise ValueError('Preflight anchor changed')
    before=io.read_json(before_path)
    if before['status']!='passed':raise ValueError('Preflight failed')
    baseline.check_existing(root,before['existing_files'])
    if not a.output.resolve().is_relative_to(inv.HERE/'validation'):p.error('Use a new directory under this module/validation')
    output=io.reserve(a.output,(source,runs,inv.SNAPSHOT,a.preflight,a.runtime));codes=code_files()
    report={'utc':datetime.now(timezone.utc).isoformat(),'scope':'Seven CytoNorm input bundles and frozen adapter; synthetic execution only',
        'real_cohort_fit':False,'provider_rebuild':False,'new_UMAP_metrics_survival':False,'clean_install':False,
        'preflight':{'path':str(before_path),'sha256':PREFLIGHT_SHA}}
    env=os.environ.copy();env.pop('PYTHONPATH',None);env.pop('LANG',None);env['PYTHONDONTWRITEBYTECODE']='1'
    def command(argv,key,cwd):
        with (output/(key+'.stdout.log')).open('x')as stdout,(output/(key+'.stderr.log')).open('x')as stderr:
            result=subprocess.run(list(map(str,argv)),cwd=cwd,env=env,stdout=stdout,stderr=stderr)
        if result.returncode:raise RuntimeError(key+' failed; see logs')
        print('PASS '+key,flush=True)
        return {'command':list(map(str,argv)),'returncode':result.returncode,
            'logs':{n:io.record(output/(key+'.'+n+'.log'))for n in ('stdout','stderr')}}
    try:
        report['source_verification']=inv.verify_source(source);catalog=inv.verify_local()
        print('PASS source verification',flush=True)
        runtime=a.runtime.resolve(strict=True);r=io.read_json(runtime/'report.json')
        if r['status']!='passed' or r['real_cohort_fit'] is not False or len(r['commands'])!=1 or any(c['returncode']!=0 for c in r['commands']):raise ValueError('Invalid R evidence')
        for n,row in r['codes'].items():io.check(inv.HERE/n,row)
        for n,row in r['files'].items():io.check(runtime/n,row)
        req=io.read_json(runtime/'REQUEST.json');io.check(req['rscript'],req['executable'])
        contract=probe_runtime.check_summary(r['summary'])
        if req['runtime_contract']!=contract or probe_runtime.runtime_contract(Path(req['library']))!=contract:raise ValueError('Runtime request/library contract differs')
        if (r['summary']['adapter_helper_sha256']!=io.sha(inv.SNAPSHOT/'adapter/cytonorm_repro.R') or
            r['summary']['parameters']!=inv.PARAMETERS or [s['batches']for s in r['summary']['results']]!=[2,3] or
            any(s['maximum_absolute_difference']!=0 for s in r['summary']['results'])):raise ValueError('Synthetic parity differs')
        report['runtime']={'path':str(runtime/'report.json'),**io.record(runtime/'report.json'),'summary':r['summary']}
        report['bundles']=[]
        with tempfile.TemporaryDirectory(prefix='cytonorm_validation_')as name:
            temp=Path(name).resolve();module=temp/'module'
            shutil.copytree(inv.HERE,module,ignore=shutil.ignore_patterns('runs','validation','__pycache__','.DS_Store'))
            report['tests']=command([sys.executable,'-B','-m','unittest','discover','-s',module/'tests','-v'],'tests',temp)
            report['python_tests_passed']=sum(map(int,re.findall(r'Ran (\d+) tests? in',(output/'tests.stderr.log').read_text())))
            for spec in catalog['branches']:
                bundle=runs/(spec['case']+'_inputs_20261011');actual,files=study.verify_bundle(bundle)
                if actual!=spec:raise ValueError('Bundle case differs')
                m=io.read_json(bundle/'MANIFEST.json')
                if m['code_records']!=study.code_records():raise ValueError('Preparation code changed')
                for row in files:
                    original=io.within(source,row['source']) if row['origin']=='source' else io.within(inv.SNAPSHOT,row['snapshot_path'])
                    io.check(original,row)
                    if (bundle/row['path']).read_bytes()!=original.read_bytes():raise ValueError('Bundle bytes differ from original')
                with tempfile.TemporaryDirectory(prefix='moved bundle ',dir=temp)as moved_name:
                    moved=Path(moved_name)/'inputs';shutil.copytree(bundle,moved)
                    cmd=command([sys.executable,'-B',module/'study.py','verify','--bundle',moved],spec['case'],temp)
                report['bundles'].append({'case':spec['case'],'path':str(bundle),'manifest_sha256':io.sha(bundle/'MANIFEST.json'),
                    'payload_files':len(files),'bytes':sum(row['bytes']for row in files),'counts':m['counts'],
                    'moved_verification':cmd,'fit_performed':False})
        report['common_inputs']=command([sys.executable,'-B',root/'benchmarks/integration_inputs/inputs.py','verify-staged',
            '--input-root',root/'benchmarks/integration_inputs/runs/frozen_inputs_20261010'],'common_inputs',output)
        prior=baseline.capture(root,source,output)
        if prior!=before['prior']:raise ValueError('Accepted prior evidence changed')
        baseline.check_existing(root,before['existing_files']);report['unchanged_existing_files']=len(before['existing_files'])
        report['prior_verification']={'fastmnn_report_sha256':baseline.FAST_REPORT,
            'stdout':io.record(output/'baseline.stdout.log'),'stderr':io.record(output/'baseline.stderr.log'),'equal_to_preflight':True}
        report['postcheck']=inv.verify_source(source)
        if report['postcheck']!=report['source_verification']:raise ValueError('Source verification changed')
        if code_files()!=codes:raise ValueError('Module changed during validation')
        report['current_files']=codes;report['catalog_sha256']=inv.CATALOG_SHA;report['status']='passed'
        io.create_json(output/'report.json',report)
    except BaseException as error:
        io.create_json(output/'FAILED.json',{**report,'error':repr(error)});raise
    print(output/'report.json')


if __name__=='__main__':main()
