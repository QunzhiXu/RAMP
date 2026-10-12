"""Verify the output-boundary-only fix and retained numerical runs; no fitting."""
import argparse
import ast
import copy
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

import baseline
import compare_execution as comparison
import execution as ex
import fastmnn_io as io
import inventory as inv
import study

REPORT_SHA = '8babdb357d8323b4d3ab8fc3cd08b65fdbaa0c6e5578d3773904fd2a69c18a9e'
ARCHIVE = inv.HERE/'provenance/seven_execution_20261010'
CURRENT = ('execution.py','tests/test_execution.py','BATCH_BOUNDARY_FIX.md','validate_batch_boundary_fix.py')


def definitions(path):
    return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef)}


def check_change_scope():
    old,new=definitions(ARCHIVE/'execution.py'),definitions(inv.HERE/'execution.py')
    if set(new)-set(old)!={'check_controller_records'} or set(old)-set(new): raise ValueError('Unexpected function addition/removal')
    changed=[n for n in old if old[n]!=new[n]]
    if set(changed)!={'batch','verify_run'}: raise ValueError('Unexpected execution change: '+str(changed))
    if io.sha(ARCHIVE/'execution.py')!=ex.ACCEPTED_R2_CONTROLLER_SHA: raise ValueError('Archive anchor changed')
    return {'unchanged_functions':[n for n in old if old[n]==new[n]],'changed_functions':changed,
            'added_function':'check_controller_records','archived_controller_sha256':ex.ACCEPTED_R2_CONTROLLER_SHA}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source-root','study-root','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();source=a.source_root.resolve(strict=True);root=a.study_root.resolve(strict=True)
    if not a.output.resolve().is_relative_to(inv.HERE/'validation'):p.error('Use a new validation directory')
    rp=inv.HERE/'validation/seven_execution_20261010/report.json'
    if io.sha(rp)!=REPORT_SHA:raise ValueError('Accepted numerical report changed')
    accepted=io.read_json(rp);codes={n:io.record(inv.HERE/n) for n in CURRENT}
    for n,row in accepted['current_files'].items():
        io.check((ARCHIVE if n in ('execution.py','tests/test_execution.py') else inv.HERE)/n,row)
    pre=io.read_json(inv.HERE/'validation/seven_execution_preflight_20261010_r2/report.json')
    if io.sha(inv.HERE/'validation/seven_execution_preflight_20261010_r2/report.json')!=accepted['execution_preflight_sha256']:
        raise ValueError('Execution preflight changed')
    outside=io.read_json(inv.HERE/'validation/execution_preflight_20261010/report.json')
    baseline.check_existing(root,outside['existing_files'])
    output=io.reserve(a.output,(source,inv.HERE/'runs',inv.SNAPSHOT,ARCHIVE))
    report={'numerical_report_sha256':REPORT_SHA,'real_fit_repeated':False,'scope':'Batch output boundary and exact historical-controller recognition'}
    env=os.environ.copy();env.pop('PYTHONPATH',None);env.pop('LANG',None);env['PYTHONDONTWRITEBYTECODE']='1'
    def command(argv,key,cwd):
        with (output/(key+'.stdout.log')).open('x') as stdout,(output/(key+'.stderr.log')).open('x') as stderr:
            proc=subprocess.run(list(map(str,argv)),env=env,cwd=cwd,stdout=stdout,stderr=stderr)
        if proc.returncode:raise RuntimeError(key+' failed; see logs')
        print('PASS '+key,flush=True)
        return {'returncode':proc.returncode,'stdout':io.record(output/(key+'.stdout.log')),'stderr':io.record(output/(key+'.stderr.log'))}
    try:
        report['change_scope']=check_change_scope()
        phase=io.read_json(inv.HERE/'validation/handoff_20261010_r2/report.json')
        for n,row in phase['current_files'].items():io.check(inv.HERE/n,row)
        report['source']=inv.verify_source(source)
        with tempfile.TemporaryDirectory(prefix='fastmnn_boundary_verify_') as name:
            temp=Path(name).resolve();module=temp/'module'
            shutil.copytree(inv.HERE,module,ignore=shutil.ignore_patterns('runs','validation','__pycache__','.DS_Store'))
            report['tests']=command([sys.executable,'-B','-m','unittest','discover','-s',module/'tests','-v'],'tests',temp)
            report['python_tests_passed']=sum(map(int,re.findall(r'Ran (\d+) tests? in',(output/'tests.stderr.log').read_text())))
            report['cases']={}
            for spec in inv.verify_local()['branches']:
                case=spec['case'];bundle=inv.HERE/'runs'/(case+'_inputs_20261010');run=inv.HERE/'runs/seven_fit_20261010_r2'/case
                now=comparison.compare(run,bundle,source)
                old=copy.deepcopy(accepted['cases'][case]);old.pop('moved_bundle_verification');old.pop('current_environment_check')
                if now!=old:raise ValueError('Accepted numerical evidence changed: '+case)
                report['cases'][case]={'matches_accepted_report':True,'core_byte_identical':now['core_byte_identical'],
                    'csv_maximum_absolute_difference':now['csv_maximum_absolute_difference'],'run_manifest':now['run_manifest']}
                with tempfile.TemporaryDirectory(prefix='moved bundle ',dir=temp) as moved_name:
                    moved=Path(moved_name)/'bundle';shutil.copytree(bundle,moved)
                    report['cases'][case]['moved_verification']=command([sys.executable,'-B',module/'execution.py','verify-run',
                        '--bundle',moved,'--run',run],'moved_'+case,temp)
        for n,row in pre['existing_module_files'].items():
            path=(ARCHIVE if n in ('execution.py','tests/test_execution.py') else inv.HERE)/n
            if 'symlink' in row:
                if not path.is_symlink() or str(path.resolve(strict=True))!=row['symlink']:raise ValueError('Prior link changed')
            else:io.check(path,row)
        baseline.check_existing(root,outside['existing_files'])
        report['prior_files_checked']=len(pre['existing_module_files']);report['outside_files_checked']=len(outside['existing_files'])
        for n,row in codes.items():io.check(inv.HERE/n,row)
        if io.sha(rp)!=REPORT_SHA:raise ValueError('Accepted report changed')
        report['current_files']=codes;report['status']='passed';io.create_json(output/'report.json',report)
    except BaseException as error:io.create_json(output/'FAILED.json',{**report,'error':repr(error)});raise
    print(output/'report.json')


if __name__=='__main__':main()
