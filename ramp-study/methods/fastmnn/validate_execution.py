"""Read-only preflight/final verification for all seven real fastMNN fits.

Writes only a new evidence directory; never launches real fitting.
"""
import argparse
from datetime import datetime, timezone
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

PHASE1_SHA = 'c864526be0df1e56c3503aa9a49e3d92adbee19e3eec95b593fbecda683df2d2'
OUTSIDE_PREFLIGHT_SHA = 'a4b6e4704da7717a328e9a76c431143f8a88c37f02eb77bc98ca6a9beea45a6a'
NEW_FILES = ('execution.py','check_environment.R','compare_execution.py','validate_execution.py',
             'tests/test_execution.py','STUDY_EXECUTION.md','AUDIT_EXECUTION.md')


def existing_files():
    result = {}
    for p in sorted(inv.HERE.rglob('*')):
        relative = p.relative_to(inv.HERE)
        if any(n.startswith('.') or n=='__pycache__' for n in relative.parts): continue
        if p.is_symlink(): result[str(relative)] = {'symlink':str(p.resolve(strict=True))}
        elif p.is_file(): result[str(relative)] = io.record(p)
    return result


def unchanged(rows):
    for name,row in rows.items():
        p=inv.HERE/name
        if 'symlink' in row:
            if not p.is_symlink() or str(p.resolve(strict=True))!=row['symlink']: raise ValueError('Prior link changed')
        else: io.check(p,row)


def accepted_phase():
    path = inv.HERE/'validation/handoff_20261010_r2/report.json'
    if io.sha(path)!=PHASE1_SHA: raise ValueError('Accepted phase-1 report changed')
    r=io.read_json(path)
    for n,row in r['current_files'].items(): io.check(inv.HERE/n,row)
    inv.verify_local()
    for row in r['bundles']:
        bundle=inv.HERE/'runs'/(row['case']+'_inputs_20261010')
        if io.sha(bundle/'MANIFEST.json')!=row['manifest_sha256']: raise ValueError('Accepted bundle changed')
        study.verify_bundle(bundle)
    return PHASE1_SHA


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('preflight','verify'))
    for n in ('source-root','study-root','bundles','output'): p.add_argument('--'+n,type=Path,required=True)
    for n in ('batch','preflight'): p.add_argument('--'+n,type=Path)
    p.add_argument('--preflight-sha256')
    a=p.parse_args()
    source,root,bundles=(x.resolve(strict=True) for x in (a.source_root,a.study_root,a.bundles))
    if a.mode=='verify' and not all((a.batch,a.preflight,a.preflight_sha256)): p.error('verify requires batch/preflight/preflight-sha256')
    if not a.output.resolve().is_relative_to(inv.HERE/'validation'): p.error('Evidence must be under this module/validation')
    outside=inv.HERE/'validation/execution_preflight_20261010/report.json'
    if io.sha(outside)!=OUTSIDE_PREFLIGHT_SHA: raise ValueError('Outside baseline changed')
    before=io.read_json(outside); baseline.check_existing(root,before['existing_files']); accepted_phase()
    codes={n:io.record(inv.HERE/n) for n in NEW_FILES}
    old=existing_files() if a.mode=='preflight' else None
    if a.mode=='verify':
        if io.sha(a.preflight/'report.json')!=a.preflight_sha256: raise ValueError('Execution preflight changed')
        pre=io.read_json(a.preflight/'report.json')
        if pre['status']!='passed' or pre['current_files']!=codes: raise ValueError('Preflight code state differs')
        unchanged(pre['existing_module_files'])
    output=io.reserve(a.output,(source,bundles,inv.SNAPSHOT,*(inv.HERE/n for n in NEW_FILES)))
    report={'utc':datetime.now(timezone.utc).isoformat(),'mode':a.mode,'scope':'Seven frozen-input fastMNN corrected embeddings',
        'validator_runs_real_fit':False,'provider_preprocessing_rebuilt':False,'UMAP_metrics_survival':False,'clean_install':False,
        'phase1_report_sha256':PHASE1_SHA,'outside_preflight_sha256':OUTSIDE_PREFLIGHT_SHA}
    env=os.environ.copy();env.pop('PYTHONPATH',None);env.pop('LANG',None);env['PYTHONDONTWRITEBYTECODE']='1'
    def command(argv,key,cwd):
        with (output/(key+'.stdout.log')).open('x') as out,(output/(key+'.stderr.log')).open('x') as err:
            process=subprocess.run(list(map(str,argv)),cwd=cwd,env=env,stdout=out,stderr=err)
        if process.returncode: raise RuntimeError(key+' failed; see logs')
        print('PASS '+key,flush=True)
        return {'command':list(map(str,argv)),'returncode':process.returncode,
                'stdout':io.record(output/(key+'.stdout.log')),'stderr':io.record(output/(key+'.stderr.log'))}
    try:
        report['source_verification']=inv.verify_source(source)
        with tempfile.TemporaryDirectory(prefix='fastmnn_execution_verify_') as name:
            temp=Path(name).resolve();module=temp/'module'
            shutil.copytree(inv.HERE,module,ignore=shutil.ignore_patterns('runs','validation','__pycache__','.DS_Store'))
            report['tests']=command([sys.executable,'-B','-m','unittest','discover','-s',module/'tests','-v'],'tests',temp)
            report['python_tests_passed']=sum(map(int,re.findall(r'Ran (\d+) tests? in',(output/'tests.stderr.log').read_text())))
            if a.mode=='preflight':
                report['existing_module_files']=old
            else:
                batch=a.batch.resolve(strict=True);b=io.complete(batch,'BATCH_MANIFEST.json')
                specs=inv.verify_local()['branches']
                if b['status']!='complete' or b['history_used_by_fit'] is not False or [r['case'] for r in b['cases']]!=[s['case'] for s in specs]:
                    raise ValueError('Batch task list differs')
                report['cases']={}
                for spec,row in zip(specs,b['cases']):
                    case=spec['case'];bundle=bundles/(case+'_inputs_20261010');run=batch/case
                    if row['run']!=case: raise ValueError('Batch path differs')
                    io.check(run/'RUN_MANIFEST.json',row['manifest'])
                    if row['result']!=ex.verify_run(run,bundle): raise ValueError('Batch result differs')
                    result=comparison.compare(run,bundle,source)
                    report['cases'][case]=result;print('PASS numerical comparison '+case,flush=True)
                    with tempfile.TemporaryDirectory(prefix='moved bundle ',dir=temp) as moved_name:
                        moved=Path(moved_name)/'inputs';shutil.copytree(bundle,moved)
                        result['moved_bundle_verification']=command([sys.executable,'-B',module/'execution.py','verify-run',
                            '--run',run,'--bundle',moved],'moved_'+case,temp)
                    req=io.read_json(run/'REQUEST.json');io.check(req['rscript'],req['rscript_record'])
                    with tempfile.TemporaryDirectory(prefix='fastmnn_env_',dir='/private/tmp') as rt:
                        old_env=env;env={**ex.environment(env),'TMPDIR':str(Path(rt).resolve())}
                        result['current_environment_check']=command(ex.command_plan(run,spec,req['rscript'])['environment'],'current_env_'+case,output)
                        env=old_env
                    if (output/('current_env_'+case+'.stdout.log')).read_bytes()!=(run/'environment_before.stdout.log').read_bytes():
                        raise ValueError('Current R environment differs from fit')
                report['batch_manifest']=io.record(batch/'BATCH_MANIFEST.json')
                report['execution_preflight_sha256']=a.preflight_sha256
                report['unchanged_existing_module_files']=len(pre['existing_module_files']);unchanged(pre['existing_module_files'])
        report['common_inputs']=command([sys.executable,'-B',root/'benchmarks/integration_inputs/inputs.py','verify-staged',
            '--input-root',root/'benchmarks/integration_inputs/runs/frozen_inputs_20261010'],'common_inputs',output)
        prior=baseline.capture(root,source,output)
        if prior!=before['prior']: raise ValueError('Prior accepted results changed')
        baseline.check_existing(root,before['existing_files']);report['unchanged_outside_files']=len(before['existing_files'])
        accepted_phase();report['postcheck']=inv.verify_source(source)
        if report['postcheck']!=report['source_verification']: raise ValueError('Source verification changed')
        for n,row in codes.items():io.check(inv.HERE/n,row)
        report['current_files']=codes;report['status']='passed';io.create_json(output/'report.json',report)
    except BaseException as error:
        io.create_json(output/'FAILED.json',{**report,'error':repr(error)});raise
    print(output/'report.json')


if __name__=='__main__':main()
