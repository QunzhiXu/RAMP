"""Read existing HNSCC correction and write new audit evidence; never rerun R."""
import argparse
from datetime import datetime, timezone
import importlib.util
import os
from pathlib import Path
import re
import shutil
import tempfile

import inventory as inv
import hnscc
import study
from validate_bm_execution import old_work
from validate_technical_execution import bm_baseline, command

TECHNICAL_REPORT_SHA = 'a9eb4316e3f0f4011f873a05a2bab2138b00301524ff4ddd70d1d7896a36099b'


def prior_technical(source):
    path = inv.HERE / 'validation/technical_execution_20261010/report.json'
    if inv.sha(path) != TECHNICAL_REPORT_SHA: raise ValueError('Accepted technical report changed')
    report = inv.read_json(path)
    for name, digest in report['current_code_sha256'].items():
        if inv.sha(inv.HERE / name) != digest: raise ValueError('Accepted technical code/doc changed: ' + name)
    checks = []
    for row in report['cases']:
        bundle, run = Path(row['bundle']), Path(row['run'])
        if inv.sha(bundle/'MANIFEST.json') != row['bundle_manifest_sha256'] or inv.sha(run/'RUN_MANIFEST.json') != row['run_manifest_sha256']:
            raise ValueError('Prior run manifest changed')
        checks.append({'case': row['case'], 'verification': study.verify_run(run,bundle),
                       'bundle_manifest_sha256': row['bundle_manifest_sha256'], 'run_manifest_sha256': row['run_manifest_sha256']})
    return {'report_sha256': TECHNICAL_REPORT_SHA, 'unchanged_current_files': len(report['current_code_sha256']),
            'cases': checks, 'BM': bm_baseline(source)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('source-root','python','output'): parser.add_argument('--'+flag,type=Path,required=True)
    a = parser.parse_args(argv); source=a.source_root.resolve(strict=True)
    if not a.output.resolve().is_relative_to(inv.HERE/'validation'): parser.error('Use a new validation directory')
    output=inv.reserve(a.output,(source,inv.SNAPSHOT,inv.HERE/'runs',inv.HERE/'provenance'))
    bundle=inv.HERE/'runs/hnscc_story_inputs_20261010'; run=inv.HERE/'runs/hnscc_story_correction_20261010'
    report={'utc':datetime.now(timezone.utc).isoformat(),'scope':'HNSCC correction from frozen method-ready input',
            'R_fit_by_validator':False,'provider_extraction':False,'clinical_analysis':False,'clean_install':False}
    env=os.environ.copy();env.pop('PYTHONPATH',None);env['PYTHONDONTWRITEBYTECODE']='1'
    try:
        report['prior_work']=old_work(source)
        report['technical_baseline']=prior_technical(source)
        spec,files=hnscc.verify_bundle(bundle)
        report['comparison']=hnscc.compare(run,bundle,source)
        report['bundle']={'path':str(bundle),'payload_files':len(files),'payload_bytes':sum(f['bytes'] for f in files),
                           'manifest_sha256':inv.sha(bundle/'MANIFEST.json')}
        report['run']={'path':str(run),'manifest_sha256':inv.sha(run/'RUN_MANIFEST.json')}
        report['request']=inv.read_json(run/'REQUEST.json');report['commands']=inv.read_json(run/'COMMANDS.json')
        for name,record in report['request']['code_records'].items():inv.check(inv.HERE/name,record)
        report['runtime_copies']=[]
        for row in files:
            target=hnscc.runtime_path(row)
            if target is not None:
                inv.check(run/target,row)
                report['runtime_copies'].append({'runtime_path':str(target),**row})
        report['environment']=inv.read_json(run/'environment_before.stdout.log')
        report['story_runtime']=inv.read_json(run/'story_runtime_before.stdout.log')
        historical_session=source/'figure_HNSCC/Comparison/methods/cycombine/sessionInfo.txt'
        if 'data.table_1.17.8' not in historical_session.read_text():raise ValueError('Story dependency version not historically supported')
        report['historical_data_table_version_source']={'path':str(historical_session),**inv.record(historical_session)}
        with tempfile.TemporaryDirectory(prefix='hnscc_cycombine_audit_') as tmpname:
            tmp=Path(tmpname).resolve();copied=tmp/'module'
            shutil.copytree(inv.HERE,copied,ignore=shutil.ignore_patterns('runs','validation','__pycache__','.DS_Store'))
            report['tests']=command([a.python.absolute(),'-B','-m','unittest','discover','-s',copied/'tests','-v'],tmp,env)
            report['tests_passed']=sum(map(int,re.findall(r'Ran (\d+) tests? in',report['tests']['stderr'])))
            moved=tmp/'moved_bundle';shutil.copytree(bundle,moved)
            report['moved_verification']={'reran_R':False,**command([a.python.absolute(),'-B',copied/'hnscc.py',
                'verify-run','--bundle',moved,'--run',run],tmp,env)}
        report['common_inputs']=command([a.python.absolute(),'-B',inv.STUDY/'benchmarks/integration_inputs/inputs.py',
            'verify-staged','--input-root',inv.STUDY/'benchmarks/integration_inputs/runs/frozen_inputs_20261010'],output,env)
        module_spec=importlib.util.spec_from_file_location('bm_readonly',inv.STUDY/'benchmarks/bone_marrow/verify.py')
        module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
        report['BM_RAMP_checks']=module.verify(inv.STUDY/'benchmarks/bone_marrow/runs/rebuild_20261010')
        if not report['BM_RAMP_checks']['all_reference_checks_passed']:raise ValueError('BM RAMP results changed')
        report['postcheck']=old_work(source)
        prior_technical(source);hnscc.verify_run(run,bundle)
        report['current_code_sha256']={name:inv.sha(inv.HERE/name) for name in
            ('hnscc.py','check_hnscc_runtime.R','tests/test_hnscc.py','validate_hnscc_execution.py',
             'STUDY_HNSCC.md','AUDIT_HNSCC_EXECUTION.md')}
        report['status']='passed';inv.create_json(output/'report.json',report)
    except BaseException as error:
        inv.create_json(output/'FAILED.json',{**report,'error':repr(error)});raise
    print(output/'report.json')


if __name__=='__main__':main()
