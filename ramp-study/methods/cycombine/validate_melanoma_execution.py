"""Read existing Melanoma correction and write new audit evidence; never rerun R."""
import argparse
from datetime import datetime, timezone
import importlib.util
import os
from pathlib import Path
import re
import shutil
import tempfile

import inventory as inv
import melanoma
import study
from validate_bm_execution import old_work
from validate_technical_execution import command
from validate_hnscc_execution import prior_technical
import hnscc

HNSCC_REPORT_SHA = '6372c715bca6e360b51b703410ab50eb71aa452ee0ac14cef7a95dfbe6961371'


def prior_hnscc(source):
    path = inv.HERE / 'validation/hnscc_execution_20261010_r2/report.json'
    if inv.sha(path) != HNSCC_REPORT_SHA: raise ValueError('Accepted HNSCC report changed')
    report = inv.read_json(path)
    for name, digest in report['current_code_sha256'].items():
        if inv.sha(inv.HERE / name) != digest: raise ValueError('Accepted HNSCC code/doc changed: ' + name)
    bundle, run = Path(report['bundle']['path']), Path(report['run']['path'])
    if (inv.sha(bundle/'MANIFEST.json') != report['bundle']['manifest_sha256']
            or inv.sha(run/'RUN_MANIFEST.json') != report['run']['manifest_sha256']):
        raise ValueError('Accepted HNSCC run changed')
    return {'report_sha256': HNSCC_REPORT_SHA,
            'unchanged_files': len(report['current_code_sha256']),
            'comparison': hnscc.compare(run, bundle, source)}


def controller_derivation():
    """Only case/path/provenance strings differ; the accepted controller stays intact."""
    before = (inv.HERE / 'hnscc.py').read_text()
    expected = (before.replace('HNSCC', 'Melanoma').replace('hnscc', 'melanoma')
                .replace('figure_Melanoma', 'figure_melanoma')
                .replace('check_melanoma_runtime.R', 'check_hnscc_runtime.R'))
    if (inv.HERE / 'melanoma.py').read_text() != expected:
        raise ValueError('Unexpected changes beyond the declared controller derivation')
    _, earlier = hnscc.plan()
    _, current = melanoma.plan()
    scripts = lambda files: {r['path']: r['sha256'] for r in files if r['path'].startswith('scripts/')}
    if scripts(earlier) != scripts(current): raise ValueError('Historical story R/Python scripts differ')
    return {'controller_is_exact_case_path_derivative': True,
            'shared_historical_script_hashes': scripts(current)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('source-root','python','output'): parser.add_argument('--'+flag,type=Path,required=True)
    a = parser.parse_args(argv); source=a.source_root.resolve(strict=True)
    if not a.output.resolve().is_relative_to(inv.HERE/'validation'): parser.error('Use a new validation directory')
    output=inv.reserve(a.output,(source,inv.SNAPSHOT,inv.HERE/'runs',inv.HERE/'provenance'))
    bundle=inv.HERE/'runs/melanoma_story_inputs_20261010'; run=inv.HERE/'runs/melanoma_story_correction_20261010'
    report={'utc':datetime.now(timezone.utc).isoformat(),'scope':'Melanoma correction from frozen method-ready input',
            'R_fit_by_validator':False,'provider_extraction':False,'clinical_analysis':False,'clean_install':False}
    env=os.environ.copy();env.pop('PYTHONPATH',None);env['PYTHONDONTWRITEBYTECODE']='1'
    try:
        report['prior_work']=old_work(source)
        report['technical_baseline']=prior_technical(source)
        report['HNSCC_baseline']=prior_hnscc(source)
        report['controller_derivation']=controller_derivation()
        spec,files=melanoma.verify_bundle(bundle)
        report['comparison']=melanoma.compare(run,bundle,source)
        report['bundle']={'path':str(bundle),'payload_files':len(files),'payload_bytes':sum(f['bytes'] for f in files),
                           'manifest_sha256':inv.sha(bundle/'MANIFEST.json')}
        report['run']={'path':str(run),'manifest_sha256':inv.sha(run/'RUN_MANIFEST.json')}
        report['request']=inv.read_json(run/'REQUEST.json');report['commands']=inv.read_json(run/'COMMANDS.json')
        for name,record in report['request']['code_records'].items():inv.check(inv.HERE/name,record)
        report['runtime_copies']=[]
        for row in files:
            target=melanoma.runtime_path(row)
            if target is not None:
                inv.check(run/target,row)
                report['runtime_copies'].append({'runtime_path':str(target),**row})
        report['environment']=inv.read_json(run/'environment_before.stdout.log')
        report['story_runtime']=inv.read_json(run/'story_runtime_before.stdout.log')
        historical_session=source/'figure_melanoma/Comparison/methods/cycombine/sessionInfo.txt'
        if 'data.table_1.17.8' not in historical_session.read_text():raise ValueError('Story dependency version not historically supported')
        report['historical_data_table_version_source']={'path':str(historical_session),**inv.record(historical_session)}
        with tempfile.TemporaryDirectory(prefix='melanoma_cycombine_audit_') as tmpname:
            tmp=Path(tmpname).resolve();copied=tmp/'module'
            shutil.copytree(inv.HERE,copied,ignore=shutil.ignore_patterns('runs','validation','__pycache__','.DS_Store'))
            report['tests']=command([a.python.absolute(),'-B','-m','unittest','discover','-s',copied/'tests','-v'],tmp,env)
            report['tests_passed']=sum(map(int,re.findall(r'Ran (\d+) tests? in',report['tests']['stderr'])))
            moved=tmp/'moved_bundle';shutil.copytree(bundle,moved)
            report['moved_verification']={'reran_R':False,**command([a.python.absolute(),'-B',copied/'melanoma.py',
                'verify-run','--bundle',moved,'--run',run],tmp,env)}
        report['common_inputs']=command([a.python.absolute(),'-B',inv.STUDY/'benchmarks/integration_inputs/inputs.py',
            'verify-staged','--input-root',inv.STUDY/'benchmarks/integration_inputs/runs/frozen_inputs_20261010'],output,env)
        module_spec=importlib.util.spec_from_file_location('bm_readonly',inv.STUDY/'benchmarks/bone_marrow/verify.py')
        module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
        report['BM_RAMP_checks']=module.verify(inv.STUDY/'benchmarks/bone_marrow/runs/rebuild_20261010')
        if not report['BM_RAMP_checks']['all_reference_checks_passed']:raise ValueError('BM RAMP results changed')
        report['postcheck']=old_work(source)
        prior_technical(source);prior_hnscc(source);melanoma.verify_run(run,bundle)
        report['current_code_sha256']={name:inv.sha(inv.HERE/name) for name in
            ('melanoma.py','tests/test_melanoma.py','validate_melanoma_execution.py',
             'STUDY_MELANOMA.md','AUDIT_MELANOMA_EXECUTION.md')}
        report['status']='passed';inv.create_json(output/'report.json',report)
    except BaseException as error:
        inv.create_json(output/'FAILED.json',{**report,'error':repr(error)});raise
    print(output/'report.json')


if __name__=='__main__':main()
