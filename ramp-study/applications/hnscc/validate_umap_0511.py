"""Read-only recovery of one completed fit whose first comparison used a wrong ID name.

The original diagnostic, FAILED record and every output remain untouched. Only
the pure table comparator is substituted while inspecting the archived payload.
This module has no fitting or scientific parameter selection path.
"""
import argparse
import gzip
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from common import HERE, read, write, sha, record, files, links, reserve, resource_state

REQUEST_SHA = 'c6e64942305b7c39b9aa4a47e53e7e592ee8ea4f17f5db7bbde23783ff4bffdc'
FAILED_SHA = 'ac90907d1e226a97015a4af8c575d1c903b059f4281eb40aa181e505dbe4bee4'
EXECUTION_CODE = {
    'common.py': {'bytes':4825,'sha256':'e8ae88167cd0f5ad86116b082171ec445975492e453f561733271d514b496595'},
    'umap_version_diagnostic.py': {'bytes':16477,'sha256':'831fde08f52eaa836581bd830c4b675457c6d96929ad34456cc1cbcefac503f2'}}


def compare_coordinates(new, old):
    import numpy as np
    import pandas as pd
    a = pd.read_csv(new, low_memory=False, float_precision='round_trip')
    b = pd.read_csv(old, low_memory=False, float_precision='round_trip')
    coord = ['UMAP1','UMAP2']
    if list(a.columns) != list(b.columns) or 'cell_uid' not in a or not set(coord) <= set(a) or len(a) != 94488 or len(b) != 94488:
        raise ValueError('Coordinate schema or cohort differs')
    if a.cell_uid.isna().any() or a.cell_uid.duplicated().any() or not a.drop(columns=coord).equals(b.drop(columns=coord)):
        raise ValueError('Coordinate identity, metadata or row order differs')
    if not np.isfinite(a[coord].to_numpy()).all() or not np.isfinite(b[coord].to_numpy()).all():
        raise ValueError('Nonfinite coordinates')
    columns = {c: {'exact':bool(np.array_equal(a[c].to_numpy(), b[c].to_numpy())),
                   'changed_values':int(np.count_nonzero(a[c].to_numpy() != b[c].to_numpy())),
                   'max_abs_difference':float(np.max(np.abs(a[c].to_numpy()-b[c].to_numpy())))} for c in coord}
    return {'cells':len(a), 'metadata_and_row_order_exact':True, 'columns':columns,
            'coordinates_exact':all(v['exact'] for v in columns.values()),
            'gzip_decompressed_equal':gzip.open(new,'rb').read() == gzip.open(old,'rb').read(),
            'byte_identical':sha(new) == sha(old)}


def inspect(run):
    run=Path(run).resolve()
    # This exception covers one fixed reporting failure, not arbitrary failed runs.
    if sha(run/'REQUEST.json') != REQUEST_SHA or sha(run/'FAILED.json') != FAILED_SHA or (run/'COMPLETE.json').exists():
        raise ValueError('Unrecognized diagnostic or reporting failure')
    if files(run/'controller') != EXECUTION_CODE or read(run/'REQUEST.json')['controller'] != EXECUTION_CODE:
        raise ValueError('Unrecognized executed controller')
    spec=importlib.util.spec_from_file_location('executed_umap_diagnostic',run/'controller/umap_version_diagnostic.py')
    executed=importlib.util.module_from_spec(spec); spec.loader.exec_module(executed)
    # Preserve every check, threshold and scientific operation. The table ID is
    # cell_uid in the actual frozen schema, not cell_id in the first comparator.
    executed.comparison=compare_coordinates
    result=executed.check_payload(run)
    return {'result':result, 'execution_request_sha256':REQUEST_SHA,
            'retained_reporting_failure_sha256':FAILED_SHA,
            'previous_run_manifest_sha256':executed.PRIOR_MANIFEST_SHA,
            'base_lock_sha256':executed.LOCK_SHA,
            'previous_comparison_sha256':executed.PRIOR_REPORT_SHA,
            'execution_manifest':{'files':files(run),'links':links(run)}}


def verification_code():
    return {name:record(HERE/name) for name in ('validate_umap_0511.py','common.py','umap_version_diagnostic.py')}


def check_report(run, folder):
    folder=Path(folder)
    if (folder/'FAILED.json').exists() or read(folder/'COMPLETE.json') != {'status':'complete','manifest_sha256':sha(folder/'MANIFEST.json')}:
        raise ValueError('Unsealed validation evidence')
    actual=files(folder); actual.pop('MANIFEST.json'); actual.pop('COMPLETE.json')
    if links(folder) or read(folder/'MANIFEST.json') != actual:
        raise ValueError('Validation evidence payload differs')
    report=read(folder/'report.json')
    observed=inspect(run)
    if report['verification_code'] != verification_code() or report['inspection'] != {k:v for k,v in observed.items() if k!='execution_manifest'}:
        raise ValueError('Validation report disagrees with actual execution')
    if read(folder/'EXECUTION_MANIFEST.json') != observed['execution_manifest']:
        raise ValueError('Preserved execution payload changed')
    expected_status='exact_coordinates_pending_external_audit' if observed['result']['historical_comparison']['coordinates_exact'] else 'completed_with_differences_pending_external_audit'
    if report['status'] != expected_status:
        raise ValueError('Validation report status differs')
    return report['inspection']


def validate(run, output):
    run=Path(run).resolve()
    before=inspect(run)
    req=read(run/'REQUEST.json')
    protected=[run,Path(req['module_root'])/'frozen_20261011',Path(req['module_root'])/'runs/rebuild_20261011']
    protected += [Path(v['path']) for v in req['bindings'].values()]
    out=reserve(output,protected)
    try:
        write(out/'REQUEST.json',{'run':str(run),'verification_code':verification_code(),'execution_request_sha256':REQUEST_SHA})
        (out/'controller').mkdir()
        for name in verification_code(): shutil.copyfile(HERE/name,out/'controller'/name)
        write(out/'EXECUTION_MANIFEST.json',before['execution_manifest'])
        tests=[]
        with tempfile.TemporaryDirectory(prefix='hnscc_0511_checks_',dir='/private/tmp') as temp:
            tmp=Path(temp).resolve(); controller=tmp/'controller with spaces'; controller.mkdir()
            for p in HERE.glob('*.py'): shutil.copyfile(p,controller/p.name)
            shutil.copytree(HERE/'tests',controller/'tests',ignore=shutil.ignore_patterns('__pycache__'))
            shutil.copytree(HERE/'frozen_20261011',controller/'frozen_20261011')
            for label,python in [('system',sys.executable),('bm_venv',req['python'])]:
                env=os.environ.copy(); env.update(PYTHONDONTWRITEBYTECODE='1',TMPDIR='/private/tmp')
                cmd=[python,'-B','-m','unittest','discover','-s','tests','-v']
                p=subprocess.run(cmd,cwd=controller,env=env,text=True,capture_output=True)
                with (out/(label+'_tests.stdout')).open('x') as s:s.write(p.stdout)
                with (out/(label+'_tests.stderr')).open('x') as s:s.write(p.stderr)
                if p.returncode:raise ValueError('Temporary tests failed: '+label)
                tests.append({'label':label,'command':cmd,'returncode':p.returncode,'TMPDIR':'/private/tmp'})
            cmd=[sys.executable,'-B',str(controller/'validate_umap_0511.py'),'inspect','--run',str(run)]
            p=subprocess.run(cmd,cwd=controller,text=True,capture_output=True)
            with (out/'moved_controller.stdout').open('x') as s:s.write(p.stdout)
            with (out/'moved_controller.stderr').open('x') as s:s.write(p.stderr)
            if p.returncode or json.loads(p.stdout) != before: raise ValueError('Moved controller inspection differs')
            copied_bytes=sum(p.stat().st_size for p in tmp.rglob('*') if p.is_file())
        # Old application output and original source indexes are checked in place.
        from study import verify_run
        from prepare import verify as verify_lock
        from baseline import check as check_prior
        from validate import original_indexes
        lockdir=Path(req['bindings']['lock']['path']).parent
        priorrun=Path(req['bindings']['prior_manifest']['path']).parent
        previous=verify_run(lockdir,priorrun); verify_lock(lockdir,source=True)
        external=check_prior(HERE/'validation/preflight_20261011'); original=original_indexes()
        after=inspect(run)
        if before != after: raise ValueError('Diagnostic changed during validation')
        result=before['result']
        report={'status':'exact_coordinates_pending_external_audit' if result['historical_comparison']['coordinates_exact'] else 'completed_with_differences_pending_external_audit',
                'inspection':{k:v for k,v in before.items() if k!='execution_manifest'},
                'verification_code':verification_code(),'tests':tests,'moved_controller_verified':True,
                'temporary_copied_bytes':copied_bytes,'previous_run':previous,'external_baseline':external,
                'original_indexes':original,'resources':resource_state(),
                'scope':'one additional 0.5.11 UMAP fit only; no downstream rerun and no final-figure QA',
                'retained_initial_test_issue':'Two path-safety fixtures used unresolved macOS /var TMPDIR; physical TMPDIR=/private/tmp passes. This does not alter production protection or the fit.',
                'retained_failure':'Fit completed; initial postflight comparison used cell_id rather than actual cell_uid. Only read-only comparison is recovered here; old FAILED and all files remain unchanged.'}
        write(out/'report.json',report)
        write(out/'MANIFEST.json',files(out))
        write(out/'COMPLETE.json',{'status':'complete','manifest_sha256':sha(out/'MANIFEST.json')})
        return report
    except BaseException as exc:
        write(out/'FAILED.json',{'error':repr(exc)});raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['inspect','validate','check-report'])
    p.add_argument('--run',required=True);p.add_argument('--out');p.add_argument('--report')
    a=p.parse_args()
    result=inspect(a.run) if a.mode=='inspect' else check_report(a.run,a.report) if a.mode=='check-report' else validate(a.run,a.out)
    print(json.dumps(result,indent=2))
