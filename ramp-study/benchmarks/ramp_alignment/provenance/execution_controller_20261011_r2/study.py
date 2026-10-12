"""Frozen-input RAMP preparation/execution. BM is a read-only accepted reference."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

import inventory as inv
import safe_io as io

CODE=('safe_io.py','inventory.py','study.py','worker.py')


def code_records():
    return {n:io.record(inv.HERE/n) for n in CODE}


def check_code(rows):
    if rows!=code_records():raise ValueError('Execution code changed')


def child_environment(run,case=None):
    from worker import thread_values
    env=dict(os.environ)
    for k in ('PYTHONPATH','PYTHONHOME'):env.pop(k,None)
    env['PYTHONDONTWRITEBYTECODE']='1'
    for k,v in thread_values(case).items():
        if v is None:env.pop(k,None)
        else:env[k]=v
    for k,part in {'TMPDIR':'tmp','TMP':'tmp','TEMP':'tmp','MPLCONFIGDIR':'cache/mpl','NUMBA_CACHE_DIR':'cache/numba','XDG_CACHE_HOME':'cache/xdg','JOBLIB_TEMP_FOLDER':'tmp'}.items():env[k]=str(Path(run)/part)
    return env


def plan(case):
    spec=inv.spec(case);c=inv.catalog();rows={}
    for n,r in spec['inputs'].items():rows['input/'+n]={k:r[k] for k in ('bytes','sha256')}
    for n in ('__init__.py','ramp.py','ramp_ver2.py','run_ramp.py'):
        key='sources/'+inv.LUNG+'/vendor/ramp/'+n if case=='lung_story' and n!='run_ramp.py' else 'vendor/ramp/'+n
        rows['vendor/ramp/'+n]={k:c['copies'][key][k] for k in ('bytes','sha256')}
    if case=='lung_story':rows['helpers/frozen_maps.py']={k:c['copies']['helpers/frozen_maps.py'][k] for k in ('bytes','sha256')}
    return {'case':case,'catalog_sha256':inv.CATALOG_SHA,'files':rows}


def protected(*extra):
    c=inv.catalog()
    return (inv.LEGACY,Path(c['source_root']),inv.SNAPSHOT,inv.STUDY/'vendor',
            inv.STUDY/'benchmarks/bone_marrow',inv.STUDY/'benchmarks/integration_inputs',
            inv.STUDY/'methods',*[inv.HERE/n for n in CODE],*extra)


def prepare(case,inputs_root,output):
    inputs_root=Path(inputs_root).resolve(strict=True)
    spec=inv.spec(case);expected=plan(case)
    sources={}
    for n,row in expected['files'].items():
        if n.startswith('input/'):src=inputs_root/case/Path(n).name
        elif case=='lung_story' and n.startswith('vendor/') and not n.endswith('run_ramp.py'):src=inv.SNAPSHOT/'sources'/inv.LUNG/n
        else:src=inv.SNAPSHOT/n
        io.check(src,row);sources[n]=src
    codes=code_records();out=io.reserve(output,protected(inputs_root))
    try:
        for n,row in expected['files'].items():io.copy(sources[n],out/n,row)
        for n,row in expected['files'].items():io.check(sources[n],row)
        check_code(codes);io.write(out/'MANIFEST.json',expected)
        io.payload(out,expected['files'],('MANIFEST.json',))
        io.write(out/'COMPLETE.json',{'status':'complete','manifest_sha256':io.record(out/'MANIFEST.json')['sha256']})
    except BaseException as e:io.failed(out,e);raise
    return out


def verify_bundle(bundle):
    bundle=Path(bundle).resolve(strict=True)
    manifest=io.read(io.child(bundle,'MANIFEST.json'));expected=plan(manifest['case'])
    if manifest!=expected:raise ValueError('Bundle not bound to accepted catalog')
    done=io.read(io.child(bundle,'COMPLETE.json'))
    if done!={'status':'complete','manifest_sha256':io.record(bundle/'MANIFEST.json')['sha256']}:raise ValueError('Incomplete bundle')
    io.payload(bundle,expected['files'],('MANIFEST.json','COMPLETE.json'))
    return inv.spec(manifest['case'])


def check_runtime(run,spec):
    io.payload(Path(run)/'runtime',plan(spec['case'])['files'])


def validate_outputs(run,spec):
    import numpy as np
    import pandas as pd
    run=Path(run);result=io.read(run/'WORKER_RESULT.json');out=run/'result'
    if result['case']!=spec['case'] or result['family']!=spec['family'] or result['pair_fit']!=spec['pair_fit']:raise ValueError('Worker result identity differs')
    if result['historical_results_read'] is not False or result['metadata_columns_read_for_fitting']!=['dataset','unit_key']:raise ValueError('Fitting scope differs')
    if result['canonical_rank_inversions']!=0:raise ValueError('Rank inversions')
    invs=result['independent_invariants']
    if invs['strict_inversions']!=0 or invs['split_raw_ties']!=0 or invs['maximum_displacement_float64']>0.75+1e-10 or invs['tolerance']!=1e-10:raise ValueError('Invariant failure')
    names=set(spec['matrices'])|set(spec['audits'])|set(spec['extras'])
    if set(result['outputs'])!=names:raise ValueError('Output set differs')
    io.payload(out,result['outputs'])
    for n,(_,dtype) in spec['matrices'].items():
        x=np.load(out/n,allow_pickle=False)
        if list(x.shape)!=spec['shape'] or str(x.dtype)!=dtype or not np.isfinite(x).all():raise ValueError('Output matrix shape/dtype/finiteness differs')
    coupling=pd.read_csv(out/next(n for n,a in spec['audits'].items() if a=='coupling_audit'))
    if set(coupling.family)!={spec['family']}:raise ValueError('Coupling family differs')
    for k,v in spec['pair_fit'].items():
        if set(coupling['pair_fit_'+k])!={v}:raise ValueError('Pair fit policy differs')
    audit=pd.read_csv(out/next(n for n,a in spec['audits'].items() if a=='algorithm_audit'),dtype=str)
    a=dict(zip(audit.audit_key,audit.audit_value))
    for k,v in {'algorithm_version':'2.3.0','graph_k_nonself':'15','cell_marker_values_over_maximum':'0','pair_fit_maximum':str(spec['pair_fit']['maximum'])}.items():
        if a.get(k)!=v:raise ValueError('Algorithm audit differs: '+k)
    for name in spec['extras']:
        with np.load(out/name,allow_pickle=False) as z:
            if not z.files or any(not np.isfinite(z[n]).all() for n in z.files):raise ValueError('Invalid map/profile archive')
    before=io.read(run/'ENVIRONMENT_BEFORE.json');after=io.read(run/'ENVIRONMENT_AFTER.json')
    if before!=after:raise ValueError('Environment changed')
    from worker import VERSIONS,check_thread_policy
    if before['versions']!=VERSIONS or before['python']!='3.14.1':raise ValueError('Unaccepted runtime')
    check_thread_policy(spec['case'],before['thread_limits'],before['threadpools'])
    return result


def fit(bundle,output):
    from worker import thread_values
    bundle=Path(bundle).resolve(strict=True);spec=verify_bundle(bundle)
    if spec['reference_only']:raise ValueError('BM already accepted; reference-only in this module')
    codes=code_records();python=Path(sys.executable).resolve(strict=True)
    b_record=io.record(bundle/'MANIFEST.json');py_record=io.record(python)
    out=io.reserve(output,protected(bundle,python))
    try:
        request={'case':spec['case'],'run':str(out),'bundle':str(bundle),'bundle_manifest':b_record,
                 'catalog_sha256':inv.CATALOG_SHA,'code_records':codes,'python':py_record,'python_path':str(python),
                 'thread_environment':thread_values(spec['case'])}
        io.write(out/'REQUEST.json',request)
        for n,row in plan(spec['case'])['files'].items():io.copy(bundle/n,out/'runtime'/n,row)
        for part in ('tmp','cache/mpl','cache/numba','cache/xdg'):(out/part).mkdir(parents=True,exist_ok=True)
        command=[str(python),'-B',str(inv.HERE/'worker.py'),'--run',str(out)]
        io.write(out/'COMMAND.json',{'argv':command,'cwd':str(out)})
        with (out/'stdout.log').open('x') as stdout,(out/'stderr.log').open('x') as stderr:
            process=subprocess.run(command,cwd=out,env=child_environment(out,spec['case']),stdout=stdout,stderr=stderr)
        if process.returncode:raise RuntimeError(f'RAMP worker failed ({process.returncode}); logs retained')
        result=validate_outputs(out,spec);verify_bundle(bundle);check_runtime(out,spec);check_code(codes)
        io.check(python,py_record);io.check(bundle/'MANIFEST.json',b_record)
        io.write(out/'RUN_MANIFEST.json',{'case':spec['case'],'catalog_sha256':inv.CATALOG_SHA,
                 'bundle_manifest':b_record,'code_records':codes,'files':io.files(out)})
        # All potentially failing checks occur before the outer completion marker.
        verify_run(out,bundle,require_complete=False)
        io.write(out/'COMPLETE.json',{'status':'complete','manifest_sha256':io.record(out/'RUN_MANIFEST.json')['sha256']})
    except BaseException as e:io.failed(out,e);raise
    return out


def verify_run(run,bundle,require_complete=True):
    from worker import thread_values
    run=Path(run).resolve(strict=True);bundle=Path(bundle).resolve(strict=True);spec=verify_bundle(bundle)
    if spec['reference_only']:raise ValueError('BM reference only')
    m=io.read(run/'RUN_MANIFEST.json');req=io.read(run/'REQUEST.json')
    if m['case']!=spec['case'] or req['case']!=spec['case'] or m['catalog_sha256']!=inv.CATALOG_SHA or req['catalog_sha256']!=inv.CATALOG_SHA:raise ValueError('Run identity differs')
    if req['run']!=str(run):raise ValueError('Run directory relocation not supported; bundles may move')
    check_code(req['code_records'])
    if req['thread_environment']!=thread_values(spec['case']):raise ValueError('Run thread policy differs')
    if m['code_records']!=req['code_records']:raise ValueError('Manifest code differs')
    io.check(bundle/'MANIFEST.json',req['bundle_manifest'])
    if m['bundle_manifest']!=req['bundle_manifest']:raise ValueError('Bundle manifest differs')
    io.check(req['python_path'],req['python'])
    command=io.read(run/'COMMAND.json')
    if command!={'argv':[req['python_path'],'-B',str(inv.HERE/'worker.py'),'--run',str(run)],'cwd':str(run)}:
        # Moving the controller is supported; authenticate its old basename and pinned bytes.
        argv=command.get('argv',[])
        if len(argv)!=5 or argv[:2]!=[req['python_path'],'-B'] or Path(argv[2]).name!='worker.py' or argv[3:]!=['--run',str(run)] or command.get('cwd')!=str(run):raise ValueError('Unexpected command')
    expected_runtime=plan(spec['case'])['files']
    for n,r in expected_runtime.items():
        if m['files'].get('runtime/'+n)!=r:raise ValueError('Runtime not bound to bundle')
    extras=['RUN_MANIFEST.json']
    if require_complete:
        if io.read(run/'COMPLETE.json')!={'status':'complete','manifest_sha256':io.record(run/'RUN_MANIFEST.json')['sha256']}:raise ValueError('Incomplete run')
        extras.append('COMPLETE.json')
    io.payload(run,m['files'],extras);check_runtime(run,spec)
    result=validate_outputs(run,spec)
    return {'case':spec['case'],'files':len(m['files']),'outputs':len(result['outputs']),'shape':spec['shape']}


def batch(bundles,output):
    bundles=Path(bundles).resolve(strict=True);plans={}
    for c in inv.EXECUTABLE:
        s=verify_bundle(bundles/c)
        if s['case']!=c:raise ValueError('Swapped case bundle')
        plans[c]=io.record(bundles/c/'MANIFEST.json')
    codes=code_records();out=io.reserve(output,protected(bundles,Path(sys.executable).resolve()))
    try:
        io.write(out/'BATCH_REQUEST.json',{'cases':list(inv.EXECUTABLE),'code_records':codes,'bundles':plans})
        runs={}
        for c in inv.EXECUTABLE:
            check_code(codes);io.check(bundles/c/'MANIFEST.json',plans[c]);print('START '+c,flush=True)
            fit(bundles/c,out/c);runs[c]=io.record(out/c/'RUN_MANIFEST.json');print('FINISHED '+c,flush=True)
        for c in inv.EXECUTABLE:
            verify_run(out/c,bundles/c);io.check(out/c/'RUN_MANIFEST.json',runs[c])
            if io.read(out/c/'REQUEST.json')['code_records']!=codes:raise ValueError('Mixed controller versions')
        check_code(codes)
        io.write(out/'BATCH_MANIFEST.json',{'cases':list(inv.EXECUTABLE),'code_records':codes,'runs':runs})
        io.write(out/'COMPLETE.json',{'status':'complete','manifest_sha256':io.record(out/'BATCH_MANIFEST.json')['sha256']})
    except BaseException as e:io.failed(out,e);raise
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    q=sub.add_parser('prepare');q.add_argument('--case',choices=inv.CASES,required=True);q.add_argument('--inputs-root',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    for mode in ('fit','verify','verify-run'):
        q=sub.add_parser(mode);q.add_argument('--bundle',type=Path,required=True)
        if mode=='fit':q.add_argument('--output',type=Path,required=True)
        if mode=='verify-run':q.add_argument('--run',type=Path,required=True)
    q=sub.add_parser('batch');q.add_argument('--bundles',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.mode=='prepare':r=prepare(a.case,a.inputs_root,a.output)
    elif a.mode=='fit':r=fit(a.bundle,a.output)
    elif a.mode=='batch':r=batch(a.bundles,a.output)
    elif a.mode=='verify-run':r=verify_run(a.run,a.bundle)
    else:r={'case':verify_bundle(a.bundle)['case']}
    print(r)


if __name__=='__main__':main()
