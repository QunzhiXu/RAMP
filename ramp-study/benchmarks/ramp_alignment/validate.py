"""Audit evidence writer. Reads original results; never fits or modifies them."""
import argparse
import datetime
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

import compare
import inventory as inv
import safe_io as io
import study

CYTONORM_REPORT_SHA='975d55ab4de6b9196c90a29365e4a4d89bdbf752ecbf93676056c4d5237e1cd1'
FIRST_PRECHECK=inv.HERE/'validation/preflight_20261011/report.json'
FIRST_PRECHECK_SHA='8814c79b8275ce1714b57872b493e741c64dd172aeef0d11f318ef81117a27b6'
PRECHECK=inv.HERE/'validation/preflight_20261011_r2/report.json'
PRECHECK_SHA='fa19d319a1948ccf767b981459aa3a601094127237c46a03573227081bb5b773'


def retained_first_attempt():
    prefixes=('runs/six_fit_20261011','provenance/prefit_20261011',
              'provenance/thread_policy_before_fix_20261011',
              'validation/preflight_20261011',
              'validation/melanoma_input_diagnosis_20261011',
              'validation/melanoma_input_diagnosis_20261011_r2',
              'validation/melanoma_default_runtime_20261011')
    return {prefix+'/'+n:r for prefix in prefixes for n,r in io.files(inv.HERE/prefix).items()}


def runtime_fix_evidence(source):
    """Verify, do not suppress, the failed first parity attempt and exact diagnostic."""
    import ast
    import numpy as np
    import pandas as pd
    spec=inv.spec('melanoma_story')
    first=inv.HERE/'runs/six_fit_20261011/melanoma_story/result'
    diagnostic=inv.HERE/'validation/melanoma_default_runtime_20261011'
    raw_report=inv.HERE/'validation/melanoma_input_diagnosis_20261011_r2/report.json'
    reconstruction=io.read(raw_report)
    if reconstruction['raw_array_equal'] is not True or reconstruction['maximum_difference']!=0 or reconstruction['changed_values']!=0 or reconstruction['identity_equal'] is not True:
        raise ValueError('Raw reconstruction was not exact')
    for path,row in reconstruction['source_records'].items():io.check(path,row)
    old_code=io.read(FIRST_PRECHECK)['fitting_code']
    diagnostic_report=io.read(diagnostic/'report.json')
    if reconstruction['controller_unchanged']!=old_code or diagnostic_report['controller_unchanged']!=old_code:
        raise ValueError('Diagnostic did not use the original controller')
    io.payload(diagnostic/'result',diagnostic_report['outputs'])
    denv=io.read(diagnostic/'ENVIRONMENT.json')
    from worker import check_thread_policy
    check_thread_policy('melanoma_story',denv['threads'],denv['threadpools'])
    if denv['canonical']['sha256']!=inv.CANONICAL_SHA:
        raise ValueError('Default runtime diagnostic contract differs')
    diag=compare.compare_files(diagnostic/'result',spec,source)
    old=np.load(source/spec['references']['ramp_aligned.npy']['source'],allow_pickle=False)
    bad=np.load(first/'ramp_aligned.npy',allow_pickle=False)
    delta=np.abs(old-bad)
    baseline='kmeanslow_baseline.npy';coupling='coupling_audit.csv'
    for name in (baseline,coupling):io.check(first/name,spec['references'][name])
    p0=pd.read_csv(source/spec['references']['profile_audit.csv']['source'])
    p1=pd.read_csv(first/'profile_audit.csv')
    keys=['query_dataset','reference_dataset','unit_key','marker_index']
    if not p0[keys].equals(p1[keys]):raise ValueError('Profile identities differ')
    flips=p0.curve_profile_fitted!=p1.curve_profile_fitted
    scientific=('validate_input','independent_invariants','calculate','save_result')
    before=inv.HERE/'provenance/thread_policy_before_fix_20261011'
    def functions(path):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
    original=functions(before/'worker.py');current=functions(inv.HERE/'worker.py')
    if any(original[n]!=current[n] for n in scientific):raise ValueError('Numerical wrapper changed during runtime fix')
    for name in ('safe_io.py','inventory.py','compare.py'):io.check(inv.HERE/name,io.record(before/name))
    return {'first_batch_accepted_as_whole':False,
            'first_melanoma_maximum_difference':float(delta.max()),
            'first_melanoma_changed_values':int(np.count_nonzero(delta)),
            'changed_profile_fit_decisions':int(flips.sum()),
            'affected_profiles':p0.loc[flips,keys].to_dict('records'),
            'first_baseline_and_coupling_byte_identical':True,
            'input_reconstruction_report':io.record(raw_report),
            'raw_and_identity_exact':True,
            'default_runtime_diagnostic':diag,
            'diagnostic_environment':io.read(diagnostic/'ENVIRONMENT.json'),
            'unchanged_numerical_wrapper_functions':list(scientific),
            'mechanistic_limit':'Thread/CPU-cap environment identified; individual native operation responsible was not isolated. No tolerance or algorithm change.'}


def outside_files():
    """Include old run payloads; exclude environments/caches and this new module."""
    rows={}
    for folder,dirs,names in os.walk(inv.STUDY):
        here=Path(folder)
        dirs[:]=sorted(d for d in dirs if not d.startswith('.') and d not in ('__pycache__','cache','tmp','node_modules') and (here/d).resolve()!=inv.HERE)
        for name in sorted(names):
            if name.startswith('.') or name.endswith('.pyc'):continue
            p=here/name;key=p.relative_to(inv.STUDY).as_posix()
            rows[key]={'symlink':str(p.resolve())} if p.is_symlink() else io.record(p)
        for d in dirs[:]:
            p=here/d
            if p.is_symlink():rows[p.relative_to(inv.STUDY).as_posix()]={'symlink':str(p.resolve())};dirs.remove(d)
    return rows


def original_indexes():
    root=inv.STUDY.parent;counts={}
    for name in ('source_census.json','curated_sources.json'):
        doc=io.read(inv.STUDY/'inventory_20261010/evidence'/name);n=0
        for row in doc['files']:
            if row.get('sha256'):
                io.check(root/row['path'],row);n+=1
        counts[name]=n
    return counts


def source_contract_checks(source):
    """Check frozen metadata/units against original producer inputs without importing it."""
    import numpy as np
    import pandas as pd
    from worker import validate_input
    rows={}
    for c in inv.CASES:
        identity_sources={}
        s=inv.spec(c);reg=next(r for r in io.read(inv.SNAPSHOT/'registry.json')['cases'] if r['case']==c)
        md=pd.read_csv(source/s['inputs']['fit_metadata.csv.gz']['path'],low_memory=False)
        raw=np.load(source/s['inputs']['raw.npy']['path'],allow_pickle=False)
        validate_input(raw,md,s['markers'],s)
        evaluation=pd.read_csv(source/reg['files']['evaluation_metadata']['path'],low_memory=False)
        for col in ('dataset','unit_key'):
            if not md[col].astype(str).equals(evaluation[col].astype(str)):raise ValueError('Evaluation row identity differs')
        if 'cell_id' in md and not md.cell_id.astype(str).equals(evaluation.cell_id.astype(str)):raise ValueError('Cell ID order differs')
        if evaluation.cell_id.isna().any() or evaluation.cell_id.duplicated().any():raise ValueError('Invalid evaluation cell IDs')
        script=source/s['source_script']
        if c in inv.CASES[:4]:
            if inv.literal(script,'DATASETS')!=s['datasets'] or inv.literal(script,'MARKERS')!=s['markers']:raise ValueError('Source code panel/datasets differ')
        elif c=='hnscc_story':
            if inv.literal(script,'DATASETS')!=s['datasets']:raise ValueError('Source dataset order differs')
            identity_path=source/'figure_HNSCC/alignment/input/cell_metadata.csv.gz'
            identity_sources[str(identity_path.relative_to(source))]=io.record(identity_path)
            original=pd.read_csv(identity_path,low_memory=False)
            for key in ('dataset','unit_key'):
                if not original[key].astype(str).equals(md[key].astype(str)):raise ValueError('Original HNSCC fit identities differ')
            if not original.cell_uid.astype(str).equals(md.cell_id.astype(str)):raise ValueError('Original HNSCC row IDs differ')
            original_raw=source/'figure_HNSCC/alignment/input/raw_shared_markers.float64.npy'
            identity_sources[str(original_raw.relative_to(source))]=io.record(original_raw)
            if not np.array_equal(raw,np.load(original_raw,allow_pickle=False)):raise ValueError('Original HNSCC raw matrix differs')
        elif c=='melanoma_story':
            if inv.literal(script,'DATASET_ORDER')!=s['datasets'] or inv.literal(script,'CLEAN15')!=s['markers']:raise ValueError('Source clean15 contract differs')
            identity_path=source/'figure_melanoma/analysis/alignment/output/cell_metadata.csv.gz'
            identity_sources[str(identity_path.relative_to(source))]=io.record(identity_path)
            original=pd.read_csv(identity_path,low_memory=False)
            for key in ('dataset','unit_key'):
                if not original[key].astype(str).equals(md[key].astype(str)):raise ValueError('Original melanoma fit identities differ')
            if not original.cell_uid.astype(str).equals(md.cell_id.astype(str)):raise ValueError('Original melanoma row IDs differ')
        else:
            design=io.read(source/s['historical_manifest'])
            if design['datasets']!=s['datasets'] or design['markers']!=s['markers']:raise ValueError('Source lung design differs')
            identity_path=source/inv.LUNG/'inputs/CELL_METADATA.csv.gz'
            identity_sources[str(identity_path.relative_to(source))]=io.record(identity_path)
            original=pd.read_csv(identity_path,low_memory=False)
            for key in ('dataset','unit_key'):
                if not original[key].astype(str).equals(md[key].astype(str)):raise ValueError('Original lung fit identities differ')
            if not original.cell_key.astype(str).equals(md.cell_id.astype(str)):raise ValueError('Original lung row IDs differ')
            original_raw=source/inv.LUNG/'inputs/RAW_SHARED9.npy'
            identity_sources[str(original_raw.relative_to(source))]=io.record(original_raw)
            if not np.array_equal(raw,np.load(original_raw,allow_pickle=False)):raise ValueError('Original lung raw matrix differs')
        for path,row in identity_sources.items():io.check(source/path,row)
        rows[c]={'shape':s['shape'],'dtype':str(raw.dtype),'dataset_order':s['datasets'],'dataset_cells':s['dataset_cells'],
                 'dataset_units':s['dataset_units'],'cell_identity_order_verified':True,'family':s['family'],'pair_fit':s['pair_fit'],'identity_sources':identity_sources}
    return rows


def run_tests(out):
    with tempfile.TemporaryDirectory(prefix='ramp_alignment_tests_') as tmp:
        copy=Path(tmp).resolve()/'module'
        shutil.copytree(inv.HERE,copy,ignore=shutil.ignore_patterns('runs','validation','__pycache__','.DS_Store'))
        cmd=[sys.executable,'-B','-m','unittest','discover','-s',str(copy/'tests'),'-v']
        env=dict(os.environ);env.pop('PYTHONPATH',None);env['PYTHONDONTWRITEBYTECODE']='1';env.pop('LANG',None)
        with (out/'tests.stdout.log').open('x') as stdout,(out/'tests.stderr.log').open('x') as stderr:
            p=subprocess.run(cmd,cwd=copy,env=env,stdout=stdout,stderr=stderr)
        if p.returncode:raise ValueError('Tests failed; see logs')
        count=re.search(r'Ran (\d+) tests', (out/'tests.stderr.log').read_text())
        if count is None:raise ValueError('Missing test summary')
        return {'command':cmd,'count':int(count.group(1)),'returncode':p.returncode}


def current_environment(run,out,case):
    """Read-only probe in a fresh interpreter; import definitions, never fit."""
    script="import sys,json;sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2]);from ramp import ramp_ver2;import worker;print(json.dumps(worker.environment(sys.argv[3]),sort_keys=True))"
    cmd=[sys.executable,'-B','-c',script,str(inv.HERE),str(run/'runtime/vendor'),case]
    with (out/(case+'.environment.json')).open('x') as stdout,(out/(case+'.environment.stderr.log')).open('x') as stderr:
        p=subprocess.run(cmd,cwd=out,env=study.child_environment(out,case),stdout=stdout,stderr=stderr)
    if p.returncode:raise ValueError('Live environment probe failed')
    observed=io.read(out/(case+'.environment.json'))
    if observed!=io.read(run/'ENVIRONMENT_BEFORE.json'):raise ValueError('Current environment differs from real run')
    return {'command':cmd,'matches_recorded_environment':True,'record':io.record(out/(case+'.environment.json'))}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['preflight','verify']);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--bundles',type=Path,required=True);p.add_argument('--runs',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    source=a.source_root.resolve(strict=True);bundles=a.bundles.resolve(strict=True)
    if not a.output.resolve().is_relative_to(inv.HERE/'validation'):p.error('Use a new validation directory in this module')
    pre=None
    if a.mode=='verify':
        if io.record(PRECHECK)['sha256']!=PRECHECK_SHA:raise ValueError('Preflight anchor changed')
        pre=io.read(PRECHECK)
        if pre['status']!='passed':raise ValueError('Preflight failed')
        study.check_code(pre['fitting_code'])
    old=outside_files()
    if io.record(FIRST_PRECHECK)['sha256']!=FIRST_PRECHECK_SHA:raise ValueError('First preflight changed')
    if old!=io.read(FIRST_PRECHECK)['outside_files']:raise ValueError('Old study baseline changed since first preflight')
    retained=retained_first_attempt()
    if pre and retained!=pre['retained_first_attempt']:raise ValueError('First attempt or diagnostic evidence changed')
    if pre and old!=pre['outside_files']:raise ValueError('Previously accepted study files changed')
    out=io.reserve(a.output,study.protected(bundles,*([a.runs] if a.runs else [])))
    try:
        report={'mode':a.mode,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'six frozen-input RAMP alignments; BM reference only',
                'source_verification':inv.verify_source(source),'original_indexes':original_indexes(),'contracts':source_contract_checks(source),
                'fitting_code':study.code_records(),'outside_files':old,'tests':run_tests(out),
                'retained_first_attempt':retained,'first_preflight_sha256':FIRST_PRECHECK_SHA}
        if pre and report['contracts']!=pre['contracts']:raise ValueError('Original input identity evidence changed')
        cyto=inv.STUDY/'methods/cytonorm/validation/seven_execution_20261011/report.json'
        if io.record(cyto)['sha256']!=CYTONORM_REPORT_SHA:raise ValueError('Accepted CytoNorm report changed')
        report['cytonorm_report_sha256']=CYTONORM_REPORT_SHA
        report['bundles']={c:io.record(bundles/c/'MANIFEST.json') for c in inv.CASES}
        for c in inv.CASES:
            if study.verify_bundle(bundles/c)['case']!=c:raise ValueError('Swapped bundle')
        bm=inv.STUDY/'benchmarks/bone_marrow/runs/rebuild_20261010/alignment/data'
        report['BM_accepted_reference']=compare.compare_files(bm,inv.spec('bone_marrow'),source)
        if a.mode=='verify':
            if a.runs is None:p.error('--runs required')
            runs=a.runs.resolve(strict=True);batch=io.read(runs/'BATCH_MANIFEST.json')
            report['runtime_fix_evidence']=runtime_fix_evidence(source)
            if io.read(runs/'COMPLETE.json')!={'status':'complete','manifest_sha256':io.record(runs/'BATCH_MANIFEST.json')['sha256']}:raise ValueError('Incomplete batch')
            if batch['cases']!=list(inv.EXECUTABLE) or set(batch['runs'])!=set(inv.EXECUTABLE):raise ValueError('Batch cases differ')
            if batch['code_records']!=pre['fitting_code']:raise ValueError('Batch code changed')
            request=io.read(runs/'BATCH_REQUEST.json')
            if request!={'cases':list(inv.EXECUTABLE),'code_records':pre['fitting_code'],'bundles':{c:pre['bundles'][c] for c in inv.EXECUTABLE}}:raise ValueError('Batch request differs')
            report['cases']={}
            report['current_environment']={}
            for c in inv.EXECUTABLE:
                io.check(runs/c/'RUN_MANIFEST.json',batch['runs'][c])
                report['cases'][c]=compare.compare(runs/c,bundles/c,source)
                report['current_environment'][c]=current_environment(runs/c,out,c)
                if io.read(runs/c/'REQUEST.json')['code_records']!=pre['fitting_code']:raise ValueError('Mixed controller versions')
            report['identical_six_request_code_records']=all(io.read(runs/c/'REQUEST.json')['code_records']==pre['fitting_code'] for c in inv.EXECUTABLE)
            report['moved_bundle_verification']={}
            with tempfile.TemporaryDirectory(prefix='ramp moved bundles ') as tmp:
                for c in inv.EXECUTABLE:
                    moved=Path(tmp)/c;shutil.copytree(bundles/c,moved)
                    report['moved_bundle_verification'][c]=study.verify_run(runs/c,moved)
            report['batch_manifest']=io.record(runs/'BATCH_MANIFEST.json')
        if outside_files()!=old:raise ValueError('Earlier files changed during validation')
        if retained_first_attempt()!=retained:raise ValueError('Retained first attempt changed during validation')
        inv.verify_source(source);original_indexes();study.check_code(report['fitting_code'])
        if source_contract_checks(source)!=report['contracts']:raise ValueError('Original identities changed during validation')
        report['current_code']={n:io.record(inv.HERE/n) for n in (*study.CODE,'compare.py','validate.py','diagnose_melanoma.py','tests/test_workflow.py','README.md','AUDIT_PROMPT.md','THREAD_POLICY_FIX_20261011.md')}
        report['outside_unchanged_files']=len(old);report['status']='passed'
        io.write(out/'report.json',report)
    except BaseException as e:io.failed(out,e);raise
    print(out/'report.json')


if __name__=='__main__':main()
