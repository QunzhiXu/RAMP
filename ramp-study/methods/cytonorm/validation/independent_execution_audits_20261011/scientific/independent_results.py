"""Independent read-only real-output audit. Does not import fitting code."""
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

ROOT=Path('/Users/qxu6/Documents/New project')
SRC=ROOT/'final_xu';MOD=ROOT/'ramp-study/methods/cytonorm'
OUT=Path(__file__).resolve().parent
SNAP=MOD/'frozen_20261011';BATCH=MOD/'runs/seven_fit_20261011'
MODEL_AUDIT=Path('/private/tmp/cytonorm_bm_model_audit_zgnnxJ')
ID=['cell_id','dataset','unit_key']
CODE=('execution.py','check_environment.R','probe_runtime.py','cytonorm_io.py','inventory.py','study.py')
REFINED={'compare_execution.py','validate_execution.py','tests/test_execution_comparison.py','STUDY_EXECUTION.md','AUDIT_EXECUTION.md'}
def read(p):return json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def rec(p):return {'bytes':p.stat().st_size,'sha256':sha(p)}
def check(p,r):assert rec(p)=={k:r[k] for k in ('bytes','sha256')},str(p)
def table(p):return pd.read_csv(p,dtype={n:str for n in ID},keep_default_na=False,low_memory=False,float_precision='round_trip')
def complete(root,name):
    assert read(root/'COMPLETE.json')=={'status':'complete','manifest_sha256':sha(root/name)}
    return read(root/name)
def write(name,value):
    with (OUT/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def source_checks():
    assert sha(SNAP/'catalog.json')=='532c2950133dce0d6435f7debbaed059a5a7a806bd5be396945463458814dc12'
    c=complete(SNAP,'catalog.json')
    for r in c['files']:
        check(SRC/r['source'],r)
        if 'copy' in r:check(SNAP/r['copy'],r)
    for row in c['links']:
        v=read(SRC/row['manifest'])
        for key in row['fields']:v=v[key]
        assert v==sha(SRC/row['target'])
    return c

if sys.argv[1]=='results':
    c=source_checks()
    phase=read(MOD/'validation/handoff_20261011_r2/report.json')
    assert sha(MOD/'validation/handoff_20261011_r2/report.json')=='c91c922cf0c4e08b950ca9f1bf84cd730190b3fd6db8744099c17cb5db591a36'
    pins={b['case']:b['manifest_sha256'] for b in phase['bundles']}
    code={n:rec(MOD/n) for n in CODE}
    bm=complete(BATCH,'BATCH_MANIFEST.json');breq=read(BATCH/'BATCH_REQUEST.json')
    cases=[s['case'] for s in c['branches']]
    assert bm['status']=='complete' and bm['history_used_by_fit'] is False
    assert [r['case'] for r in bm['cases']]==breq['cases']==cases
    assert bm['code_records']==breq['code_records']==code
    with (SNAP/'environment/runtime_lock.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    versions={r['package']:r['version'] for r in rows};assert len(versions)==69
    runtime={'lock':rec(SNAP/'environment/runtime_lock.csv'),'versions':versions}
    model_report=read(MODEL_AUDIT/'all_models.json');assert model_report['status']=='passed'
    prototype=(MODEL_AUDIT/'strict_model_compare.R').read_bytes()
    assert (MOD/'compare_models.R').read_bytes().startswith(prototype)
    assert sha(MOD/'compare_models.R')=='9a66fc7be49e586466f616351a32631ef1f6dceccf4d35d87302d8ef3e10b0b5'
    results={}
    for s,row in zip(c['branches'],bm['cases'],strict=True):
        case=s['case'];run=BATCH/case;bundle=MOD/'runs'/(case+'_inputs_20261011')
        m=complete(run,'RUN_MANIFEST.json');req=read(run/'REQUEST.json');bund=complete(bundle,'MANIFEST.json')
        assert sha(bundle/'MANIFEST.json')==pins[case]
        check(bundle/'MANIFEST.json',breq['bundle_manifests'][case]);check(run/'RUN_MANIFEST.json',row['manifest'])
        assert req['case']==case and req['history_used_by_fit'] is False and m['history_used_by_fit'] is False
        assert req['code_records']==code and req['runtime_contract']==runtime
        assert req['catalog_sha256']==sha(SNAP/'catalog.json')
        check(Path(req['library'])/'runtime_lock.csv',runtime['lock']);check(Path(req['rscript']),req['rscript_record'])
        actual={p.relative_to(run).as_posix() for p in run.rglob('*') if p.is_file() and not p.is_symlink()}
        links={p.relative_to(run).as_posix():str(p.resolve(strict=True)) for p in run.rglob('*') if p.is_symlink()}
        assert actual==set(m['files'])|{'RUN_MANIFEST.json','COMPLETE.json'}
        assert links==m['links']=={'runtime/tree/algorithm/CytoNorm/.Rlib':req['library']}
        for name,r in m['files'].items():check(run/name,r)
        for f in bund['files']:
            check(bundle/f['path'],f)
            if f['path'].startswith('adapter/'):target=run/'runtime/tree/algorithm/CytoNorm'/Path(f['path']).relative_to('adapter')
            elif f['path'].startswith('inputs/'):target=run/'runtime/tree'/f['source']
            elif f['path']=='environment/runtime_lock.csv':target=run/'runtime/environment/runtime_lock.csv'
            else:target=run/'runtime/tree'/s['base']/f['path']
            check(target,f);assert target.read_bytes()==(bundle/f['path']).read_bytes()
        assert (run/'environment_before.stdout.log').read_bytes()==(run/'environment_after.stdout.log').read_bytes()
        env=read(run/'environment_before.stdout.log')
        assert env['runtime_versions']==versions and env['runtime_lock_sha256']==runtime['lock']['sha256'] and env['R']=='4.3.2'
        assert env['environment']=={'LC_ALL':'C.UTF-8','LANG':'C.UTF-8'}
        assert env['locale']=={k:('C' if k=='LC_NUMERIC' else 'C.UTF-8') for k in ('LC_COLLATE','LC_CTYPE','LC_MONETARY','LC_NUMERIC','LC_TIME','LC_MESSAGES')}
        result=run/'runtime/tree/algorithm_test/cytonorm_execution'/case/'result' if s['kind']=='technical' else run/'runtime/tree'/s['base']/'methods/cytonorm'
        old=SRC/s['result_dir'];commands=read(run/'COMMANDS.json')
        assert commands['environment']==[req['rscript'],'--vanilla',str(Path(req['controller_root'])/'check_environment.R'),str(run/'runtime/tree/algorithm/CytoNorm'),str(run/'runtime/environment/runtime_lock.csv')]
        if s['kind']=='technical':
            expected_cmd=[req['rscript'],'--vanilla',str(run/'runtime/tree/algorithm/CytoNorm/run_cytonorm.R'),'--input',str(run/'runtime/tree'/s['input']),'--markers',str(run/'runtime/tree'/s['marker_file']),'--config',str(run/'runtime_config.json'),'--output',str(result)]
        else:
            here=run/'runtime/tree'/s['base'];expected_cmd=[req['rscript'],'--vanilla',str(here/'scripts/run_r.R'),str(here),'cytonorm']
        assert commands['correction']==expected_cmd
        # Every final file is either byte exact or one explicitly checked field.
        assert {p.name for p in result.iterdir()}=={p.name for p in old.iterdir()}
        exceptions={'model.rds','sessionInfo.txt','COMPLETE.json'}|({'warnings.txt','resolved_config.json'} if s['kind']=='technical' else {'R_COMPLETE.json'})
        equality={p.name:p.read_bytes()==(old/p.name).read_bytes() for p in result.iterdir()}
        assert all(equal for name,equal in equality.items() if name not in exceptions)
        inp=table(bundle/'inputs'/s['input']);newtable=table(result/'corrected_expression.csv.gz');oldtable=table(old/'corrected_expression.csv.gz')
        assert list(newtable)==['cell_id',*s['markers']] and newtable.cell_id.equals(inp.cell_id)
        x=newtable.iloc[:,1:].to_numpy(float);assert np.isfinite(x).all() and np.array_equal(x,oldtable.iloc[:,1:].to_numpy(float))
        assert inp[ID].equals(table(SRC/s['canonical']['evaluation_metadata'])[ID])
        assert table(result/'row_metadata.csv.gz').equals(inp[ID])
        controls=table(result/'aggregate_cells.csv.gz');clusters=table(result/'cell_clusters.csv.gz')
        assert controls.equals(table(old/'aggregate_cells.csv.gz')) and controls.cell_id.is_unique
        assert controls.equals(inp.set_index('cell_id').loc[controls.cell_id].reset_index()[ID])
        assert controls.groupby('unit_key').size().equals(inp.groupby('unit_key').size().clip(upper=2000))
        assert clusters[ID].equals(inp[ID]) and clusters.cluster.between(1,10).all()
        assert clusters.equals(table(old/'cell_clusters.csv.gz'))
        for name in ('spline_diagnostics.csv','cluster_batch_coverage.csv','cluster_cv.csv'):assert table(result/name).equals(table(old/name))
        oldsession=(old/'sessionInfo.txt').read_text().splitlines();newsession=(result/'sessionInfo.txt').read_text().splitlines()
        assert len(oldsession)==len(newsession)
        osdiff=[]
        for a,b in zip(oldsession,newsession):
            if a!=b:assert a.startswith('Running under: macOS ') and b.startswith('Running under: macOS ');osdiff.append({'old':a,'new':b})
        assert len(osdiff)<=1
        details={'case':case,'cells':len(inp),'markers':len(s['markers']),'units':int(inp.unit_key.nunique()),
                 'controls':len(controls),'zero_input_rows':int(np.all(inp[s['markers']].to_numpy(float)==0,axis=1).sum()),
                 'csv_max_absolute_difference':float(np.max(np.abs(x-oldtable.iloc[:,1:].to_numpy(float)))),
                 'byte_identical_files':sum(equality.values()),'total_result_files':len(equality),
                 'different_files':[n for n,v in equality.items() if not v],'session_os_difference':osdiff}
        if s['kind']=='technical':
            a=read(old/'resolved_config.json');b=read(result/'resolved_config.json')
            oldprefix=Path(a['preprocessing_file']).parent.parent/'alignment'
            assert b['preprocessing_file']==str(run/'runtime/tree'/s['input_manifest'])
            assert a['preprocessing_sha256']==b['preprocessing_sha256']==sha(run/'runtime/tree'/s['input_manifest'])
            a['preprocessing_file']=b['preprocessing_file'];assert a==b==read(run/'runtime_config.json')
            assert read(result/'integrity.json')==s['integrity']
            wa=(old/'warnings.txt').read_text().splitlines();wb=(result/'warnings.txt').read_text().splitlines();assert len(wa)==len(wb)
            changed=[]
            for a,b in zip(wa,wb):
                if a==b:continue
                pairs=[]
                for text,prefix in ((a,oldprefix),(b,result)):
                    match=re.fullmatch(r"cannot remove file '"+re.escape(str(prefix))+r"\.staging-([0-9]+)/work/normalize/(Norm_batch_0[123]_fsom(?:[1-9]|10)\.fcs)', reason 'No such file or directory'",text)
                    assert match,(case,text);pairs.append(match.groups())
                assert pairs[0][1]==pairs[1][1];changed.append({'file':pairs[0][1],'old_pid':pairs[0][0],'new_pid':pairs[1][0]})
            details['cleanup_path_only_warning_differences']=changed
            for directory in (old,result):
                done=read(directory/'COMPLETE.json');assert done['status']=='COMPLETE' and set(done['outputs'])==set(equality)-{'COMPLETE.json'}
                for name,digest in done['outputs'].items():assert sha(directory/name)==digest
        else:
            for name,nested in (('R_COMPLETE.json',False),('COMPLETE.json',True)):
                a=read(old/name);b=read(result/name)
                va=a['runtime'] if nested else a;vb=b['runtime'] if nested else b
                for v in (va,vb):assert not isinstance(v['elapsed_seconds'],bool) and math.isfinite(v['elapsed_seconds']) and v['elapsed_seconds']>=0;v.pop('elapsed_seconds')
                assert a==b,(case,name)
            assert read(result/'R_COMPLETE.json')['result']==s['integrity']
            a=np.load(old/'values.npy',allow_pickle=False);b=np.load(result/'values.npy',allow_pickle=False)
            default=pd.read_csv(result/'corrected_expression.csv.gz',low_memory=False).iloc[:,1:].to_numpy(float)
            assert a.dtype==b.dtype==np.float64 and np.array_equal(a,b) and np.array_equal(b,default)
            assert np.array_equal(np.load(bundle/'inputs'/s['ids'],allow_pickle=False),newtable.cell_id)
            details['npy_max_absolute_difference']=float(np.max(np.abs(a-b)))
        mr=model_report['cases'][case]
        assert mr['status']=='passed' and sha(old/'model.rds')==mr['old_sha256'] and sha(result/'model.rds')==mr['new_sha256']
        details['model_strict_semantic_parity']=True;details['spline_closures']=mr['spline_closures']
        results[case]=details
        print('PASS independent real result',case,'CSV error',details['csv_max_absolute_difference'],'strict model',True,flush=True)
    write('results_evidence.json',{'status':'passed','cases':results,'code_records':code,'batch_manifest':rec(BATCH/'BATCH_MANIFEST.json'),
                                 'source_hashes':288,'exact_hash_relationships':357,'model_comparator':rec(MOD/'compare_models.R')})
elif sys.argv[1]=='final':
    reportdir=MOD/'validation/seven_execution_20261011';r=read(reportdir/'report.json');assert r['status']=='passed'
    evidence=read(OUT/'results_evidence.json');models=read(MODEL_AUDIT/'all_models.json')
    prepath=MOD/'validation/execution_preflight_20261011/report.json'
    assert sha(prepath)==r['execution_preflight_sha256']=='2fbc7534fb22a7840a564ddd8d6d5249128effdf8b5e2f4ac6bfea6aa9511751'
    pre=read(prepath);assert pre['status']=='passed'
    archive=MOD/'provenance/execution_preflight_20261011';am=read(archive/'MANIFEST.json')
    assert sha(archive/'MANIFEST.json')==r['validator_refinement']['archive_sha256']=='122807df7c7c6ca98afac9573b369ce90edffd541dea92df89c2eac576399ae6'
    assert am['preflight_sha256']==sha(prepath)
    for name,record in am['files'].items():check(archive/name,record);assert pre['current_files'][name]==record
    for name,record in pre['existing_module_files'].items():
        path=archive/name if name in REFINED else MOD/name
        if 'symlink' in record:assert path.is_symlink() and str(path.resolve(strict=True))==record['symlink']
        else:check(path,record)
    for name,record in evidence['code_records'].items():assert pre['existing_module_files'][name]==record;check(MOD/name,record)
    for name,record in r['current_files'].items():check(MOD/name,record);check(OUT/'module'/name,record)
    check(BATCH/'BATCH_MANIFEST.json',r['batch_manifest']);assert r['batch_manifest']==evidence['batch_manifest']
    assert r['python_tests_passed']==101 and r['model_negative_tests']['returncode']==0
    for name in ('tests','common_inputs','model_negative_tests'):
        assert r[name]['returncode']==0
        for stream in ('stdout','stderr'):check(reportdir/(name+'.'+stream+'.log'),r[name][stream])
    assert read(reportdir/'model_negative_tests.stdout.log')['tests']==15
    for case,s in evidence['cases'].items():
        rc=r['cases'][case]
        assert rc['csv_maximum_absolute_difference']==s['csv_max_absolute_difference']==0
        assert rc['byte_identical_files']==s['byte_identical_files'] and rc['total_files']==s['total_result_files']
        if 'npy_max_absolute_difference' in s:assert rc['npy_maximum_absolute_difference']==s['npy_max_absolute_difference']==0
        expected={k:v for k,v in models['cases'][case].items() if k not in ('old_sha256','new_sha256')}
        actual={k:v for k,v in rc['model_semantic_comparison'].items() if k not in ('checker','stderr')}
        assert actual==expected,(case,'model report differs from independent model audit')
        for record,key in ((rc['moved_bundle_verification'],'moved_'+case),(rc['current_environment_check'],'current_env_'+case)):
            assert record['returncode']==0
            for stream in ('stdout','stderr'):check(reportdir/(key+'.'+stream+'.log'),record[stream])
        assert (reportdir/('current_env_'+case+'.stdout.log')).read_bytes()==(BATCH/case/'environment_before.stdout.log').read_bytes()
    outside=read(MOD/'validation/preflight_20261011/report.json')
    assert sha(MOD/'validation/preflight_20261011/report.json')==r['outside_preflight_sha256']=='3099bf86316519d18dbc7a4e618baebe42839a47679d0ddbd68ee0d4976298aa'
    existing={};study=ROOT/'ramp-study'
    for directory,dirs,files in os.walk(study):
        rel=Path(directory).relative_to(study)
        dirs[:]=sorted(d for d in dirs if not d.startswith('.') and d not in ('runs','__pycache__') and (rel/d).as_posix()!='methods/cytonorm')
        for name in sorted(files):
            if name.startswith('.'):continue
            p=Path(directory)/name;existing[p.relative_to(study).as_posix()]={'symlink':str(p.resolve(strict=True))} if p.is_symlink() else rec(p)
    assert existing==outside['existing_files'] and len(existing)==r['unchanged_outside_files']==1337
    assert read(reportdir/'baseline.stdout.log')==outside['prior']
    source_checks()
    assert r['source_verification']==r['postcheck']=={'files':288,'copies':101,'hash_relations':357,'branches':7}
    phase1=read(MOD/'validation/handoff_20261011_r2/report.json')
    for name,record in phase1['current_files'].items():check(MOD/name,record)
    assert sha(MOD/'validation/handoff_20261011_r2/report.json')==r['phase1_report_sha256']=='c91c922cf0c4e08b950ca9f1bf84cd730190b3fd6db8744099c17cb5db591a36'
    assert r['identical_seven_request_code_records'] and r['validator_refinement']['fit_code_unchanged']
    for key in ('validator_runs_real_fit','provider_preprocessing_rebuilt','UMAP_metrics_survival','clean_install'):assert r[key] is False
    (OUT/'reviewed_final_report.json').write_bytes((reportdir/'report.json').read_bytes())
    write('final_evidence.json',{'status':'passed','final_report':rec(reportdir/'report.json'),'preflight':rec(prepath),
         'baseline_module_entries':len(pre['existing_module_files']),
         'current_unchanged_baseline_module_entries':len(pre['existing_module_files'])-len(REFINED),
         'exact_archived_refined_baseline_module_entries':sorted(REFINED),'unchanged_outside_files':1337,
         'all_seven_numeric_and_semantic_reports_match_independent_checks':True,'unchanged_fit_code_records':evidence['code_records']})
    print('PASS final report, all model/output evidence, preflight archive, unchanged fit code and 1337-file prior baseline')
