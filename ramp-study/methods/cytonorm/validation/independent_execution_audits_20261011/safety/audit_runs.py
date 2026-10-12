import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

WORK=Path('/private/tmp/cytonorm_phase2_safety_zJXCnl54')
MOD=WORK/'module'
ORIG=Path('/Users/qxu6/Documents/New project/ramp-study/methods/cytonorm')
SOURCE=ORIG.parents[2]/'final_xu'
RUNS=ORIG/'runs/seven_fit_20261011'
sys.path.insert(0,str(MOD))
import execution as ex
import inventory as inv
import baseline
import numpy as np
import pandas as pd

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def row(p):return {'bytes':Path(p).stat().st_size,'sha256':sha(p)}
def table(p):return pd.read_csv(p,dtype={n:str for n in ('cell_id','dataset','unit_key')},keep_default_na=False,low_memory=False,float_precision='round_trip')
def save(name,value):
    with (WORK/name).open('x') as f:json.dump(value,f,indent=2)

assert sha(RUNS/'BATCH_MANIFEST.json')=='5c6585fe32276f7bea4d24777002f55d37860c03113dd79ce91e92ba6e18e45e'
assert sha(ORIG/'validation/execution_preflight_20261011/report.json')=='2fbc7534fb22a7840a564ddd8d6d5249128effdf8b5e2f4ac6bfea6aa9511751'
assert sha(ORIG/'provenance/execution_preflight_20261011/MANIFEST.json')=='122807df7c7c6ca98afac9573b369ce90edffd541dea92df89c2eac576399ae6'
b=read(RUNS/'BATCH_MANIFEST.json');br=read(RUNS/'BATCH_REQUEST.json')
assert read(RUNS/'COMPLETE.json')=={'status':'complete','manifest_sha256':sha(RUNS/'BATCH_MANIFEST.json')}
cat=read(MOD/'frozen_20261011/catalog.json');specs=cat['branches'];cases=[s['case'] for s in specs]
assert [r['case'] for r in b['cases']]==br['cases']==cases
assert b['code_records']==br['code_records']==ex.code_records()
for name,expected in b['code_records'].items():assert row(ORIG/name)==expected
assert {p.name for p in RUNS.iterdir()}==set(cases)|{'BATCH_REQUEST.json','BATCH_MANIFEST.json','COMPLETE.json'}
preflight=read(ORIG/'validation/execution_preflight_20261011/report.json')
archive=ORIG/'provenance/execution_preflight_20261011';ar=read(archive/'MANIFEST.json')
for name,expected in ar['files'].items():assert row(archive/name)==expected==preflight['current_files'][name]
old_changes=[]
for name,expected in preflight['existing_module_files'].items():
    p=ORIG/name
    actual={'symlink':str(p.resolve(strict=True))} if p.is_symlink() else row(p)
    if actual!=expected:
        assert name in ar['files'] and name not in b['code_records'],name
        assert row(archive/name)==expected
        old_changes.append(name)
before=read(ORIG/'validation/preflight_20261011/report.json')
baseline.check_existing(ORIG.parents[1],before['existing_files'])
save('baseline.json',{'outside_files_unchanged':len(before['existing_files']),
 'preflight_existing_module_files':len(preflight['existing_module_files']),
 'changed_comparison_or_documentation_files_with_archives':old_changes})

expected_versions={r['package']:r['version'] for r in csv.DictReader((MOD/'frozen_20261011/environment/runtime_lock.csv').open())}
case_results=[]
for spec,batchrow in zip(specs,b['cases']):
    case=spec['case'];run=RUNS/case;bundle=ORIG/'runs'/(case+'_inputs_20261011')
    assert batchrow['run']==case and Path(batchrow['bundle'])==bundle
    assert row(bundle/'MANIFEST.json')==br['bundle_manifests'][case]
    assert row(run/'RUN_MANIFEST.json')==batchrow['manifest']
    assert ex.verify_run(run,bundle)==batchrow['result']
    req=read(run/'REQUEST.json');m=read(run/'RUN_MANIFEST.json')
    assert req['code_records']==b['code_records']
    assert Path(req['bundle'])==bundle and req['bundle_manifest']==row(bundle/'MANIFEST.json')
    assert Path(req['controller_root'])==ORIG and req['rscript']=='/usr/local/bin/Rscript'
    assert req['rscript_record']==row(req['rscript'])
    assert req['library']==str(SOURCE/'algorithm/CytoNorm/.Rlib')
    assert read(run/'COMPLETE.json')=={'status':'complete','manifest_sha256':sha(run/'RUN_MANIFEST.json')}
    _,files=ex.verify_bundle(bundle)
    result=run/ex.result_path(spec)
    resultfiles=ex.TECH_FILES if spec['kind']=='technical' else (*ex.STORY_FILES,'values.npy','COMPLETE.json')
    required={ex.runtime_path(f,spec).as_posix() for f in files}
    required|={(ex.result_path(spec)/name).as_posix() for name in resultfiles}
    required|={'REQUEST.json','COMMANDS.json'}|{n+'.'+s+'.log' for n in ('environment_before','correction','environment_after') for s in ('stdout','stderr')}
    if spec['kind']=='technical':required|={'runtime_config.json','CONFIG_RELOCATION.json'}
    assert set(m['files'])==required,(case,set(m['files'])^required)
    for name,expected in m['files'].items():assert row(run/name)==expected
    assert not (run/'FAILED.json').exists()
    links={p.relative_to(run).as_posix():str(p.resolve(strict=True)) for p in run.rglob('*') if p.is_symlink()}
    assert links==m['links']=={'runtime/tree/algorithm/CytoNorm/.Rlib':req['library']}
    for f in files:assert (run/ex.runtime_path(f,spec)).read_bytes()==(bundle/f['path']).read_bytes()
    commands=read(run/'COMMANDS.json')
    assert commands==ex.command_plan(run,spec,req['rscript'],ORIG)
    for phase in ('environment_before','environment_after'):
        env=read(run/(phase+'.stdout.log'))
        assert env['runtime_versions']==expected_versions
        assert env['runtime_lock_sha256']==sha(MOD/'frozen_20261011/environment/runtime_lock.csv')
    assert (run/'environment_before.stdout.log').read_bytes()==(run/'environment_after.stdout.log').read_bytes()
    childenv=ex.environment();childenv['TMPDIR']=str(WORK)
    with (WORK/(case+'.environment.stdout.log')).open('x') as stdout,(WORK/(case+'.environment.stderr.log')).open('x') as stderr:
        proc=subprocess.run(commands['environment'],env=childenv,cwd=WORK,stdout=stdout,stderr=stderr)
    assert proc.returncode==0
    assert (WORK/(case+'.environment.stdout.log')).read_bytes()==(run/'environment_before.stdout.log').read_bytes()
    inp=table(bundle/'inputs'/spec['input']);corrected=table(result/'corrected_expression.csv.gz')
    ident=['cell_id','dataset','unit_key'];markers=spec['markers']
    assert list(inp)==ident+markers and list(corrected)==['cell_id']+markers
    assert inp.cell_id.equals(corrected.cell_id)
    assert inp[ident].equals(table(SOURCE/spec['canonical']['evaluation_metadata'])[ident])
    assert inp[ident].equals(table(result/'row_metadata.csv.gz'))
    controls=table(result/'aggregate_cells.csv.gz')
    assert controls.equals(inp.set_index('cell_id').loc[controls.cell_id].reset_index()[ident])
    assert controls.groupby('unit_key').size().equals(inp.groupby('unit_key').size().clip(upper=2000))
    clusters=table(result/'cell_clusters.csv.gz')
    assert clusters[ident].equals(inp[ident]) and clusters.cluster.isin(range(1,11)).all()
    olddir=SOURCE/spec['result_dir']
    oldout=table(olddir/'corrected_expression.csv.gz')
    assert np.array_equal(oldout.to_numpy(),corrected.to_numpy())
    core=['corrected_expression.csv.gz','aggregate_cells.csv.gz','cell_clusters.csv.gz','row_metadata.csv.gz','cluster_batch_coverage.csv','cluster_cv.csv','spline_diagnostics.csv']
    if spec['kind']=='technical':core+=['integrity.json','markers.txt']
    for name in core:assert (olddir/name).read_bytes()==(result/name).read_bytes(),(case,name)
    if spec['kind']=='story':
        newvalues=np.load(result/'values.npy',allow_pickle=False);oldvalues=np.load(olddir/'values.npy',allow_pickle=False)
        assert newvalues.dtype==np.float64 and np.array_equal(newvalues,oldvalues)
        assert np.array_equal(newvalues,pd.read_csv(result/'corrected_expression.csv.gz',low_memory=False).iloc[:,1:].to_numpy(float))
    else:
        originalcfg=read(bundle/'inputs'/spec['config']);newcfg=read(run/'runtime_config.json')
        assert [n for n in originalcfg if originalcfg[n]!=newcfg[n]]==['preprocessing_file']
        assert Path(newcfg['preprocessing_file']).is_relative_to(run)
        assert sha(newcfg['preprocessing_file'])==originalcfg['preprocessing_sha256']
    differing=[name for name in resultfiles if sha(olddir/name)!=sha(result/name)]
    case_results.append({'case':case,'payload_records':len(required),'runtime_copies':len(files),
       'cells':len(inp),'markers':len(markers),'units':int(inp.unit_key.nunique()),
       'controls':len(controls),'csv_difference':0,'npy_difference':0 if spec['kind']=='story' else None,
       'byte_identical_scientific_tables':core,'differing_files_pending_semantic_review':differing})
    print('PASS',case,flush=True)
save('runs.json',case_results)
save('source_verification.json',inv.verify_source(SOURCE))
assert sha(RUNS/'BATCH_MANIFEST.json')=='5c6585fe32276f7bea4d24777002f55d37860c03113dd79ce91e92ba6e18e45e'
for name,expected in b['code_records'].items():assert row(ORIG/name)==expected
print('PASS source and frozen controllers',flush=True)
