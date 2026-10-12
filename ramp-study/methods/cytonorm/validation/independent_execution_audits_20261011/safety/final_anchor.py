import hashlib
import json
from pathlib import Path
import sys

WORK=Path('/private/tmp/cytonorm_phase2_safety_zJXCnl54')
MOD=WORK/'module_final'
ORIG=Path('/Users/qxu6/Documents/New project/ramp-study/methods/cytonorm')
sys.path.insert(0,str(MOD))
import baseline
import execution as ex

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def row(p):return {'bytes':Path(p).stat().st_size,'sha256':sha(p)}
def logcheck(logdir,key,r):
    for stream in ('stdout','stderr'):assert row(logdir/(key+'.'+stream+'.log'))==r[stream]

p=ORIG/'validation/seven_execution_20261011/report.json'
assert sha(p)==sys.argv[1]
r=read(p);assert r['status']=='passed' and r['python_tests_passed']==101
assert r['phase1_report_sha256']=='c91c922cf0c4e08b950ca9f1bf84cd730190b3fd6db8744099c17cb5db591a36'
assert r['execution_preflight_sha256']=='2fbc7534fb22a7840a564ddd8d6d5249128effdf8b5e2f4ac6bfea6aa9511751'
assert r['source_verification']==r['postcheck']=={'files':288,'copies':101,'hash_relations':357,'branches':7}
assert r['batch_manifest']==row(ORIG/'runs/seven_fit_20261011/BATCH_MANIFEST.json')
assert r['batch_manifest']['sha256']=='5c6585fe32276f7bea4d24777002f55d37860c03113dd79ce91e92ba6e18e45e'
assert r['identical_seven_request_code_records'] is True
for name,expected in r['current_files'].items():assert row(ORIG/name)==expected==row(MOD/name),name
for key in ('tests','common_inputs','model_negative_tests'):logcheck(p.parent,key,r[key])
for case,data in r['cases'].items():
    independent=read(WORK/('semantic_'+case+'.json'))
    actual={k:v for k,v in data.items() if k not in ('moved_bundle_verification','current_environment_check')}
    assert actual==independent,case
    logcheck(p.parent,'moved_'+case,data['moved_bundle_verification'])
    logcheck(p.parent,'current_env_'+case,data['current_environment_check'])
    assert (p.parent/('current_env_'+case+'.stdout.log')).read_bytes()==(WORK/(case+'.environment.stdout.log')).read_bytes()
assert read(p.parent/'model_negative_tests.stdout.log')==read(WORK/'model_negative.stdout.log')
before=read(ORIG/'validation/preflight_20261011/report.json')
assert read(p.parent/'baseline.stdout.log')==before['prior']
baseline.check_existing(ORIG.parents[1],before['existing_files'])
pre=read(ORIG/'validation/execution_preflight_20261011/report.json')
archive=ORIG/'provenance/execution_preflight_20261011';ar=read(archive/'MANIFEST.json')
assert sha(archive/'MANIFEST.json')=='122807df7c7c6ca98afac9573b369ce90edffd541dea92df89c2eac576399ae6'
changed=[]
for name,expected in pre['existing_module_files'].items():
    original=ORIG/name
    current={'symlink':str(original.resolve(strict=True))} if original.is_symlink() else row(original)
    if current!=expected:
        assert name in r['validator_refinement']['files'] and row(archive/name)==expected==ar['files'][name]
        changed.append(name)
assert sorted(changed)==r['validator_refinement']['files']
assert len(pre['existing_module_files'])==440 and len(changed)==5
bm=read(ORIG/'runs/seven_fit_20261011/BATCH_MANIFEST.json')
for name,expected in bm['code_records'].items():assert row(ORIG/name)==expected==pre['existing_module_files'][name]
value={'status':'passed','final_report_sha256':sha(p),'all_seven_comparisons_equal_independent_reruns':True,
 'current_r_environment_equal_independent_readonly_checks':True,'outside_baseline_files_unchanged':len(before['existing_files']),
 'prefit_current_files_unchanged':len(pre['existing_module_files'])-len(changed),
 'prefit_refined_files_preserved_in_authenticated_archive':changed,
 'six_fit_code_records_unchanged':True,'all_referenced_final_logs_verified':True}
with (WORK/'final_anchor.json').open('x') as f:json.dump(value,f,indent=2)
print(json.dumps(value,indent=2))
