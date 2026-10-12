"""Small-copy handoff checks; reports are bound to the verified current run."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from common import HERE, STUDY, read, write, sha, record, reserve, resource_state
from study import verify_run
from prepare import verify as verify_lock
from baseline import check as check_prior

def report_binding(lockdir,run,comparison):
    folder=Path(comparison)
    if (folder/'FAILED.json').exists(): raise ValueError('Failed comparison')
    digest=sha(folder/'report.json')
    if read(folder/'COMPLETE.json')!={'status':'complete','report_sha256':digest}:
        raise ValueError('Unsealed comparison')
    report=read(folder/'report.json')
    expected={'run_manifest_sha256':sha(Path(run)/'RUN_MANIFEST.json'),
              'lock_sha256':sha(Path(lockdir)/'LOCK.json')}
    for k,v in expected.items():
        if report.get(k)!=v: raise ValueError('Comparison binding differs: '+k)
    if report.get('controller')!=record(HERE/'compare.py'):
        raise ValueError('Unknown comparison controller')
    from common import codes
    if report.get('execution_controllers') != codes(): raise ValueError('Comparison execution controllers differ')
    if report.get('status') not in ('exact_with_disclosed_reuse_scope','completed_with_differences'):
        raise ValueError('Unknown comparison status')
    return report, {'comparison_report_sha256':digest,**expected}

def original_indexes():
    answer={}
    for name in ('source_census.json','curated_sources.json'):
        records=read(STUDY/'inventory_20261010/evidence'/name)['files']; checked=0
        for item in records:
            if item.get('sha256'):
                if sha(STUDY.parent/item['path'])!=item['sha256']:
                    raise ValueError('Original indexed file changed: '+item['path'])
                checked+=1
        answer[name]=checked
    return answer

def comparison_coverage(report,lock):
    expected={k for k,v in lock['files'].items() if v['role']=='reference'}
    items=[]
    for kind in ('tables','arrays','npzs','jsons','texts'):
        if not isinstance(report.get(kind),list): raise ValueError('Comparison category missing: '+kind)
        items.extend(report[kind])
    paths=[item['path'] for item in items]
    if len(paths)!=len(set(paths)) or set(paths)!=expected or report.get('reference_files_compared')!=len(expected):
        raise ValueError('Comparison reference coverage differs')
    if any(type(item.get('byte_identical')) is not bool for item in items):
        raise ValueError('Comparison byte identity field differs')
    if report.get('byte_identical_files')!=sum(item['byte_identical'] for item in items):
        raise ValueError('Comparison byte identity count differs')
    return {'reference_files':len(expected)}

def validate(lockdir,run,comparison,output):
    lockdir=Path(lockdir).resolve(); run=Path(run).resolve(); comparison=Path(comparison).resolve()
    structure=verify_run(lockdir,run); lock=verify_lock(lockdir,source=True)
    report,binding=report_binding(lockdir,run,comparison)
    coverage=comparison_coverage(report,lock)
    out=reserve(output,[Path(lock['source_root']),lockdir,run,comparison])
    try:
        (out/'controller').mkdir()
        for p in HERE.glob('*.py'): shutil.copyfile(p,out/'controller'/p.name)
        for name,rec in read(run/'REQUEST.json')['code_records'].items():
            if record(out/'controller'/name)!=rec: raise ValueError('Archived execution code changed')
        with tempfile.TemporaryDirectory(prefix='lung_handoff_') as tmp:
            tmp=Path(tmp).resolve(); controller=tmp/'controller'; controller.mkdir()
            for p in HERE.glob('*.py'): shutil.copyfile(p,controller/p.name)
            shutil.copytree(HERE/'tests',controller/'tests',ignore=shutil.ignore_patterns('__pycache__'))
            shutil.copytree(lockdir,controller/'frozen_20261011')
            p=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','tests','-v'],cwd=controller,text=True,capture_output=True)
            (out/'tests.stdout').write_text(p.stdout); (out/'tests.stderr').write_text(p.stderr)
            if p.returncode: raise ValueError('Temporary tests failed')
            moved=tmp/'moved lock with spaces'; (controller/'frozen_20261011').rename(moved)
            p=subprocess.run([sys.executable,'-B',str(controller/'study.py'),'verify-run','--lockdir',str(moved),'--run',str(run)],text=True,capture_output=True)
            (out/'moved.stdout').write_text(p.stdout); (out/'moved.stderr').write_text(p.stderr)
            if p.returncode: raise ValueError('Moved verification failed')
            copied_bytes=sum(p.stat().st_size for p in tmp.rglob('*') if p.is_file())
            negatives=[]
            for mutation in ('run_hash','lock_hash','controller','execution_controller','duplicate_payload','missing_payload','status','unsealed','failed'):
                fixture=tmp/('negative_'+mutation); fixture.mkdir(); altered=json.loads(json.dumps(report))
                if mutation=='run_hash': altered['run_manifest_sha256']='0'*64
                if mutation=='lock_hash': altered['lock_sha256']='0'*64
                if mutation=='controller': altered['controller']['sha256']='0'*64
                if mutation=='execution_controller': altered['execution_controllers']['worker.py']['sha256']='0'*64
                if mutation=='duplicate_payload': altered['tables'][1]=altered['tables'][0]
                if mutation=='missing_payload': altered['arrays'].pop()
                if mutation=='status': altered['status']='passed'
                write(fixture/'report.json',altered)
                write(fixture/'COMPLETE.json',{'status':'complete','report_sha256':'0'*64 if mutation=='unsealed' else sha(fixture/'report.json')})
                if mutation=='failed': write(fixture/'FAILED.json',{})
                target=tmp/(mutation+'_output'); before={p.name:record(p) for p in fixture.iterdir()}
                command=[sys.executable,'-B',str(HERE/'validate.py'),'--lockdir',str(lockdir),'--run',str(run),'--comparison',str(fixture),'--out',str(target)]
                p=subprocess.run(command,text=True,capture_output=True)
                if p.returncode==0 or target.exists() or before!={p.name:record(p) for p in fixture.iterdir()}:
                    raise ValueError('CLI accepted mismatched evidence: '+mutation)
                negatives.append({'mutation':mutation,'returncode':p.returncode,'output_created':False,'stderr':p.stderr})
            write(out/'cli_negatives.json',negatives)
        prior=check_prior(HERE/'validation/preflight_20261011'); original=original_indexes()
        if report_binding(lockdir,run,comparison)!=(report,binding): raise ValueError('Comparison changed during handoff')
        verify_run(lockdir,run); verify_lock(lockdir,source=True)
        result={'status':'internally_checked_pending_external_audit','bindings':binding,'structure':structure,
                'comparison_status':report['status'],'comparison_coverage':coverage,'tests_returncode':0,'moved_controller_and_lock_returncode':0,
                'cli_negative_cases':len(negatives),'temporary_copied_bytes':copied_bytes,'prior':prior,'original_indexes':original,
                'current_code':{p.name:record(p) for p in sorted(HERE.glob('*.py'))},
                'resources':resource_state(),'no_final_figure_or_full_provider_extraction':True,
                'known_scope_limit':'current Senosain140k release; frozen full-cell inputs and immune measurement tables; reuse accepted RAMP'}
        write(out/'report.json',result); write(out/'COMPLETE.json',{'status':'complete','report_sha256':sha(out/'report.json')})
        return result
    except BaseException as e:
        write(out/'FAILED.json',{'error':repr(e)}); raise

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--lockdir',required=True); p.add_argument('--run',required=True)
    p.add_argument('--comparison',required=True); p.add_argument('--out',required=True)
    a=p.parse_args(); print(json.dumps(validate(a.lockdir,a.run,a.comparison,a.out),indent=2))
