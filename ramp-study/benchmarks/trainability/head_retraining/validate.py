"""Post-fit handoff checks in disposable copies; no real head fitting."""
import argparse
import copy as copying
import re
import subprocess
import sys
import tempfile
sys.dont_write_bytecode=True
from contract import *
import preflight
from compare import compare

def run_command(command,output,expect=0):
    p=subprocess.run(command,env=child_environment(),capture_output=True,text=True)
    write(output,dict(command=command,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
    require(p.returncode==expect if expect==0 else p.returncode!=0,'Unexpected CLI status: '+str(command))
    return p

def reseal(folder,details):
    for n in ('RUN_MANIFEST.json','COMPLETE.json'):(folder/n).unlink()
    seal(folder,details)

def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    bundle=a.bundle.resolve();run=a.run.resolve();source=a.source_root.resolve()
    out=reserve(a.output,[bundle,run,source]);commands=[]
    try:
        old=read(HERE/'validation/preflight_20261011/report.json');require(preflight.capture()==old,'Old baseline changed')
        plan=verify_bundle(bundle);before=dict(run=files(run),bundle=files(bundle))
        report=compare(bundle,run,source)
        require(report==read(HERE/'validation/comparison_20261011/report.json'),'Comparison changed')
        require(report['status']=='exact','Pilot numerical differences require separate handoff')
        with tempfile.TemporaryDirectory(prefix='head_pilot_audit_') as temp:
            root=Path(temp).resolve();mod=root/'ramp-study/benchmarks/trainability/head_retraining';mod.mkdir(parents=True)
            for p in HERE.glob('*.py'):shutil.copy2(p,mod/p.name)
            shutil.copytree(HERE/'locks',mod/'locks');shutil.copytree(HERE/'tests',mod/'tests')
            for n,v in plan['training_files'].items():
                if n.startswith('code/'):copy(member(source,v['source']),root/'final_xu'/v['source'],v['record'])
            result=run_command([sys.executable,'-B','-m','unittest','discover','-s',str(mod/'tests'),'-v'],out/'tests.json')
            match=re.search(r'Ran (\d+) tests',result.stderr);require(match is not None,'Missing test count');test_count=int(match[1])
            moved=root/'moved bundle with spaces';shutil.copytree(bundle,moved)
            base=[sys.executable,'-B',str(mod/'study.py')]
            run_command(base+['verify-run','--bundle',str(moved),'--run',str(run)],out/'moved_bundle.json')
            bad=root/'malformed batch';shutil.copytree(run,bad);details=read(bad/'RUN_MANIFEST.json')['details']
            pristine={n:(bad/n).read_bytes() for n in ('report.json','BATCH_REQUEST.json','RUN_MANIFEST.json','COMPLETE.json')}
            negative=[]
            cases=('duplicate','missing','extra','wrong_batch_identity')
            for name in cases:
                for n,data in pristine.items():(bad/n).write_bytes(data)
                if name=='wrong_batch_identity':
                    d=read(bad/'BATCH_REQUEST.json');d['scope']='all32370';(bad/'BATCH_REQUEST.json').write_text(json.dumps(d))
                else:
                    d=read(bad/'report.json')
                    if name=='duplicate':d['jobs']=[d['jobs'][0]]*45
                    elif name=='missing':d['jobs'].pop()
                    else:d['jobs'].append(dict(d['jobs'][0],id='unregistered_job'))
                    (bad/'report.json').write_text(json.dumps(d))
                reseal(bad,details)
                fingerprint=files(bad)
                r=run_command(base+['verify-run','--bundle',str(moved),'--run',str(bad)],out/(name+'.json'),expect=1)
                require('Pilot task grid differs' in r.stderr or 'Batch request differs' in r.stderr,'Wrong refusal reason')
                require(files(bad)==fingerprint,'Read-only CLI wrote to malformed run');negative.append(name)
            # Tampering a moved input is rejected, not repaired or silently reloaded.
            native=next(n for n in plan['training_files'] if n.endswith('ledger.json'));p=member(moved,native)
            with p.open('ab') as f:f.write(b' ')
            r=run_command(base+['verify','--bundle',str(moved)],out/'input_tamper.json',expect=1)
            require('Bundle payload differs' in r.stderr,'Wrong input refusal')
            negative.append('input_tamper')
        probe=run_command([sys.executable,'-B','-c','import sys,json;sys.path.insert(0,sys.argv[1]);from contract import environment;print(json.dumps(environment()))',str(HERE)],out/'runtime_probe.json')
        current=json.loads(probe.stdout)
        for ident in plan['tasks']:require(current==read(run/ident/'environment_before.json'),'Current environment differs')
        require(dict(run=files(run),bundle=files(bundle))==before,'Real artifacts changed during validation')
        require(preflight.capture()==old,'Old baseline changed during validation')
        request_codes=[read(run/i/'REQUEST.json')['code_records'] for i in plan['tasks']]
        require(all(v==codes() for v in request_codes),'Request code records differ')
        manifest=dict(status='passed_internal_validation_pending_external_audit',tests=test_count,cli_rejections=negative,
            exact_jobs=report['exact_jobs'],total_predictions=report['total_predictions'],changed_predictions=report['changed_predictions'],
            code_records_identical=all(v==request_codes[0] for v in request_codes),
            old_trainability_files=len(old['trainability_files']),external_non_run_files=len(old['external']),prior=old['prior'],
            current_code={p.relative_to(HERE).as_posix():record(p) for p in sorted(HERE.rglob('*.py')) if not any(part in {'runs','bundles','validation','__pycache__'} for part in p.relative_to(HERE).parts)},
            plan=record(HERE/'locks/pilot_20261011.json'),comparison=record(HERE/'validation/comparison_20261011/report.json'),
            run_manifest=record(run/'RUN_MANIFEST.json'),bundle_manifest=record(bundle/'MANIFEST.json'),
            limits='Pilot45; no representation refitting, no full task grid, no new class/arm/parameter choices, no clean-install or cross-host validation')
        write(out/'report.json',manifest);seal(out,dict(scope='pilot_handoff_validation'))
        print({k:manifest[k] for k in ('status','tests','exact_jobs','total_predictions','changed_predictions')})
    except BaseException as e:
        if not(out/'FAILED.json').exists():write(out/'FAILED.json',dict(error=repr(e)))
        raise

if __name__=='__main__':main()
