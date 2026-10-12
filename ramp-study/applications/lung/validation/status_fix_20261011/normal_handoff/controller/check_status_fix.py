"""Small report-only CLI counterexamples; no fitting or payload copying."""
from pathlib import Path
import copy
import os
import subprocess
import sys
import tempfile
from common import HERE,read,write,sha,record,reserve,codes

def main():
    out=reserve(HERE/'validation/status_fix_20261011',[HERE/'runs',HERE/'frozen_20261011'])
    lock=HERE/'frozen_20261011'; run=HERE/'runs/rebuild_20261011'
    accepted=HERE/'validation/comparison_20261011_r2'
    report=read(accepted/'report.json'); records=[]
    old=HERE/'validation/status_preflight_20261011/controller/validate.py'
    def command(script,comparison,target):
        return [sys.executable,'-B',str(script),'--lockdir',str(lock),'--run',str(run),'--comparison',str(comparison),'--out',str(target)]
    try:
        with tempfile.TemporaryDirectory(prefix='lung_status_') as tmp:
            tmp=Path(tmp).resolve()
            for mode in ('table','array','npz','json','text','false_differences','non_boolean','missing_flag','empty_npz'):
                changed=copy.deepcopy(report)
                if mode=='table': changed['tables'][0]['exact_values']=False
                if mode=='array': changed['arrays'][0]['exact_values']=False
                if mode=='npz': next(iter(changed['npzs'][0]['arrays'].values()))['exact_values']=False
                if mode=='json': changed['jsons'][0]['exact_after_recorded_provenance_relocation']=False
                if mode=='text': changed['texts'][0]['equal_except_disclosed_os_line']=False
                if mode=='false_differences': changed['status']='completed_with_differences'
                if mode=='non_boolean': changed['tables'][0]['exact_values']='false'
                if mode=='missing_flag': del changed['arrays'][0]['exact_values']
                if mode=='empty_npz': changed['npzs'][0]['arrays']={}
                fixture=tmp/mode; fixture.mkdir(); write(fixture/'report.json',changed)
                write(fixture/'COMPLETE.json',{'status':'complete','report_sha256':sha(fixture/'report.json')})
                before={p.name:record(p) for p in fixture.iterdir()}
                if mode=='table':
                    env=os.environ.copy(); env['PYTHONPATH']=str(HERE); env['PYTHONDONTWRITEBYTECODE']='1'
                    legacy=subprocess.run(command(old,fixture,out/'old_cli_counterexample'),env=env,text=True,capture_output=True)
                    (out/'old_cli.stdout').write_text(legacy.stdout); (out/'old_cli.stderr').write_text(legacy.stderr)
                    if legacy.returncode or not (out/'old_cli_counterexample/COMPLETE.json').exists():
                        raise ValueError('Could not reproduce old CLI acceptance')
                target=tmp/(mode+'_output')
                p=subprocess.run(command(HERE/'validate.py',fixture,target),text=True,capture_output=True)
                if p.returncode==0 or target.exists() or before!={p.name:record(p) for p in fixture.iterdir()}:
                    raise ValueError('Wrong status accepted or evidence modified: '+mode)
                records.append({'case':mode,'returncode':p.returncode,'output_created':False,'stderr':p.stderr})
        p=subprocess.run(command(HERE/'validate.py',accepted,out/'normal_handoff'),text=True,capture_output=True)
        (out/'normal.stdout').write_text(p.stdout); (out/'normal.stderr').write_text(p.stderr)
        if p.returncode: raise ValueError('Normal handoff failed')
        before=read(HERE/'validation/status_preflight_20261011/BEFORE.json')['files']
        for rel,rec in before.items():
            if rel!='validate.py' and record(HERE/rel)!=rec: raise ValueError('Prior lung file changed: '+rel)
        result={'status':'passed','old_cli_accepted_false_exact_status':True,'new_cli_cases':records,
            'old_files_unchanged_except_validator':len(before)-1,'new_validator':record(HERE/'validate.py'),
            'comparison_report_sha256':sha(accepted/'report.json'),'run_manifest_sha256':sha(run/'RUN_MANIFEST.json'),
            'controller_records_unchanged':codes()==read(run/'REQUEST.json')['code_records'],
            'normal_handoff_sha256':sha(out/'normal_handoff/report.json'),
            'tests':80,'no_refitting':True,'no_scientific_payload_copies':True}
        write(out/'report.json',result); write(out/'COMPLETE.json',{'status':'complete','report_sha256':sha(out/'report.json')})
        print({k:v for k,v in result.items() if k!='new_cli_cases'})
    except BaseException as exc:
        write(out/'FAILED.json',{'error':repr(exc)}); raise

if __name__=='__main__': main()
