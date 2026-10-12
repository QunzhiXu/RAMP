"""Create-only evidence for the J1 probability-boundary verifier correction."""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from common import *
import preflight
import validate_p2_fix as previous

OLD_REPORT = HERE/'validation/p2_fix_20261011/report.json'
OLD_REPORT_SHA = '986675da3681bcfacbd4623350ea1290e3d04253703787c3d457810f7447794a'
CAPTURE = HERE/'validation/probability_preflight_20261011'
CAPTURE_SHA = '2933fc1aeb180f720949df9cf103811e79a8be1d32e7ca43318a220663b2646f'


def capture(out):
    require(record(OLD_REPORT)['sha256']==OLD_REPORT_SHA,'Previous report changed')
    old=read(OLD_REPORT)
    for name, rec in old['current_code'].items():require(record(HERE/name)==rec,'Previous code changed')
    for name, rec in old['tests'].items():require(record(HERE/'tests'/name)==rec,'Previous tests changed')
    files=previous.module_files()
    prior=preflight.check(HERE/'validation/preflight_20261011/report.json')
    dest=output(out,PARENT.parents[2]/'final_xu',[OLD_REPORT])
    (dest/'controller').mkdir()
    for name in CODE:
        shutil.copy2(HERE/name,dest/'controller'/name)
        require(record(dest/'controller'/name)==files[name],'Archive mismatch')
    write(dest/'report.json',dict(status='captured',files=files,prior=prior,old_report_sha256=OLD_REPORT_SHA,code_records=code_records()))
    write(dest/'COMPLETE.json',dict(status='complete',report=record(dest/'report.json')))
    print(dict(files=len(files),report=record(dest/'report.json')),flush=True)


def preserved():
    require(record(CAPTURE/'report.json')['sha256']==CAPTURE_SHA,'Boundary preflight changed')
    before=read(CAPTURE/'report.json')
    allowed={'replay.py','validate_probability_fix.py'}
    for name,rec in before['files'].items():
        if name not in allowed:require(record(member(HERE,name))==rec,'Preserved file changed: '+name)
    for name,rec in before['code_records'].items():require(record(CAPTURE/'controller'/name)==rec,'Archive changed')
    require(record(OLD_REPORT)['sha256']==OLD_REPORT_SHA,'Previous P2 report changed')
    require(preflight.check(HERE/'validation/preflight_20261011/report.json')==before['prior'],'Prior baseline changed')
    old=(CAPTURE/'controller/replay.py').read_text()
    original="finite_number(score[name], 'J1 ' + name, 0, 1)"
    replacement="finite_number(score[name], 'J1 ' + name, -SCORE_TOLERANCE, 1 + SCORE_TOLERANCE)"
    require(old.count(original)==1 and old.replace(original,replacement)==(HERE/'replay.py').read_text(),
            'Production diff is not the single probability-boundary line')
    return dict(unchanged_files=len(before['files'])-len(allowed),prior=before['prior'],
                single_production_line=True,old_report_sha256=OLD_REPORT_SHA)


def validate(out):
    import binding
    import compare
    import replay
    before=preserved();source=PARENT.parents[2]/'final_xu'
    run=HERE/'runs/all_predictions_20261011';index=HERE/'bindings/current_predictions_20261011/catalog.json'
    code={p.name:record(p) for p in HERE.glob('*.py')};tests={p.name:record(p) for p in (HERE/'tests').glob('*.py')}
    dest=output(out,source,[run,index]);commands={}
    try:
        def invoke(argv,name,expected=0,message=None):
            env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env.pop('LANG',None)
            with (dest/(name+'.stdout')).open('xb') as stdout,(dest/(name+'.stderr')).open('xb') as stderr:
                result=subprocess.run([str(x) for x in argv],stdout=stdout,stderr=stderr,env=env)
            commands[name]=dict(command=[str(x) for x in argv],returncode=result.returncode,
                                stdout=record(dest/(name+'.stdout')),stderr=record(dest/(name+'.stderr')))
            require((result.returncode==0) if expected==0 else (result.returncode!=0),'Unexpected exit: '+name)
            if message:require(message in (dest/(name+'.stderr')).read_text(),'Unexpected rejection: '+name)
            print(name,result.returncode,flush=True)
        with tempfile.TemporaryDirectory(prefix='prediction_probability_fix_') as tmp:
            scratch=Path(tmp).resolve();local=previous.faithful_copy(scratch)
            probe=local/'tests/test_probability_boundary.py'
            # Same actual-scorer synthetic branch, three versions of the real
            # verifier, no monkeypatch of acceptance logic.
            for name,controller,rc in [
                ('original_accepts',previous.PREFLIGHT/'controller/replay.py',0),
                ('p2_guard_rejects',CAPTURE/'controller/replay.py',1),
                ('boundary_fix_accepts',local/'replay.py',0),
            ]:
                branch=scratch/name
                invoke([sys.executable,'-B',probe,'probe',controller,branch],name,rc,
                       'J1 accuracy outside allowed range' if rc else None)
                with __import__('gzip').open(branch/'scores.jsonl.gz','rt') as stream:saved=json.loads(next(stream))
                require(saved['score']['accuracy']==1.0000000000000002 and saved['max_abs_difference']==0.,'Edge fixture changed')
            edge=dict(class_counts=[6,23,1],patients=1,cells=30,accuracy=saved['score']['accuracy'],
                      max_abs_difference=0.,original_accepts=True,previous_p2_rejects=True,current_accepts=True)
            invoke([sys.executable,'-B','-m','unittest','discover','-s',local/'tests','-v'],'tests')
            count=re.search(r'Ran (\d+) tests',(dest/'tests.stderr').read_text());require(count is not None,'No test count')
            moved=scratch/'index with spaces';moved.mkdir()
            for name in ('catalog.json','COMPLETE.json'):shutil.copy2(index.parent/name,moved/name)
            def cli(mode,folder,destination=None):
                command=[sys.executable,'-B',local/'replay.py',mode,'--index',moved/'catalog.json',
                         '--anchor',previous.INDEX_ANCHOR,'--run',folder]
                return command+(['--output',destination] if destination else [])
            invoke(cli('verify-run',run),'valid_moved_verify')
            cases={
                'out_of_tolerance':'max_abs_difference','missing_diagnostics':'J1 score fields',
                'wrong_metric_count':'Branch metric count','nan_error':'max_abs_difference',
                'infinite_error':'max_abs_difference','negative_error':'max_abs_difference',
                'invalid_patient_support':'J1 patient support','j2_missing_baseline':'J2 score fields',
                'j2_missing_units':'J2 score fields','j2_wrong_status_count':'J2 status counts',
                'j2_wrong_baseline_support':'J2 baseline target support','wrong_prediction_hash':'prediction binding',
                'unknown_controller':'Batch identity/controller','partial_controller':'Batch identity/controller',
                'batch_report_count':'Batch report branches',
            }
            rejected={}
            for name,message in cases.items():
                with tempfile.TemporaryDirectory(dir=scratch,prefix=name+'_') as case_dir:
                    bad=Path(case_dir)/'batch';shutil.copytree(run,bad);previous.mutate_batch(bad,name)
                    hashes=previous.tree_records(bad)
                    invoke(cli('verify-run',bad),'verify_'+name,1,message)
                    target=Path(case_dir)/'rejected aggregate'
                    invoke(cli('aggregate',bad,target),'aggregate_'+name,1,message)
                    require(not target.exists(),'Rejected aggregate created output')
                    require(hashes==previous.tree_records(bad),'Verifier changed input')
                    rejected[name]=dict(reason=message,no_output=True,input_unchanged=True)
            fresh=scratch/'aggregate'
            invoke(cli('aggregate',run,fresh),'valid_aggregate')
            original=HERE/'validation/aggregation_20261011'
            require(read(fresh/'report.json')==read(original/'report.json'),'Aggregation report changed')
            tables={p.name:record(p) for p in fresh.glob('*.csv')}
            for name,rec in tables.items():require(record(original/name)==rec,'Aggregation CSV changed')
        catalog=binding.load(index,previous.INDEX_ANCHOR)
        comparison,_=compare.compare(catalog,run,previous.INDEX_ANCHOR)
        require(comparison==read(HERE/'validation/per_job_20261011/report.json'),'Per-job comparison changed')
        require(before==preserved(),'Preserved files changed')
        require(code=={p.name:record(p) for p in HERE.glob('*.py')} and tests=={p.name:record(p) for p in (HERE/'tests').glob('*.py')},'Code changed during validation')
        report=dict(status='passed',scope='One J1 verifier range-check line; no scientific rescoring or result rewrites',
                    preflight_sha256=CAPTURE_SHA,previous_fix_report_sha256=OLD_REPORT_SHA,tests_passed=int(count.group(1)),
                    regression=edge,run=comparison['run'],negative_cli=rejected,commands=commands,
                    aggregation_tables=tables,aggregation_report_unchanged=True,per_job_report_unchanged=True,
                    preserved=before,current_code=code,tests=tests,external_audit_approval=False)
        write(dest/'report.json',report);write(dest/'COMPLETE.json',dict(status='complete',files={p.name:record(p) for p in dest.iterdir()}))
        print(dict(status='passed',tests=report['tests_passed'],run=report['run']),flush=True)
    except BaseException as error:
        write(dest/'FAILED.json',dict(error=repr(error)));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['capture','validate'])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();(capture if args.mode=='capture' else validate)(args.output)


if __name__=='__main__':main()
