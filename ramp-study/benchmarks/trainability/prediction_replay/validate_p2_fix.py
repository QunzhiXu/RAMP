"""Separate, create-only evidence for the saved-score verifier fix; never rescore."""
import argparse
import ast
import copy
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
from common import *
import preflight

PREFLIGHT = HERE / 'validation/p2_preflight_20261011'
INDEX_ANCHOR = '82eafb7e4e41652b23844b53247081e3242163cc606004771734d2ba51ea7c4d'
PREFLIGHT_ANCHOR = '046280a661911bf746597135688954f2f43c0c9ca80d65b57b2d9f8a44275a0c'


def module_files():
    result = {}
    for root, dirs, names in os.walk(HERE):
        dirs[:] = sorted(d for d in dirs if d != '__pycache__' and not d.startswith('.'))
        for name in sorted(names):
            if name == '.DS_Store' or name.endswith('.pyc'):
                continue
            path = Path(root) / name
            result[path.relative_to(HERE).as_posix()] = record(path)
    return result


def capture(destination):
    # Capture before reserving the new evidence directory; no old record is edited.
    records = module_files()
    prior = preflight.check(HERE / 'validation/preflight_20261011/report.json')
    out = output(destination, PARENT.parents[2] / 'final_xu', [HERE / 'validate_p2_fix.py'])
    try:
        (out / 'controller').mkdir()
        for name in CODE:
            shutil.copy2(HERE / name, out / 'controller' / name)
            require(record(out / 'controller' / name) == records[name], 'Archive changed')
        write(out / 'report.json', dict(status='captured', files=records, prior=prior,
                                       code_records=code_records()))
        write(out / 'COMPLETE.json', dict(status='complete', report=record(out / 'report.json')))
        print(dict(status='captured', files=len(records), report=record(out / 'report.json')))
    except BaseException as error:
        write(out / 'FAILED.json', dict(error=repr(error)))
        raise


def preserved():
    require(record(PREFLIGHT/'report.json')['sha256']==PREFLIGHT_ANCHOR,'P2 preflight changed')
    before=read(PREFLIGHT/'report.json')
    allowed={'replay.py','validate_p2_fix.py'}
    for name, expected in before['files'].items():
        if name not in allowed:require(record(member(HERE,name))==expected,'Old file changed: '+name)
    require(preflight.check(HERE/'validation/preflight_20261011/report.json')==before['prior'],'Prior baseline changed')
    for name, expected in before['code_records'].items():
        require(record(PREFLIGHT/'controller'/name)==expected,'Archived controller changed')
    # Function-level AND module-level comparison: all pre-existing code except
    # the two verification functions is identical; no fitting/scoring edits.
    old=ast.parse((PREFLIGHT/'controller/replay.py').read_text())
    new=ast.parse((HERE/'replay.py').read_text())
    previous={x.name:x for x in old.body if isinstance(x,(ast.FunctionDef,ast.ClassDef))}
    current={x.name:x for x in new.body if isinstance(x,(ast.FunctionDef,ast.ClassDef))}
    changed=[n for n,x in previous.items() if ast.dump(x)!=ast.dump(current[n])]
    require(set(changed)=={'verify_branch','verify_run'},'Unexpected existing function changed')
    old_statements=[ast.dump(x) for x in old.body if not isinstance(x,(ast.FunctionDef,ast.ClassDef))]
    new_statements=[ast.dump(x) for x in new.body if not isinstance(x,(ast.FunctionDef,ast.ClassDef))]
    for statement in old_statements:require(statement in new_statements,'Old module-level statement changed')
    extras=[x for x in new.body if not isinstance(x,(ast.FunctionDef,ast.ClassDef)) and ast.dump(x) not in old_statements]
    require(len(extras)==3 and isinstance(extras[0],ast.ImportFrom) and extras[0].module=='collections'
            and [a.name for a in extras[0].names]==['Counter'] and
            [x.targets[0].id for x in extras[1:]]==['ACCEPTED_REPLAY_CODE','SCORE_TOLERANCE'],'Unexpected new module-level statement')
    return dict(unchanged_old_module_files=len(before['files'])-len(allowed),changed_files=sorted(allowed),
                changed_existing_functions=changed,prior=before['prior'])


def faithful_copy(scratch, legacy=False):
    parent=scratch/'ramp-study/benchmarks/trainability'
    parent.mkdir(parents=True,exist_ok=True)
    local=parent/('prediction_replay_legacy' if legacy else 'prediction_replay')
    local.mkdir()
    for path in (PREFLIGHT/'controller' if legacy else HERE).glob('*.py'):shutil.copy2(path,local/path.name)
    if not legacy:shutil.copytree(HERE/'tests',local/'tests')
    if not (parent/SNAPSHOT.name).exists():
        for path in PARENT.glob('*.py'):shutil.copy2(path,parent/path.name)
        shutil.copytree(SNAPSHOT,parent/SNAPSHOT.name)
        dest=parent/'validation/handoff_20261011_r2';dest.mkdir(parents=True)
        shutil.copy2(PARENT/'validation/handoff_20261011_r2/report.json',dest/'report.json')
    return local


def tree_records(root):
    return {p.relative_to(root).as_posix():record(p) for p in root.rglob('*') if p.is_file()}


def replace_json(path,value):
    # For deliberately malformed artifacts in a private temporary copy ONLY.
    path.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=True)+'\n')


def mutate_batch(root, name):
    """Rehash every affected manifest, so probes exercise semantics, not hashes."""
    branch='J2__bone_marrow' if name.startswith('j2_') else 'J1__bone_marrow'
    folder=root/'branches'/branch
    with gzip.open(folder/'scores.jsonl.gz','rt') as stream:rows=[json.loads(line) for line in stream]
    report=read(folder/'report.json');summary=read(root/'report.json');request=read(root/'REQUEST.json')
    if name in ('out_of_tolerance','nan_error','infinite_error','negative_error'):
        value={'out_of_tolerance':1.0,'nan_error':float('nan'),'infinite_error':float('inf'),'negative_error':-1e-20}[name]
        rows[0]['max_abs_difference']=value
        report['max_abs_difference']=value
    elif name=='missing_diagnostics':
        for row in rows:
            for field in ('confusion_matrix','patients','cells','classes','zero_division','patient_confusion_matrices'):del row['score'][field]
    elif name=='wrong_metric_count':report['metric_rows']=0
    elif name=='invalid_patient_support':rows[0]['score']['patients']+=1
    elif name=='j2_missing_baseline':del rows[0]['score']['source_constant_baseline']
    elif name=='j2_missing_units':del rows[0]['score']['units']
    elif name=='j2_wrong_status_count':rows[0]['score']['rho_status_counts']['ok']=0
    elif name=='j2_wrong_baseline_support':rows[0]['score']['source_constant_baseline']['units'][0]['cells']+=1
    elif name=='wrong_prediction_hash':rows[0]['prediction_sha256']='0'*64
    elif name=='unknown_controller':request['code_records']['scores.py']['sha256']='0'*64
    elif name=='partial_controller':del request['code_records']['scores.py']
    elif name=='batch_report_count':summary['branches'][0]['metric_rows']=0
    else:raise ValueError('Unknown mutation')
    with gzip.open(folder/'scores.jsonl.gz','wt') as stream:
        for row in rows:stream.write(json.dumps(row,allow_nan=True)+'\n')
    replace_json(folder/'report.json',report)
    if name!='batch_report_count':
        summary['branches']=[report if (r['case'],r['task_type'])==(report['case'],report['task_type']) else r for r in summary['branches']]
    replace_json(root/'REQUEST.json',request);replace_json(root/'report.json',summary)
    manifest=read(folder/'RUN_MANIFEST.json')
    replace_json(folder/'RUN_MANIFEST.json',{n:record(folder/n) for n in manifest})
    replace_json(folder/'COMPLETE.json',dict(status='complete',manifest=record(folder/'RUN_MANIFEST.json')))
    batch=read(root/'BATCH_MANIFEST.json');batch['files']={n:record(root/n) for n in batch['files']}
    replace_json(root/'BATCH_MANIFEST.json',batch)
    replace_json(root/'COMPLETE.json',dict(status='complete',manifest=record(root/'BATCH_MANIFEST.json')))


def validate(destination):
    import binding
    import replay
    import compare
    import re
    before=preserved();before_code={p.name:record(p) for p in HERE.glob('*.py')}
    tests_code={p.name:record(p) for p in (HERE/'tests').glob('*.py')}
    require(replay.ACCEPTED_REPLAY_CODE==read(PREFLIGHT/'report.json')['code_records'],'Historical allowlist differs')
    source=PARENT.parents[2]/'final_xu';run=HERE/'runs/all_predictions_20261011'
    index=HERE/'bindings/current_predictions_20261011/catalog.json'
    dest=output(destination,source,[index,run]);commands={}
    try:
        catalog=binding.load(index,INDEX_ANCHOR)
        def invoke(argv,name,expected=0,message=None):
            env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env.pop('LANG',None)
            with (dest/(name+'.stdout')).open('xb') as stdout,(dest/(name+'.stderr')).open('xb') as stderr:
                proc=subprocess.run([str(x) for x in argv],stdout=stdout,stderr=stderr,env=env)
            text=(dest/(name+'.stderr')).read_text()
            result=dict(command=[str(x) for x in argv],returncode=proc.returncode,
                        stdout=record(dest/(name+'.stdout')),stderr=record(dest/(name+'.stderr')))
            commands[name]=result
            require((proc.returncode==0) if expected==0 else (proc.returncode!=0),'Unexpected CLI exit: '+name)
            if message:require(message in text,'Unexpected CLI rejection: '+name)
            print(name,proc.returncode,flush=True)
            return result
        with tempfile.TemporaryDirectory(prefix='prediction_p2_validation_') as tmp:
            scratch=Path(tmp).resolve();local=faithful_copy(scratch);old=faithful_copy(scratch,True)
            moved=scratch/'index moved with spaces';moved.mkdir()
            for name in ('catalog.json','COMPLETE.json'):shutil.copy2(index.parent/name,moved/name)
            invoke([sys.executable,'-B','-m','unittest','discover','-s',local/'tests','-v'],'tests')
            tested=re.search(r'Ran (\d+) tests', (dest/'tests.stderr').read_text())
            require(tested is not None,'No test count')
            def cli(code,mode,folder,output_path=None):
                argv=[sys.executable,'-B',code/'replay.py',mode,'--index',moved/'catalog.json','--anchor',INDEX_ANCHOR,'--run',folder]
                return argv+(['--output',output_path] if output_path else [])
            invoke(cli(local,'verify-run',run),'valid_moved_verify')
            cases={
                'out_of_tolerance':'max_abs_difference', 'missing_diagnostics':'J1 score fields',
                'wrong_metric_count':'Branch metric count', 'nan_error':'max_abs_difference',
                'infinite_error':'max_abs_difference', 'negative_error':'max_abs_difference',
                'invalid_patient_support':'J1 patient support', 'j2_missing_baseline':'J2 score fields',
                'j2_missing_units':'J2 score fields', 'j2_wrong_status_count':'J2 status counts',
                'j2_wrong_baseline_support':'J2 baseline target support', 'wrong_prediction_hash':'prediction binding',
                'unknown_controller':'Batch identity/controller', 'partial_controller':'Batch identity/controller',
                'batch_report_count':'Batch report branches',
            }
            probes={}
            for name,message in cases.items():
                with tempfile.TemporaryDirectory(dir=scratch,prefix=name+'_') as case_dir:
                    bad=Path(case_dir)/'batch';shutil.copytree(run,bad);mutate_batch(bad,name)
                    hashes=tree_records(bad)
                    # Reproduce the auditor's exact three defects against the
                    # archived controller, not a mock or reconstructed CLI.
                    if name in ('out_of_tolerance','missing_diagnostics','wrong_metric_count'):
                        invoke(cli(old,'verify-run',bad),'old_'+name)
                        require("'status': 'passed'" in (dest/('old_'+name+'.stdout')).read_text(),'Old CLI did not pass')
                        if name=='out_of_tolerance':
                            old_out=Path(case_dir)/'old aggregate'
                            invoke(cli(old,'aggregate',bad,old_out),'old_aggregate_out_of_tolerance')
                            require(read(old_out/'report.json')['status']=='passed','Old aggregate not reproduced')
                    invoke(cli(local,'verify-run',bad),'new_'+name,1,message)
                    rejected=Path(case_dir)/'rejected aggregate'
                    invoke(cli(local,'aggregate',bad,rejected),'aggregate_'+name,1,message)
                    require(not rejected.exists(),'Rejected aggregate created output')
                    require(hashes==tree_records(bad),'Read-only verifier changed malformed input')
                    probes[name]=dict(rejected=True,aggregate_rejected_before_output=True,input_unchanged=True,reason=message)
            # Recreate only reporting tables in scratch; compare the entire
            # numeric report and all table bytes to the retained original.
            fresh=scratch/'new aggregate'
            invoke(cli(local,'aggregate',run,fresh),'valid_aggregate')
            original=HERE/'validation/aggregation_20261011'
            require(read(fresh/'report.json')==read(original/'report.json'),'Aggregation report changed')
            aggregate_files={p.name:record(p) for p in fresh.glob('*.csv')}
            for name,expected in aggregate_files.items():require(record(original/name)==expected,'Aggregation table changed')
        per_job,_=compare.compare(catalog,run,INDEX_ANCHOR)
        require(per_job==read(HERE/'validation/per_job_20261011/report.json'),'Per-job report changed')
        require(before==preserved(),'Preserved baselines changed')
        require(before_code=={p.name:record(p) for p in HERE.glob('*.py')} and
                tests_code=={p.name:record(p) for p in (HERE/'tests').glob('*.py')},'Code changed during verification')
        report=dict(status='passed',scope='Verifier-only P2 fix; no rescoring, fitting or result rewrites',
                    preflight_sha256=PREFLIGHT_ANCHOR,index_sha256=INDEX_ANCHOR,tests_passed=int(tested.group(1)),
                    run=per_job['run'],negative_cli=probes,commands=commands,
                    old_cli_defects_reproduced=3,old_aggregate_defect_reproduced=True,
                    per_job_report_unchanged=True,aggregation_report_unchanged=True,aggregation_tables=aggregate_files,
                    preserved=before,current_code=before_code,tests=tests_code,
                    old_controller=replay.ACCEPTED_REPLAY_CODE,external_audit_approval=False)
        write(dest/'report.json',report)
        write(dest/'COMPLETE.json',dict(status='complete',files={p.name:record(p) for p in dest.iterdir()}))
        print(dict(status='passed',tests=report['tests_passed'],run=report['run']),flush=True)
    except BaseException as error:
        write(dest/'FAILED.json',dict(error=repr(error)));raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['capture','validate'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    (capture if args.mode=='capture' else validate)(args.output)


if __name__ == '__main__':
    main()
