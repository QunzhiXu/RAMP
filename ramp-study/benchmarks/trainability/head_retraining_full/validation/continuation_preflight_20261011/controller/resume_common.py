"""Explicit continuation policy; the original controller and failed batch stay immutable."""
import full_common as original
from full_common import *

INTERRUPTION_SHA='c21ee267d4f804869c0568183298a4b51ea3fec2d4e80c6cdb3fc1c26f3dc363'
EXTRA_CODE=('resume_common.py','resume_worker.py','resume_run.py')
RESERVE_BYTES=15*1024**3
RESOURCE_POLICY=dict(reserve_bytes=RESERVE_BYTES,authorization='User approved continuation with 15 GiB reserve; only remaining 810 jobs; no deletion',scientific_parameters_changed=False)

def interruption():
    p=HERE/'validation/interruption_20261011/report.json'
    require(record(p)['sha256']==INTERRUPTION_SHA,'Interruption anchor changed')
    value=read(p)
    require(original.codes()==value['code_records'],'Original controller changed')
    return value

def codes():
    interruption()
    return original.codes()|{n:record(HERE/n) for n in EXTRA_CODE}

def remaining(root):
    prior=interruption();accepted=prior['sealed_groups']
    require(len(accepted)==845 and sum(v['jobs'] for v in accepted.values())==31560,'Prior sealed grid differs')
    require(set(accepted)<=set(root['groups']),'Unknown prior group')
    todo=[i for i in root['groups'] if i not in accepted]
    require(set(todo)==set(prior['failed_groups'])|set(prior['not_started_groups']),'Remaining group grid differs')
    require(len(todo)==15 and sum(len(root['groups'][i]['jobs']) for i in todo)==810,'Remaining job grid differs')
    return todo

def prior_binding(run,root):
    run=Path(run);prior=interruption()
    require(not(run/'COMPLETE.json').exists() and not(run/'BATCH_MANIFEST.json').exists(),'Interrupted batch was relabelled complete')
    require(record(run/'FAILED.json')==prior['batch_failure'],'Prior failure evidence changed')
    req=read(run/'BATCH_REQUEST.json')
    require(req['scope']=='full32370' and req['lock_sha256']==LOCK_SHA and req['code_records']==prior['code_records'],'Original batch identity differs')
    check_grid(req['groups'],root['groups'])
    require({p.name for p in run.iterdir()}==set(prior['sealed_groups'])|set(prior['failed_groups'])|{'BATCH_REQUEST.json','FAILED.json'},'Interrupted batch inventory differs')
    for ident,r in prior['sealed_groups'].items():
        p=member(run,ident)
        require(record(p/'RUN_MANIFEST.json')==r['manifest'],'Prior sealed manifest changed')
        require(read(p/'COMPLETE.json')==dict(status='complete',manifest=r['manifest']) and not(p/'FAILED.json').exists(),'Prior group incomplete')
    for ident,r in prior['failed_groups'].items():
        require(record(member(run,ident)/'FAILED.json')==r['evidence'] and not(member(run,ident)/'COMPLETE.json').exists(),'Failed group evidence changed')
    remaining(root)
    return prior

def disk_check(path,needed=0):
    free=shutil.disk_usage(path).free
    require(free>=RESERVE_BYTES+needed,'Continuation disk reserve insufficient')
    return free

def expected_request(ident,lockdir,source,initial):
    return original.expected_request(ident,lockdir,source,initial)|dict(continuation_of=INTERRUPTION_SHA,resource_policy=RESOURCE_POLICY)

