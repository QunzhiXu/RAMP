"""An explicit two-generation index, not a repaired COMPLETE for the failed batch."""
import argparse
import sys
sys.dont_write_bytecode=True
from resume_common import *
import full_run
import resume_run

PREFLIGHT_SHA='5ae48896e651074ea5c1285a97cb9a543e796e1bdb191712dca8f07fb7aa1e1b'

def controller_binding():
    p=HERE/'validation/continuation_preflight_20261011/report.json'
    require(record(p)['sha256']==PREFLIGHT_SHA,'Continuation preflight changed')
    require(read(p)['code_records']==codes(),'Continuation controller differs from preflight')

def expected_routes(lockdir,root,original_run,continuation):
    controller_binding();prior=prior_binding(original_run,root);_,batch=resume_run.binding(lockdir,continuation)
    return [dict(id=i,origin='original' if i in prior['sealed_groups'] else 'continuation',
                 manifest=prior['sealed_groups'][i]['manifest'] if i in prior['sealed_groups'] else batch['groups'][i])
            for i in root['groups']]

def check_routes(actual,root,expected):
    require(isinstance(actual,list) and all(isinstance(r,dict) and set(r)=={'id','origin','manifest'} for r in actual),'Combined route schema differs')
    check_grid([r['id'] for r in actual],root['groups'])
    require(actual==expected,'Combined routes differ')

def group_path(original_run,continuation,route):
    require(route['origin'] in ('original','continuation'),'Unknown group origin')
    return member(original_run if route['origin']=='original' else continuation,route['id'])

def binding(lockdir,original_run,continuation,combined):
    root=load_lock(lockdir);expected=expected_routes(lockdir,root,original_run,continuation)
    combined=Path(combined);m=read(combined/'BATCH_MANIFEST.json')
    require(set(m)=={'scope','lock_sha256','interruption_sha256','preflight_sha256','continuation_manifest','controllers','groups','jobs','predictions'},'Combined manifest schema differs')
    require(m['scope']=='combined_full32370' and m['lock_sha256']==LOCK_SHA and m['interruption_sha256']==INTERRUPTION_SHA and m['preflight_sha256']==PREFLIGHT_SHA,'Combined provenance differs')
    require(m['controllers']==dict(original=interruption()['code_records'],continuation=codes()),'Combined controller versions differ')
    require(m['continuation_manifest']==record(Path(continuation)/'BATCH_MANIFEST.json'),'Continuation manifest changed')
    check_routes(m['groups'],root,expected)
    require(type(m['jobs']) is int and m['jobs']==32370 and type(m['predictions']) is int and m['predictions']>0,'Combined counts differ')
    require(verify_seal(combined)['details']==dict(scope='combined_full32370'),'Combined seal differs')
    return root,m

def verify_payloads(lockdir,source,original_run,continuation,root,routes):
    import ast
    trees=[ast.parse((HERE/n).read_text()) for n in ('full_worker.py','resume_worker.py')]
    same=ast.dump(next(n for n in trees[0].body if isinstance(n,ast.FunctionDef)))==ast.dump(next(n for n in trees[1].body if isinstance(n,ast.FunctionDef)))
    require(same,'Worker execute scientific logic changed')
    reports=[]
    for route in routes:
        p=group_path(original_run,continuation,route)
        require(record(p/'RUN_MANIFEST.json')==route['manifest'],'Combined group changed')
        verifier=full_run.verify_group if route['origin']=='original' else resume_run.verify_group
        report=verifier(lockdir,source,p,route['id'],root);reports.append(report)
        if len(reports)%50==0:print('VERIFIED',len(reports),'/ 860',flush=True)
    jobs=sum(r['jobs'] for r in reports);require(jobs==32370,'Verified job count differs')
    return dict(status='structurally_verified',groups=860,jobs=jobs,predictions=sum(r['predictions'] for r in reports),
        original_groups=845,original_jobs=sum(r['jobs'] for r in reports[:845]),continuation_groups=15,continuation_jobs=sum(r['jobs'] for r in reports[845:]),
        controller_versions=2,scientific_worker_execute_identical=same)

def verify_run(lockdir,source,original_run,continuation,combined):
    root,m=binding(lockdir,original_run,continuation,combined)
    report=verify_payloads(lockdir,source,original_run,continuation,root,m['groups'])
    require(report['jobs']==m['jobs'] and report['predictions']==m['predictions'],'Combined payload totals differ')
    require(report==read(Path(combined)/'verification.json'),'Combined saved verification differs')
    return report

def assemble(lockdir,source,original_run,continuation,output):
    import ast
    root=load_lock(lockdir);routes=expected_routes(lockdir,root,original_run,continuation)
    # Do not equate different controllers by assertion: compare the numerical execute AST.
    a=ast.parse((HERE/'full_worker.py').read_text());b=ast.parse((HERE/'resume_worker.py').read_text())
    require(ast.dump(next(n for n in a.body if isinstance(n,ast.FunctionDef)))==ast.dump(next(n for n in b.body if isinstance(n,ast.FunctionDef))),'Worker execute scientific logic changed')
    out=reserve(output,[lockdir,source,original_run,continuation])
    try:
        report=verify_payloads(lockdir,source,original_run,continuation,root,routes)
        require(routes==expected_routes(lockdir,root,original_run,continuation),'Group bindings changed during verification')
        write(out/'BATCH_MANIFEST.json',dict(scope='combined_full32370',lock_sha256=LOCK_SHA,interruption_sha256=INTERRUPTION_SHA,preflight_sha256=PREFLIGHT_SHA,
            continuation_manifest=record(Path(continuation)/'BATCH_MANIFEST.json'),controllers=dict(original=interruption()['code_records'],continuation=codes()),
            groups=routes,jobs=report['jobs'],predictions=report['predictions']))
        write(out/'verification.json',report);seal(out,dict(scope='combined_full32370'));return report
    except BaseException as e:write(out/'FAILED.json',dict(error=repr(e)));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['assemble','verify-run','check-binding'])
    for n in ('lockdir','source-root','original-run','continuation','run'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();args=(a.lockdir,a.source_root,a.original_run,a.continuation,a.run)
    print(assemble(*args) if a.mode=='assemble' else verify_run(*args) if a.mode=='verify-run' else binding(a.lockdir,a.original_run,a.continuation,a.run)[1]['jobs'])

