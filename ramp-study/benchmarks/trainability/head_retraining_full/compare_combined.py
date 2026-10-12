"""Exhaustive historical comparison of both controller generations; no fitting."""
import argparse
import sys
sys.dont_write_bytecode=True
from full_common import *
import combined
from full_compare import compare_group

def compare_all(lockdir,source,original_run,continuation,run,output):
    root,batch=combined.binding(lockdir,original_run,continuation,run)
    structural=combined.verify_run(lockdir,source,original_run,continuation,run)
    out=reserve(output,[lockdir,source,original_run,continuation,run]);(out/'groups').mkdir()
    try:
        summaries={};exact=0;total=0;predictions=0;changed=0;candidates=0;npz=0
        for route in batch['groups']:
            ident=route['id'];path=combined.group_path(original_run,continuation,route)
            report=compare_group(lockdir,source,path,ident,root)
            write(out/'groups'/(ident+'.json'),report);summaries[ident]=record(out/'groups'/(ident+'.json'))
            exact+=report['exact_jobs'];total+=len(report['jobs']);predictions+=report['predictions'];changed+=report['changed_predictions']
            candidates+=sum(r['candidate_count'] for r in report['jobs']);npz+=sum(r['prediction_npz_bytes_equal'] for r in report['jobs'])
            if len(summaries)%50==0:print('COMPARED',len(summaries),'/ 860 exact',exact,'/',total,flush=True)
        require(total==32370 and predictions==batch['predictions'],'Comparison grid/count incomplete')
        combined.binding(lockdir,original_run,continuation,run)
        report=dict(status='exact' if exact==total else 'completed_with_differences',scope='Full32370 frozen-representation head retraining, explicit original plus continuation',
            structural=structural,batch_manifest=record(Path(run)/'BATCH_MANIFEST.json'),jobs=total,exact_jobs=exact,predictions=predictions,changed_predictions=changed,
            candidate_records=candidates,byte_identical_npz=npz,groups=summaries,
            exclusions='Unchanged pilot whitelist: elapsed/training/inference seconds, candidate/selection seconds, prediction receipt path and separately checked NPZ hash')
        write(out/'report.json',report);seal(out,dict(scope='full_comparison',status=report['status']));return report
    except BaseException as e:write(out/'FAILED.json',dict(error=repr(e)));raise

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('lockdir','source-root','original-run','continuation','run','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();r=compare_all(a.lockdir,a.source_root,a.original_run,a.continuation,a.run,a.output)
    print({k:r[k] for k in ('status','jobs','exact_jobs','predictions','changed_predictions')})
