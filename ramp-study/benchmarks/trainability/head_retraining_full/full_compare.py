"""Post-fit exhaustive comparisons. Differences are evidence, not parameter-tuning targets."""
import argparse
import io
import sys
sys.dont_write_bytecode=True
from full_common import *
from full_run import verify_run
import compare as pilot_compare

def compare_group(lockdir,source,run,ident,root):
    import numpy as np
    g=group_lock(lockdir,root,ident);rows=[]
    def historical(name):
        data=member(source,name).read_bytes();require(dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())==g['references'][name],'Historical reference changed');return data
    for jobid,row in g['tasks'].items():
        new=Path(run)/'runtime/results'/jobid;prefix=row['folder']+'/'
        old_data=historical(prefix+'predictions.npz');receipt=json.loads(historical(prefix+'predictions.json'));old=json.loads(historical(prefix+'RESULT.json'))
        ledger=bound_json(source,prefix+'ledger.json',g['inputs'])
        require(read(new/'ledger.json')==ledger and old['prediction_receipt']==receipt,'Historical result/ledger binding')
        require(receipt['sha256']==hashlib.sha256(old_data).hexdigest() and receipt['prediction_file'].endswith('/'+prefix+'predictions.npz'),'Historical prediction binding')
        require(receipt['provenance']==dict(ledger_sha256=digest(ledger),pool_sha256=ledger['pool_sha256'],selection=old['selection']),'Historical selection binding')
        with np.load(io.BytesIO(old_data),allow_pickle=False) as a,np.load(new/'predictions.npz',allow_pickle=False) as b:
            require(set(a.files)==set(b.files)=={'cell_ids','predictions'},'Unexpected prediction fields')
            require(a['cell_ids'].dtype==b['cell_ids'].dtype and np.array_equal(a['cell_ids'],b['cell_ids']),'Prediction order differs')
            x=a['predictions'];y=b['predictions'];require(x.dtype==y.dtype and x.shape==y.shape,'Prediction shape/dtype differs')
            changes=int(np.count_nonzero(x!=y));delta=float(np.max(np.abs(x-y))) if x.dtype.kind=='f' else None
        actual=read(new/'RESULT.json');diff=pilot_compare.tree_differences(pilot_compare.normalize(actual),pilot_compare.normalize(old))
        rows.append(dict(id=jobid,job=row['job'],predictions=len(x),changed_predictions=changes,max_abs_prediction_difference=delta,
            prediction_npz_bytes_equal=record(new/'predictions.npz')['sha256']==hashlib.sha256(old_data).hexdigest(),
            candidate_count=len(actual['candidates']),scientific_result_differences=diff,exact=changes==0 and not diff))
    return dict(id=ident,jobs=rows,exact_jobs=sum(r['exact'] for r in rows),predictions=sum(r['predictions'] for r in rows),changed_predictions=sum(r['changed_predictions'] for r in rows))

def compare_all(lockdir,source,run,output):
    root=load_lock(lockdir);structural=verify_run(lockdir,source,run);out=reserve(output,[lockdir,source,run]);(out/'groups').mkdir()
    try:
        summaries={};exact=0;total=0;predictions=0;changed=0;candidates=0;npz=0
        for ident in root['groups']:
            report=compare_group(lockdir,source,Path(run)/ident,ident,root)
            write(out/'groups'/(ident+'.json'),report);summaries[ident]=record(out/'groups'/(ident+'.json'))
            exact+=report['exact_jobs'];total+=len(report['jobs']);predictions+=report['predictions'];changed+=report['changed_predictions']
            candidates+=sum(r['candidate_count'] for r in report['jobs']);npz+=sum(r['prediction_npz_bytes_equal'] for r in report['jobs'])
            if len(summaries)%50==0:print('COMPARED',len(summaries),'/',860,'exact',exact,'/',total,flush=True)
        require(total==32370,'Comparison grid incomplete')
        report=dict(status='exact' if exact==total else 'completed_with_differences',scope='Full32370 frozen-representation head retraining',structural=structural,batch_manifest=record(Path(run)/'BATCH_MANIFEST.json'),
            jobs=total,exact_jobs=exact,predictions=predictions,changed_predictions=changed,candidate_records=candidates,byte_identical_npz=npz,groups=summaries,
            exclusions='Only original pilot whitelist: elapsed/training/inference seconds, candidate/selection seconds, prediction receipt path and separately checked NPZ hash')
        write(out/'report.json',report);seal(out,dict(scope='full_comparison',status=report['status']));return report
    except BaseException as e:write(out/'FAILED.json',dict(error=repr(e)));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lockdir',type=Path,required=True);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=compare_all(a.lockdir,a.source_root,a.run,a.output);print({k:r[k] for k in ('status','jobs','exact_jobs','predictions','changed_predictions')})
